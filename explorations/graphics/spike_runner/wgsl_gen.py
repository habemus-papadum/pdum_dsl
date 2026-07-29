"""WGSL generation, split clean from every device act.

Adapted from ``conformance/wgsl_executor.py`` (the marker tables and
``_Gen`` are lifted near-verbatim; the LEAF rows are rewritten). Two
differences from the original, both of them the point of the spike:

1. **No device calls.** ``_translate`` / ``render_wgpu`` interleave source
   generation with ``create_shader_module`` / ``create_buffer_with_data`` /
   ``begin_render_pass`` / ``submit`` / ``read_buffer`` in one function
   body, so there is no way to get the source without also getting a
   submitted frame. Here generation returns a description; the runner does
   the device work.

2. **Residency-aware leaves.** The original indexes every buffer as if it
   were contiguous row-major over its own dims, which is only true because
   the caller uploaded ``ascontiguousarray(t.to_numpy(...))`` — the upload
   MANUFACTURES the layout the shader assumes. Here every leaf reads
   through ``residency.index_expr`` against the view's real strides, so one
   resident buffer serves a record's field views and the record itself.

The fragment's color0 is a scalar f32 in the lowering, so the surface
expansion (how one channel becomes a swap-chain's four) is a parameter of
generation, not a property of the shader — see ``target`` below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from residency import Residency, index_expr

from pdum.tl.dialect import _thaw_params, walk_region

# --- lifted verbatim from wgsl_executor.py -----------------------------------
_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_CMP = {"lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "==", "ne": "!="}
_FNS = {f: f for f in ("sqrt", "exp", "log", "tanh", "abs", "floor", "sin", "cos")}
_CORE_INFIX = {"core.add": "+", "core.sub": "-", "core.mul": "*", "core.div": "/"}


class Untranslatable(Exception):
    """This region has no WGSL translation yet — the reason names the op."""


def _lit(v) -> str:
    s = repr(float(v))
    return s if ("." in s or "e" in s or "inf" in s or "nan" in s) else s + ".0"


class _Gen:
    """A per-stage expression generator over the shared marker tables; the
    stage supplies its LEAF row. Lifted from wgsl_executor._Gen."""

    def __init__(self, leaf):
        self.leaf = leaf
        self.lines: list[str] = []
        self.names: dict[int, str] = {}
        self.bools: set[int] = set()
        self.n = 0

    def go(self, node):
        if id(node) in self.names:
            return self.names[id(node)]
        expr, is_bool = self.expr(node)
        var = f"e{self.n}"
        self.n += 1
        self.lines.append(f"  let {var}: {'bool' if is_bool else 'f32'} = {expr};")
        self.names[id(node)] = var
        if is_bool:
            self.bools.add(id(node))
        return var

    def operand(self, node):
        v = self.go(node)
        return f"select(0.0, 1.0, {v})" if id(node) in self.bools else v

    def cond(self, node):
        v = self.go(node)
        return v if id(node) in self.bools else f"({v} != 0.0)"

    def expr(self, node):
        got = self.leaf(node, self)
        if got is not None:
            return got
        attrs = dict(node.attrs)
        op = node.op
        if op in ("core.const", "tl.const"):
            return _lit(attrs["value"]), False
        if op == "tl.pointwise":
            f = attrs["f"]
            ops = [self.operand(a) for a in node.args]
            if f in _INFIX:
                return f"({ops[0]} {_INFIX[f]} {ops[1]})", False
            if f == "neg":
                return f"(-{ops[0]})", False
            if f in _CMP:
                return f"({ops[0]} {_CMP[f]} {ops[1]})", True
            if f == "where":
                return f"select({ops[2]}, {ops[1]}, {self.cond(node.args[0])})", False
            if f in ("maximum", "minimum"):
                return f"{'max' if f == 'maximum' else 'min'}({ops[0]}, {ops[1]})", False
            if f in _FNS:
                return f"{_FNS[f]}({ops[0]})", False
            raise Untranslatable(f"marker {f!r}")
        if op in _CORE_INFIX:
            a, b = (self.operand(x) for x in node.args)
            return f"({a} {_CORE_INFIX[op]} {b})", False
        if op == "core.neg":
            return f"(-{self.operand(node.args[0])})", False
        if op == "core.cmp":
            a, b = (self.operand(x) for x in node.args)
            return f"({a} {_CMP[attrs['pred']]} {b})", True
        if op == "core.select":
            t_, e_ = self.operand(node.args[1]), self.operand(node.args[2])
            return f"select({e_}, {t_}, {self.cond(node.args[0])})", False
        if op.startswith("pw."):
            f = op[3:]
            if f in _FNS:
                return f"{_FNS[f]}({self.operand(node.args[0])})", False
            if f in ("maximum", "minimum"):
                a, b = (self.operand(x) for x in node.args)
                return f"{'max' if f == 'maximum' else 'min'}({a}, {b})", False
        if op == "tl.sample":
            cy, cx = (self.operand(a) for a in node.args)
            i = attrs["idx"]
            return f"textureSampleLevel(tex{i}, smp{i}, vec2<f32>({cx}, {cy}), f32({attrs['lod']})).x", False
        raise Untranslatable(op)


# --- binding plans -----------------------------------------------------------


@dataclass
class BufferBinding:
    """One resident device buffer, at one binding index of one bind group."""

    binding: int
    resident: object  # residency.Resident
    label: str


@dataclass
class ComputeProgram:
    source: str
    buffers: tuple  # BufferBinding, in binding order
    slot_binding: int | None  # the U buffer's binding, or None
    slots: tuple  # (offset, fmt) sorted by offset — the staging read plan
    workgroups: tuple  # the dispatch, precomputed (extents are compile-time)
    axes: tuple


@dataclass
class RenderProgram:
    source: str
    buffers: tuple  # BufferBinding
    v_slot_binding: int | None
    v_slots: tuple
    f_slot_binding: int | None
    f_slots: tuple
    tex_base: int
    tex_pairs: tuple
    draw_count: int
    varyings: tuple
    v_ctx: dict = field(repr=False, default_factory=dict)
    f_ctx: dict = field(repr=False, default_factory=dict)
    fparams: tuple = ()


class _Binder:
    """Assigns bind-group slots to residents in first-use order, so that a
    buffer touched by two params (a record's two field views) binds ONCE."""

    def __init__(self, res: Residency):
        self.res = res
        self.order: list[BufferBinding] = []
        self._by_id: dict[int, BufferBinding] = {}

    def slot(self, tensor, label: str) -> int:
        key = id(tensor.buffer)
        got = self._by_id.get(key)
        if got is None:
            got = BufferBinding(binding=len(self.order), resident=self.res.resident(tensor, label), label=label)
            self._by_id[key] = got
            self.order.append(got)
        return got.binding


# --- the compute stage -------------------------------------------------------


def gen_compute(art, values, res: Residency, labels=()) -> ComputeProgram:
    """Kernel region -> WGSL compute shader, reading and writing through
    resident buffers at their real strides."""
    region = art.region
    params = list(region.params)
    if not art.tensor_params:
        raise Untranslatable("a kernel with no tensor parameters")
    lattice = params[len(art.tensor_params) - 1]
    axes = tuple(d.name for d in lattice.type.dims)
    if len(axes) > 2:
        raise Untranslatable("rank-3+ lattices (the 2D/1D subset translates today)")
    comp = {n: c for n, c in zip(axes, ("y", "x") if len(axes) == 2 else ("x",))}
    extents = {d.name: (d.start, d.stop) for d in lattice.type.dims}

    slots = list(art.uniforms and [(o, f) for _, o, f in art.uniforms] or [])
    for _pname, _fixed, _wrap, base, plan, _extract in art.arg_slots:
        for s in plan.slots:
            slots.append((base + s.dest.offset, s.dest.fmt))
    slots.sort()
    slot_index = {off: i for i, (off, _) in enumerate(slots)}

    binder = _Binder(res)
    labels = tuple(labels) or tuple(art.tensor_params) + tuple(f"tap{i}" for i in range(len(params)))

    def read_at(p) -> str:
        i = params.index(p)
        view = values[i]
        coords = {}
        for d in view.layout.dims:
            if d.name not in comp:
                raise Untranslatable(f"a buffer dim {d.name!r} outside the launch lattice")
            coords[d.name] = f"i32(gid.{comp[d.name]})"
        b = binder.slot(view, labels[i] if i < len(labels) else f"p{i}")
        return f"b{b}[{index_expr(view, coords)}]"

    def leaf(n, g):
        if n.op == "core.param":
            return read_at(n), False
        if n.op == "tl.iota":
            name = dict(n.attrs)["name"]
            if name not in comp:
                raise Untranslatable(f"an iota over dim {name!r} outside the launch lattice")
            return f"f32(gid.{comp[name]})", False
        if n.op == "abi.slot":
            return f"U[{slot_index[dict(n.attrs)['offset']]}]", False
        if n.op == "tl.read":
            tex, *idx = n.args
            if tex.op != "core.param":
                raise Untranslatable("a read of a non-parameter tensor")
            i = params.index(tex)
            view = values[i]
            coords = {d.name: f"i32({g.operand(ix)})" for d, ix in zip(view.layout.dims, idx)}
            b = binder.slot(view, labels[i] if i < len(labels) else f"p{i}")
            return f"b{b}[{index_expr(view, coords)}]", False
        return None

    gen = _Gen(leaf)
    stores: list[str] = []
    for n in walk_region(region):
        if n.op == "tl.store":
            _tok, dst, val = n.args
            if dst.op != "core.param":
                raise Untranslatable("a store into a non-parameter tensor")
            stores.append(f"  {read_at(dst)} = {gen.operand(val)};")
        elif n.op in ("tl.token", "core.yield"):
            continue

    decls = [
        f"@group(0) @binding({b.binding}) var<storage, read_write> b{b.binding}: array<f32>;"
        for b in binder.order
    ]
    slot_binding = None
    if slots:
        slot_binding = len(binder.order)
        decls.append(f"@group(0) @binding({slot_binding}) var<storage, read> U: array<f32>;")

    wg = "@workgroup_size(8, 8, 1)" if len(axes) == 2 else "@workgroup_size(64, 1, 1)"
    guards = " || ".join(f"gid.{comp[a]} >= {extents[a][1] - extents[a][0]}u" for a in axes)
    src = "\n".join(
        [
            *decls,
            f"@compute {wg}",
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
            f"  if ({guards}) {{ return; }}",
            *gen.lines,
            *stores,
            "}",
        ]
    )
    ext = [extents[a][1] - extents[a][0] for a in axes]
    groups = ((ext[1] + 7) // 8, (ext[0] + 7) // 8, 1) if len(axes) == 2 else ((ext[0] + 63) // 64, 1, 1)
    return ComputeProgram(
        source=src,
        buffers=tuple(binder.order),
        slot_binding=slot_binding,
        slots=tuple(slots),
        workgroups=groups,
        axes=axes,
    )


# --- the render stage --------------------------------------------------------

_SURFACE = {
    # target format -> (wgsl return type, expression from the scalar color0)
    "r32float": ("f32", "{c}"),
    "bgra8unorm": ("vec4<f32>", "vec4<f32>({c}, {c}, {c}, 1.0)"),
    "rgba8unorm": ("vec4<f32>", "vec4<f32>({c}, {c}, {c}, 1.0)"),
    "bgra8unorm-srgb": ("vec4<f32>", "vec4<f32>({c}, {c}, {c}, 1.0)"),
    "rgba8unorm-srgb": ("vec4<f32>", "vec4<f32>({c}, {c}, {c}, 1.0)"),
}


def gen_render(pso, vs_args, fs_args, res: Residency, target: str = "r32float") -> RenderProgram:
    """PSO -> one WGSL module with vs_main/fs_main, plus its binding plan.

    Adapted from ``wgsl_executor.render_wgpu``'s first half. Everything the
    original did to a device after building ``source`` is the runner's job."""
    import struct

    from pdum.dsl.pack import pack_into
    from pdum.tl.graphics import _check_pairing, _lower_fragment, _lower_vertex
    from pdum.tl.producer import _captured

    if target not in _SURFACE:
        raise Untranslatable(f"no surface expansion for target format {target!r}")
    ret_ty, ret_expr = _SURFACE[target]

    v_region, vnames, flats, count, _lattice, v_ctx = _lower_vertex(pso.vs, tuple(vs_args))
    _check_pairing(frozenset(vnames), pso.required)
    plan = list(v_ctx.get("g.param_plan", ()))
    buf_params = list(v_region.params[: len(plan)])
    binder = _Binder(res)

    def view_of(pi: int):
        i, fname = plan[pi]
        b = vs_args[i]
        return (b.field(fname), f"vb{i}.{fname}") if fname is not None else (b, f"vb{i}")

    def pulled(pi: int, extra_coords: dict) -> str:
        view, label = view_of(pi)
        coords = dict(extra_coords)
        for d in view.layout.dims:
            if d.name == "vertex_id":
                coords[d.name] = "i32(vid)"
            elif d.name not in coords:
                raise Untranslatable(f"a vertex buffer dim {d.name!r} with no selected coordinate")
        b = binder.slot(view, label)
        return f"b{b}[{index_expr(view, coords)}]"

    v_slots = sorted((dict(nd.attrs)["offset"], dict(nd.attrs)["fmt"]) for nd in v_ctx["k.uniforms"].values())
    v_slot_index = {off: i for i, (off, _) in enumerate(v_slots)}

    def v_leaf(n, g):
        if n.op == "tl.iota" and dict(n.attrs).get("name") == "vertex_id":
            return "f32(vid)", False
        if n.op == "abi.slot":
            return f"VU[{v_slot_index[dict(n.attrs)['offset']]}]", False
        if n.op == "tl.select" and n.args and n.args[0] in buf_params:
            pi = buf_params.index(n.args[0])
            return pulled(pi, _thaw_params(dict(n.attrs))["coords"]), False
        if n.op == "core.param":
            if n in buf_params:
                return pulled(buf_params.index(n), {}), False
            raise Untranslatable("a vertex-stage param beyond the pulled buffers")
        return None

    vg = _Gen(v_leaf)
    v_outs = [vg.operand(a) for a in v_region.body[-1].args[0].args]

    # the fragment over the pixel lattice, varyings as params
    from pdum.tl.dialect import tensor_type
    from pdum.tl.tensor import Tensor

    ref_lat = Tensor.from_numpy(np.zeros((2, 2)), ("y", "x"))
    v_types = {n: tensor_type(ref_lat) for n in vnames}
    f_region, f_ctx, _touched, fparams = _lower_fragment(pso.fs, tuple(fs_args), v_types)
    if f_ctx["k.fn_markers"]:
        raise Untranslatable("an oracle-class fragment fn-argument on the device")
    bound = dict(zip(fparams, fs_args))

    f_slots, staging = [], bytearray(f_ctx["k.uniform_size"])
    env = _captured(pso.fs.fn) if f_ctx["k.uniforms"] else {}
    for name, nd in f_ctx["k.uniforms"].items():
        at = dict(nd.attrs)
        struct.pack_into(at["fmt"], staging, at["offset"], env[name])
        f_slots.append((at["offset"], at["fmt"]))
    for blk in f_ctx["k.arg_plans"].values():
        f = bound[blk["pname"]] if blk["pname"] is not None else blk["fixed"]
        if blk["wrap"] is not None:
            f = blk["wrap"](f)
        view = memoryview(staging)[blk["base"] : blk["base"] + blk["plan"].staging_size]
        pack_into(blk["plan"], view, blk["extract"](f.captures, ()))
        for s in blk["plan"].slots:
            f_slots.append((blk["base"] + s.dest.offset, s.dest.fmt))
    f_slots.sort()
    f_slot_index = {off: i for i, (off, _) in enumerate(f_slots)}
    f_params = list(f_region.params)

    def f_leaf(n, g):
        if n.op == "core.param":
            return f"vin.f{f_params.index(n)}", False
        if n.op == "abi.slot":
            return f"U[{f_slot_index[dict(n.attrs)['offset']]}]", False
        return None

    fg = _Gen(f_leaf)
    f_out = fg.operand(f_region.body[-1].args[0])

    fields = [
        f"  @location({i}){' @interpolate(flat)' if n in flats else ''} f{i}: f32,"
        for i, n in enumerate(vnames)
    ]
    lines = ["struct VOut {", "  @builtin(position) pos: vec4<f32>,", *fields, "}"]
    for b in binder.order:
        lines.append(f"@group(0) @binding({b.binding}) var<storage, read> b{b.binding}: array<f32>;")
    nxt = len(binder.order)
    v_slot_binding = None
    if v_slots:
        v_slot_binding = nxt
        lines.append(f"@group(0) @binding({nxt}) var<storage, read> VU: array<f32>;")
        nxt += 1
    f_slot_binding = None
    if f_slots:
        f_slot_binding = nxt
        lines.append(f"@group(0) @binding({nxt}) var<storage, read> U: array<f32>;")
        nxt += 1
    tex_pairs = tuple(f_ctx["g.textures"])
    tex_base = nxt
    for i in range(len(tex_pairs)):
        lines.append(f"@group(0) @binding({tex_base + 2 * i}) var tex{i}: texture_2d<f32>;")
        lines.append(f"@group(0) @binding({tex_base + 2 * i + 1}) var smp{i}: sampler;")
    lines += [
        "@vertex",
        "fn vs_main(@builtin(vertex_index) vid: u32) -> VOut {",
        *vg.lines,
        "  var o: VOut;",
        f"  o.pos = vec4<f32>({v_outs[0]}, {v_outs[1]}, 0.0, 1.0);",
        *[f"  o.f{i} = {v};" for i, v in enumerate(v_outs[2:])],
        "  return o;",
        "}",
        "@fragment",
        f"fn fs_main(vin: VOut) -> @location(0) {ret_ty} {{",
        *fg.lines,
        f"  return {ret_expr.format(c=f_out)};",
        "}",
    ]
    return RenderProgram(
        source="\n".join(lines),
        buffers=tuple(binder.order),
        v_slot_binding=v_slot_binding,
        v_slots=tuple(v_slots),
        f_slot_binding=f_slot_binding,
        f_slots=tuple(f_slots),
        tex_base=tex_base,
        tex_pairs=tex_pairs,
        draw_count=count,
        varyings=tuple(vnames),
        v_ctx=v_ctx,
        f_ctx=f_ctx,
        fparams=tuple(fparams),
    )
