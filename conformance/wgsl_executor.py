"""The WebGPU CONFORMANCE executor (200 §7, the P8 gate) — translation-only.

This is NOT a performance backend (the Not-doing list): it renders a
kernel REGION to naive WGSL — one thread per lattice point, no tiling,
no fusion — compiles it through wgpu, and executes it over real device
buffers. It plugs into the executor COLUMN built at 240 C5.2:
``dataclasses.replace(artifact, executor=compile_wgsl(artifact))``
reuses the whole launch protocol (staging pack, rebind, the overlap
refusal) and swaps only the backend column behind the same content-key
discipline (`WGPU_FP` is `_EXECUTOR_FP`'s second value).

THE AMBIENT DECISION, resolved here (recorded in memory + the ledger):
``tl.iota`` stays the ambient's spelling. Its translation row IS a
one-line lookup — dim name → global-invocation-id component — because
the alignment law makes every kernel-region iota a coordinate function
of the thread. The unification thesis survives into the device era; no
dedicated ambient op is needed.

Numeric policy (210, both sides): the device computes f32 (WGSL has no
f64); differentials against the f64 reference state their tolerances.
The device does NOT bounds-check reads — the reference refuses OOB, so
any case in-bounds on reference is certified UB-free before it runs
here (the keying-ladder ruling).

Honest skips: a region containing an op with no translation row raises
``Untranslatable`` naming the op; the battery skips with that reason.
"""

from __future__ import annotations

import struct
from dataclasses import replace

import numpy as np
from pdum.tl.dialect import _thaw_params, walk_region
from pdum.tl.tensor import Tensor

WGPU_FP = ("wgsl", "wgpu")  # the backend column's second value


class Untranslatable(Exception):
    """This region has no WGSL translation yet — the reason names the op."""


_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_CMP = {"lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "==", "ne": "!="}
_FNS = {f: f for f in ("sqrt", "exp", "log", "tanh", "abs", "floor", "sin", "cos")}
_CORE_INFIX = {"core.add": "+", "core.sub": "-", "core.mul": "*", "core.div": "/"}


def _device():
    from pdum.tl.graphics import _device as dev

    return dev()


def _dims_of(node):
    return tuple(d.name for d in node.type.dims)


def _translate(art) -> tuple[str, dict]:
    """Region -> WGSL compute shader. Returns (source, meta): buffer
    order, per-buffer dims, the writable set, the uniform slot list."""
    region = art.region
    params = list(region.params)
    lattice = params[len(art.tensor_params) - 1] if art.tensor_params else None
    if lattice is None:
        raise Untranslatable("a kernel with no tensor parameters")
    axes = _dims_of(lattice)  # dim name -> gid component, by the writable's order
    if len(axes) > 2:
        raise Untranslatable("rank-3+ lattices (the 2D/1D subset translates today)")
    comp = {name: c for name, c in zip(axes, ("y", "x") if len(axes) == 2 else ("x",))}
    extents = {d.name: (d.start, d.stop) for d in lattice.type.dims}

    slots = list(art.uniforms)  # (name, offset, fmt) — the kernel's own captures
    for _, _, _, base, plan, _ in art.arg_slots:  # spliced fn-arg blocks
        for s in plan.slots:
            slots.append((f"arg{base}", base + s.dest.offset, s.dest.fmt))
    slot_index = {off: i for i, (_, off, _) in enumerate(sorted(slots, key=lambda s: s[1]))}

    names: dict[int, str] = {}
    lines: list[str] = []
    bools: set[int] = set()
    stores: list[str] = []
    counter = [0]

    def buf_index(p) -> str:
        """The linear index of a buffer at the thread's coordinates —
        row-major over ITS dims, each coordinate the gid component of
        that dim name (the alignment law's gift)."""
        dims = p.type.dims
        strides, acc = [], 1
        for d in reversed(dims):
            strides.append(acc)
            acc *= d.size
        parts = []
        for d, s in zip(dims, reversed(strides)):
            if d.name not in comp:
                raise Untranslatable(f"a buffer dim {d.name!r} outside the launch lattice")
            parts.append(f"i32(gid.{comp[d.name]}) * {s}")
        return " + ".join(parts) or "0"

    def as_f32(nid, expr):
        return f"select(0.0, 1.0, {expr})" if nid in bools else expr

    def go(n):
        if id(n) in names:
            return names[id(n)]
        expr, is_bool = _expr(n)
        var = f"v{counter[0]}"
        counter[0] += 1
        ty = "bool" if is_bool else "f32"
        lines.append(f"  let {var}: {ty} = {expr};")
        names[id(n)] = var
        if is_bool:
            bools.add(id(n))
        return var

    def operand(n) -> str:
        v = go(n)
        return as_f32(id(n), v)

    def _expr(n) -> tuple[str, bool]:
        attrs = dict(n.attrs)
        if n.op == "core.param":
            i = params.index(n)
            return f"buf{i}[{buf_index(n)}]", False
        if n.op == "tl.iota":
            name = attrs["name"]
            if name not in comp:
                raise Untranslatable(f"an iota over dim {name!r} outside the launch lattice")
            return f"f32(gid.{comp[name]})", False  # THE one-line ambient row
        if n.op == "core.const":
            return _lit(attrs["value"]), False
        if n.op == "tl.const":
            return _lit(attrs["value"]), False
        if n.op == "abi.slot":
            return f"U[{slot_index[attrs['offset']]}]", False
        if n.op == "tl.pointwise":
            f = attrs["f"]
            ops = [operand(a) for a in n.args]
            if f in _INFIX:
                return f"({ops[0]} {_INFIX[f]} {ops[1]})", False
            if f == "neg":
                return f"(-{ops[0]})", False
            if f in _CMP:
                return f"({ops[0]} {_CMP[f]} {ops[1]})", True
            if f == "where":
                cond = go(n.args[0])
                c = cond if id(n.args[0]) in bools else f"({as_f32(id(n.args[0]), cond)} != 0.0)"
                return f"select({ops[2]}, {ops[1]}, {c})", False
            if f == "maximum":
                return f"max({ops[0]}, {ops[1]})", False
            if f == "minimum":
                return f"min({ops[0]}, {ops[1]})", False
            if f in _FNS:
                return f"{_FNS[f]}({ops[0]})", False
            raise Untranslatable(f"marker {f!r}")
        if n.op in _CORE_INFIX:
            a, b = (operand(x) for x in n.args)
            return f"({a} {_CORE_INFIX[n.op]} {b})", False
        if n.op == "core.neg":
            return f"(-{operand(n.args[0])})", False
        if n.op == "core.cmp":
            a, b = (operand(x) for x in n.args)
            return f"({a} {_CMP[attrs['pred']]} {b})", True
        if n.op == "core.select":
            c = go(n.args[0])
            cond = c if id(n.args[0]) in bools else f"({as_f32(id(n.args[0]), c)} != 0.0)"
            return f"select({operand(n.args[2])}, {operand(n.args[1])}, {cond})", False
        if n.op.startswith("pw."):
            f = n.op[3:]
            if f in _FNS:
                return f"{_FNS[f]}({operand(n.args[0])})", False
            if f in ("maximum", "minimum"):
                return f"{'max' if f == 'maximum' else 'min'}({operand(n.args[0])}, {operand(n.args[1])})", False
            raise Untranslatable(f"scalar op pw.{f}")
        if n.op == "tl.read":
            tex, *idx = n.args
            if tex.op != "core.param":
                raise Untranslatable("a read of a non-parameter tensor")
            i = params.index(tex)
            dims = tex.type.dims
            strides, acc = [], 1
            for d in reversed(dims):
                strides.append(acc)
                acc *= d.size
            parts = []
            for ix, d, s in zip(idx, dims, reversed(strides)):
                parts.append(f"i32({operand(ix)}) * {s}")  # no bounds check: reference-certified
            return f"buf{i}[{' + '.join(parts)}]", False
        raise Untranslatable(n.op)

    def _lit(v) -> str:
        s = repr(float(v))
        return s if ("." in s or "e" in s or "inf" in s or "nan" in s) else s + ".0"

    # walk once: emit stores in token order; everything else demand-driven
    for n in walk_region(region):
        if n.op == "tl.store":
            _, dst, val = n.args
            if dst.op != "core.param":
                raise Untranslatable("a store into a non-parameter tensor")
            i = params.index(dst)
            stores.append(f"  buf{i}[{buf_index(dst)}] = {operand(val)};")
        elif n.op in ("tl.token", "core.yield"):
            continue

    bindings = []
    for i, p in enumerate(params):
        writable = i < len(art.tensor_params) and art.tensor_params[i] in art.writable
        writable = writable or i >= len(art.tensor_params)  # tap buffers write
        mode = "read_write" if writable else "read"
        bindings.append(f"@group(0) @binding({i}) var<storage, {mode}> buf{i}: array<f32>;")
    if slots:  # layout="auto" prunes unused bindings: U exists only when slots do
        bindings.append(f"@group(0) @binding({len(params)}) var<storage, read> U: array<f32>;")

    wg = "@workgroup_size(8, 8, 1)" if len(axes) == 2 else "@workgroup_size(64, 1, 1)"
    guards = " || ".join(
        f"gid.{comp[a]} >= {extents[a][1] - extents[a][0]}u" for a in axes
    )
    src = "\n".join(
        [
            *bindings,
            f"@compute {wg}",
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
            f"  if ({guards}) {{ return; }}",
            *lines,
            *stores,
            "}",
        ]
    )
    meta = {
        "slots": sorted(slots, key=lambda s: s[1]),
        "axes": axes,
        "extents": extents,
        "n_params": len(params),
    }
    return src, meta


def compile_wgsl(art):
    """The backend column's second executor: (values, staging) -> effect,
    behind the same signature run_region serves. Buffers upload f32,
    writables read back into the tensors' own buffers."""
    source, meta = _translate(art)
    device = _device()
    import wgpu

    module = device.create_shader_module(code=source)
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": "main"}
    )

    def executor(values, staging):
        writable = set(art.writable)
        bufs, orders = [], []
        for i, t in enumerate(values):
            order = tuple(d.name for d in t.layout.dims)
            arr = np.ascontiguousarray(t.to_numpy(order=order), dtype=np.float32)
            name = art.tensor_params[i] if i < len(art.tensor_params) else f"tap:{i}"
            usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
            bufs.append(device.create_buffer_with_data(data=arr.tobytes(), usage=usage))
            orders.append((name, order, arr.shape))
        entries = [{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)]
        if meta["slots"]:
            uvals = [struct.unpack_from(fmt, staging, off)[0] for _, off, fmt in meta["slots"]]
            ubuf = device.create_buffer_with_data(
                data=np.asarray(uvals, dtype=np.float32).tobytes(), usage=wgpu.BufferUsage.STORAGE
            )
            entries.append({"binding": len(bufs), "resource": {"buffer": ubuf, "offset": 0, "size": ubuf.size}})
        bind = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)
        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bind)
        axes = meta["axes"]
        ext = [meta["extents"][a][1] - meta["extents"][a][0] for a in axes]
        if len(axes) == 2:
            cp.dispatch_workgroups((ext[1] + 7) // 8, (ext[0] + 7) // 8, 1)
        else:
            cp.dispatch_workgroups((ext[0] + 63) // 64, 1, 1)
        cp.end()
        device.queue.submit([enc.finish()])
        from pdum.tl.ir import Token, _store

        for (name, order, shape), buf, t in zip(orders, bufs, values):
            if name in writable:
                raw = np.frombuffer(device.queue.read_buffer(buf), dtype=np.float32).reshape(shape)
                _store(Token(), t, Tensor.from_numpy(raw.astype(np.float64), order))
        return None

    return executor


def wgpu_artifact(art):
    """The drop-in: the same artifact with only the backend column
    swapped — launch() then runs staging/rebind/overlap identically and
    lands on the device."""
    return replace(art, executor=compile_wgsl(art))


# --- the render path: the quad+f golden (the P8 gate's named item) -----------


def _lit_of(v) -> str:
    s = repr(float(v))
    return s if ("." in s or "e" in s or "inf" in s or "nan" in s) else s + ".0"


class _Gen:
    """A per-stage expression generator over the shared marker tables;
    the stage supplies its LEAF row (ambient / params / slots)."""

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
            return _lit_of(attrs["value"]), False
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


def render_wgpu(pso, *args, shape):
    """The PSO through a REAL vertex/fragment pipeline into an offscreen
    r32float target — the reference interpolator's device twin (exact
    f32 readback, no rgba8 quantization). Vertex buffers ride STORAGE-
    BUFFER VERTEX PULLING (250/stage-5b): each buffer binds read-only in
    the vertex stage and attribute fields read at @builtin(vertex_index)
    — classic vertex attributes are an L4 backend lowering, not a
    semantic. Args follow render(): vertex buffers first, then fragment
    args. Returns the (H, W) image in the reference's row convention
    (row 0 at NDC y = -1; WebGPU's y-up flips at readback)."""
    from pdum.dsl.pack import pack_into
    from pdum.tl.graphics import _check_pairing, _lower_fragment, _lower_vertex
    from pdum.tl.producer import _captured

    device = _device()
    import wgpu

    H, W = shape
    n_vs = pso.vs.fn.__code__.co_argcount
    vs_args, fs_args = args[:n_vs], args[n_vs:]
    v_region, vnames, flats, count, lattice, _v_ctx = _lower_vertex(pso.vs, tuple(vs_args))
    _check_pairing(frozenset(vnames), pso.required)  # buffer-taking shaders check here
    plan = list(_v_ctx.get("g.param_plan", ()))  # (buffer index, field | None) per region param
    buf_params = list(v_region.params[: len(plan)])
    used_bufs: set[int] = set()

    def _pulled_read(i: int, coords: dict) -> str:
        """Row-major element index over buffer i's own dims: vertex_id
        rides vid, component dims take their selected lattice constant."""
        dims = vs_args[i].layout.dims
        strides, acc = [], 1
        for d in reversed(dims):
            strides.append(acc)
            acc *= d.size
        parts = []
        for d, st in zip(dims, reversed(strides)):
            if d.name == "vertex_id":
                parts.append(f"i32(vid) * {st}" if d.start == 0 else f"(i32(vid) - {d.start}) * {st}")
            elif d.name in coords:
                parts.append(str((coords[d.name] - d.start) * st))
            else:
                raise Untranslatable(f"a vertex buffer dim {d.name!r} with no selected coordinate")
        used_bufs.add(i)
        return f"vb{i}[{' + '.join(parts)}]"

    def v_leaf(n, g):
        if n.op == "tl.iota" and dict(n.attrs).get("name") == "vertex_id":
            return "f32(vid)", False
        if n.op == "tl.select" and n.args and n.args[0] in buf_params:
            i, f = plan[buf_params.index(n.args[0])]
            if f is not None:
                raise Untranslatable("select on a record field param")
            return _pulled_read(i, _thaw_params(dict(n.attrs))["coords"]), False
        if n.op == "core.param":
            if n in buf_params:
                i, f = plan[buf_params.index(n)]
                if f is not None:  # a record field: the WGSL struct member, pulled at vid
                    used_bufs.add(i)
                    return f"vb{i}[vid].{f}", False
                return _pulled_read(i, {}), False  # a 1-D buffer used whole
            raise Untranslatable("a vertex-stage param beyond the pulled buffers")
        return None

    v_slots = sorted((dict(nd.attrs)["offset"], dict(nd.attrs)["fmt"]) for nd in _v_ctx["k.uniforms"].values())
    v_slot_index = {off: i for i, (off, _) in enumerate(v_slots)}

    def v_leaf_slots(n, g):
        if n.op == "abi.slot":  # the vertex stage's captured scalars (spun's angle)
            return f"VU[{v_slot_index[dict(n.attrs)['offset']]}]", False
        return v_leaf(n, g)

    vg = _Gen(v_leaf_slots)
    v_outs = [vg.operand(a) for a in v_region.body[-1].args[0].args]  # (px, py, *varyings)

    # the fragment, lowered over the pixel lattice with varyings as params
    ref_lat = Tensor.from_numpy(np.zeros((H, W)), ("y", "x"))
    from pdum.tl.dialect import tensor_type

    v_types = {n2: tensor_type(ref_lat) for n2 in vnames}
    f_region, f_ctx, _, fparams = _lower_fragment(pso.fs, tuple(fs_args), v_types)
    if f_ctx["k.fn_markers"]:
        raise Untranslatable("an oracle-class fragment fn-argument on the device")
    bound = dict(zip(fparams, fs_args))

    slots, staging = [], bytearray(f_ctx["k.uniform_size"])
    env = _captured(pso.fs.fn) if f_ctx["k.uniforms"] else {}
    for name, nd in f_ctx["k.uniforms"].items():
        at = dict(nd.attrs)
        struct.pack_into(at["fmt"], staging, at["offset"], env[name])
        slots.append((at["offset"], at["fmt"]))
    for blk in f_ctx["k.arg_plans"].values():
        f = bound[blk["pname"]] if blk["pname"] is not None else blk["fixed"]
        if blk["wrap"] is not None:
            f = blk["wrap"](f)
        view = memoryview(staging)[blk["base"] : blk["base"] + blk["plan"].staging_size]
        pack_into(blk["plan"], view, blk["extract"](f.captures, ()))
        for s2 in blk["plan"].slots:
            slots.append((blk["base"] + s2.dest.offset, s2.dest.fmt))
    slots.sort()
    slot_index = {off: i for i, (off, _) in enumerate(slots)}
    f_params = list(f_region.params)

    def f_leaf(n, g):
        if n.op == "core.param":
            return f"vin.f{f_params.index(n)}", False
        if n.op == "abi.slot":
            return f"U[{slot_index[dict(n.attrs)['offset']]}]", False
        return None

    fg = _Gen(f_leaf)
    f_out = fg.operand(f_region.body[-1].args[0])

    fields = []
    for i, n2 in enumerate(vnames):
        interp = " @interpolate(flat)" if n2 in flats else ""
        fields.append(f"  @location({i}){interp} f{i}: f32,")
    src_lines = [
        "struct VOut {",
        "  @builtin(position) pos: vec4<f32>,",
        *fields,
        "}",
    ]
    for i in sorted(used_bufs):  # only USED buffers declare (layout="auto" prunes)
        b = vs_args[i]
        if b.dtype.fields is not None:  # a RECORD buffer: a real WGSL struct (flat f32 v1)
            members = " ".join(f"{f}: f32," for f in b.dtype.names)
            src_lines.append(f"struct Rec{i} {{ {members} }}")
            src_lines.append(f"@group(0) @binding({i}) var<storage, read> vb{i}: array<Rec{i}>;")
        else:
            src_lines.append(f"@group(0) @binding({i}) var<storage, read> vb{i}: array<f32>;")
    vu_b = len(vs_args)
    if v_slots:
        src_lines.append(f"@group(0) @binding({vu_b}) var<storage, read> VU: array<f32>;")
    u_b = vu_b + (1 if v_slots else 0)
    if slots:
        src_lines.append(f"@group(0) @binding({u_b}) var<storage, read> U: array<f32>;")
    tex_pairs = f_ctx["g.textures"]
    base_b = u_b + (1 if slots else 0)
    for i in range(len(tex_pairs)):
        src_lines.append(f"@group(0) @binding({base_b + 2 * i}) var tex{i}: texture_2d<f32>;")
        src_lines.append(f"@group(0) @binding({base_b + 2 * i + 1}) var smp{i}: sampler;")
    src_lines += [
        "@vertex",
        "fn vs_main(@builtin(vertex_index) vid: u32) -> VOut {",
        *vg.lines,
        "  var o: VOut;",
        f"  o.pos = vec4<f32>({v_outs[0]}, {v_outs[1]}, 0.0, 1.0);",
        *[f"  o.f{i} = {v};" for i, v in enumerate(v_outs[2:])],
        "  return o;",
        "}",
        "@fragment",
        "fn fs_main(vin: VOut) -> @location(0) f32 {",
        *fg.lines,
        f"  return {f_out};",
        "}",
    ]
    source = "\n".join(src_lines)

    module = device.create_shader_module(code=source)
    pipeline = device.create_render_pipeline(
        layout="auto",
        vertex={"module": module, "entry_point": "vs_main", "buffers": []},
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        fragment={
            "module": module,
            "entry_point": "fs_main",
            "targets": [{"format": wgpu.TextureFormat.r32float}],
        },
    )
    tex = device.create_texture(
        size=(W, H, 1),
        format=wgpu.TextureFormat.r32float,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    enc = device.create_command_encoder()
    rp = enc.begin_render_pass(
        color_attachments=[
            {
                "view": tex.create_view(),
                "clear_value": (0.0, 0.0, 0.0, 0.0),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }
        ]
    )
    rp.set_pipeline(pipeline)
    entries = []
    for i in sorted(used_bufs):  # bind groups are OURS to build: automatic, never user-visible
        b = vs_args[i]
        if b.dtype.fields is not None:  # pack the record fields f32, dtype order = struct order
            flat = np.stack([b.field(f).to_numpy() for f in b.dtype.names], axis=-1).astype(np.float32)
        else:
            flat = np.ascontiguousarray(b.to_numpy(order=tuple(d.name for d in b.layout.dims)), dtype=np.float32)
        vbuf = device.create_buffer_with_data(data=flat.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        entries.append({"binding": i, "resource": {"buffer": vbuf, "offset": 0, "size": vbuf.size}})
    if v_slots:
        from pdum.tl.graphics import _env_staging

        v_staging = _env_staging(_v_ctx, pso.vs.fn)
        vvals = [struct.unpack_from(fmt, v_staging, off)[0] for off, fmt in v_slots]
        vubuf = device.create_buffer_with_data(
            data=np.asarray(vvals, dtype=np.float32).tobytes(), usage=wgpu.BufferUsage.STORAGE
        )
        entries.append({"binding": vu_b, "resource": {"buffer": vubuf, "offset": 0, "size": vubuf.size}})
    if slots:
        uvals = [struct.unpack_from(fmt, staging, off)[0] for off, fmt in slots]
        ubuf = device.create_buffer_with_data(
            data=np.asarray(uvals, dtype=np.float32).tobytes(), usage=wgpu.BufferUsage.STORAGE
        )
        entries.append({"binding": u_b, "resource": {"buffer": ubuf, "offset": 0, "size": ubuf.size}})
    for i, (t, s_) in enumerate(tex_pairs):
        entries.append({"binding": base_b + 2 * i, "resource": t.create_view()})
        entries.append({"binding": base_b + 2 * i + 1, "resource": s_})
    if entries:
        bind = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)
        rp.set_bind_group(0, bind)
    rp.draw(count)
    rp.end()
    bpr = (W * 4 + 255) // 256 * 256
    out_buf = device.create_buffer(size=bpr * H, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC)
    enc.copy_texture_to_buffer(
        {"texture": tex}, {"buffer": out_buf, "bytes_per_row": bpr, "rows_per_image": H}, (W, H, 1)
    )
    device.queue.submit([enc.finish()])
    raw = np.frombuffer(device.queue.read_buffer(out_buf), dtype=np.float32).reshape(H, bpr // 4)[:, :W]
    return np.flipud(raw).astype(np.float64)  # WebGPU y-up -> the reference row convention
