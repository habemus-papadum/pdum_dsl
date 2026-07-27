"""@vertex/@fragment — the graphics tier (200 §S.4, P8).

The kind rule: RETURN IS MANDATORY, CLAIMS ARE OPTIONAL — a vertex
shader returns exactly ``position(x, y)``, a fragment shader returns
exactly color0; everything else is claimed by naming it (the one
claiming law, tagless). A varying is a claimed site whose sink is the
INTERPOLATOR; interpolation is declared at the vertex site
(perspective-correct by default, ``flat(...)`` the sole annotation) and
is a production detail excluded from the interface. The
vertex→fragment boundary is a record TYPE: the fragment consumes by
attribute access and its required record is INFERRED from the fields
it touches; pairing checks produced ⊇ required.

Both kinds lower through the ONE dsl Lowerer as rule-pack layers over
the kernel/tl packs (the dialect-hierarchy law): the vertex pack adds
the vertex ambient (``vertex_index`` — the raw), data-flow branches
(``a if cond else b`` lowers to ``where``; ``or``/``and`` are mask
max/min — the committed S.4 spellings), and the position yield; the
fragment IS a step-kind body over the pixel lattice, so fn-valued
arguments splice through the same machinery as compute kernels.

Vertex ARRAYS are ordinary tensors over the ``vid`` dim passed as
vertex-shader parameters — per-vertex attributes are ``.select()``
fields of them, and the draw count is the vid extent. A shader with no
vertex inputs draws the screen quad (six ids, two triangles).

The rasterizer here is the MINIMAL REFERENCE INTERPOLATOR (S.4):
barycentric coverage at pixel centers, linear interpolation (flat
takes the provoking vertex), triangles composed in order. The GPU path
arrives with the conformance executor and the L4-era backends.
"""

from __future__ import annotations

import ast
import struct
from dataclasses import dataclass

import numpy as np
from pdum.dsl.ir import Region
from pdum.dsl.lower import Lowerer, check_coherence
from pdum.dsl.ops import CORE_OPS
from pdum.dsl.pack import ABI_OPS, pack_into
from pdum.dsl.registry import DEFAULT

from .dialect import TL_OPS, TensorType, capture_shim, run_region, tensor_type
from .ir import Token, _store
from .kernel import _ARG_BINDINGS, KERNEL_RULES, _lookup
from .lifting import _Intrinsic
from .producer import _captured, _fn_ast
from .tensor import Tensor

vertex_index = _Intrinsic("vertex_index")
instance_index = _Intrinsic("instance_index")  # reserved: instancing arrives with a consumer


@dataclass(frozen=True)
class _Position:
    """The mandatory vertex return: clip-space (x, y). A host-tagged pair
    at lower time — the vertex kind's yield protocol consumes it."""

    x: object
    y: object


def position(x, y) -> _Position:
    return _Position(x, y)


@dataclass(frozen=True)
class _Flat:
    """``flat(v)``: the sole interpolation annotation — the claimed site
    takes the provoking vertex's value, no interpolation."""

    value: object


def flat(v) -> _Flat:
    return _Flat(v)


# --- the vertex rule pack: a layer over the kernel pack ----------------------


def _v_call(ctx, node):
    if isinstance(node.func, ast.Name):
        obj = _lookup(ctx, node.func.id)
        if isinstance(obj, _Intrinsic) and obj.name == "vertex_index":
            lattice = ctx.root.params[-1]  # the vid lattice (hidden param, or the first vertex buffer)
            return ctx.emit("tl.iota", lattice, node=node, name="vid")
        if obj is position:
            x, y = (ctx.lower(a) for a in node.args)
            return _Position(x, y)
        if obj is flat:
            return _Flat(ctx.lower(node.args[0]))
    from .kernel import _k_call

    return _k_call(ctx, node)


def _v_assign(ctx, node):
    """Vertex bindings CLAIM (varyings — one law, this sink is the
    interpolator); ``flat(...)`` marks the site it binds."""
    tgt = node.targets[0]
    if isinstance(tgt, ast.Name):
        value = ctx.lower(node.value)
        if isinstance(value, _Flat):
            ctx.context["g.flat"].add(tgt.id)
            value = value.value
        ctx.locals[tgt.id] = value
        if hasattr(value, "op") and isinstance(value.type, TensorType):
            claims = ctx.context["g.claims"]
            if tgt.id in claims:
                ctx.context["g.invalid"][tgt.id] = "bound at more than one site"
            else:
                claims[tgt.id] = value
        return None
    from .kernel import _k_assign

    return _k_assign(ctx, node)


def _v_ifexp(ctx, node):
    """The committed vertex spelling ``a if cond else b`` over tensor
    predicates is DATA FLOW: it lowers to ``where`` (both sides
    evaluated); host conditions branch on the host."""
    cond = ctx.lower(node.test)
    if hasattr(cond, "op") and isinstance(cond.type, TensorType):
        a, b = ctx.lower(node.body), ctx.lower(node.orelse)
        lift = _scalar_lift(ctx, node)
        return ctx.emit("tl.pointwise", cond, lift(a), lift(b), node=node, f="where")
    return ctx.lower(node.body) if cond else ctx.lower(node.orelse)


def _v_boolop(ctx, node):
    """``or``/``and`` over predicate masks are max/min — data flow, the
    committed spelling; host booleans stay host."""
    vals = [ctx.lower(v) for v in node.values]
    if any(hasattr(v, "op") and isinstance(v.type, TensorType) for v in vals):
        f = "maximum" if isinstance(node.op, ast.Or) else "minimum"
        lift = _scalar_lift(ctx, node)
        out = lift(vals[0])
        for v in vals[1:]:
            out = ctx.emit("tl.pointwise", out, lift(v), node=node, f=f)
        return out
    return any(vals) if isinstance(node.op, ast.Or) else all(vals)


def _scalar_lift(ctx, node):
    from pdum.dsl.types import f64

    def lift(v):
        return v if hasattr(v, "op") else ctx.emit("core.const", node=node, type=f64, value=float(v))

    return lift


VERTEX_RULES = {
    **KERNEL_RULES,
    ast.Call: _v_call,
    ast.Assign: _v_assign,
    ast.IfExp: _v_ifexp,
    ast.BoolOp: _v_boolop,
}


# --- the two kinds -----------------------------------------------------------


@dataclass(frozen=True)
class VertexShader:
    fn: object


@dataclass(frozen=True)
class FragmentShader:
    fn: object


def vertex(fn) -> VertexShader:
    return VertexShader(fn)


def fragment(fn) -> FragmentShader:
    return FragmentShader(fn)


@dataclass(frozen=True)
class PSO:
    """The PAIR is the artifact unit (real APIs compile PSOs whole).
    Pairing is its own composition — never ``|``."""

    vs: VertexShader
    fs: FragmentShader


def pair(vs: VertexShader, fs: FragmentShader) -> PSO:
    if not isinstance(vs, VertexShader) or not isinstance(fs, FragmentShader):
        raise TypeError("pair(vertex, fragment) — PSO pairing takes one shader of each kind")
    return PSO(vs, fs)


def _fresh_context(kind: str) -> dict:
    return {
        "registry": DEFAULT,
        "tl.kind": kind,
        "k.claims": {},
        "k.invalid": {},
        "k.claimed": set(),
        "k.fn_params": {},
        "k.fn_markers": {},
        "k.iotas": [],
        "k.stored": [],
        "k.pname_of": {},
        "k.uniforms": {},
        "k.uniform_size": 0,
        "k.arg_plans": {},
        "k.geom": None,
        "k.globals": {},
        "k.grid_node": None,
        "g.claims": {},
        "g.invalid": {},
        "g.flat": set(),
    }


def _lower_vertex(vs: VertexShader, buffers: tuple):
    """The vertex body over the vid lattice. Region params: the vertex
    buffers, then the hidden vid lattice; yields (px, py, *varyings).
    Returns (region, varying names in claim order, flat set, count)."""
    fn = vs.fn
    handle = capture_shim(fn)
    check_coherence(handle)
    ctx = Lowerer(handle, VERTEX_RULES, {**CORE_OPS, **TL_OPS, **ABI_OPS}, {}, context=_fresh_context("compute"))
    names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    if len(names) != len(buffers):
        raise TypeError(f"{fn.__qualname__} takes {len(names)} vertex buffers, got {len(buffers)}")
    for b in buffers:
        if not (isinstance(b, Tensor) and any(d.name == "vid" for d in b.layout.dims)):
            raise TypeError("vertex buffers are tensors with a 'vid' dim (per-vertex attributes)")
    count = buffers[0].layout.dim("vid").size if buffers else 6  # no inputs: the screen quad's six ids
    lattice = Tensor.from_numpy(np.zeros(count), ("vid",)) if not buffers else None
    params = []
    for i, b in enumerate(buffers):
        params.append(ctx.builder.param(i, tensor_type(b)))
    hidden = ctx.builder.param(len(params), tensor_type(lattice)) if lattice is not None else params[0]
    all_params = tuple(params) + ((hidden,) if lattice is not None else ())
    ctx.params = all_params
    ctx.locals.update(zip(names, params))
    tree = _fn_ast(fn)
    result = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Return):
            result = ctx.lower(stmt.value)
            break
        ctx.lower(stmt)
    if not isinstance(result, _Position):
        raise ValueError(f"{fn.__qualname__}: a vertex shader must return position(x, y) — return is MANDATORY")
    c = ctx.context
    varying_names = tuple(c["g.claims"])
    lift = _scalar_lift(ctx, tree)

    def as_field(v):  # scalars broadcast over the vid lattice
        if hasattr(v, "op") and isinstance(v.type, TensorType):
            return v
        dims = tuple((d[0], (d[1], d[2])) for d in hidden.type.dims)
        return ctx.builder.emit("tl.const", value=float(v), dims=dims) if not hasattr(v, "op") else lift(v)

    outs = (as_field(result.x), as_field(result.y)) + tuple(c["g.claims"][n] for n in varying_names)
    yielded = ctx.builder.emit("core.tuple", *outs)
    region = Region(params=all_params, body=(ctx.builder.emit("core.yield", yielded),))
    return region, varying_names, frozenset(c["g.flat"]), count, lattice


class _VaryingProbe:
    """The fragment's varying record: attribute access yields the
    interpolated FIELD and records the touch — the required record is
    INFERRED from use (declaring a type is honored later; inference is
    the floor)."""

    def __init__(self, fields: dict, touched: set):
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_touched", touched)

    def __getattr__(self, name):
        if name not in self._fields:
            raise AttributeError(
                f"the paired vertex shader produces no varying {name!r} (produced: {sorted(self._fields)})"
            )
        self._touched.add(name)
        return self._fields[name]


def _lower_fragment(fs: FragmentShader, fn_args: tuple, varying_types: dict):
    """The fragment IS a step-kind body over the pixel lattice: params
    are the varying fields; fn-valued arguments splice exactly as in
    compute kernels; the return is color0 (mandatory)."""
    fn = fs.fn
    handle = capture_shim(fn)
    check_coherence(handle)
    ctx = Lowerer(handle, KERNEL_RULES, {**CORE_OPS, **TL_OPS, **ABI_OPS}, {}, context=_fresh_context("compute"))
    names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    if not names:
        raise TypeError(f"{fn.__qualname__}: a fragment shader's last parameter is the varying record")
    fparams, vname = names[:-1], names[-1]
    if len(fparams) != len(fn_args):
        raise TypeError(f"{fn.__qualname__} takes {len(fparams)} arguments before the varying, got {len(fn_args)}")
    params = {n: ctx.builder.param(i, t) for i, (n, t) in enumerate(varying_types.items())}
    ctx.params = tuple(params.values())
    touched: set = set()
    ctx.locals[vname] = _VaryingProbe(params, touched)
    ctx.locals.update(zip(fparams, fn_args))
    ctx.context["k.fn_params"] = dict(zip(fparams, fn_args))  # fn-args rebind like kernel params
    tree = _fn_ast(fn)
    result = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Return):
            result = ctx.lower(stmt.value)
            break
        ctx.lower(stmt)
    if result is None:
        raise ValueError(f"{fn.__qualname__}: a fragment shader must return color0 — return is MANDATORY")
    lift = _scalar_lift(ctx, tree)
    region = Region(params=tuple(params.values()), body=(ctx.builder.emit("core.yield", lift(result)),))
    return region, ctx.context, frozenset(touched), tuple(fparams)


def _rasterize(px, py, varys: dict, flats: frozenset, target: Tensor):
    """The minimal reference interpolator: barycentric coverage at pixel
    centers over the target lattice; linear interpolation; flat takes
    the provoking (first) vertex; triangles compose in order."""
    (hd, wd) = target.layout.dims
    H, W = hd.size, wd.size
    xs = (np.arange(W) + 0.5) / W * 2.0 - 1.0
    ys = (np.arange(H) + 0.5) / H * 2.0 - 1.0
    X, Y = np.meshgrid(xs, ys)
    covered = np.zeros((H, W), dtype=bool)
    fields = {k: np.zeros((H, W)) for k in varys}
    for t in range(len(px) // 3):
        i0, i1, i2 = 3 * t, 3 * t + 1, 3 * t + 2
        d = (py[i1] - py[i2]) * (px[i0] - px[i2]) + (px[i2] - px[i1]) * (py[i0] - py[i2])
        if d == 0.0:
            continue  # a degenerate triangle covers nothing
        w0 = ((py[i1] - py[i2]) * (X - px[i2]) + (px[i2] - px[i1]) * (Y - py[i2])) / d
        w1 = ((py[i2] - py[i0]) * (X - px[i2]) + (px[i0] - px[i2]) * (Y - py[i2])) / d
        w2 = 1.0 - w0 - w1
        cov = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        for k, vals in varys.items():
            interp = vals[i0] if k in flats else w0 * vals[i0] + w1 * vals[i1] + w2 * vals[i2]
            fields[k] = np.where(cov, interp, fields[k])
        covered |= cov
    return covered, fields


def render(pso: PSO, *args, target: Tensor):
    """One draw into the target: run the vertex stage over the vid
    lattice, rasterize through the reference interpolator, run the
    fragment stage over the covered pixel lattice, compose. The host
    owns the pass; this is the encodable's reference semantics."""
    n_vs = pso.vs.fn.__code__.co_argcount
    vs_args, fs_args = args[:n_vs], args[n_vs:]
    region, vnames, flats, count, lattice = _lower_vertex(pso.vs, tuple(vs_args))
    values = list(vs_args) + ([lattice] if lattice is not None else [])
    outs = run_region(region, values)
    order = ("vid",)
    px, py = outs[0].to_numpy(order=order), outs[1].to_numpy(order=order)
    varys = {n: v.to_numpy(order=order) for n, v in zip(vnames, outs[2:])}
    covered, fields = _rasterize(px, py, varys, flats, target)
    dim_names = tuple(d.name for d in target.layout.dims)
    f_tensors = {n: Tensor.from_numpy(a, dim_names) for n, a in fields.items()}
    v_types = {n: tensor_type(t) for n, t in f_tensors.items()}
    f_region, f_ctx, _, fparams = _lower_fragment(pso.fs, tuple(fs_args), v_types)
    bound = dict(zip(fparams, fs_args))
    for pname_, mnames in f_ctx["k.fn_markers"].items():  # oracle-class fn-args rebind
        for mname in mnames:
            _ARG_BINDINGS[mname] = bound[pname_]
    staging = b""
    if f_ctx["k.uniform_size"]:
        staging = bytearray(f_ctx["k.uniform_size"])
        env = _captured(pso.fs.fn) if f_ctx["k.uniforms"] else {}
        for name, nd in f_ctx["k.uniforms"].items():
            at = dict(nd.attrs)
            struct.pack_into(at["fmt"], staging, at["offset"], env[name])
        for blk in f_ctx["k.arg_plans"].values():
            f = bound[blk["pname"]] if blk["pname"] is not None else blk["fixed"]
            if blk["wrap"] is not None:
                f = blk["wrap"](f)
            view = memoryview(staging)[blk["base"] : blk["base"] + blk["plan"].staging_size]
            pack_into(blk["plan"], view, blk["extract"](f.captures, ()))
        staging = bytes(staging)
    color = run_region(f_region, [f_tensors[n] for n in v_types], uniforms=staging)
    composed = np.where(covered, color.to_numpy(order=dim_names), target.to_numpy(order=dim_names))
    _store(Token(), target, Tensor.from_numpy(composed, dim_names))
    return None
