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
the vertex ambient (``thread_idx("vertex_id")`` — the one ambient
function, the draw domain as its lattice), data-flow branches
(``a if cond else b`` lowers to ``where``; ``or``/``and`` are mask
max/min — the committed S.4 spellings), and the position yield; the
fragment IS a step-kind body over the pixel lattice, so fn-valued
arguments splice through the same machinery as compute kernels.

Vertex ARRAYS are ordinary tensors over the ``vertex_id`` dim passed
as vertex-shader parameters — per-vertex attributes are ``.select()``
fields of them (records replace the component dim when the records arc
lands), and the draw count is the vertex_id extent. A shader with no
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
from pdum.dsl.cache import Memo
from pdum.dsl.ir import Region
from pdum.dsl.lower import Lowerer, check_coherence
from pdum.dsl.ops import CORE_OPS
from pdum.dsl.pack import ABI_OPS, pack_into
from pdum.dsl.registry import DEFAULT
from pdum.dsl.types import Type

from .coords import admit_frame
from .dialect import TL_OPS, CoordType, TensorType, capture_shim, check_tier, run_region, tensor_type
from .encoding import FormatEncoding
from .kernel import _ARG_BINDINGS, KERNEL_RULES, _code_fp, _env_fp, _lookup
from .lifting import _Intrinsic
from .producer import _captured, _fn_ast
from .tensor import Tensor, Token, _store

# The vertex ambient is thread_idx over the DRAW DOMAIN (250, the one
# ambient function; kind-dependent lattice): canonical dims "vertex_id"
# (and "instance_id" when instancing arrives with its consumer).


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
        if isinstance(obj, _Intrinsic) and obj.name == "thread_idx":
            # the ONE ambient function; the vertex kind's lattice is the
            # DRAW DOMAIN (250) — Coordinates over vertex_id/instance_id
            lattice = ctx.root.params[-1]  # hidden param, or the first vertex buffer
            out = []
            for cst in node.args:
                n = cst.value
                if n == "instance_id":
                    raise ValueError("instancing arrives with its consumer — the draw domain is vertex_id today")
                if n != "vertex_id":
                    raise ValueError(f'the vertex ambient is the draw domain: thread_idx("vertex_id"), not {n!r}')
                backing = ctx.emit("tl.iota", lattice, node=node, name=n)
                frame = next(f for f in lattice.type.dims if f.name == n)
                out.append(ctx.emit("tl.coord", backing, node=node, frame=frame))
            return tuple(out)
        if obj is position:
            x, y = (ctx.lower(a) for a in node.args)
            return _Position(x, y)
        if obj is flat:
            return _Flat(ctx.lower(node.args[0]))
    from .kernel import _k_call

    return _k_call(ctx, node)


def _v_assign(ctx, node):
    """Vertex bindings CLAIM (varyings — one law, this sink is the
    interpolator); ``flat(...)`` marks the site it binds. Host-scalar
    bindings claim too — a constant varying broadcasts over the vid
    lattice at the yield."""
    tgt = node.targets[0]
    if isinstance(tgt, ast.Name):
        value = ctx.lower(node.value)
        if isinstance(value, _Flat):
            ctx.context["g.flat"].add(tgt.id)
            value = value.value
        ctx.locals[tgt.id] = value
        tensorish = hasattr(value, "op") and isinstance(value.type, TensorType)
        if tensorish or isinstance(value, (int, float)):
            claims = ctx.context["g.claims"]
            if tgt.id in claims:
                ctx.context["g.invalid"][tgt.id] = "bound at more than one site"
            else:
                claims[tgt.id] = value
        return None
    from .kernel import _k_assign

    return _k_assign(ctx, node)


def _scalar_lift(ctx, node):
    from pdum.dsl.types import f64

    def lift(v):
        return v if hasattr(v, "op") else ctx.emit("core.const", node=node, type=f64, value=float(v))

    return lift


def _v_subscript(ctx, node):
    """``verts[v]`` on a RECORD buffer: the subscript law's record face —
    ONE Coordinate, admitted against the buffer's vertex_id frame (strict
    identity, containment extent); the element's fields come by NAME."""
    if isinstance(node.value, ast.Name):
        base = ctx.locals.get(node.value.id)
        if isinstance(base, _RecordBuffer):
            idx = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            vals = [ctx.lower(i) for i in idx]
            if len(vals) != 1 or not (hasattr(vals[0], "type") and isinstance(vals[0].type, CoordType)):
                raise TypeError(
                    'a record vertex buffer subscripts by ONE Coordinate — thread_idx("vertex_id")'
                )
            admit_frame(base.frame, vals[0].type.frame)
            return _RecordElement(base.fields)
    from .kernel import _k_subscript

    return _k_subscript(ctx, node)


VERTEX_RULES = {  # IfExp/BoolOp ride KERNEL_RULES now — the vertex spelling, promoted (211 §1.2)
    **KERNEL_RULES,
    ast.Call: _v_call,
    ast.Assign: _v_assign,
    ast.Subscript: _v_subscript,
}


def _f_call(ctx, node):
    """The fragment layer: ``sample(t, s, (row, col), lod=0)`` — the
    texture/sampler resolve to REGISTERED runtime objects (captures or
    locals), the coords lower as fields, and the node carries the ledger
    index plus the sampler's descriptor. v1 limits, recorded in
    BOUNDARIES.md: lod=0 only; textures in compute kernels are future
    work."""
    if isinstance(node.func, ast.Name):
        obj = _lookup(ctx, node.func.id)
        if obj is sample:
            t = _lookup(ctx, node.args[0].id)
            s_ = _lookup(ctx, node.args[1].id)
            if _TEX_MIRRORS is None or t not in _TEX_MIRRORS:
                raise ValueError("sample: the first argument must be an uploaded texture (upload(t))")
            if _SAMPLER_DESCS is None or s_ not in _SAMPLER_DESCS:
                raise ValueError("sample: the second argument must be a sampler (sampler(...))")
            lod = 0
            for kw in node.keywords:
                if kw.arg == "lod":
                    lod = ast.literal_eval(kw.value)
            if lod != 0:
                raise ValueError("sample: explicit lod=0 only today — mip chains are recorded future work")
            cy, cx = (ctx.lower(e) for e in node.args[2].elts)
            filt, addr = _SAMPLER_DESCS[s_]
            ledger = ctx.context["g.textures"]
            ledger.append((t, s_))
            return ctx.emit(
                "tl.sample", cy, cx, node=node, idx=len(ledger) - 1, lod=lod, filter=filt, address=addr
            )
    from .kernel import _k_call

    return _k_call(ctx, node)


FRAGMENT_RULES = {**KERNEL_RULES, ast.Call: _f_call}


# --- the two kinds -----------------------------------------------------------


@dataclass(frozen=True)
class VertexShader:
    fn: object

    def varyings(self) -> dict:
        """Introspection: every claimed site with its interpolation mode.
        Flat-ness is a production detail EXCLUDED from the interface type
        the fragment pairs against — it appears here, never there."""
        _, vnames, flats, _, _, _ = _lower_vertex(self, ())
        return {n: {"flat": n in flats} for n in vnames}


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
    Pairing is its own composition — never ``|``. ``pso[config(taps=...)]``
    binds fragment taps to render buffers at the pass — MRT."""

    vs: VertexShader
    fs: FragmentShader
    required: frozenset  # the fragment's inferred interface (names only)

    def __getitem__(self, cfg) -> "_BoundPSO":
        from .kernel import Config

        if not isinstance(cfg, Config):
            raise TypeError("pso[...] takes a config(...) object")
        if cfg.blocks is not None or cfg.threads is not None or cfg.shared_mem is not None:
            raise ValueError("a render pass takes taps only — the draw defines its own geometry")
        return _BoundPSO(self, cfg)


@dataclass(frozen=True)
class _BoundPSO:
    pso: PSO
    cfg: object


# The fragment ARTIFACT tier: the interface, content-keyed by the fragment's
# own fingerprints — one artifact serves EVERY vertex shader producing a
# superset, so ``pair(rich, shade)`` after ``pair(lean, shade)`` is a HIT
# (``fragment.miss`` never fires; the committed subset-pairing pin).
FRAGMENTS = Memo("fragment", capacity=1 << 20)


def _required_fields(fs: FragmentShader) -> frozenset:
    """The fragment's REQUIRED record, inferred from the fields its body
    touches (attribute access on the varying parameter — inference is the
    floor; a declared record type is honored when that door opens)."""
    fn = fs.fn
    names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    if not names:
        raise TypeError(f"{fn.__qualname__}: a fragment shader's last parameter is the varying record")
    vname = names[-1]
    tree = _fn_ast(fn)
    return frozenset(
        nd.attr
        for nd in ast.walk(tree)
        if isinstance(nd, ast.Attribute) and isinstance(nd.value, ast.Name) and nd.value.id == vname
    )


def pair(vs: VertexShader, fs: FragmentShader) -> PSO:
    """PSO pairing: the boundary is a record TYPE and strings never cross
    it — the check is produced ⊇ required (width subtyping), so adding a
    varying breaks no existing pairing."""
    if not isinstance(vs, VertexShader) or not isinstance(fs, FragmentShader):
        raise TypeError("pair(vertex, fragment) — PSO pairing takes one shader of each kind")
    key = (_code_fp(fs.fn), _env_fp(fs.fn))
    iface = FRAGMENTS.get_or_compile(key, lambda: {"required": _required_fields(fs)})
    required = iface["required"]
    if vs.fn.__code__.co_argcount == 0:  # buffer-taking shaders check at render, when buffers exist
        _, produced, _, _, _, _ = _lower_vertex(vs, ())
        _check_pairing(frozenset(produced), required)
    return PSO(vs, fs, required)


def _check_pairing(produced: frozenset, required: frozenset):
    missing = required - produced
    if missing:
        raise ValueError(
            f"pairing refused: the fragment requires varyings {sorted(required)} but the vertex "
            f"shader produces {sorted(produced)} — missing {sorted(missing)}"
        )


def _fresh_context(kind: str) -> dict:
    return {
        "registry": DEFAULT,
        "tl.kind": kind,
        "k.claims": {},
        "k.invalid": {},
        "k.claimed": set(),
        "k.fn_params": {},
        "k.fn_markers": {},
        "k.stored": [],
        "k.pname_of": {},
        "k.uniforms": {},
        "k.uniform_size": 0,
        "k.arg_plans": {},
        "k.geom": None,
        "k.grid_node": None,
        "g.claims": {},
        "g.invalid": {},
        "g.flat": set(),
        "g.textures": [],
    }


class _RecordBuffer:
    """A record-element vertex buffer at LOWER time: the structured dtype
    is the declaration (the 200 §4 encoding is the memory shape), each
    field a region param — pure lowering bookkeeping, zero new IR.
    ``verts[v]`` admits the Coordinate and yields the element."""

    def __init__(self, frame, fields: dict):
        self.frame = frame  # the vertex_id Frame
        self.fields = fields  # field name -> the field-view param node


class _RecordElement:
    """One record, per-vertex: fields by NAME (each a field over the draw
    lattice); anything else refuses listing the fields."""

    def __init__(self, fields: dict):
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name):
        if name not in self._fields:
            raise AttributeError(f"the record has no field {name!r} (fields: {sorted(self._fields)})")
        return self._fields[name]


def _record_fields(b: Tensor):
    """The v1 record-buffer contract: dims exactly (vertex_id,), flat
    float64 fields — nested/mixed records are a recorded boundary."""
    names = tuple(d.name for d in b.layout.dims)
    if names != ("vertex_id",):
        raise TypeError(f"a record vertex buffer's dims are exactly ('vertex_id',), got {names}")
    for fname, (fdt, _off) in b.dtype.fields.items():
        if fdt.kind != "f" or fdt.fields is not None:
            raise TypeError(
                f"record field {fname!r}: flat float fields only today ({fdt}) — "
                f"nested/mixed records are a recorded boundary (BOUNDARIES.md)"
            )
    return tuple(b.dtype.names)


def _lower_vertex(vs: VertexShader, buffers: tuple):
    """The vertex body over the draw domain. Region params: the vertex
    buffers — a RECORD buffer (structured dtype) expands to one param per
    FIELD (its field views; the complex-element door, automated) — then
    the hidden vertex_id lattice; yields (px, py, *varyings)."""
    fn = vs.fn
    handle = capture_shim(fn)
    check_coherence(handle)
    ctx = Lowerer(handle, VERTEX_RULES, {**CORE_OPS, **TL_OPS, **ABI_OPS}, {}, context=_fresh_context("vertex"))
    names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    if len(names) != len(buffers):
        raise TypeError(f"{fn.__qualname__} takes {len(names)} vertex buffers, got {len(buffers)}")
    for b in buffers:
        if not (isinstance(b, Tensor) and any(d.name == "vertex_id" for d in b.layout.dims)):
            raise TypeError("vertex buffers are tensors with a 'vertex_id' dim (per-vertex attributes)")
    counts = {b.layout.dim("vertex_id").size for b in buffers}
    if len(counts) > 1:  # the draw count DERIVES from the buffers: they must agree
        raise ValueError(f"vertex buffers disagree on the draw count: vertex_id extents {sorted(counts)}")
    count = buffers[0].layout.dim("vertex_id").size if buffers else 6  # no inputs: the screen quad's six ids
    lattice = Tensor.from_numpy(np.zeros(count), ("vertex_id",)) if not buffers else None
    params, plan, locals_ = [], [], []  # plan: (buffer index, field name | None) per param
    for i, b in enumerate(buffers):
        if b.dtype.fields is not None:  # a RECORD buffer: one param per field view
            fields = {}
            for fname in _record_fields(b):
                plan.append((i, fname))
                fields[fname] = ctx.builder.param(len(params), tensor_type(b.field(fname)))
                params.append(fields[fname])
            locals_.append(_RecordBuffer(b.layout.dim("vertex_id").frame, fields))
        else:
            plan.append((i, None))
            p = ctx.builder.param(len(params), tensor_type(b))
            params.append(p)
            locals_.append(p)
    hidden = ctx.builder.param(len(params), tensor_type(lattice)) if lattice is not None else params[0]
    all_params = tuple(params) + ((hidden,) if lattice is not None else ())
    ctx.params = all_params
    ctx.context["g.param_plan"] = tuple(plan)
    ctx.locals.update(zip(names, locals_))
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
        dims = tuple((d.name, (d.start, d.stop)) for d in hidden.type.dims)
        return ctx.builder.emit("tl.const", value=float(v), dims=dims) if not hasattr(v, "op") else lift(v)

    outs = (as_field(result.x), as_field(result.y)) + tuple(as_field(c["g.claims"][n]) for n in varying_names)
    yielded = ctx.builder.emit("core.tuple", *outs)
    region = check_tier(Region(params=all_params, body=(ctx.builder.emit("core.yield", yielded),)), "vertex")
    return region, varying_names, frozenset(c["g.flat"]), count, lattice, c


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


def _lower_fragment(fs: FragmentShader, fn_args: tuple, varying_types: dict, tap_names: tuple = ()):
    """The fragment IS a step-kind body over the pixel lattice: params
    are the varying fields; fn-valued arguments splice exactly as in
    compute kernels; the return is color0 (mandatory)."""
    fn = fs.fn
    handle = capture_shim(fn)
    check_coherence(handle)
    ctx = Lowerer(handle, FRAGMENT_RULES, {**CORE_OPS, **TL_OPS, **ABI_OPS}, {}, context=_fresh_context("fragment"))
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
    c = ctx.context
    tap_nodes = []
    for tname in tap_names:  # fragment taps: claimed sites sunk to render buffers (MRT)
        if tname in c["k.invalid"]:
            raise ValueError(f"tap {tname!r} is invalid: {c['k.invalid'][tname]}")
        if tname not in c["k.claims"]:
            raise ValueError(f"the fragment has no site named {tname!r} (sites: {sorted(c['k.claims'])})")
        tap_nodes.append(lift(c["k.claims"][tname]))
    yielded = lift(result) if not tap_nodes else ctx.builder.emit("core.tuple", lift(result), *tap_nodes)
    region = Region(params=tuple(params.values()), body=(ctx.builder.emit("core.yield", yielded),))
    check_tier(region, "fragment")
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


def _env_staging(c, fn) -> bytes:
    """A stage's captured-scalar staging bytes (the abi channel — the
    uniform doctrine reaching the graphics kinds)."""
    if not c["k.uniform_size"] or not c["k.uniforms"]:
        return b""
    staging = bytearray(c["k.uniform_size"])
    env = _captured(fn)
    for name, nd in c["k.uniforms"].items():
        at = dict(nd.attrs)
        struct.pack_into(at["fmt"], staging, at["offset"], env[name])
    return bytes(staging)


def render(pso, *args, target: Tensor):
    """One draw into the target: run the vertex stage over the vid
    lattice, rasterize through the reference interpolator, run the
    fragment stage over the covered pixel lattice, compose. The host
    owns the pass; this is the encodable's reference semantics. A bound
    PSO's taps write additional render targets (MRT), coverage-masked
    like color0; the buffers are invocation data."""
    taps = {}
    if isinstance(pso, _BoundPSO):
        taps = dict(pso.cfg.taps)
        pso = pso.pso
    n_vs = pso.vs.fn.__code__.co_argcount
    vs_args, fs_args = args[:n_vs], args[n_vs:]
    region, vnames, flats, count, lattice, v_ctx = _lower_vertex(pso.vs, tuple(vs_args))
    _check_pairing(frozenset(vnames), pso.required)  # the deferred half (buffer-taking shaders)
    plan = v_ctx.get("g.param_plan", ())
    values = [vs_args[i] if f is None else vs_args[i].field(f) for i, f in plan]
    values += [lattice] if lattice is not None else []
    outs = run_region(region, values, uniforms=_env_staging(v_ctx, pso.vs.fn))
    order = ("vertex_id",)
    px, py = outs[0].to_numpy(order=order), outs[1].to_numpy(order=order)
    varys = {n: v.to_numpy(order=order) for n, v in zip(vnames, outs[2:])}
    covered, fields = _rasterize(px, py, varys, flats, target)
    dim_names = tuple(d.name for d in target.layout.dims)
    f_tensors = {n: Tensor.from_numpy(a, dim_names) for n, a in fields.items()}
    v_types = {n: tensor_type(t) for n, t in f_tensors.items()}
    f_region, f_ctx, _, fparams = _lower_fragment(pso.fs, tuple(fs_args), v_types, tuple(taps))
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
    mirrors = [_TEX_MIRRORS[t] for t, _ in f_ctx["g.textures"]]
    out = run_region(f_region, [f_tensors[n] for n in v_types], uniforms=staging, textures=mirrors)
    color, tap_vals = (out, ()) if not taps else (out[0], out[1:])
    composed = np.where(covered, color.to_numpy(order=dim_names), target.to_numpy(order=dim_names))
    _store(Token(), target, Tensor.from_numpy(composed, dim_names))
    tap_types = [n.type for n in f_region.body[-1].args[0].args[1:]] if taps else []
    for (tname, buf), tv, tt in zip(taps.items(), tap_vals, tap_types):  # MRT: each tap a second target
        if isinstance(tv, tuple):  # a RECORD tap: per-field stores through the struct views (200 §4)
            fnames = [fn for fn, _ in tt.fields]
            if buf.dtype.names is None or list(buf.dtype.names) != fnames:
                raise TypeError(
                    f"record tap {tname!r} wants a struct-element buffer with fields {fnames} "
                    f"(in order); got dtype {buf.dtype}"
                )
            for fname, ftv in zip(fnames, tv):
                view = buf.field(fname)
                merged = np.where(covered, ftv.to_numpy(order=dim_names), view.to_numpy(order=dim_names))
                _store(Token(), view, Tensor.from_numpy(merged, dim_names))
            continue
        merged = np.where(covered, tv.to_numpy(order=dim_names), buf.to_numpy(order=dim_names))
        _store(Token(), buf, Tensor.from_numpy(merged, dim_names))
    return None


# --- textures v1 (S.4 amendment): runtime objects, ONE format ----------------

_RGBA8_SRGB = FormatEncoding("unorm8-srgb")
_GPU: list = []  # the lazy device singleton (first upload/sampler creates it)
# runtime-object ledgers (weak: the wgpu runtime owns the objects):
# texture -> its decoded linear-light mirror (what hardware sampling returns,
# quantization included) for REFERENCE sampling; sampler -> its descriptor
_TEX_MIRRORS = None  # created lazily (WeakKeyDictionary)
_SAMPLER_DESCS = None



@dataclass(frozen=True)
class VocabType(Type):
    """Captured VOCABULARY — an intrinsic riding a closure. Code, not
    data: identity is the name; nothing marshals."""

    name: str


class _VocabKind:
    def typeof(self, v, table):
        return VocabType(v.name)

    def fingerprint(self, v, table):
        return ("vocab", v.name)

    def flatten(self, v, table):
        return ()


DEFAULT.table.register(_Intrinsic, _VocabKind())


@dataclass(frozen=True)
class TextureType(Type):
    """The texture LEAF KIND (S.4): a runtime object the type system
    recognizes — never a dressed-up tensor. v1 has exactly one format."""

    fmt: str = "rgba8unorm-srgb"


@dataclass(frozen=True)
class SamplerType(Type):
    """The sampler leaf kind — the interpolation object."""


def _register_wgpu_kinds(wgpu):
    """RECOGNITION, not impersonation: the wgpu classes get leaf
    ValueKinds so textures/samplers capture and fingerprint as their own
    types (type-keyed, never by value)."""
    if getattr(_register_wgpu_kinds, "_done", False):
        return

    class _Leaf:
        def __init__(self, t):
            self._t = t

        def typeof(self, v, table):
            return self._t

        def fingerprint(self, v, table):
            return (type(self._t).__name__,)

        def flatten(self, v, table):
            return (v,)  # the leaves channel: runtime handles never byte-pack

    DEFAULT.table.register(wgpu.GPUTexture, _Leaf(TextureType()))
    DEFAULT.table.register(wgpu.GPUSampler, _Leaf(SamplerType()))
    _register_wgpu_kinds._done = True


def _device():
    """The WebGPU device. Textures and samplers are wgpu RUNTIME objects
    (owned by the runtime, recognized by our type system, never
    impersonated) — so they exist only where a WebGPU adapter does; the
    refusal names the dependency."""
    try:
        import wgpu
    except ImportError:
        raise RuntimeError(
            "textures are wgpu runtime objects — install wgpu (the workspace dev group carries it)"
        ) from None
    _register_wgpu_kinds(wgpu)
    if not _GPU:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        _GPU.append(adapter.request_device_sync())
    return _GPU[0]


def upload(t: Tensor):
    """The upload door: a TEXTURE built from a tensor — rgba8unorm-srgb,
    the ONE v1 format, encoded through the existing FormatEncoding (the
    format registry is later work). The result is the wgpu texture —
    never a Tensor."""
    import wgpu

    global _TEX_MIRRORS
    if _TEX_MIRRORS is None:
        import weakref

        _TEX_MIRRORS = weakref.WeakKeyDictionary()
    hd, wd = t.layout.dims
    lum = _RGBA8_SRGB.encode(t.to_numpy(order=(hd.name, wd.name)))
    rgba = np.stack([lum, lum, lum, np.full_like(lum, 255)], axis=-1)  # luminance replicated, alpha 1
    device = _device()
    tex = device.create_texture(
        size=(wd.size, hd.size, 1),
        format=wgpu.TextureFormat.rgba8unorm_srgb,
        usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
    )
    device.queue.write_texture(
        {"texture": tex},
        np.ascontiguousarray(rgba).tobytes(),
        {"bytes_per_row": 4 * wd.size, "rows_per_image": hd.size},
        (wd.size, hd.size, 1),
    )
    _TEX_MIRRORS[tex] = _RGBA8_SRGB.decode(lum)  # what hardware sampling returns
    return tex


def sampler(*, filter: str = "linear", address: str = "clamp"):
    """The SAMPLER — the interpolation object, a wgpu runtime object like
    the texture. v1 vocabulary: filter linear|nearest, address
    clamp|repeat."""
    import wgpu

    modes = {"clamp": wgpu.AddressMode.clamp_to_edge, "repeat": wgpu.AddressMode.repeat}
    filters = {"linear": wgpu.FilterMode.linear, "nearest": wgpu.FilterMode.nearest}
    if filter not in filters or address not in modes:
        raise ValueError(f"sampler: filter is linear|nearest, address is clamp|repeat (got {filter!r}, {address!r})")
    global _SAMPLER_DESCS
    if _SAMPLER_DESCS is None:
        import weakref

        _SAMPLER_DESCS = weakref.WeakKeyDictionary()
    smp = _make_sampler(modes, filters, filter, address)
    _SAMPLER_DESCS[smp] = (filter, address)
    return smp


def _make_sampler(modes, filters, filter, address):
    return _device().create_sampler(
        address_mode_u=modes[address],
        address_mode_v=modes[address],
        mag_filter=filters[filter],
        min_filter=filters[filter],
    )


# Sampling vocabulary: lowers inside device bodies. Explicit-LOD first;
# auto-LOD is ANALYTIC (log2 footprint from the wrt-ambient gradient, no
# 2x2 quad) — the lowering arrives with the conformance executor's
# translation table, where sampling has real hardware semantics to match.
sample = _Intrinsic("sample")
