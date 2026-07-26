"""@compute kernels — lowered through the ONE dsl Lowerer (240 C4.2).

A kernel body IS the value language plus the dialect extensions (the
one-body-language law, 200 §S.3 amendment): the thread AMBIENT
(``thread_idx("y", "x")`` — coordinates are never positional
parameters), explicit token-threaded STORES into writable arguments
(``img[y, x] = v``), and — arriving P8/P9 — buffer READS at computed
indices. Function-valued arguments apply at the thread coordinates,
including tuple-returning ones (``v, (dy, dx) = f(y, x)`` — the
destructuring pattern declares the structure). Writability is inferred
from the body: an argument is writable iff it is stored to.

Since the pivot's kernel switch, kernels lower through the DSL Lowerer
with the KERNEL RULE PACK — a layer over the tl dialect pack, which is a
layer over the base value pack — onto ``tl.*`` Regions (``tl.iota`` over
the writable lattice, ``tl.pointwise`` bodies, token-threaded
``tl.store``), and ``run_region`` executes them. The iota unification is
now literal: thread coordinates ARE ``tl.iota`` nodes.

Claiming is TAGLESS (S.4 amendment): every uniquely-named binding is a
claimable site — ``config(taps={"dist": t})`` binds the binding named
``dist``; a name bound more than once is invalidated with the reason,
never auto-suffixed.

**Function-valued arguments** carry their FnType identity (``handle.fp``:
code + env TYPES, never values) in the kernel key: swapping a stage is a
new artifact; new captured values on the same shape are a WARM HIT, with
the values riding a per-launch rebind (the uniform channel at reference
tier). Per-element host dispatch through the spelled oracle is the
sanctioned oracle-grade execution. A body may also host-STAGE a declared
``@staged`` transform of a parameter (``g = value_and_grad(f, ...)``) —
the recipe replays on the CURRENT binding every launch.

**Invocation is the bracket** — ``kernel[config(blocks, threads,
taps={...})](args)``; geometry is VALIDATED LAUNCHER DATA: one entry
binds one geometry, launches are checked against it, and it never
enters identity (specialization-on-geometry is a declared policy door
for the backend era); the tap NAME SET specializes, tap tensors are
invocation data.

Day-one contract (210): a writable argument overlapping any READABLE
argument refuses at dispatch with the ping-pong message; in-place returns
only ever as an L2-certified rewrite.
"""

from __future__ import annotations

import ast
import hashlib
import struct
from dataclasses import dataclass

import numpy as np
from pdum.dsl import events
from pdum.dsl.cache import ArtifactCache, Memo
from pdum.dsl.ir import Node, Region
from pdum.dsl.lower import Lowerer, check_coherence, lower_handle
from pdum.dsl.ops import CORE_OPS
from pdum.dsl.pack import _FMTS, ABI_OPS, NORMALIZE_ENV, build_extractor, pack_into, plan_from_types
from pdum.dsl.registry import DEFAULT
from pdum.dsl.rewrite import run_stage
from pdum.dsl.types import LiteralValue, f64
from pdum.dsl.value import _assign as _value_assign
from pdum.dsl.value import _name as _value_name
from pdum.dsl.value import _subscript as _value_subscript
from pdum.dsl.valuekind import typeof

from .dialect import (
    TL_OPS,
    TL_RULES,
    TensorType,
    _globals_of,
    _lookup,
    _tl_call,
    capture_shim,
    run_region,
    tensor_type,
    walk_region,
)
from .lifting import _Intrinsic
from .markers import Marker
from .producer import _captured
from .tensor import Tensor

KERNELS = Memo("kernel", capacity=1 << 30)  # tier 1: the specialization -> launch plan
ARTIFACTS = ArtifactCache()  # tier 2, content-addressed: identical IR builds ONE executor (240 C5)
# The backend column's kernel seat. The numpy interpreter is the one executor
# today; the conformance executor (P8, wgpu) brings the second value — its
# render+compile slots into _executor behind this same content key.
_EXECUTOR_FP = ("run_region", "numpy")
_ARG_BINDINGS: dict[str, object] = {}  # marker name -> the CURRENT handle (per launch)

thread_idx = _Intrinsic("thread_idx")
block_idx = _Intrinsic("block_idx")  # the RAW pair's other half (S.3: raws are the primitives)


@dataclass(frozen=True)
class Shared:
    """Reserved: shared-memory declarations (the tile tier, L4)."""

    layouts: tuple


def shared(**layouts) -> Shared:
    return Shared(tuple(sorted(layouts.items())))


@dataclass(frozen=True)
class Config:
    """The bracket config (040 §3c's contract, 200-era): each component
    declares its SPECIALIZATION REGIME, defaulting to invocation-only —
    blocks/threads are VALIDATED LAUNCHER DATA (owner-ruled, P8): one
    entry binds one geometry, every launch is checked against it, and
    geometry never enters identity — specialization-on-geometry is a
    declared policy door for the backend era; the tap NAME SET
    specializes (a different tap set is a different artifact, the P5
    identity law) while the tap TENSORS are invocation data; shared_mem
    is structural (specializes) and arrives with the tile tier."""

    blocks: tuple | None = None
    threads: tuple | None = None
    taps: tuple = ()  # sorted (name, tensor) pairs
    shared_mem: Shared | None = None


def config(blocks=None, threads=None, taps=None, shared_mem=None) -> Config:
    return Config(
        blocks=blocks,
        threads=threads,
        taps=tuple(sorted((taps or {}).items())),
        shared_mem=shared_mem,
    )


@dataclass(frozen=True)
class ComputeKernel:
    fn: object

    def __call__(self, *args):
        return self._invoke(Config(), args)

    def __getitem__(self, cfg: Config) -> "_Bound":
        if not isinstance(cfg, Config):
            raise TypeError("kernel[...] takes a config(...) object")
        return _Bound(self, cfg)

    def _invoke(self, cfg: Config, args):
        if cfg.shared_mem is not None:
            raise NotImplementedError(
                "shared memory arrives with the tile tier (L4) — the config "
                "slot is reserved; see test_kernel_spec for the committed syntax"
            )
        tap_names = tuple(n for n, _ in cfg.taps)
        geom = (cfg.blocks, cfg.threads) if cfg.blocks is not None or cfg.threads is not None else None
        key = (_code_fp(self.fn), _env_fp(self.fn), tuple(_arg_fp(a) for a in args), tap_names)
        art = KERNELS.get_or_compile(key, lambda: _compile(self.fn, args, tap_names, geom))
        if _norm_geom(geom) != _norm_geom(art.geometry):
            # geometry is VALIDATED LAUNCHER DATA (owner-ruled): it never keys
            # and never re-renders — one entry binds one geometry, checked here;
            # a specialization POLICY is a declared door for the backend era
            raise ValueError(
                f"this kernel entry was built under geometry {art.geometry!r} and the launch "
                f"presents {geom!r} — geometry is validated launcher data, never identity; "
                f"launch with the geometry the entry was built under"
            )
        if geom is not None and _norm_geom(geom) is None:  # an explicit one-block launch
            _check_coverage(geom, art, args)
        return art.launch(args, dict(cfg.taps))

    def taps(self, *args) -> dict:
        """Introspection: the kernel's tap SITES for these argument shapes —
        {name: {"valid": bool, "dims": (...) | None, "reason": str | None}}.
        Compiles (cached) with an empty tap set to discover the sites."""
        key = (_code_fp(self.fn), _env_fp(self.fn), tuple(_arg_fp(a) for a in args), ())
        art = KERNELS.get_or_compile(key, lambda: _compile(self.fn, args, ()))
        out = {}
        for name, dims in art.tap_sites.items():
            out[name] = {"valid": True, "dims": dims, "reason": None}
        for name, reason in art.invalid_taps.items():
            out[name] = {"valid": False, "dims": None, "reason": reason}
        return out


@dataclass(frozen=True)
class _Bound:
    kernel: ComputeKernel
    cfg: Config

    def __call__(self, *args):
        return self.kernel._invoke(self.cfg, args)


def compute(fn) -> ComputeKernel:
    """Mark a compute kernel: explicit stores, ambient thread coordinates."""
    return ComputeKernel(fn)


def _code_fp(fn) -> tuple:
    code = fn.__code__
    return (code.co_qualname, hashlib.sha256(code.co_code).hexdigest()[:16])


def _env_fp(fn, _seen=None) -> tuple:
    """Fingerprint the captured environment the body can SEE (referenced
    names only) — the kernel key's answer to the dsl tier's guards (240 C1):
    a rebound global or an edited helper is a MISS, never a stale artifact.
    User functions recurse; pdum.* library callables key by qualified name
    (stable within a process); opaque objects key by type — types over
    values, as everywhere."""
    import builtins

    seen = set() if _seen is None else _seen
    if fn.__code__ in seen:
        return ("cycle",)
    seen.add(fn.__code__)
    env, code = _captured(fn), fn.__code__
    out = []
    for n in sorted(set(code.co_names) | set(code.co_freevars)):
        if n not in env or env[n] is getattr(builtins, n, _env_fp):
            continue
        v = env[n]
        if isinstance(v, Marker):
            out.append((n, "marker", v.name))
        elif getattr(v, "fp", None) is not None:
            out.append((n, "fn", v.fp))
        elif isinstance(v, LiteralValue):  # declared structural: the VALUE keys
            out.append((n, "lit", v.value))
        elif isinstance(v, (int, float, bool)):  # unmarked scalars are DATA: type keys, value flows
            out.append((n, "data", type(v).__name__))
        elif isinstance(v, (str, bytes, type(None))):
            out.append((n, "val", v))
        elif callable(v) and hasattr(v, "__code__"):
            mod = getattr(v, "__module__", "") or ""
            if mod.startswith("pdum."):
                out.append((n, "lib", mod, v.__qualname__))
            else:
                out.append((n, "code", _code_fp(v), _env_fp(v, seen)))
        elif isinstance(v, type(hashlib)):  # a module
            out.append((n, "mod", v.__name__))
        else:
            out.append((n, "type", type(v).__qualname__))
    return tuple(out)


def _arg_fp(a) -> tuple:
    fp = getattr(a, "fp", None)
    if fp is not None:  # a Handle/Pipeline: FnType identity — types, never values
        return ("fn", fp)
    if isinstance(a, Tensor):
        # the fingerprint IS the type identity (frame included) + dtype, the
        # compute-layer fact the type does not yet carry
        return ("tensor", tensor_type(a).dims, str(a.dtype))
    raise TypeError(f"@compute arguments are tensors or kernel values, got {a!r}")


# ---- the kernel rule pack: a layer over the dialect pack --------------------


def _claim(ctx, name, value):
    """The naming law IS the claiming mechanism (tagless, S.4 amendment):
    every uniquely-named binding is a site; a name bound more than once is
    invalidated with the reason, never auto-suffixed."""
    c = ctx.context
    if name in c["k.claimed"]:
        c["k.claims"].pop(name, None)
        c["k.invalid"][name] = "bound at more than one site (rebinding or inlining made it non-unique)"
        return
    c["k.claimed"].add(name)
    if isinstance(value, Node) and isinstance(value.type, TensorType):
        c["k.claims"][name] = value
    else:
        c["k.invalid"][name] = "not a lattice value (nothing to store)"


def _lower_args(ctx, call):
    """Lower call arguments tuple-aware: an AST tuple stays a Python tuple
    of nodes (coordinate PAIRS ride as tuples through pipelines)."""
    return tuple(tuple(ctx.lower(e) for e in a.elts) if isinstance(a, ast.Tuple) else ctx.lower(a) for a in call.args)


def _norm_geom(geom):
    """Geometry, semantically: all-ones blocks (or no blocks) IS the
    default one-block geometry — identical meaning, identical IR — so
    both bind one entry. Multi-block geometry compares verbatim."""
    if geom is None:
        return None
    blocks, _ = geom
    if not blocks or all(b == 1 for b in blocks):
        return None
    return geom


def _check_coverage(geom, art, args):
    """Launcher-data validation for an explicit one-block launch: the
    block's threads must COVER the writable lattice (over-provisioned
    threads idle, like a device's guarded excess; under-provisioning
    would silently write nothing, so it refuses)."""
    _, threads = geom
    if not threads or not art.tensor_params:
        return
    target = args[art.params.index(art.tensor_params[-1])]
    dims = target.layout.dims
    for i, thr in enumerate(threads):
        if i < len(dims) and thr < dims[i].size:
            raise ValueError(
                f"config(threads={threads}) does not cover the writable lattice "
                f"{tuple((d.name, d.size) for d in dims)} — a one-block launch spans it "
                f"(over-provisioned threads idle; under-provisioning writes nothing)"
            )


def _axis_pairs(ctx, lattice):
    """The launch geometry's per-axis (block dim, thread dim) pairs on the
    writable lattice: consecutive dims whose extents match config(blocks,
    threads). The raws are iotas of these dims — a tensor already split to
    the launch lattice is indexed by the raw pair directly (S.3)."""
    blocks, threads = ctx.context["k.geom"]
    dims = lattice.type.dims
    pairs, i = [], 0
    for b, t in zip(blocks or (), threads or ()):
        if i + 1 < len(dims) and dims[i][2] - dims[i][1] == b and dims[i + 1][2] - dims[i + 1][1] == t:
            pairs.append((dims[i][0], dims[i + 1][0]))
            i += 2
        else:
            raise ValueError(
                f"config(blocks={blocks}, threads={threads}): the raw pair indexes a launch-"
                f"lattice-shaped target, and this writable's dims {tuple(d[0] for d in dims)} "
                f"are not split to the geometry — split the target (raw indexing then aligns "
                f"for free), or compute global indices explicitly via "
                f"global_thread_idx(block_idx(...), thread_idx(...), grid_layout())"
            )
    return pairs


def _ambient_iota(ctx, kind, axis, name, lattice, node):
    """One raw coordinate — NO MAGIC (owner-ruled): the raws mean ONE
    thing always. thread_idx is the within-block coordinate; block_idx is
    the block coordinate. Under the DEFAULT geometry one block spans the
    lattice, so thread_idx EQUALS the global coordinate structurally (not
    modally) and block_idx is the zero field. Under explicit geometry the
    raws are iotas of the launch lattice — served by the writable's
    per-axis (block, thread) dim pairs when it is split to the geometry
    (the smart-user case: raw indexing aligns for free, never a
    requirement); a global index is always the EXPLICIT spelling,
    global_thread_idx(block, thread, grid)."""
    if _norm_geom(ctx.context["k.geom"]) is None:  # ONE block spans the lattice (default, or all-ones blocks)
        if kind == "block_idx":
            dims = tuple((d[0], (d[1], d[2])) for d in lattice.type.dims)
            return ctx.emit("tl.const", node=node, value=0.0, dims=dims)
        return ctx.emit("tl.iota", lattice, node=node, name=name)
    b_dim, t_dim = _axis_pairs(ctx, lattice)[axis]
    return ctx.emit("tl.iota", lattice, node=node, name=b_dim if kind == "block_idx" else t_dim)


def _k_call(ctx, node):
    """The kernel layer of call resolution: the ambient (iota-recording)
    and function-valued arguments; everything else — markers, the staging
    door, helper inlining, the S.1 vocabulary — is the shared dialect pack."""
    c = ctx.context
    if isinstance(node.func, ast.Name):
        name = node.func.id
        obj = _lookup(ctx, name)
        if isinstance(obj, _Intrinsic) and obj.name in ("thread_idx", "block_idx"):
            lattice = ctx.root.params[-1]  # the writable target (S.3 convention)
            out = tuple(_ambient_iota(ctx, obj.name, i, cst.value, lattice, node) for i, cst in enumerate(node.args))
            c["k.iotas"].extend(out)
            return out  # ALWAYS a tuple
        if obj is not None and getattr(obj, "fp", None) is not None:
            return _fn_arg(ctx, obj, _lower_args(ctx, node), None, node)
    return _tl_call(ctx, node)


def _resolve_staged(ctx, obj):
    """Walk the staged-recipe chain (240 C1) from ``obj`` down to the ONE
    kernel parameter underneath; return (pname, replay)."""
    c = ctx.context
    pname = next((n for n, v in c["k.fn_params"].items() if v is obj), None)
    if pname is not None:
        return pname, lambda cur: cur
    rec = c.get("tl.recipes", {}).get(id(obj))
    if rec is None:
        return None, None
    fn, args, kwargs = rec
    a_replays, k_replays, found = [], {}, []
    for a in args:
        p, r = _resolve_staged(ctx, a) if getattr(a, "fp", None) is not None else (None, None)
        a_replays.append(r)
        if p is not None:
            found.append(p)
    for key, v in kwargs.items():
        p, r = _resolve_staged(ctx, v) if getattr(v, "fp", None) is not None else (None, None)
        k_replays[key] = r
        if p is not None:
            found.append(p)
    if len(found) != 1:
        raise ValueError(
            f"a staged transform used inside a kernel must reference exactly one kernel "
            f"parameter (found {len(found)}) — stage it outside the body instead"
        )

    def replay(cur, fn=fn, args=args, kwargs=kwargs, ar=tuple(a_replays), kr=dict(k_replays)):
        return fn(
            *[r(cur) if r is not None else a for a, r in zip(args, ar)],
            **{k: (kr[k](cur) if kr[k] is not None else v) for k, v in kwargs.items()},
        )

    return found[0], replay


# the scalar value-dialect ops a spliced fn-argument may contain; anything
# else (bounded control flow above all) stays on the ORACLE path — per-element
# dispatch through the spelled reference, until tl grows predication
_LIFT = {"core.add": "add", "core.sub": "sub", "core.mul": "mul", "core.div": "div", "core.neg": "neg"}
_LIFT["core.select"] = "where"
_STRUCTURAL_OPS = {"core.param", "core.env", "core.const", "core.yield", "core.tuple", "core.extract"}


def _scalar_region(handle, nargs):
    """The fn-argument lowered by the dsl's OWN machinery to a scalar
    region: pipelines compose, staged/derived transforms build, captures
    become env paths — zero kernel-side lowering logic (240 C5.3)."""
    ops = {**CORE_OPS, **ABI_OPS, **DEFAULT.ops}
    rules, ctx = dict(DEFAULT.lower_rules), {"registry": DEFAULT}
    region = lower_handle(handle, rules, ops, arg_types=(f64,) * nargs, derived=DEFAULT.derived, context=ctx)
    return run_stage(region, NORMALIZE_ENV, ops)


def _liftable(region) -> bool:
    """Is this scalar region pointwise-liftable over the thread lattice?
    Straight-line value ops only — a region-carrying op (if/for) is the
    oracle class."""
    for n in walk_region(region):
        if n.regions:
            return False
        if not (n.op in _STRUCTURAL_OPS or n.op in _LIFT or n.op == "core.cmp" or n.op.startswith("pw.")):
            return False
    return True


def _fn_arg(ctx, handle, args, out_spec, node):
    """A function-valued argument applied at the thread coordinates.
    The LIFTABLE class (straight-line value bodies) INLINES: the fn's own
    scalar region splices into the kernel region as pointwise rows, its
    captures riding arg-rooted abi slots re-extracted per launch (240
    C5.3 — the compile-once thesis holds with zero per-element dispatch,
    and the destructure pattern is CHECKED against the fn's actual result
    shape). Bodies with bounded control flow stay on the per-element
    ORACLE path (oracle-grade by doctrine) until tl grows predication.
    Staged transforms of a parameter replay their recipe chain on the
    CURRENT launch binding — both classes."""
    pname, wrap = _resolve_staged(ctx, handle)
    if all(isinstance(a, Node) for a in args) and hasattr(handle, "fntype"):
        region = _scalar_region(handle, len(args))
        if _liftable(region):
            return _splice_fn(ctx, handle, region, args, out_spec, node, pname, wrap)
    return _fn_arg_oracle(ctx, handle, args, out_spec, node, pname, wrap)


def _splice_fn(ctx, handle, region, args, out_spec, node, pname, wrap):
    """Scalar region -> kernel region: params bind the coordinate tensors,
    value ops become pointwise rows when a tensor is in play (all-scalar
    subtrees stay in the value dialect — run_region has their rows), and
    env reads become abi slots at ("arg", pname, *path) with offsets from
    the fn's OWN pack plan, block-based into the kernel staging bytes."""
    c = ctx.context
    key = (pname, handle.fp)
    block = c["k.arg_plans"].get(key)
    if block is None:
        plan = plan_from_types(handle.env_types, (), DEFAULT.table)
        base = c["k.uniform_size"]
        c["k.uniform_size"] = base + plan.staging_size
        block = c["k.arg_plans"][key] = {
            "pname": pname,
            "fixed": None if pname is not None else handle,
            "wrap": wrap,
            "base": base,
            "plan": plan,
            "extract": build_extractor(handle.env_types, (), plan, DEFAULT.table),
            "by_path": {(s.source.index, *s.source.sub): s for s in plan.slots},
            "slots": {},
        }
    memo: dict[int, object] = {}

    def go(nd):
        if id(nd) not in memo:
            memo[id(nd)] = _one(nd)
        return memo[id(nd)]

    def _one(nd):
        attrs = dict(nd.attrs)
        if nd.op == "core.param":
            return args[attrs["index"]]
        if nd.op == "core.const":
            return attrs["value"]
        if nd.op == "core.env":
            path = attrs["slot"]
            cached = block["slots"].get(path)
            if cached is None:
                spec = block["by_path"][path]
                cached = block["slots"][path] = ctx.emit(
                    "abi.slot",
                    node=node,
                    type=nd.type,
                    src=("arg", pname or handle.fntype.template.label, *path),
                    offset=block["base"] + spec.dest.offset,
                    fmt=spec.dest.fmt,
                )
            return cached
        if nd.op == "core.tuple":
            return tuple(go(a) for a in nd.args)
        if nd.op == "core.extract":
            return go(nd.args[0])[attrs["index"]]
        ops_v = [go(a) for a in nd.args]

        def lift(o):
            return o if isinstance(o, Node) else ctx.emit("core.const", node=node, type=f64, value=float(o))

        lifted = [lift(o) for o in ops_v]
        if any(isinstance(o, Node) and isinstance(o.type, TensorType) for o in ops_v):
            f = _LIFT.get(nd.op) or (attrs["pred"] if nd.op == "core.cmp" else nd.op[3:])
            return ctx.emit("tl.pointwise", *lifted, node=node, f=f)
        return ctx.emit(nd.op, *lifted, node=node, **attrs)  # an all-scalar subtree stays value-dialect

    result = go(region.body[-1].args[0])
    if out_spec is None:
        if isinstance(result, tuple):
            raise ValueError(
                f"the function returns {len(result)} values — destructure them (the pattern declares the outputs)"
            )
        return result
    if not isinstance(result, tuple) or len(result) != len(out_spec):
        got = len(result) if isinstance(result, tuple) else 1
        raise ValueError(f"the destructuring pattern names {len(out_spec)} outputs but the function returns {got}")
    flat = []
    for s, part in zip(out_spec, result):
        if s is None:
            if isinstance(part, tuple):
                raise ValueError("the destructuring pattern binds one name where the function returns a tuple")
            flat.append(part)
        else:
            if not isinstance(part, tuple) or len(part) != s:
                got = len(part) if isinstance(part, tuple) else 1
                raise ValueError(f"the destructuring pattern names {s} grouped outputs but the function returns {got}")
            flat.extend(part)
    return tuple(flat)


def _fn_arg_oracle(ctx, handle, args, out_spec, node, pname, wrap):
    """The per-element ORACLE path: pointwise instrs over launch-rebindable
    markers dispatching through the spelled reference. The oracle class is
    exactly what cannot lift yet (bounded control flow, tuple-shaped
    coordinates); everything else inlines (240 C5.3)."""
    c = ctx.context
    flat, spec = [], []
    for a in args:
        if isinstance(a, Node):
            flat.append(a)
            spec.append(None)
        else:
            spec.append(len(a))
            flat.extend(a)
    n_out = 1 if out_spec is None else sum(1 if s is None else s for s in out_spec)
    outs = []
    for k in range(n_out):
        fp_key = (handle.fp, tuple(spec), out_spec, k)
        mname = f"kernel.fn.{hashlib.sha256(repr(fp_key).encode()).hexdigest()[:10]}"

        def _make(mname=mname, spec=tuple(spec), out_spec=out_spec, k=k, wrap=wrap):
            def apply(*coords):
                from pdum.dsl.reference import reference

                f = _ARG_BINDINGS[mname]
                if wrap is not None:  # replay the recipe chain on the CURRENT binding
                    f = wrap(f)

                def call(*cs):
                    it = iter(cs)
                    rebuilt = [float(next(it)) if s is None else tuple(float(next(it)) for _ in range(s)) for s in spec]
                    res = reference(f)(*rebuilt)
                    if out_spec is None:
                        return res
                    flat_res = []
                    for s, part in zip(out_spec, res):
                        flat_res.append(part) if s is None else flat_res.extend(part)
                    return flat_res[k]

                return np.vectorize(call)(*coords)

            return Marker(mname, apply)

        from .registry import MARKERS

        MARKERS.derive(mname, _make)
        if pname is not None:
            c["k.fn_markers"].setdefault(pname, []).append(mname)
        outs.append(ctx.emit("tl.pointwise", *flat, node=node, f=mname))
    return outs[0] if out_spec is None else tuple(outs)


def _check_indices(ctx, sub):
    idx = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
    for i in idx:
        v = ctx.lower(i)
        if not (isinstance(v, Node) and any(v == t for t in ctx.context["k.iotas"])):
            raise ValueError(
                "kernel subscripts take exactly the thread coordinates "
                "(data-dependent indexing is take/scatter_add, arriving P9)"
            )


def _k_assign(ctx, node):
    c = ctx.context
    if len(node.targets) != 1:
        return _value_assign(ctx, node)
    tgt = node.targets[0]
    if isinstance(tgt, ast.Subscript):  # the store, at exactly the thread coordinates
        target = ctx.lower(tgt.value)
        if not (isinstance(target, Node) and isinstance(target.type, TensorType)) or id(target) not in c["k.pname_of"]:
            raise ValueError("stores write into a tensor argument")
        _check_indices(ctx, tgt)
        value = _k_call(ctx, node.value) if isinstance(node.value, ast.Call) else ctx.lower(node.value)
        if not hasattr(value, "type"):  # a scalar store broadcasts over the target lattice
            dims = tuple((d.name, (d.start, d.stop)) for d in target.type.layout.dims)
            value = ctx.emit("tl.const", node=node, value=float(value), dims=dims)
        tok = c.get("tl.token") or ctx.emit("tl.token", node=node)
        c["tl.token"] = ctx.emit("tl.store", tok, target, value, node=node)
        c["k.stored"].append(c["k.pname_of"][id(target)])
        return None
    if isinstance(tgt, ast.Tuple) and all(isinstance(e, (ast.Name, ast.Tuple)) for e in tgt.elts):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            fname = node.value.func.id
            obj = _lookup(ctx, fname)
            if obj is not None and getattr(obj, "fp", None) is not None:
                # the destructuring PATTERN declares the output structure
                out_spec, groups = [], []
                for e in tgt.elts:
                    if isinstance(e, ast.Name):
                        out_spec.append(None)
                        groups.append((e.id,))
                    elif all(isinstance(x, ast.Name) for x in e.elts):
                        out_spec.append(len(e.elts))
                        groups.append(tuple(x.id for x in e.elts))
                    else:
                        raise ValueError("destructure a function result into names or name-tuples")
                outs = iter(_fn_arg(ctx, obj, _lower_args(ctx, node.value), tuple(out_spec), node.value))
                for group in groups:
                    for n in group:
                        v = next(outs)
                        ctx.locals[n] = v
                        _claim(ctx, n, v)
                return None
        value = _k_call(ctx, node.value) if isinstance(node.value, ast.Call) else ctx.lower(node.value)
        if isinstance(value, tuple) and all(isinstance(e, ast.Name) for e in tgt.elts):
            for e, v in zip(tgt.elts, value):
                ctx.locals[e.id] = v
                _claim(ctx, e.id, v)
            return None
        return _value_assign(ctx, node)
    if isinstance(tgt, ast.Name):
        value = _k_call(ctx, node.value) if isinstance(node.value, ast.Call) else ctx.lower(node.value)
        ctx.locals[tgt.id] = value
        _claim(ctx, tgt.id, value)
        return None
    return _value_assign(ctx, node)


def _k_subscript(ctx, node):
    """A LOAD at the thread coordinates: the view itself."""
    if isinstance(node.value, ast.Name):
        base = ctx.locals.get(node.value.id)
        if isinstance(base, Node) and isinstance(base.type, TensorType):
            _check_indices(ctx, node)
            return base
    return _value_subscript(ctx, node)


def _k_name(ctx, node):
    """Names resolve locals-first. The LITERAL DOCTRINE (240 C4.2b,
    owner-ruled): an UNMARKED host scalar is DATA — it becomes a per-launch
    uniform slot, warm on change, its value never entering identity; only a
    ``literal(...)``-wrapped value bakes as a compile-time constant (and
    then recompiling on change is chosen). Host CITIZENS (helpers, markers,
    staged results) are consumed at call sites, never as expression
    values."""
    if node.id in ctx.locals:
        return ctx.locals[node.id]  # Nodes and host bindings alike
    v = ctx.handle.env.get(node.id, None)
    if v is None:
        v = _globals_of(ctx).get(node.id)
    if isinstance(v, LiteralValue):  # declared structural: bake, value-keyed
        return ctx.emit("core.const", node=node, type=typeof(v.value), value=v.value)
    if isinstance(v, (int, float, bool)):  # unmarked: DATA — an abi slot (ONE marshaling dialect, 240 C5)
        c = ctx.context
        cached = c["k.uniforms"].get(node.id)
        if cached is None:
            t = typeof(v)
            fmt = _FMTS[t.kind]
            off = c["k.uniform_size"]
            cached = ctx.emit("abi.slot", node=node, type=t, src=("env", node.id), offset=off, fmt=fmt)
            c["k.uniform_size"] = off + struct.calcsize(fmt)
            c["k.uniforms"][node.id] = cached
        return cached
    if v is not None:
        raise ValueError(f"{node.id!r} is a host citizen here — kernels use it at a call site, not as a value")
    return _value_name(ctx, node)  # the proper refusal voice


KERNEL_RULES = {
    **TL_RULES,
    ast.Assign: _k_assign,
    ast.Call: _k_call,
    ast.Subscript: _k_subscript,
    ast.Name: _k_name,
}


# ---- compilation ------------------------------------------------------------


@dataclass(frozen=True)
class _Artifact:
    region: Region
    executor: object  # tier-2 artifact: (values, staging) -> effect; shared by content key
    fn: object  # the kernel fn — abi slots re-read its environment per launch
    uniforms: tuple  # (name, offset, fmt) abi slots (DATA: extracted + packed fresh every launch)
    arg_slots: tuple  # spliced fn-arg capture blocks: (pname, fixed, wrap, base, plan, extract)
    staging_size: int  # total staging bytes (kernel env slots + fn-arg blocks)
    params: tuple  # kernel parameter names, in order
    tensor_params: tuple  # tensor parameter names, region-param order
    writable: tuple  # parameter names that are stored to (incl. tap buffers)
    fn_markers: dict  # parameter name -> marker names (launch rebind slots)
    tap_sites: dict  # site name -> lattice dim names (valid sites)
    invalid_taps: dict  # site name -> reason (the naming law met inlining)
    requested_taps: tuple = ()  # the name set this artifact was built for
    geometry: tuple | None = None  # the (blocks, threads) this entry binds; launches validate against it

    def launch(self, args, taps=None):
        bound = dict(zip(self.params, args))
        for name in self.requested_taps:  # tap buffers are writable inputs
            bound[f"tap:{name}"] = (taps or {})[name]
        for w in self.writable:  # the day-one overlap refusal (210)
            for name, a in bound.items():
                if name in self.writable or not isinstance(a, Tensor):
                    continue
                if bound[w].overlaps(a):
                    raise ValueError(
                        f"writable argument {w!r} overlaps readable argument {name!r} — "
                        f"in-place returns exist only as an L2-certified rewrite; "
                        f"ping-pong between two buffers instead"
                    )
        for name, mnames in self.fn_markers.items():  # values ride the rebind channel
            for mname in mnames:
                _ARG_BINDINGS[mname] = bound[name]
        values = [bound[n] for n in self.tensor_params] + [bound[f"tap:{n}"] for n in self.requested_taps]
        staging = b""
        if self.staging_size:  # extract -> pack -> launch (the dsl's hit-path shape)
            staging = bytearray(self.staging_size)
            env = _captured(self.fn) if self.uniforms else {}
            for name, off, fmt in self.uniforms:
                struct.pack_into(fmt, staging, off, env[name])
            for pname, fixed, wrap, base, plan, extract in self.arg_slots:
                f = bound[pname] if pname is not None else fixed
                if wrap is not None:  # replay the recipe chain on the CURRENT binding
                    f = wrap(f)
                pack_into(plan, memoryview(staging)[base : base + plan.staging_size], extract(f.captures, ()))
            staging = bytes(staging)
        self.executor(values, staging)
        return None  # stores are the effect; kernels return nothing (taps included)


def _executor(region: Region):
    """The numpy interpreter's 'compile': close over the region. This is
    the backend column's seam — a device executor replaces this body with
    render + compile, behind the same content key."""
    return lambda values, staging: run_region(region, values, uniforms=staging)


def _compile(fn, args, tap_names=(), geom=None) -> _Artifact:
    handle = capture_shim(fn)
    check_coherence(handle)
    ctx = Lowerer(
        handle, KERNEL_RULES, {**CORE_OPS, **TL_OPS, **ABI_OPS}, {}, context={"registry": DEFAULT, "tl.kind": "compute"}
    )
    c = ctx.context
    c.update(
        {
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
            "k.geom": geom,
        }
    )
    names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    if len(names) != len(args):
        raise TypeError(f"{fn.__qualname__} takes {len(names)} arguments, got {len(args)}")
    p_nodes, tensor_names = [], []
    for name, a in zip(names, args):
        if isinstance(a, Tensor):
            p = ctx.builder.param(len(p_nodes), tensor_type(a))
            p_nodes.append(p)
            tensor_names.append(name)
            ctx.locals[name] = p
            c["k.pname_of"][id(p)] = name
        else:
            ctx.locals[name] = a  # a kernel value: identity in the key, values rebind
            c["k.fn_params"][name] = a
    if not p_nodes:
        raise TypeError("@compute needs at least one tensor argument (the thread lattice)")
    ctx.params = tuple(p_nodes)
    tree = next(n for n in ast.parse(handle.snapshot.text).body if isinstance(n, ast.FunctionDef))
    with events.span("kernel.lower", fn.__qualname__):
        for stmt in tree.body:
            if isinstance(stmt, ast.Return):
                raise ValueError("kernels return nothing — stores into writable arguments are the effect")
            ctx.lower(stmt)
    tok = c.get("tl.token")
    if tok is None:
        raise ValueError(f"{fn.__qualname__} stores nothing — a kernel's effect is its stores")
    tap_params = []
    for tname in tap_names:  # requested taps become token-threaded stores
        if tname in c["k.invalid"]:
            raise ValueError(f"tap {tname!r} is INVALID: {c['k.invalid'][tname]}")
        if tname not in c["k.claims"]:
            have = sorted(c["k.claims"]) or ["<none>"]
            raise ValueError(f"no tap site {tname!r} — sites: {', '.join(have)}")
        claimed = c["k.claims"][tname]
        p = ctx.builder.param(len(p_nodes) + len(tap_params), claimed.type)
        tap_params.append(p)
        tok = ctx.builder.emit("tl.store", tok, p, claimed)
        c["k.stored"].append(f"tap:{tname}")
    region = Region(
        params=tuple(p_nodes) + tuple(tap_params),
        body=(ctx.builder.emit("core.yield", tok),),
    )
    tap_sites = {n: tuple(d[0] for d in v.type.dims) for n, v in c["k.claims"].items()}
    return _Artifact(
        region=region,
        executor=ARTIFACTS.get_or_compile((region.key, _EXECUTOR_FP), lambda: _executor(region)),
        fn=fn,
        uniforms=tuple((n, dict(nd.attrs)["offset"], dict(nd.attrs)["fmt"]) for n, nd in c["k.uniforms"].items()),
        arg_slots=tuple(
            (b["pname"], b["fixed"], b["wrap"], b["base"], b["plan"], b["extract"]) for b in c["k.arg_plans"].values()
        ),
        staging_size=c["k.uniform_size"],
        params=tuple(names),
        tensor_params=tuple(tensor_names),
        writable=tuple(dict.fromkeys(c["k.stored"])),
        fn_markers=dict(c["k.fn_markers"]),
        tap_sites=tap_sites,
        invalid_taps=dict(c["k.invalid"]),
        requested_taps=tuple(tap_names),
        geometry=geom,
    )


__all__ = ["ComputeKernel", "Config", "block_idx", "compute", "config", "shared", "thread_idx"]
