"""@compute kernels on the reference evaluator (200 §S.3, P7).

A kernel body IS the value language plus the dialect extensions (the
one-body-language law, 200 §S.3 amendment): the thread AMBIENT
(``thread_idx("y", "x")`` — coordinates are never positional
parameters), explicit token-threaded STORES into writable arguments
(``img[y, x] = v``), and — arriving P8/P9 — buffer READS at computed
indices. Function-valued arguments apply at the thread coordinates,
including tuple-returning ones (``v, (dy, dx) = f(y, x)`` — the
destructuring pattern declares the structure). Writability is inferred
from the body: an argument is writable iff it is stored to. Ordering is
TOKEN THREADING: one implicit token threads through all stores in
statement order (the frontend policy); tokens never appear in user
syntax.

Claiming is TAGLESS (S.4 amendment): every uniquely-named binding is a
claimable site — ``config(taps={"dist": t})`` binds the binding named
``dist``; a name bound more than once is invalidated with the reason,
never auto-suffixed.

The reference lowering is the IOTA UNIFICATION: thread coordinates are
coordinate iotas over the writable target's lattice, the body lowers to
pointwise/store instructions over them, and the whole kernel is one tl
Program run by ``ir.run`` — the same kernel is expressible by hand as
pointwise-over-iotas, and the two are differential-tested.

**Function-valued arguments** carry their FnType identity (``handle.fp``:
code + env TYPES, never values) in the kernel key: swapping a stage is a
new artifact; new captured values on the same shape are a WARM HIT, with
the values riding a per-launch rebind (the uniform channel at reference
tier). Per-element host dispatch through the spelled oracle is the
sanctioned oracle-grade execution. Guard policy for argument handles,
recorded: identity rides the FnType fp in the key; values rebind at every
launch; cell guards are not needed at the reference tier (the device tier
revisits when argument handles bake into artifacts).

**Invocation is the bracket** — ``kernel[config(blocks, threads,
taps={...})](args)``; geometry is invocation-only and never enters any
key (threads-per-block becomes the value-specialized carve-out when
device backends exist); the tap NAME SET specializes, tap tensors are
invocation data.

Day-one contract (210): a writable argument overlapping any READABLE
argument refuses at dispatch with the ping-pong message; in-place returns
only ever as an L2-certified rewrite.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

import numpy as np
from pdum.dsl import events
from pdum.dsl.cache import Memo

from .ir import Program, run
from .lifting import _T, _Intrinsic, _Lifter
from .markers import Marker
from .producer import _captured, _fn_ast
from .registry import MARKERS
from .tensor import Tensor

KERNELS = Memo("kernel", capacity=1 << 30)
_ARG_BINDINGS: dict[str, object] = {}  # marker name -> the CURRENT handle (per launch)

thread_idx = _Intrinsic("thread_idx")


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
    blocks/threads are invocation data (threads-per-block is the recorded
    value-specialized carve-out when device backends declare it); the tap
    NAME SET specializes (a different tap set is a different artifact,
    the P5 identity law) while the tap TENSORS are invocation data;
    shared_mem is structural (specializes) and arrives with the tile
    tier."""

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
        key = (_code_fp(self.fn), tuple(_arg_fp(a) for a in args), tap_names)
        art = KERNELS.get_or_compile(key, lambda: _compile(self.fn, args, tap_names))
        return art.launch(args, dict(cfg.taps))

    def taps(self, *args) -> dict:
        """Introspection: the kernel's tap SITES for these argument shapes —
        {name: {"valid": bool, "dims": (...) | None, "reason": str | None}}.
        Compiles (cached) with an empty tap set to discover the sites."""
        key = (_code_fp(self.fn), tuple(_arg_fp(a) for a in args), ())
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


def _arg_fp(a) -> tuple:
    fp = getattr(a, "fp", None)
    if fp is not None:  # a Handle/Pipeline: FnType identity — types, never values
        return ("fn", fp)
    if isinstance(a, Tensor):
        return ("tensor", tuple((d.name, d.start, d.stop) for d in a.layout.dims), str(a.dtype))
    raise TypeError(f"@compute arguments are tensors or kernel values, got {a!r}")


# ---- compilation -----------------------------------------------------------


@dataclass(frozen=True)
class _Artifact:
    program: Program
    params: tuple  # kernel parameter names, in order
    writable: tuple  # parameter names that are stored to
    fn_markers: dict  # parameter name -> marker names (launch rebind slots; one per flat output)
    tap_sites: dict  # site name -> lattice dim names (valid sites)
    invalid_taps: dict  # site name -> reason (the naming law met inlining)
    requested_taps: tuple = ()  # the name set this artifact was built for

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
        inputs = {name: a for name, a in bound.items() if isinstance(a, Tensor)}
        run(self.program, inputs)
        return None  # stores are the effect; kernels return nothing (taps included)


class _KernelLowerer(_Lifter):
    """thread_idx, subscript loads/stores, and function-argument calls on
    top of the shared lowering machinery."""

    def __init__(self, env: dict):
        super().__init__(env)
        self.threads: dict[str, _T] = {}  # thread name -> iota _T
        self.target: _T | None = None  # the lattice source (first writable use)
        self.token: str | None = None
        self.stored: list[str] = []  # parameter names stored to, in order
        self.fn_markers: dict[str, list] = {}  # param -> marker names (one per flat output)
        self.param_names: tuple = ()  # kernel parameters — fn-arg slots live here ONLY
        self.tap_vars: dict[str, str] = {}  # site name -> SSA var
        self.invalid_taps: dict[str, str] = {}  # site name -> reason
        self.claimed: set[str] = set()  # every binding name ever seen (uniqueness law)

    def child(self, env: dict) -> "_KernelLowerer":
        """Helpers inlined into a kernel share its claiming/thread context
        (the site dicts are shared by reference, so a helper's bindings
        register and collide honestly). Stores inside helpers are not yet
        supported."""
        inner = super().child(env)
        inner.threads, inner.target = self.threads, self.target
        inner.tap_vars, inner.invalid_taps = self.tap_vars, self.invalid_taps
        inner.claimed = self.claimed
        inner.fn_markers, inner.param_names = self.fn_markers, self.param_names
        return inner

    def claim(self, name: str, value) -> None:
        """The naming law IS the claiming mechanism (S.4 amendment, tagless):
        every uniquely-named binding is a site — free unless requested. A
        name bound more than once (rebinding, or a helper inlined twice) is
        INVALIDATED with the reason; the law never auto-suffixes."""
        if name in self.claimed:
            self.tap_vars.pop(name, None)
            self.invalid_taps[name] = "bound at more than one site (rebinding or inlining made it non-unique)"
            return
        self.claimed.add(name)
        if isinstance(value, _T):
            self.tap_vars[name] = value.var
        else:
            self.invalid_taps[name] = "not a lattice value (nothing to store)"

    def _i_thread_idx(self, *names):
        if self.target is None:
            raise ValueError("thread_idx needs a writable argument to define the lattice")
        out = []
        for n in names:
            if n not in {d.name for d in self.target.shadow.dims}:
                raise ValueError(f"thread_idx({n!r}): the writable argument has no dim {n!r}")
            t = self.emit("iota", (self.target.var,), f"tid_{n}", name=n)
            self.threads[n] = t
            out.append(t)
        return tuple(out)

    def statement(self, stmt) -> None:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Tuple)
            and isinstance(stmt.value, ast.Call)
            and self._fn_arg_destructure(stmt)
        ):
            return
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Subscript):
            target = self.value(stmt.targets[0].value)
            if not isinstance(target, _T):
                raise ValueError("stores write into a tensor argument")
            self._check_indices(stmt.targets[0], target)
            value = self.value(stmt.value)
            if not isinstance(value, _T):
                value = self.const_like(float(value), target)
            tok = self.token or self.b.emit("token", (), hint="tok")
            self.token = self.b.emit("store", (tok, target.var, value.var), hint="st")
            self.shadows[self.token] = self.shadows.get(self.token)
            self.stored.append(target.var)
            return
        super().statement(stmt)

    def _check_indices(self, sub: ast.Subscript, target: _T) -> None:
        idx = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
        vals = [self.value(i) for i in idx]
        tids = {t.var for t in self.threads.values()}
        for v in vals:
            if not (isinstance(v, _T) and v.var in tids):
                raise ValueError(
                    "kernel subscripts take exactly the thread coordinates "
                    "(data-dependent indexing is take/scatter_add, arriving P9)"
                )

    def value(self, node):
        if isinstance(node, ast.Subscript):
            base = (
                super().value(node.value)
                if not isinstance(node.value, ast.Name)
                else self.adopt(self.env.get(node.value.id))
            )
            if isinstance(base, _T):  # a LOAD at the thread coordinates: the view itself
                self._check_indices(node, base)
                return base
        return super().value(node)

    def compute_call(self, target, args, kwargs):
        from .markers import Marker
        from .mdsl import CompositeMarker

        if isinstance(target, (Marker, CompositeMarker)):
            # a kernel body is SCALAR-TIER code — per-thread values, like a
            # marker body — so bare names apply directly (S.2's one
            # definition, two consumers: the same gelu serves both)
            return self.pointwise(target.name, *args, hint=target.name.rsplit(".", 1)[-1])
        made = super().compute_call(target, args, kwargs)
        if made is not None:
            return made
        fp = getattr(target, "fp", None)
        if fp is not None and all(
            isinstance(a, _T) or (isinstance(a, tuple) and all(isinstance(x, _T) for x in a)) for a in args
        ):
            return self._fn_arg_call(target, args)
        return None

    def _fn_arg_destructure(self, stmt) -> bool:
        """``v, (dy, dx) = f(y, x)`` — a tuple-returning function argument.
        The DESTRUCTURING PATTERN declares the output structure (the same
        spec discipline as tuple arguments): one pointwise per flat output,
        each a component of the same per-element oracle call. Every bound
        name claims (the naming law)."""
        call = stmt.value
        if not isinstance(call.func, ast.Name) or call.keywords:
            return False
        handle = self.env.get(call.func.id)
        if getattr(handle, "fp", None) is None:
            return False
        args = tuple(self.value(a) for a in call.args)
        if not all(isinstance(a, _T) or (isinstance(a, tuple) and all(isinstance(x, _T) for x in a)) for a in args):
            return False
        out_spec, groups = [], []
        for e in stmt.targets[0].elts:
            if isinstance(e, ast.Name):
                out_spec.append(None)
                groups.append((e.id,))
            elif isinstance(e, ast.Tuple) and all(isinstance(x, ast.Name) for x in e.elts):
                out_spec.append(len(e.elts))
                groups.append(tuple(x.id for x in e.elts))
            else:
                raise ValueError("destructure a function result into names or name-tuples")
        outs = iter(self._fn_arg_call(handle, args, out_spec=tuple(out_spec)))
        for group in groups:
            for n in group:
                v = self.rebind(next(outs), n)
                self.env[n] = v
                self.claim(n, v)
        return True

    def _fn_arg_call(self, handle, args, out_spec=None):
        """A function-valued argument applied at the thread coordinates:
        pointwise instrs over launch-rebindable markers — per-element
        dispatch through the spelled oracle (oracle-grade by doctrine).
        A tuple argument (``f((y, x))``) flattens into the operands and
        regroups per element; a tuple RESULT flattens per ``out_spec``
        (from the destructuring pattern), one instr per flat component."""
        pname = next((n for n in self.param_names if self.env.get(n) is handle), None)
        flat, spec = [], []
        for a in args:
            if isinstance(a, _T):
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

            def _make(mname=mname, spec=tuple(spec), out_spec=out_spec, k=k):
                def apply(*coords):
                    from pdum.dsl.reference import reference

                    f = _ARG_BINDINGS[mname]

                    def call(*cs):
                        it = iter(cs)
                        rebuilt = [
                            float(next(it)) if s is None else tuple(float(next(it)) for _ in range(s)) for s in spec
                        ]
                        res = reference(f)(*rebuilt)
                        if out_spec is None:
                            return res
                        flat_res = []
                        for s, part in zip(out_spec, res):
                            flat_res.append(part) if s is None else flat_res.extend(part)
                        return flat_res[k]

                    return np.vectorize(call)(*coords)

                return Marker(mname, apply)

            MARKERS.derive(mname, _make)
            if pname is not None:
                self.fn_markers.setdefault(pname, []).append(mname)
            outs.append(self.pointwise(mname, *flat, hint="fx"))
        return outs[0] if out_spec is None else tuple(outs)


def _compile(fn, args, tap_names=()) -> _Artifact:
    tree = _fn_ast(fn)
    params = [a.arg for a in tree.args.args]
    if len(params) != len(args):
        raise TypeError(f"{fn.__qualname__} takes {len(params)} arguments, got {len(args)}")
    lo = _KernelLowerer(_captured(fn))
    lo.param_names = tuple(params)
    writable_target = next((a for a in args if isinstance(a, Tensor)), None)
    if writable_target is None:
        raise TypeError("@compute needs at least one tensor argument (the thread lattice)")
    for name, a in zip(params, args):
        if isinstance(a, Tensor):
            lo.b.input(name)
            lo.shadows[name] = a.layout
            lo.env[name] = _T(name, a.layout)
        else:
            lo.env[name] = a  # a kernel value: identity in the key, values rebind
    # the thread lattice: the writable argument's layout — discovered from the
    # body's stores, but thread_idx may precede the store syntactically, so we
    # seed with the LAST tensor argument (the S.3 convention: outputs last)
    lo.target = next((lo.env[n] for n, a in reversed(list(zip(params, args))) if isinstance(a, Tensor)), None)
    with events.span("kernel.lower", fn.__qualname__):
        for stmt in tree.body:
            if isinstance(stmt, ast.Return):
                raise ValueError("kernels return nothing — stores into writable arguments are the effect")
            lo.statement(stmt)
    if not lo.stored:
        raise ValueError(f"{fn.__qualname__} stores nothing — a kernel's effect is its stores")
    for name in tap_names:  # requested taps become token-threaded stores
        if name in lo.invalid_taps:
            raise ValueError(f"tap {name!r} is INVALID: {lo.invalid_taps[name]}")
        if name not in lo.tap_vars:
            have = sorted(lo.tap_vars) or ["<none>"]
            raise ValueError(f"no tap site {name!r} — sites: {', '.join(have)}")
        buf = lo.b.input(f"tap:{name}")
        lo.shadows[buf] = lo.shadows[lo.tap_vars[name]]
        tok = lo.token or lo.b.emit("token", (), hint="tok")
        lo.token = lo.b.emit("store", (tok, buf, lo.tap_vars[name]), hint="st")
        lo.stored.append(buf)
    writable = tuple(dict.fromkeys(lo.stored))
    tap_sites = {n: tuple(d.name for d in lo.shadows[v].dims) for n, v in lo.tap_vars.items()}
    return _Artifact(
        program=lo.b.program(),
        params=tuple(params),
        writable=writable,
        fn_markers=dict(lo.fn_markers),
        tap_sites=tap_sites,
        invalid_taps=dict(lo.invalid_taps),
        requested_taps=tuple(tap_names),
    )


__all__ = ["ComputeKernel", "Config", "compute", "config", "shared", "thread_idx"]
