"""@compute kernels on the reference evaluator (200 §S.3, P7).

A kernel body speaks the shared expression syntax with two additions:
``thread_idx("y", "x")`` — the AMBIENT intrinsic (thread coordinates are
never positional parameters) — and explicit STORES into writable
arguments (``img[y, x] = v``). Writability is inferred from the body: an
argument is writable iff it is stored to. Ordering is TOKEN THREADING:
one implicit token threads through all stores in statement order (the
frontend policy); tokens never appear in user syntax.

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

**Launch config is invocation-only** — it rides the launcher and never
enters any key; threads-per-block becomes a value-specialized bracket
when device backends exist.

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

from .compute import _tensor_like
from .ir import Program, run
from .lifting import _T, _Intrinsic, _Lifter
from .markers import Marker
from .producer import _captured, _fn_ast
from .registry import MARKERS
from .tensor import Tensor

KERNELS = Memo("kernel", capacity=1 << 30)
_ARG_BINDINGS: dict[str, object] = {}  # marker name -> the CURRENT handle (per launch)

thread_idx = _Intrinsic("thread_idx")


def grid(blocks=None, threads=None) -> dict:
    """Launch config: pure invocation data — never part of any key."""
    return {"blocks": blocks, "threads": threads}


@dataclass(frozen=True)
class ComputeKernel:
    fn: object

    def __call__(self, *args, launch: dict | None = None):
        key = (_code_fp(self.fn), tuple(_arg_fp(a) for a in args))
        art = KERNELS.get_or_compile(key, lambda: _compile(self.fn, args))
        return art.launch(args, launch)


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
    fn_markers: dict  # parameter name -> marker name (launch rebind slots)

    def launch(self, args, launch_cfg=None):
        bound = dict(zip(self.params, args))
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
        for name, mname in self.fn_markers.items():  # values ride the rebind channel
            _ARG_BINDINGS[mname] = bound[name]
        inputs = {name: a for name, a in bound.items() if isinstance(a, Tensor)}
        run(self.program, inputs)
        return None  # stores are the effect; kernels return nothing


class _KernelLowerer(_Lifter):
    """thread_idx, subscript loads/stores, and function-argument calls on
    top of the shared lowering machinery."""

    def __init__(self, env: dict):
        super().__init__(env)
        self.threads: dict[str, _T] = {}  # thread name -> iota _T
        self.target: _T | None = None  # the lattice source (first writable use)
        self.token: str | None = None
        self.stored: list[str] = []  # parameter names stored to, in order
        self.fn_markers: dict[str, str] = {}
        self.param_names: tuple = ()  # kernel parameters — fn-arg slots live here ONLY

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
            and isinstance(stmt.targets[0], ast.Subscript)
        ):
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
            base = super().value(node.value) if not isinstance(node.value, ast.Name) else self.adopt(
                self.env.get(node.value.id)
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

    def _fn_arg_call(self, handle, args):
        """A function-valued argument applied at the thread coordinates:
        ONE pointwise instr over a launch-rebindable marker — per-element
        dispatch through the spelled oracle (oracle-grade by doctrine).
        A tuple argument (``f((y, x))``) flattens into the operands and
        regroups per element — the pipe threads one value, so coordinate
        PAIRS ride as tuples through pipelines."""
        pname = next((n for n in self.param_names if self.env.get(n) is handle), None)
        flat, spec = [], []
        for a in args:
            if isinstance(a, _T):
                flat.append(a)
                spec.append(None)
            else:
                spec.append(len(a))
                flat.extend(a)
        mname = f"kernel.fn.{hashlib.sha256(repr((handle.fp, tuple(spec))).encode()).hexdigest()[:10]}"

        def _make(mname=mname, spec=tuple(spec)):
            def apply(*coords):
                from pdum.dsl.reference import reference

                f = _ARG_BINDINGS[mname]

                def call(*cs):
                    it = iter(cs)
                    rebuilt = [
                        float(next(it)) if k is None else tuple(float(next(it)) for _ in range(k))
                        for k in spec
                    ]
                    return reference(f)(*rebuilt)

                return np.vectorize(call)(*coords)

            return Marker(mname, apply)

        MARKERS.derive(mname, _make)
        if pname is not None:
            self.fn_markers[pname] = mname
        return self.pointwise(mname, *flat, hint="fx")


def _compile(fn, args) -> _Artifact:
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
    lo.target = next(
        (lo.env[n] for n, a in reversed(list(zip(params, args))) if isinstance(a, Tensor)), None
    )
    with events.span("kernel.lower", fn.__qualname__):
        for stmt in tree.body:
            if isinstance(stmt, ast.Return):
                raise ValueError("kernels return nothing — stores into writable arguments are the effect")
            lo.statement(stmt)
    if not lo.stored:
        raise ValueError(f"{fn.__qualname__} stores nothing — a kernel's effect is its stores")
    writable = tuple(dict.fromkeys(lo.stored))
    return _Artifact(
        program=lo.b.program(),
        params=tuple(params),
        writable=writable,
        fn_markers=dict(lo.fn_markers),
    )


__all__ = ["ComputeKernel", "compute", "grid", "thread_idx"]

_ = _tensor_like  # noqa: F841 — keep the import surface stable for kernels
