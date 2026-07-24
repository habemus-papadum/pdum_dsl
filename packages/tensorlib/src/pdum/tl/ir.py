"""The linear SSA IR and its reference interpreter (Level 0).

A program is a straight-line sequence of instructions — no branching, no
phi. Each instruction is (var, op, operands, params): the variable it
defines, the operation name, the earlier SSA variables it consumes, and
exact parameter data (ranges, factors, marker names — never floats where
the layout layer would refuse them). Fan-out (one var consumed many times)
is allowed and is what makes reverse-mode AD interesting.

Ops:
- leaves: `input` (value supplied at run time), `const` (a literal scalar
  broadcast over declared dims — layout-native constants), `iota` (operand
  referenced for its LAYOUT only; no value dependency, no gradient).
- layout ops: slice, select, shift, rename, repeat, flip, split, merge,
  diagonal, window, decimate, pad, stencil, strip_charts, simplify — thin
  adapters over the Tensor methods.
- compute: pointwise / reduce / scan, with markers referenced by NAME
  through the pw/red registries (programs stay data, not closures).
- `materialize` (params: order) — the one op that copies: identity
  computation exported in a chosen dim order, giving controlled dense
  strides. Needed where a later `merge` requires real stride nesting
  (e.g. the zero-stuffing adjoint of decimate); also the honest IR home of
  "export order is a property of materialization" (D5).

`run` evaluates over the reference compute layer. `infer` propagates
LAYOUTS only (no data): layout ops run for real, compute ops mirror the
reference layer's dense-wrap rule. Shadow strides use a uniform 8-byte
itemsize — only relative nesting matters (merge checks), and it is
consistent within any fabricated tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np
from pdum.dsl import events

from .compute import Marker, Reducer, iota, pointwise, pw, red, reduce, scan
from .guarded import GuardedLayout, pad_layout, stencil_layout
from .layout import Dim, Layout
from .registry import MARKERS, REDUCERS
from .tensor import Tensor, alignment

PW = {m.name: m for m in vars(pw).values() if isinstance(m, Marker)}
RED = {m.name: m for m in vars(red).values() if isinstance(m, Reducer)}


def pw_marker(name: str):
    # resolve a pointwise marker name: primitives, then registered composites
    if name in PW:
        return PW[name]
    if name in MARKERS:
        return MARKERS[name]
    raise KeyError(f"unknown pointwise marker {name!r}")


def reducer(name: str):
    if name in RED:
        return RED[name]
    if name in REDUCERS:
        return REDUCERS[name]
    raise KeyError(f"unknown reducer {name!r}")


_LEAF_OPS = ("input", "const", "iota", "random")
_COMPUTE_OPS = ("pointwise", "reduce", "scan", "materialize", "with_value_units", "fold")


@dataclass(frozen=True, eq=False)
class Instr:
    var: str
    op: str
    operands: tuple[str, ...] = ()
    params: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        # snapshot: callers cannot mutate an Instr after Program validation
        # (nested values, e.g. a ranges dict's tuples, are already immutable
        # by convention; the top-level mapping is the aliasing hazard)
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    def __repr__(self) -> str:
        args = ", ".join(self.operands)
        ps = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        inner = args + ("; " + ps if ps else "")
        return f"{self.var} = {self.op}({inner})"


@dataclass(frozen=True, eq=False)
class Program:
    instrs: tuple[Instr, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for ins in self.instrs:
            if ins.op not in _LEAF_OPS + _COMPUTE_OPS and ins.op not in _LAYOUT_OPS:
                raise ValueError(f"unknown op {ins.op!r} in {ins!r}")
            if ins.var in seen:
                raise ValueError(f"SSA violation: {ins.var!r} assigned twice")
            for o in ins.operands:
                if o not in seen:
                    raise ValueError(f"{ins!r}: operand {o!r} not yet defined")
            seen.add(ins.var)
        # the compile-ish seam (200 §1.10): building a Program announces
        # itself, so forbid("program.build") pins "this loop builds nothing"
        events.emit("program.build", len(self.instrs))

    @property
    def vars(self) -> tuple[str, ...]:
        return tuple(i.var for i in self.instrs)

    def instr(self, var: str) -> Instr:
        for i in self.instrs:
            if i.var == var:
                return i
        raise KeyError(f"no instruction defines {var!r}")

    def __repr__(self) -> str:
        return "\n".join(repr(i) for i in self.instrs)


# ----------------------------------------------------------------------
# op adapters
# ----------------------------------------------------------------------

_LAYOUT_OPS = {
    "slice": lambda t, p: t.slice(**p["ranges"]),
    "select": lambda t, p: t.select(**p["coords"]),
    "shift": lambda t, p: t.shift(**p["deltas"]),
    "rename": lambda t, p: t.rename(**p["mapping"]),
    "repeat": lambda t, p: t.repeat(p["name"], p["extent"], p.get("chart"), p.get("labels")),
    "flip": lambda t, p: t.flip(p["name"]),
    "split": lambda t, p: t.split(p["name"], **p["parts"]),
    "merge": lambda t, p: t.merge(tuple(p["parts"]), p["name"], p.get("start", 0)),
    "diagonal": lambda t, p: t.diagonal(tuple(p["parts"]), p["name"], p.get("chart")),
    "window": lambda t, p: t.window(p["name"], p["k_name"], p["k"], p.get("dilation", 1)),
    "decimate": lambda t, p: t.decimate(p["name"], p["factor"], p.get("phase", 0)),
    "pad": lambda t, p: t.pad(p["fill"], **p["extents"]),
    "stencil": lambda t, p: t.stencil(p["name"], p["k"], p.get("k_name"), p.get("fill", 0), p.get("dilation", 1)),
    "strip_charts": lambda t, p: t.strip_charts(),
    "with_charts": lambda t, p: t.with_charts(**p["charts"]),
    "with_labels": lambda t, p: t.with_labels(**p["labels"]),
    "bind": lambda t, p: t.bind(**p["levels"]),
    "simplify": lambda t, p: t.simplify(),
}


def _const(p) -> Tensor:
    value = np.asarray(p["value"], dtype=p.get("dtype", np.float64))
    if value.ndim != 0:
        raise ValueError("const takes a scalar value (broadcast over dims)")
    t = Tensor.from_numpy(value, ())
    for name, extent in p.get("dims", ()):
        t = t.repeat(name, extent)
    return t


# ----------------------------------------------------------------------
# fold — the tensor-state scan: programs as first-class step functions
# ----------------------------------------------------------------------
#
# fold(var, operands = k state inits ++ m element tensors, params:
#   step:    an IR Program; its `input` vars are the state names + element
#            names below
#   dim:     the scan dim (must be chartless/unlabeled in the reference —
#            glue time charts back onto the result with with_charts)
#   state:   tuple of step-input names receiving the carried state (order
#            matches the first k operands); k >= 1
#   element: tuple of step-input names receiving per-step SLICES of the
#            element operands (the scan dim is select-ed away)
#   carry:   {state name -> step var producing the next state}; the carry
#            must keep the state's exact layout (checked, D17-style)
#   out:     ("emit", v) stacks step var v along the scan dim (inclusive
#            scan), or ("final", v) returns a carry var's final value
#   extent:  (start, stop), required only when there are no elements)
#
# Sequential by definition — the reference semantics of time-stepped state
# (PDE leapfrog, linear-attention/SSM matrix states). An associative tensor
# COMBINE is a future declaration the compiler may exploit for parallel
# evaluation; it does not change this denotation.


def _fold_parts(p):
    return (
        p["step"],
        p["dim"],
        tuple(p["state"]),
        tuple(p["element"]),
        dict(p["carry"]),
        tuple(p["out"]),
    )


def _fold_extent(ins: Instr, opshadows) -> tuple[int, int]:
    _, dim, state_names, elem_names, _, _ = _fold_parts(ins.params)
    if not state_names:
        raise ValueError("fold needs at least one state tensor")
    if elem_names:
        d = opshadows[ins.operands[len(state_names)]].dim(dim)
        return d.start, d.stop
    return tuple(ins.params["extent"])


def _fold_step_layouts(ins: Instr, opshadows) -> dict:
    """Input layouts for the step program: state layouts as carried, element
    layouts with the scan dim dropped."""
    _, dim, state_names, elem_names, _, _ = _fold_parts(ins.params)
    k = len(state_names)
    layouts = {}
    for sn, ov in zip(state_names, ins.operands[:k]):
        sh = opshadows[ov]
        layouts[sn] = sh.layout if isinstance(sh, Tensor) else sh
    for en, ov in zip(elem_names, ins.operands[k:]):
        sh = opshadows[ov]
        sh = sh.layout if isinstance(sh, Tensor) else sh
        layouts[en] = _dense_like(tuple(d for d in sh.dims if d.name != dim))
    return layouts


def _fold_infer(ins: Instr, opshadows) -> Layout:
    step, dim, state_names, elem_names, carry, (out_kind, out_var) = _fold_parts(ins.params)
    layouts = _fold_step_layouts(ins, opshadows)
    ss = infer(step, layouts)
    for sn in state_names:
        want = {(d.name, d.start, d.stop) for d in layouts[sn].dims}
        got = {(d.name, d.start, d.stop) for d in ss[carry[sn]].dims}
        if want != got:
            raise ValueError(f"fold carry {sn!r} changes the state layout: {sorted(want)} -> {sorted(got)}")
    if out_kind == "final":
        if out_var not in carry.values():
            raise ValueError("fold out=('final', v) requires v to be a carry output")
        return _dense_like(ss[out_var].dims)
    start, stop = _fold_extent(ins, opshadows)
    return _dense_like((Dim(dim, 0, start, stop),) + tuple(ss[out_var].dims))


def _run_fold(ins: Instr, env: dict) -> Tensor:
    from .compute import _tensor_like

    step, dim, state_names, elem_names, carry, (out_kind, out_var) = _fold_parts(ins.params)
    k = len(state_names)
    inits = [env[o] for o in ins.operands[:k]]
    elems = [env[o] for o in ins.operands[k:]]
    for e in elems:
        d = e.layout.dim(dim)
        if d.chart is not None or d.labels is not None:
            raise ValueError(f"fold scan dim {dim!r} must be chartless/unlabeled (strip_charts first)")
    shadow = _fold_infer(ins, {o: env[o].layout for o in ins.operands})  # validates carry/out too
    start, stop = _fold_extent(ins, {o: env[o].layout for o in ins.operands})
    carried = dict(zip(state_names, inits))
    emitted = []
    for t in range(start, stop):
        bound = dict(carried)
        for en, e in zip(elem_names, elems):
            bound[en] = e.select(**{dim: t})
        senv = run(step, bound)
        new = {}
        for sn in state_names:
            nv = senv[carry[sn]]
            issues = alignment(carried[sn], nv)
            if issues:
                details = "\n".join(f"  {msg!r}" for msg in issues)
                raise ValueError(f"fold carry {sn!r} drifted from its state layout:\n{details}")
            new[sn] = nv
        carried = new
        if out_kind == "emit":
            emitted.append(senv[out_var])
    if out_kind == "final":
        for sn in state_names:
            if carry[sn] == out_var:
                return carried[sn]
    if not emitted:
        return _tensor_like(np.zeros(tuple(d.size for d in shadow.dims)), shadow.dims)
    onames = emitted[0].names
    arr = np.stack([e.to_numpy(order=onames) if onames else e.to_numpy() for e in emitted], axis=0)
    return _tensor_like(arr, shadow.dims)


def _materialize(t: Tensor, p) -> Tensor:
    from .compute import _tensor_like

    order = tuple(p["order"])
    arr = t.to_numpy(order=order)
    dims = tuple(t.layout.dim(n) for n in order)
    plain = tuple(replace(d, chart=None, labels=None) for d in dims)
    return _tensor_like(arr, plain)


def run(prog: Program, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    """Evaluate the program over the reference compute layer; returns the
    full environment (every SSA var's value)."""
    env: dict[str, Tensor] = {}
    for ins in prog.instrs:
        if ins.op == "input":
            if ins.var not in inputs:
                raise KeyError(
                    f"missing input {ins.var!r} — virtual leaves analyze for free but "
                    f"execute only once provisioned: provision(root, source=init(...)"
                    f"|safetensors(...)) (200 §1.7)"
                )
            env[ins.var] = inputs[ins.var]
        elif ins.op == "const":
            env[ins.var] = _const(ins.params)
        elif ins.op == "iota":
            env[ins.var] = iota(env[ins.operands[0]], ins.params["name"], ins.params.get("unit"))
        elif ins.op == "random":
            from .random import _field

            env[ins.var] = _field(ins.params["dist"], ins.params["key"], env[ins.operands[0]])
        elif ins.op == "pointwise":
            env[ins.var] = pointwise(pw_marker(ins.params["f"]), *[env[o] for o in ins.operands])
        elif ins.op == "reduce":
            vals = tuple(env[o] for o in ins.operands)
            env[ins.var] = reduce(
                reducer(ins.params["f"]),
                vals[0] if len(vals) == 1 else vals,
                ins.params["dims"],
                ins.params.get("zero"),
            )
        elif ins.op == "scan":
            vals = tuple(env[o] for o in ins.operands)
            env[ins.var] = scan(
                reducer(ins.params["f"]),
                vals[0] if len(vals) == 1 else vals,
                ins.params["dim"],
                ins.params.get("zero"),
            )
        elif ins.op == "materialize":
            env[ins.var] = _materialize(env[ins.operands[0]], ins.params)
        elif ins.op == "fold":
            env[ins.var] = _run_fold(ins, env)
        elif ins.op == "with_value_units":
            env[ins.var] = env[ins.operands[0]].with_value_units(ins.params["value_units"])
        else:
            env[ins.var] = _LAYOUT_OPS[ins.op](env[ins.operands[0]], ins.params)
    return env


# ----------------------------------------------------------------------
# layout-only inference (shadows) — what the AD transform reads
# ----------------------------------------------------------------------


def _dense_like(dims: tuple[Dim, ...]) -> Layout:
    """A plain layout over the given dims with fabricated C-order strides
    (uniform 8-byte itemsize; only relative nesting matters)."""
    strides = []
    acc = 8
    for d in reversed(dims):
        strides.append(acc)
        acc *= max(d.size, 1)
    new = tuple(replace(d, stride=s, chart=d.chart, labels=d.labels) for d, s in zip(dims, reversed(strides)))
    return Layout(new, offset=-sum(d.stride * d.start for d in new))


def infer(prog: Program, input_layouts: dict) -> dict[str, Layout | GuardedLayout]:
    """Propagate layouts (no data) through the program. `input_layouts` maps
    each `input` var to a Layout/GuardedLayout or a Tensor (whose layout is
    taken)."""
    shadows: dict[str, Layout | GuardedLayout] = {}
    for ins in prog.instrs:
        shadows[ins.var] = infer_instr(ins, shadows, input_layouts)
    return shadows


def infer_instr(ins: Instr, shadows: dict, input_layouts: dict | None = None):
    """One instruction's shadow from its operands' — the single inference
    dispatch `infer` loops and the step lifter consumes incrementally."""
    if ins.op == "input":
        src = (input_layouts or {})[ins.var]
        return src.layout if isinstance(src, Tensor) else src
    if ins.op == "const":
        dims = tuple(Dim(name, 0, *_extent(extent)) for name, extent in ins.params.get("dims", ()))
        # stride-0 broadcast, exactly like run's _const
        return Layout(dims)
    if ins.op == "iota":
        base = shadows[ins.operands[0]]
        d = base.dim(ins.params["name"])
        new = tuple(replace(x, stride=(8 if x.name == d.name else 0)) for x in base.dims)
        return Layout(new, offset=-8 * d.start)
    if ins.op == "random":
        return _dense_like(shadows[ins.operands[0]].dims)
    if ins.op == "pointwise":
        return _dense_like(shadows[ins.operands[0]].dims)
    if ins.op == "reduce":
        dims = ins.params["dims"]
        names = (dims,) if isinstance(dims, str) else tuple(dims)
        survivors = tuple(d for d in shadows[ins.operands[0]].dims if d.name not in names)
        return _dense_like(survivors)
    if ins.op == "scan":
        return _dense_like(shadows[ins.operands[0]].dims)
    if ins.op == "materialize":
        order = tuple(ins.params["order"])
        src = shadows[ins.operands[0]]
        dims = tuple(replace(src.dim(n), chart=None, labels=None) for n in order)
        return _dense_like(dims)
    if ins.op == "fold":
        return _fold_infer(ins, shadows)
    if ins.op == "pad":
        return pad_layout(shadows[ins.operands[0]], ins.params["extents"])
    if ins.op == "stencil":
        p = ins.params
        return stencil_layout(
            shadows[ins.operands[0]],
            p["name"],
            p["k"],
            p.get("k_name"),
            p.get("dilation", 1),
        )
    if ins.op == "simplify":
        s = shadows[ins.operands[0]]
        return s.simplify() if isinstance(s, GuardedLayout) else s
    if ins.op == "with_value_units":
        return shadows[ins.operands[0]]
    # Layout/GuardedLayout share these ops' names and signatures with
    # Tensor, so run and infer dispatch through ONE table — a new
    # layout op is added in exactly one place. (pad/stencil/simplify
    # are the three genuine special cases, handled above.)
    return _LAYOUT_OPS[ins.op](shadows[ins.operands[0]], ins.params)


def _extent(spec) -> tuple[int, int]:
    if isinstance(spec, tuple):
        return spec
    return (0, spec)
