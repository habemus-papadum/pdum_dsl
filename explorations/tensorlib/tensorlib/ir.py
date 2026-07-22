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

import numpy as np

from .compute import Marker, Reducer, iota, pointwise, pw, red, reduce, scan
from .guarded import GuardedLayout, pad_layout, stencil_layout
from .layout import Dim, Layout
from .tensor import Tensor

PW = {m.name: m for m in vars(pw).values() if isinstance(m, Marker)}
RED = {m.name: m for m in vars(red).values() if isinstance(m, Reducer)}

_LEAF_OPS = ("input", "const", "iota")
_COMPUTE_OPS = ("pointwise", "reduce", "scan", "materialize")


@dataclass(frozen=True, eq=False)
class Instr:
    var: str
    op: str
    operands: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)

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
                raise KeyError(f"missing input {ins.var!r}")
            env[ins.var] = inputs[ins.var]
        elif ins.op == "const":
            env[ins.var] = _const(ins.params)
        elif ins.op == "iota":
            env[ins.var] = iota(env[ins.operands[0]], ins.params["name"], ins.params.get("unit"))
        elif ins.op == "pointwise":
            env[ins.var] = pointwise(PW[ins.params["f"]], *[env[o] for o in ins.operands])
        elif ins.op == "reduce":
            env[ins.var] = reduce(
                RED[ins.params["f"]],
                env[ins.operands[0]],
                ins.params["dims"],
                ins.params.get("zero"),
            )
        elif ins.op == "scan":
            env[ins.var] = scan(
                RED[ins.params["f"]],
                env[ins.operands[0]],
                ins.params["dim"],
                ins.params.get("zero"),
            )
        elif ins.op == "materialize":
            env[ins.var] = _materialize(env[ins.operands[0]], ins.params)
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
        if ins.op == "input":
            src = input_layouts[ins.var]
            shadows[ins.var] = src.layout if isinstance(src, Tensor) else src
        elif ins.op == "const":
            dims = tuple(Dim(name, 0, *_extent(extent)) for name, extent in ins.params.get("dims", ()))
            shadows[ins.var] = _dense_like(dims)
        elif ins.op == "iota":
            base = shadows[ins.operands[0]]
            d = base.dim(ins.params["name"])
            new = tuple(replace(x, stride=(8 if x.name == d.name else 0)) for x in base.dims)
            shadows[ins.var] = Layout(new, offset=-8 * d.start)
        elif ins.op == "pointwise":
            shadows[ins.var] = _dense_like(shadows[ins.operands[0]].dims)
        elif ins.op == "reduce":
            dims = ins.params["dims"]
            names = (dims,) if isinstance(dims, str) else tuple(dims)
            survivors = tuple(d for d in shadows[ins.operands[0]].dims if d.name not in names)
            shadows[ins.var] = _dense_like(survivors)
        elif ins.op == "scan":
            shadows[ins.var] = _dense_like(shadows[ins.operands[0]].dims)
        elif ins.op == "materialize":
            order = tuple(ins.params["order"])
            src = shadows[ins.operands[0]]
            dims = tuple(replace(src.dim(n), chart=None, labels=None) for n in order)
            shadows[ins.var] = _dense_like(dims)
        elif ins.op == "pad":
            shadows[ins.var] = pad_layout(shadows[ins.operands[0]], ins.params["extents"])
        elif ins.op == "stencil":
            p = ins.params
            shadows[ins.var] = stencil_layout(
                shadows[ins.operands[0]],
                p["name"],
                p["k"],
                p.get("k_name"),
                p.get("dilation", 1),
            )
        elif ins.op == "simplify":
            s = shadows[ins.operands[0]]
            shadows[ins.var] = s.simplify() if isinstance(s, GuardedLayout) else s
        else:
            layout = shadows[ins.operands[0]]
            method = {
                "slice": lambda: layout.slice(**ins.params["ranges"]),
                "select": lambda: layout.select(**ins.params["coords"]),
                "shift": lambda: layout.shift(**ins.params["deltas"]),
                "rename": lambda: layout.rename(**ins.params["mapping"]),
                "repeat": lambda: layout.repeat(
                    ins.params["name"],
                    ins.params["extent"],
                    ins.params.get("chart"),
                    ins.params.get("labels"),
                ),
                "flip": lambda: layout.flip(ins.params["name"]),
                "split": lambda: layout.split(ins.params["name"], **ins.params["parts"]),
                "merge": lambda: layout.merge(
                    tuple(ins.params["parts"]),
                    ins.params["name"],
                    ins.params.get("start", 0),
                ),
                "diagonal": lambda: layout.diagonal(
                    tuple(ins.params["parts"]), ins.params["name"], ins.params.get("chart")
                ),
                "window": lambda: layout.window(
                    ins.params["name"],
                    ins.params["k_name"],
                    ins.params["k"],
                    ins.params.get("dilation", 1),
                ),
                "decimate": lambda: layout.decimate(
                    ins.params["name"], ins.params["factor"], ins.params.get("phase", 0)
                ),
                "strip_charts": lambda: layout.strip_charts(),
            }[ins.op]
            shadows[ins.var] = method()
    return shadows


def _extent(spec) -> tuple[int, int]:
    if isinstance(spec, tuple):
        return spec
    return (0, spec)
