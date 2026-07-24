"""Static ops counting over the IR — exact scalar-operation tallies.

Design position on the mul/add/MAC question: COUNT primitives by NAME and
never collapse to a single "flops" scalar. A Counter keyed by marker name
("mul": 12, "add": 9, "exp": 4, "copy": 8) is the exact, layout-derived
fact; what a mul, an add, or an exp COSTS is a property of a machine, so it
belongs in a cost model (a weights dict applied at the end), not in the
count. In particular exp is just the bucket "exp" — whether it is 1 unit,
~20 flops of polynomial, or a table lookup is the cost model's opinion.

MACs are not a primitive either — they are a FUSION the hardware performs on
the contraction pattern. `fuse_mac=True` recognizes exactly that pattern in
the IR (a pointwise mul consumed solely by a reduce-sum) and reclassifies:
the muls become "mac" and the reduce's adds are absorbed (the accumulator
runs inside the MAC chain, starting at the identity) — so a matmul counts
m·n·k macs, matching the standard figure. Anything fancier (fusing adds
into FMA chains inside arbitrary expressions) is a compiler decision and is
deliberately not guessed at here.

Counts follow the REFERENCE semantics, sizes from `ir.infer` shadows:
- pointwise: (tree ops of the marker; a primitive is one op) x output numel.
- reduce: (numel_in - numel_out) combine ops (fold cost, identity-seeded);
  mean adds numel_out divides. Composite reducers: lift ops x numel_in,
  combine ops x (numel_in - lines), project ops x lines (reduce) or
  x numel_in (scan), where lines = numel_in / dim size.
- scan: (numel_in - lines) combine ops.
- materialize: numel "copy" (the one moving op — counted in its own bucket,
  never conflated with arithmetic).
- layout ops, iota, const, metadata: zero. Guarded operands count over the
  guard BOX (the reference layer evaluates fills too) — a λ-proportional
  refinement can come later; no silent narrowing here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from .ir import Program, _fold_extent, _fold_step_layouts, infer
from .nodes import Prim
from .registry import MARKERS, REDUCERS

_RED_COMBINE = {"sum": "add", "prod": "mul", "max": "maximum", "min": "minimum", "mean": "add"}


def _tree_ops(node) -> Counter:
    c: Counter = Counter()
    if isinstance(node, Prim):
        c[node.op] += 1
        for a in node.args:
            c += _tree_ops(a)
    return c


def _marker_ops(name: str) -> Counter:
    if name in MARKERS:
        return _tree_ops(MARKERS[name].body)
    return Counter({name: 1})


def _numel(layout) -> int:
    n = 1
    for d in layout.dims:
        n *= d.size
    return n


def _scale(ops: Counter, n: int) -> Counter:
    return Counter({k: v * n for k, v in ops.items()}) if n > 0 else Counter()


@dataclass(frozen=True)
class ProgramOps:
    per_var: Mapping[str, Counter]
    total: Counter

    def weighted(self, weights: Mapping[str, float] | None = None, default: float = 1.0) -> float:
        """Collapse the named buckets under a cost model. `weights` maps op
        names to costs (e.g. {"exp": 20, "div": 4, "mac": 1, "copy": 0});
        unnamed ops cost `default`."""
        weights = weights or {}
        return float(sum(v * weights.get(k, default) for k, v in self.total.items()))


def ops_count(prog: Program, input_layouts: dict, fuse_mac: bool = False) -> ProgramOps:
    """Count scalar operations per instruction (and in total), by name."""
    shadows = infer(prog, input_layouts)
    per: dict[str, Counter] = {}
    for ins in prog.instrs:
        c: Counter = Counter()
        if ins.op == "pointwise":
            c = _scale(_marker_ops(ins.params["f"]), _numel(shadows[ins.var]))
        elif ins.op in ("reduce", "scan"):
            f = ins.params["f"]
            nin = _numel(shadows[ins.operands[0]])
            if ins.op == "reduce":
                lines = _numel(shadows[ins.var])
            else:
                size = shadows[ins.operands[0]].dim(ins.params["dim"]).size
                lines = nin // size if size else 0
            folds = max(nin - lines, 0)
            if f in _RED_COMBINE:
                c[_RED_COMBINE[f]] = folds
                if f == "mean":
                    c["div"] += lines if ins.op == "reduce" else nin
            else:
                r = REDUCERS[f]
                lift = sum((_tree_ops(n) for n in r.lift), Counter())
                combine = sum((_tree_ops(n) for n in r.combine), Counter())
                project = _tree_ops(r.project)
                c = _scale(lift, nin) + _scale(combine, folds)
                c += _scale(project, lines if ins.op == "reduce" else nin)
        elif ins.op == "materialize":
            c["copy"] = _numel(shadows[ins.var])
        elif ins.op == "fold":
            # per-step cost x step count, recursively (nested folds compose)
            start, stop = _fold_extent(ins, shadows)
            sub = ops_count(ins.params["step"], _fold_step_layouts(ins, shadows), fuse_mac=fuse_mac)
            c = Counter({name: v * max(stop - start, 0) for name, v in sub.total.items()})
        per[ins.var] = c

    if fuse_mac:
        defs = {i.var: i for i in prog.instrs}
        consumers: Counter = Counter()
        for ins in prog.instrs:
            for o in ins.operands:
                consumers[o] += 1
        for ins in prog.instrs:
            if ins.op != "reduce" or ins.params["f"] != "sum" or len(ins.operands) != 1:
                continue
            src = defs.get(ins.operands[0])
            if src is None or src.op != "pointwise" or src.params["f"] != "mul":
                continue
            if consumers[src.var] != 1:
                continue  # the product is observable elsewhere; no fusion
            muls = per[src.var].pop("mul", 0)
            if muls:
                per[src.var]["mac"] = muls
            per[ins.var].pop("add", None)

    total: Counter = Counter()
    for c in per.values():
        total += c
    return ProgramOps(per, +total)
