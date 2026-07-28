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
- take: numel_out in the "take" bucket — one read+write per output element
  (200 §1.9), in its OWN bucket, never "copy": random access vs sequential
  copy is a machine property the cost model prices, so the count keeps them
  apart. scatter_add: numel_values "scatter" (the movement) + numel_values
  "add" (the accumulate) — the dual: one read+add+write per input element.
- layout ops, iota, const, metadata: zero. Guarded operands count over the
  guard BOX (the reference layer evaluates fills too) — a λ-proportional
  refinement can come later; no silent narrowing here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from .nodes import Prim
from .registry import MARKERS, REDUCERS

_RED_COMBINE = {"sum": "add", "prod": "mul", "max": "maximum", "min": "minimum", "mean": "add"}
_FREE_OPS = ("iota", "const", "random", "round_to", "repeat_like", "with_value_units")


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


def ops_count(region, input_layouts: dict | None = None, fuse_mac: bool = False, *, names=None) -> ProgramOps:
    """Count scalar operations per named value (and in total), by name.
    Sizes ride the region nodes' layout shadows (``input_layouts`` is not
    consulted; the argument survives for call-site symmetry with the other
    analyses); reports key by the naming law — pass ``names``."""
    return _region_ops_count(region, fuse_mac, names)


def _region_ops_count(region, fuse_mac: bool, names) -> ProgramOps:
    """The Region face — the same tallies over dialect nodes. Sizes come from
    the nodes' layout shadows; consts are deferred scalars (zero-cost
    plumbing, no report row); fuse_mac's sole-consumer test is DAG
    in-degree, with yields and the yielded tuple never counting as
    consumers (outputs are not consumption, matching the Program face)."""
    from .dialect import LAYOUT_OP_NAMES, region_names, walk_region  # lazy: dialect imports this module's consumers

    if names is None:
        raise ValueError(
            "region ops_count joins on names — pass names=region_names(region, param_names) "
            "(LiftedStep.names / Assemblage.names carry it)"
        )
    per: dict[str, Counter] = {}
    for p in region.params:
        per[names[id(p)]] = Counter()
    yielded = region.body[-1].args[0] if region.body else None
    seen: set[int] = {id(p) for p in region.params}
    for n in walk_region(region):
        if id(n) in seen or n.op in ("core.yield", "core.tuple", "core.const"):
            continue
        seen.add(id(n))
        if not n.op.startswith("tl."):
            raise KeyError(f"no ops_count rule for op {n.op!r}")
        base, p = n.op[3:], dict(n.attrs)
        c: Counter = Counter()
        if base == "pointwise":
            c = _scale(_marker_ops(p["f"]), _numel(n.type.layout))
        elif base in ("reduce", "scan"):
            f = p["f"]
            nin = _numel(n.args[0].type.layout)
            if base == "reduce":
                lines = _numel(n.type.layout)
            else:
                size = n.args[0].type.layout.dim(p["dim"]).size
                lines = nin // size if size else 0
            folds = max(nin - lines, 0)
            if f in _RED_COMBINE:
                c[_RED_COMBINE[f]] = folds
                if f == "mean":
                    c["div"] += lines if base == "reduce" else nin
            else:
                r = REDUCERS[f]
                lift = sum((_tree_ops(x) for x in r.lift), Counter())
                combine = sum((_tree_ops(x) for x in r.combine), Counter())
                project = _tree_ops(r.project)
                c = _scale(lift, nin) + _scale(combine, folds)
                c += _scale(project, lines if base == "reduce" else nin)
        elif base == "materialize":
            c["copy"] = _numel(n.type.layout)
        elif base == "take":
            c["take"] = _numel(n.type.layout)
        elif base == "scatter_add":
            nv = _numel(n.args[0].type.layout)
            c["scatter"] = nv
            c["add"] = nv
        elif base in ("argtopk", "argsort"):
            c[base] = _numel(n.args[0].type.layout)
        elif base == "fold":
            # per-step cost x step count, recursively (nested folds compose)
            state, element = tuple(p["state"]), tuple(p["element"])
            if element:
                d0 = next(dd for dd in n.args[len(state)].type.dims if dd.name == p["dim"])
                start, stop = d0.start, d0.stop
            else:
                start, stop = tuple(p["extent"])
            sub = _region_ops_count(n.regions[0], fuse_mac, region_names(n.regions[0], state + element))
            c = Counter({name: v * max(stop - start, 0) for name, v in sub.total.items()})
        elif base in _FREE_OPS or base in LAYOUT_OP_NAMES:
            pass  # closed forms, metadata, views, and broadcast declarations cost nothing
        else:
            raise KeyError(f"no ops_count rule for op {n.op!r}")
        per[names[id(n)]] = c

    if fuse_mac:
        consumers: Counter = Counter()
        for n in walk_region(region):
            # outputs are not consumption: yields and the yielded tuple are
            # plumbing; a yielded tl node still consumes its own operands
            if n.op == "core.yield" or (n.op == "core.tuple" and n is yielded):
                continue
            for a in n.args:
                consumers[id(a)] += 1
        for n in walk_region(region):
            if n.op != "tl.reduce" or dict(n.attrs).get("f") != "sum" or len(n.args) != 1:
                continue
            src = n.args[0]
            if src.op != "tl.pointwise" or dict(src.attrs).get("f") != "mul":
                continue
            if consumers[id(src)] != 1:
                continue  # the product is observable elsewhere; no fusion
            muls = per[names[id(src)]].pop("mul", 0)
            if muls:
                per[names[id(src)]]["mac"] = muls
            per[names[id(n)]].pop("add", None)

    total: Counter = Counter()
    for c in per.values():
        total += c
    return ProgramOps(per, +total)
