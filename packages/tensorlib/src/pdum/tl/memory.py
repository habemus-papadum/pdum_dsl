"""L1 — the peak-memory simulator: measure first (LEVELS.md, REPRESENTATIONS.md).

Walks a schedule of the program tracking live bytes. What makes this exact
rather than heuristic is the white-box representation:

- LAYOUT OPS ARE ALIASES: every layout op (slice/shift/repeat/pad/window/
  merge/...) is a zero-byte view; a view keeps its ROOT allocation alive
  until the last use of ANY alias.
- CLOSED FORMS ARE FREE: iota and const occupy nothing (FunctionalBuffers
  and stride-0 broadcasts) — masks, positions, and broadcast scalars drop
  out of the budget structurally, exactly the "some tensors are free to
  save" claim.
- ALLOCATIONS are the compute ops (pointwise/reduce/scan/fold/materialize)
  and inputs; sizes come from layout shadows.

The SCHEDULE is a separate, optimizable object (the L1 thesis): pass
`order` — any topological order of the same instructions — and the peak
moves; minimizing it is the pebbling problem the later passes attack.

Model simplifications (documented, all conservative-or-neutral):
- uniform 8-byte itemsize (the shadow convention; dtype-exact sizes later);
- numpy's internal temporaries are ignored (reference-layer artifact);
- fold's transient = new-carry bytes + a recursive simulation of one step
  (step inputs alias outer storage but are counted — an upper bound);
- guarded shadows count their box (fills are reads, not bytes);
- a value with no later use frees immediately after its instruction;
- inputs are resident for the whole run unless free_inputs=True.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# The DECLARED interior itemsize (200 §4): the reference oracle computes in
# f64 — a declared property, never semantics. L2's encoding assignment
# replaces this for materialized intermediates; BOUNDARY values take their
# byte truth from the inputs themselves (a Tensor's dtype, a Descriptor's
# encoding) — descriptor-fed, byte-exact, never the old uniform convention.
_INTERIOR_ITEM = 8
_FREE = frozenset({"iota", "const", "random"})


def _boundary_bytes(value, layout, local: bool) -> int | None:
    """Byte truth for an INPUT: Descriptor -> encoding-fed; Tensor -> its
    dtype's itemsize; a bare Layout has no boundary fact (None)."""
    enc = getattr(getattr(value, "encoding", None), "nbytes", None)
    if enc is not None:
        return value.encoding.nbytes(_numel(value.layout, local))
    itemsize = getattr(getattr(value, "dtype", None), "itemsize", None)
    if itemsize is not None:
        return _numel(layout, local) * itemsize
    return None


def _numel(layout, local: bool = False) -> int:
    n = 1
    for d in layout.dims:
        if local and d.level is not None:
            continue  # bound dims index devices, not this device's memory
        n *= d.size
    return n


@dataclass(frozen=True)
class MemoryReport:
    peak_bytes: int
    peak_at: str  # var whose instruction hits the high-water mark
    live_at_peak: tuple  # ((root_var_or_label, bytes), ...) at the peak
    timeline: tuple  # (var, live_bytes_after_instr) per scheduled instr
    alloc_bytes: Mapping[str, int]  # root var -> allocated bytes
    input_bytes: int


def peak_memory(
    region, input_layouts: dict | None = None, order=None, free_inputs: bool = False, local: bool = False, *, names=None
) -> MemoryReport:
    """With local=True, machine-bound dims (L3, PLACEMENT.md) count as
    device indices rather than memory: sizes become per-device shard bytes.
    Layouts ride the region nodes' shadows; ``input_layouts`` contributes
    boundary byte facts only (Tensors/Descriptors by name); ``order`` is a
    name sequence over the naming law's assignment — pass ``names``."""
    return _region_peak_memory(region, input_layouts or {}, order, free_inputs, local, names)


def _region_peak_memory(region, inputs, order, free_inputs, local, names) -> MemoryReport:
    """The Region face — the same live-byte walk over dialect nodes. The
    default schedule is params-then-walk-order (the export order, so the
    Program face's trajectory is reproduced exactly); consts are deferred
    scalars (zero bytes, never scheduled); yields and the yielded tuple are
    plumbing, never uses — an unconsumed output dies at its own node,
    matching the Program face."""
    from .dialect import LAYOUT_OP_NAMES, region_names, walk_region  # lazy: dialect's consumers import this module

    if names is None:
        raise ValueError(
            "region peak_memory joins on names — pass names=region_names(region, param_names) "
            "(LiftedStep.names / Assemblage.names carry it)"
        )
    param_ids = {id(p) for p in region.params}
    yielded = region.body[-1].args[0] if region.body else None
    nodes = list(region.params)
    seen: set[int] = set(param_ids)
    for n in walk_region(region):
        if id(n) in seen or n.op in ("core.yield", "core.const") or (n.op == "core.tuple" and n is yielded):
            continue
        seen.add(id(n))
        nodes.append(n)

    if order is not None:
        by_name = {names[id(n)]: n for n in nodes}
        if sorted(order) != sorted(by_name):
            raise ValueError("order must be a permutation of the program's vars")
        placed: set[int] = set()
        reordered = []
        for v in order:
            n = by_name[v]
            for a in n.args:
                if a.op != "core.const" and id(a) in seen and id(a) not in placed:
                    raise ValueError(f"order is not topological: {v!r} runs before its operand {names[id(a)]!r}")
            placed.add(id(n))
            reordered.append(n)
        nodes = reordered

    root: dict[int, int | None] = {}
    size: dict[int, int] = {}
    for n in nodes:
        if id(n) in param_ids:
            root[id(n)] = id(n)
            size[id(n)] = _numel(n.type.layout, local) * _INTERIOR_ITEM
            b = _boundary_bytes(inputs.get(names[id(n)]), n.type.layout, local)
            if b is not None:
                size[id(n)] = b
            continue
        if not n.op.startswith("tl."):
            raise KeyError(f"no peak_memory rule for op {n.op!r}")
        base = n.op[3:]
        if base in LAYOUT_OP_NAMES or base in ("with_value_units", "repeat_like"):
            root[id(n)] = root[id(n.args[0])]
        elif base in _FREE:
            root[id(n)] = None
        else:
            root[id(n)] = id(n)
            size[id(n)] = _numel(n.type.layout, local) * _INTERIOR_ITEM

    def transient(n) -> int:
        if n.op != "tl.fold":
            return 0
        p = dict(n.attrs)
        state, element = tuple(p["state"]), tuple(p["element"])
        k = len(state)
        carry_bytes = sum(size.get(id(a), _numel(a.type.layout, local) * _INTERIOR_ITEM) for a in n.args[:k])
        step = n.regions[0]
        sub = _region_peak_memory(step, {}, None, True, local, region_names(step, state + element))
        return carry_bytes + sub.peak_bytes

    last: dict[int, int] = {}
    for i, n in enumerate(nodes):
        if id(n) not in param_ids:
            for a in n.args:
                r = root.get(id(a))
                if r is not None:
                    last[r] = i
        r = root[id(n)]
        if r is not None:
            last.setdefault(r, i)  # dead values free immediately

    live = 0
    live_set: dict[int, int] = {}
    peak, peak_at, at_peak = 0, "", ()
    timeline = []
    for i, n in enumerate(nodes):
        alloc = size[id(n)] if root[id(n)] == id(n) else 0
        tr = transient(n)
        if live + alloc + tr > peak:
            peak, peak_at = live + alloc + tr, names[id(n)]
            snap = {names[r]: b for r, b in live_set.items()}
            if alloc:
                snap[names[id(n)]] = alloc
            if tr:
                snap["(fold transient)"] = tr
            at_peak = tuple(sorted(snap.items()))
        if alloc:
            live += alloc
            live_set[id(n)] = alloc
        for r, li in list(last.items()):
            if li == i and r in live_set and (free_inputs or r not in param_ids):
                live -= live_set.pop(r)
        timeline.append((names[id(n)], live))
    return MemoryReport(
        peak_bytes=peak,
        peak_at=peak_at,
        live_at_peak=at_peak,
        timeline=tuple(timeline),
        alloc_bytes={names[r]: b for r, b in size.items()},
        input_bytes=sum(size[id(p)] for p in region.params),
    )
