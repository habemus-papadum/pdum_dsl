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

from .ir import _LAYOUT_OPS, Program, _fold_step_layouts, infer

_ITEM = 8
_FREE = frozenset({"iota", "const"})


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
    prog: Program, input_layouts: dict, order=None, free_inputs: bool = False, local: bool = False
) -> MemoryReport:
    """With local=True, machine-bound dims (L3, PLACEMENT.md) count as
    device indices rather than memory: sizes become per-device shard
    bytes."""
    shadows = infer(prog, input_layouts)
    instrs = list(prog.instrs)
    if order is not None:
        by = {i.var: i for i in instrs}
        if sorted(order) != sorted(by):
            raise ValueError("order must be a permutation of the program's vars")
        seen: set[str] = set()
        reordered = []
        for v in order:
            for o in by[v].operands:
                if o not in seen:
                    raise ValueError(f"order is not topological: {v!r} runs before its operand {o!r}")
            seen.add(v)
            reordered.append(by[v])
        instrs = reordered

    defs = {i.var: i for i in instrs}
    root: dict[str, str | None] = {}
    size: dict[str, int] = {}
    for ins in instrs:
        if ins.op in _LAYOUT_OPS or ins.op == "with_value_units":
            root[ins.var] = root[ins.operands[0]]
        elif ins.op in _FREE:
            root[ins.var] = None
        else:
            root[ins.var] = ins.var
            size[ins.var] = _numel(shadows[ins.var], local) * _ITEM

    last_use: dict[str, int] = {}
    for i, ins in enumerate(instrs):
        for o in ins.operands:
            r = root[o]
            if r is not None:
                last_use[r] = i
        r = root[ins.var]
        if r is not None:
            last_use.setdefault(r, i)  # dead values free immediately

    def transient(ins) -> int:
        if ins.op != "fold":
            return 0
        state_names = tuple(ins.params["state"])
        k = len(state_names)
        carry_bytes = sum(size.get(o, _numel(shadows[o], local) * _ITEM) for o in ins.operands[:k])
        step_layouts = _fold_step_layouts(ins, shadows)
        sub = peak_memory(ins.params["step"], step_layouts, free_inputs=True, local=local)
        return carry_bytes + sub.peak_bytes

    live = 0
    live_set: dict[str, int] = {}
    peak, peak_at, at_peak = 0, "", ()
    timeline = []
    for i, ins in enumerate(instrs):
        alloc = size[ins.var] if root[ins.var] == ins.var else 0
        tr = transient(ins)
        if live + alloc + tr > peak:
            peak, peak_at = live + alloc + tr, ins.var
            snap = dict(live_set)
            if alloc:
                snap[ins.var] = alloc
            if tr:
                snap["(fold transient)"] = tr
            at_peak = tuple(sorted(snap.items()))
        if alloc:
            live += alloc
            live_set[ins.var] = alloc
        for r, li in list(last_use.items()):
            if li == i and r in live_set and (free_inputs or defs[r].op != "input"):
                live -= live_set.pop(r)
        timeline.append((ins.var, live))
    return MemoryReport(
        peak_bytes=peak,
        peak_at=peak_at,
        live_at_peak=at_peak,
        timeline=tuple(timeline),
        alloc_bytes=dict(size),
        input_bytes=sum(size[i.var] for i in instrs if i.op == "input"),
    )
