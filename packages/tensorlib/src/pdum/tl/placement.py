"""L3-lite: the machine tree and the traffic pass (PLACEMENT.md).

The machine is DATA — a tuple of levels (cluster → … → lane capable),
each with a count, a memory capacity, and a sibling link. The IR never
names hardware; dims bind to LEVEL NAMES and the machine gives them
meaning (LEVELS.md's two-artifact principle). v1 exercises a single mesh
level; nothing here assumes depth-1.

There are NO collective ops. `traffic` walks an ordinary program and reads
communication off the existing algebra applied to machine-bound dims:

- reduce over a bound dim  -> all-reduce  (2·(p−1)/p × result-local bytes,
  ring; the result drops the mesh dim and is replicated — exactly reduce's
  value semantics)
- merge of a bound part    -> all-gather  ((p−1)/p × merged bytes)
- repeat+bind / split+bind -> free (distributing what every device holds,
  or viewing a shard of it — the mesh analogue of "masks are free")

Everything the model does NOT cover refuses loudly (D17): lattice surgery
on bound dims, scan/fold along a bound dim, unknown levels, mesh extents
exceeding the level count. Forward programs only in v1 — gradients do not
yet carry bindings (PLACEMENT.md out-of-scope list).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

from .ir import Program, _fold_parts, _fold_step_layouts, infer

_ITEM = 8
_SURGERY = ("slice", "select", "shift", "flip", "decimate", "window", "stencil", "pad", "split", "diagonal")


@dataclass(frozen=True)
class Level:
    name: str
    count: int
    capacity: int | None = None  # bytes of this level's memory, per instance
    link_bandwidth: float | None = None  # bytes/s between siblings
    link_latency: float = 0.0  # seconds per message


@dataclass(frozen=True)
class Machine:
    """Outermost -> innermost levels. Swap the tree, retarget the model."""

    levels: tuple[Level, ...]

    def __post_init__(self) -> None:
        names = [lv.name for lv in self.levels]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate level names: {names}")

    def level(self, name: str) -> Level:
        for lv in self.levels:
            if lv.name == name:
                return lv
        raise KeyError(f"machine has no level {name!r} (levels: {[lv.name for lv in self.levels]})")


def mesh(count: int, name: str = "gpu", **kw) -> Machine:
    """The v1 machine: one mesh level."""
    return Machine((Level(name, count, **kw),))


def local_bytes(layout, itemsize: int = _ITEM) -> int:
    """Per-device bytes of one value: bound dims index devices, not memory."""
    n = 1
    for d in layout.dims:
        if d.level is None:
            n *= d.size
    return n * itemsize


@dataclass(frozen=True)
class Collective:
    var: str  # the instruction whose algebra implies the communication
    kind: str  # "all_reduce" | "all_gather"
    level: str
    bytes: int  # per participating device


@dataclass(frozen=True)
class TrafficReport:
    collectives: tuple[Collective, ...]
    per_level: Counter  # level -> total bytes per device

    def time(self, machine: Machine) -> float:
        """α-β estimate: latency per collective + bytes over link bandwidth.
        Levels without a declared bandwidth contribute latency only."""
        t = 0.0
        for c in self.collectives:
            lv = machine.level(c.level)
            t += lv.link_latency
            if lv.link_bandwidth:
                t += c.bytes / lv.link_bandwidth
        return t


def _check_bound(layout, machine: Machine) -> None:
    for d in layout.dims:
        if d.level is None:
            continue
        lv = machine.level(d.level)  # raises for unknown levels
        if d.size > lv.count:
            raise ValueError(f"dim {d.name}: mesh extent {d.size} exceeds level {d.level!r} count {lv.count}")


def traffic(prog: Program, input_layouts: dict, machine: Machine) -> TrafficReport:
    """Read the program's communication off its algebra on bound dims."""
    shadows = infer(prog, input_layouts)
    events: list[Collective] = []
    for ins in prog.instrs:
        for o in (ins.var, *ins.operands):
            _check_bound(shadows[o], machine)
        if ins.op == "reduce":
            names = ins.params["dims"]
            names = (names,) if isinstance(names, str) else tuple(names)
            src = shadows[ins.operands[0]]
            for n in names:
                d = src.dim(n)
                if d.level is not None:
                    p = d.size
                    nbytes = int(Fraction(2 * (p - 1), p) * local_bytes(shadows[ins.var]))
                    events.append(Collective(ins.var, "all_reduce", d.level, nbytes))
        elif ins.op == "merge":
            src = shadows[ins.operands[0]]
            for n in ins.params["parts"]:
                d = src.dim(n)
                if d.level is not None:
                    p = d.size
                    nbytes = int(Fraction(p - 1, p) * local_bytes(shadows[ins.var]))
                    events.append(Collective(ins.var, "all_gather", d.level, nbytes))
        elif ins.op == "scan":
            if shadows[ins.operands[0]].dim(ins.params["dim"]).level is not None:
                raise NotImplementedError("scan along a machine-bound dim (distributed scan) is not modeled")
        elif ins.op == "fold":
            _, dim, _, elem_names, _, _ = _fold_parts(ins.params)
            k = len(ins.params["state"])
            probe = shadows[ins.operands[k]] if elem_names else None
            if probe is not None and probe.dim(dim).level is not None:
                raise NotImplementedError("fold along a machine-bound dim is not modeled")
            start, stop = (probe.dim(dim).start, probe.dim(dim).stop) if probe is not None else ins.params["extent"]
            sub = traffic(ins.params["step"], _fold_step_layouts(ins, shadows), machine)
            for c in sub.collectives:
                events.append(Collective(ins.var, c.kind, c.level, c.bytes * max(stop - start, 0)))
        elif ins.op in _SURGERY:
            src = shadows[ins.operands[0]]
            keys = {
                "slice": "ranges",
                "select": "coords",
                "shift": "deltas",
                "pad": "extents",
            }
            if ins.op in keys:
                touched = tuple(ins.params[keys[ins.op]])
            elif ins.op == "diagonal":
                touched = tuple(ins.params["parts"])
            else:
                touched = (ins.params["name"],)
            for n in touched:
                if n in src.names and src.dim(n).level is not None:
                    raise NotImplementedError(f"{ins.op} on machine-bound dim {n!r} is not modeled (unbind first)")
    per_level: Counter = Counter()
    for c in events:
        per_level[c.level] += c.bytes
    return TrafficReport(tuple(events), per_level)
