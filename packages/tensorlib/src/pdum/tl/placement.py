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
on bound dims, scan/fold along a bound dim, take along a bound taken dim /
scatter_add over a bound consumed dim (all-to-all and partial-sum
all-reduce — 200 §1.9's later work; the refusals quote the fix), unknown
levels, mesh extents exceeding the level count. Forward programs only in v1 — gradients do not
yet carry bindings (PLACEMENT.md out-of-scope list).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

# The DECLARED interior itemsize (200 §4): collectives in the reference
# move f64 interior values — a declared oracle property. L2's encoding
# assignment makes per-value bytes real; boundary facts enter via the
# descriptors the capacity checks read.
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


def traffic(region, input_layouts: dict | None, machine: Machine, *, names=None) -> TrafficReport:
    """Read the region's communication off its algebra on bound dims.
    Layouts ride the nodes' shadows (``input_layouts`` is not consulted);
    events name their nodes through the naming law — pass ``names``."""
    return _region_traffic(region, machine, names)


def _region_traffic(region, machine: Machine, names) -> TrafficReport:
    """The Region face — the same algebra read off dialect nodes. Consts are
    deferred scalars (no layout to check; their materialized twins inherit
    the reference operand's bindings, which are checked at that operand)."""
    from .dialect import _thaw_params, region_names, walk_region  # lazy: dialect's consumers import this module

    if names is None:
        raise ValueError(
            "region traffic joins on names — pass names=region_names(region, param_names) "
            "(LiftedStep.names / Assemblage.names carry it)"
        )
    events: list[Collective] = []
    yielded = region.body[-1].args[0] if region.body else None
    seen: set[int] = set()
    for p in region.params:
        _check_bound(p.type.layout, machine)
        seen.add(id(p))
    for n in walk_region(region):
        if id(n) in seen or n.op in ("core.yield", "core.const") or (n.op == "core.tuple" and n is yielded):
            continue
        seen.add(id(n))
        if not n.op.startswith("tl."):
            raise KeyError(f"no traffic rule for op {n.op!r}")
        base, p = n.op[3:], _thaw_params(dict(n.attrs))
        lays = [t.type.layout for t in (n, *n.args) if getattr(t.type, "layout", None) is not None]
        for lay in lays:
            _check_bound(lay, machine)
        var = names[id(n)]
        if base == "reduce":
            dims = p["dims"]
            dims = (dims,) if isinstance(dims, str) else tuple(dims)
            src = n.args[0].type.layout
            for dn in dims:
                d = src.dim(dn)
                if d.level is not None:
                    events.append(
                        Collective(
                            var,
                            "all_reduce",
                            d.level,
                            int(Fraction(2 * (d.size - 1), d.size) * local_bytes(n.type.layout)),
                        )
                    )
        elif base == "merge":
            src = n.args[0].type.layout
            for dn in p["parts"]:
                d = src.dim(dn)
                if d.level is not None:
                    events.append(
                        Collective(
                            var,
                            "all_gather",
                            d.level,
                            int(Fraction(d.size - 1, d.size) * local_bytes(n.type.layout)),
                        )
                    )
        elif base == "take":
            if n.args[0].type.layout.dim(p["dim"]).level is not None:
                raise NotImplementedError(
                    f"take along machine-bound dim {p['dim']!r} is an "
                    f"all-to-all, not modeled in v1 — colocate (unbind the table) "
                    f"or all-gather the table (merge the bound dim) first"
                )
        elif base == "scatter_add":
            vsh, ish = n.args[0].type.layout, n.args[1].type.layout
            consumed = p.get("over") or tuple(ish.names)
            for dn in ish.names:
                lv, li = vsh.dim(dn).level, ish.dim(dn).level
                if lv != li:
                    raise NotImplementedError(
                        f"scatter_add: values and idx place shared dim {dn!r} on "
                        f"different machine levels ({lv!r} vs {li!r}) — colocate them"
                    )
                if li is not None and dn in consumed:
                    raise NotImplementedError(
                        f"scatter_add over machine-bound dim {dn!r} leaves per-device "
                        f"partial sums that need an all-reduce, not modeled in v1 — "
                        f"unbind (gather) the consumed dim first"
                    )
        elif base in ("argtopk", "argsort"):
            if n.args[0].type.layout.dim(p["dim"]).level is not None:
                raise NotImplementedError(
                    f"{base} along a machine-bound dim (a distributed sort) is "
                    f"not modeled — gather (merge the bound dim) first"
                )
        elif base == "scan":
            if n.args[0].type.layout.dim(p["dim"]).level is not None:
                raise NotImplementedError("scan along a machine-bound dim (distributed scan) is not modeled")
        elif base == "fold":
            state, element = tuple(p["state"]), tuple(p["element"])
            if element:
                d0 = n.args[len(state)].type.layout.dim(p["dim"])
                if d0.level is not None:
                    raise NotImplementedError("fold along a machine-bound dim is not modeled")
                start, stop = d0.start, d0.stop
            else:
                start, stop = tuple(p["extent"])
            step = n.regions[0]
            sub = _region_traffic(step, machine, region_names(step, state + element))
            for c in sub.collectives:
                events.append(Collective(var, c.kind, c.level, c.bytes * max(stop - start, 0)))
        elif base in _SURGERY:
            src = n.args[0].type.layout
            keys = {"slice": "ranges", "select": "coords", "shift": "deltas", "pad": "extents"}
            if base in keys:
                touched = tuple(p[keys[base]])
            elif base == "diagonal":
                touched = tuple(p["parts"])
            else:
                touched = (p["name"],)
            for dn in touched:
                if dn in src.names and src.dim(dn).level is not None:
                    raise NotImplementedError(f"{base} on machine-bound dim {dn!r} is not modeled (unbind first)")
    per_level: Counter = Counter()
    for c in events:
        per_level[c.level] += c.bytes
    return TrafficReport(tuple(events), per_level)
