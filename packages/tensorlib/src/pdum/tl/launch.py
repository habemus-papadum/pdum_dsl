"""Grid & launch (340): the data-parallel quotient as a plan artifact.

The launch never enters the IR — it rides the fusion Group as ordered
``(output dim, tile)`` pairs, the program count DERIVED as a product of
cdivs. Oversubscription replaces processor counts: the machine stays a
table (340 §5), and no column names an SM.

Feasibility is analytic: the per-program resident set — staged tiles
double-buffered plus fold-carried state, launch-clipped, at padded pow2
extents — must fit the machine's tightest level. The scorer is a
conservative model (fold carries sum where sequential loops could
share); triton's OutOfResources stays as the backstop tripwire.

Proposals are a pow2 ladder over the LEADING output dim, from the
smallest tile above the granularity floor (maximum oversubscription —
the analytic default) up to the largest feasible tile (maximum staged
reuse); the measured pick through the ledger settles that tension
(autotune-LITE, 340 §4). The floor default is a PLACEHOLDER for the
per-device measured fact (ruling 2) — a ledger row, never an
architectural constant.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dialect import _thaw_params

_ITEM = 4  # the translated column is f32 (320 §8)


def _pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


@dataclass(frozen=True)
class TileLevel:
    """One row of the 320 §5 machine table, per-program view."""

    name: str
    granule: int  # bytes per transaction line
    capacity: int  # bytes resident per program
    bandwidth: float | None = None  # bytes/s


@dataclass(frozen=True)
class TileMachine:
    levels: tuple[TileLevel, ...]

    def tightest(self) -> int:
        return min(lv.capacity for lv in self.levels)


def _nodes(region):
    seen, stack = set(), list(region.body)
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        yield n
        stack.extend(n.args)
        for r in n.regions:
            stack.extend(r.body)


def _clipped_bytes(dims, launch) -> int:
    b = _ITEM
    for d in dims:
        size = d.stop - d.start
        b *= _pow2(min(launch.get(d.name, size), size))
    return b


def footprint(region, launch: tuple = ()) -> int:
    """Per-program resident bytes under a launch: stages ×2 (double
    buffer) + fold carries. Conservative — see the module docstring."""
    lm = dict(launch)
    total = 0
    for n in _nodes(region):
        if n.op == "tl.stage":
            total += 2 * _clipped_bytes(n.type.dims, lm)
        elif n.op == "tl.fold":
            k = len(tuple(_thaw_params(dict(n.attrs))["state"]))
            for init in n.args[:k]:
                total += _clipped_bytes(init.type.dims, lm)
    return total


def feasible(region, launch, machine: TileMachine) -> tuple[bool, int]:
    fp = footprint(region, launch)
    return fp <= machine.tightest(), fp


def propose(region, machine: TileMachine, floor: int = 1024) -> tuple:
    """Feasible launches over the leading output dim, smallest tile first
    (the analytic default). Empty when nothing fits — the caller keeps
    grid (1,) and the tripwire owns the crash."""
    yld = region.body[-1].args[0]
    dims = yld.type.dims
    lead = dims[0]
    extent = lead.stop - lead.start
    rest = 1
    for d in dims[1:]:
        rest *= d.stop - d.start
    launches = []
    t = _pow2(extent)
    while True:
        ok, _ = feasible(region, ((lead.name, t),), machine)
        if ok:
            launches.append(((lead.name, t),))
        if t == 1 or rest * t <= floor:  # never shrink below the floor's work
            break
        t //= 2
    launches.reverse()
    return tuple(launches)
