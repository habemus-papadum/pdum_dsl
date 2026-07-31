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
    """Feasible launches, smallest tile first, laddering EACH output dim
    in turn — the analytic default picks the dim by feasibility (a dV
    kernel grids its big axis, not its leading one; 340 §4's closed
    forms stay per-row). Leading-dim candidates come first, so every
    pick that existed before still wins its tie. Empty when nothing
    fits — the caller keeps grid (1,) and the tripwire owns the crash."""
    yld = region.body[-1].args[0]
    if yld.op == "core.tuple":
        yld = yld.args[0]  # artifacts ride the output's launch (§7.8)
    dims = yld.type.dims
    launches = []
    for i, lead in enumerate(dims):
        extent = lead.stop - lead.start
        rest = 1
        for j, d in enumerate(dims):
            if j != i:
                rest *= d.stop - d.start
        found = []
        t = _pow2(extent)
        while True:
            ok, _ = feasible(region, ((lead.name, t),), machine)
            if ok:
                found.append(((lead.name, t),))
            if t == 1 or rest * t <= floor:  # never shrink below the floor's work
                break
            t //= 2
        found.reverse()
        launches.extend(found)
    return tuple(launches)


# --- closed-form emptiness (340 §4b): the general engine ------------------
#
# Three-valued interval evaluation of a CLOSED forest (pointwise/consts
# over iota leaves) on coordinate boxes. Sound by construction: every
# unspelled op evaluates to "M" (maybe) — unproven never means wrong,
# only unpruned. The engine is geometry-free; templates own the mapping
# from fold iterations to boxes and the INERTNESS claim itself.

_INF = float("inf")


def tri_eval(n, boxes: dict) -> tuple:
    """("bool", "F"|"T"|"M") or ("num", lo, hi) of ``n`` over inclusive
    integer boxes keyed by dim name."""
    if n.op == "tl.iota":
        lo, hi = boxes[_thaw_params(dict(n.attrs))["name"]]
        return ("num", float(lo), float(hi))
    if n.op in ("core.const", "tl.const"):
        v = float(_thaw_params(dict(n.attrs))["value"])
        return ("num", v, v)
    if n.op in ("tl.repeat_like", "tl.repeat", "tl.simplify", "tl.rename"):
        if n.op == "tl.rename":
            return ("bool", "M")  # renames would need box rekeying — unspelled
        return tri_eval(n.args[0], boxes)
    if n.op == "tl.pointwise":
        return _tri_apply(_thaw_params(dict(n.attrs))["f"], [tri_eval(x, boxes) for x in n.args])
    return ("bool", "M")


def _tri_apply(f, args) -> tuple:
    nums = all(a[0] == "num" for a in args)
    if f in ("le", "lt", "ge", "gt") and nums and len(args) == 2:
        (_, a1, b1), (_, a2, b2) = args
        if f in ("ge", "gt"):
            (a1, b1), (a2, b2), f = (a2, b2), (a1, b1), {"ge": "le", "gt": "lt"}[f]
        if f == "le":
            return ("bool", "T" if b1 <= a2 else "F" if a1 > b2 else "M")
        return ("bool", "T" if b1 < a2 else "F" if a1 >= b2 else "M")
    if nums:
        iv = [(a[1], a[2]) for a in args]
        if f == "add":
            return ("num", iv[0][0] + iv[1][0], iv[0][1] + iv[1][1])
        if f == "sub":
            return ("num", iv[0][0] - iv[1][1], iv[0][1] - iv[1][0])
        if f == "neg":
            return ("num", -iv[0][1], -iv[0][0])
        if f == "mul":
            c = [x * y for x in iv[0] for y in iv[1]]
            return ("num", min(c), max(c))
        if f == "maximum":
            return ("num", max(iv[0][0], iv[1][0]), max(iv[0][1], iv[1][1]))
        if f == "minimum":
            return ("num", min(iv[0][0], iv[1][0]), min(iv[0][1], iv[1][1]))
        return ("num", -_INF, _INF)  # unspelled numeric: the infinite hull
    if all(a[0] == "bool" for a in args):
        ts = [a[1] for a in args]
        if f == "mul":  # AND
            return ("bool", "F" if "F" in ts else "T" if all(t == "T" for t in ts) else "M")
        if f == "maximum":  # OR
            return ("bool", "T" if "T" in ts else "F" if all(t == "F" for t in ts) else "M")
    if f == "where" and len(args) == 3 and args[0][0] == "bool":
        c, a, b = args
        if c[1] == "T":
            return a
        if c[1] == "F":
            return b
        if a[0] == b[0] == "bool":
            return ("bool", a[1] if a[1] == b[1] else "M")
        if a[0] == b[0] == "num":
            return ("num", min(a[1], b[1]), max(a[2], b[2]))
    return ("bool", "M")


def fit_affine(seq, lo_nat: int, hi_nat: int):
    """(a, d, c) such that clamp((a + d*g) // c, lo_nat, hi_nat) == seq[g]
    for every g, or None. Clamped ends are part of the fit (windows land
    on one line), and the floor-affine form covers every tile-ratio case
    (BT < SI gives ceil patterns, affine over a denominator). Exact,
    never approximate — a failed fit means no prune, never a wrong one."""
    n = len(seq)
    if n == 1:
        return (seq[0], 0, 1)

    def ok(a, d, c):
        return all(min(hi_nat, max(lo_nat, (a + d * g) // c)) == s for g, s in enumerate(seq))

    for d in sorted({seq[g + 1] - seq[g] for g in range(n - 1)}):  # pure affine first
        for g0 in range(n):
            a = seq[g0] - d * g0
            if ok(a, d, 1):
                return (a, d, 1)
    span = seq[-1] - seq[0]
    for c in range(2, 65):
        d0 = round(span * c / (n - 1))
        for d in (d0 - 1, d0, d0 + 1):
            for g0 in (0, n // 2, n - 1):
                base = seq[g0] * c - d * g0
                for r in range(c):
                    if ok(base + r, d, c):
                        return (base + r, d, c)
    return None
