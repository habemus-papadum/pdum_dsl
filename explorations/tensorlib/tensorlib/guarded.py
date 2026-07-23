"""Guarded layouts (step 2): affine map + box domain + guards + fill.

The one planned extension to the core family. A Guard is a linear form over
named LATTICE coordinates with half-open bounds:

    lo <= sum(c_d * i_d) < hi

Inside the box domain, coordinates that violate any guard resolve to no
location at all (the tensor supplies a fill value); coordinates that satisfy
all guards resolve through the underlying affine map.

This stays white-box: every view operation rewrites the guards' coefficients
and bounds *algebraically* (substitution into the linear form), and
`simplify` discharges guards by interval arithmetic over the box — exactly
the reasoning a compiler would do (e.g. peel stencil boundaries, keep the
pure affine map in the interior). All the substitutions are linear:

    select   i = v            -> bounds shift, coeff drops
    shift    i -> i + d       -> bounds shift
    flip     i -> C - i       -> coeff negates, bounds shift
    split    x = const + sum(w_j * i_j)  -> coeff fans out over the parts
    decimate i = factor*j + p -> coeff scales, bounds shift
    diagonal x = z, y = z     -> coefficients add
    window/stencil x -> x + dilation*k   -> coeff gains a k term
    merge    only when coeffs are proportional to the mixed-radix weights
             (true for any guard that came from a split) — else refused,
             since the inverse map is div/mod and would not stay linear.

Charts (step 3) sit strictly above this machinery: physical coordinates are
normalized to lattice ints at the resolve/op boundary, and guards never see a
Quantity. Chart management (with_charts / strip_charts / recenter) never
touches the guards, because it never touches the lattice.

`pad` and `stencil` are the two constructors:
  - pad extends a dim's box beyond the mapped region (guard: 1*x in old box),
  - stencil adds a kernel dim x_k with x's stride (guard: x + dilation*x_k
    in x's box).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterator

from .chart import Chart
from .layout import (
    Dim,
    Injectivity,
    Layout,
    RangeSpec,
    _axis_of,
    _no_labels,
    _split_dim,
    _tap,
    as_range,
)


@dataclass(frozen=True)
class Guard:
    coeffs: tuple[tuple[str, int], ...]  # linear form sum(c * i), zero coeffs dropped
    lo: int  # inclusive
    hi: int  # exclusive

    def __post_init__(self) -> None:
        cleaned = tuple(sorted((n, c) for n, c in self.coeffs if c != 0))
        object.__setattr__(self, "coeffs", cleaned)

    def value(self, coords: dict[str, int]) -> int:
        return sum(c * coords[n] for n, c in self.coeffs)

    def ok(self, coords: dict[str, int]) -> bool:
        return self.lo <= self.value(coords) < self.hi

    def range_over(self, base: Layout) -> tuple[int, int]:
        """Inclusive [min, max] of the form over the box (box assumed non-empty)."""
        mn = mx = 0
        for n, c in self.coeffs:
            d = base.dim(n)
            a, b = c * d.start, c * (d.stop - 1)
            mn += min(a, b)
            mx += max(a, b)
        return mn, mx

    def __repr__(self) -> str:
        form = " + ".join(f"{c}*{n}" if c != 1 else n for n, c in self.coeffs) or "0"
        return f"Guard({self.lo} <= {form} < {self.hi})"


@dataclass(frozen=True)
class GuardedLayout:
    base: Layout  # its box is the (extended) view domain
    guards: tuple[Guard, ...]

    # ------------------------------------------------------------------
    # queries (mostly delegated to the affine base)
    # ------------------------------------------------------------------

    @property
    def dims(self) -> tuple[Dim, ...]:
        return self.base.dims

    @property
    def names(self) -> tuple[str, ...]:
        return self.base.names

    @property
    def offset(self) -> int:
        return self.base.offset

    def dim(self, name: str) -> Dim:
        return self.base.dim(name)

    def sizes(self) -> dict[str, int]:
        return self.base.sizes()

    @property
    def numel(self) -> int:
        return self.base.numel

    def domain(self) -> Iterator[dict[str, int]]:
        return self.base.domain()

    def is_contiguous(self, itemsize: int) -> bool:
        """Contiguity of the BOX (ignoring guards): whether the view's
        address image, fill slots included, packs one dense block."""
        return self.base.is_contiguous(itemsize)

    def phys(self, name: str, i: int):
        return self.base.phys(name, i)

    def snap(self, name: str, value, mode: str = "nearest") -> int:
        return self.base.snap(name, value, mode)

    def resolve(self, **coords) -> int | None:
        """The extended contract: a byte location, or None meaning 'fill'.
        Physical coordinates are normalized to the lattice first; guards
        always evaluate on lattice ints."""
        lat = self.base.to_lattice(coords)
        loc = self.base.get_loc(**lat)  # raises outside the box
        for g in self.guards:
            if not g.ok(lat):
                return None
        return loc

    def injectivity(self) -> Injectivity:
        """Conservative: guards only remove coordinates, so an injective base
        stays injective on the real (non-fill) subset. An aliased base might
        in principle be rescued by guards, so we refuse to certify either way."""
        inj = self.base.injectivity()
        return inj if inj is Injectivity.INJECTIVE else Injectivity.UNKNOWN

    def footprint(self, itemsize: int = 1) -> tuple[int, int] | None:
        """Byte interval of *real* locations, honoring guards where the
        algebra allows it exactly.

        When a guard's dims have strides proportional to its coefficients
        (s_d = lam * c_d — true for pad, stencil, and their images under the
        rewrites above), the group's address contribution is exactly
        lam * form, so the guard's bounds clamp it. Dims not covered this way
        fall back to their box interval (a conservative overestimate)."""
        if self.base.numel == 0:
            return None
        lo = hi = self.base.offset
        consumed: set[str] = set()
        leftover = {d.name for d in self.base.dims}
        for g in self.guards:
            if not g.coeffs:
                if not (g.lo <= 0 < g.hi):
                    return None  # constant guard always violated: all fill
                continue
            supp = [n for n, _ in g.coeffs]
            if any(n in consumed for n in supp):
                continue  # overlapping guards: keep the conservative path
            lam = _proportional(self.base, g)
            if lam is None:
                continue
            mn, mx = g.range_over(self.base)
            emn, emx = max(g.lo, mn), min(g.hi - 1, mx)
            if emn > emx:
                return None  # guard unsatisfiable over the box: all fill
            a, b = lam * emn, lam * emx
            lo += min(a, b)
            hi += max(a, b)
            consumed.update(supp)
            leftover.difference_update(supp)
        for name in leftover:
            d = self.base.dim(name)
            a, b = d.stride * d.start, d.stride * (d.stop - 1)
            lo += min(a, b)
            hi += max(a, b)
        return lo, hi + itemsize

    def with_offset_delta(self, delta: int) -> "GuardedLayout":
        return replace(self, base=self.base.with_offset_delta(delta))

    def __repr__(self) -> str:
        gs = ", ".join(repr(g) for g in self.guards)
        return f"GuardedLayout({self.base!r}, [{gs}])"

    # ------------------------------------------------------------------
    # algebraic simplification
    # ------------------------------------------------------------------

    def simplify(self) -> "GuardedLayout | Layout":
        """Discharge guards by interval arithmetic over the box. A guard the
        box can never violate is dropped; with no guards left, the layout
        collapses back into the core affine family."""
        if self.base.numel == 0:
            return self.base
        keep = []
        for g in self.guards:
            mn, mx = g.range_over(self.base)
            if g.lo <= mn and mx < g.hi:
                continue  # vacuous over this box
            keep.append(g)
        if not keep:
            return self.base
        return replace(self, guards=tuple(keep))

    def always_fill(self) -> bool:
        """True if some guard is unsatisfiable over the box: no coordinate
        resolves to a real location."""
        if self.base.numel == 0:
            return False
        for g in self.guards:
            if g.lo >= g.hi:
                return True  # the guard's own interval is empty
            mn, mx = g.range_over(self.base)
            if mx < g.lo or mn >= g.hi:
                return True
        return False

    def canonical(self) -> "GuardedLayout":
        """Order-free identity: canonical base plus deterministically sorted
        guards (their coefficient lists are already name-sorted)."""
        return GuardedLayout(
            self.base.canonical(),
            tuple(sorted(self.guards, key=lambda g: (g.coeffs, g.lo, g.hi))),
        )

    # ------------------------------------------------------------------
    # chart management — lattice untouched, so guards are untouched
    # ------------------------------------------------------------------

    def with_labels(self, **labels) -> "GuardedLayout":
        return replace(self, base=self.base.with_labels(**labels))

    def with_charts(self, **charts) -> "GuardedLayout":
        return replace(self, base=self.base.with_charts(**charts))

    def strip_charts(self) -> "GuardedLayout":
        return replace(self, base=self.base.strip_charts())

    def bind(self, **levels) -> "GuardedLayout":
        return replace(self, base=self.base.bind(**levels))

    def recenter(self, **deltas) -> "GuardedLayout":
        return replace(self, base=self.base.recenter(**deltas))

    # ------------------------------------------------------------------
    # view operations — base op + guard rewrite (on lattice ints)
    # ------------------------------------------------------------------

    def slice(self, **ranges) -> "GuardedLayout":
        # Shrinking the box never changes what the guards mean.
        return replace(self, base=self.base.slice(**ranges))

    def select(self, **coords) -> "GuardedLayout":
        # Substitute i = v into each form: bounds shift by -c*v, coeff drops.
        lat = self.base.to_lattice(coords)
        new_base = self.base.select(**lat)
        new_guards = []
        for g in self.guards:
            shift = sum(c * lat[n] for n, c in g.coeffs if n in lat)
            kept = tuple((n, c) for n, c in g.coeffs if n not in lat)
            new_guards.append(Guard(kept, g.lo - shift, g.hi - shift))
        return GuardedLayout(new_base, tuple(new_guards))

    def shift(self, **deltas) -> "GuardedLayout":
        # Relabel i -> i + delta: the form gains sum(c * delta) on both bounds.
        lat = self.base.to_lattice_deltas(deltas)
        new_base = self.base.shift(**lat)
        new_guards = []
        for g in self.guards:
            move = sum(c * lat.get(n, 0) for n, c in g.coeffs)
            new_guards.append(Guard(g.coeffs, g.lo + move, g.hi + move))
        return GuardedLayout(new_base, tuple(new_guards))

    def rename(self, **mapping: str) -> "GuardedLayout":
        new_base = self.base.rename(**mapping)
        new_guards = tuple(Guard(tuple((mapping.get(n, n), c) for n, c in g.coeffs), g.lo, g.hi) for g in self.guards)
        return GuardedLayout(new_base, new_guards)

    def repeat(self, name: str, extent: RangeSpec, chart: Chart | None = None, labels=None) -> "GuardedLayout":
        return replace(self, base=self.base.repeat(name, extent, chart, labels))

    def flip(self, name: str) -> "GuardedLayout":
        # Substitute i = C - i' (C = start + stop - 1): coeff negates, the
        # constant c*C moves into the bounds.
        d = self.base.dim(name)
        c_anchor = d.start + d.stop - 1
        new_base = self.base.flip(name)
        new_guards = []
        for g in self.guards:
            c = dict(g.coeffs).get(name)
            if c is None:
                new_guards.append(g)
                continue
            kept = tuple((n, -k if n == name else k) for n, k in g.coeffs)
            new_guards.append(Guard(kept, g.lo - c * c_anchor, g.hi - c * c_anchor))
        return GuardedLayout(new_base, tuple(new_guards))

    def split(self, name: str, **parts: RangeSpec) -> "GuardedLayout":
        # x = const + sum(w_j * i_j): substitute into every form mentioning x.
        d = self.base.dim(name)
        norm = tuple((n, as_range(spec)) for n, spec in parts.items())
        _, weights, const = _split_dim(d, norm)
        new_base = self.base.split(name, **parts)
        new_guards = []
        for g in self.guards:
            c = dict(g.coeffs).get(name)
            if c is None:
                new_guards.append(g)
                continue
            kept = [(n, k) for n, k in g.coeffs if n != name]
            kept += [(pn, c * w) for (pn, _), w in zip(norm, weights)]
            new_guards.append(Guard(tuple(kept), g.lo - c * const, g.hi - c * const))
        return GuardedLayout(new_base, tuple(new_guards))

    def merge(self, parts: tuple[str, ...], name: str, start: int = 0) -> "GuardedLayout":
        # The inverse map is div/mod, so a guard survives only when its
        # coefficients on the parts are proportional to the mixed-radix
        # weights (c_j = c * w_j) — then sum(c_j * i_j) = c * (x - start + W0)
        # and the form stays linear in the merged coordinate.
        ds = [self.base.dim(n) for n in parts]
        sizes = [d.size for d in ds]
        weights = [math.prod(sizes[j + 1 :]) for j in range(len(parts))]
        w0 = sum(w * d.start for w, d in zip(weights, ds))
        new_base = self.base.merge(parts, name, start=start)
        new_guards = []
        for g in self.guards:
            cs = dict(g.coeffs)
            if not any(pn in cs for pn in parts):
                new_guards.append(g)
                continue
            c = cs.get(parts[-1], 0)  # innermost weight is 1
            if any(cs.get(pn, 0) != c * w for pn, w in zip(parts, weights)):
                raise ValueError(
                    f"cannot merge {parts} under {g!r}: coefficients are not "
                    f"proportional to the mixed-radix weights {weights}, so the "
                    f"form would need div/mod in the merged coordinate"
                )
            kept = [(n, k) for n, k in g.coeffs if n not in parts] + [(name, c)]
            move = c * (start - w0)
            new_guards.append(Guard(tuple(kept), g.lo + move, g.hi + move))
        return GuardedLayout(new_base, tuple(new_guards))

    def diagonal(self, parts: tuple[str, ...], name: str, chart=None) -> "GuardedLayout":
        # Substitute x_j = z for every part: the coefficients add. A form
        # that cancels to a constant is evaluated over the diagonal correctly
        # (e.g. a guard on x - y is identically 0 there).
        new_base = self.base.diagonal(parts, name, chart)
        new_guards = []
        for g in self.guards:
            cs = dict(g.coeffs)
            total = sum(cs.get(p, 0) for p in parts)
            if all(cs.get(p, 0) == 0 for p in parts):
                new_guards.append(g)
                continue
            kept = [(n, k) for n, k in g.coeffs if n not in parts]
            kept.append((name, total))
            new_guards.append(Guard(tuple(kept), g.lo, g.hi))
        return GuardedLayout(new_base, tuple(new_guards))

    def window(self, name: str, k_name: str, k, dilation: int = 1) -> "GuardedLayout":
        # The tap position along x becomes x + dilation*k, so every form
        # mentioning x gains a dilation-scaled k term (same rewrite stencil
        # applies to pre-existing guards).
        new_base = self.base.window(name, k_name, k, dilation)
        new_guards = []
        for g in self.guards:
            c = dict(g.coeffs).get(name)
            if c is None:
                new_guards.append(g)
            else:
                new_guards.append(Guard(g.coeffs + ((k_name, c * dilation),), g.lo, g.hi))
        return GuardedLayout(new_base, tuple(new_guards))

    def decimate(self, name: str, factor: int, phase=0) -> "GuardedLayout":
        # Substitute i = factor*j + p: coeff scales by factor, bounds shift
        # by -c*p.
        d = self.base.dim(name)
        p = d.delta_to_lattice(phase) % factor
        new_base = self.base.decimate(name, factor, phase)
        new_guards = []
        for g in self.guards:
            c = dict(g.coeffs).get(name)
            if c is None:
                new_guards.append(g)
                continue
            kept = tuple((n, k * factor if n == name else k) for n, k in g.coeffs)
            new_guards.append(Guard(kept, g.lo - c * p, g.hi - c * p))
        return GuardedLayout(new_base, tuple(new_guards))


def _proportional(base: Layout, g: Guard) -> int | None:
    """If strides of g's support are an integer multiple lam of its coeffs
    (s_d = lam * c_d for all d), return lam, else None."""
    (n0, c0) = g.coeffs[0]
    s0 = base.dim(n0).stride
    if s0 % c0 != 0:
        return None
    lam = s0 // c0
    for n, c in g.coeffs:
        if base.dim(n).stride != lam * c:
            return None
    return lam


# ----------------------------------------------------------------------
# constructors
# ----------------------------------------------------------------------


def _unwrap(layout: Layout | GuardedLayout) -> tuple[Layout, tuple[Guard, ...]]:
    if isinstance(layout, GuardedLayout):
        return layout.base, layout.guards
    return layout, ()


def pad_layout(layout: Layout | GuardedLayout, extents: dict) -> GuardedLayout:
    """Extend dims' boxes beyond the mapped region; out-of-old-box reads
    become fill. The dual of slice. Extents may be lattice ints or on-lattice
    Quantities (positions on the dim's chart); the chart itself is unchanged
    — the box grows, the labeling stays."""
    base, guards = _unwrap(layout)
    new_guards = list(guards)
    new_dims = []
    pending = dict(extents)
    for d in base.dims:
        if d.name in pending:
            _no_labels(d, "pad")
            spec = pending.pop(d.name)
            if isinstance(spec, tuple):
                start, stop = (d.to_lattice(x) for x in spec)
            else:
                start, stop = as_range(spec)
            if start > d.start or stop < d.stop:
                raise ValueError(
                    f"pad range [{start}, {stop}) must contain {d.name}'s "
                    f"domain [{d.start}, {d.stop}) — to shrink, slice instead"
                )
            new_guards.append(Guard(((d.name, 1),), d.start, d.stop))
            d = replace(d, start=start, stop=stop)
        new_dims.append(d)
    if pending:
        raise KeyError(f"unknown dims in pad: {sorted(pending)}")
    return GuardedLayout(Layout(tuple(new_dims), base.offset), tuple(new_guards))


def stencil_layout(
    layout: Layout | GuardedLayout,
    name: str,
    k: tuple,
    k_name: str | None = None,
    dilation: int = 1,
) -> GuardedLayout:
    """Add a kernel dim x_k with stride stride_x * dilation and INCLUSIVE
    tap-index range k=(kmin, kmax) (the agreed D1 sugar for symmetric kernels
    like (-1, 1); bounds may be physical displacements on a charted dim):

        new[x=X, x_k=K] -> old[x = X + dilation*K],
        or fill when the tap leaves x's box.

    x's own box is unchanged; the guard is on the linear form
    x + dilation*x_k. On a charted dim the kernel gets a displacement chart
    on x's axis with step step_x * dilation, so taps carry physical labels.

    Because the tap position along x is now x + dilation*x_k, any *existing*
    guard mentioning x gets the same substitution (coeff c on x gains coeff
    c*dilation on x_k) — e.g. stenciling a padded tensor must see the padding
    as fill."""
    base, guards = _unwrap(layout)
    if not isinstance(dilation, int) or dilation < 1:
        raise ValueError(f"dilation must be a positive int, got {dilation!r}")
    d = base.dim(name)
    _no_labels(d, "stencil")
    k_min, k_max = (_tap(d, x, dilation) for x in k)
    if k_max < k_min:
        raise ValueError(f"empty stencil range ({k_min}, {k_max})")
    if k_name is None:
        k_name = f"{name}_k"
    if k_name in base.names:
        raise ValueError(f"dim {k_name!r} already exists")
    k_chart = None
    if d.chart is not None:
        k_chart = Chart(0 * d.chart.step, dilation * d.chart.step, "displacement", _axis_of(d))
    new_dims = []
    for x in base.dims:
        new_dims.append(x)
        if x.name == name:
            new_dims.append(
                Dim(
                    k_name,
                    stride=d.stride * dilation,
                    start=k_min,
                    stop=k_max + 1,
                    chart=k_chart,
                )
            )
    rewritten = []
    for g in guards:
        c = dict(g.coeffs).get(name)
        if c is None:
            rewritten.append(g)
        else:
            rewritten.append(Guard(g.coeffs + ((k_name, c * dilation),), g.lo, g.hi))
    guard = Guard(((name, 1), (k_name, dilation)), d.start, d.stop)
    return GuardedLayout(Layout(tuple(new_dims), base.offset), tuple(rewritten) + (guard,))
