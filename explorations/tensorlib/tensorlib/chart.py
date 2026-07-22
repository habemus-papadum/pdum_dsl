"""Coordinate charts (step 3): exact physical labels for lattice dims.

A Chart is an affine relabeling of a dim's integer lattice:

    phys(i) = origin + i * step        (origin, step exact Quantities)

Today's plain-integer indexing is the degenerate chart (origin 0, step 1,
dimensionless) left implicit; attaching a chart changes no address or guard
arithmetic — the lattice keeps full citizenship, charts are the API surface.

`kind` records the affine/vector distinction: positions have an origin
(stage coordinates, timestamps); displacements are origin-free differences
(kernel taps, block-local offsets).

`axis` names the physical axis a dim lives on. Several dims may share one
axis — split parts, window/stencil kernel dims — and the invariant is:

    physical position on axis A = sum of the chart labels of all dims
    tagged A (at most one of which is a position; the rest displacements).

Ops maintain the invariant mechanically: split tags its parts with the
parent's axis, kernels get the anchor's axis, and select folds a removed
dim's label into a remaining sibling on the same axis (promoting a
displacement to a position when the position itself was selected). This is
what makes "physical labels stay glued to the data" a theorem across ops
rather than a per-op convention.

Membership is exact-only: `lattice()` raises off-lattice; `snap()` is the
explicit, deliberate way to round onto the lattice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .units import Quantity
from .units import q as _q

KINDS = ("position", "displacement")


def _as_quantity(value) -> Quantity:
    if isinstance(value, Quantity):
        return value
    return _q(value)  # str/int/Fraction; floats rejected inside


def _fr_gcd(a: Fraction, b: Fraction) -> Fraction:
    return Fraction(
        math.gcd(a.numerator * b.denominator, b.numerator * a.denominator),
        a.denominator * b.denominator,
    )


@dataclass(frozen=True)
class Chart:
    origin: Quantity
    step: Quantity
    kind: str = "position"
    axis: str | None = None  # name of the physical axis this dim labels

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _as_quantity(self.origin))
        object.__setattr__(self, "step", _as_quantity(self.step))
        if self.origin.dims != self.step.dims:
            raise ValueError(f"origin {self.origin!r} and step {self.step!r} have different dimensions")
        if not self.step:
            raise ValueError("chart step must be nonzero")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")

    def phys(self, i: int) -> Quantity:
        return self.origin + i * self.step

    def ratio(self, value) -> Fraction:
        """(value - origin) / step as an exact rational lattice position."""
        return (_as_quantity(value) - self.origin) / self.step

    def lattice(self, value) -> int:
        """Exact-only membership: raises if `value` is off-lattice."""
        r = self.ratio(value)
        if r.denominator != 1:
            raise ValueError(
                f"{_as_quantity(value)!r} is off-lattice for {self!r} "
                f"(lattice position {r}); use snap() to round deliberately"
            )
        return int(r)

    def snap(self, value, mode: str = "nearest") -> int:
        """Deliberate rounding onto the lattice."""
        r = self.ratio(value)
        if mode == "floor":
            return math.floor(r)
        if mode == "ceil":
            return math.ceil(r)
        if mode == "nearest":
            return round(r)  # ties to even, like round()
        raise ValueError(f"mode must be floor/ceil/nearest, got {mode!r}")

    def commensurable(self, other: "Chart") -> bool:
        """Can the two lattices interoperate pointwise (no resampling)?
        True iff same dimensions and the origins differ by an integer
        multiple of the gcd of the steps."""
        if self.step.dims != other.step.dims:
            return False
        g = _fr_gcd(abs(self.step.base), abs(other.step.base))
        return ((self.origin.base - other.origin.base) / g).denominator == 1

    def __repr__(self) -> str:
        k = "pos" if self.kind == "position" else "disp"
        ax = f"{self.axis}: " if self.axis is not None else ""
        return f"{k}[{ax}{self.origin!r} step {self.step!r}]"


def chart(origin, step, kind: str = "position", axis: str | None = None) -> Chart:
    """Convenience constructor coercing strings/ints/Fractions exactly:
    chart("-1.5 um", "0.25 um")."""
    return Chart(_as_quantity(origin), _as_quantity(step), kind, axis)


def characteristic(rate, label_along: str):
    """Diagonal chart combinator for a rate-related pair of axes (x = rate·t,
    e.g. a wave characteristic): validates that the lattice is EXACTLY
    characteristic — step(label_along) == rate * step(other), the CFL-style
    commensurability condition — and labels z along `label_along` (a dim
    name). All the physics lives here, named and testable; `diagonal` itself
    never guesses.

        xi = t2d.diagonal(("x", "t"), "xi",
                          chart=characteristic(q("2 um/ms"), "x"))
    """
    rate = _as_quantity(rate)

    def combine(dims) -> Chart:
        if len(dims) != 2:
            raise ValueError("characteristic() combines exactly two dims")
        by = {d.name: d for d in dims}
        if label_along not in by:
            raise KeyError(f"{label_along!r} is not one of {sorted(by)}")
        d_lab = by.pop(label_along)
        (d_other,) = by.values()
        if d_lab.chart is None or d_other.chart is None:
            raise TypeError("characteristic() needs charts on both dims")
        expect = rate * d_other.chart.step
        if d_lab.chart.step != expect:
            raise ValueError(
                f"lattice is not characteristic for rate {rate!r}: "
                f"step({d_lab.name}) = {d_lab.chart.step!r} but rate * "
                f"step({d_other.name}) = {expect!r}"
            )
        return d_lab.chart

    return combine
