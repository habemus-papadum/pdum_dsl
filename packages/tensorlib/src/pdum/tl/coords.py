"""The coordinate algebra (design 250): Frame, Coordinate, Displacement, Slice.

A dim's integer lattice is a ℤ-torsor: points and differences are distinct
sorts, and charts extend the torsor exactly to ℚ·units. These objects are
the nouns for laws the layout algebra already obeys — none of them ever
holds a stride or an offset; they bind to the observable frame, never to
the shadow, which is what makes a coordinate portable across every tensor
sharing the frame.

The torsor law, with refusals as theorems:

    Coordinate − Coordinate  → Displacement      (same frame)
    Coordinate ± Displacement → Coordinate       (bounds re-checked)
    Displacement ± Displacement, int·Displacement, −Displacement
    Coordinate + Coordinate   REFUSES            (adding points)
    cross-frame arithmetic    REFUSES            (geometry, not lattice)
    numeric use of a point    REFUSES            (coerce explicitly)

Promotion happens at operator boundaries only: an int (or integral
Fraction) or a step-multiple Quantity promotes to a Displacement,
exact-only — `snap` remains the one deliberate rounding door. A scalar
never promotes to a Coordinate: there is no origin-free way to make a
point.

A Coordinate is in-bounds by construction: there is no out-of-bounds
coordinate, only a refusal. One consequence: the frame-end exclusive
endpoint has no point, so a Slice to the end of the domain is spelled by
omission (stop=None; `t[y[128]:]` in subscript position).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from fractions import Fraction

from .chart import Chart
from .units import Quantity

_NOT_A_NUMBER = (
    "no numeric arithmetic on a Coordinate — a point is not a number; "
    "coerce explicitly (.i for the lattice int, .phys for the physical reading)"
)


def _lattice_int(x) -> int | None:
    """An integral value -> plain int; bool refused; None for non-integral."""
    if isinstance(x, bool):
        raise TypeError("bool is not a lattice value; pass an int")
    if isinstance(x, int):
        return x
    try:
        return operator.index(x)
    except TypeError:
        return None


def _cross(a: "Frame", b: "Frame") -> TypeError:
    return TypeError(
        f"cross-frame arithmetic ({a!r} vs {b!r}) refuses: "
        f"geometry, not lattice structure — coerce both sides explicitly"
    )


@dataclass(frozen=True)
class Frame:
    """One dim's observable identity: what TensorType keys on, reified.

    Frame is Dim minus the stride — name, half-open domain, and the
    optional labeling (chart | labels) or machine binding (level). It is
    also the host-tier point factory: `y[128]` is a Coordinate, and the
    handle's `__getitem__` is point-only — slices are built from
    Coordinate endpoints (`t[y[128] : y[256] : 2]`), never inside the
    handle."""

    name: str
    start: int
    stop: int
    chart: Chart | None = None
    labels: tuple[str, ...] | None = None
    level: str | None = None

    def __post_init__(self) -> None:
        if self.stop < self.start:
            raise ValueError(f"frame {self.name}: stop {self.stop} < start {self.start}")
        if self.labels is not None:
            object.__setattr__(self, "labels", tuple(self.labels))
            if self.chart is not None:
                raise ValueError(f"frame {self.name}: a frame carries a chart or labels, not both")
            if len(self.labels) != self.size:
                raise ValueError(f"frame {self.name}: {len(self.labels)} labels for {self.size} lattice points")
            if len(set(self.labels)) != len(self.labels):
                raise ValueError(f"frame {self.name}: labels must be unique")
        if self.level is not None and (self.chart is not None or self.labels is not None):
            raise ValueError(f"frame {self.name}: machine-bound frames are addresses — chartless and unlabeled")

    @property
    def size(self) -> int:
        return self.stop - self.start

    def contains(self, i: int) -> bool:
        return self.start <= i < self.stop

    # ---- the point factory ------------------------------------------------

    def __getitem__(self, spec) -> "Coordinate":
        if isinstance(spec, slice):
            raise TypeError(
                f"the frame handle makes points only; build slices from Coordinate "
                f"endpoints — t[{self.name}[a] : {self.name}[b] : step] — or Slice(start, stop, step)"
            )
        return Coordinate(self, self._to_lattice(spec))

    def snap(self, value, mode: str = "nearest") -> "Coordinate":
        """Deliberately round a physical value onto the lattice -> Coordinate."""
        if self.chart is None:
            raise TypeError(f"frame {self.name} has no chart to snap onto")
        return Coordinate(self, self.chart.snap(value, mode))

    def _to_lattice(self, coord) -> int:
        """A point spec -> lattice int. Quantities need a chart; strings need
        labels; integral values always mean the lattice."""
        if isinstance(coord, Quantity):
            if self.chart is None:
                raise TypeError(f"frame {self.name} has no chart; pass a lattice int, not {coord!r}")
            return self.chart.lattice(coord)
        if isinstance(coord, str):
            if self.labels is None:
                raise TypeError(f"frame {self.name} has no labels; got {coord!r}")
            try:
                return self.start + self.labels.index(coord)
            except ValueError:
                raise KeyError(f"{coord!r} is not a label of frame {self.name}; have {self.labels}") from None
        i = _lattice_int(coord)
        if i is None:
            raise TypeError(f"point for {self.name} must be int, Quantity, or label, got {coord!r}")
        return i

    def _delta_steps(self, x) -> int:
        """A displacement spec -> whole lattice steps (exact-only)."""
        if isinstance(x, Displacement):
            if x.frame != self:
                raise _cross(self, x.frame)
            return x.k
        if isinstance(x, Quantity):
            if self.chart is None:
                raise TypeError(f"frame {self.name} has no chart; pass lattice steps (int), not {x!r}")
            r = x / self.chart.step
            if isinstance(r, Quantity):
                raise ValueError(f"delta {x!r} has wrong dimensions for {self.name}'s chart")
            if r.denominator != 1:
                raise ValueError(
                    f"delta {x!r} is not a whole number of steps ({self.chart.step!r}) for frame {self.name}"
                )
            return int(r)
        if isinstance(x, Fraction):
            if x.denominator != 1:
                raise ValueError(f"delta {x} is not a whole number of lattice steps for frame {self.name}")
            return int(x)
        i = _lattice_int(x)
        if i is not None:
            return i
        raise TypeError(
            f"cannot use {x!r} as a displacement on {self.name}: whole lattice steps only — "
            f"an int, or a step-multiple Quantity (exact; snap() rounds positions deliberately)"
        )

    def __repr__(self) -> str:
        base = f"{self.name}[{self.start}:{self.stop})"
        if self.level is not None:
            return f"{base} %{self.level}"
        if self.chart is not None:
            return f"{base} @{self.chart}"
        if self.labels is not None:
            return f"{base} #[{','.join(self.labels)}]"
        return base


@dataclass(frozen=True)
class Coordinate:
    """An affine point on a frame's lattice, in-bounds by construction."""

    frame: Frame
    i: int

    def __post_init__(self) -> None:
        if not self.frame.contains(self.i):
            raise IndexError(
                f"{self.frame.name}={self.i} outside [{self.frame.start}, {self.frame.stop}): "
                f"there is no out-of-bounds Coordinate"
            )

    @property
    def phys(self) -> Quantity:
        if self.frame.chart is None:
            raise TypeError(f"frame {self.frame.name} has no chart")
        return self.frame.chart.phys(self.i)

    @property
    def label(self) -> str:
        if self.frame.labels is None:
            raise TypeError(f"frame {self.frame.name} has no labels")
        return self.frame.labels[self.i - self.frame.start]

    # ---- the torsor law ---------------------------------------------------

    def __add__(self, other) -> "Coordinate":
        if isinstance(other, Coordinate):
            raise TypeError(
                "cannot add two Coordinates: adding points is the affine crime — "
                "subtract for a Displacement, or add a Displacement to a point"
            )
        return Coordinate(self.frame, self.i + self.frame._delta_steps(other))

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Coordinate):
            if other.frame != self.frame:
                raise _cross(self.frame, other.frame)
            return Displacement(self.frame, self.i - other.i)
        return Coordinate(self.frame, self.i - self.frame._delta_steps(other))

    def __rsub__(self, other):
        raise TypeError(f"cannot subtract a Coordinate from {other!r}: a point is not a number")

    def _refuse_numeric(self, *_):
        raise TypeError(_NOT_A_NUMBER)

    __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _refuse_numeric
    __floordiv__ = __rfloordiv__ = __mod__ = __rmod__ = _refuse_numeric
    __pow__ = __rpow__ = __neg__ = __pos__ = _refuse_numeric

    def _ordered(self, other, op) -> bool:
        if not isinstance(other, Coordinate):
            return NotImplemented
        if other.frame != self.frame:
            raise _cross(self.frame, other.frame)
        return op(self.i, other.i)

    def __lt__(self, other):
        return self._ordered(other, operator.lt)

    def __le__(self, other):
        return self._ordered(other, operator.le)

    def __gt__(self, other):
        return self._ordered(other, operator.gt)

    def __ge__(self, other):
        return self._ordered(other, operator.ge)

    def __repr__(self) -> str:
        return f"{self.frame.name}[{self.i}]"


@dataclass(frozen=True)
class Displacement:
    """A vector along a frame's lattice: whole steps, unbounded — the domain
    bounds points, not differences."""

    frame: Frame
    k: int

    @property
    def phys(self) -> Quantity:
        if self.frame.chart is None:
            raise TypeError(f"frame {self.frame.name} has no chart")
        return self.k * self.frame.chart.step

    def __add__(self, other):
        if isinstance(other, Coordinate):
            return other + self
        return Displacement(self.frame, self.k + self.frame._delta_steps(other))

    def __radd__(self, other):
        return Displacement(self.frame, self.frame._delta_steps(other) + self.k)

    def __sub__(self, other):
        return Displacement(self.frame, self.k - self.frame._delta_steps(other))

    def __rsub__(self, other):
        return Displacement(self.frame, self.frame._delta_steps(other) - self.k)

    def __neg__(self) -> "Displacement":
        return Displacement(self.frame, -self.k)

    def __mul__(self, n) -> "Displacement":
        i = _lattice_int(n)
        if i is None:
            raise TypeError(
                f"only integers scale a Displacement on the lattice, not {n!r}; "
                f"physical scaling goes through .phys explicitly"
            )
        return Displacement(self.frame, self.k * i)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return f"{self.frame.name}[{self.k:+d}]"


@dataclass(frozen=True)
class Slice:
    """An arithmetic progression on one frame: half-open [start, stop),
    walking by a positive step Displacement. stop=None means the frame's
    stop — the frame-end exclusive endpoint has no point (Coordinates are
    in-bounds by construction), so slicing to the end is spelled by
    omission. A negative or zero step refuses: a Slice is a forward
    progression, and orientation is flip's job."""

    start: Coordinate
    stop: Coordinate | None = None
    step: Displacement | int | Quantity = 1

    def __post_init__(self) -> None:
        if not isinstance(self.start, Coordinate):
            raise TypeError(f"Slice start must be a Coordinate, got {self.start!r}")
        if self.stop is not None:
            if not isinstance(self.stop, Coordinate):
                raise TypeError(f"Slice stop must be a Coordinate (or None for the frame's stop), got {self.stop!r}")
            if self.stop.frame != self.start.frame:
                raise _cross(self.start.frame, self.stop.frame)
        step = (
            self.step
            if isinstance(self.step, Displacement)
            else Displacement(self.frame, self.frame._delta_steps(self.step))
        )
        if step.frame != self.frame:
            raise _cross(self.frame, step.frame)
        object.__setattr__(self, "step", step)
        if step.k < 1:
            raise ValueError(
                f"Slice step must be positive, got {step.k}: a Slice is a forward "
                f"progression — orientation is flip's job"
            )
        if self.stop_i < self.start_i:
            raise ValueError(f"empty-inverted Slice [{self.start_i}, {self.stop_i}) on {self.frame.name}")

    @property
    def frame(self) -> Frame:
        return self.start.frame

    @property
    def start_i(self) -> int:
        return self.start.i

    @property
    def stop_i(self) -> int:
        return self.frame.stop if self.stop is None else self.stop.i

    @property
    def step_k(self) -> int:
        return self.step.k

    @property
    def size(self) -> int:
        """The number of lattice points the progression visits."""
        span = self.stop_i - self.start_i
        return (span + self.step_k - 1) // self.step_k

    def __repr__(self) -> str:
        stop = "" if self.stop is None else str(self.stop.i)
        step = "" if self.step_k == 1 else f":{self.step_k}"
        return f"{self.frame.name}[{self.start.i}:{stop}{step}]"
