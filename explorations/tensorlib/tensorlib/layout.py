"""Core layout algebra (steps 1 + 3): affine map + box domain.

Semantic ground truth:

    loc(coords) = offset + sum(stride_d * i_d)

with each *raw* lattice coordinate i_d in the half-open domain [start_d, stop_d).

Conventions (agreed design decisions):
  - D1: domains are half-open [start, stop).
  - D2: the offset is kept; it is the constant term of the affine map.
  - D3: raw coordinates enter the sum (not i - start). Domains are purely
    *domains*: slicing shrinks them and never touches addressing. A negative
    coordinate is a literal coordinate, not Python's "from the end".
  - D4: strides and offsets here are in BYTES; element-unit convenience lives
    at the Tensor level.
  - D5: dimension identity is its NAME. The order of `dims` carries no
    addressing semantics; it is retained only as a default for display and
    materialization. `canonical()` is the order-free identity.
  - D7 (step 3): a dim may carry a Chart — an exact affine physical labeling
    phys(i) = origin + i*step over the same integer lattice. Charts change no
    address or guard arithmetic; every op below rewrites them mechanically.
    Coordinates given as Quantities are converted at the boundary
    (exact-only); plain ints always mean lattice coordinates (compiler mode).
  - Axis invariant (see chart.py): the physical position on an axis is the
    sum of the labels of all dims tagged with that axis, at most one of
    which is a position. Ops preserve this: charts stay glued to the data
    under shift/flip, and select folds a removed dim's label into a
    remaining sibling on the same axis.
  - D16: diagonal never guesses a labeling. Same-axis parts get the forced
    label-sum chart; otherwise z is uncharted unless the caller supplies a
    Chart or a combinator (consumed_dims) -> Chart | None.
  - D18: a dim may instead carry categorical LABELS — the nominal rung of
    the measurement ladder: a bijection lattice <-> names with no arithmetic
    (no step, no displacements). Ops with no meaning on nominal data (split,
    merge, window, stencil, diagonal, pad) refuse labeled dims; slice, shift,
    flip, decimate, select thread the labels glued to the data. Attaching a
    chart REPLACES the labels — the nominal -> interval upgrade.
"""

from __future__ import annotations

import enum
import itertools
import math
import operator
from dataclasses import dataclass, replace
from typing import Iterator, Mapping

from .chart import Chart
from .chart import chart as make_chart
from .units import Quantity

# An extent spec: an int n (meaning the lattice range [0, n)) or an explicit
# (start, stop) whose bounds may be lattice ints or on-lattice Quantities.
RangeSpec = "int | tuple[int | Quantity, int | Quantity]"


def _lattice_int(x) -> int | None:
    """Coerce an integral value (int, numpy integer, anything with __index__)
    to a plain int. bool is refused — a boolean coordinate is a bug, not an
    index. Returns None for non-integral values."""
    if isinstance(x, bool):
        raise TypeError("bool is not a lattice coordinate; pass an int")
    if isinstance(x, int):
        return x
    try:
        return operator.index(x)
    except TypeError:
        return None


def as_range(spec) -> tuple[int, int]:
    """Lattice-only range spec: int n -> (0, n), or an int pair."""
    if isinstance(spec, tuple):
        raw_start, raw_stop = spec
    else:
        raw_start, raw_stop = 0, spec
    start, stop = _lattice_int(raw_start), _lattice_int(raw_stop)
    if start is None or stop is None:
        raise TypeError(f"expected lattice ints in range {spec!r}")
    if stop < start:
        raise ValueError(f"empty-inverted range [{start}, {stop})")
    return start, stop


@dataclass(frozen=True)
class Dim:
    name: str
    stride: int  # bytes; may be zero (repeat) or negative (flip)
    start: int  # inclusive
    stop: int  # exclusive
    chart: Chart | None = None  # optional exact physical labeling of the lattice
    labels: tuple[str, ...] | None = None  # optional categorical labeling (D18)

    def __post_init__(self) -> None:
        if self.stop < self.start:
            raise ValueError(f"dim {self.name}: stop {self.stop} < start {self.start}")
        if self.labels is not None:
            if self.chart is not None:
                raise ValueError(f"dim {self.name}: a dim carries a chart or labels, not both")
            if len(self.labels) != self.size:
                raise ValueError(f"dim {self.name}: {len(self.labels)} labels for {self.size} lattice points")
            if len(set(self.labels)) != len(self.labels):
                raise ValueError(f"dim {self.name}: labels must be unique")

    @property
    def size(self) -> int:
        return self.stop - self.start

    def contains(self, i: int) -> bool:
        return self.start <= i < self.stop

    def to_lattice(self, coord) -> int:
        """A coordinate -> lattice int. Quantities need a chart; strings need
        labels; integral values always mean the lattice."""
        if isinstance(coord, Quantity):
            if self.chart is None:
                raise TypeError(f"dim {self.name} has no chart; pass a lattice int, not {coord!r}")
            return self.chart.lattice(coord)
        if isinstance(coord, str):
            if self.labels is None:
                raise TypeError(f"dim {self.name} has no labels; got {coord!r}")
            try:
                return self.start + self.labels.index(coord)
            except ValueError:
                raise KeyError(f"{coord!r} is not a label of dim {self.name}; have {self.labels}") from None
        i = _lattice_int(coord)
        if i is None:
            raise TypeError(f"coordinate for {self.name} must be int, Quantity, or label, got {coord!r}")
        return i

    def delta_to_lattice(self, delta) -> int:
        """A displacement -> whole number of lattice steps (exact-only)."""
        if isinstance(delta, Quantity):
            if self.chart is None:
                raise TypeError(f"dim {self.name} has no chart; pass a lattice int delta")
            r = delta / self.chart.step
            if isinstance(r, Quantity):
                raise ValueError(f"delta {delta!r} has wrong dimensions for {self.name}'s chart")
            if r.denominator != 1:
                raise ValueError(
                    f"delta {delta!r} is not a whole number of steps ({self.chart.step!r}) for dim {self.name}"
                )
            return int(r)
        i = _lattice_int(delta)
        if i is None:
            raise TypeError(f"delta for {self.name} must be int or Quantity, got {delta!r}")
        return i

    def phys(self, i: int) -> Quantity:
        if self.chart is None:
            raise TypeError(f"dim {self.name} has no chart")
        return self.chart.phys(i)

    def label(self, i: int) -> str:
        if self.labels is None:
            raise TypeError(f"dim {self.name} has no labels")
        if not self.contains(i):
            raise IndexError(f"{self.name}={i} outside [{self.start}, {self.stop})")
        return self.labels[i - self.start]

    def __repr__(self) -> str:
        base = f"{self.name}[{self.start}:{self.stop})*{self.stride}"
        if self.chart is not None:
            return f"{base} @{self.chart}"
        if self.labels is not None:
            return f"{base} #[{','.join(self.labels)}]"
        return base


def _coord_range(d: Dim, spec) -> tuple[int, int]:
    """A slice range for one dim: int n -> (0, n); pair bounds may be
    lattice ints, on-lattice Quantities (positions), or labels."""
    if isinstance(spec, tuple):
        lo_raw, hi_raw = spec
        lo, hi = d.to_lattice(lo_raw), d.to_lattice(hi_raw)
        if hi < lo:
            if not (isinstance(lo_raw, int) and isinstance(hi_raw, int)):
                raise ValueError(
                    f"slice on {d.name} inverted at the lattice (negative-step chart?); slice in lattice coordinates"
                )
            raise ValueError(f"empty-inverted range [{lo}, {hi})")
        return lo, hi
    n = _lattice_int(spec)
    if n is not None:
        return 0, n
    raise TypeError(f"bad range spec for {d.name}: {spec!r}")


def _tap(d: Dim, x, dilation: int) -> int:
    """A kernel tap spec -> tap index. Ints are tap indices; Quantities are
    physical displacements, which must be multiples of step*dilation."""
    if isinstance(x, Quantity):
        lat = d.delta_to_lattice(x)
        if lat % dilation:
            raise ValueError(
                f"displacement {x!r} is not a multiple of the dilated step ({dilation} * {d.chart.step!r})"
            )
        return lat // dilation
    i = _lattice_int(x)
    if i is None:
        raise TypeError(f"tap for {d.name} must be int or Quantity, got {x!r}")
    return i


def _tap_range(d: Dim, spec, dilation: int) -> tuple[int, int]:
    """Half-open tap-index range for window kernels."""
    if isinstance(spec, tuple):
        lo, hi = (_tap(d, x, dilation) for x in spec)
        if hi < lo:
            raise ValueError(f"empty-inverted range [{lo}, {hi})")
        return lo, hi
    n = _lattice_int(spec)
    if n is not None:
        return 0, n
    raise TypeError(f"bad tap range spec for {d.name}: {spec!r}")


def _axis_of(d: Dim) -> str:
    """The physical axis a charted dim labels (defaults to its own name)."""
    return d.chart.axis if d.chart.axis is not None else d.name


def _no_labels(d: Dim, op: str) -> None:
    if d.labels is not None:
        raise TypeError(
            f"dim {d.name} is categorical (labels {d.labels}); {op} has no "
            f"meaning on nominal data — attach a chart first if a numeric "
            f"scale exists"
        )


class Injectivity(enum.Enum):
    INJECTIVE = "injective"  # provably one-to-one
    ALIASED = "aliased"  # provably not (e.g. stride 0 with size > 1)
    UNKNOWN = "unknown"  # the sufficient condition failed; no proof either way


@dataclass(frozen=True)
class Layout:
    dims: tuple[Dim, ...]
    offset: int = 0  # bytes

    def __post_init__(self) -> None:
        names = [d.name for d in self.dims]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate dimension names: {names}")
        axis_dims: dict[str, tuple] = {}
        for d in self.dims:
            ch = d.chart
            if ch is not None and ch.axis is not None:
                prev = axis_dims.setdefault(ch.axis, ch.step.dims)
                if prev != ch.step.dims:
                    raise ValueError(
                        f"axis {ch.axis!r}: siblings label different physical "
                        f"dimensions ({dict(prev)} vs {dict(ch.step.dims)})"
                    )
        seen_pos: set[str] = set()
        for d in self.dims:
            ch = d.chart
            if ch is not None and ch.axis is not None and ch.kind == "position":
                if ch.axis in seen_pos:
                    raise ValueError(
                        f"axis {ch.axis!r} has more than one position dim; siblings on one axis must be displacements"
                    )
                seen_pos.add(ch.axis)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.dims)

    def dim(self, name: str) -> Dim:
        for d in self.dims:
            if d.name == name:
                return d
        raise KeyError(f"no dimension named {name!r}; have {self.names}")

    def sizes(self) -> dict[str, int]:
        return {d.name: d.size for d in self.dims}

    @property
    def numel(self) -> int:
        return math.prod(d.size for d in self.dims)

    def to_lattice(self, coords: Mapping) -> dict[str, int]:
        """Normalize named coordinates (ints, Quantities, labels) to lattice ints."""
        return {name: self.dim(name).to_lattice(v) for name, v in coords.items()}

    def to_lattice_deltas(self, deltas: Mapping) -> dict[str, int]:
        return {name: self.dim(name).delta_to_lattice(v) for name, v in deltas.items()}

    def get_loc(self, **coords) -> int:
        """The contract: named raw coordinates -> byte location. Coordinates
        may be lattice ints, on-lattice Quantities, or labels."""
        if set(coords) != set(self.names):
            raise KeyError(f"coordinates {sorted(coords)} do not match dims {sorted(self.names)}")
        loc = self.offset
        for d in self.dims:
            i = d.to_lattice(coords[d.name])
            if not d.contains(i):
                raise IndexError(f"coordinate {d.name}={i} outside domain [{d.start}, {d.stop})")
            loc += d.stride * i
        return loc

    def resolve(self, **coords) -> int | None:
        """Uniform contract shared with guarded layouts: a core layout always
        resolves to a real location (never a fill)."""
        return self.get_loc(**coords)

    def footprint(self, itemsize: int = 1) -> tuple[int, int] | None:
        """Half-open byte interval [lo, hi) the layout can touch.

        None if the domain is empty."""
        if self.numel == 0:
            return None
        lo = hi = self.offset
        for d in self.dims:
            a, b = d.stride * d.start, d.stride * (d.stop - 1)
            lo += min(a, b)
            hi += max(a, b)
        return lo, hi + itemsize

    def injectivity(self) -> Injectivity:
        """Conservative one-to-one check on element start addresses.

        Uses the classic sufficient condition: with dims (size > 1) sorted by
        |stride|, each |stride| must exceed the total span of the smaller
        ones. INJECTIVE and ALIASED are proofs; UNKNOWN means the sufficient
        condition failed for a layout that is not obviously aliased."""
        active = [d for d in self.dims if d.size > 1]
        if any(d.stride == 0 for d in active):
            return Injectivity.ALIASED
        active.sort(key=lambda d: abs(d.stride))
        span = 0
        for d in active:
            if abs(d.stride) <= span:
                return Injectivity.UNKNOWN
            span += abs(d.stride) * (d.size - 1)
        return Injectivity.INJECTIVE

    def is_contiguous(self, itemsize: int) -> bool:
        """True iff the layout's image packs one dense block, each element
        visited once: dims (size > 1) sorted by stride must have positive
        strides that nest exactly (s_0 == itemsize, s_k == s_{k-1}*n_{k-1}).
        Trivially true when no dim has size > 1."""
        active = sorted((d for d in self.dims if d.size > 1), key=lambda d: d.stride)
        expect = itemsize
        for d in active:
            if d.stride != expect:
                return False
            expect *= d.size
        return True

    def domain(self) -> Iterator[dict[str, int]]:
        """Iterate all lattice coordinate tuples, first-listed dim slowest."""
        ranges = [range(d.start, d.stop) for d in self.dims]
        for tup in itertools.product(*ranges):
            yield dict(zip(self.names, tup))

    def with_offset_delta(self, delta: int) -> "Layout":
        return replace(self, offset=self.offset + delta)

    def phys(self, name: str, i: int) -> Quantity:
        """Physical label of lattice coordinate i on a charted dim."""
        return self.dim(name).phys(i)

    def snap(self, name: str, value, mode: str = "nearest") -> int:
        """Deliberately round a physical value onto a charted dim's lattice."""
        d = self.dim(name)
        if d.chart is None:
            raise TypeError(f"dim {name} has no chart")
        return d.chart.snap(value, mode)

    def canonical(self) -> "Layout":
        """Order-free identity: dims sorted by name. Dim order carries no
        addressing semantics (D5), so two layouts that differ only in
        presentation order have equal canonical forms — names are semantic,
        order is not. (There is deliberately no permute op: export order is a
        parameter of materialization, not a property of the layout.)"""
        return replace(self, dims=tuple(sorted(self.dims, key=lambda d: d.name)))

    def __repr__(self) -> str:
        inner = ", ".join(repr(d) for d in self.dims)
        return f"Layout(offset={self.offset}, {inner})"

    # ------------------------------------------------------------------
    # chart & label management — lattice and addressing untouched
    # ------------------------------------------------------------------

    def with_charts(self, **charts) -> "Layout":
        """Attach/replace charts: a Chart, an (origin, step[, kind[, axis]])
        tuple, or None to clear. A chart without an axis is tagged with the
        dim's own name. Attaching a chart to a labeled dim REPLACES the
        labels (the nominal -> interval upgrade). Pure metadata; addresses
        never change."""
        new = []
        for d in self.dims:
            if d.name in charts:
                c = charts.pop(d.name)
                if isinstance(c, tuple):
                    c = make_chart(*c)
                if not (c is None or isinstance(c, Chart)):
                    raise TypeError(f"chart for {d.name}: expected Chart/tuple/None, got {c!r}")
                if c is not None and c.axis is None:
                    c = replace(c, axis=d.name)
                d = replace(d, chart=c, labels=None)
            new.append(d)
        if charts:
            raise KeyError(f"unknown dims in with_charts: {sorted(charts)}")
        return replace(self, dims=tuple(new))

    def with_labels(self, **labels) -> "Layout":
        """Attach/replace categorical labels (one name per lattice point;
        None clears). Replaces any chart on the dim: nominal and interval
        labelings are mutually exclusive."""
        new = []
        for d in self.dims:
            if d.name in labels:
                lb = labels.pop(d.name)
                if lb is not None:
                    lb = tuple(lb)
                d = replace(d, labels=lb, chart=None)
            new.append(d)
        if labels:
            raise KeyError(f"unknown dims in with_labels: {sorted(labels)}")
        return replace(self, dims=tuple(new))

    def strip_charts(self) -> "Layout":
        """Compiler mode: drop all physical labeling (charts AND labels),
        keep the pure lattice."""
        return replace(self, dims=tuple(replace(d, chart=None, labels=None) for d in self.dims))

    def recenter(self, **deltas) -> "Layout":
        """Move the physical frame: origin += delta. The lattice, the
        addresses, and the data are untouched — only the labels move.
        (Compare shift, which relabels the lattice and keeps physical labels
        glued to the data.)"""
        new = []
        for d in self.dims:
            if d.name in deltas:
                if d.chart is None:
                    raise TypeError(f"dim {d.name} has no chart to recenter")
                delta = deltas.pop(d.name)
                if not isinstance(delta, Quantity):
                    raise TypeError(f"recenter delta for {d.name} must be a Quantity")
                d = replace(d, chart=replace(d.chart, origin=d.chart.origin + delta))
            new.append(d)
        if deltas:
            raise KeyError(f"unknown dims in recenter: {sorted(deltas)}")
        return replace(self, dims=tuple(new))

    # ------------------------------------------------------------------
    # view operations — each returns a new Layout over the same buffer
    # ------------------------------------------------------------------

    def slice(self, **ranges) -> "Layout":
        """Shrink domains. Raw coordinates keep their identity: strides,
        offset, and charts are untouched; only [start, stop) changes. Bounds
        may be on-lattice Quantities (or labels) on labeled dims. An empty
        range is a subset of any domain and is allowed anywhere (align
        recipes rely on this for disjoint operands). Categorical labels are
        subset along with the domain."""
        new = []
        for d in self.dims:
            if d.name in ranges:
                start, stop = _coord_range(d, ranges.pop(d.name))
                if start != stop and (start < d.start or stop > d.stop):
                    raise IndexError(f"slice [{start}, {stop}) not within {d.name} domain [{d.start}, {d.stop})")
                lb = d.labels
                if lb is not None:
                    lb = lb[start - d.start : max(start, stop) - d.start]
                d = replace(d, start=start, stop=stop, labels=lb)
            new.append(d)
        if ranges:
            raise KeyError(f"unknown dims in slice: {sorted(ranges)}")
        return replace(self, dims=tuple(new))

    def select(self, **coords) -> "Layout":
        """Fix dims to points: drop them, folding stride*i into the offset.
        Coordinates may be labels on labeled dims.

        Axis bookkeeping: a selected charted dim's physical label is folded
        into a remaining sibling on the same axis (the position dim if
        present, else the widest-step displacement, which is promoted to a
        position). This keeps 'sum of labels per axis == physical position'
        true — e.g. selecting a phase after a split yields correctly-labeled
        decimation, and selecting a block yields absolute in-block positions.
        If no sibling remains, the axis is fully collapsed and the
        contribution is dropped. (Categorical labels have no sum; nothing to
        fold.)"""
        delta = 0
        removed: dict[str, Quantity] = {}
        keep = []
        for d in self.dims:
            if d.name in coords:
                i = d.to_lattice(coords.pop(d.name))
                if not d.contains(i):
                    raise IndexError(f"select {d.name}={i} outside [{d.start}, {d.stop})")
                delta += d.stride * i
                if d.chart is not None:
                    axis = _axis_of(d)
                    lbl = d.chart.phys(i)
                    removed[axis] = removed[axis] + lbl if axis in removed else lbl
            else:
                keep.append(d)
        if coords:
            raise KeyError(f"unknown dims in select: {sorted(coords)}")
        if removed:
            keep = _compensate_axes(keep, removed)
        return Layout(dims=tuple(keep), offset=self.offset + delta)

    def shift(self, **deltas) -> "Layout":
        """Relabel the lattice: translate domains, compensating the offset so
        the same memory is addressed. A chart's origin compensates too, so
        physical labels stay glued to the data — shift is a storage-side
        relabeling, not a physics change. Deltas may be step-multiple
        Quantities on charted dims. Categorical labels move with the domain
        and stay glued automatically."""
        delta_off = 0
        new = []
        for d in self.dims:
            if d.name in deltas:
                s = d.delta_to_lattice(deltas.pop(d.name))
                delta_off -= d.stride * s
                ch = d.chart
                if ch is not None:
                    ch = replace(ch, origin=ch.origin - s * ch.step)
                d = replace(d, start=d.start + s, stop=d.stop + s, chart=ch)
            new.append(d)
        if deltas:
            raise KeyError(f"unknown dims in shift: {sorted(deltas)}")
        return Layout(dims=tuple(new), offset=self.offset + delta_off)

    def rename(self, **mapping: str) -> "Layout":
        """Rename dims. Axis tags are free-standing labels shared between
        siblings, so they do NOT follow a dim rename."""
        unknown = set(mapping) - set(self.names)
        if unknown:
            raise KeyError(f"unknown dims in rename: {sorted(unknown)}")
        new = tuple(replace(d, name=mapping.get(d.name, d.name)) for d in self.dims)
        return replace(self, dims=new)  # __post_init__ re-checks uniqueness

    def repeat(
        self,
        name: str,
        extent,
        chart: Chart | None = None,
        labels: tuple[str, ...] | None = None,
    ) -> "Layout":
        """Add a stride-0 dimension: every coordinate aliases the same data.
        May carry a chart or categorical labels (e.g. an RGB channel dim)."""
        if name in self.names:
            raise ValueError(f"dim {name!r} already exists")
        start, stop = as_range(extent)
        if labels is not None:
            labels = tuple(labels)
        return replace(
            self,
            dims=self.dims + (Dim(name, stride=0, start=start, stop=stop, chart=chart, labels=labels),),
        )

    def flip(self, name: str) -> "Layout":
        """Reverse a dim about its own domain: i <-> (start + stop - 1) - i.
        Stride negates; the domain is unchanged; the offset compensates. A
        chart follows the data (origin re-anchors, step negates): flip is a
        storage reversal, and the physics of each datum is invariant.
        Categorical labels reverse for the same reason."""
        d = self.dim(name)
        ch = d.chart
        if ch is not None:
            ch = replace(
                ch,
                origin=ch.origin + (d.start + d.stop - 1) * ch.step,
                step=-ch.step,
            )
        lb = d.labels[::-1] if d.labels is not None else None
        new = tuple(replace(x, stride=-x.stride, chart=ch, labels=lb) if x.name == name else x for x in self.dims)
        return Layout(dims=new, offset=self.offset + d.stride * (d.start + d.stop - 1))

    def split(self, name: str, **parts) -> "Layout":
        """Blocking: replace dim x by parts (kwargs order = outer..inner) via
        the affine bijection

            x = const + sum(w_j * i_j),   w_j = prod(sizes of inner parts)

        Each part is an int size or an explicit lattice (start, stop) — the
        caller controls the new index ranges so coordinates stay natural. The
        constant folds into the offset (scaled by x's stride). If x is
        charted, all parts share x's axis: the outermost inherits a position
        chart (step scaled by its weight — the block pitch) and inner parts
        get displacement charts — physically, block position +
        offset-within-block."""
        d = self.dim(name)
        _no_labels(d, "split")
        norm = tuple((n, as_range(spec)) for n, spec in parts.items())
        new_dims, _, const = _split_dim(d, norm)
        out = []
        for x in self.dims:
            if x.name == name:
                out.extend(new_dims)
            else:
                if x.name in parts:
                    raise ValueError(f"part name {x.name!r} collides with existing dim")
                out.append(x)
        return Layout(dims=tuple(out), offset=self.offset + d.stride * const)

    def merge(self, parts: tuple[str, ...], name: str, start: int = 0) -> "Layout":
        """Inverse of split, for compatibly nested dims (outer..inner order):
        requires stride_outer == stride_inner * size_inner along the chain.
        Charts must be all present (sharing one axis, with steps nesting the
        same way — the merged labeling must stay affine) or all absent."""
        ds = [self.dim(n) for n in parts]
        for d in ds:
            _no_labels(d, "merge")
        for outer, inner in zip(ds, ds[1:]):
            if outer.stride != inner.stride * inner.size:
                raise ValueError(
                    f"cannot merge: {outer.name}.stride ({outer.stride}) != "
                    f"{inner.name}.stride*size ({inner.stride}*{inner.size})"
                )
        stride = ds[-1].stride
        n = math.prod(d.size for d in ds)
        mchart = _merge_charts(ds, start)
        merged = Dim(name, stride=stride, start=start, stop=start + n, chart=mchart)
        delta = sum(d.stride * d.start for d in ds) - stride * start
        out, placed = [], False
        for x in self.dims:
            if x.name in parts:
                if not placed:
                    out.append(merged)
                    placed = True
            else:
                out.append(x)
        return Layout(dims=tuple(out), offset=self.offset + delta)

    def diagonal(self, parts: tuple[str, ...], name: str, chart=None) -> "Layout":
        """(x_1, ..., x_n) -> z walking all parts at once (n >= 2):
        stride_z = sum of strides, domain = intersection of the boxes. Raw
        coords make this offset-free.

        Charts (D16): the library never guesses a labeling. If all parts
        share ONE axis, z gets the forced label-sum chart (origins and steps
        add; a zero step-sum leaves z uncharted). Otherwise z is UNCHARTED
        unless the caller supplies `chart` — a Chart, or a combinator
        callable (consumed_dims) -> Chart | None which receives the full Dim
        objects (charts, domains, names) of the parts it consumes. See
        chart.characteristic for the rate-diagonal combinator."""
        if len(parts) < 2:
            raise ValueError("diagonal needs at least two dims")
        ds = [self.dim(n) for n in parts]
        for d in ds:
            _no_labels(d, "diagonal")
        start = max(d.start for d in ds)
        stop = max(start, min(d.stop for d in ds))
        ch = _diagonal_chart(ds, chart)
        diag = Dim(name, stride=sum(d.stride for d in ds), start=start, stop=stop, chart=ch)
        out, placed = [], False
        for x in self.dims:
            if x.name in parts:
                if not placed:
                    out.append(diag)
                    placed = True
            else:
                out.append(x)
        return replace(self, dims=tuple(out))

    def window(self, name: str, k_name: str, k, dilation: int = 1) -> "Layout":
        """Sliding window (the stencil *interior*): add k with stride
        stride_x * dilation and shrink x so that x + dilation*k always lands
        in x's old domain. No guard, no fill — that is step 2's `stencil`.

        k is a half-open TAP-INDEX range (dilation taps sit `dilation` lattice
        steps apart); bounds may be physical displacements on a charted dim.
        The kernel gets a displacement chart on x's axis with step
        step_x * dilation, so taps carry physical labels."""
        if k_name in self.names:
            raise ValueError(f"dim {k_name!r} already exists")
        if not isinstance(dilation, int) or dilation < 1:
            raise ValueError(f"dilation must be a positive int, got {dilation!r}")
        d = self.dim(name)
        _no_labels(d, "window")
        k_start, k_stop = _tap_range(d, k, dilation)
        if k_stop <= k_start:
            raise ValueError("window needs a non-empty k range")
        new_start = d.start - dilation * k_start
        new_stop = d.stop - dilation * (k_stop - 1)
        if new_stop < new_start:
            raise ValueError(f"window [{k_start}, {k_stop}) wider than {name}'s domain")
        k_chart = None
        if d.chart is not None:
            k_chart = Chart(0 * d.chart.step, dilation * d.chart.step, "displacement", _axis_of(d))
        new = []
        for x in self.dims:
            if x.name == name:
                new.append(replace(x, start=new_start, stop=new_stop))
                new.append(
                    Dim(
                        k_name,
                        stride=d.stride * dilation,
                        start=k_start,
                        stop=k_stop,
                        chart=k_chart,
                    )
                )
            else:
                new.append(x)
        return replace(self, dims=tuple(new))

    def decimate(self, name: str, factor: int, phase=0) -> "Layout":
        """Keep every factor-th element (lattice congruence i = phase mod
        factor) as a view: the stride scales by factor and the lattice is
        renumbered (j = (i - phase)/factor — the one op besides shift that
        relabels). A chart keeps the physical truth: origin gains phase*step
        and the step scales by factor, so labels stay glued to the data.
        Categorical labels are subsampled the same way. Phase may be a
        step-multiple Quantity displacement."""
        if not isinstance(factor, int) or factor < 1:
            raise ValueError(f"factor must be a positive int, got {factor!r}")
        d = self.dim(name)
        p = d.delta_to_lattice(phase) % factor
        j0 = -((-(d.start - p)) // factor)  # ceil((start - p) / factor)
        j1 = (d.stop - 1 - p) // factor  # floor((stop - 1 - p) / factor)
        start, stop = j0, max(j0, j1 + 1)
        ch = d.chart
        if ch is not None:
            ch = replace(ch, origin=ch.origin + p * ch.step, step=factor * ch.step)
        lb = d.labels
        if lb is not None:
            lb = tuple(lb[(factor * j + p) - d.start] for j in range(start, stop))
        new = tuple(
            replace(x, stride=d.stride * factor, start=start, stop=stop, chart=ch, labels=lb) if x.name == name else x
            for x in self.dims
        )
        return Layout(dims=new, offset=self.offset + d.stride * p)


def _compensate_axes(dims: list[Dim], removed: Mapping[str, Quantity]) -> list[Dim]:
    """Fold selected dims' physical labels into a remaining sibling on the
    same axis, so 'sum of labels per axis == physical position' stays true.

    Target: the axis's position dim if it remains; otherwise the
    widest-step displacement (promoted to a position) — an order-free
    choice, per D5. No sibling: the axis is fully collapsed; drop."""
    out = list(dims)
    for axis, lbl in removed.items():
        target = None
        best_step = None
        for idx, d in enumerate(out):
            ch = d.chart
            if ch is None or _axis_of(d) != axis:
                continue
            if ch.kind == "position":
                target = idx
                break
            key = (abs(ch.step.base), d.name)
            if best_step is None or key > best_step:
                best_step = key
                target = idx
        if target is None:
            continue
        d = out[target]
        out[target] = replace(d, chart=replace(d.chart, origin=d.chart.origin + lbl, kind="position"))
    return out


def _split_dim(d: Dim, parts: tuple[tuple[str, tuple[int, int]], ...]) -> tuple[tuple[Dim, ...], tuple[int, ...], int]:
    """Shared split machinery (also used to rewrite guards in step 2).

    Returns (new_dims, weights, const) with x = const + sum(w_j * i_j)."""
    if not parts:
        raise ValueError("split needs at least one part")
    sizes = [stop - start for _, (start, stop) in parts]
    if math.prod(sizes) != d.size:
        raise ValueError(f"split sizes {sizes} (product {math.prod(sizes)}) != size of {d.name} ({d.size})")
    weights = []
    for j in range(len(parts)):
        weights.append(math.prod(sizes[j + 1 :]))
    const = d.start - sum(w * start for (_, (start, _)), w in zip(parts, weights))
    charts: list[Chart | None] = [None] * len(parts)
    if d.chart is not None:
        c = d.chart
        axis = _axis_of(d)
        for j, w in enumerate(weights):
            if j == 0:
                charts[j] = Chart(c.origin + const * c.step, w * c.step, c.kind, axis)
            else:
                charts[j] = Chart(0 * c.step, w * c.step, "displacement", axis)
    new_dims = tuple(
        Dim(n, stride=d.stride * w, start=start, stop=stop, chart=ch)
        for (n, (start, stop)), w, ch in zip(parts, weights, charts)
    )
    return new_dims, tuple(weights), const


def _merge_charts(ds: list[Dim], start: int) -> Chart | None:
    charts = [d.chart for d in ds]
    if all(c is None for c in charts):
        return None
    if any(c is None for c in charts):
        raise ValueError("merge: parts must be all charted or all uncharted")
    axes = {_axis_of(d) for d in ds}
    if len(axes) != 1:
        raise ValueError(f"merge: parts label different axes {sorted(axes)}")
    step_m = charts[-1].step
    w = 1
    for d, c in zip(reversed(ds), reversed(charts)):
        if c.step != w * step_m:
            raise ValueError(
                f"merge: {d.name}'s chart step {c.step!r} != {w} * {step_m!r}; the merged labeling would not be affine"
            )
        w *= d.size
    p0 = charts[0].phys(ds[0].start)
    for d, c in zip(ds[1:], charts[1:]):
        p0 = p0 + c.phys(d.start)
    kind = "position" if any(c.kind == "position" for c in charts) else "displacement"
    return Chart(p0 - start * step_m, step_m, kind, axes.pop())


def _diagonal_chart(ds: list[Dim], spec) -> Chart | None:
    """Resolve diagonal's chart parameter (D16): explicit Chart, a combinator
    (consumed_dims) -> Chart | None, or None — which forces the label-sum
    chart when all parts share one axis and leaves z uncharted otherwise."""
    if isinstance(spec, Chart):
        return spec
    if callable(spec):
        out = spec(tuple(ds))
        if not (out is None or isinstance(out, Chart)):
            raise TypeError(f"diagonal chart combinator returned {out!r}, not Chart/None")
        return out
    if spec is not None:
        raise TypeError(f"diagonal chart must be Chart, callable, or None, got {spec!r}")
    charts = [d.chart for d in ds]
    if all(c is not None for c in charts) and len({_axis_of(d) for d in ds}) == 1:
        step = charts[0].step
        origin = charts[0].origin
        for c in charts[1:]:
            step = step + c.step
            origin = origin + c.origin
        if not step:
            return None  # e.g. a dim and its mirror: constant position
        kind = "position" if any(c.kind == "position" for c in charts) else "displacement"
        return Chart(origin, step, kind, _axis_of(ds[0]))
    return None
