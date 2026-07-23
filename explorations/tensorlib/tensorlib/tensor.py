"""Tensor = Buffer + Layout + DType (+ fill from step 2, + units from step 3).

Every operation returns a new Tensor sharing the same buffer with a
different layout — nothing here moves data. The read path (`item`,
`to_numpy`) exists to *exercise* the layout math against ground truth; it is
a testing convenience, not a compute layer.

Step 3 adds two kinds of physical labeling, both strictly metadata over the
unchanged lattice machinery:

- coordinate charts live on dims (see layout.py); Tensor just forwards
  Quantities to the layout and exposes with_charts / strip_charts / recenter
  / snap / phys;
- `value_units` labels the VALUE space — a Unit for a scalar dtype, or a
  mapping field -> Unit for structured dtypes, threaded by `field()`. Values
  themselves are inexact machine numbers, so this is type-level metadata for
  the future compute layer (dimensional checking), not exact arithmetic:
  `item()` returns raw scalars.

`alignment(*tensors)` is the elementwise-compute gatekeeper — a DIAGNOSIS of
what stops operands from being combinable, with the exact primitive that
fixes each item. The library never applies the fixes: aligning is the
caller's conscious act (D17), spent one cheap view op at a time.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from .buffer import Buffer, host_view
from .dtypes import CARRIERS, carrier_of
from .guarded import GuardedLayout, pad_layout, stencil_layout
from .layout import Dim, Injectivity, Layout, RangeSpec, as_range
from .units import Quantity


@dataclass(frozen=True)
class Tensor:
    buffer: Buffer
    dtype: np.dtype
    layout: Layout | GuardedLayout
    fill: object = None  # step 2: value produced where guards say "no location"
    value_units: object = None  # step 3: Unit, or {field: Unit} for structured dtypes
    carrier: str | None = None  # the algebraic object values approximate (dtypes.CARRIERS)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if self.carrier is None:
            object.__setattr__(self, "carrier", carrier_of(self.dtype))
        elif self.carrier not in CARRIERS:
            raise ValueError(f"carrier must be one of {CARRIERS}, got {self.carrier!r}")

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_numpy(cls, arr, names: tuple[str, ...]) -> "Tensor":
        """Wrap a numpy array. C-contiguous input shares memory (mutations to
        the source stay visible through the view); otherwise a contiguous
        copy is taken. numpy's byte strides are already 'fully expanded'."""
        arr = np.asarray(arr)
        if arr.ndim != len(names):
            raise ValueError(f"{len(names)} names for {arr.ndim}-d array")
        carr = np.ascontiguousarray(arr)
        buf = Buffer(nbytes=carr.nbytes, data=host_view(carr))
        dims = tuple(Dim(n, stride=s, start=0, stop=e) for n, s, e in zip(names, carr.strides, carr.shape))
        return cls(buf, carr.dtype, Layout(dims)).check()

    @classmethod
    def dense(cls, dtype, device: str = "cpu", **extents: RangeSpec) -> "Tensor":
        """Allocate a compact tensor. Per the fully-expanded-strides
        convention the FIRST listed dim is fastest (stride = one element);
        each later dim's stride is the product of the earlier extents. The
        offset compensates non-zero starts so the min corner sits at byte 0."""
        dtype = np.dtype(dtype)
        dims = []
        stride = dtype.itemsize
        for name, spec in extents.items():
            start, stop = as_range(spec)
            dims.append(Dim(name, stride=stride, start=start, stop=stop))
            stride *= stop - start
        offset = -sum(d.stride * d.start for d in dims)
        layout = Layout(tuple(dims), offset=offset)
        return cls(Buffer.allocate(layout.numel * dtype.itemsize, device), dtype, layout)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    @property
    def itemsize(self) -> int:
        return self.dtype.itemsize

    @property
    def names(self) -> tuple[str, ...]:
        return self.layout.names

    def sizes(self) -> dict[str, int]:
        return self.layout.sizes()

    @property
    def numel(self) -> int:
        return self.layout.numel

    @property
    def device(self) -> str:
        return self.buffer.device

    def strides_in_elements(self) -> dict[str, int]:
        """The element-unit view of the byte strides (D4)."""
        out = {}
        for d in self.layout.dims:
            if d.stride % self.itemsize != 0:
                raise ValueError(f"dim {d.name}: byte stride {d.stride} is not a multiple of itemsize {self.itemsize}")
            out[d.name] = d.stride // self.itemsize
        return out

    def injectivity(self) -> Injectivity:
        return self.layout.injectivity()

    def is_contiguous(self) -> bool:
        return self.layout.is_contiguous(self.itemsize)

    def footprint(self) -> tuple[int, int] | None:
        return self.layout.footprint(self.itemsize)

    def check(self) -> "Tensor":
        """Validate that every real location lands inside the buffer."""
        fp = self.footprint()
        if fp is not None:
            lo, hi = fp
            if lo < 0 or hi > self.buffer.nbytes:
                raise ValueError(f"layout footprint [{lo}, {hi}) exceeds buffer [0, {self.buffer.nbytes})")
        return self

    def overlaps(self, other: "Tensor") -> bool:
        """Conservative may-alias: same buffer and intersecting footprints."""
        if self.buffer is not other.buffer:
            return False
        a, b = self.footprint(), other.footprint()
        if a is None or b is None:
            return False
        return a[0] < b[1] and b[0] < a[1]

    def phys(self, name: str, i: int) -> Quantity:
        """Physical label of lattice coordinate i on a charted dim."""
        return self.layout.phys(name, i)

    def snap(self, name: str, value, mode: str = "nearest") -> int:
        """Deliberately round a physical value onto a charted dim's lattice."""
        return self.layout.snap(name, value, mode)

    # ------------------------------------------------------------------
    # the read path (exercises the contract; not a compute layer)
    # ------------------------------------------------------------------

    def item(self, **coords):
        loc = self.layout.resolve(**coords)
        if loc is None:
            if self.fill is None:
                raise RuntimeError("coordinate resolves to fill, but no fill is set")
            return self.fill
        return self.buffer.read(loc, self.dtype)

    def to_numpy(self, order: tuple[str, ...] | None = None) -> np.ndarray:
        """Materialize (naively) for testing, coordinates re-based to 0.

        Axis order defaults to the stored (presentation) dim order; pass
        `order` to choose the export order explicitly — order is a property
        of the export, not of the tensor."""
        dims = self.layout.dims
        if order is not None:
            if sorted(order) != sorted(self.names):
                raise KeyError(f"order {order} must be a permutation of {self.names}")
            dims = tuple(self.layout.dim(n) for n in order)
        names = [d.name for d in dims]
        out = np.zeros([d.size for d in dims], dtype=self.dtype)
        for tup in itertools.product(*(range(d.start, d.stop) for d in dims)):
            idx = tuple(i - d.start for i, d in zip(tup, dims))
            out[idx] = self.item(**dict(zip(names, tup)))
        return out

    # ------------------------------------------------------------------
    # view operations (all: same buffer, new layout)
    # ------------------------------------------------------------------

    def _via(self, layout: Layout | GuardedLayout) -> "Tensor":
        return replace(self, layout=layout)

    def slice(self, **ranges) -> "Tensor":
        return self._via(self.layout.slice(**ranges))

    def select(self, **coords) -> "Tensor":
        return self._via(self.layout.select(**coords))

    def shift(self, **deltas) -> "Tensor":
        return self._via(self.layout.shift(**deltas))

    def rename(self, **mapping: str) -> "Tensor":
        return self._via(self.layout.rename(**mapping))

    def canonical(self) -> "Tensor":
        return self._via(self.layout.canonical())

    def repeat(self, name: str, extent: RangeSpec, chart=None, labels=None) -> "Tensor":
        return self._via(self.layout.repeat(name, extent, chart, labels))

    def flip(self, name: str) -> "Tensor":
        return self._via(self.layout.flip(name))

    def split(self, name: str, **parts: RangeSpec) -> "Tensor":
        return self._via(self.layout.split(name, **parts))

    def merge(self, parts: tuple[str, ...], name: str, start: int = 0) -> "Tensor":
        return self._via(self.layout.merge(parts, name, start=start))

    def diagonal(self, parts: tuple[str, ...], name: str, chart=None) -> "Tensor":
        """N-ary diagonal (see Layout.diagonal). `chart` is a Chart, a
        combinator (consumed_dims) -> Chart | None, or None — same-axis
        parts get the forced label-sum chart, anything else is uncharted
        unless you say otherwise (D16: the library never guesses)."""
        return self._via(self.layout.diagonal(parts, name, chart))

    def window(self, name: str, k_name: str, k, dilation: int = 1) -> "Tensor":
        return self._via(self.layout.window(name, k_name, k, dilation))

    def decimate(self, name: str, factor: int, phase=0) -> "Tensor":
        """Keep every factor-th element as a view (see Layout.decimate);
        the chart keeps physical labels glued to the kept data."""
        return self._via(self.layout.decimate(name, factor, phase))

    def with_charts(self, **charts) -> "Tensor":
        """Attach physical labels: Chart, (origin, step[, kind[, axis]])
        tuple, or None to clear. Addresses never change."""
        return self._via(self.layout.with_charts(**charts))

    def with_labels(self, **labels) -> "Tensor":
        """Attach categorical labels (the nominal rung, D18): one name per
        lattice point, e.g. with_labels(c=("R", "G", "B")). Replaces any
        chart on the dim; attaching a chart later replaces the labels — the
        nominal -> interval upgrade."""
        return self._via(self.layout.with_labels(**labels))

    def strip_charts(self) -> "Tensor":
        """Compiler mode: pure lattice — no charts, no labels."""
        return self._via(self.layout.strip_charts())

    def recenter(self, **deltas) -> "Tensor":
        """Move the physical frame (origin += delta); lattice and data
        untouched. Compare shift, which relabels the lattice and keeps the
        physical labels glued to the data."""
        return self._via(self.layout.recenter(**deltas))

    def field(self, name: str) -> "Tensor":
        """View one field of a structured dtype: offset bump + dtype change,
        strides untouched. Padding between fields is skipped for free. A
        value_units mapping is narrowed to the selected field's unit."""
        if self.dtype.fields is None:
            raise TypeError(f"dtype {self.dtype} has no fields")
        fdt, off = self.dtype.fields[name][0], self.dtype.fields[name][1]
        vu = self.value_units
        if isinstance(vu, Mapping):
            vu = vu.get(name)
        return replace(
            self,
            dtype=fdt,
            layout=self.layout.with_offset_delta(off),
            value_units=vu,
            carrier=carrier_of(fdt),  # the field's own carrier, re-inferred
        )

    def with_carrier(self, carrier: str | None) -> "Tensor":
        """Declare the algebraic object the values approximate (None re-infers
        from the dtype). Semantics lives here; the dtype is representation."""
        if carrier is None:
            return replace(self, carrier=carrier_of(self.dtype))
        return replace(self, carrier=carrier)

    def with_value_units(self, value_units) -> "Tensor":
        """Label the value space: a Unit for scalar dtypes, or a mapping
        field -> Unit for structured dtypes. Metadata for the future compute
        layer; item() still returns raw machine numbers."""
        return replace(self, value_units=value_units)

    # ------------------------------------------------------------------
    # step 2: guarded views
    # ------------------------------------------------------------------

    def _merge_fill(self, fill) -> object:
        if self.fill is not None and fill != self.fill:
            raise ValueError(f"tensor already has fill {self.fill!r}; one fill per tensor in this sketch")
        return fill

    def pad(self, fill, **extents) -> "Tensor":
        """Extend domains beyond the mapped region; outside reads give fill.
        Extents may be lattice ints or on-lattice Quantities."""
        return replace(self, layout=pad_layout(self.layout, extents), fill=self._merge_fill(fill))

    def stencil(
        self,
        name: str,
        k: tuple,
        k_name: str | None = None,
        fill=0,
        dilation: int = 1,
    ) -> "Tensor":
        """new[x=X, x_k=K] = old[x=X+dilation*K], or fill out of bounds.
        k is an INCLUSIVE tap-index range (D1 sugar): k=(-1, 1) is a 3-tap
        kernel; bounds may be physical displacements on a charted dim."""
        return replace(
            self,
            layout=stencil_layout(self.layout, name, k, k_name, dilation),
            fill=self._merge_fill(fill),
        )

    def simplify(self) -> "Tensor":
        """Discharge vacuous guards; may collapse back to a core Layout."""
        if isinstance(self.layout, GuardedLayout):
            return self._via(self.layout.simplify())
        return self

    def __repr__(self) -> str:
        extra = f", fill={self.fill!r}" if self.fill is not None else ""
        if self.value_units is not None:
            extra += f", values={self.value_units!r}"
        if self.carrier != carrier_of(self.dtype):
            extra += f", carrier={self.carrier}"
        return f"Tensor({self.dtype}, {self.layout!r}, {self.buffer!r}{extra})"


# ----------------------------------------------------------------------
# alignment: diagnosis for elementwise combination — never surgery (D17)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Misalignment:
    operand: int  # index into the argument list
    dim: str
    problem: str
    fix: str  # the exact primitive call that addresses it; "" if none exists

    def __repr__(self) -> str:
        tail = f"  ->  {self.fix}" if self.fix else ""
        return f"operand {self.operand}, dim {self.dim!r}: {self.problem}{tail}"


def aligned(*tensors: Tensor) -> bool:
    """True iff alignment(*tensors) reports nothing."""
    return not alignment(*tensors)


def alignment(*tensors: Tensor) -> tuple[Misalignment, ...]:
    """Diagnosis, never surgery: report exactly what stops these operands
    from being elementwise-combinable — every shared dim name identical in
    domain and labeling — and, for each item, the primitive call that fixes
    it (flip / shift / slice / repeat / decimate / with_charts: all cheap
    views the caller applies CONSCIOUSLY). Empty result means aligned.

    Diagnosis is iterative: some fixes unlock others (flip before shift
    before slice), so apply and re-run until clean. The reference frame for
    each dim name is the first operand that has it. Only layout-level
    alignment is checked; dtype and value-unit compatibility belong to the
    compute layer."""
    if len(tensors) < 2:
        return ()
    issues: list[Misalignment] = []
    ref: dict[str, Dim] = {}
    for t in tensors:
        for d in t.layout.dims:
            ref.setdefault(d.name, d)
    frame_true: dict[str, list[tuple[int, int]]] = {name: [] for name in ref}
    for idx, t in enumerate(tensors):
        for name, rd in ref.items():
            if name not in t.layout.names:
                fix = f"repeat({name!r}, ({rd.start}, {rd.stop})"
                if rd.chart is not None:
                    fix += ", chart=<reference chart>"
                if rd.labels is not None:
                    fix += ", labels=<reference labels>"
                fix += ")"
                issues.append(Misalignment(idx, name, "missing dim (broadcast needed)", fix))
                continue
            found = _frame_issue(rd, t.layout.dim(name))
            if found is not None:
                issues.append(Misalignment(idx, name, *found))
            else:
                d = t.layout.dim(name)
                frame_true[name].append((d.start, d.stop))
    # domain intersection recipes — only for dims whose frames all agree
    for name, spans in frame_true.items():
        if len(spans) < 2 or any(m.dim == name for m in issues):
            continue
        lo = max(s for s, _ in spans)
        hi = max(lo, min(e for _, e in spans))
        for idx, t in enumerate(tensors):
            if name in t.layout.names:
                d = t.layout.dim(name)
                if (d.start, d.stop) != (lo, hi):
                    note = " (empty intersection)" if lo == hi else ""
                    issues.append(
                        Misalignment(
                            idx,
                            name,
                            f"domain [{d.start}, {d.stop}) exceeds the common [{lo}, {hi}){note}",
                            f"slice({name}=({lo}, {hi}))",
                        )
                    )
    issues.sort(key=lambda m: (m.dim, m.operand))
    return tuple(issues)


def _frame_issue(rd: Dim, d: Dim) -> tuple[str, str] | None:
    """(problem, fix) when d's labeling frame differs from the reference
    dim rd's, else None. Domains are phase 2's business."""
    if (rd.labels is None) != (d.labels is None):
        return ("one operand is categorical, the other is not", "")
    if rd.labels is not None:
        if rd.labels != d.labels:
            return (f"categorical labels differ ({d.labels} vs {rd.labels})", "")
        if d.start != rd.start:
            return (
                "label frames offset at the lattice",
                f"shift({rd.name}={rd.start - d.start})",
            )
        return None
    ca, cb = rd.chart, d.chart
    if (ca is None) != (cb is None):
        return (
            "one operand is charted, the other is not (frames incomparable)",
            "with_charts(...) or strip_charts()",
        )
    if ca is None:
        return None  # both uncharted: the raw lattice is the shared frame
    if cb.step == -ca.step:
        return ("chart runs reversed (flipped storage)", f"flip({rd.name!r})")
    if cb.step != ca.step:
        fix = ""
        ratio = cb.step / ca.step
        if not isinstance(ratio, Quantity):
            if ratio.denominator == 1 and ratio > 1:
                # this operand is coarser: the (finer) reference must decimate
                fix = f"decimate({rd.name!r}, {int(ratio)}, phase=...) on the reference operand"
            elif ratio.numerator == 1 and 0 < ratio < 1:
                # this operand is finer: decimate it
                fix = f"decimate({rd.name!r}, {int(1 / ratio)}, phase=...)"
        return (
            f"steps differ ({cb.step!r} vs {ca.step!r}); resampling is not a view",
            fix,
        )
    if cb.kind != ca.kind or cb.axis != ca.axis:
        return (
            f"charts disagree on kind/axis ({cb.kind}/{cb.axis} vs {ca.kind}/{ca.axis})",
            "",
        )
    r = (cb.origin - ca.origin) / ca.step
    if isinstance(r, Quantity) or r.denominator != 1:
        return (
            "origins differ by a non-integer number of steps; the lattices do "
            "not interoperate (Chart.commensurable is the test)",
            "",
        )
    if r != 0:
        return (f"frames offset by {int(r)} steps", f"shift({rd.name}={int(r)})")
    return None
