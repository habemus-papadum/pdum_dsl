"""The two computational primitives + iota — reference (numpy) semantics.

This is the deliberately inefficient correctness layer from COMPUTE.md §3:
check alignment, export operands to numpy, apply the marker's numpy
function, wrap the result back with the surviving dims' charts and labels.
It materializes stride-0 repeats and overlapping windows — a real backend
must treat those views as virtual; here O(m·k·n) memory for a matmul is the
point, not a bug. It is the denotational semantics later layers must match.

- `pointwise(f, A, B, ...)`: operands must be 100% ALIGNED (same dim names,
  identical domains, equal charts/labels — `alignment()` is the gatekeeper
  and its diagnosis is quoted in the error). Guarded operands participate as
  their filled materialization; the result is a plain tensor.
- `reduce(f, A, dims, zero=None)`: fold the named dims; reduced dims drop,
  surviving dims keep their charts/labels. `zero` overrides the marker's
  identity (markers carry their own identities, so it is rarely needed).
  A normalizing reducer (`red.mean`) divides by the reduced numel — which is
  STATIC, known exactly from the layout.
- `scan(f, A, dim, zero=None)`: the inclusive prefix reduce — reduce
  keeping every intermediate accumulator, so the dim SURVIVES (cumsum =
  scan(red.sum)). A reverse (suffix) scan is flip . scan . flip — layout
  ops, free.
- `iota(t, name, unit=None)`: a dim's coordinates as data — the bridge from
  label-space to value-space, built TIGHT: a FunctionalBuffer declaring
  read(loc) = const + coeff*(loc//scale) with exact rational coefficients —
  no memory, and closed-form under every view op (the closure invariant:
  layout ops rewrite only the layout, so iota-ness cannot be destroyed;
  window on iota yields tap positions x+k, decimate yields factor*j+phase,
  all still declared). Default: the lattice face, carrier "int". With
  `unit=`: the physical face — values are the EXACT rational magnitudes of
  phys(i) in that unit (carrier "rat", recorded with value_units); the cast
  to the machine dtype happens only at the read. Semantics never mentions
  precision; the dtype is representation.

Markers (`pw.*`, `red.*`) are DECLARATIONS, not callbacks: a name, a numpy
function for this reference layer, and (for reducers) identity/associativity
— the algebra a compiler and the AD layer read. Unit-signature rules for
markers are future work; `pointwise` currently drops value_units.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from .buffer import Buffer, FunctionalBuffer
from .layout import Dim, Layout
from .tensor import Tensor, alignment
from .units import Quantity, Unit, u


@dataclass(frozen=True)
class Marker:
    """A pointwise primitive: a declared operation, not a callback."""

    name: str
    fn: object  # numpy callable applied elementwise to aligned arrays

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Reducer:
    """A reduction primitive: a declared monoid (plus numpy ufunc for the
    reference layer). `identity` is None when no dtype-independent identity
    exists (max/min); pass `zero=` to reduce() in that case if the reduced
    extent can be empty."""

    name: str
    fn: object  # numpy ufunc; .reduce(arr, axis=...) is used
    identity: object = None
    associative: bool = True
    commutative: bool = True
    normalize: bool = False  # divide by the (static) reduced numel

    def __repr__(self) -> str:
        return self.name


class pw:
    """The initial pointwise marker set."""

    add = Marker("add", np.add)
    sub = Marker("sub", np.subtract)
    mul = Marker("mul", np.multiply)
    div = Marker("div", np.divide)
    neg = Marker("neg", np.negative)
    exp = Marker("exp", np.exp)
    log = Marker("log", np.log)
    maximum = Marker("maximum", np.maximum)
    minimum = Marker("minimum", np.minimum)
    where = Marker("where", np.where)  # ternary select
    eq = Marker("eq", np.equal)
    ne = Marker("ne", np.not_equal)
    le = Marker("le", np.less_equal)
    lt = Marker("lt", np.less)
    ge = Marker("ge", np.greater_equal)
    gt = Marker("gt", np.greater)


class red:
    """The initial reducer set."""

    sum = Reducer("sum", np.add, identity=0)
    prod = Reducer("prod", np.multiply, identity=1)
    max = Reducer("max", np.maximum)
    min = Reducer("min", np.minimum)
    mean = Reducer("mean", np.add, identity=0, normalize=True)


def _tensor_like(arr: np.ndarray, dims: tuple[Dim, ...], value_units=None) -> Tensor:
    """Wrap a numpy array (in the dims' presentation order) as a fresh dense
    tensor carrying the template dims' domains, charts, and labels."""
    carr = np.ascontiguousarray(arr)
    try:
        mv = memoryview(carr).cast("B")
    except ValueError, TypeError:
        mv = memoryview(bytearray(carr.tobytes()))
    buf = Buffer(nbytes=carr.nbytes, data=mv)
    new_dims = tuple(replace(d, stride=s) for d, s in zip(dims, carr.strides))
    offset = -sum(d.stride * d.start for d in new_dims)
    return Tensor(buf, carr.dtype, Layout(new_dims, offset), value_units=value_units).check()


def pointwise(f: Marker, *tensors: Tensor) -> Tensor:
    """C[i, j, ...] = f(A[i, j, ...], B[i, j, ...], ...).

    All operands must be aligned; the error quotes the `alignment()`
    diagnosis, whose fixes (flip/shift/slice/repeat/...) the caller applies
    consciously (D17). The result carries the shared dims verbatim."""
    if not isinstance(f, Marker):
        raise TypeError(f"first argument must be a pointwise Marker, got {f!r}")
    if not tensors:
        raise ValueError("pointwise needs at least one operand")
    issues = alignment(*tensors)
    if issues:
        details = "\n".join(f"  {m!r}" for m in issues)
        raise ValueError(f"pointwise({f}) requires aligned operands:\n{details}")
    order = tensors[0].names
    arrays = [t.to_numpy(order=order) if len(order) else t.to_numpy() for t in tensors]
    out = f.fn(*arrays)
    return _tensor_like(np.asarray(out), tensors[0].layout.dims)


def reduce(f: Reducer, a: Tensor, dims, zero=None) -> Tensor:
    """Fold the named dims of `a` with the reducer's monoid; surviving dims
    keep their charts and labels. `zero` overrides the marker's identity."""
    if not isinstance(f, Reducer):
        raise TypeError(f"first argument must be a Reducer, got {f!r}")
    if isinstance(dims, str):
        dims = (dims,)
    if not dims:
        raise ValueError("reduce needs at least one dim")
    for name in dims:
        a.layout.dim(name)  # raises KeyError for unknown names
    arr = a.to_numpy()
    axes = tuple(i for i, n in enumerate(a.names) if n in dims)
    if len(axes) != len(set(dims)):
        raise ValueError(f"duplicate names in dims {dims}")
    initial = zero if zero is not None else f.identity
    if initial is not None:
        out = f.fn.reduce(arr, axis=axes, initial=initial)
    else:
        out = f.fn.reduce(arr, axis=axes)
    if f.normalize:
        n = 1
        for name in dims:
            n *= a.layout.dim(name).size
        out = out / n
    survivors = tuple(d for d in a.layout.dims if d.name not in dims)
    return _tensor_like(np.asarray(out), survivors)


def iota(t: Tensor | Layout, name: str, unit=None, dtype=None) -> Tensor:
    """Materialize `name`'s coordinates as data, aligned with `t` (constant
    along every other dim via stride-0 dims; no memory at all — the buffer
    is a FunctionalBuffer declaring the closed form).

    Default: the lattice face — values are the raw lattice ints, carrier
    "int". With `unit=` (a Unit or string): the physical face — values are
    the exact rational magnitudes of phys(i) in that unit, carrier "rat",
    `value_units` recording the unit; representation (the dtype, float64 by
    default) enters only at the read. A gradient-free constant field."""
    layout = t.layout if isinstance(t, Tensor) else t
    d = layout.dim(name)
    if unit is None:
        dt = np.dtype(dtype or np.int64)
        coeff, const = Fraction(1), Fraction(d.start)
        value_units, carrier = None, "int"
    else:
        if isinstance(unit, str):
            unit = u.parse_unit(unit)
        if not isinstance(unit, Unit):
            raise TypeError(f"unit must be a Unit or string, got {unit!r}")
        if d.chart is None:
            raise TypeError(f"dim {name} has no chart; physical iota needs one (labels have no numeric face)")
        one = Quantity(1, unit)
        step_u = d.chart.step / one
        origin_u = d.chart.origin / one
        if isinstance(step_u, Quantity) or isinstance(origin_u, Quantity):
            raise ValueError(f"unit {unit!r} has different dimensions than {name}'s chart")
        dt = np.dtype(dtype or np.float64)
        coeff, const = step_u, origin_u + d.start * step_u
        value_units, carrier = unit, "rat"
    scale = dt.itemsize
    buf = FunctionalBuffer(nbytes=d.size * scale, scale=scale, coeff=coeff, const=const)
    new_dims = tuple(replace(x, stride=scale if x.name == name else 0) for x in layout.dims)
    return Tensor(
        buf,
        dt,
        Layout(new_dims, offset=-scale * d.start),
        value_units=value_units,
        carrier=carrier,
    ).check()


def scan(f: Reducer, a: Tensor, dim: str, zero=None) -> Tensor:
    """Inclusive prefix reduce along ONE dim: out[.., t, ..] folds
    a[.., 0..t, ..]. The dim survives with its domain, chart, and labels —
    scan is reduce keeping every intermediate accumulator (cumsum =
    scan(red.sum); running max = scan(red.max)). A reverse (suffix) scan is
    flip . scan . flip — layout ops, free. A normalizing reducer (red.mean)
    divides prefix t by its running count.

    Reference limitation: the marker must be a numpy ufunc (accumulate);
    scans over composite pair-state operators (SSM recurrences) await the
    marker DSL."""
    if not isinstance(f, Reducer):
        raise TypeError(f"first argument must be a Reducer, got {f!r}")
    if not isinstance(dim, str):
        raise TypeError("scan folds exactly one dim (pass its name)")
    a.layout.dim(dim)  # raises KeyError for unknown names
    arr = a.to_numpy()
    axis = a.names.index(dim)
    out = f.fn.accumulate(arr, axis=axis)
    if zero is not None:
        out = f.fn(out, zero)
    if f.normalize:
        n = a.layout.dim(dim).size
        shape = [1] * arr.ndim
        shape[axis] = n
        out = out / np.arange(1, n + 1).reshape(shape)
    return _tensor_like(np.asarray(out), a.layout.dims)
