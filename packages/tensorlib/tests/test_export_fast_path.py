"""to_numpy's strided fast path vs the per-element oracle (the launch-repack
fix — 211 §4: the loop was 88–100% of every device launch).

The affine law makes the fast path exact: ``get_loc = offset + Σ stride·coord``
IS numpy's strided model, so a plain Layout over host bytes exports as one
bounds-checked ndarray view + copy. The oracle loop STAYS (``_to_numpy_oracle``)
and every view-op composition must agree between the two — including negative
strides (flip), stride-0 dims (repeat), and permuted export orders. Guarded
layouts (fill) and functional buffers DECLINE the fast path and keep the loop.
"""

import numpy as np
import pytest

from pdum.tl import Tensor
from pdum.tl.compute import iota


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


RNG = np.random.default_rng(3)
BASE = T(RNG.standard_normal((6, 8)), ("y", "x"))


CASES = {
    "identity": lambda: BASE,
    "flip": lambda: BASE.flip("x"),
    "double-flip": lambda: BASE.flip("x").flip("y"),
    "slice": lambda: BASE.slice(y=(1, 5), x=(2, 7)),
    "shift": lambda: BASE.shift(y=2),
    "decimate": lambda: BASE.decimate("x", 2, 1),
    "window": lambda: BASE.window("x", "x_k", (-1, 2)),
    "merge": lambda: BASE.merge(("y", "x"), "yx"),
    "split": lambda: BASE.split("x", xo=2, xi=4),
    "diagonal": lambda: T(RNG.standard_normal((5, 5)), ("a", "b")).diagonal(("a", "b"), "d"),
    "repeat": lambda: T(RNG.standard_normal(7), ("x",)).repeat("y", (0, 4)),
    "composition": lambda: BASE.flip("y").decimate("x", 2).slice(y=(1, 4)),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_fast_path_agrees_with_the_oracle(name):
    t = CASES[name]()
    dims = t.layout.dims
    fast = t._strided_export(dims)
    assert fast is not None, f"{name}: a plain affine layout must take the fast path"
    np.testing.assert_array_equal(fast, t._to_numpy_oracle(dims))


def test_permuted_export_order_agrees():
    dims = tuple(BASE.layout.dim(n) for n in ("x", "y"))
    np.testing.assert_array_equal(BASE._strided_export(dims), BASE._to_numpy_oracle(dims))
    np.testing.assert_array_equal(BASE.to_numpy(order=("x", "y")), BASE.to_numpy().T)


def test_the_export_never_aliases_the_buffer():
    out = BASE.to_numpy()
    out[0, 0] += 1.0
    assert BASE.item(y=0, x=0) != out[0, 0]  # materialize means COPY


def test_guarded_and_functional_decline_to_the_oracle():
    """Fill semantics and computed reads are not strided maps — they keep
    the per-element loop and stay correct."""
    padded = BASE.pad(fill=-1.0, x=(0, 9))
    assert padded._strided_export(tuple(padded.layout.dims)) is None
    got = padded.to_numpy(order=("y", "x"))
    want = np.concatenate([BASE.to_numpy(), np.full((6, 1), -1.0)], axis=1)
    np.testing.assert_array_equal(got, want)

    st = BASE.stencil("x", k=(-1, 1), fill=-7.0)
    assert st._strided_export(tuple(st.layout.dims)) is None

    iv = iota(BASE, "y")  # a FunctionalBuffer: values are computed, not stored
    assert iv._strided_export(tuple(iv.layout.dims)) is None
    assert float(iv.to_numpy(order=("y", "x"))[3, 2]) == 3.0


def test_record_dtype_rides_the_fast_path():
    rec = np.dtype([("a", "<f8"), ("b", "<f8")])
    arr = np.zeros(4, dtype=rec)
    arr["a"], arr["b"] = np.arange(4.0), np.arange(4.0) * 10
    t = Tensor.from_numpy(arr, ("i",))
    dims = t.layout.dims
    fast = t._strided_export(dims)
    if fast is None:  # a structured dtype the buffer protocol refused to cast
        pytest.skip("structured dtype fell back to the oracle — allowed, not expected")
    np.testing.assert_array_equal(fast, t._to_numpy_oracle(dims))
