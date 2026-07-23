"""The compute reference layer: pointwise, reduce, iota (COMPUTE.md §2-3)."""

import numpy as np
import pytest
from tensorlib import Tensor, iota, pointwise, pw, q, red, reduce, u

# ----------------------------------------------------------------------
# pointwise
# ----------------------------------------------------------------------


def test_pointwise_requires_alignment_and_quotes_the_diagnosis():
    a = Tensor.from_numpy(np.arange(4, dtype=np.int64), ("x",))
    b = a.shift(x=1)
    with pytest.raises(ValueError) as e:
        pointwise(pw.add, a, b)
    assert "slice(x=(1, 4))" in str(e.value)  # the D17 recipe, verbatim


def test_pointwise_basic_and_charts_survive():
    arr = np.arange(6, dtype=np.float64).reshape(2, 3)
    a = Tensor.from_numpy(arr, ("i", "j")).with_charts(i=("0 um", "1 um"))
    b = Tensor.from_numpy(arr * 10, ("i", "j")).with_charts(i=("0 um", "1 um"))
    c = pointwise(pw.add, a, b)
    np.testing.assert_array_equal(c.to_numpy(), arr * 11)
    assert c.layout.dim("i").chart == a.layout.dim("i").chart
    assert c.item(i=q("1 um"), j=2) == arr[1, 2] * 11


def test_pointwise_is_presentation_order_free():
    arr = np.arange(6, dtype=np.int64).reshape(2, 3)
    a = Tensor.from_numpy(arr, ("i", "j"))
    b = Tensor.from_numpy(arr.T.copy(), ("j", "i"))  # same data, dims listed j, i
    c = pointwise(pw.sub, a, b)  # D5: order carries no meaning
    np.testing.assert_array_equal(c.to_numpy(order=("i", "j")), np.zeros((2, 3)))


def test_pointwise_reads_guarded_operands_as_filled():
    arr = np.arange(1, 5, dtype=np.int64)
    p = Tensor.from_numpy(arr, ("x",)).pad(fill=0, x=(-1, 5))
    plain = Tensor.from_numpy(np.ones(6, dtype=np.int64), ("x",)).shift(x=-1)
    c = pointwise(pw.mul, p, plain)
    np.testing.assert_array_equal(c.to_numpy(), [0, 1, 2, 3, 4, 0])


# ----------------------------------------------------------------------
# reduce
# ----------------------------------------------------------------------


def test_reduce_basics_and_surviving_charts():
    arr = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    t = Tensor.from_numpy(arr, ("i", "j", "k")).with_charts(i=("0 ms", "1 ms"))
    np.testing.assert_array_equal(reduce(red.sum, t, ("j",)).to_numpy(), arr.sum(axis=1))
    np.testing.assert_array_equal(reduce(red.max, t, ("j", "k")).to_numpy(), arr.max(axis=(1, 2)))
    s = reduce(red.sum, t, ("j", "k"))
    assert s.layout.dim("i").chart == t.layout.dim("i").chart


def test_reduce_all_dims_yields_scalar():
    arr = np.arange(6, dtype=np.int64).reshape(2, 3)
    t = Tensor.from_numpy(arr, ("i", "j"))
    loss = reduce(red.sum, t, ("i", "j"))
    assert loss.layout.dims == ()
    assert loss.item() == arr.sum()


def test_mean_normalizes_by_static_numel():
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    t = Tensor.from_numpy(arr, ("i", "j"))
    np.testing.assert_allclose(reduce(red.mean, t, ("j",)).to_numpy(), arr.mean(axis=1))


def test_reduce_zero_override():
    arr = np.arange(4, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",))
    assert reduce(red.sum, t, "x", zero=100).item() == 100 + arr.sum()


# ----------------------------------------------------------------------
# the canon, as programs
# ----------------------------------------------------------------------


def test_matmul_is_repeat_mul_reduce():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((3, 4))
    b = rng.standard_normal((4, 5))
    A = Tensor.from_numpy(a, ("m", "k")).repeat("n", 5)
    B = Tensor.from_numpy(b, ("k", "n")).repeat("m", 3)
    C = reduce(red.sum, pointwise(pw.mul, A, B), ("k",))
    np.testing.assert_allclose(C.to_numpy(order=("m", "n")), a @ b)


def test_valid_conv1d_is_window_mul_reduce():
    x = np.arange(8, dtype=np.float64)
    w = np.array([1.0, -2.0, 0.5])
    X = Tensor.from_numpy(x, ("x",)).window("x", "k", 3)
    W = Tensor.from_numpy(w, ("k",)).repeat("x", (0, 6))
    y = reduce(red.sum, pointwise(pw.mul, X, W), ("k",))
    np.testing.assert_allclose(y.to_numpy(), np.correlate(x, w, mode="valid"))


def test_padded_conv_uses_the_guard_as_boundary():
    x = np.arange(1, 5, dtype=np.float64)
    w = np.ones(3)
    X = Tensor.from_numpy(x, ("x",)).stencil("x", k=(-1, 1), fill=0.0)
    # conscious alignment: relabel the weights' taps onto the stencil's [-1, 2)
    W = Tensor.from_numpy(w, ("k",)).rename(k="x_k").shift(x_k=-1).repeat("x", 4)
    y = reduce(red.sum, pointwise(pw.mul, X, W), ("x_k",))
    np.testing.assert_allclose(y.to_numpy(), np.convolve(x, w, mode="same"))


def test_softmax_is_reduce_repeat_pointwise():
    rng = np.random.default_rng(1)
    s = rng.standard_normal((3, 5))
    S = Tensor.from_numpy(s, ("i", "v"))
    m = reduce(red.max, S, "v")
    e = pointwise(pw.exp, pointwise(pw.sub, S, m.repeat("v", 5)))
    z = reduce(red.sum, e, "v")
    P = pointwise(pw.div, e, z.repeat("v", 5))
    expect = np.exp(s - s.max(axis=1, keepdims=True))
    expect /= expect.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(P.to_numpy(), expect)


def test_causal_attention_mask_via_iota():
    rng = np.random.default_rng(2)
    s = rng.standard_normal((4, 4))
    S = Tensor.from_numpy(s, ("i", "j"))
    mask = pointwise(pw.le, iota(S, "j"), iota(S, "i"))  # j <= i
    neg = Tensor.from_numpy(np.float64(-1e30), ()).repeat("i", 4).repeat("j", 4)
    masked = pointwise(pw.where, mask, S, neg)
    out = masked.to_numpy(order=("i", "j"))
    expect = np.where(np.tril(np.ones((4, 4), dtype=bool)), s, -1e30)
    np.testing.assert_array_equal(out, expect)


# ----------------------------------------------------------------------
# iota
# ----------------------------------------------------------------------


def test_iota_lattice_face():
    t = Tensor.dense(np.float32, x=2, y=3, z=4)
    Y = iota(t, "y")
    assert Y.item(x=1, y=2, z=3) == 2
    assert Y.buffer.nbytes == 3 * 8  # one coordinate array + stride-0 repeats
    assert Y.value_units is None
    sh = Tensor.from_numpy(np.arange(5, dtype=np.int64), ("x",)).shift(x=-2)
    np.testing.assert_array_equal(iota(sh, "x").to_numpy(), [-2, -1, 0, 1, 2])


def test_iota_physical_face_records_the_unit():
    t = Tensor.from_numpy(np.zeros(4), ("x",)).with_charts(x=("0 um", "0.25 um"))
    X = iota(t, "x", unit=u.um)
    np.testing.assert_allclose(X.to_numpy(), [0.0, 0.25, 0.5, 0.75])
    assert X.value_units == u.um
    Xnm = iota(t, "x", unit="nm")
    np.testing.assert_allclose(Xnm.to_numpy(), [0.0, 250.0, 500.0, 750.0])
    with pytest.raises(ValueError):
        iota(t, "x", unit=u.s)  # wrong dimensions
    with pytest.raises(TypeError):
        iota(t.strip_charts(), "x", unit=u.um)  # no chart, no physical face


def test_iota_keeps_all_dim_charts():
    t = Tensor.dense(np.float32, x=2, y=3).with_charts(x=("0 um", "1 um"), y=("0 ms", "1 ms"))
    X = iota(t, "x", unit=u.um)
    assert X.layout.dim("x").chart == t.layout.dim("x").chart
    assert X.layout.dim("y").chart == t.layout.dim("y").chart


# ----------------------------------------------------------------------
# tight iota: the closure invariant
# ----------------------------------------------------------------------


def test_iota_is_functional_not_materialized():
    from tensorlib import FunctionalBuffer

    t = Tensor.from_numpy(np.arange(6, dtype=np.int64), ("x",))
    ix = iota(t, "x")
    assert isinstance(ix.buffer, FunctionalBuffer)
    assert ix.buffer.data is None  # no memory at all
    np.testing.assert_array_equal(ix.to_numpy(), np.arange(6))


def test_layout_ops_cannot_destroy_iota():
    from tensorlib import FunctionalBuffer

    t = Tensor.from_numpy(np.arange(6, dtype=np.int64), ("x",))
    ix = iota(t, "x")
    W = ix.window("x", "k", 3)
    assert W.buffer is ix.buffer  # ops rewrite layouts, never buffers
    assert isinstance(W.buffer, FunctionalBuffer)
    assert W.item(x=2, k=1) == 3  # value = x + k: tap positions, closed form
    D = ix.decimate("x", 2, phase=1)
    assert isinstance(D.buffer, FunctionalBuffer)
    np.testing.assert_array_equal(D.to_numpy(), [1, 3, 5])  # factor*j + phase
    F = ix.flip("x")
    np.testing.assert_array_equal(F.to_numpy(), [5, 4, 3, 2, 1, 0])
    S = ix.shift(x=10)
    assert S.item(x=12) == 2  # labels glued: same datum, same value
    P = ix.pad(fill=-1, x=(-2, 8))
    assert P.item(x=-1) == -1 and P.item(x=3) == 3  # guarded, still functional
    assert isinstance(P.buffer, FunctionalBuffer)


def test_physical_iota_is_exact_rationals_cast_at_read():
    from fractions import Fraction

    from tensorlib import FunctionalBuffer

    t = Tensor.from_numpy(np.zeros(4), ("x",)).with_charts(x=("0 um", "1/3 um"))
    X = iota(t, "x", unit=u.um)
    assert isinstance(X.buffer, FunctionalBuffer)
    assert X.buffer.coeff == Fraction(1, 3)  # exact inside...
    assert X.item(x=2) == np.float64(float(Fraction(2, 3)))  # ...cast at the read
    assert X.carrier == "rat" and X.dtype == np.float64  # semantics vs representation


# ----------------------------------------------------------------------
# carriers
# ----------------------------------------------------------------------


def test_carriers_infer_from_dtype_and_thread():
    t = Tensor.from_numpy(np.arange(4, dtype=np.int64), ("x",))
    assert t.carrier == "int"
    assert t.slice(x=(0, 2)).carrier == "int"
    f = Tensor.from_numpy(np.zeros(3, dtype=np.float32), ("x",))
    assert f.carrier == "real"
    mask = pointwise(pw.le, iota(t, "x"), iota(t, "x"))
    assert mask.carrier == "bool"


def test_carrier_override_and_field_reinference():
    t = Tensor.from_numpy(np.arange(4, dtype=np.float64), ("x",))
    r = t.with_carrier("rat")  # exact values represented in floats
    assert r.carrier == "rat"
    assert r.with_carrier(None).carrier == "real"
    dt = np.dtype([("re", "<f4"), ("im", "<f4")])
    rec = np.zeros(3, dt)
    c = Tensor.from_numpy(rec, ("x",)).with_carrier("complex")
    assert c.field("re").carrier == "real"  # the field's own carrier
    with pytest.raises(ValueError):
        t.with_carrier("quaternion")


def test_lattice_iota_carrier_is_int():
    t = Tensor.dense(np.float32, x=4)
    assert iota(t, "x").carrier == "int"


# ----------------------------------------------------------------------
# scan
# ----------------------------------------------------------------------


def test_scan_is_cumsum():
    from tensorlib import scan

    arr = np.arange(24, dtype=np.float64).reshape(4, 6)
    t = Tensor.from_numpy(arr, ("i", "j")).with_charts(j=("0 ms", "1 ms"))
    s = scan(red.sum, t, "j")
    np.testing.assert_array_equal(s.to_numpy(), np.cumsum(arr, axis=1))
    assert s.layout.dim("j").chart == t.layout.dim("j").chart  # dim SURVIVES


def test_scan_running_max_and_mean():
    from tensorlib import scan

    arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    t = Tensor.from_numpy(arr, ("x",))
    np.testing.assert_array_equal(scan(red.max, t, "x").to_numpy(), np.maximum.accumulate(arr))
    np.testing.assert_allclose(
        scan(red.mean, t, "x").to_numpy(),
        np.cumsum(arr) / np.arange(1, 6),  # running mean: count is per-prefix
    )


def test_reverse_scan_is_flip_scan_flip():
    from tensorlib import scan

    arr = np.arange(1, 6, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",))
    suffix = scan(red.sum, t.flip("x"), "x").flip("x")
    np.testing.assert_array_equal(suffix.to_numpy(), np.cumsum(arr[::-1])[::-1])


def test_scan_zero_override():
    from tensorlib import scan

    t = Tensor.from_numpy(np.arange(4, dtype=np.int64), ("x",))
    np.testing.assert_array_equal(scan(red.sum, t, "x", zero=100).to_numpy(), 100 + np.cumsum(np.arange(4)))
