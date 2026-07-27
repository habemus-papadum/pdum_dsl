"""Host indexing (design 250 §6, stage 3): the subscript law on Tensor —
named, order-free, typed (strict identity, containment extent), point
drops the dim, Slice keeps it, never promote to scalar, and the one
store-side promotion (scalar → memoryless const broadcast)."""

import numpy as np
import pytest
from pdum.tl import Frame, Slice, Tensor, q


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


@pytest.fixture
def img():
    # from_numpy SHARES memory; hand the tensor a copy so `arr` stays a
    # pristine reference for expectations.
    arr = np.arange(64, dtype=np.float64).reshape(8, 8)
    return arr, T(arr.copy(), ("y", "x"))


# ---------------------------------------------------------------------------
# The factory
# ---------------------------------------------------------------------------


def test_frames_factory(img):
    arr, t = img
    y, x = t.frames("y", "x")
    assert y == Frame("y", 0, 8) and x == Frame("x", 0, 8)
    assert t.frames() == (y, x)  # unnamed: all dims, presentation order
    with pytest.raises(KeyError, match="no dimension named"):
        t.frames("z")


def test_frames_carry_the_labeling():
    t = T(np.arange(8), ("x",)).with_charts(x=("0 um", "0.25 um"))
    (x,) = t.frames("x")
    assert x.chart is not None
    assert x[q("1.5 um")].i == 6  # the chart door works off the factory


# ---------------------------------------------------------------------------
# Points: select, order-free, rank-0, never promote
# ---------------------------------------------------------------------------


def test_point_drops_the_dim(img):
    arr, t = img
    y, x = t.frames("y", "x")
    row = t[y[3]]
    assert row.names == ("x",)
    np.testing.assert_array_equal(row.to_numpy(), arr[3])


def test_order_free_and_unmentioned_dims_pass(img):
    arr, t = img
    y, x = t.frames("y", "x")
    a, b = t[y[2], x[5]], t[x[5], y[2]]
    assert a.item() == b.item() == arr[2, 5]


def test_full_point_is_rank_zero_never_scalar(img):
    arr, t = img
    y, x = t.frames("y", "x")
    cell = t[y[2], x[5]]
    assert isinstance(cell, Tensor)  # NEVER a Python scalar
    assert cell.names == () and cell.numel == 1
    assert cell.to_numpy().shape == ()
    assert cell.item() == arr[2, 5]  # .item() is the one explicit exit


def test_coordinates_are_portable_across_tensors(img):
    arr, t = img
    other = T(arr * 2, ("y", "x"))
    y, x = t.frames("y", "x")
    assert other[y[1], x[1]].item() == 2 * arr[1, 1]


# ---------------------------------------------------------------------------
# Slices: the colon display, steps, one-sided forms
# ---------------------------------------------------------------------------


def test_slice_keeps_the_dim(img):
    arr, t = img
    y, x = t.frames("y", "x")
    crop = t[y[2] : y[6], x[0] : x[4]]
    assert crop.sizes() == {"y": 4, "x": 4}
    np.testing.assert_array_equal(crop.to_numpy(), arr[2:6, 0:4])


def test_slice_object_and_colon_display_agree(img):
    arr, t = img
    (y,) = t.frames("y")
    np.testing.assert_array_equal(t[Slice(y[2], y[6])].to_numpy(), t[y[2] : y[6]].to_numpy())


def test_stepped_slice_decimates(img):
    arr, t = img
    (y,) = t.frames("y")
    np.testing.assert_array_equal(t[y[0] :: 2].to_numpy(), arr[0:8:2])
    np.testing.assert_array_equal(t[y[3] :: 2].to_numpy(), arr[3:8:2])  # phase ≠ 0
    np.testing.assert_array_equal(t[y[1] : y[7] : 3].to_numpy(), arr[1:7:3])


def test_one_sided_forms(img):
    arr, t = img
    (y,) = t.frames("y")
    np.testing.assert_array_equal(t[y[5] :].to_numpy(), arr[5:])
    np.testing.assert_array_equal(t[: y[3]].to_numpy(), arr[:3])
    np.testing.assert_array_equal(t[y[0] :: 3].to_numpy(), arr[::3])


def test_empty_slice(img):
    arr, t = img
    (y,) = t.frames("y")
    assert t[y[3] : y[3]].sizes() == {"y": 0, "x": 8}


def test_chart_stays_glued_through_a_stepped_slice():
    t = T(np.arange(8), ("x",)).with_charts(x=("0 um", "0.25 um"))
    (x,) = t.frames("x")
    s = t[x[1] :: 2]  # points 1, 3, 5, 7
    d = s.layout.dim("x")
    assert d.chart.step == q("0.5 um")
    assert [s.phys("x", j) for j in range(d.start, d.stop)] == [q("0.25 um") * k for k in (1, 3, 5, 7)]


# ---------------------------------------------------------------------------
# The type check: strict identity, containment extent
# ---------------------------------------------------------------------------


def test_containment_is_the_one_relaxation(img):
    arr, t = img
    big = Frame("y", 0, 100)  # same name, no labeling, larger extent
    assert t[big[3]].to_numpy()[0] == arr[3, 0]  # contained: accepted
    with pytest.raises(IndexError, match="outside the target's domain"):
        t[big[50]]


def test_strict_identity_on_charts(img):
    arr, t = img
    tc = T(arr, ("y", "x")).with_charts(y=("0 um", "1 um"))
    yc = tc.frames("y")[0]
    y = t.frames("y")[0]
    with pytest.raises(TypeError, match="strict identity"):
        t[yc[2]]  # charted coordinate, chartless dim
    with pytest.raises(TypeError, match="strict identity"):
        tc[y[2]]  # chartless coordinate, charted dim
    other = T(arr, ("y", "x")).with_charts(y=("0 um", "2 um"))
    with pytest.raises(TypeError, match="strict identity"):
        other[yc[2]]  # different charts


def test_labels_agree_at_the_indexed_points():
    rgb = Frame("chan", 0, 3, labels=("r", "g", "b"))
    rgx = Frame("chan", 0, 3, labels=("r", "g", "x"))
    # a labeled tensor via repeat (a stride-0 labeled dim reads fine)
    lab = T(np.arange(3), ("k",)).repeat("chan", 3, labels=("r", "g", "b"))
    assert lab[rgb["g"]].sizes() == {"k": 3}  # matching labels: admitted
    with pytest.raises(TypeError, match="label 'x' vs 'b'"):
        lab[rgx["x"]]
    with pytest.raises(TypeError, match="labeled-meets-unlabeled"):
        lab[Frame("chan", 0, 3)[1]]


def test_rename_is_the_adapter(img):
    arr, t = img
    row = Frame("row", 0, 8)
    with pytest.raises(KeyError, match="no dimension named"):
        t[row[3]]
    np.testing.assert_array_equal(t.rename(y="row")[row[3]].to_numpy(), arr[3])


# ---------------------------------------------------------------------------
# Refusals at the door
# ---------------------------------------------------------------------------


def test_raw_ints_refuse_toward_frames(img):
    arr, t = img
    with pytest.raises(TypeError, match="make points via frames"):
        t[3]
    with pytest.raises(TypeError, match="make points via frames"):
        t[1:4]


def test_bare_colon_refuses(img):
    arr, t = img
    with pytest.raises(TypeError, match="bare \\[:\\]"):
        t[:]


def test_duplicate_dim_refuses(img):
    arr, t = img
    y, x = t.frames("y", "x")
    with pytest.raises(TypeError, match="duplicate index"):
        t[y[1], y[2]]
    with pytest.raises(TypeError, match="duplicate index"):
        t[y[1], y[2] : y[4]]


# ---------------------------------------------------------------------------
# The store side
# ---------------------------------------------------------------------------


def test_scalar_store_into_a_point(img):
    arr, t = img
    y, x = t.frames("y", "x")
    t[y[2], x[5]] = 99.0
    expect = arr.copy()
    expect[2, 5] = 99.0
    np.testing.assert_array_equal(t.to_numpy(), expect)


def test_scalar_broadcast_over_a_view(img):
    arr, t = img
    y, x = t.frames("y", "x")
    t[y[0] : y[2]] = 7.0  # pointwise's law: const broadcast over the view
    expect = arr.copy()
    expect[0:2, :] = 7.0
    np.testing.assert_array_equal(t.to_numpy(), expect)


def test_tensor_store_through_the_alignment_law(img):
    arr, t = img
    y, x = t.frames("y", "x")
    patch = T(np.full((2, 8), -1.0), ("y", "x"))  # same frames: [0,2) x [0,8)
    t[: y[2]] = patch
    expect = arr.copy()
    expect[0:2, :] = -1.0
    np.testing.assert_array_equal(t.to_numpy(), expect)
    with pytest.raises(TypeError, match="ALIGNED"):
        t[: y[2]] = T(np.zeros((3, 8)), ("y", "x"))  # domain [0,3) vs view [0,2)
    with pytest.raises(TypeError, match="aligned to the view"):
        t[: y[2]] = T(np.zeros((2, 8)), ("y", "z"))  # wrong dim names


def test_store_order_free(img):
    arr, t = img
    y, x = t.frames("y", "x")
    t2 = T(arr.copy(), ("y", "x"))
    t[y[1], x[2]] = 5.0
    t2[x[2], y[1]] = 5.0
    np.testing.assert_array_equal(t.to_numpy(), t2.to_numpy())


def test_aliased_writes_refuse():
    t = T(np.arange(3), ("k",)).repeat("r", 2)
    (r,) = t.frames("r")
    with pytest.raises(ValueError, match="aliased views"):
        t[r[0] :] = 1.0


def test_readonly_buffer_refuses_writes():
    arr = np.arange(8, dtype=np.float64)
    arr.flags.writeable = False
    t = Tensor.from_numpy(arr, ("x",))
    (x,) = t.frames("x")
    with pytest.raises(RuntimeError, match="read-only"):
        t[x[0]] = 1.0


def test_coordinate_is_not_a_value(img):
    arr, t = img
    y, x = t.frames("y", "x")
    with pytest.raises(TypeError, match="not a value"):
        t[y[0], x[0]] = y[3]
