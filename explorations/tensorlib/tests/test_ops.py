"""Step 1: Tensor view operations checked against numpy ground truth."""

import numpy as np
import pytest
from tensorlib import Dim, Injectivity, Layout, Tensor, bfloat16


@pytest.fixture
def t3():
    arr = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    return arr, Tensor.from_numpy(arr, ("i", "j", "k"))


def test_from_numpy_roundtrip(t3):
    arr, t = t3
    assert t.sizes() == {"i": 2, "j": 3, "k": 4}
    np.testing.assert_array_equal(t.to_numpy(), arr)


def test_memory_is_shared_not_copied(t3):
    arr, t = t3
    arr[1, 2, 3] = 999
    assert t.item(i=1, j=2, k=3) == 999


def test_slice(t3):
    arr, t = t3
    np.testing.assert_array_equal(t.slice(j=(1, 3)).to_numpy(), arr[:, 1:3, :])


def test_slice_keeps_coordinates_natural(t3):
    arr, t = t3
    sub = t.slice(j=(1, 3))
    # element j=2 is still called j=2 after slicing (D3: raw coordinates)
    assert sub.item(i=0, j=2, k=1) == arr[0, 2, 1]


def test_select(t3):
    arr, t = t3
    np.testing.assert_array_equal(t.select(i=1).to_numpy(), arr[1])


def test_shift(t3):
    arr, t = t3
    moved = t.shift(k=10)
    assert moved.item(i=1, j=1, k=12) == arr[1, 1, 2]
    np.testing.assert_array_equal(moved.to_numpy(), arr)


def test_to_numpy_export_order(t3):
    arr, t = t3
    np.testing.assert_array_equal(t.to_numpy(order=("k", "j", "i")), arr.transpose(2, 1, 0))
    with pytest.raises(KeyError):
        t.to_numpy(order=("k", "j"))  # not a permutation


def test_canonical_erases_presentation_order():
    a = Layout((Dim("x", 1, 0, 4), Dim("y", 4, 0, 3)))
    b = Layout((Dim("y", 4, 0, 3), Dim("x", 1, 0, 4)))
    assert a != b  # dataclass equality sees presentation order
    assert a.canonical() == b.canonical()  # canonical erases it
    assert a.canonical() != a.rename(x="z").canonical()  # names stay semantic


def test_flip(t3):
    arr, t = t3
    np.testing.assert_array_equal(t.flip("k").to_numpy(), arr[:, :, ::-1])


def test_repeat_is_stride_zero_broadcast():
    arr = np.arange(5, dtype=np.float64)
    t = Tensor.from_numpy(arr, ("x",))
    r = t.repeat("b", 3)
    np.testing.assert_array_equal(r.to_numpy(), np.broadcast_to(arr[:, None], (5, 3)))
    assert r.injectivity() is Injectivity.ALIASED


def test_split_into_blocks():
    arr = np.arange(16, dtype=np.int64).reshape(4, 4)
    t = Tensor.from_numpy(arr, ("i", "j"))
    b = t.split("i", io=2, ii=2).split("j", jo=2, ji=2)
    assert b.names == ("io", "ii", "jo", "ji")
    np.testing.assert_array_equal(b.to_numpy(), arr.reshape(2, 2, 2, 2))
    for a in range(2):
        for x in range(2):
            for c in range(2):
                for d in range(2):
                    assert b.item(io=a, ii=x, jo=c, ji=d) == arr[2 * a + x, 2 * c + d]


def test_split_with_natural_ranges():
    arr = np.arange(24, dtype=np.int64).reshape(4, 6)
    t = Tensor.from_numpy(arr, ("i", "j"))
    # blocks of j indexed 1..3, within-block 0..1: j = (jb - 1) * 2 + jw
    tn = t.split("j", jb=(1, 4), jw=2)
    for i in range(4):
        for jb in range(1, 4):
            for jw in range(2):
                assert tn.item(i=i, jb=jb, jw=jw) == arr[i, (jb - 1) * 2 + jw]


def test_merge_roundtrip():
    arr = np.arange(16, dtype=np.int64).reshape(4, 4)
    t = Tensor.from_numpy(arr, ("i", "j"))
    b = t.split("i", io=2, ii=2)
    np.testing.assert_array_equal(b.merge(("io", "ii"), "i").to_numpy(), arr)
    # merge with a chosen start relabels but addresses the same memory
    m = b.merge(("io", "ii"), "i", start=5)
    for x in range(4):
        assert m.item(i=5 + x, j=1) == arr[x, 1]


def test_merge_flattens_compatible_dims():
    arr = np.arange(24, dtype=np.int64).reshape(4, 6)
    t = Tensor.from_numpy(arr, ("i", "j"))
    np.testing.assert_array_equal(t.merge(("i", "j"), "f").to_numpy(), arr.reshape(-1))
    with pytest.raises(ValueError):
        t.merge(("j", "i"), "f")  # wrong nesting order


def test_diagonal():
    arr = np.arange(16, dtype=np.int64).reshape(4, 4)
    t = Tensor.from_numpy(arr, ("i", "j"))
    np.testing.assert_array_equal(t.diagonal(("i", "j"), "d").to_numpy(), np.diagonal(arr))


def test_window_matches_sliding_window_view():
    arr = np.arange(6, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",))
    w = t.window("x", "k", 3)
    np.testing.assert_array_equal(w.to_numpy(), np.lib.stride_tricks.sliding_window_view(arr, 3))


def test_field_selection_skips_padding():
    dt = np.dtype([("a", "u1"), ("b", "<i8")], align=True)
    assert dt.itemsize == 16  # 7 bytes of padding after 'a'
    arr = np.zeros(4, dt)
    arr["a"] = [1, 2, 3, 4]
    arr["b"] = [10, 20, 30, 40]
    t = Tensor.from_numpy(arr, ("x",))
    np.testing.assert_array_equal(t.field("b").to_numpy(), arr["b"])
    np.testing.assert_array_equal(t.field("a").to_numpy(), arr["a"])


def test_dense_first_listed_dim_is_fastest():
    t = Tensor.dense(np.float32, x=3, y=(-1, 2))
    assert t.strides_in_elements() == {"x": 1, "y": 3}
    assert t.footprint() == (0, 9 * 4)  # min corner at byte 0 despite y start=-1
    t.check()


def test_bounds_check_rejects_oversized_layout():
    t = Tensor.from_numpy(np.arange(4, dtype=np.int64), ("x",))
    bad = Tensor(t.buffer, t.dtype, Layout((Dim("x", 8, 0, 5),)))
    with pytest.raises(ValueError):
        bad.check()


def test_overlaps_is_footprint_based():
    arr = np.arange(8, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",))
    lo, hi = t.slice(x=(0, 4)), t.slice(x=(4, 8))
    assert not lo.overlaps(hi)
    assert lo.overlaps(t.slice(x=(3, 5)))
    other = Tensor.from_numpy(arr.copy(), ("x",))
    assert not lo.overlaps(other)  # different buffers


@pytest.mark.skipif(bfloat16 is None, reason="ml_dtypes not installed")
def test_bfloat16_roundtrip():
    arr = np.arange(6, dtype=bfloat16).reshape(2, 3)
    t = Tensor.from_numpy(arr, ("i", "j"))
    assert t.itemsize == 2
    np.testing.assert_array_equal(t.to_numpy(), arr)
