"""Step 2: guard + fill layouts — pad, stencil, and the guard-rewrite algebra."""

import numpy as np
import pytest
from pdum.tl import GuardedLayout, Layout, Tensor


@pytest.fixture
def t1():
    arr = np.arange(1, 6, dtype=np.int64)  # 1..5, distinct from fill values
    return arr, Tensor.from_numpy(arr, ("x",))


# ----------------------------------------------------------------------
# pad
# ----------------------------------------------------------------------


def test_pad_basic(t1):
    arr, t = t1
    p = t.pad(fill=-1, x=(-2, 7))
    assert p.sizes() == {"x": 9}
    expected = np.concatenate([[-1, -1], arr, [-1, -1]])
    np.testing.assert_array_equal(p.to_numpy(), expected)
    assert p.item(x=-1) == -1
    assert p.item(x=0) == arr[0]
    p.check()  # footprint honors the guard: no negative byte addresses


def test_pad_rejects_shrinking(t1):
    _, t = t1
    with pytest.raises(ValueError):
        t.pad(fill=0, x=(1, 4))


def test_pad_slice_interior_simplifies_to_core(t1):
    arr, t = t1
    p = t.pad(fill=-1, x=(-2, 7)).slice(x=(0, 5)).simplify()
    assert isinstance(p.layout, Layout)  # guard discharged, back to core family
    np.testing.assert_array_equal(p.to_numpy(), arr)


def test_pad_slice_into_padding_is_always_fill(t1):
    arr, t = t1
    edge = t.pad(fill=-1, x=(-2, 7)).slice(x=(5, 7))
    assert isinstance(edge.layout, GuardedLayout)
    assert edge.layout.always_fill()
    np.testing.assert_array_equal(edge.to_numpy(), [-1, -1])


# ----------------------------------------------------------------------
# stencil
# ----------------------------------------------------------------------


def stencil_reference(arr, k_min, k_max, fill):
    n = len(arr)
    out = np.full((n, k_max - k_min + 1), fill, dtype=arr.dtype)
    for i in range(n):
        for k in range(k_min, k_max + 1):
            if 0 <= i + k < n:
                out[i, k - k_min] = arr[i + k]
    return out


def test_stencil_semantics(t1):
    arr, t = t1
    s = t.stencil("x", k=(-1, 1), fill=-7)  # k range is inclusive (D1 sugar)
    assert s.sizes() == {"x": 5, "x_k": 3}
    assert s.item(x=0, x_k=-1) == -7
    assert s.item(x=0, x_k=0) == arr[0]
    assert s.item(x=2, x_k=1) == arr[3]
    assert s.item(x=4, x_k=1) == -7
    np.testing.assert_array_equal(s.to_numpy(), stencil_reference(arr, -1, 1, -7))
    s.check()  # guard-aware footprint stays inside the buffer


def test_stencil_interior_equals_window(t1):
    arr, t = t1
    interior = t.stencil("x", k=(-1, 1), fill=0).slice(x=(1, 4)).simplify()
    assert isinstance(interior.layout, Layout)  # boundary guard discharged
    w = t.window("x", "x_k", (-1, 2))  # window takes a half-open range
    np.testing.assert_array_equal(interior.to_numpy(), w.to_numpy())


# ----------------------------------------------------------------------
# the guard-rewrite algebra
# ----------------------------------------------------------------------


def test_split_through_stencil_guard():
    arr = np.arange(1, 9, dtype=np.int64)  # 1..8
    t = Tensor.from_numpy(arr, ("x",))
    s = t.stencil("x", k=(-1, 1), fill=0).split("x", xo=2, xi=4)
    for a in range(2):
        for b in range(4):
            for c in (-1, 0, 1):
                x = 4 * a + b + c
                expected = arr[x] if 0 <= x < 8 else 0
                assert s.item(xo=a, xi=b, x_k=c) == expected


def test_split_with_natural_ranges_through_guard():
    arr = np.arange(1, 9, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",))
    # xo indexed 1..2: x = 4 * (xo - 1) + xi
    s = t.stencil("x", k=(-1, 1), fill=0).split("x", xo=(1, 3), xi=4)
    for a in range(1, 3):
        for b in range(4):
            for c in (-1, 0, 1):
                x = 4 * (a - 1) + b + c
                expected = arr[x] if 0 <= x < 8 else 0
                assert s.item(xo=a, xi=b, x_k=c) == expected


def test_shift_through_guard(t1):
    arr, t = t1
    s = t.stencil("x", k=(-1, 1), fill=-7).shift(x=10)
    assert s.item(x=10, x_k=-1) == -7  # was x=0: still the left boundary
    assert s.item(x=11, x_k=1) == arr[2]
    assert s.item(x=14, x_k=1) == -7


def test_select_through_guard(t1):
    arr, t = t1
    s = t.stencil("x", k=(-1, 1), fill=-7)
    row = s.select(x=0)
    assert row.item(x_k=-1) == -7
    assert row.item(x_k=0) == arr[0]
    assert row.item(x_k=1) == arr[1]
    # selecting everything leaves a 0-d tensor with a constant guard
    assert s.select(x=0, x_k=-1).item() == -7
    assert s.select(x=2, x_k=1).item() == arr[3]


def test_rename_through_guard(t1):
    arr, t = t1
    s = t.stencil("x", k=(-1, 1), fill=-7).rename(x="t")
    assert s.item(t=0, x_k=-1) == -7
    assert s.item(t=1, x_k=-1) == arr[0]


def test_charted_pad_and_stencil():
    from pdum.tl import q

    arr = np.arange(1, 6, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",)).with_charts(x=("0 nm", "25 nm"))
    p = t.pad(fill=-1, x=(q("-50 nm"), q("175 nm")))
    assert p.layout.dim("x").start == -2 and p.layout.dim("x").stop == 7
    assert p.item(x=q("-25 nm")) == -1
    assert p.item(x=q("0 nm")) == arr[0]
    p.check()

    s = t.stencil("x", k=(q("-25 nm"), q("25 nm")), fill=0)
    ck = s.layout.dim("x_k").chart
    assert ck.kind == "displacement" and ck.step == q("25 nm")
    assert s.item(x=q("0 nm"), x_k=q("-25 nm")) == 0  # boundary tap: fill
    assert s.item(x=q("50 nm"), x_k=q("25 nm")) == arr[3]
    # guard rewrites keep working with physical coordinates
    row = s.select(x=q("0 nm"))
    assert row.item(x_k=q("-25 nm")) == 0 and row.item(x_k=q("0 nm")) == arr[0]
    moved = s.shift(x=q("250 nm"))  # lattice relabel; physics glued
    assert moved.item(x=q("0 nm"), x_k=q("-25 nm")) == 0


def test_canonical_on_guarded(t1):
    arr, t = t1
    s = t.rename(x="z").stencil("z", k=(-1, 1), k_name="a", fill=-7)
    c = s.canonical()
    assert c.names == ("a", "z")  # sorted by name; addressing unchanged
    assert c.item(z=0, a=-1) == s.item(z=0, a=-1) == -7
    assert c.item(z=1, a=-1) == s.item(z=1, a=-1) == arr[0]


def test_pad_then_stencil_compose():
    arr = np.arange(1, 5, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",))
    ps = t.pad(fill=0, x=(0, 6)).stencil("x", k=(0, 1), fill=0)
    # x=3: right neighbor is padding; x=5: both taps are padding
    assert ps.item(x=3, x_k=0) == arr[3]
    assert ps.item(x=3, x_k=1) == 0
    assert ps.item(x=5, x_k=0) == 0


def test_one_fill_per_tensor(t1):
    _, t = t1
    p = t.pad(fill=0, x=(-1, 6))
    with pytest.raises(ValueError):
        p.stencil("x", k=(-1, 1), fill=9)


def test_flip_through_guard(t1):
    arr, t = t1
    p = t.pad(fill=-1, x=(-2, 7))
    f = p.flip("x")
    # storage reversed, boundary follows: old x=-1 (fill) is now x=5
    np.testing.assert_array_equal(f.to_numpy(), p.to_numpy()[::-1])
    f.check()


def test_window_through_guard(t1):
    arr, t = t1
    p = t.pad(fill=0, x=(0, 7))  # two fill slots on the right
    w = p.window("x", "k", (0, 2))
    # taps read the padding as fill, exactly like a stencil would
    assert w.item(x=3, k=1) == arr[4]
    assert w.item(x=4, k=1) == 0
    assert w.item(x=5, k=1) == 0


def test_diagonal_through_guard():
    arr = np.arange(16, dtype=np.int64).reshape(4, 4)
    t = Tensor.from_numpy(arr, ("i", "j"))
    p = t.pad(fill=-1, i=(-1, 5), j=(-1, 5))
    d = p.diagonal(("i", "j"), "z")
    assert d.item(z=-1) == -1  # outside the real region: fill
    assert d.item(z=0) == arr[0, 0]
    assert d.item(z=3) == arr[3, 3]
    # offset diagonal via shift: walks (i=z, j=z+1), falls off at the corner
    sup = p.shift(j=-1).diagonal(("i", "j"), "z")
    assert sup.item(z=0) == arr[0, 1]
    assert sup.item(z=3) == -1  # (3, 4) is padding


def test_merge_through_guard_roundtrip():
    arr = np.arange(1, 9, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",))
    s = t.stencil("x", k=(-1, 1), fill=0)
    rt = s.split("x", xo=2, xi=4).merge(("xo", "xi"), "x")
    assert rt.layout.guards == s.layout.guards
    np.testing.assert_array_equal(rt.to_numpy(), s.to_numpy())


def test_merge_through_guard_refuses_nonproportional():
    arr = np.arange(16, dtype=np.int64).reshape(4, 4)
    t = Tensor.from_numpy(arr, ("i", "j"))
    p = t.pad(fill=0, i=(-1, 5))  # guard on i alone
    with pytest.raises(ValueError):
        p.slice(i=(0, 4)).merge(("i", "j"), "f")  # guard not expressible in f


def test_decimate_through_guard(t1):
    arr, t = t1
    s = t.stencil("x", k=(-1, 1), fill=0).decimate("x", 2)
    # kept anchors are x in {0, 2, 4} renumbered to j in {0, 1, 2}
    assert s.item(x=0, x_k=-1) == 0  # tap at -1: boundary
    assert s.item(x=1, x_k=-1) == arr[1]  # anchor 2, tap at 1
    assert s.item(x=2, x_k=1) == 0  # anchor 4, tap at 5: boundary
    s.check()


def test_dilated_stencil(t1):
    arr, t = t1
    s = t.stencil("x", k=(-1, 1), fill=0, dilation=2)
    assert s.item(x=2, x_k=-1) == arr[0]  # tap at 2 - 2
    assert s.item(x=2, x_k=1) == arr[4]  # tap at 2 + 2
    assert s.item(x=1, x_k=-1) == 0  # tap at -1: boundary
    assert s.item(x=4, x_k=1) == 0  # tap at 6: boundary
    s.check()  # guard-aware footprint handles the dilated coefficient
