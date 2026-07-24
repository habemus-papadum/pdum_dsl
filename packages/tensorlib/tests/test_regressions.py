"""Regressions from the review pass, plus invariants the review suggested
locking in."""

import numpy as np
import pytest
from pdum.tl import GuardedLayout, Layout, Tensor, aligned, alignment, chart, q, u

# ----------------------------------------------------------------------
# defect 1: align on disjoint domains must yield empty views, not crash
# ----------------------------------------------------------------------


def test_alignment_diagnoses_disjoint_domains():
    a = Tensor.from_numpy(np.arange(3, dtype=np.int64), ("x",))
    b = Tensor.from_numpy(np.arange(3, dtype=np.int64), ("x",)).shift(x=10)
    fixes = alignment(a, b)
    assert {m.fix for m in fixes} == {"slice(x=(10, 10))"}
    assert all("empty intersection" in m.problem for m in fixes)
    va, vb = a.slice(x=(10, 10)), b.slice(x=(10, 10))
    assert aligned(va, vb)
    assert va.sizes() == vb.sizes() == {"x": 0}
    assert va.footprint() is None and va.to_numpy().shape == (0,)


def test_empty_slice_is_allowed_anywhere():
    t = Tensor.from_numpy(np.arange(4, dtype=np.int64), ("x",))
    e = t.slice(x=(10, 10))  # empty set is a subset of any domain
    assert e.numel == 0 and e.footprint() is None
    with pytest.raises(IndexError):
        t.slice(x=(10, 11))  # non-empty out-of-domain still refused


# ----------------------------------------------------------------------
# defect 2: always_fill must see a guard whose own interval is empty
# ----------------------------------------------------------------------


def test_always_fill_on_empty_guard_interval():
    t = Tensor.from_numpy(np.zeros(0, dtype=np.int32), ("x",))
    p = t.pad(7, x=(-1, 2))  # guard: 0 <= x < 0, unsatisfiable
    assert p.layout.always_fill()
    assert p.footprint() is None
    assert all(p.layout.resolve(**c) is None for c in p.layout.domain())
    np.testing.assert_array_equal(p.to_numpy(), [7, 7, 7])


# ----------------------------------------------------------------------
# defect 3: dimensionless spellings must agree
# ----------------------------------------------------------------------


def test_unit_pow_zero_is_dimensionless():
    assert (u.m**0).dims == ()
    assert q("3 m") / q("1 m") == 3  # exact Fraction
    assert (3 * u.m**0) < 4  # comparable with plain numbers


# ----------------------------------------------------------------------
# defect 4: coordinate integer policy — numpy ints yes, bool no
# ----------------------------------------------------------------------


def test_numpy_ints_are_valid_coordinates():
    arr = np.arange(8, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",))
    assert t.item(x=np.int64(3)) == arr[3]
    assert t.slice(x=(np.int32(1), np.int64(4))).sizes() == {"x": 3}
    assert t.select(x=np.int64(2)).item() == arr[2]
    assert t.shift(x=np.int64(5)).item(x=8) == arr[3]
    assert t.window("x", "k", np.int64(3)).item(x=0, k=np.int64(2)) == arr[2]
    assert np.int64(3) * u.mm == q("3 mm")


def test_bool_coordinates_are_refused():
    t = Tensor.from_numpy(np.arange(4, dtype=np.int64), ("x",))
    with pytest.raises(TypeError):
        t.item(x=True)
    with pytest.raises(TypeError):
        True * u.mm


# ----------------------------------------------------------------------
# minor: parsing
# ----------------------------------------------------------------------


def test_zero_denominator_is_a_parse_error():
    with pytest.raises(ValueError):
        q("1/0 m")


def test_unit_expressions_are_left_associative():
    assert q("1 m/s*s") == q("1 m")  # (m/s)*s, pint's convention
    assert q("1 m/s/s") == q("1 m/s**2")


def test_mixed_dimension_axis_is_rejected_at_construction():
    t = Tensor.from_numpy(np.arange(6, dtype=np.int64).reshape(2, 3), ("a", "b"))
    with pytest.raises(ValueError):
        t.with_charts(
            a=chart("0 um", "1 um", axis="shared"),
            b=chart("0 s", "1 s", "displacement", axis="shared"),
        )


# ----------------------------------------------------------------------
# invariants worth locking in (behaved correctly; now pinned)
# ----------------------------------------------------------------------


def test_guarded_footprint_contains_every_real_location():
    arr = np.arange(1, 9, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",)).with_charts(x=("0 nm", "25 nm"))
    views = [
        t.pad(fill=0, x=(-2, 10)).flip("x"),
        t.stencil("x", k=(-1, 1), fill=0, dilation=2),
        t.stencil("x", k=(-1, 1), fill=0).decimate("x", 2),
        t.pad(fill=0, x=(-1, 8)).stencil("x", k=(0, 1), fill=0).split("x", xo=3, xi=3),
    ]
    for v in views:
        lo, hi = v.layout.footprint(v.itemsize)
        for coords in v.layout.domain():
            loc = v.layout.resolve(**coords)
            if loc is not None:
                assert lo <= loc and loc + v.itemsize <= hi


def test_decimate_composes():
    arr = np.arange(36, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",))
    np.testing.assert_array_equal(
        t.decimate("x", 2, phase=1).decimate("x", 3, phase=2).to_numpy(),
        t.decimate("x", 6, phase=5).to_numpy(),  # i = 2*(3k+2)+1 = 6k+5
    )


def test_flip_is_an_involution_with_charts():
    t = Tensor.from_numpy(np.arange(6, dtype=np.int64), ("x",)).with_charts(x=("0 um", "0.25 um"))
    assert t.flip("x").flip("x").layout == t.layout


def test_alignment_trivial_cases():
    t = Tensor.from_numpy(np.arange(4, dtype=np.int64), ("x",))
    assert alignment() == () and alignment(t) == ()
    assert aligned(t, t)


def test_diagonal_same_axis_combines_charts():
    arr = np.arange(24, dtype=np.int64) * 100
    t = Tensor.from_numpy(arr, ("x",)).with_charts(x=("0 nm", "1 nm"))
    d = t.split("x", a=(0, 4), b=(0, 6)).diagonal(("a", "b"), "z")
    c = d.layout.dim("z").chart
    assert c.step == q("7 nm") and c.kind == "position" and c.axis == "x"
    assert d.item(z=q("14 nm")) == arr[14]  # label == true physical position
    # a window's anchor and kernel share the axis: z walks 2 steps per unit
    w = t.window("x", "k", (0, 2)).diagonal(("x", "k"), "z")
    assert w.layout.dim("z").chart.step == q("2 nm")
    assert w.item(z=q("2 nm")) == arr[2]
    # a dim and its mirror sum to zero step: constant position, uncharted
    anti = t.window("x", "k", (0, 2)).flip("k").diagonal(("x", "k"), "z")
    assert anti.layout.dim("z").chart is None


def test_diagonal_different_axes_is_uncharted_by_default():
    arr = np.arange(16, dtype=np.int64).reshape(4, 4)
    t = Tensor.from_numpy(arr, ("i", "j")).with_charts(i=("0 um", "6.5 um"), j=("0 um", "6.5 um"))
    d = t.diagonal(("i", "j"), "z")
    assert d.layout.dim("z").chart is None  # D16: the library never guesses
    d2 = t.diagonal(("i", "j"), "z", chart=chart("0 um", "6.5 um", axis="i"))
    assert d2.item(z=q("13 um")) == arr[2, 2]


def test_quantity_hash_contract():
    assert hash(q(3)) == hash(3)
    assert len({q(3), 3}) == 1
    assert {3: "x"}[q(3)] == "x"
    assert hash(q("1 mm")) == hash(q("1000 um"))


def test_alignment_reports_flipped_operands():
    arr = np.arange(6, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",)).with_charts(x=("0 um", "0.25 um"))
    (issue,) = alignment(t, t.flip("x"))
    assert issue.operand == 1 and issue.fix == "flip('x')"
    tb = t.flip("x").flip("x")  # the conscious fix
    assert aligned(t, tb)
    for i in range(6):
        assert tb.item(x=i) == t.item(x=i) == arr[i]


def test_is_contiguous():
    t = Tensor.from_numpy(np.arange(12, dtype=np.int64).reshape(3, 4), ("i", "j"))
    assert t.is_contiguous()
    assert not t.slice(j=(0, 2)).is_contiguous()
    assert t.split("i", a=1, b=3).is_contiguous()  # size-1 dims ignored
    assert not t.flip("j").is_contiguous()
    assert t.pad(fill=0, i=(0, 4)).is_contiguous()  # guarded: box contiguity
    assert not t.pad(fill=0, j=(0, 6)).is_contiguous()  # inner pad breaks nesting


def test_noop_pad_simplifies_away():
    t = Tensor.from_numpy(np.arange(4, dtype=np.int64), ("x",))
    p = t.pad(fill=0, x=(0, 4)).simplify()
    assert isinstance(p.layout, Layout)
    assert not isinstance(p.layout, GuardedLayout)
