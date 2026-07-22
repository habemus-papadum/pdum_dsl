"""Step 3: coordinate charts over the unchanged lattice."""

from fractions import Fraction

import numpy as np
import pytest
from tensorlib import Chart, Dim, Layout, Tensor, chart, q, u


@pytest.fixture
def tc():
    arr = np.arange(10, dtype=np.int64) * 10  # values 0, 10, ..., 90
    t = Tensor.from_numpy(arr, ("x",))
    return arr, t.with_charts(x=("0 um", "0.25 um"))


def test_chart_phys_lattice_roundtrip():
    c = chart("-1.5 um", "0.25 um")
    assert c.phys(0) == q("-1.5 um")
    assert c.phys(6) == q("0 um")
    assert c.lattice(q("0.25 um")) == 7
    for i in range(-3, 9):
        assert c.lattice(c.phys(i)) == i


def test_off_lattice_is_an_error_and_snap_is_deliberate():
    c = chart("0 um", "0.25 um")
    with pytest.raises(ValueError):
        c.lattice(q("0.3 um"))
    assert c.snap(q("0.3 um"), "nearest") == 1
    assert c.snap(q("0.3 um"), "floor") == 1
    assert c.snap(q("0.3 um"), "ceil") == 2


def test_commensurable():
    a = chart("0 um", "0.25 um")
    assert a.commensurable(chart("0.5 um", "0.75 um"))
    assert not a.commensurable(chart("0.125 um", "0.25 um"))
    assert not a.commensurable(chart("0 s", "0.25 s"))


def test_both_faces_of_indexing(tc):
    arr, t = tc
    assert t.item(x=3) == t.item(x=q("0.75 um")) == arr[3]
    assert t.phys("x", 3) == q("0.75 um")


def test_ints_always_mean_lattice(tc):
    arr, t = tc
    # compiler mode: plain ints work identically with or without charts
    assert t.strip_charts().item(x=3) == t.item(x=3)
    with pytest.raises(TypeError):
        t.strip_charts().item(x=q("0.75 um"))


def test_physical_slice_matches_lattice_slice(tc):
    arr, t = tc
    a = t.slice(x=(q("0.25 um"), q("1 um")))
    b = t.slice(x=(1, 4))
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    # coordinates stay natural: same physical label addresses the same datum
    assert a.item(x=q("0.5 um")) == arr[2]


def test_select_by_quantity(tc):
    arr, t = tc
    assert t.select(x=q("0.5 um")).item() == arr[2]


def test_shift_keeps_physics_glued_to_data(tc):
    arr, t = tc
    s = t.shift(x=4)
    assert s.item(x=4) == arr[0]  # lattice labels moved...
    assert s.phys("x", 4) == q("0 um")  # ...physical labels followed the data
    assert s.item(x=q("0.75 um")) == t.item(x=q("0.75 um"))
    # a physical delta shifts by delta/step lattice steps
    s2 = t.shift(x=q("0.5 um"))
    assert s2.layout.dim("x").start == 2
    with pytest.raises(ValueError):
        t.shift(x=q("0.3 um"))  # not a whole number of steps


def test_recenter_moves_the_frame_not_the_data(tc):
    arr, t = tc
    r = t.recenter(x=q("10 um"))
    assert r.layout.dim("x").start == 0  # lattice untouched
    assert r.phys("x", 0) == q("10 um")  # labels moved
    assert r.item(x=0) == arr[0]  # data untouched


def test_flip_keeps_physics_glued_to_data(tc):
    arr, t = tc
    f = t.flip("x")
    c = f.layout.dim("x").chart
    assert c.step == -q("0.25 um")
    # each datum keeps its physical label: lattice moves, physics doesn't
    assert f.item(x=q("0.25 um")) == t.item(x=q("0.25 um")) == arr[1]


def test_split_derives_block_and_offset_charts():
    arr = np.arange(16, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",)).with_charts(x=("0 um", "0.25 um"))
    b = t.split("x", xb=4, xi=4)
    cb, ci = b.layout.dim("xb").chart, b.layout.dim("xi").chart
    assert cb.kind == "position" and cb.step == q("1 um") and cb.origin == q("0 um")
    assert ci.kind == "displacement" and ci.step == q("0.25 um") and ci.origin == q("0 um")
    # physical block position + within-block displacement address naturally
    assert b.item(xb=q("2 um"), xi=q("0.75 um")) == arr[11]


def test_merge_restores_the_chart():
    arr = np.arange(16, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",)).with_charts(x=("0 um", "0.25 um"))
    m = t.split("x", xb=4, xi=4).merge(("xb", "xi"), "x")
    assert m.layout.dim("x").chart == t.layout.dim("x").chart
    np.testing.assert_array_equal(m.to_numpy(), arr)


def test_merge_rejects_mixed_charting():
    arr = np.arange(16, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",)).with_charts(x=("0 um", "0.25 um"))
    b = t.split("x", xb=4, xi=4).with_charts(xi=None)
    with pytest.raises(ValueError):
        b.merge(("xb", "xi"), "x")


def test_window_kernel_gets_displacement_chart():
    arr = np.arange(6, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",)).with_charts(x=("0 ms", "1 ms"))
    w = t.window("x", "k", (q("-1 ms"), q("2 ms")))
    ck = w.layout.dim("k").chart
    assert ck.kind == "displacement" and ck.step == q("1 ms") and ck.origin == q("0 ms")
    assert w.item(x=q("2 ms"), k=q("-1 ms")) == arr[1]


def test_diagonal_default_uncharted_across_axes():
    arr = np.arange(16, dtype=np.int64).reshape(4, 4)
    t = Tensor.from_numpy(arr, ("i", "j")).with_charts(i=("0 um", "6.5 um"), j=("0 um", "6.5 um"))
    d = t.diagonal(("i", "j"), "z")
    assert d.layout.dim("z").chart is None  # D16: no guessing across axes
    d2 = t.diagonal(("i", "j"), "z", chart=chart("0 um", "6.5 um", axis="i"))
    assert d2.item(z=q("13 um")) == arr[2, 2]


def test_canonical_sees_through_unit_spelling():
    a = Layout((Dim("x", 8, 0, 4, chart("1000 nm", "250 nm")),))
    b = Layout((Dim("x", 8, 0, 4, chart("1 um", "0.25 um")),))
    assert a.canonical() == b.canonical()


def test_chart_construction_rejects_nonsense():
    with pytest.raises(ValueError):
        Chart(q("0 um"), q("0 um"))  # zero step
    with pytest.raises(ValueError):
        Chart(q("0 um"), q("1 s"))  # mixed dimensions
    with pytest.raises(TypeError):
        chart(0.0, "1 um")  # float origin


def test_value_units_metadata_threads_through_field():
    reg_v = u.define("V_test", dim="voltage_test")
    dt = np.dtype([("v", "<f4"), ("flags", "u1")], align=True)
    rec = np.zeros(3, dt)
    rec["v"] = [1.5, 2.5, 3.5]
    t = Tensor.from_numpy(rec, ("t",)).with_value_units({"v": reg_v})
    fv = t.field("v")
    assert fv.value_units == reg_v
    assert t.field("flags").value_units is None
    assert fv.item(t=1) == np.float32(2.5)  # raw machine number, not a Quantity


def test_exactness_survives_fraction_steps():
    # a step of 1/3 um: no fixed-point denominator, still exact
    arr = np.arange(7, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",)).with_charts(x=("0 um", Fraction(1, 3) * u.um))
    assert t.item(x=q("2 um")) == arr[6]
    with pytest.raises(ValueError):
        t.item(x=q("0.5 um"))
