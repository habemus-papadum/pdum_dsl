"""Step 3b: axis identity — the per-axis label-sum invariant and the ops
built on it (select compensation, decimate, dilation, align)."""

import numpy as np
import pytest
from pdum.tl import Dim, Layout, Tensor, aligned, alignment, chart, q


@pytest.fixture
def tc():
    arr = np.arange(8, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",))
    return arr, t.with_charts(x=("0 um", "0.25 um"))


def test_axis_defaults_to_dim_name(tc):
    _, t = tc
    assert t.layout.dim("x").chart.axis == "x"


def test_split_parts_share_the_axis(tc):
    _, t = tc
    b = t.split("x", xc=4, xp=2)
    assert b.layout.dim("xc").chart.axis == "x"
    assert b.layout.dim("xp").chart.axis == "x"


def test_one_position_per_axis_is_enforced():
    with pytest.raises(ValueError):
        Layout(
            (
                Dim("a", 8, 0, 4, chart("0 um", "1 um", axis="x")),
                Dim("b", 4, 0, 2, chart("0 um", "1 um", axis="x")),
            )
        )


def test_select_phase_compensates_the_position_sibling(tc):
    arr, t = tc
    b = t.split("x", xc=4, xp=2)
    dec = b.select(xp=q("0.25 um"))
    c = dec.layout.dim("xc").chart
    assert c.kind == "position" and c.origin == q("0.25 um") and c.step == q("0.5 um")
    # labels now tell the truth: lattice 1 holds arr[3] at 0.75 um
    assert dec.item(xc=1) == arr[3]
    assert dec.phys("xc", 1) == q("0.75 um")
    assert dec.item(xc=q("0.75 um")) == arr[3]


def test_select_block_promotes_the_displacement(tc):
    arr, t = tc
    b = t.split("x", xc=4, xp=2)
    blk = b.select(xc=q("1 um"))  # extract block 2 (elements 4, 5)
    c = blk.layout.dim("xp").chart
    assert c.kind == "position" and c.origin == q("1 um") and c.step == q("0.25 um")
    assert blk.item(xp=q("1.25 um")) == arr[5]  # absolute in-block positions


def test_select_whole_axis_collapses(tc):
    arr, t = tc
    b = t.split("x", xc=4, xp=2)
    scalar = b.select(xc=2, xp=1)
    assert scalar.layout.dims == ()
    assert scalar.item() == arr[5]


def test_select_window_tap_keeps_labels_glued(tc):
    arr, t = tc
    w = t.window("x", "k", (-1, 2))
    right = w.select(k=q("0.25 um"))  # "right neighbor" view
    assert right.phys("x", 1) == q("0.5 um")  # label = position of the datum read
    assert right.item(x=q("0.5 um")) == arr[2]


def test_decimate_matches_split_select(tc):
    arr, t = tc
    dec = t.decimate("x", 2, phase=1)
    via_split = t.split("x", xc=4, xp=2).select(xp=1).rename(xc="x")
    assert dec.layout.dim("x").chart == via_split.layout.dim("x").chart
    np.testing.assert_array_equal(dec.to_numpy(), via_split.to_numpy())
    np.testing.assert_array_equal(dec.to_numpy(), arr[1::2])


def test_decimate_lattice_and_values(tc):
    arr, t = tc
    np.testing.assert_array_equal(t.decimate("x", 2).to_numpy(), arr[::2])
    np.testing.assert_array_equal(t.decimate("x", 3, phase=2).to_numpy(), arr[2::3])
    # phase as a physical displacement
    np.testing.assert_array_equal(t.decimate("x", 2, phase=q("0.25 um")).to_numpy(), arr[1::2])


def test_decimate_non_divisible_and_negative_start():
    arr = np.arange(7, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",))
    np.testing.assert_array_equal(t.decimate("x", 2).to_numpy(), arr[::2])
    sh = t.shift(x=-3)  # domain [-3, 4)
    d = sh.decimate("x", 2)  # keeps even lattice coords {-2, 0, 2}
    assert (d.layout.dim("x").start, d.layout.dim("x").stop) == (-1, 2)
    np.testing.assert_array_equal(d.to_numpy(), arr[1::2])


def test_dilated_window(tc):
    arr, t = tc
    w = t.window("x", "k", (-1, 2), dilation=2)  # taps at -2, 0, +2 steps
    assert w.layout.dim("x").start == 2 and w.layout.dim("x").stop == 6
    assert w.item(x=2, k=-1) == arr[0]
    assert w.item(x=2, k=1) == arr[4]
    assert w.layout.dim("k").chart.step == q("0.5 um")  # dilated tap pitch
    # physical tap spec uses the dilated pitch
    assert w.item(x=3, k=q("-0.5 um")) == arr[1]


def test_alignment_diagnoses_offset_grids_and_recipes_work():
    a_arr = np.arange(8, dtype=np.int64)
    b_arr = np.arange(8, dtype=np.int64) * 10
    ta = Tensor.from_numpy(a_arr, ("x",)).with_charts(x=("0 um", "0.25 um"))
    tb = Tensor.from_numpy(b_arr, ("x",)).with_charts(x=("0.5 um", "0.25 um"))
    (issue,) = alignment(ta, tb)
    assert issue.operand == 1 and issue.fix == "shift(x=2)"
    tb2 = tb.shift(x=2)  # the caller applies the recipe consciously
    fixes = alignment(ta, tb2)
    assert len(fixes) == 2 and {m.fix for m in fixes} == {"slice(x=(2, 8))"}
    va, vb = ta.slice(x=(2, 8)), tb2.slice(x=(2, 8))
    assert aligned(va, vb)
    for i in range(2, 8):
        pos = va.phys("x", i)
        assert va.item(x=pos) == a_arr[i] and vb.item(x=pos) == b_arr[i - 2]


def test_alignment_reports_missing_dims():
    row = Tensor.from_numpy(np.arange(4, dtype=np.int64), ("j",))
    img = Tensor.from_numpy(np.arange(12, dtype=np.int64).reshape(3, 4), ("i", "j"))
    (issue,) = alignment(row, img)
    assert issue.operand == 0 and issue.dim == "i"
    assert issue.fix == "repeat('i', (0, 3))"
    row2 = row.repeat("i", (0, 3))
    assert aligned(row2, img)
    assert row2.item(i=2, j=3) == row.item(j=3)  # broadcast alias


def test_alignment_uncharted_uses_raw_lattice():
    a = Tensor.from_numpy(np.arange(8, dtype=np.int64), ("x",)).shift(x=2)
    b = Tensor.from_numpy(np.arange(8, dtype=np.int64), ("x",))
    fixes = alignment(a, b)
    assert {m.fix for m in fixes} == {"slice(x=(2, 8))"}
    assert aligned(a.slice(x=(2, 8)), b.slice(x=(2, 8)))


def test_alignment_reports_incompatibilities_without_fixes():
    a = Tensor.from_numpy(np.arange(4, dtype=np.int64), ("x",)).with_charts(x=("0 um", "0.25 um"))
    b = Tensor.from_numpy(np.arange(4, dtype=np.int64), ("x",)).with_charts(x=("0 um", "0.5 um"))
    (issue,) = alignment(a, b)
    assert "resampling is not a view" in issue.problem
    assert "decimate('x', 2" in issue.fix  # conscious decimating alignment hint
    c = Tensor.from_numpy(np.arange(4, dtype=np.int64), ("x",)).with_charts(x=("0.1 um", "0.25 um"))
    (issue,) = alignment(a, c)
    assert "non-integer" in issue.problem and issue.fix == ""


def test_alignment_works_through_guards():
    arr = np.arange(1, 6, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",)).with_charts(x=("0 um", "0.25 um"))
    p = t.pad(fill=0, x=(q("-0.5 um"), q("1.5 um")))  # box [-2, 6)
    plain = Tensor.from_numpy(np.arange(6, dtype=np.int64), ("x",)).with_charts(x=("-0.5 um", "0.25 um"))
    (issue,) = alignment(p, plain)
    assert issue.operand == 1 and issue.fix == "shift(x=-2)"
    plain2 = plain.shift(x=-2)  # domain [-2, 4)
    fixes = alignment(p, plain2)
    assert {m.fix for m in fixes} == {"slice(x=(-2, 4))"}
    vp = p.slice(x=(-2, 4))
    assert aligned(vp, plain2)
    assert vp.item(x=q("-0.25 um")) == 0  # still fill through the recipes
