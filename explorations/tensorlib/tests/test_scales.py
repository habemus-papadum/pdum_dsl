"""D16-D18: the diagonal chart contract, the rate combinator, and
categorical labels — the nominal rung of the measurement ladder."""

import numpy as np
import pytest
from tensorlib import Chart, Dim, Tensor, characteristic, chart, q

# ----------------------------------------------------------------------
# n-ary diagonal and the D16 chart contract
# ----------------------------------------------------------------------


def test_nary_diagonal_lattice():
    arr = np.arange(64, dtype=np.int64).reshape(4, 4, 4)
    t = Tensor.from_numpy(arr, ("i", "j", "k"))
    z = t.diagonal(("i", "j", "k"), "z")
    np.testing.assert_array_equal(z.to_numpy(), np.einsum("iii->i", arr))
    with pytest.raises(ValueError):
        t.diagonal(("i",), "z")  # needs at least two dims


def test_nary_diagonal_same_axis_combines_charts():
    arr = np.arange(24, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",)).with_charts(x=("0 nm", "1 nm"))
    z = t.split("x", a=(0, 2), b=(0, 2), c=(0, 6)).diagonal(("a", "b", "c"), "z")
    ch = z.layout.dim("z").chart
    assert ch.step == q("19 nm")  # weights 12 + 6 + 1
    assert z.item(z=q("19 nm")) == arr[19]


def test_diagonal_chart_param_forms():
    arr = np.arange(16, dtype=np.int64).reshape(4, 4)
    t = Tensor.from_numpy(arr, ("x", "t")).with_charts(x=("0 um", "1 um"), t=("0 s", "1 s"))
    assert t.diagonal(("x", "t"), "z").layout.dim("z").chart is None
    explicit = t.diagonal(("x", "t"), "z", chart=chart("0 um", "1 um", axis="x"))
    assert explicit.item(z=q("2 um")) == arr[2, 2]
    # a combinator receives the consumed dims (with their charts and domains)
    combi = t.diagonal(("x", "t"), "z", chart=lambda dims: dims[1].chart)
    assert combi.layout.dim("z").chart.step == q("1 s")
    with pytest.raises(TypeError):
        t.diagonal(("x", "t"), "z", chart=42)
    with pytest.raises(TypeError):
        t.diagonal(("x", "t"), "z", chart=lambda dims: "nonsense")


def test_characteristic_rate_diagonal():
    arr = np.arange(16, dtype=np.int64).reshape(4, 4)
    t = Tensor.from_numpy(arr, ("x", "t")).with_charts(x=("0 um", "0.5 um"), t=("0 ms", "0.25 ms"))
    xi = t.diagonal(("x", "t"), "xi", chart=characteristic(q("2 um/ms"), "x"))
    assert xi.layout.dim("xi").chart == chart("0 um", "0.5 um", axis="x")
    assert xi.item(xi=q("1 um")) == arr[2, 2]
    # the CFL-style commensurability condition is validated exactly
    with pytest.raises(ValueError):
        t.diagonal(("x", "t"), "xi", chart=characteristic(q("3 um/ms"), "x"))
    with pytest.raises(KeyError):
        t.diagonal(("x", "t"), "xi", chart=characteristic(q("2 um/ms"), "y"))


# ----------------------------------------------------------------------
# categorical labels (D18)
# ----------------------------------------------------------------------


@pytest.fixture
def rgb():
    arr = np.arange(12, dtype=np.int64).reshape(3, 4)
    t = Tensor.from_numpy(arr, ("c", "x")).with_labels(c=("R", "G", "B"))
    return arr, t.with_charts(x=("0 um", "6.5 um"))


def test_labels_both_faces(rgb):
    arr, t = rgb
    assert t.item(c="G", x=1) == t.item(c=1, x=1) == arr[1, 1]
    assert t.layout.dim("c").label(2) == "B"
    with pytest.raises(KeyError):
        t.item(c="Y", x=0)


def test_labels_select_and_slice(rgb):
    arr, t = rgb
    np.testing.assert_array_equal(t.select(c="B").to_numpy(), arr[2])
    s = t.slice(c=("G", "B"))  # half-open, like every range: keeps just G
    assert s.layout.dim("c").labels == ("G",)
    np.testing.assert_array_equal(s.to_numpy()[0], arr[1])


def test_labels_glued_under_shift_flip_decimate(rgb):
    arr, t = rgb
    sh = t.shift(c=5)
    assert sh.item(c="G", x=0) == arr[1, 0]  # names still find their data
    f = t.flip("c")
    assert f.layout.dim("c").labels == ("B", "G", "R")
    assert f.item(c="R", x=0) == arr[0, 0]  # glued through the reversal
    d = t.decimate("c", 2)
    assert d.layout.dim("c").labels == ("R", "B")
    assert d.item(c="B", x=0) == arr[2, 0]


def test_labels_refuse_arithmetic_ops(rgb):
    arr, t = rgb
    with pytest.raises(TypeError):
        t.window("c", "k", 2)
    with pytest.raises(TypeError):
        t.stencil("c", k=(0, 1), fill=0)
    with pytest.raises(TypeError):
        t.pad(fill=0, c=(-1, 4))
    with pytest.raises(TypeError):
        t.diagonal(("c", "x"), "z")
    t3 = Tensor.from_numpy(np.arange(4, dtype=np.int64), ("c",)).with_labels(c=("a", "b", "c", "d"))
    with pytest.raises(TypeError):
        t3.split("c", hi=2, lo=2)


def test_labels_validation():
    with pytest.raises(ValueError):
        Dim("c", 8, 0, 3, labels=("R", "G"))  # wrong length
    with pytest.raises(ValueError):
        Dim("c", 8, 0, 2, labels=("R", "R"))  # duplicates
    with pytest.raises(ValueError):
        Dim("c", 8, 0, 2, chart=chart("0 um", "1 um"), labels=("R", "G"))


def test_nominal_to_interval_upgrade(rgb):
    arr, t = rgb
    # "later we can replace the categorical data with a coordinate system"
    up = t.with_charts(c=("0 ms", "5 ms"))
    assert up.layout.dim("c").labels is None
    assert up.item(c=q("5 ms"), x=0) == arr[1, 0]
    # and back: attaching labels replaces the chart
    down = up.with_labels(c=("R", "G", "B"))
    assert down.layout.dim("c").chart is None
    assert down.item(c="G", x=0) == arr[1, 0]


def test_repeat_with_labels():
    arr = np.arange(4, dtype=np.int64)
    t = Tensor.from_numpy(arr, ("x",)).repeat("c", 3, labels=("R", "G", "B"))
    assert t.item(c="G", x=2) == t.item(c="B", x=2) == arr[2]  # aliases


def test_strip_charts_also_strips_labels(rgb):
    arr, t = rgb
    lat = t.strip_charts()
    with pytest.raises(TypeError):
        lat.item(c="G", x=0)
    assert lat.item(c=1, x=0) == arr[1, 0]


def test_isinstance_chart_wins_over_callable():
    # a Chart instance passed as `chart` is used verbatim, not called
    c = Chart(q("0 um"), q("1 um"), "position", "x")
    arr = np.arange(16, dtype=np.int64).reshape(4, 4)
    t = Tensor.from_numpy(arr, ("x", "t"))
    d = t.diagonal(("x", "t"), "z", chart=c)
    assert d.layout.dim("z").chart == c
