"""The coordinate algebra (design 250, stage 1): torsor laws, every
refusal, the chart/label doors, exact-only conversion, and promotion at
operator boundaries. Nothing consumes these objects yet — this battery IS
the contract."""

from fractions import Fraction

import pytest

from pdum.tl import Coordinate, Displacement, Frame, Slice, chart, extent, q

Y = Frame("y", 0, 512)
X = Frame("x", 0, 512)
YC = Frame("y", 0, 512, chart=chart("0 um", "0.25 um"))
CHAN = Frame("chan", 0, 3, labels=("red", "green", "blue"))


# ---------------------------------------------------------------------------
# Frame: construction and validation
# ---------------------------------------------------------------------------


def test_frame_validation():
    with pytest.raises(ValueError, match="stop .* < start"):
        Frame("y", 4, 2)
    with pytest.raises(ValueError, match="chart or labels, not both"):
        Frame("y", 0, 2, chart=chart("0 um", "1 um"), labels=("a", "b"))
    with pytest.raises(ValueError, match="labels for"):
        Frame("y", 0, 3, labels=("a", "b"))
    with pytest.raises(ValueError, match="unique"):
        Frame("y", 0, 2, labels=("a", "a"))
    with pytest.raises(ValueError, match="chartless and unlabeled"):
        Frame("y", 0, 2, chart=chart("0 um", "1 um"), level="block")
    assert Y.size == 512
    assert Y.contains(0) and Y.contains(511) and not Y.contains(512)


def test_frame_is_hashable_identity():
    assert Y == Frame("y", 0, 512)
    assert Y != X
    assert Y != YC  # a chart is part of the observable frame
    assert len({Y, Frame("y", 0, 512), X, YC}) == 3


# ---------------------------------------------------------------------------
# The point factory: one door per labeling
# ---------------------------------------------------------------------------


def test_point_doors():
    assert Y[128] == Coordinate(Y, 128)
    assert YC[q("1.5 um")].i == 6  # chart door, exact-only
    assert CHAN["green"].i == 1  # label door
    negative = Frame("y", -4, 4)
    assert negative[-4].i == -4  # a literal coordinate, never "from the end"


def test_point_door_refusals():
    with pytest.raises(IndexError, match="no out-of-bounds Coordinate"):
        Y[512]
    with pytest.raises(TypeError, match="bool is not a lattice value"):
        Y[True]
    with pytest.raises(ValueError, match="off-lattice"):
        YC[q("0.3 um")]  # chart.lattice's own refusal, naming snap
    with pytest.raises(TypeError, match="has no chart"):
        Y[q("1 um")]
    with pytest.raises(TypeError, match="has no labels"):
        Y["red"]
    with pytest.raises(KeyError, match="not a label"):
        CHAN["cyan"]
    with pytest.raises(TypeError, match="must be int, Quantity, or label"):
        Y[1.5]


def test_frame_handle_is_point_only():
    with pytest.raises(TypeError, match="points only"):
        Y[128:256]


def test_snap_is_the_deliberate_rounding_door():
    assert YC.snap(q("0.3 um")) == YC[1]
    assert YC.snap(q("0.3 um"), "ceil") == YC[2]
    with pytest.raises(TypeError, match="no chart to snap"):
        Y.snap(q("0.3 um"))


# ---------------------------------------------------------------------------
# Coordinate: readings and identity
# ---------------------------------------------------------------------------


def test_coordinate_readings():
    assert YC[6].phys == q("1.5 um")
    assert CHAN[2].label == "blue"
    with pytest.raises(TypeError, match="has no chart"):
        Y[3].phys
    with pytest.raises(TypeError, match="has no labels"):
        Y[3].label


def test_coordinate_equality_is_structural():
    assert Y[3] == Coordinate(Y, 3)
    assert Y[3] != YC[3]  # different frames: different points
    assert Y[3] != 3  # a point is not a number (Python eq convention: False)
    assert len({Y[3], Coordinate(Y, 3), Y[4]}) == 2


def test_coordinate_ordering_same_frame_only():
    assert Y[3] < Y[7] <= Y[7]
    with pytest.raises(TypeError, match="cross-frame"):
        Y[3] < X[7]
    with pytest.raises(TypeError):
        Y[3] < 7  # ordering against numbers is unsupported


# ---------------------------------------------------------------------------
# The torsor law
# ---------------------------------------------------------------------------


def test_point_minus_point_is_a_displacement():
    d = Y[256] - Y[128]
    assert d == Displacement(Y, 128)
    assert Y[128] - Y[256] == Displacement(Y, -128)


def test_point_plus_displacement_is_a_point():
    d = Displacement(Y, 5)
    assert Y[100] + d == Y[105]
    assert d + Y[100] == Y[105]
    assert Y[100] - d == Y[95]


def test_ints_and_quantities_promote_to_displacements():
    assert Y[100] + 5 == Y[105]
    assert 5 + Y[100] == Y[105]
    assert Y[100] - 5 == Y[95]
    assert YC[4] + q("0.5 um") == YC[6]  # 2 exact steps
    assert q("0.5 um") + YC[4] == YC[6]
    assert YC[4] - q("0.25 um") == YC[3]
    assert Y[100] + Fraction(2, 1) == Y[102]


def test_result_bounds_are_rechecked():
    with pytest.raises(IndexError, match="no out-of-bounds Coordinate"):
        Y[511] + 1
    with pytest.raises(IndexError, match="no out-of-bounds Coordinate"):
        Y[0] - 1


def test_adding_points_is_the_affine_crime():
    with pytest.raises(TypeError, match="affine crime"):
        Y[1] + Y[2]


def test_cross_frame_arithmetic_refuses():
    with pytest.raises(TypeError, match="cross-frame"):
        Y[3] - X[1]
    with pytest.raises(TypeError, match="cross-frame"):
        Y[3] + Displacement(X, 1)
    with pytest.raises(TypeError, match="cross-frame"):
        Y[3] - YC[3]  # same name, different frame: still cross-frame


def test_exactness_of_quantity_displacements():
    with pytest.raises(ValueError, match="not a whole number of steps"):
        YC[4] + q("0.3 um")
    with pytest.raises(ValueError, match="wrong dimensions"):
        YC[4] + q("1 s")
    with pytest.raises(TypeError, match="has no chart"):
        Y[4] + q("1 um")
    with pytest.raises(ValueError, match="not a whole number"):
        Y[4] + Fraction(1, 2)


def test_numeric_use_of_a_point_refuses_toward_coercion():
    for op in (
        lambda: Y[3] * 2,
        lambda: 2 * Y[3],
        lambda: Y[3] / 2,
        lambda: 2 / Y[3],
        lambda: Y[3] // 2,
        lambda: Y[3] % 2,
        lambda: Y[3] ** 2,
        lambda: -Y[3],
        lambda: +Y[3],
    ):
        with pytest.raises(TypeError, match="a point is not a number"):
            op()
    with pytest.raises(TypeError, match="whole lattice steps only"):
        Y[3] + 0.5
    with pytest.raises(TypeError, match="a point is not a number"):
        2 - Y[3]


# ---------------------------------------------------------------------------
# Displacement: the vector space (ℤ-module on the lattice)
# ---------------------------------------------------------------------------


def test_displacement_algebra():
    a, b = Displacement(Y, 3), Displacement(Y, 4)
    assert a + b == Displacement(Y, 7)
    assert a - b == Displacement(Y, -1)
    assert -a == Displacement(Y, -3)
    assert 2 * a == a * 2 == Displacement(Y, 6)
    assert a + 1 == Displacement(Y, 4)
    assert 1 + a == Displacement(Y, 4)
    assert 10 - a == Displacement(Y, 7)


def test_displacement_quantity_boundary():
    d = Displacement(YC, 2)
    assert d + q("0.25 um") == Displacement(YC, 3)
    assert d.phys == q("0.5 um")
    with pytest.raises(TypeError, match="has no chart"):
        Displacement(Y, 2).phys


def test_displacement_refusals():
    with pytest.raises(TypeError, match="cross-frame"):
        Displacement(Y, 1) + Displacement(X, 1)
    with pytest.raises(TypeError, match="only integers scale"):
        Displacement(Y, 3) * 0.5
    with pytest.raises(TypeError, match="whole lattice steps only"):
        Displacement(Y, 3) + 0.5
    with pytest.raises(TypeError, match="bool is not a lattice value"):
        Displacement(Y, 3) * True


def test_displacements_are_unbounded():
    # The domain bounds points, not differences.
    assert Displacement(Y, 10_000) + Displacement(Y, -20_000) == Displacement(Y, -10_000)


# ---------------------------------------------------------------------------
# Slice: half-open forward progressions
# ---------------------------------------------------------------------------


def test_slice_basics():
    s = Slice(Y[128], Y[256])
    assert (s.start_i, s.stop_i, s.step_k) == (128, 256, 1)
    assert s.size == 128
    assert s.frame is Y


def test_slice_step_promotion():
    assert Slice(Y[0], Y[10], 2).step == Displacement(Y, 2)
    assert Slice(YC[0], YC[8], q("0.5 um")).step == Displacement(YC, 2)
    assert Slice(Y[0], Y[10], Displacement(Y, 3)).size == 4  # 0,3,6,9


def test_slice_to_the_frame_end_is_spelled_by_omission():
    s = Slice(Y[128])
    assert s.stop is None and s.stop_i == 512
    assert s.size == 384
    # The frame-end exclusive endpoint has no point:
    with pytest.raises(IndexError, match="no out-of-bounds Coordinate"):
        Slice(Y[128], Y[512])


def test_empty_slice_is_legal():
    assert Slice(Y[7], Y[7]).size == 0


def test_slice_refusals():
    with pytest.raises(ValueError, match="forward progression"):
        Slice(Y[0], Y[10], -1)
    with pytest.raises(ValueError, match="step must be positive"):
        Slice(Y[0], Y[10], 0)
    with pytest.raises(ValueError, match="empty-inverted"):
        Slice(Y[10], Y[2])
    with pytest.raises(TypeError, match="cross-frame"):
        Slice(Y[0], X[10])
    with pytest.raises(TypeError, match="cross-frame"):
        Slice(Y[0], Y[10], Displacement(X, 1))
    with pytest.raises(TypeError, match="start must be a Coordinate"):
        Slice(3, Y[10])
    with pytest.raises(TypeError, match="stop must be a Coordinate"):
        Slice(Y[0], 10)


def test_slice_size_rounds_up():
    assert Slice(Y[0], Y[10], 3).size == 4  # 0,3,6,9
    assert Slice(Y[0], Y[9], 3).size == 3  # 0,3,6
    assert Slice(Y[0], Y[1], 5).size == 1


# ---------------------------------------------------------------------------
# Reprs read like the spellings
# ---------------------------------------------------------------------------


def test_reprs():
    assert repr(Y[128]) == "y[128]"
    assert repr(Y[256] - Y[128]) == "y[+128]"
    assert repr(Y[128] - Y[256]) == "y[-128]"
    assert repr(Slice(Y[128], Y[256], 2)) == "y[128:256:2]"
    assert repr(Slice(Y[128])) == "y[128:]"
    assert repr(Frame("y", 0, 4, labels=("a", "b", "c", "d"))) == "y[0:4) #[a,b,c,d]"


def test_extent_reads_the_frame_width():
    assert extent(Y[128]) == 512  # a host int — promote explicitly to join float math
    with pytest.raises(TypeError, match="wants a Coordinate"):
        extent(3)
