"""Step 1: the affine + box core, at the Layout level (no buffers)."""

import pytest
from pdum.tl import Dim, Injectivity, Layout


def make():
    # x: stride 8, domain [2, 5); y: stride -4, domain [-1, 3); offset 100
    return Layout((Dim("x", 8, 2, 5), Dim("y", -4, -1, 3)), offset=100)


def test_get_loc_is_raw_affine():
    lay = make()
    assert lay.get_loc(x=3, y=0) == 100 + 24
    assert lay.get_loc(x=2, y=-1) == 100 + 16 + 4
    assert lay.get_loc(x=4, y=2) == 100 + 32 - 8


def test_domain_errors():
    lay = make()
    with pytest.raises(IndexError):
        lay.get_loc(x=5, y=0)  # out of [2, 5)
    with pytest.raises(KeyError):
        lay.get_loc(x=3)  # missing y
    with pytest.raises(KeyError):
        lay.get_loc(x=3, y=0, z=0)  # extra name


def test_sizes_and_numel():
    lay = make()
    assert lay.sizes() == {"x": 3, "y": 4}
    assert lay.numel == 12


def test_footprint_with_negative_stride():
    lay = make()
    # x contributes [16, 32]; y (stride -4 over [-1, 2]) contributes [-8, 4]
    assert lay.footprint() == (100 + 16 - 8, 100 + 32 + 4 + 1)


def test_footprint_empty_domain():
    lay = Layout((Dim("x", 8, 2, 2),))
    assert lay.numel == 0
    assert lay.footprint() is None


def test_injectivity_three_ways():
    dense = Layout((Dim("x", 1, 0, 4), Dim("y", 4, 0, 3)))
    assert dense.injectivity() is Injectivity.INJECTIVE
    aliased = Layout((Dim("x", 1, 0, 4), Dim("r", 0, 0, 3)))
    assert aliased.injectivity() is Injectivity.ALIASED
    murky = Layout((Dim("x", 1, 0, 2), Dim("y", 1, 0, 2)))
    assert murky.injectivity() is Injectivity.UNKNOWN


def test_flip_addresses_same_memory_reversed():
    lay = Layout((Dim("x", 8, 0, 4),))
    flipped = lay.flip("x")
    assert flipped.dim("x").start == 0 and flipped.dim("x").stop == 4
    for i in range(4):
        assert flipped.get_loc(x=i) == lay.get_loc(x=3 - i)
    assert flipped.footprint(8) == lay.footprint(8)


def test_shift_relabels_without_moving():
    lay = make()
    moved = lay.shift(x=10)
    assert moved.dim("x").start == 12 and moved.dim("x").stop == 15
    for i in range(2, 5):
        for j in range(-1, 3):
            assert moved.get_loc(x=i + 10, y=j) == lay.get_loc(x=i, y=j)


def test_slice_touches_only_the_domain():
    lay = make()
    sub = lay.slice(x=(3, 5))
    assert sub.offset == lay.offset
    assert sub.dim("x").stride == 8
    assert sub.get_loc(x=3, y=0) == lay.get_loc(x=3, y=0)  # same coordinate name
    with pytest.raises(IndexError):
        lay.slice(x=(0, 5))  # not a sub-range
