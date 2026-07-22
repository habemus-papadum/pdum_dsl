"""Step 3: the exact unit/quantity system."""

from fractions import Fraction

import pytest
from tensorlib import Quantity, UnitRegistry, q, u


def test_construction_forms_are_exact():
    assert q("0.75 um").magnitude == Fraction(3, 4)
    assert q("3/4 um") == q("0.75 um")
    assert 3 * u.mm == q("3 mm")
    assert Fraction(3, 4) * u.um == q("0.75 um")
    assert q("1e-3 m") == q("1 mm")
    assert q("0.1 um").magnitude == Fraction(1, 10)  # exact decimal, not float


def test_t_string_construction_is_exact():
    val = Fraction(1, 3)
    assert q(t"{val} m").magnitude == Fraction(1, 3)
    assert q(t"{5} mm") == 5 * u.mm


def test_floats_are_rejected_everywhere():
    with pytest.raises(TypeError):
        0.1 * u.um
    with pytest.raises(TypeError):
        q("1 mm") * 0.5
    with pytest.raises(TypeError):
        bad = 0.1
        q(t"{bad} um")


def test_exact_arithmetic_across_units():
    assert q("1 mm") + q("250 um") == q("1.25 mm")
    assert q("1 mm") - q("250 um") == q("750 um")
    assert 2 * q("0.5 um") == q("1 um")
    assert -q("3 nm") == q("-3 nm")


def test_division_of_like_quantities_is_an_exact_ratio():
    r = q("1 mm") / q("250 um")
    assert isinstance(r, Fraction)
    assert r == 4


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        q("1 mm") + q("1 s")
    with pytest.raises(ValueError):
        q("1 mm") < q("1 s")
    assert q("1 mm") != q("1 s")  # equality is total, just False


def test_comparisons_convert_exactly():
    assert q("999 um") < q("1 mm")
    assert q("1000 um") <= q("1 mm")
    assert q("1 mm") == q("1000 um")


def test_compound_units():
    v = q("3 m") / q("2 s")
    assert v == q("1.5 m/s")
    assert q("2 Hz") == q("2 s**-1")


def test_to_converts_display_unit_exactly():
    assert repr(q("1 mm").to(u.um)) == "1000 um"
    with pytest.raises(ValueError):
        q("1 mm").to(u.s)


def test_registry_define():
    reg = UnitRegistry()
    reg.define("px", dim="pixel")
    assert (3 * reg.px).dims == (("pixel", 1),)
    with pytest.raises(ValueError):
        reg.quantity("1 px") + reg.quantity("1 m")
    reg.define("tick", "1/60 s")
    assert reg.quantity("120 tick") == reg.quantity("2 s")


def test_repr_prefers_exact_decimal():
    assert repr(q("0.25 um")) == "0.25 um"
    assert repr(q("1/3 m")) == "1/3 m"
    assert repr(q("-0.5 ms")) == "-0.5 ms"


def test_quantity_is_hashable_semantically():
    assert hash(q("1 mm")) == hash(q("1000 um"))
    assert len({q("1 mm"), q("1000 um")}) == 1
    assert isinstance(q("1 mm"), Quantity)
