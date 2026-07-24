"""Value-type expansion (200 §S.2, P4): records NEST and carry methods —
a nested record field flattens through the same aspects, addresses through
the same child walker, and its methods inline like any other."""

from dataclasses import dataclass

import pdum.dsl  # noqa: F401 — batteries
from pdum.dsl import Registry, install, jit
from pdum.dsl.cache import no_compile
from pdum.dsl.reference import reference
from pdum.dsl.surfaces import record


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def dot(self, other):
        return self.x * other.x + self.y * other.y


@dataclass(frozen=True)
class Segment:
    a: Point
    b: Point
    weight: float

    def stretch(self):
        return self.b.x - self.a.x


def _fresh() -> Registry:
    reg = install(Registry())
    record(reg, Point)
    record(reg, Segment)
    return reg


def test_nested_record_type_nests():
    _fresh()
    rec = Segment.__dsl_record__
    assert [n for n, _ in rec.fields] == ["a", "b", "weight"]
    assert rec.fields[0][1] is Point.__dsl_record__  # the field IS the nested Record type


def test_nested_fields_compute_through_the_reference(sample=None):
    reg = _fresh()

    def make(s):
        @jit()
        def f(t):
            return (s.b.x - s.a.x) * t + s.weight

        return f

    s = Segment(Point(1.0, 2.0), Point(4.0, 6.0), 0.5)
    f = make(s)
    assert reg.dispatch(f, (2.0,), backend="reference") == (4.0 - 1.0) * 2.0 + 0.5
    with no_compile():  # new VALUES, same nested Record type: a warm hit
        s2 = Segment(Point(0.0, 0.0), Point(1.0, 1.0), 9.0)
        assert reg.dispatch(make(s2), (1.0,), backend="reference") == 1.0 + 9.0


def test_methods_on_nested_records_inline():
    reg = _fresh()

    def make(s):
        @jit()
        def f(t):
            return s.stretch() * t + s.a.dot(s.b)

        return f

    s = Segment(Point(1.0, 2.0), Point(3.0, 5.0), 0.0)
    got = reg.dispatch(make(s), (10.0,), backend="reference")
    assert got == (3.0 - 1.0) * 10.0 + (1.0 * 3.0 + 2.0 * 5.0)


def test_nested_record_flattens_leaf_per_scalar_field():
    reg = _fresh()
    s = Segment(Point(1.0, 2.0), Point(3.0, 4.0), 5.0)
    assert reg.table.flatten(s) == (1.0, 2.0, 3.0, 4.0, 5.0)  # depth-first field order


def test_unregistered_nested_class_refuses_naming_the_fix():
    import pytest

    reg = install(Registry())

    @dataclass(frozen=True)
    class Inner:
        v: float

    @dataclass(frozen=True)
    class Outer:
        i: Inner

    with pytest.raises(TypeError, match="register the nested record first"):
        record(reg, Outer)


def test_reference_evaluates_via_dispatch():
    """The spelled door works for nested-record captures on DEFAULT too."""
    from pdum.dsl import DEFAULT

    if not hasattr(DEFAULT, "_nested_records_registered"):
        record(DEFAULT, Point)
        record(DEFAULT, Segment)
        DEFAULT._nested_records_registered = True

    def make(s):
        @jit()
        def f(t):
            return s.a.x + s.b.y * t

        return f

    s = Segment(Point(7.0, 8.0), Point(9.0, 10.0), 0.0)
    assert reference(make(s))(2.0) == 7.0 + 10.0 * 2.0
