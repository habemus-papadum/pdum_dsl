"""Value-type expansion (200 §S.2, P4): records NEST and carry methods —
a nested record field flattens through the same aspects, addresses through
the same child walker, and its methods inline like any other."""

from dataclasses import dataclass

from pdum.dsl.cache import no_compile
from pdum.dsl.reference import reference
from pdum.dsl.surfaces import record

import pdum.dsl  # noqa: F401 — batteries
from pdum.dsl import Registry, install, jit


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


def test_records_construct_in_body_and_fields_fold():
    """Construction is a DECLARED door (surface C): RG(a, b) in a body
    builds the record; field access on an in-body construction FOLDS to
    the argument, so no record op survives lowering."""
    reg = _fresh()

    @jit()
    def f(t):
        p = Point(t * 2.0, t + 1.0)
        return p.x * p.y

    assert reg.dispatch(f, (3.0,), backend="reference") == 6.0 * 4.0


def test_unregistered_class_construction_refuses():
    """Declarations over recognition, both faces: an unregistered class in
    a closure refuses at CAPTURE (no ValueKind); an unregistered global
    name misses the vocabulary."""
    import pytest

    reg = install(Registry())

    @dataclass(frozen=True)
    class Plain:
        v: float

    def make():
        @jit()
        def f(t):
            return Plain(t).v

        return f

    with pytest.raises(Exception, match="no ValueKind registered"):
        reg.dispatch(make(), (1.0,), backend="reference")

    @jit()
    def g(t):
        return _Unregistered(t).v

    with pytest.raises(Exception, match="cannot call '_Unregistered'"):
        reg.dispatch(g, (1.0,), backend="reference")


@dataclass(frozen=True)
class _Unregistered:
    v: float


def test_with_respect_to_mirrors_the_record_type():
    """The derivative type law's record clause: d(RG)/dy is an RG,
    per-field — nested records recurse."""
    from pdum.dsl.registry import DEFAULT

    if not hasattr(DEFAULT, "_nested_records_registered"):
        record(DEFAULT, Point)
        record(DEFAULT, Segment)
        DEFAULT._nested_records_registered = True

    @jit()
    def g(y, x):
        c = Point(y * x, y + x)
        dc = with_respect_to(c, y)  # noqa: F821 — Point(d x/dy, d y/dy) = Point(x, 1)
        return dc.x + dc.y

    from pdum.dsl import DEFAULT as D

    assert D.dispatch(g, (2.0, 5.0), backend="reference") == 5.0 + 1.0
