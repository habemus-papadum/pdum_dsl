"""M0.1 — capture + types + cache. The central acceptance test: ``closure(5)``
and ``closure(6)`` share one ``FnType`` and trigger exactly one compile."""

import pytest
from pdum.dsl_reference import (
    BoolType,
    FloatType,
    IntType,
    SpecCache,
    TupleType,
    bump_generation,
    jit,
    typeof,
)
from pdum.dsl_reference.types import BigIntError, f64, i64


def make_closure(x):
    """Mirrors docs/closure_specialization.md: an inner closing over ``x``."""

    @jit(kind="device")
    def inner(y):
        return x + y

    return inner


# --- typeof -----------------------------------------------------------------


def test_typeof_scalars_and_int_bucketing():
    assert typeof(True) == BoolType()  # bool checked before int
    assert typeof(5) == IntType(64, True)
    assert typeof(2**63) == IntType(64, False)  # overflows i64 -> u64
    assert typeof(1.5) == FloatType(64)
    with pytest.raises(BigIntError):
        typeof(2**70)


def test_typeof_tuple_is_elementwise_with_arity():
    assert typeof((1, 2.0)) == TupleType((i64, f64))
    assert typeof((1, 2)) != typeof((1, 2, 3))  # arity is part of the type


# --- FnType: structural, value-keyed ---------------------------------------


def test_fntype_stable_across_capture_values():
    f5, f6 = make_closure(5), make_closure(6)
    assert f5.fntype == f6.fntype  # same code object + same env_types
    assert f5.env == {"x": 5} and f6.env == {"x": 6}  # values differ, type does not


def test_fntype_differs_on_capture_type():
    assert make_closure(5).fntype != make_closure(3.0).fntype


def test_fntype_env_types_in_freevar_order():
    f5 = make_closure(5)
    assert f5.fntype.env_types == (i64,)
    assert f5.freevars == ("x",)


# --- the cache: one compile per (FnType, arg_types, generation) -------------


def test_one_compile_across_capture_values():
    cache = SpecCache()
    compiles = []

    def compile_inner(label):
        compiles.append(label)
        return f"artifact:{label}"

    f5, f6 = make_closure(5), make_closure(6)
    arg_types = (i64,)
    a = cache.get_or_compile(f5.fntype, arg_types, lambda: compile_inner("first"))
    b = cache.get_or_compile(f6.fntype, arg_types, lambda: compile_inner("second"))

    assert a == b == "artifact:first"  # f6 reused f5's artifact
    assert cache.compile_count == 1  # THE THESIS
    assert cache.hit_count == 1


def test_recompiles_on_arg_type_change():
    cache = SpecCache()
    f = make_closure(5)
    cache.get_or_compile(f.fntype, (i64,), lambda: "int")
    cache.get_or_compile(f.fntype, (f64,), lambda: "float")
    assert cache.compile_count == 2


def test_generation_bump_invalidates():
    cache = SpecCache()
    f = make_closure(5)
    cache.get_or_compile(f.fntype, (i64,), lambda: "gen0")
    bump_generation()
    cache.get_or_compile(f.fntype, (i64,), lambda: "gen1")
    assert cache.compile_count == 2
