"""Step 1 — the type lattice: structural values, template identity."""

import dataclasses

import pytest

from pdum.dsl.kernel import types as T

# --- structural equality and hashing -----------------------------------------


def test_types_are_structural_values():
    assert T.Scalar("f64") == T.f64
    assert T.Scalar("f64") is not T.f64
    assert hash(T.Scalar("f64")) == hash(T.f64)
    assert {T.f64: "a"}[T.Scalar("f64")] == "a"  # usable as a dict key


def test_vec_and_record_structural():
    assert T.Vec(T.f32, 3) == T.Vec(T.f32, 3)
    assert T.Vec(T.f32, 3) != T.Vec(T.f32, 4)
    color = T.Record("Color", (("r", T.f32), ("g", T.f32), ("b", T.f32)))
    assert color == T.Record("Color", (("r", T.f32), ("g", T.f32), ("b", T.f32)))
    assert hash(color) == hash(T.Record("Color", (("r", T.f32), ("g", T.f32), ("b", T.f32))))


def test_tuple_type_structural():
    assert T.Tuple((T.i64, T.f64)) == T.Tuple((T.i64, T.f64))
    assert T.Tuple((T.i64,)) != T.Tuple((T.i64, T.i64))  # arity in the identity
    assert T.Tuple((T.Tuple((T.f64,)),)) == T.Tuple((T.Tuple((T.f64,)),))  # recursive hash/eq
    assert repr(T.Tuple((T.i64, T.f64))) == "(i64, f64)"


def test_types_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        T.f64.kind = "f32"


def test_validation_is_loud():
    with pytest.raises(ValueError):
        T.Scalar("f16")
    with pytest.raises(ValueError):
        T.Vec(T.f32, 5)
    with pytest.raises(TypeError):
        T.Vec(T.Vec(T.f32, 2), 2)  # no vectors of vectors


def test_literal_type_carries_value_in_identity():
    a = T.LiteralType(T.i64, 8)
    b = T.LiteralType(T.i64, 9)
    assert a != b  # the ONE value-in-type opt-in: value participates in the key
    assert a == T.LiteralType(T.i64, 8)
    assert hash(a) == hash(T.LiteralType(T.i64, 8))


def test_literal_values_are_type_aware():
    # Python's == is cross-type (1 == 1.0 == True; 0.0 == -0.0). If these
    # collided, a specialization compiled for one constant would be silently
    # served for another — the wrong-hit failure class.
    assert T.LiteralType(T.f64, 1) != T.LiteralType(T.f64, 1.0)
    assert T.LiteralType(T.f64, 1) != T.LiteralType(T.f64, True)
    assert T.LiteralType(T.f64, 0.0) != T.LiteralType(T.f64, -0.0)
    assert T.LiteralType(T.f64, (1,)) != T.LiteralType(T.f64, (1.0,))
    nan = float("nan")
    assert T.LiteralType(T.f64, nan) == T.LiteralType(T.f64, nan)  # keys need reflexivity


def test_reprs_read_well():
    assert repr(T.f64) == "f64"
    assert repr(T.Vec(T.f32, 3)) == "vec3<f32>"
    assert repr(T.LiteralType(T.i64, 8)) == "Literal[i64 = 8]"


# --- template identity: code objects compare by VALUE -------------------------

SRC = "def f(x):\n    return x + k\n"
SRC_EDITED = "def f(x):\n    return x + k + 1\n"


def _code(src):
    ns = {}
    exec(compile(src, "<ch02-preview>", "exec"), ns)
    return ns["f"].__code__


def test_base_template_hits_across_rerun_and_misses_on_edit():
    a, b = T.Base(_code(SRC)), T.Base(_code(SRC))
    assert a == b and hash(a) == hash(b)  # unchanged re-run: value-equal code
    assert T.Base(_code(SRC)) != T.Base(_code(SRC_EDITED))  # edit: natural miss


def test_derived_never_collides_with_base():
    base = T.Base(_code(SRC))
    grad = T.Derived("grad", base, (("wrt", 0),))
    assert grad != base
    assert grad == T.Derived("grad", T.Base(_code(SRC)), (("wrt", 0),))  # rebuilt per frame: hit
    assert grad != T.Derived("grad", base, (("wrt", 1),))
    assert grad.label == "grad(f)"


def test_fntype_is_the_thesis_in_one_value():
    base = T.Base(_code(SRC))
    a = T.FnType(base, (T.i64,))
    assert a == T.FnType(T.Base(_code(SRC)), (T.i64,))  # same code, same env types
    assert a != T.FnType(base, (T.f64,))  # a capture changed type
    assert repr(a) == "Fn<f>(i64)"
