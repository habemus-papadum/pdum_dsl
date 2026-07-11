"""Step 1 — typeof/fingerprint: summaries, bucketing, and the soundness law."""

import random

import pytest

from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.valuekind import BUILTINS, BigIntError, fingerprint, typeof


def test_typeof_scalars():
    assert typeof(True) == T.boolean  # bool dispatched before int (exact-type lookup)
    assert typeof(5) == T.i64
    assert typeof(-(2**63)) == T.i64
    assert typeof(2**63) == T.u64  # overflows i64 -> u64: value-dependent typing since M0
    assert typeof(2**64 - 1) == T.u64
    assert typeof(1.5) == T.f64


def test_int_bucketing_is_loud_at_the_edges():
    with pytest.raises(BigIntError):
        typeof(2**64)
    with pytest.raises(BigIntError):
        typeof(-(2**63) - 1)
    with pytest.raises(BigIntError):
        fingerprint(2**70)  # the fast tag must bucket identically


def test_tuples_summarize_elementwise():
    # Honest identity: Tuple, never Vec — vec-ness is a dialect interpretation.
    assert typeof((1.0, 2.0)) == T.Tuple((T.f64, T.f64))
    assert typeof((1, 2.0)) == T.Tuple((T.i64, T.f64))  # heterogeneous is fine at this layer
    assert typeof(((1.0, 2.0), 3)) == T.Tuple((T.Tuple((T.f64, T.f64)), T.i64))  # recursive
    assert typeof((1, 2)) != typeof((1, 2, 3))  # arity is part of the type
    assert typeof(()) == T.Tuple(())


def test_tuple_loudness_comes_only_from_elements():
    with pytest.raises(BigIntError):
        typeof((1.0, 2**70))
    with pytest.raises(TypeError, match="no ValueKind registered"):
        typeof((1.0, object()))


def test_unregistered_type_is_loud():
    with pytest.raises(TypeError, match="no ValueKind registered"):
        typeof(object())
    with pytest.raises(TypeError, match="no ValueKind registered"):
        typeof(None)  # None is deliberately unsupported until a use case forces it


def test_mro_dispatch():
    class MyFloat(float): ...

    assert typeof(MyFloat(2.0)) == T.f64  # exact-type miss, MRO walk hits float


def test_extended_table_overrides_reach_nested_elements():
    # Kinds get the dispatching table per call, so a child table's override
    # must apply inside composites too (the layering that surface C formalizes).
    class WeirdFloatKind:
        def typeof(self, v, table):
            return T.f32

        def fingerprint(self, v, table):
            return "f32"

    mine = BUILTINS.extend()
    mine.register(float, WeirdFloatKind())
    assert mine.typeof(1.5) == T.f32
    assert mine.typeof((1.5, 2.5)) == T.Tuple((T.f32, T.f32))  # nested elements see the override
    assert mine.fingerprint((1.5, 2.5)) == ("t", ("f32", "f32"))
    assert BUILTINS.typeof(1.5) == T.f64  # the parent table is untouched


# --- the soundness law: fingerprint(a) == fingerprint(b) => same typeof outcome
# (This generator has a pedagogical twin in docs/book/ch01-types-are-values.ipynb;
#  when the value universe grows — new kinds, new buckets — update BOTH.)


def _outcome(fn, v):
    try:
        return ("ok", fn(v))
    except TypeError as e:  # BigIntError ⊂ TypeError
        return ("raise", type(e).__name__)


def _random_value(rng, depth=0):
    roll = rng.random()
    if roll < 0.15:
        return rng.choice([True, False])
    if roll < 0.45:
        span = rng.choice([2**8, 2**62, 2**63, 2**64, 2**66])
        return rng.randrange(-span, span)
    if roll < 0.7 or depth >= 2:
        return rng.uniform(-1e9, 1e9)
    return tuple(_random_value(rng, depth + 1) for _ in range(rng.randrange(0, 6)))


def test_fingerprint_soundness_fuzz():
    rng = random.Random(20260711)
    by_fp = {}
    for _ in range(3000):
        v = _random_value(rng)
        fp = _outcome(BUILTINS.fingerprint, v)
        ty = _outcome(BUILTINS.typeof, v)
        if fp[0] == "raise":
            assert ty[0] == "raise", f"fingerprint raised but typeof did not for {v!r}"
            continue
        prior = by_fp.setdefault(fp, (v, ty))
        assert prior[1] == ty, (
            f"SOUNDNESS VIOLATION: {prior[0]!r} and {v!r} share fingerprint {fp!r} "
            f"but typeof outcomes differ: {prior[1]!r} vs {ty!r}"
        )
