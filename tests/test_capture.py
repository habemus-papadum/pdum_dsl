"""Step 2 — phase-A capture: Handles, FnTypes from live closures, snapshots."""

import pytest

from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.capture import Handle, make_handle
from pdum.dsl.kernel.valuekind import fingerprint, typeof


def make_closure(x):
    @jit(kind="device")
    def inner(y):
        return x + y

    return inner


def test_fntype_stable_across_capture_values():
    f5, f6 = make_closure(5), make_closure(6)
    assert f5.fntype == f6.fntype  # same code (by value) + same env types
    assert f5.fp == f6.fp  # handle fingerprints agree too
    assert f5.env == {"x": 5} and f6.env == {"x": 6}  # values differ, identity doesn't


def test_fntype_differs_on_capture_type():
    assert make_closure(5).fntype != make_closure(3.0).fntype
    assert make_closure(5).fntype != make_closure(2**63).fntype  # i64 vs u64 bucketing


def test_cell_rerun_hits_and_edit_misses():
    src = "def f(k):\n    def g():\n        return k\n    return g\n"

    def build(source):
        ns = {}
        exec(compile(source, "<cell>", "exec"), ns)
        return make_handle(ns["f"](7), "device")

    assert build(src).fntype == build(src).fntype  # unchanged re-run: value-equal code
    assert build(src).fntype != build(src.replace("return k", "return k + 1")).fntype


def test_tuple_capture_unlocks_center():
    center = (320.0, 240.0)

    @jit()
    def shader():
        return center

    assert shader.env_types == (T.Tuple((T.f64, T.f64)),)  # the M0 gap, closed at capture


def test_nested_handles_compose_structurally():
    def dense(w, b):
        @jit()
        def layer(x):
            return w * x + b

        return layer

    def net(w1, b1, w2, b2):
        inner = dense(w1, b1)

        @jit()
        def outer(x):
            return inner(x) * w2 + b2

        return outer

    a, b = net(1.0, 2.0, 3.0, 4.0), net(9.0, 8.0, 7.0, 6.0)
    assert a.fntype == b.fntype  # the program is the parameter container
    assert any(isinstance(t, T.FnType) for t in a.env_types)  # child FnType inside parent identity
    assert net(1.0, 2.0, 3.0, 4)  # int b2 ...
    assert net(1.0, 2.0, 3.0, 4).fntype != a.fntype  # ... is a different composed identity


def test_handles_are_first_class_values():
    h = make_closure(5)
    assert typeof(h) == h.fntype  # a Handle summarizes as its FnType
    assert fingerprint(h) == h.fp
    assert typeof((h, h)) == T.Tuple((h.fntype, h.fntype))  # tuples of handles compose


def test_self_referential_closure_phase_a_succeeds():
    captured = {}

    def deco(fn):
        captured["h"] = make_handle(fn, "device")
        return fn

    def factory():
        @deco
        def rec(n):
            return 1 if n == 0 else rec(n - 1)  # rec's own cell is empty at decoration

        return rec

    factory()
    h = captured["h"]
    assert isinstance(h, Handle)
    assert h.env == {}  # the unbound self-cell is skipped, phase A does not fail
    assert h.freevars == ("rec",)  # but the template knows the name


def test_missing_source_does_not_fail_phase_a():
    ns = {}
    exec(compile("def f(k):\n    def g():\n        return k\n    return g\n", "<no-file>", "exec"), ns)
    h = make_handle(ns["f"](1), "device")
    assert h.snapshot is None  # NoSourceError is phase B's job (step 6)


def test_snapshot_taken_once_per_code_object():
    a, b = make_closure(1), make_closure(2)
    assert a.snapshot is b.snapshot  # memoized (WeakKeyDictionary on the code object)
    assert a.snapshot.qualname.endswith("inner")
    assert "def inner(y):" in a.snapshot.text


def test_untypeable_capture_is_loud_at_the_def_site():
    obj = object()
    with pytest.raises(TypeError, match="no ValueKind registered"):

        @jit()
        def f():
            return obj
