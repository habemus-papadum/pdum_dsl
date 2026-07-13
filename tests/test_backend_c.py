"""Step 11 — the C backend: golden renders (no compiler needed), then
probe-gated compile-and-run differentials against the Python twin."""

import pytest

np = pytest.importorskip("numpy")

import pdum.dsl  # noqa: F401, E402
from pdum.dsl.backends import c  # noqa: E402
from pdum.dsl.kernel.api import jit  # noqa: E402
from pdum.dsl.kernel.registry import DEFAULT  # noqa: E402
from pdum.dsl.stdlib.arrays import Named  # noqa: E402

pytestmark = pytest.mark.skipif(not c.is_available(), reason="no C compiler on PATH")


def c_registry():
    ext = DEFAULT.extend()
    c.install(ext, default=True)
    return ext


def both(target, args):
    """One kernel, two backends, same closure — the differential harness."""
    ext = c_registry()
    return DEFAULT.dispatch(target, args), ext.dispatch(target, args)


def test_scalar_kernel_matches_twin():
    def make(cx, gain):
        @jit()
        def f(x):
            return gain * (x - cx) * (x + cx)

        return f

    py, cc = both(make(0.3, 2.0), (1.7,))
    assert abs(py - cc) < 1e-12


def test_branches_loops_and_intrinsics_match():
    @jit()
    def f(x):
        acc = 0.0
        for i in range(8):
            v = sqrt(float(i) + x)  # noqa: F821
            if v > 2.0:
                acc = acc + v
            else:
                acc = acc - 0.5
        return acc

    py, cc = both(f, (1.25,))
    assert abs(py - cc) < 1e-12


def test_multi_carry_scalarization():
    @jit()
    def f(x):
        a = x
        b = 1.0
        for i in range(6):
            a = a + b
            b = b * 1.5
        return a + b

    py, cc = both(f, (0.5,))
    assert abs(py - cc) < 1e-12


def test_array_capture_and_named_axes_on_c():
    t = np.arange(12.0).reshape(3, 4)
    nt = Named(np.arange(12.0).reshape(3, 4), ("y", "x"))

    def make_pos(t):
        @jit()
        def f(i, j):
            return t[i, j] * 2.0

        return f

    def make_named(t):
        @jit()
        def g(i, j):
            return t.isel(x=j, y=i) * 2.0

        return g

    py, cc = both(make_pos(t), (2, 3))
    assert py == cc == 22.0
    py, cc = both(make_named(nt), (2, 3))
    assert py == cc == 22.0


def test_trunc_div_mod_policy_agrees():
    """070's numeric policy: trunc div/mod (C native). The twin compensates
    for Python's floored operators — negative operands are where they differ."""

    @jit()
    def f(a, b):
        return float(a / b) + float(a % b)

    for args in ((7, 2), (-7, 2), (7, -2), (-7, -2)):
        py, cc = both(f, args)
        assert py == cc, f"{args}: twin {py} vs C {cc}"


def test_int_div_is_exact_past_float_precision():
    """Review-caught: the first twin routed i64 div through double, losing
    exactness above 2^53. The helpers are integer-exact now."""

    @jit()
    def q(a, b):
        return a / b

    @jit()
    def r(a, b):
        return a % b

    big = 2**60 + 1
    assert both(q, (big, 1)) == (big, big)
    assert both(r, (big, 2)) == (1, 1)


def test_float_mod_is_fmod_on_both_targets():
    """Review-caught twice over: C's `%` does not compile for doubles, and
    Python's `%` is floored — both sides now spell trunc fmod."""
    import math

    @jit()
    def f(a, b):
        return a % b

    py, cc = both(f, (-7.0, 2.0))
    assert py == cc == math.fmod(-7.0, 2.0) == -1.0


def test_zero_trip_and_nested_join_on_c():
    """The two riskiest C render paths, pinned (review-caught coverage gap):
    zero-trip loops (carry-is-init declare/assign) and a branch join INSIDE
    a multi-carry loop (nested tuple-lane assignment)."""

    def make(n):
        @jit()
        def f(x):
            a = x
            b = 1.0
            for i in range(n):
                if a > b:
                    b = b + a
                else:
                    a = a + 0.5
            return a * 10.0 + b

        return f

    for n in (0, 5):
        py, cc = both(make(n), (0.25,))
        assert py == cc, f"n={n}: twin {py} vs C {cc}"


def test_rendered_c_is_content_cached():
    ext = c_registry()

    def make(k):
        @jit()
        def f(x):
            return x * k

        return f

    ext.dispatch(make(2.0), (1.0,))
    n = len(ext.artifacts._store) if hasattr(ext.artifacts, "_store") else None
    ext.dispatch(make(3.0), (1.0,))  # same types, new value: same ARTIFACT too
    if n is not None:
        assert len(ext.artifacts._store) == n


def test_golden_render_shape():
    """Structure of the emitted C, pinned (hardware-free)."""
    t = np.ones((2, 2))

    def make(t):
        @jit()
        def f(i, j):
            acc = 0.0
            for k in range(2):
                acc = acc + t[i, k]
            return acc

        return f

    ext = c_registry()
    f = make(t)
    ext.dispatch(f, (0, 1))
    fp = tuple(ext.table.fingerprint(a) for a in (0, 1))
    rec = ext.specializations.probe(ext.specializations.key_for(f, fp, ext.backend_for(f.kind).fp))
    src = rec.artifact.__pdum_source__
    assert "for (int64_t" in src and "bufs[0]" in src and "ld_i64(staging" in src
    assert "pdum-restype: f64" in src
