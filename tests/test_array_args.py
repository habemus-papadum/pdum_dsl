"""Stage 2a — ARGUMENT arrays: the arg-side twin of step-11's captures.

The plan/extractor/pack machinery was arg-generic from step 7; what landed
here is the lowering (`core.param` bases in `_linear_index`, `array.dim`
for shape/stride staging reads) and the renderer resolution. Everything a
captured array can do, an argument array can do — including rank-generic
cache hits across shapes and named contraction."""

import pytest

np = pytest.importorskip("numpy")

import pdum.dsl  # noqa: F401, E402
from pdum.dsl.kernel.api import jit  # noqa: E402
from pdum.dsl.kernel.cache import no_compile  # noqa: E402
from pdum.dsl.kernel.registry import DEFAULT  # noqa: E402
from pdum.dsl.stdlib.arrays import Named  # noqa: E402


@jit()
def load2(t, i, j):
    return t[i, j] * 2.0


def test_positional_arg_array():
    x = np.arange(12.0).reshape(3, 4)
    assert load2(x, 1, 2) == x[1, 2] * 2.0


def test_arg_shape_is_not_identity():
    load2(np.ones((3, 4)), 0, 0)
    with no_compile():  # a NEW shape as an argument: rank-generic hit
        assert load2(np.full((7, 9), 5.0), 6, 8) == 10.0


def test_named_arg_array_isel():
    @jit()
    def g(t, k):
        return t.isel(x=k) + 100.0

    assert g(Named(np.arange(4.0), ("x",)), 2) == 102.0


def test_capture_and_arg_arrays_mix():
    def make(table):
        @jit()
        def h(t, i):
            return table[i] + t[i]

        return h

    h = make(np.array([10.0, 20.0, 30.0]))
    assert h(np.array([1.0, 2.0, 3.0]), 1) == 22.0


def test_arg_arrays_on_c_backend():
    from pdum.dsl.backends import c

    if not c.is_available():
        pytest.skip("no C compiler")
    ext = DEFAULT.extend()
    c.install(ext, default=True)
    x = np.arange(12.0).reshape(3, 4)
    assert ext.dispatch(load2, (x, 2, 3)) == x[2, 3] * 2.0


def test_matmul_over_argument_arrays():
    @jit()
    def cell(A, B, i, j):
        return matmul(A, B, i, j)  # noqa: F821

    A = Named(np.arange(6.0).reshape(2, 3), ("row", "inner"))
    B = Named(np.arange(12.0).reshape(3, 4), ("inner", "col"))
    got = np.array([[cell(A, B, i, j) for j in range(4)] for i in range(2)])
    assert np.allclose(got, A.array @ B.array)
    with no_compile():  # bigger operands, same names: the trip count is a staging value
        cell(Named(np.ones((5, 7)), ("row", "inner")), Named(np.ones((7, 2)), ("inner", "col")), 4, 1)


def test_jvp_treats_arg_arrays_as_constants():
    """Array args have no tangents yet (110/130 deferral) — the derivative
    w.r.t. the FLOAT args must still be exact with an array arg present...
    which today refuses loudly at the jvp arg check (documented posture)."""
    from pdum.dsl.kernel.ir import VerifyError
    from pdum.dsl.stdlib.transforms import jvp

    @jit()
    def f(t, x):
        return t[0] * x * x

    with pytest.raises(VerifyError, match="float args"):
        jvp(f)(np.array([2.0]), 1.5, np.array([0.0]), 1.0)
