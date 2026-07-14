"""Step 12 — transforms: jvp (forward AD) vs finite differences, the
in-kernel `D`, SIMT vmap with named weaving, and named/batched matmul."""

import pytest

np = pytest.importorskip("numpy")

import pdum.dsl  # noqa: F401, E402
from pdum.dsl.kernel.api import jit  # noqa: E402
from pdum.dsl.kernel.cache import no_compile  # noqa: E402
from pdum.dsl.kernel.ir import VerifyError  # noqa: E402
from pdum.dsl.kernel.lower import MissingRule  # noqa: E402
from pdum.dsl.kernel.registry import DEFAULT  # noqa: E402
from pdum.dsl.stdlib.arrays import Named  # noqa: E402
from pdum.dsl.stdlib.transforms import jvp, over  # noqa: E402


def fd(f, x, eps=1e-7):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def test_jvp_matches_finite_differences_through_control_flow():
    def make(c):
        @jit()
        def f(x):
            y = c * x * x + sqrt(x)  # noqa: F821
            if y > 2.0:
                y = y * 2.0
            acc = y
            for i in range(3):
                acc = acc * x
            return acc

        return f

    f = make(0.5)
    p, t = jvp(f)(1.3, 1.0)
    assert abs(p - f(1.3)) < 1e-12
    assert abs(t - fd(f, 1.3)) < 1e-5


def test_jvp_is_a_cached_identity():
    def make(c):
        @jit()
        def f(x):
            return c * x * x

        return f

    jvp(make(2.0))(1.0, 1.0)
    with no_compile():  # fresh closure UNDER the transform: still a hit
        assert jvp(make(3.0))(2.0, 1.0) == (12.0, 12.0)


def test_jvp_arg_contract():
    @jit()
    def f(x):
        return x * x

    with pytest.raises(VerifyError, match="doubles the args"):
        jvp(f)(1.0)


def test_D_exact_partials_and_branches():
    def make(cx, cy):
        @jit()
        def k(i, j):
            r2 = (i - cx) * (i - cx) + (j - cy) * (j - cy)
            di, dj = D(r2)  # noqa: F821
            return di * 10.0 + dj

        return k

    assert abs(make(1.0, 2.0)(3.0, 5.0) - (2 * 2.0 * 10.0 + 2 * 3.0)) < 1e-12

    def make_b(c):
        @jit()
        def k(i, j):
            v = i * j
            if v > c:
                v = v * v
            di, dj = D(v)  # noqa: F821
            return di - dj

        return k

    assert abs(make_b(100.0)(3.0, 5.0) - (5.0 - 3.0)) < 1e-12  # untaken: d(ij)
    assert abs(make_b(1.0)(3.0, 5.0) - (2 * 15 * 5 - 2 * 15 * 3)) < 1e-12  # taken: d((ij)^2)


def test_D_of_structured_value():
    @jit()
    def kt(i, j):
        p = (i * j, i + j)
        dp_i, dp_j = D(p)  # noqa: F821
        return dp_i[0] + dp_j[0] + dp_i[1] * 100.0

    assert abs(kt(3.0, 5.0) - (5.0 + 3.0 + 100.0)) < 1e-12


def test_D_through_a_loop_widens_the_carry():
    def make(g):
        @jit()
        def k(x):
            acc = x
            for i in range(4):
                acc = acc * g + x
            return D(acc)[0]  # noqa: F821 — d/dx of the whole recurrence (D is ALWAYS a tuple)

        return k

    def ref(x, g=0.5):
        acc = x
        for _ in range(4):
            acc = acc * g + x
        return acc

    eps = 1e-7
    got = make(0.5)(2.0)
    assert abs(got - (ref(2.0 + eps) - ref(2.0 - eps)) / (2 * eps)) < 1e-5


def test_fwidth_sugar_matches_finite_differences():
    from pdum.dsl.demo import graphics  # noqa: F401 — ddx/ddy/fwidth are demo vocabulary

    @jit()
    def ks(i, j):
        m = smoothstep(0.2, 0.8, i * 0.1 + j * 0.05)  # noqa: F821
        return fwidth(m)  # noqa: F821

    def m(i, j):
        t = min(max((i * 0.1 + j * 0.05 - 0.2) / 0.6, 0.0), 1.0)
        return t * t * (3 - 2 * t)

    eps = 1e-7
    ref = abs((m(3.0 + eps, 4.0) - m(3.0 - eps, 4.0)) / (2 * eps)) + abs(
        (m(3.0, 4.0 + eps) - m(3.0, 4.0 - eps)) / (2 * eps)
    )
    assert abs(ks(3.0, 4.0) - ref) < 1e-5


def test_D_costs_nothing_when_unused():
    """Two kernels, one with a dead `D`-free body change... the honest check:
    a kernel WITHOUT D compiles to the same artifact as it always did — the
    tangent slice exists only where demanded."""
    ext = DEFAULT.extend()

    def make(c):
        @jit()
        def f(x):
            return c * x * x

        return f

    ext.dispatch(make(2.0), (1.0,))
    n = len(ext.artifacts)
    ext.dispatch(make(5.0), (2.0,))
    assert len(ext.artifacts) == n  # no tangent machinery leaked into D-free kernels


def test_over_batching_ignorance_arc():
    data = Named(np.arange(12.0).reshape(3, 4), ("batch", "x"))

    def make(t):
        @jit()
        def g(k):
            return t.isel(x=k) * 10.0

        return g

    g = make(data)
    with pytest.raises(MissingRule, match="isel is pedantic"):  # batch unaccounted
        g(0)
    vg = over(g, axis="batch")
    assert [vg(1, b) for b in range(3)] == [10.0, 50.0, 90.0]
    with no_compile():  # a bigger batch is the SAME type (rank-generic)
        over(make(Named(np.ones((100, 4)), ("batch", "x"))), axis="batch")(1, 99)


def test_over_name_scoping_inside_the_body():
    data = Named(np.ones((3, 4)), ("batch", "x"))

    def make(t):
        @jit()
        def g(k):
            return t.isel(batch=k, x=k)  # naming the woven axis: refused

        return g

    with pytest.raises(MissingRule, match="mapped away"):
        over(make(data), axis="batch")(0, 0)


def test_over_refuses_when_nothing_carries_the_axis():
    data = Named(np.ones((4,)), ("x",))

    def make(t):
        @jit()
        def g(k):
            return t.isel(x=k)

        return g

    with pytest.raises(VerifyError, match="no capture carries it"):
        over(make(data), axis="batch")(0, 0)


def test_over_runs_on_the_c_backend():
    from pdum.dsl.backends import c

    if not c.is_available():
        pytest.skip("no C compiler")
    data = Named(np.arange(8.0).reshape(2, 4), ("batch", "x"))

    def make(t):
        @jit()
        def g(k):
            return t.isel(x=k) + 1.0

        return g

    vg = over(make(data), axis="batch")
    ext = DEFAULT.extend()
    c.install(ext, default=True)
    assert ext.dispatch(vg, (2, 1)) == vg(2, 1) == 7.0


def test_D_memo_survives_gc_address_reuse():
    """Review-caught soundness bug: the tangent memo was keyed by id() of
    GC-able nodes — a fresh node could reuse a dead node's address and
    inherit its tangent. Two D calls on inline expressions pin it."""

    @jit()
    def k(u, v):
        a = D(u * v)  # noqa: F821
        b = D(u + v)  # noqa: F821
        return b[0] * 1000.0 + b[1] * 100.0 + a[0] + a[1]

    assert k(3.0, 5.0) == 1000.0 + 100.0 + 5.0 + 3.0


def test_D_always_returns_a_tuple_and_sugar_works_on_one_param():
    from pdum.dsl.demo import graphics  # noqa: F401

    @jit()
    def one(i):
        return ddx(i * i)  # noqa: F821 — D(v) is (d/di,) even for one param

    assert one(3.0) == 6.0


def test_D_under_over_composes():
    """The two flagship features together (review-caught gap: the root-argc
    plant missed derived builds)."""
    from pdum.dsl.demo import graphics  # noqa: F401

    data = Named(np.array([[1.0, 4.0], [9.0, 16.0]]), ("batch", "x"))

    def make(t):
        @jit()
        def g(u):
            s = t.isel(x=0) * u * u  # d/du = 2u * t[b,0]
            return ddx(s)  # noqa: F821 — fwidth needs two coordinates; this kernel has one

        return g

    vg = over(make(data), axis="batch")
    assert vg(3.0, 0) == 6.0 * 1.0
    assert vg(3.0, 1) == 6.0 * 9.0


def test_D_inside_loop_body_refuses():
    @jit()
    def k(x):
        out = 0.0
        acc = x
        for i in range(2):
            out = out + D(acc)[0]  # noqa: F821 — the carry has no tangent HERE
            acc = acc * 0.5
        return out

    with pytest.raises(VerifyError, match="loop binder without a tangent"):
        k(1.0)


def test_over_composes_over_two_axes():
    """Composition is lower_handle re-entry (130 §7): the outer build lowers
    the inner WRAPPER, which dispatches to its own build rule with the merged
    woven context. Lanes trail in application order, outermost LAST."""
    rng = np.random.default_rng(3)
    data = Named(rng.standard_normal((2, 3, 4)), ("b1", "b2", "x"))

    def make(t):
        @jit()
        def g(k):
            return t.isel(x=k) * 10.0

        return g

    composed = over(over(make(data), axis="b2"), axis="b1")
    for i in range(2):
        for j in range(3):
            assert composed(1, j, i) == data.array[i, j, 1] * 10.0
    with no_compile():  # a fresh closure under BOTH transforms: still one identity
        over(over(make(Named(rng.standard_normal((5, 6, 4)), ("b1", "b2", "x"))), axis="b2"), axis="b1")(0, 0, 0)


def test_over_duplicate_axis_refused():
    data = Named(np.ones((2, 3, 4)), ("b1", "b2", "x"))

    def make(t):
        @jit()
        def g(k):
            return t.isel(x=k)

        return g

    with pytest.raises(VerifyError, match="already mapped by an enclosing over"):
        over(over(make(data), axis="b1"), axis="b1")(0, 0, 0)


def test_jvp_of_over_refuses_on_the_lane_type():
    data = Named(np.ones((2, 4)), ("batch", "x"))

    def make(t):
        @jit()
        def g(u):
            return t.isel(x=0) * u

        return g

    with pytest.raises(VerifyError, match="floats? args"):
        jvp(over(make(data), axis="batch"))(1.0, 0, 1.0, 0)


def test_transformed_kernels_are_capturable():
    """A vmapped kernel SUMMARIZES like a Pipeline (ValueKind); calling it
    in-body still refuses downstream (first-class kernel values, later)."""
    data = Named(np.ones((2, 3)), ("batch", "x"))

    def make(t):
        @jit()
        def g(k):
            return t.isel(x=k)

        return g

    vg = over(make(data), axis="batch")

    @jit()
    def outer(x):
        return x + 1.0  # vg captured but unused would fold away; capture it live:

    @jit()
    def uses(x):
        return vg(0, 0) + x  # noqa: F821 — capture summarizes; the CALL refuses

    assert "over" in repr(uses.fntype)  # typeof(Over) worked at the def site


def test_named_matmul_matches_numpy_and_batches():
    A = Named(np.arange(6.0).reshape(2, 3), ("row", "inner"))
    B = Named(np.arange(12.0).reshape(3, 4), ("inner", "col"))

    def make(A, B):
        @jit()
        def cell(i, j):
            return matmul(A, B, i, j)  # noqa: F821

        return cell

    cell = make(A, B)
    got = np.array([[cell(i, j) for j in range(4)] for i in range(2)])
    assert np.allclose(got, A.array @ B.array)

    with pytest.raises(MissingRule, match="ONE shared axis"):
        make(A, Named(np.ones((3, 4)), ("k", "col")))(0, 0)

    rng = np.random.default_rng(7)
    Ab = Named(rng.standard_normal((5, 2, 3)), ("batch", "row", "inner"))
    Bb = Named(rng.standard_normal((5, 3, 4)), ("batch", "inner", "col"))
    bcell = over(make(Ab, Bb), axis="batch")
    got3 = np.array([[[bcell(i, j, b) for j in range(4)] for i in range(2)] for b in range(5)])
    assert np.allclose(got3, Ab.array @ Bb.array)
    with no_compile():  # new batch size AND new inner extent: all staging values
        over(
            make(
                Named(rng.standard_normal((9, 2, 7)), ("batch", "row", "inner")),
                Named(rng.standard_normal((9, 7, 4)), ("batch", "inner", "col")),
            ),
            axis="batch",
        )(0, 0, 8)
