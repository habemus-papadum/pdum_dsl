"""Step 11 — arrays as captures: rank-generic caching, the shape dial,
named axes (the xarray exercise), and the pedantic indexing refusals."""

import pytest

np = pytest.importorskip("numpy")

import pdum.dsl  # noqa: F401, E402
from pdum.dsl.kernel.api import jit  # noqa: E402
from pdum.dsl.kernel.cache import no_compile  # noqa: E402
from pdum.dsl.kernel.lower import MissingRule  # noqa: E402
from pdum.dsl.kernel.registry import DEFAULT  # noqa: E402
from pdum.dsl.stdlib.arrays import Named, Shaped  # noqa: E402


def make_positional(t):
    @jit()
    def f(i, j):
        return t[i, j]

    return f


def test_positional_indexing_matches_numpy():
    t = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    f = make_positional(t)
    assert f(1, 2) == 6.0 and f(0, 0) == 1.0


def test_rank_generic_shape_is_not_identity():
    """THE thesis, extended to data: shape is a VALUE (staging bytes), only
    rank/dtype/device are identity — a new shape is a cache hit."""
    make_positional(np.ones((2, 3)))(0, 0)
    with no_compile():
        assert make_positional(np.full((7, 5), 4.0))(6, 4) == 4.0


def test_buffer_contents_are_data_not_identity():
    t = np.zeros((2, 2))
    f = make_positional(t)
    assert f(1, 1) == 0.0
    t[1, 1] = 42.0  # in-place mutation: the pointer travels fresh each call
    assert f(1, 1) == 42.0


def test_shaped_dial_specializes_per_shape():
    def make(t):
        @jit()
        def g(i, j):
            return t[i, j]

        return g

    g = make(Shaped(np.ones((2, 3))))
    g(0, 0)
    c0 = DEFAULT.specializations.compiles
    make(Shaped(np.ones((3, 2))))(0, 0)  # same rank, new shape: a NEW TYPE here
    assert DEFAULT.specializations.compiles - c0 == 1
    fp = (DEFAULT.table.fingerprint(0),) * 2
    rec = DEFAULT.specializations.probe(DEFAULT.specializations.key_for(g, fp, DEFAULT.backend_for(g.kind).fp))
    assert rec.plan.staging_size == 16  # two i64 args ONLY: shape/strides left staging for the type


def test_named_axes_isel_keyword_order_is_free():
    t = Named(np.arange(6.0).reshape(2, 3), ("y", "x"))

    def make(t):
        @jit()
        def h(a, b):
            return t.isel(x=b, y=a)

        return h

    assert make(t)(1, 2) == 5.0


def test_named_array_refuses_positional():
    t = Named(np.ones((2, 2)), ("y", "x"))
    with pytest.raises(MissingRule, match="axes have NAMES.*isel"):
        make_positional(t)(0, 0)


def test_isel_pedantry_names_every_axis():
    t = Named(np.ones((2, 2)), ("y", "x"))

    def make(t):
        @jit()
        def h(a):
            return t.isel(y=a)  # x missing

        return h

    with pytest.raises(MissingRule, match="every axis exactly once"):
        make(t)(0)


def test_indices_are_strict_i64():
    t = np.ones((2,))

    @jit()
    def f(x):
        return t[x]

    with pytest.raises(TypeError, match="strict i64"):
        f(0.5)


def test_partial_indexing_refused():
    t = np.ones((2, 3))

    @jit()
    def f(i):
        return t[i]

    with pytest.raises(MissingRule, match="every axis exactly once"):
        f(0)


def test_noncontiguous_refused_at_the_def_site():
    base = np.ones((4, 4))
    with pytest.raises(TypeError, match="C-contiguous"):
        make_positional(base.T)(0, 0)


def test_xarray_dataarray_adopts_with_dims():
    xr = pytest.importorskip("xarray")
    da = xr.DataArray(np.arange(12.0).reshape(3, 4), dims=("lat", "lon"))

    def make(t):
        @jit()
        def f(a, b):
            return t.isel(lon=b, lat=a)

        return f

    assert make(da)(2, 3) == 11.0
    with no_compile():  # fresh DataArray, same dims/rank/dtype: the type is equal
        make(xr.DataArray(np.zeros((5, 5)), dims=("lat", "lon")))(0, 0)


def test_renamed_dims_are_a_different_type():
    xr = pytest.importorskip("xarray")

    def make(t):
        @jit()
        def f(a, b):
            return t.isel(lon=b, lat=a)

        return f

    da = xr.DataArray(np.ones((2, 2)), dims=("north", "east"))
    with pytest.raises(MissingRule, match="pedantic on purpose"):
        make(da)(0, 0)  # isel(lon=...) against north/east axes


def test_zero_dim_array_refused_cleanly():
    z = np.array(3.0)

    @jit()
    def f(x):
        return z[()] + x

    with pytest.raises(MissingRule, match="0-d array has no axes"):
        f(1.0)


def test_shaped_shape_drift_fails_loud():
    """Shape lives in the TYPE for Shaped captures, but guards are identity
    triples — an in-place metadata mutation must fail LOUD, never serve
    strides baked for the old shape (review-caught)."""
    x = np.arange(6.0).reshape(2, 3)

    def make(t):
        @jit()
        def f(i, j):
            return t[i, j]

        return f

    f = make(Shaped(x))
    assert f(1, 0) == 3.0
    x.resize(3, 2)  # in-place metadata mutation: same identity, new strides
    with pytest.raises(RuntimeError, match="Shaped capture drifted"):
        f(1, 0)


def test_named_and_positional_share_an_artifact():
    """The free lunch, pinned: names are ERASED at emission, so an isel
    kernel and its positional twin have identical content keys — two
    specializations, ONE compiled artifact."""
    ext = DEFAULT.extend()
    raw = np.arange(6.0).reshape(2, 3)

    def pos(t):
        @jit()
        def k(i, j):
            return t[i, j]

        return k

    def named(t):
        @jit()
        def k(i, j):
            return t.isel(y=i, x=j)

        return k

    ext.dispatch(pos(raw), (0, 1))
    ext.dispatch(named(Named(raw, ("y", "x"))), (0, 1))
    assert len(ext.specializations._ready) == 2
    assert len(ext.artifacts) == 1


def test_loop_over_captured_table_differential():
    t = np.array([0.5, 1.5, 2.5, 3.5])

    @jit()
    def f(x):
        acc = x
        for i in range(4):
            acc = acc + t[i]
        return acc

    assert abs(f(1.0) - (1.0 + t.sum())) < 1e-12
