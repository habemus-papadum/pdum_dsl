"""Stage 2b — the GRID family: params are integer domain coordinates, and
the domain loop lives IN the artifact. One dispatch fills the out array —
the answer to the ray-march verdict (per-lane dispatch drowned a 7× body
win). `over` composes unchanged: the lane is one more coordinate."""

import pytest

np = pytest.importorskip("numpy")

import pdum.dsl  # noqa: F401, E402
from pdum.dsl.backends import c  # noqa: E402
from pdum.dsl.kernel.api import jit  # noqa: E402
from pdum.dsl.kernel.cache import no_compile  # noqa: E402
from pdum.dsl.kernel.registry import DEFAULT  # noqa: E402
from pdum.dsl.stdlib.arrays import Named  # noqa: E402
from pdum.dsl.stdlib.transforms import over  # noqa: E402

pytestmark = pytest.mark.skipif(not c.is_available(), reason="no C compiler")


def grid_registry():
    ext = DEFAULT.extend()
    c.install_grid(ext, default=True)
    return ext


def test_grid_fills_a_2d_domain_in_one_dispatch():
    def make(scale):
        @jit()
        def cell(i, j):
            return float(i) * scale + float(j)

        return cell

    g = grid_registry()
    out = g.dispatch(make(10.0), (), np.empty((3, 4)))
    assert np.allclose(out, [[i * 10.0 + j for j in range(4)] for i in range(3)])
    with no_compile():  # a new DOMAIN is not identity (out never enters the key)
        g.dispatch(make(10.0), (), np.empty((7, 2)))


def test_grid_allocates_from_a_shape_tuple():
    @jit()
    def cell(i):
        return float(i) * 3.0

    g = grid_registry()
    out = g.dispatch(cell, (), (5,))
    assert out.shape == (5,) and out[4] == 12.0


def test_overed_kernel_is_one_more_coordinate():
    data = Named(np.arange(8.0).reshape(2, 4), ("batch", "x"))

    def make(t):
        @jit()
        def g(k):
            return t.isel(x=k) * 10.0

        return g

    vg = over(make(data), axis="batch")
    out = grid_registry().dispatch(vg, (), np.empty((4, 2)))  # (k, lane)
    assert np.allclose(out, data.array.T * 10.0)


def make_attn(Q, K, V, S, scale):
    @jit()
    def cell(t, d):
        den = 0.0
        for s in range(S):
            den = den + exp(matmul(Q, K, t, s) * scale)  # noqa: F821
        acc = 0.0
        for s in range(S):
            w = exp(matmul(Q, K, t, s) * scale) / den  # noqa: F821
            acc = acc + w * V.isel(kseq=s, dim=d)
        return acc

    return cell


def ref_attn(q, k, v, scale):
    sc = (q @ np.swapaxes(k, -1, -2)) * scale
    w = np.exp(sc) / np.exp(sc).sum(-1, keepdims=True)
    return w @ v


def test_batched_attention_on_the_grid_matches_numpy():
    rng = np.random.default_rng(0)
    B, S, E, Dv = 3, 5, 4, 4
    qb = Named(rng.standard_normal((B, S, E)), ("batch", "seq", "embed"))
    kb = Named(rng.standard_normal((B, S, E)), ("batch", "kseq", "embed"))
    vb = Named(rng.standard_normal((B, S, Dv)), ("batch", "kseq", "dim"))
    bcell = over(make_attn(qb, kb, vb, S, 1 / np.sqrt(E)), axis="batch")
    got = grid_registry().dispatch(bcell, (), np.empty((S, Dv, B)))
    assert np.allclose(np.moveaxis(got, -1, 0), ref_attn(qb.array, kb.array, vb.array, 1 / np.sqrt(E)))


def test_the_gate_one_dispatch_beats_per_lane_by_10x():
    """The stage-2 exit gate (020 step 14): batch-ignorant attention,
    over-batched, ≥10× per-lane dispatch on C. Bench-shaped: retry once."""
    from pdum.dsl.bench import benchmark

    rng = np.random.default_rng(0)
    B, S, E, Dv = 8, 16, 8, 8
    qb = Named(rng.standard_normal((B, S, E)), ("batch", "seq", "embed"))
    kb = Named(rng.standard_normal((B, S, E)), ("batch", "kseq", "embed"))
    vb = Named(rng.standard_normal((B, S, Dv)), ("batch", "kseq", "dim"))
    bcell = over(make_attn(qb, kb, vb, S, 1 / np.sqrt(E)), axis="batch")
    grid, scalar = grid_registry(), DEFAULT.extend()
    c.install(scalar, default=True)

    def measure():
        per_lane = benchmark(
            lambda: [[[scalar.dispatch(bcell, (t, d, b)) for b in range(B)] for d in range(Dv)] for t in range(S)],
            budget_s=0.3,
        )
        one = benchmark(lambda: grid.dispatch(bcell, (), np.empty((S, Dv, B))), budget_s=0.3)
        return per_lane.minimum / one.minimum

    speed = measure()
    if speed < 10:  # one-sided noise: a real regression fails twice
        speed = measure()
    assert speed >= 10, f"gate: {speed:.1f}x"
