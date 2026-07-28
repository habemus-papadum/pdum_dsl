"""The S.1 assemblage vocabulary (amended P6): the library in the final
functional surface — pointwise/reduce application, repeat_like alignment,
mandatory contract axes — lowered AND eager, matching numpy."""

import numpy as np
import pytest
from pdum.tl import Tensor
from pdum.tl.compute import const_like, contract, iota, pointwise, red, reduce, repeat_like
from pdum.tl.dialect import run_named, walk_region
from pdum.tl.lifting import lift_step
from pdum.tl.markers import exp, le, sqrt, where


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def _run1(fn, **tensors):
    ls = lift_step(fn, **{k: v.layout for k, v in tensors.items()})
    return run_named(ls.region, tensors, ls.names)[ls.outputs[0]]


# --- the spec's helpers, S.1 style (kwonly defaults, helper inlining) -------


def rmsnorm(x, g, *, feat="e", eps=1e-5):
    ms = reduce(red.mean, x * x, feat)
    sd = pointwise(sqrt, ms + eps)
    xn = x / repeat_like(sd, x)
    return xn * repeat_like(g, x)


def layernorm(x, g, b, *, feat, eps=1e-5):
    mu = reduce(red.mean, x, feat)
    xc = x - repeat_like(mu, x)
    sd = pointwise(sqrt, reduce(red.mean, xc * xc, feat) + eps)
    return xc / repeat_like(sd, x) * repeat_like(g, x) + repeat_like(b, x)


def causal_softmax(sc, *, q="t", k="s"):
    mask = pointwise(le, iota(sc, k), iota(sc, q))
    sm = pointwise(where, mask, sc, const_like(sc, -1e9))
    e = pointwise(exp, sm - repeat_like(reduce(red.max, sm, k), sm))
    return e / repeat_like(reduce(red.sum, e, k), sm)


def test_rmsnorm_the_s1_worked_example():
    def step(x, g):
        return rmsnorm(x, g, feat="e")

    rng = np.random.default_rng(3)
    x, g = rng.standard_normal((4, 6)), rng.standard_normal(6)
    got = _run1(step, x=T(x, ("t", "e")), g=T(g, ("e",))).to_numpy(order=("t", "e"))
    want = x / np.sqrt((x * x).mean(axis=1, keepdims=True) + 1e-5) * g
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_layernorm_matches_numpy():
    def step(x, g, b):
        return layernorm(x, g, b, feat="e")

    rng = np.random.default_rng(4)
    x, g, b = rng.standard_normal((3, 5)), rng.standard_normal(5), rng.standard_normal(5)
    got = _run1(step, x=T(x, ("t", "e")), g=T(g, ("e",)), b=T(b, ("e",))).to_numpy(order=("t", "e"))
    mu = x.mean(axis=1, keepdims=True)
    xc = x - mu
    want = xc / np.sqrt((xc * xc).mean(axis=1, keepdims=True) + 1e-5) * g + b
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_causal_softmax_masks_by_closed_form():
    def step(sc):
        return causal_softmax(sc)

    rng = np.random.default_rng(5)
    sc = rng.standard_normal((4, 4))
    got = _run1(step, sc=T(sc, ("t", "s"))).to_numpy(order=("t", "s"))
    m = np.where(np.arange(4)[None, :] <= np.arange(4)[:, None], sc, -1e9)
    e = np.exp(m - m.max(axis=1, keepdims=True))
    np.testing.assert_allclose(got, e / e.sum(axis=1, keepdims=True), rtol=1e-12)


# --- contract: the unique shared axis, and the named ambiguity --------------


def test_contract_axis_is_mandatory_and_declared():
    """The owner ruling: contraction axes are structural knowledge the
    author HAS — no inference. The unknown part (batch dims) flows only
    through repeat_like."""
    with pytest.raises(TypeError, match="axis"):
        contract(T(np.zeros((2, 3)), ("t", "d")), T(np.zeros((3, 4)), ("d", "m")))

    def step(a, w):
        return contract(a, w, axis="d")

    rng = np.random.default_rng(6)
    a, w = rng.standard_normal((4, 6)), rng.standard_normal((6, 3))
    got = _run1(step, a=T(a, ("t", "d")), w=T(w, ("d", "m")))
    np.testing.assert_allclose(got.to_numpy(order=("t", "m")), a @ w, rtol=1e-12)


def test_repeat_like_makes_code_batching_unaware():
    """THE mechanism (220): the SAME body lowers for (t, d) and for
    (b, t, d) — the batch dim is never named; it flows in through
    repeat_like's layout-derived added-dim set, as ONE IR instruction."""

    def step(a, w):
        return contract(a, w, axis="d")

    rng = np.random.default_rng(13)
    w = rng.standard_normal((6, 3))
    plain = rng.standard_normal((4, 6))
    batched = rng.standard_normal((5, 4, 6))
    got = _run1(step, a=T(plain, ("t", "d")), w=T(w, ("d", "m")))
    np.testing.assert_allclose(got.to_numpy(order=("t", "m")), plain @ w, rtol=1e-12)
    gotb = _run1(step, a=T(batched, ("b", "t", "d")), w=T(w, ("d", "m")))
    np.testing.assert_allclose(
        gotb.to_numpy(order=("b", "t", "m")), np.einsum("btd,dm->btm", batched, w), rtol=1e-12
    )
    ls = lift_step(step, a=T(batched, ("b", "t", "d")).layout, w=T(w, ("d", "m")).layout)
    assert sum(1 for n in walk_region(ls.region) if n.op == "tl.repeat_like") == 2  # one per operand


def test_contract_named_axis_lets_heads_ride():
    def step(q, k):
        return contract(q, k, axis="hk")

    rng = np.random.default_rng(8)
    q, k = rng.standard_normal((4, 2, 3)), rng.standard_normal((5, 2, 3))
    got = _run1(step, q=T(q, ("t", "nh", "hk")), k=T(k, ("s", "nh", "hk")))
    want = np.einsum("tnk,snk->tsn", q, k)
    np.testing.assert_allclose(got.to_numpy(order=("t", "s", "nh")), want, rtol=1e-12)


def test_contract_tuple_axis():
    def step(cx, wo):
        return contract(cx, wo, axis=("nh", "hk"))

    rng = np.random.default_rng(9)
    cx, wo = rng.standard_normal((4, 2, 3)), rng.standard_normal((2, 3, 6))
    got = _run1(step, cx=T(cx, ("t", "nh", "hk")), wo=T(wo, ("nh", "hk", "d")))
    want = np.einsum("tnk,nkd->td", cx, wo)
    np.testing.assert_allclose(got.to_numpy(order=("t", "d")), want, rtol=1e-12)


def test_contract_bogus_axis_refuses_through_reduce():
    def step(a, b):
        return contract(a, b, axis="z")

    with pytest.raises(KeyError):
        _run1(step, a=T(np.zeros(2), ("i",)), b=T(np.zeros((2, 3)), ("i", "j")))


# --- the two-operand reduce form + calling vocabulary outside a body --------


def test_reduce_the_two_operand_form():
    from pdum.tl.zoo.attention import flashsm

    def step(se, ve):
        return reduce(flashsm, (se, ve), "s")

    rng = np.random.default_rng(10)
    s, v = rng.standard_normal(6), rng.standard_normal(6)
    got = float(_run1(step, se=T(s, ("s",)), ve=T(v, ("s",))).item())
    e = np.exp(s - s.max())
    np.testing.assert_allclose(got, (e * v).sum() / e.sum(), rtol=1e-9)


def test_contract_is_one_dual_mode_function():
    """contract is ONE visible line (repeat_like + mul + reduce) — the
    SAME plain function runs eagerly on real tensors and inlines when
    lowered; nothing recognizes it."""
    rng = np.random.default_rng(12)
    a, w = rng.standard_normal((4, 6)), rng.standard_normal((6, 3))
    eager = contract(T(a, ("t", "d")), T(w, ("d", "m")), axis="d")  # EXECUTED
    np.testing.assert_allclose(eager.to_numpy(order=("t", "m")), a @ w, rtol=1e-12)


def test_bare_marker_calls_refuse_on_tensors():
    """The owner ruling (P6): tensor-tier application is ALWAYS
    pointwise(cos, t) — eagerly AND in lowered bodies."""
    with pytest.raises(TypeError, match=r"spelled\s+pointwise\(cos"):
        from pdum.tl.markers import cos

        cos(T(np.zeros(2), ("i",)))

    def step(x):
        return exp(x)

    with pytest.raises(ValueError, match=r"spelled pointwise\(exp"):
        lift_step(step, x=T(np.zeros(2), ("i",)).layout)


def test_the_library_runs_eagerly_and_lowers_identically():
    """The naive backend IS the same code: the library executes eagerly on
    numpy-backed tensors, uncompiled — and the lowered region agrees."""
    rng = np.random.default_rng(11)
    x, g = rng.standard_normal((4, 6)), rng.standard_normal(6)
    xt, gt = T(x, ("t", "e")), T(g, ("e",))
    eager = rmsnorm(xt, gt, feat="e")  # EXECUTED, no IR anywhere
    ls = lift_step(lambda x, g: rmsnorm(x, g, feat="e"), x=xt.layout, g=gt.layout)
    lowered = run_named(ls.region, {"x": xt, "g": gt}, ls.names)[ls.outputs[0]]
    np.testing.assert_allclose(
        eager.to_numpy(order=("t", "e")), lowered.to_numpy(order=("t", "e")), rtol=1e-12
    )
    sc = rng.standard_normal((4, 4))
    egr = causal_softmax(T(sc, ("t", "s")))
    ls2 = lift_step(lambda sc: causal_softmax(sc), sc=T(sc, ("t", "s")).layout)
    low = run_named(ls2.region, {"sc": T(sc, ("t", "s"))}, ls2.names)[ls2.outputs[0]]
    np.testing.assert_allclose(
        egr.to_numpy(order=("t", "s")), low.to_numpy(order=("t", "s")), rtol=1e-12
    )


def test_binding_names_become_ssa_names():
    def step(x, g):
        return rmsnorm(x, g, feat="e")

    ls = lift_step(step, x=T(np.zeros((2, 3)), ("t", "e")).layout, g=T(np.zeros(3), ("e",)).layout)
    assert {"ms", "sd", "xn"} <= set(ls.names.values())  # the source reads back
