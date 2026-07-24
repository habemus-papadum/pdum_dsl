"""The S.1 assemblage vocabulary: rmsnorm/layernorm/causal_softmax written
in the spec's own style (200 §6.1, nearly verbatim) lower through the
lifting machinery and match numpy; contract infers the unique shared axis
and refuses genuine ambiguity; the two-operand reduce form works."""

import numpy as np
import pytest
from pdum.tl import Tensor
from pdum.tl.ir import run
from pdum.tl.lifting import const_like, contract, iota_of, lift_step, reduce_over
from pdum.tl.mdsl import exp, where


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def _run1(fn, **tensors):
    ls = lift_step(fn, **{k: v.layout for k, v in tensors.items()})
    env = run(ls.program, tensors)
    return env[ls.outputs[0]]


# --- the spec's helpers, S.1 style (kwonly defaults, helper inlining) -------


def rmsnorm(x, g, *, feat="e", eps=1e-5):
    ms = (x * x).mean(feat)
    sd = (ms + eps).sqrt()
    xn = x / sd.repeat(feat, x.extent(feat))
    return xn * g.repeat_like(x, but=feat)


def layernorm(x, g, b, *, feat, eps=1e-5):
    mu = x.mean(feat)
    xc = x - mu.repeat(feat, x.extent(feat))
    sd = ((xc * xc).mean(feat) + eps).sqrt()
    return xc / sd.repeat(feat, x.extent(feat)) * g.repeat_like(x, but=feat) + b.repeat_like(x, but=feat)


def causal_softmax(sc, *, q="t", k="s"):
    mask = iota_of(sc, k) <= iota_of(sc, q)
    sm = where(mask, sc, const_like(sc, -1e9))
    e = exp(sm - sm.max(k).repeat_like(sm, dim=k))
    return e / e.sum(k).repeat_like(e, dim=k)


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


def test_contract_infers_the_unique_shared_axis():
    def step(a, w):
        return contract(a, w)

    rng = np.random.default_rng(6)
    a, w = rng.standard_normal((4, 6)), rng.standard_normal((6, 3))
    got = _run1(step, a=T(a, ("t", "d")), w=T(w, ("d", "m")))
    np.testing.assert_allclose(got.to_numpy(order=("t", "m")), a @ w, rtol=1e-12)


def test_contract_ambiguity_refuses_naming_the_fix():
    def step(q, k):
        return contract(q, k)

    rng = np.random.default_rng(7)
    q, k = rng.standard_normal((4, 2, 3)), rng.standard_normal((5, 2, 3))
    with pytest.raises(ValueError, match=r"shared axes \['hk', 'nh'\].*contract\(a, b, axis=...\)"):
        _run1(step, q=T(q, ("t", "nh", "hk")), k=T(k, ("s", "nh", "hk")))


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


def test_contract_no_shared_axis_refuses():
    def step(a, b):
        return contract(a, b)

    with pytest.raises(ValueError, match="no shared axis"):
        _run1(step, a=T(np.zeros(2), ("i",)), b=T(np.zeros(3), ("j",)))


# --- the two-operand reduce form + calling vocabulary outside a body --------


def test_reduce_over_the_two_operand_form():
    def step(se, ve):
        return reduce_over("zoo.flashsm", (se, ve), "s")

    import pdum.tl.zoo.attention  # noqa: F401 — registers zoo.flashsm

    rng = np.random.default_rng(10)
    s, v = rng.standard_normal(6), rng.standard_normal(6)
    got = float(_run1(step, se=T(s, ("s",)), ve=T(v, ("s",))).item())
    e = np.exp(s - s.max())
    np.testing.assert_allclose(got, (e * v).sum() / e.sum(), rtol=1e-9)


def test_vocabulary_refuses_being_called_outside_a_body():
    with pytest.raises(TypeError, match="lowers by inspection"):
        contract(1, 2)


def test_binding_names_become_ssa_names():
    def step(x, g):
        return rmsnorm(x, g, feat="e")

    ls = lift_step(step, x=T(np.zeros((2, 3)), ("t", "e")).layout, g=T(np.zeros(3), ("e",)).layout)
    assert {"ms", "sd", "xn"} <= set(ls.program.vars)  # the source reads back
