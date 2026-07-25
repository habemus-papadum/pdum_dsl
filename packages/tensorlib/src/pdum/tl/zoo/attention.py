"""Attention variants — each isolating one mechanism, single-head sized,
authored as MAKERS over the S.1 vocabulary (the binding layer owns names;
q flows, k/v and gains are declared leaves at the root scope).

The flagship is `flash_attention`: the online-softmax accumulator
(running max, running denominator, running weighted sum) as a composite
reducer whose associative rescaling combine is declared once — and whose
BACKWARD pass is therefore DERIVED by the composite-reducer BPTT machinery.
L0 states the algorithm; L4 will show the fusion is what makes it fast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..assemblage import assemblage, unit
from ..ir import _dense_like
from ..layout import Dim
from ..lifting import const_like, contract, iota_of, reduce_over
from ..mdsl import defreducer, exp, maximum, where
from ..scope import scope
from ..tensor import Tensor
from .zoo_common import ZooModel, causal_softmax, np_sigmoid, np_softmax, rmsnorm, sigmoid, softmax


@dataclass(frozen=True)
class SoftmaxState:
    """The online-softmax accumulator: running max, denominator, weighted sum."""

    m: object
    den: object
    o: object


def _flash_lift(s, v):
    return SoftmaxState(s, 1.0, v)


def _flash_combine(L, R):
    # rescale both sides to the joint max — associative (the online-softmax
    # lemma); the AD machinery differentiates this BY INSPECTION, which is
    # why flash's backward is derived, not hand-written (S.2)
    m = maximum(L.m, R.m)
    sl, sr = exp(L.m - m), exp(R.m - m)
    return SoftmaxState(m, L.den * sl + R.den * sr, L.o * sl + R.o * sr)


def _flash_project(s):
    return s.o / s.den


# init is the monoid identity (exp(-1e30 - m) underflows to 0)
flashsm = defreducer(
    "zoo.flashsm",
    state=SoftmaxState,
    element=2,
    lift=_flash_lift,
    combine=_flash_combine,
    init=SoftmaxState(-1e30, 0.0, 0.0),
    project=_flash_project,
)


def _t(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def _qkv_scope(rng, root, T, E, OD):
    k = root.param("k", s=T, e=E)
    v = root.param("v", s=T, o=OD)
    inputs = {
        "q": _t(rng.standard_normal((T, E)), ("t", "e")),
        "k": _t(rng.standard_normal((T, E)), ("s", "e")),
        "v": _t(rng.standard_normal((T, OD)), ("s", "o")),
    }
    return k, v, inputs


def _qlay(T, E):
    return _dense_like((Dim("t", 0, 0, T), Dim("e", 0, 0, E)))


def sliding_attention(T=5, E=3, OD=2, W=2, seed=3) -> ZooModel:
    """Causal AND within-window: s <= t and t - s < W. Both masks are iota
    comparisons — closed forms, zero bytes."""
    rng = np.random.default_rng(seed)
    root = scope()
    k, v, inputs = _qkv_scope(rng, root, T, E, OD)

    @unit
    def attend(q):
        sc = contract(q, k)  # unique shared axis: "e"
        causal = iota_of(sc, "s") <= iota_of(sc, "t")
        inwin = (iota_of(sc, "t") - iota_of(sc, "s")) < W
        m = causal * inwin  # bool AND
        pr = softmax(where(m, sc, const_like(sc, -1e9)), k="s")
        return contract(pr, v, axis="s")

    model = assemblage(attend, scope=root, q=_qlay(T, E))

    def ref(inp):
        sc = inp["q"] @ inp["k"].T
        t, s = np.arange(T)[:, None], np.arange(T)[None, :]
        mask = (s <= t) & (t - s < W)
        return np_softmax(np.where(mask, sc, -1e9), axis=1) @ inp["v"]

    return ZooModel(model.program, inputs, model.output, ref, ("t", "o"))


def gated_attention(T=5, E=3, OD=2, seed=4) -> ZooModel:
    """Output gating (Qwen3-Next style): out = sigmoid(q @ wg) x attention."""
    rng = np.random.default_rng(seed)
    root = scope()
    k, v, inputs = _qkv_scope(rng, root, T, E, OD)
    wg = root.param("wg", e=E, o=OD)
    inputs["wg"] = _t(0.5 * rng.standard_normal((E, OD)), ("e", "o"))

    @unit
    def attend(q):
        pr = causal_softmax(contract(q, k))
        ctx = contract(pr, v, axis="s")
        gate = sigmoid(contract(q, wg))
        return gate * ctx

    model = assemblage(attend, scope=root, q=_qlay(T, E))

    def ref(inp):
        sc = inp["q"] @ inp["k"].T
        mask = np.tril(np.ones((T, T), dtype=bool))
        ctx = np_softmax(np.where(mask, sc, -1e9), axis=1) @ inp["v"]
        return np_sigmoid(inp["q"] @ inp["wg"]) * ctx

    return ZooModel(model.program, inputs, model.output, ref, ("t", "o"))


def qknorm_attention(T=5, E=3, OD=2, eps=1e-6, seed=5) -> ZooModel:
    """RMS-normalize q and k (with learned gains) before the scores."""
    rng = np.random.default_rng(seed)
    root = scope()
    k, v, inputs = _qkv_scope(rng, root, T, E, OD)
    gq = root.param("gq", e=E)
    gk = root.param("gk", e=E)
    for nm in ("gq", "gk"):
        inputs[nm] = _t(1 + 0.1 * rng.standard_normal(E), ("e",))

    @unit
    def attend(q):
        qn = rmsnorm(q, gq, feat="e", eps=eps)
        kn = rmsnorm(k, gk, feat="e", eps=eps)
        pr = causal_softmax(contract(qn, kn))
        return contract(pr, v, axis="s")

    model = assemblage(attend, scope=root, q=_qlay(T, E))

    def ref(inp):
        def rms(x, g):
            return x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps) * g

        sc = rms(inp["q"], inp["gq"]) @ rms(inp["k"], inp["gk"]).T
        mask = np.tril(np.ones((T, T), dtype=bool))
        return np_softmax(np.where(mask, sc, -1e9), axis=1) @ inp["v"]

    return ZooModel(model.program, inputs, model.output, ref, ("t", "o"))


def flash_attention(T=5, E=3, OD=2, seed=6, naive=False) -> ZooModel:
    """Masked scores fed to the online-softmax reducer (or, with
    naive=True, to materialized softmax — same denotation, different
    program; the pair is the fusion story's before/after)."""
    rng = np.random.default_rng(seed)
    root = scope()
    k, v, inputs = _qkv_scope(rng, root, T, E, OD)

    @unit
    def attend_naive(q):
        pr = causal_softmax(contract(q, k))
        return contract(pr, v, axis="s")

    @unit
    def attend_flash(q):
        sc = contract(q, k)
        m = iota_of(sc, "s") <= iota_of(sc, "t")
        sm = where(m, sc, const_like(sc, -1e9))
        se = sm.repeat_like(v, dim="o")
        ve = v.repeat_like(sm, dim="t")
        return reduce_over("zoo.flashsm", (se, ve), "s")

    model = assemblage(attend_naive if naive else attend_flash, scope=root, q=_qlay(T, E))

    def ref(inp):
        sc = inp["q"] @ inp["k"].T
        mask = np.tril(np.ones((T, T), dtype=bool))
        return np_softmax(np.where(mask, sc, -1e9), axis=1) @ inp["v"]

    return ZooModel(model.program, inputs, model.output, ref, ("t", "o"))
