"""Attention variants — each isolating one mechanism, single-head sized.

The flagship is `flash_attention`: the online-softmax accumulator
(running max, running denominator, running weighted sum) as a composite
reducer whose associative rescaling combine is declared once — and whose
BACKWARD pass is therefore DERIVED by the composite-reducer BPTT machinery.
L0 states the algorithm; L4 will show the fusion is what makes it fast."""

from __future__ import annotations

import numpy as np

from ..build import Build
from ..mdsl import defreducer, exp, maximum
from .zoo_common import ZooModel, causal_softmax, contract, np_sigmoid, np_softmax, rmsnorm, softmax, t_in

# state (m, l, o): running max, denominator, weighted sum; the combine
# rescales both sides to the joint max. Associative (the online-softmax
# lemma); init is the monoid identity (exp(-1e30 - m) underflows to 0).
flashsm = defreducer(
    "zoo.flashsm",
    state=3,
    element=2,
    lift=lambda s, v: (s, 1.0, v),
    combine=lambda L, R: (
        maximum(L[0], R[0]),
        L[1] * exp(L[0] - maximum(L[0], R[0])) + R[1] * exp(R[0] - maximum(L[0], R[0])),
        L[2] * exp(L[0] - maximum(L[0], R[0])) + R[2] * exp(R[0] - maximum(L[0], R[0])),
    ),
    init=(-1e30, 0.0, 0.0),
    project=lambda m, den, o: o / den,
)


def _qkv(rng, inputs, b, T, E, OD):
    names = []
    for nm, shape, nn in (("q", (T, E), ("t", "e")), ("k", (T, E), ("s", "e")), ("v", (T, OD), ("s", "o"))):
        names.append(t_in(inputs, nm, rng.standard_normal(shape), nn))
    for nm in names:
        b.input(nm)
    return names


def sliding_attention(T=5, E=3, OD=2, W=2, seed=3) -> ZooModel:
    """Causal AND within-window: s <= t and t - s < W. Both masks are iota
    comparisons — closed forms, zero bytes."""
    rng = np.random.default_rng(seed)
    b = Build()
    inputs: dict = {}
    q, k, v = _qkv(rng, inputs, b, T, E, OD)
    ts = [("t", (0, T)), ("s", (0, T))]
    sc = contract(b, q, k, [("s", (0, T))], [("t", (0, T))], ("e",), hint="sc")
    it = b.emit("iota", (sc,), hint="it", name="t")
    isv = b.emit("iota", (sc,), hint="is", name="s")
    causal = b.pw("le", isv, it, hint="mc")
    dist = b.pw("sub", it, isv, hint="dist")
    wc = b.const(W, ts, hint="w", dtype="int64")
    inwin = b.pw("lt", dist, wc, hint="mw")
    m = b.pw("mul", causal, inwin, hint="mask")  # bool AND
    sm = b.pw("where", m, sc, b.const(-1e9, ts, hint="ninf"), hint="scm")
    pr = softmax(b, sm, "s", (0, T), ts)
    ctx = contract(b, pr, v, [("o", (0, OD))], [("t", (0, T))], ("s",), hint="ctx")

    def ref(inp):
        sc = inp["q"] @ inp["k"].T
        t, s = np.arange(T)[:, None], np.arange(T)[None, :]
        mask = (s <= t) & (t - s < W)
        return np_softmax(np.where(mask, sc, -1e9), axis=1) @ inp["v"]

    return ZooModel(b.program(), inputs, ctx, ref, ("t", "o"))


def gated_attention(T=5, E=3, OD=2, seed=4) -> ZooModel:
    """Output gating (Qwen3-Next style): out = sigmoid(q @ wg) ⊙ attention."""
    rng = np.random.default_rng(seed)
    b = Build()
    inputs: dict = {}
    q, k, v = _qkv(rng, inputs, b, T, E, OD)
    b.input(t_in(inputs, "wg", 0.5 * rng.standard_normal((E, OD)), ("e", "o")))
    ts = [("t", (0, T)), ("s", (0, T))]
    sc = contract(b, q, k, [("s", (0, T))], [("t", (0, T))], ("e",), hint="sc")
    pr = causal_softmax(b, sc, "t", "s", ts)
    ctx = contract(b, pr, v, [("o", (0, OD))], [("t", (0, T))], ("s",), hint="ctx")
    gate = b.pw("zoo.sigmoid", contract(b, q, "wg", [("o", (0, OD))], [("t", (0, T))], ("e",), hint="gz"), hint="gate")
    out = b.pw("mul", gate, ctx, hint="out")

    def ref(inp):
        sc = inp["q"] @ inp["k"].T
        mask = np.tril(np.ones((T, T), dtype=bool))
        ctx = np_softmax(np.where(mask, sc, -1e9), axis=1) @ inp["v"]
        return np_sigmoid(inp["q"] @ inp["wg"]) * ctx

    return ZooModel(b.program(), inputs, out, ref, ("t", "o"))


def qknorm_attention(T=5, E=3, OD=2, eps=1e-6, seed=5) -> ZooModel:
    """RMS-normalize q and k (with learned gains) before the scores."""
    rng = np.random.default_rng(seed)
    b = Build()
    inputs: dict = {}
    q, k, v = _qkv(rng, inputs, b, T, E, OD)
    for nm in ("gq", "gk"):
        b.input(t_in(inputs, nm, 1 + 0.1 * rng.standard_normal(E), ("e",)))
    qn = rmsnorm(b, q, "e", (0, E), [("t", (0, T)), ("e", (0, E))], "gq", eps)
    kn = rmsnorm(b, k, "e", (0, E), [("s", (0, T)), ("e", (0, E))], "gk", eps)
    ts = [("t", (0, T)), ("s", (0, T))]
    sc = contract(b, qn, kn, [("s", (0, T))], [("t", (0, T))], ("e",), hint="sc")
    pr = causal_softmax(b, sc, "t", "s", ts)
    ctx = contract(b, pr, v, [("o", (0, OD))], [("t", (0, T))], ("s",), hint="ctx")

    def ref(inp):
        def rms(x, g):
            return x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps) * g

        sc = rms(inp["q"], inp["gq"]) @ rms(inp["k"], inp["gk"]).T
        mask = np.tril(np.ones((T, T), dtype=bool))
        return np_softmax(np.where(mask, sc, -1e9), axis=1) @ inp["v"]

    return ZooModel(b.program(), inputs, ctx, ref, ("t", "o"))


def flash_attention(T=5, E=3, OD=2, seed=6, naive=False) -> ZooModel:
    """Masked scores fed to the online-softmax reducer (or, with
    naive=True, to materialized softmax — same denotation, different
    program; the pair is the fusion story's before/after)."""
    rng = np.random.default_rng(seed)
    b = Build()
    inputs: dict = {}
    q, k, v = _qkv(rng, inputs, b, T, E, OD)
    ts = [("t", (0, T)), ("s", (0, T))]
    sc = contract(b, q, k, [("s", (0, T))], [("t", (0, T))], ("e",), hint="sc")
    it = b.emit("iota", (sc,), hint="it", name="t")
    isv = b.emit("iota", (sc,), hint="is", name="s")
    m = b.pw("le", isv, it, hint="mask")
    sm = b.pw("where", m, sc, b.const(-1e9, ts, hint="ninf"), hint="scm")
    if naive:
        pr = softmax(b, sm, "s", (0, T), ts)
        out = contract(b, pr, v, [("o", (0, OD))], [("t", (0, T))], ("s",), hint="ctx")
    else:
        se = b.bcast(sm, [("o", (0, OD))], hint="se")
        ve = b.bcast(v, [("t", (0, T))], hint="ve")
        out = b.emit("reduce", (se, ve), hint="flash", f="zoo.flashsm", dims=("s",))

    def ref(inp):
        sc = inp["q"] @ inp["k"].T
        mask = np.tril(np.ones((T, T), dtype=bool))
        return np_softmax(np.where(mask, sc, -1e9), axis=1) @ inp["v"]

    return ZooModel(b.program(), inputs, out, ref, ("t", "o"))
