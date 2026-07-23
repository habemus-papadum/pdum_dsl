"""GPT-2: LayerNorm + causal MHA + GELU MLP + residuals — the baseline canon.

Starts from hidden states (token embedding is a gather — a recorded
boundary); learned positions are just another input tensor added in.
Weights are born structured: wq is (d, nh, hk), so heads are dims, never
splits (D5 names-first)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..build import Build
from .zoo_common import ZooModel, causal_softmax, contract, layernorm, np_gelu, np_layernorm, np_softmax, t_in


@dataclass(frozen=True)
class GPT2Config:
    t: int = 4  # sequence
    d: int = 6  # model width
    nh: int = 2  # heads
    hk: int = 3  # head width
    m: int = 8  # mlp width
    layers: int = 2
    v: int = 5  # vocab (head only; no embedding gather)
    eps: float = 1e-5


def gpt2(cfg: GPT2Config = GPT2Config(), seed: int = 7) -> ZooModel:
    rng = np.random.default_rng(seed)
    T, D, H, K, M, V = cfg.t, cfg.d, cfg.nh, cfg.hk, cfg.m, cfg.v
    b = Build()
    inputs: dict = {}

    def w(name, *shape, names):
        return t_in(inputs, name, 0.4 * rng.standard_normal(shape), names)

    x = t_in(inputs, "x", rng.standard_normal((T, D)), ("t", "d"))
    pos = t_in(inputs, "pos", 0.1 * rng.standard_normal((T, D)), ("t", "d"))
    b.input(x)
    b.input(pos)
    td = [("t", (0, T)), ("d", (0, D))]
    thk = [("t", (0, T)), ("nh", (0, H)), ("hk", (0, K))]
    tsh = [("t", (0, T)), ("s", (0, T)), ("nh", (0, H))]
    h = b.pw("add", x, pos, hint="h0")
    for i in range(cfg.layers):
        p = f"L{i}."
        for nm, shape, names in (
            (p + "ln1g", (D,), ("d",)),
            (p + "ln1b", (D,), ("d",)),
            (p + "wq", (D, H, K), ("d", "nh", "hk")),
            (p + "wk", (D, H, K), ("d", "nh", "hk")),
            (p + "wv", (D, H, K), ("d", "nh", "hk")),
            (p + "wo", (H, K, D), ("nh", "hk", "d")),
            (p + "ln2g", (D,), ("d",)),
            (p + "ln2b", (D,), ("d",)),
            (p + "w1", (D, M), ("d", "m")),
            (p + "b1", (M,), ("m",)),
            (p + "w2", (M, D), ("m", "d")),
            (p + "b2", (D,), ("d",)),
        ):
            b.input(w(nm, *shape, names=names))
        a = layernorm(b, h, "d", (0, D), td, p + "ln1g", p + "ln1b", cfg.eps)
        a_s = b.emit("rename", (a,), hint="as", mapping={"t": "s"})
        heads = [("nh", (0, H)), ("hk", (0, K))]
        q = contract(b, a, p + "wq", heads, [("t", (0, T))], ("d",), hint="q")
        kk = contract(b, a_s, p + "wk", heads, [("s", (0, T))], ("d",), hint="k")
        vv = contract(b, a_s, p + "wv", heads, [("s", (0, T))], ("d",), hint="v")
        qs = b.pw("mul", q, b.const(1.0 / np.sqrt(K), thk, hint="scale"), hint="qs")
        sc = contract(b, qs, kk, [("s", (0, T))], [("t", (0, T))], ("hk",), hint="sc")
        pr = causal_softmax(b, sc, "t", "s", tsh)
        ctx = contract(b, pr, vv, [("hk", (0, K))], [("t", (0, T))], ("s",), hint="ctx")
        o = contract(b, ctx, p + "wo", [("d", (0, D))], [("t", (0, T))], ("nh", "hk"), hint="o")
        h = b.pw("add", h, o, hint="hres")
        a2 = layernorm(b, h, "d", (0, D), td, p + "ln2g", p + "ln2b", cfg.eps)
        m1 = contract(b, a2, p + "w1", [("m", (0, M))], [("t", (0, T))], ("d",), hint="m1")
        m1b = b.pw("add", m1, b.bcast(p + "b1", [("t", (0, T))]), hint="m1b")
        gg = b.pw("zoo.gelu", m1b, hint="gelu")
        m2 = contract(b, gg, p + "w2", [("d", (0, D))], [("t", (0, T))], ("m",), hint="m2")
        m2b = b.pw("add", m2, b.bcast(p + "b2", [("t", (0, T))]), hint="m2b")
        h = b.pw("add", h, m2b, hint="hres")
    b.input(w("lnfg", D, names=("d",)))
    b.input(w("lnfb", D, names=("d",)))
    b.input(w("wlm", D, V, names=("d", "v")))
    hf = layernorm(b, h, "d", (0, D), td, "lnfg", "lnfb", cfg.eps)
    logits = contract(b, hf, "wlm", [("v", (0, V))], [("t", (0, T))], ("d",), hint="logits")

    def ref(inp):
        h = inp["x"] + inp["pos"]
        mask = np.tril(np.ones((T, T), dtype=bool))
        for i in range(cfg.layers):
            p = f"L{i}."
            a = np_layernorm(h, inp[p + "ln1g"], inp[p + "ln1b"], cfg.eps)
            q = np.einsum("td,dhk->thk", a, inp[p + "wq"]) / np.sqrt(K)
            kk = np.einsum("sd,dhk->shk", a, inp[p + "wk"])
            vv = np.einsum("sd,dhk->shk", a, inp[p + "wv"])
            sc = np.einsum("thk,shk->tsh", q, kk)
            sc = np.where(mask[:, :, None], sc, -1e9)
            pr = np_softmax(sc, axis=1)
            ctx = np.einsum("tsh,shk->thk", pr, vv)
            h = h + np.einsum("thk,hkd->td", ctx, inp[p + "wo"])
            a2 = np_layernorm(h, inp[p + "ln2g"], inp[p + "ln2b"], cfg.eps)
            mm = np_gelu(a2 @ inp[p + "w1"] + inp[p + "b1"])
            h = h + mm @ inp[p + "w2"] + inp[p + "b2"]
        hf = np_layernorm(h, inp["lnfg"], inp["lnfb"], cfg.eps)
        return hf @ inp["wlm"]

    return ZooModel(b.program(), inputs, logits, ref, ("t", "v"))
