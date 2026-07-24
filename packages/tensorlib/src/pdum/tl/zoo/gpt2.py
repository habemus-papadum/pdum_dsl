"""GPT-2 as MAKERS (200 §6.2/6.3) — the flagship of the binding layer.

The library (layernorm, causal_softmax) is parameter-blind: functions from
tensors to tensors. The makers own names: declare-at-use ``s.param`` lines,
level-first paths via ``make_attn(s / "attn", cfg)``, and ``s.seq`` giving
``h.0.attn.wq, h.1.mlp.w1, ...`` — the naming law's worked example. Weights
are born structured: wq is (d, nh, hk), heads are dims, never splits (D5).
Starts from hidden states (token embedding is a gather — a recorded
boundary); learned positions are the ``wpe`` leaf added in."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..assemblage import assemblage, unit
from ..ir import _dense_like
from ..layout import Dim
from ..lifting import const_like, contract, iota_of
from ..mdsl import exp, where
from ..scope import scope
from ..tensor import Tensor
from .zoo_common import ZooModel, gelu, np_gelu, np_layernorm, np_softmax


@dataclass(frozen=True)
class GPT2Config:
    t: int = 4  # sequence
    d: int = 6  # model width
    nh: int = 2  # heads
    hk: int = 3  # head width
    m: int = 8  # mlp width
    layers: int = 2
    v: int = 5  # vocab (head only; embedding gather is a recorded boundary)
    eps: float = 1e-5


# --- library (parameter-blind): tensors to tensors, no scope, no names ------


def layernorm_t(x, g, b, *, feat, eps):
    mu = x.mean(feat)
    xc = x - mu.repeat(feat, x.extent(feat))
    sd = ((xc * xc).mean(feat) + eps).sqrt()
    return xc / sd.repeat(feat, x.extent(feat)) * g.repeat_like(x, but=feat) + b.repeat_like(x, but=feat)


def causal_softmax_t(sc, *, q="t", k="s"):
    mask = iota_of(sc, k) <= iota_of(sc, q)
    sm = where(mask, sc, const_like(sc, -1e9))
    e = exp(sm - sm.max(k).repeat_like(sm, dim=k))
    return e / e.sum(k).repeat_like(e, dim=k)


# --- the makers (binding layer): declare-at-use -----------------------------


def make_attn(s, cfg):
    D, H, K = cfg.d, cfg.nh, cfg.hk
    ln1g, ln1b = s.param("ln1g", d=D), s.param("ln1b", d=D)
    wq = s.param("wq", d=D, nh=H, hk=K)
    wk = s.param("wk", d=D, nh=H, hk=K)
    wv = s.param("wv", d=D, nh=H, hk=K)
    wo = s.param("wo", nh=H, hk=K, d=D)
    scale = 1.0 / math.sqrt(K)

    @unit
    def attn(h):
        a = layernorm_t(h, ln1g, ln1b, feat="d", eps=cfg.eps)
        q = contract(a, wq)  # unique shared axis: "d"
        k = contract(a.rename(t="s"), wk)
        v = contract(a.rename(t="s"), wv)
        sc = contract(q * scale, k, axis="hk")  # "nh" rides; axis breaks the ambiguity
        pr = causal_softmax_t(sc)
        cx = contract(pr, v, axis="s")
        o = contract(cx, wo, axis=("nh", "hk"))
        return h + o

    return attn


def make_mlp(s, cfg):
    D, M = cfg.d, cfg.m
    ln2g, ln2b = s.param("ln2g", d=D), s.param("ln2b", d=D)
    w1, b1 = s.param("w1", d=D, m=M), s.param("b1", m=M)
    w2, b2 = s.param("w2", m=M, d=D), s.param("b2", d=D)

    @unit
    def mlp(h):
        a = layernorm_t(h, ln2g, ln2b, feat="d", eps=cfg.eps)
        m = gelu(contract(a, w1) + b1.repeat_like(a, dim="t"))
        return h + contract(m, w2) + b2.repeat_like(h, but="d")

    return mlp


def make_block(s, cfg):
    return make_attn(s / "attn", cfg) | make_mlp(s / "mlp", cfg)


def make_gpt2(s, cfg):
    wpe = s.param("wpe", t=cfg.t, d=cfg.d)
    lnfg, lnfb = s.param("lnfg", d=cfg.d), s.param("lnfb", d=cfg.d)
    wlm = s.param("wlm", d=cfg.d, v=cfg.v)

    @unit
    def embed(x):
        return x + wpe

    trunk = s.seq("h", make_block, cfg, n=cfg.layers)  # h.0.attn.wq, h.1.mlp.w1, ...

    @unit
    def head(h):
        hf = layernorm_t(h, lnfg, lnfb, feat="d", eps=cfg.eps)
        return contract(hf, wlm)

    return embed | trunk | head


# --- the zoo entry: build, bind test values by contract name ----------------


def _t(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def gpt2(cfg: GPT2Config = GPT2Config(), seed: int = 7) -> ZooModel:
    rng = np.random.default_rng(seed)
    root = scope()
    xlay = _dense_like((Dim("t", 0, 0, cfg.t), Dim("d", 0, 0, cfg.d)))
    model = assemblage(make_gpt2(root, cfg), scope=root, x=xlay)
    inputs = {"x": _t(rng.standard_normal((cfg.t, cfg.d)), ("t", "d"))}
    for name, p in root.coll.leaves.items():
        shape = tuple(e for _, e in p.dims)
        std = 0.1 if name == "wpe" else 0.4
        inputs[name] = _t(std * rng.standard_normal(shape), tuple(n for n, _ in p.dims))

    T, K = cfg.t, cfg.hk

    def ref(inp):
        h = inp["x"] + inp["wpe"]
        mask = np.tril(np.ones((T, T), dtype=bool))
        for i in range(cfg.layers):
            p = f"h.{i}."
            a = np_layernorm(h, inp[p + "attn.ln1g"], inp[p + "attn.ln1b"], cfg.eps)
            q = np.einsum("td,dhk->thk", a, inp[p + "attn.wq"]) / np.sqrt(K)
            kk = np.einsum("sd,dhk->shk", a, inp[p + "attn.wk"])
            vv = np.einsum("sd,dhk->shk", a, inp[p + "attn.wv"])
            sc = np.einsum("thk,shk->tsh", q, kk)
            sc = np.where(mask[:, :, None], sc, -1e9)
            pr = np_softmax(sc, axis=1)
            ctx = np.einsum("tsh,shk->thk", pr, vv)
            h = h + np.einsum("thk,hkd->td", ctx, inp[p + "attn.wo"])
            a2 = np_layernorm(h, inp[p + "mlp.ln2g"], inp[p + "mlp.ln2b"], cfg.eps)
            mm = np_gelu(a2 @ inp[p + "mlp.w1"] + inp[p + "mlp.b1"])
            h = h + mm @ inp[p + "mlp.w2"] + inp[p + "mlp.b2"]
        hf = np_layernorm(h, inp["lnfg"], inp["lnfb"], cfg.eps)
        return hf @ inp["wlm"]

    return ZooModel(model.program, inputs, model.output, ref, ("t", "v"))
