"""GPT-2 as MAKERS (200 §6.2/6.3) — the flagship of the binding layer.

The library (layernorm, causal_softmax) is parameter-blind: functions from
tensors to tensors. The makers own names: declare-at-use ``s.param`` lines,
level-first paths via ``make_attn(s / "attn", cfg)``, and ``s.seq`` giving
``h.0.attn.wq, h.1.mlp.w1, ...`` — the naming law's worked example. Weights
are born structured: wq is (d, nh, hk), heads are dims, never splits (D5).
Starts from token IDS: embedding is ``take(wte, ids, dim="v")`` (the P9
gather — the old recorded boundary, dissolved), the head is TIED to the
same ``wte`` (contract over d), so d_wte accumulates through BOTH paths —
the scatter-add adjoint plus the contract adjoint, summed by fan-out;
learned positions are the ``wpe`` leaf added in."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..assemblage import assemblage, unit
from ..compute import const_like, contract, iota, pointwise, red, reduce, repeat_like
from ..indexing import take
from ..layout import Dim, _dense_like
from ..markers import exp, le, sqrt, where
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
    v: int = 5  # vocab: wte is tied — the §1.9 gather in, the head contract out
    eps: float = 1e-5


# --- library (parameter-blind): tensors to tensors, no scope, no names ------


def layernorm_t(x, g, b, *, feat, eps):
    mu = reduce(red.mean, x, feat)
    xc = x - repeat_like(mu, x)
    sd = pointwise(sqrt, reduce(red.mean, xc * xc, feat) + eps)
    return xc / repeat_like(sd, x) * repeat_like(g, x) + repeat_like(b, x)


def causal_softmax_t(sc, *, q="t", k="s"):
    mask = pointwise(le, iota(sc, k), iota(sc, q))
    sm = pointwise(where, mask, sc, const_like(sc, -1e9))
    e = pointwise(exp, sm - repeat_like(reduce(red.max, sm, k), sm))
    return e / repeat_like(reduce(red.sum, e, k), sm)


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
        q = contract(a, wq, axis="d")
        k = contract(a.rename(t="s"), wk, axis="d")
        v = contract(a.rename(t="s"), wv, axis="d")
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
        u = contract(a, w1, axis="d")
        m = pointwise(gelu, u + repeat_like(b1, u))
        return h + contract(m, w2, axis="m") + repeat_like(b2, h)

    return mlp


def make_block(s, cfg):
    return make_attn(s / "attn", cfg) | make_mlp(s / "mlp", cfg)


def make_gpt2(s, cfg):
    wte = s.param("wte", v=cfg.v, d=cfg.d)  # tied: embedding AND head
    wpe = s.param("wpe", t=cfg.t, d=cfg.d)
    lnfg, lnfb = s.param("lnfg", d=cfg.d), s.param("lnfb", d=cfg.d)

    @unit
    def embed(ids):
        return take(wte, ids, dim="v") + wpe  # the §1.9 gather

    trunk = s.seq("h", make_block, cfg, n=cfg.layers)  # h.0.attn.wq, h.1.mlp.w1, ...

    @unit
    def head(h):
        hf = layernorm_t(h, lnfg, lnfb, feat="d", eps=cfg.eps)
        return contract(hf, wte, axis="d")  # the tie: one leaf, two duties

    return embed | trunk | head


# --- the zoo entry: build, bind test values by contract name ----------------


def _t(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def gpt2(cfg: GPT2Config = GPT2Config(), seed: int = 7) -> ZooModel:
    rng = np.random.default_rng(seed)
    root = scope()
    ids_lay = _dense_like((Dim("t", 0, 0, cfg.t),))
    model = assemblage(make_gpt2(root, cfg), scope=root, ids=ids_lay)
    inputs = {"ids": Tensor.from_numpy(rng.integers(0, cfg.v, cfg.t), ("t",))}
    for name, p in root.coll.leaves.items():
        shape = tuple(e for _, e in p.dims)
        std = 0.1 if name == "wpe" else 0.4
        inputs[name] = _t(std * rng.standard_normal(shape), tuple(n for n, _ in p.dims))

    T, K = cfg.t, cfg.hk

    def ref(inp):
        h = inp["wte"][inp["ids"]] + inp["wpe"]
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
        return hf @ inp["wte"].T  # tied head

    return ZooModel(model.region, inputs, model.output, ref, ("t", "v"), model.names)
