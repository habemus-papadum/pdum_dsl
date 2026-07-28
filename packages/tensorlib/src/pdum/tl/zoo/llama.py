"""A Llama-style block as a MAKER: RMSNorm, RoPE, GQA, SwiGLU.

RoPE without splits: the rotary pair structure is BORN in the weights —
wq is (d, g, r, c, u) with c the pair index and u the {re, im} slot — so
rotation is selects + pointwise trig, and the score contraction runs over
the structured feature dims directly (sum of the two u-slot contractions;
no concat, no interleave). GQA the same way: query heads are (g, r) —
kv-group x query-within-group — and K/V are simply repeated over r by
declaration (contract's broadcast)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..assemblage import assemblage, unit
from ..compute import contract, extent, iota, pointwise, repeat_like
from ..layout import Dim, _dense_like
from ..markers import cos, sin
from ..scope import scope
from ..tensor import Tensor
from .zoo_common import ZooModel, causal_softmax, np_rmsnorm, np_sigmoid, np_softmax, rmsnorm, silu


@dataclass(frozen=True)
class LlamaConfig:
    t: int = 4  # sequence
    d: int = 6  # model width
    g: int = 2  # kv heads
    r: int = 2  # query heads per kv head
    c: int = 2  # rotary pairs (head width = 2c)
    kv: int = 3  # v head width
    m: int = 8  # mlp width
    eps: float = 1e-5
    base: float = 100.0  # rotary base (small: visible angles at toy sizes)


def _rope(qv, cos_v, sin_v):
    # qv: (..., c, u); rotate each pair by theta — selects + pointwise trig
    q0, q1 = qv.select(u=0), qv.select(u=1)
    cb = repeat_like(cos_v, q0)
    sb = repeat_like(sin_v, q0)
    return q0 * cb - q1 * sb, q0 * sb + q1 * cb


def make_llama_block(s, cfg):
    D, G, R, C, KV, M = cfg.d, cfg.g, cfg.r, cfg.c, cfg.kv, cfg.m
    omega = s.param("omega", c=C)
    rms1g = s.param("rms1g", d=D)
    wq = s.param("wq", d=D, g=G, r=R, c=C, u=2)
    wk = s.param("wk", d=D, g=G, c=C, u=2)
    wv = s.param("wv", d=D, g=G, kv=KV)
    wo = s.param("wo", g=G, r=R, kv=KV, d=D)
    rms2g = s.param("rms2g", d=D)
    w1 = s.param("w1", d=D, m=M)
    w3 = s.param("w3", d=D, m=M)
    w2 = s.param("w2", m=M, d=D)
    scale = 1.0 / np.sqrt(2 * C)

    @unit
    def block(x):
        a = rmsnorm(x, rms1g, feat="d", eps=cfg.eps)
        # RoPE angles: theta[t, c] = t * omega_c — positions from iota, exactly
        ot = omega.repeat("t", extent(x, "t"))  # structural construction: explicit
        th = iota(ot, "t") * ot
        cs, sn = pointwise(cos, th), pointwise(sin, th)
        q = contract(a, wq, axis="d")
        kk = contract(a.rename(t="s"), wk, axis="d")
        q0, q1 = _rope(q, cs, sn)
        k0, k1 = _rope(kk, cs.rename(t="s"), sn.rename(t="s"))
        sc = (contract(q0, k0, axis="c") + contract(q1, k1, axis="c")) * scale
        pr = causal_softmax(sc)
        vv = contract(a.rename(t="s"), wv, axis="d")
        ctx = contract(pr, vv, axis="s")
        o = contract(ctx, wo, axis=("g", "r", "kv"))
        h = x + o
        a2 = rmsnorm(h, rms2g, feat="d", eps=cfg.eps)
        hh = pointwise(silu, contract(a2, w1, axis="d")) * contract(a2, w3, axis="d")
        return h + contract(hh, w2, axis="m")

    return block


def llama_block(cfg: LlamaConfig = LlamaConfig(), seed: int = 11) -> ZooModel:
    rng = np.random.default_rng(seed)
    T, C = cfg.t, cfg.c
    root = scope()
    model = assemblage(
        make_llama_block(root, cfg),
        scope=root,
        x=_dense_like((Dim("t", 0, 0, T), Dim("d", 0, 0, cfg.d))),
    )
    inputs = {"x": _t(rng.standard_normal((T, cfg.d)), ("t", "d"))}
    inputs["omega"] = _t(cfg.base ** (-np.arange(C) / C), ("c",))
    for name, p in root.coll.leaves.items():
        if name == "omega":
            continue
        shape = tuple(e for _, e in p.dims)
        inputs[name] = _t(0.4 * rng.standard_normal(shape), tuple(n for n, _ in p.dims))

    def ref(inp):
        x, om = inp["x"], inp["omega"]
        a = np_rmsnorm(x, inp["rms1g"], cfg.eps)
        ang = np.arange(T)[:, None] * om[None, :]  # (t, c)
        cs, sn = np.cos(ang), np.sin(ang)
        q = np.einsum("td,dgrcu->tgrcu", a, inp["wq"])
        kk = np.einsum("sd,dgcu->sgcu", a, inp["wk"])
        q0 = q[..., 0] * cs[:, None, None, :] - q[..., 1] * sn[:, None, None, :]
        q1 = q[..., 0] * sn[:, None, None, :] + q[..., 1] * cs[:, None, None, :]
        k0 = kk[..., 0] * cs[:, None, :] - kk[..., 1] * sn[:, None, :]
        k1 = kk[..., 0] * sn[:, None, :] + kk[..., 1] * cs[:, None, :]
        sc = np.einsum("tgrc,sgc->tsgr", q0, k0) + np.einsum("tgrc,sgc->tsgr", q1, k1)
        sc = sc / np.sqrt(2 * C)
        mask = np.tril(np.ones((T, T), dtype=bool))
        sc = np.where(mask[:, :, None, None], sc, -1e9)
        pr = np_softmax(sc, axis=1)
        vv = np.einsum("sd,dgk->sgk", a, inp["wv"])
        ctx = np.einsum("tsgr,sgk->tgrk", pr, vv)
        h = x + np.einsum("tgrk,grkd->td", ctx, inp["wo"])
        a2 = np_rmsnorm(h, inp["rms2g"], cfg.eps)
        z1 = a2 @ inp["w1"]
        return h + (z1 * np_sigmoid(z1) * (a2 @ inp["w3"])) @ inp["w2"]

    return ZooModel(model.region, inputs, model.output, ref, ("t", "d"), model.names)


def _t(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)
