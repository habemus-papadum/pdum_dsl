"""A Llama-style block: RMSNorm, RoPE, GQA, SwiGLU.

RoPE without splits: the rotary pair structure is BORN in the weights —
wq is (d, g, r, c, u) with c the pair index and u the {re, im} slot — so
rotation is selects + pointwise trig, and the score contraction runs over
the structured feature dims directly (sum of the two u-slot contractions;
no concat, no interleave). GQA the same way: query heads are (g, r) —
kv-group × query-within-group — and K/V are simply repeated over r by
declaration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..build import Build
from .zoo_common import ZooModel, causal_softmax, contract, np_rmsnorm, np_sigmoid, np_softmax, rmsnorm, t_in


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


def llama_block(cfg: LlamaConfig = LlamaConfig(), seed: int = 11) -> ZooModel:
    rng = np.random.default_rng(seed)
    T, D, G, R, C, KV, M = cfg.t, cfg.d, cfg.g, cfg.r, cfg.c, cfg.kv, cfg.m
    b = Build()
    inputs: dict = {}

    def w(name, *shape, names):
        return t_in(inputs, name, 0.4 * rng.standard_normal(shape), names)

    x = t_in(inputs, "x", rng.standard_normal((T, D)), ("t", "d"))
    omega = t_in(inputs, "omega", cfg.base ** (-np.arange(C) / C), ("c",))
    for name in (x, omega):
        b.input(name)
    for nm, shape, names in (
        ("rms1g", (D,), ("d",)),
        ("wq", (D, G, R, C, 2), ("d", "g", "r", "c", "u")),
        ("wk", (D, G, C, 2), ("d", "g", "c", "u")),
        ("wv", (D, G, KV), ("d", "g", "kv")),
        ("wo", (G, R, KV, D), ("g", "r", "kv", "d")),
        ("rms2g", (D,), ("d",)),
        ("w1", (D, M), ("d", "m")),
        ("w3", (D, M), ("d", "m")),
        ("w2", (M, D), ("m", "d")),
    ):
        b.input(w(nm, *shape, names=names))

    td = [("t", (0, T)), ("d", (0, D))]
    a = rmsnorm(b, x, "d", (0, D), td, "rms1g", cfg.eps)
    a_s = b.emit("rename", (a,), hint="as", mapping={"t": "s"})
    # RoPE angles: theta[c, t] = t * omega_c — positions from iota, exactly
    ot = b.bcast(omega, [("t", (0, T))], hint="ot")
    pos = b.emit("iota", (ot,), hint="pos", name="t")
    th = b.pw("mul", pos, ot, hint="theta")
    cs, sn = b.pw("cos", th), b.pw("sin", th)
    cs_s = b.emit("rename", (cs,), hint="css", mapping={"t": "s"})
    sn_s = b.emit("rename", (sn,), hint="sns", mapping={"t": "s"})

    def rope(qv, reps, cos_v, sin_v):
        # qv: (..., c, u); rotate each pair by theta
        q0 = b.emit("select", (qv,), hint="q0", coords={"u": 0})
        q1 = b.emit("select", (qv,), hint="q1", coords={"u": 1})
        cb = b.bcast(cos_v, reps)
        sb = b.bcast(sin_v, reps)
        r0 = b.pw("sub", b.pw("mul", q0, cb), b.pw("mul", q1, sb), hint="rot0")
        r1 = b.pw("add", b.pw("mul", q0, sb), b.pw("mul", q1, cb), hint="rot1")
        return r0, r1

    q = contract(
        b, a, "wq", [("g", (0, G)), ("r", (0, R)), ("c", (0, C)), ("u", (0, 2))], [("t", (0, T))], ("d",), hint="q"
    )
    kk = contract(b, a_s, "wk", [("g", (0, G)), ("c", (0, C)), ("u", (0, 2))], [("s", (0, T))], ("d",), hint="k")
    q0, q1 = rope(q, [("g", (0, G)), ("r", (0, R))], cs, sn)
    k0, k1 = rope(kk, [("g", (0, G))], cs_s, sn_s)
    sc0 = contract(b, q0, k0, [("s", (0, T))], [("t", (0, T)), ("r", (0, R))], ("c",), hint="sc0")
    sc1 = contract(b, q1, k1, [("s", (0, T))], [("t", (0, T)), ("r", (0, R))], ("c",), hint="sc1")
    tsgr = [("t", (0, T)), ("s", (0, T)), ("g", (0, G)), ("r", (0, R))]
    sc = b.pw("mul", b.pw("add", sc0, sc1), b.const(1.0 / np.sqrt(2 * C), tsgr, hint="scale"), hint="sc")
    pr = causal_softmax(b, sc, "t", "s", tsgr)
    vv = contract(b, a_s, "wv", [("g", (0, G)), ("kv", (0, KV))], [("s", (0, T))], ("d",), hint="v")
    ctx = contract(b, pr, vv, [("kv", (0, KV))], [("t", (0, T)), ("r", (0, R))], ("s",), hint="ctx")
    o = contract(b, ctx, "wo", [("d", (0, D))], [("t", (0, T))], ("g", "r", "kv"), hint="o")
    h = b.pw("add", x, o, hint="hres")
    a2 = rmsnorm(b, h, "d", (0, D), td, "rms2g", cfg.eps)
    u1 = contract(b, a2, "w1", [("m", (0, M))], [("t", (0, T))], ("d",), hint="u1")
    u3 = contract(b, a2, "w3", [("m", (0, M))], [("t", (0, T))], ("d",), hint="u3")
    hh = b.pw("mul", b.pw("zoo.silu", u1, hint="silu"), u3, hint="gated")
    dn = contract(b, hh, "w2", [("d", (0, D))], [("t", (0, T))], ("m",), hint="down")
    out = b.pw("add", h, dn, hint="out")

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

    return ZooModel(b.program(), inputs, out, ref, ("t", "d"))
