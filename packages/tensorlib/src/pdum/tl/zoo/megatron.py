"""Megatron-style tensor parallelism as PLACEMENT metadata (L3-lite),
authored as a MAKER — placement declared AT THE LEAF.

One transformer block, the global-view way (PLACEMENT.md): attention heads
carry a mesh dim `g` (column-parallel QKV, heads sharded), the MLP width
carries the same `g` (column-parallel up, row-parallel down) — and `g` is
BOUND via ``s.param(..., bind={"g": level})``, so every broadcast or
constant introduced against a placed leaf binds automatically (rep_dim /
const_like copy the source dim's level). No collective ops appear anywhere:
the two per-block all-reduces Megatron's paper prescribes are simply the
two reduce-over-`g` contractions (attention output projection, MLP down
projection), which the traffic pass reads off the algebra.

`level=None` builds the ERASURE — the identical program minus bindings —
for the denotation-preservation check: placed and erased runs must agree
bit-for-bit, because placement is cost-bearing metadata, never meaning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..assemblage import assemblage, unit
from ..ir import _dense_like
from ..layout import Dim
from ..lifting import contract
from ..scope import scope
from ..tensor import Tensor
from .zoo_common import ZooModel, causal_softmax, gelu, layernorm, np_gelu, np_layernorm, np_softmax


@dataclass(frozen=True)
class MegatronConfig:
    t: int = 4  # sequence
    d: int = 6  # model width
    g: int = 2  # mesh extent (tensor-parallel ways)
    hl: int = 1  # heads per device (nh = g * hl)
    hk: int = 3  # head width
    ml: int = 4  # mlp width per device (m = g * ml)
    eps: float = 1e-5


def make_megatron_block(s, cfg, level):
    D, G, HL, HK, ML = cfg.d, cfg.g, cfg.hl, cfg.hk, cfg.ml
    gbind = {"g": level} if level is not None else None
    ln1g, ln1b = s.param("ln1g", d=D), s.param("ln1b", d=D)
    wq = s.param("wq", bind=gbind, d=D, g=G, hl=HL, hk=HK)
    wk = s.param("wk", bind=gbind, d=D, g=G, hl=HL, hk=HK)
    wv = s.param("wv", bind=gbind, d=D, g=G, hl=HL, hk=HK)
    wo = s.param("wo", bind=gbind, g=G, hl=HL, hk=HK, d=D)
    ln2g, ln2b = s.param("ln2g", d=D), s.param("ln2b", d=D)
    w1 = s.param("w1", bind=gbind, d=D, g=G, ml=ML)
    b1 = s.param("b1", bind=gbind, g=G, ml=ML)
    w2 = s.param("w2", bind=gbind, g=G, ml=ML, d=D)
    b2 = s.param("b2", d=D)
    scale = 1.0 / np.sqrt(HK)

    @unit
    def block(x):
        a = layernorm(x, ln1g, ln1b, feat="d", eps=cfg.eps)
        q = contract(a, wq)  # placement rides: broadcasts against wq bind g
        kk = contract(a.rename(t="s"), wk)
        vv = contract(a.rename(t="s"), wv)
        sc = contract(q * scale, kk, axis="hk")
        pr = causal_softmax(sc)
        ctx = contract(pr, vv, axis="s")
        o = contract(ctx, wo, axis=("g", "hl", "hk"))  # all-reduce #1
        h = x + o
        a2 = layernorm(h, ln2g, ln2b, feat="d", eps=cfg.eps)
        a1 = contract(a2, w1)
        gg = gelu(a1 + b1.repeat_like(a1, dim="t"))
        m2 = contract(gg, w2, axis=("g", "ml"))  # all-reduce #2
        return h + m2 + b2.repeat_like(h, but="d")

    return block


def megatron_block(cfg: MegatronConfig = MegatronConfig(), level: str | None = "gpu", seed: int = 29) -> ZooModel:
    rng = np.random.default_rng(seed)
    T = cfg.t
    root = scope()
    model = assemblage(
        make_megatron_block(root, cfg, level),
        scope=root,
        x=_dense_like((Dim("t", 0, 0, T), Dim("d", 0, 0, cfg.d))),
    )
    inputs = {"x": _t(rng.standard_normal((T, cfg.d)), ("t", "d"))}
    for name, p in root.coll.leaves.items():
        shape = tuple(e for _, e in p.dims)
        tensor = _t(0.4 * rng.standard_normal(shape), tuple(n for n, _ in p.dims))
        if p.levels:
            tensor = tensor.bind(**dict(p.levels))
        inputs[name] = tensor

    HK = cfg.hk

    def ref(inp):
        xx = inp["x"]
        a = np_layernorm(xx, inp["ln1g"], inp["ln1b"], cfg.eps)
        q = np.einsum("td,dglk->tglk", a, inp["wq"]) / np.sqrt(HK)
        kk = np.einsum("sd,dglk->sglk", a, inp["wk"])
        vv = np.einsum("sd,dglk->sglk", a, inp["wv"])
        sc = np.einsum("tglk,sglk->tsgl", q, kk)
        mask = np.tril(np.ones((T, T), dtype=bool))
        sc = np.where(mask[:, :, None, None], sc, -1e9)
        pr = np_softmax(sc, axis=1)
        ctx = np.einsum("tsgl,sglk->tglk", pr, vv)
        h = xx + np.einsum("tglk,glkd->td", ctx, inp["wo"])
        a2 = np_layernorm(h, inp["ln2g"], inp["ln2b"], cfg.eps)
        mm = np_gelu(np.einsum("td,dgm->tgm", a2, inp["w1"]) + inp["b1"])
        return h + np.einsum("tgm,gmd->td", mm, inp["w2"]) + inp["b2"]

    return ZooModel(model.program, inputs, model.output, ref, ("t", "d"))


def _t(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)
