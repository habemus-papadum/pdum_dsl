"""The unrolled trainer (200 §6.7) — P9's end-to-end gate.

Unroll the sampling step K times inside training: chunk 1 is the normal
teacher-forced forward (loss over all positions, xent by the ALIGNED
take); between chunks, sample the next token by top-k with a TRAINABLE
temperature (§1.8: Gumbel noise + temperature softmax + argtopk
restriction — the restriction mask is BUILT BY scatter_add — +
straight-through via stop_gradient); later chunks predict one step
further, reusing chunk 1's KV as ordinary dataflow (concat via pad+add,
the spec's spelling); the total loss sums all chunks.

The whole unroll is ONE Program, so weight sharing across chunks is
automatic — the same input vars referenced everywhere, gradients summed
by fan-out (the tying rule at macro scale). Randomness enters as NAMED,
KEYED INPUT FIELDS (the Gumbel tensors are data; the in-program
`random` op stays the dropout door) — so checkpointed recompute replays
the draws exactly by construction, and the checkpoint gate pins
store-all ≡ checkpointed gradients. The chunks are SPELLED (K−1 helper
calls): step bodies keep the straight-line refusal (S.3 — the `for`
face is the kernel tier's); a unit-tier static unroll is a recorded
future for large K.

Deliberately one single-head attention block with a tied head — the
gate exercises indexing + sampling + unroll + checkpoint-replay
together; the multi-layer naming law is gpt2's entry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..compute import const_like, contract, pointwise, red, reduce, repeat_like
from ..indexing import argtopk, scatter_add, take
from ..lifting import lift_step
from ..markers import exp, gt, log, stop_gradient, where
from ..tensor import Tensor
from .zoo_common import ZooModel, causal_softmax, layernorm, np_layernorm, np_softmax, softmax


@dataclass(frozen=True)
class TrainerConfig:
    t: int = 4  # the teacher-forced window
    d: int = 6  # model width
    hk: int = 4  # attention width (single head)
    v: int = 5  # vocab
    k_chunks: int = 3  # K: chunk 1 + (K-1) sampled decode chunks (spelled)
    topk: int = 3  # sampling restriction
    eps: float = 1e-5


def make_trainer(cfg: TrainerConfig):
    T, V, TOPK, EPS = cfg.t, cfg.v, cfg.topk, cfg.eps
    scale = 1.0 / math.sqrt(cfg.hk)

    # helpers live at maker scope (a lowered body is ONE def); they inline
    # as captured callables, weights passed explicitly

    def lse_of(logits, dim):
        mx = reduce(red.max, logits, dim)
        e = pointwise(exp, logits - repeat_like(mx, logits))
        return pointwise(log, reduce(red.sum, e, dim)) + mx

    def head(h, wte, lnfg, lnfb):
        return contract(layernorm(h, lnfg, lnfb, feat="d", eps=EPS), wte, axis="d")

    def chunk(j, loss, lastlog, ks, vs, gum, tgt, wte, wpe, lng, lnb, lnfg, lnfb, wq, wk, wv, wo, tau):
        """One sampled decode step: §1.8's estimator over §1.9's family."""
        pert = (lastlog + gum) / repeat_like(tau, lastlog)  # temperature enters
        top = argtopk(pert, dim="v", k=TOPK, k_name="r")  # the restriction set
        mask = scatter_add(const_like(top, 1.0), top, dim="v", extent=V)  # built BY scatter
        keep = pointwise(gt, mask, const_like(mask, 0.0))
        masked = pointwise(where, keep, pert, const_like(pert, -1e9))
        soft = softmax(masked, k="v")
        nxt = argtopk(masked, dim="v", k=1, k_name="c")  # the hard sample
        ehard = take(wte, nxt, dim="v").select(c=0)  # (d,)
        esoft = contract(soft, wte, axis="v")  # (d,)
        est = ehard + (esoft - pointwise(stop_gradient, esoft))  # straight-through
        e = est + wpe.select(p=T + j - 1)  # the new position
        a = layernorm(e, lng, lnb, feat="d", eps=EPS)
        qn = contract(a, wq, axis="d") * scale
        kn = contract(a, wk, axis="d")
        vn = contract(a, wv, axis="d")
        s_hi = T + j  # keys now cover [0, T+j)
        kf = ks.pad(0.0, s=(0, s_hi)) + kn.repeat("s", (s_hi - 1, s_hi)).pad(0.0, s=(0, s_hi))
        vf = vs.pad(0.0, s=(0, s_hi)) + vn.repeat("s", (s_hi - 1, s_hi)).pad(0.0, s=(0, s_hi))
        pr = softmax(contract(qn, kf, axis="hk"), k="s")  # attend over ALL keys
        hn = e + contract(contract(pr, vf, axis="s"), wo, axis="hk")  # (d,)
        ll = head(hn, wte, lnfg, lnfb)  # (v,)
        lj = lse_of(ll, "v") - take(ll, tgt.select(u=T + j - 1), dim="v")  # rank-0 xent
        return loss + lj, ll, kf, vf

    def trainer(ids, tgt, wte, wpe, lng, lnb, lnfg, lnfb, wq, wk, wv, wo, tau, gum1, gum2):
        # ---- chunk 1: the teacher-forced forward ---------------------------
        h = take(wte, ids, dim="v") + wpe.slice(p=(0, T)).rename(p="t")
        a = layernorm(h, lng, lnb, feat="d", eps=EPS)
        q = contract(a, wq, axis="d") * scale
        ks = contract(a.rename(t="s"), wk, axis="d")
        vs = contract(a.rename(t="s"), wv, axis="d")
        pr = causal_softmax(contract(q, ks, axis="hk"))
        h = h + contract(contract(pr, vs, axis="s"), wo, axis="hk")
        logits = head(h, wte, lnfg, lnfb)  # (t, v) — the tied head
        t1 = tgt.slice(u=(0, T)).rename(u="t")  # truth for positions 1..T
        loss = reduce(red.sum, lse_of(logits, "v") - take(logits, t1, dim="v"), "t")
        lastlog = logits.select(t=T - 1)

        # ---- chunks 2..K: sample, decode against the reused KV -------------
        loss, lastlog, ks, vs = chunk(
            1, loss, lastlog, ks, vs, gum1, tgt, wte, wpe, lng, lnb, lnfg, lnfb, wq, wk, wv, wo, tau
        )
        loss, lastlog, ks, vs = chunk(
            2, loss, lastlog, ks, vs, gum2, tgt, wte, wpe, lng, lnb, lnfg, lnfb, wq, wk, wv, wo, tau
        )
        return loss

    return trainer


# ---- the numpy denotation ---------------------------------------------------


def _np_trainer(cfg: TrainerConfig, inp):
    T, V, TOPK, EPS = cfg.t, cfg.v, cfg.topk, cfg.eps
    scale = 1.0 / math.sqrt(cfg.hk)
    wte, wpe = inp["wte"], inp["wpe"]

    def ln(x, g, b):
        return np_layernorm(x, g, b, EPS)

    def lse(logits):
        m = logits.max(axis=-1)
        return np.log(np.exp(logits - m[..., None]).sum(axis=-1)) + m

    def head(h):
        return ln(h, inp["lnfg"], inp["lnfb"]) @ wte.T

    h = wte[inp["ids"]] + wpe[:T]
    a = ln(h, inp["lng"], inp["lnb"])
    q = a @ inp["wq"] * scale
    ks, vs = a @ inp["wk"], a @ inp["wv"]
    sc = np.where(np.tril(np.ones((T, T), dtype=bool)), q @ ks.T, -1e9)
    h = h + np_softmax(sc, axis=1) @ vs @ inp["wo"]
    logits = head(h)
    t1 = inp["tgt"][:T]
    loss = float((lse(logits) - logits[np.arange(T), t1]).sum())
    lastlog = logits[T - 1]

    for j, gum in ((1, inp["gum1"]), (2, inp["gum2"])):
        pert = (lastlog + gum) / inp["tau"]
        top = np.argsort(-pert, kind="stable")[:TOPK]
        masked = np.where(np.isin(np.arange(V), top), pert, -1e9)
        nxt = int(np.argsort(-masked, kind="stable")[0])
        e = wte[nxt] + wpe[T + j - 1]  # forward IS the hard path
        a = ln(e, inp["lng"], inp["lnb"])
        qn, kn, vn = a @ inp["wq"] * scale, a @ inp["wk"], a @ inp["wv"]
        ks, vs = np.concatenate([ks, kn[None]]), np.concatenate([vs, vn[None]])
        prd = np_softmax(qn @ ks.T, axis=-1)
        hn = e + prd @ vs @ inp["wo"]
        ll = head(hn)
        loss += float(lse(ll) - ll[inp["tgt"][T + j - 1]])
        lastlog = ll
    return np.asarray(loss)


def unrolled_trainer(cfg: TrainerConfig = TrainerConfig(), seed: int = 5) -> ZooModel:
    rng = np.random.default_rng(seed)
    P = cfg.t + cfg.k_chunks - 1  # positions 0 .. T+K-2

    def f(arr, names):
        return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)

    inputs = {
        "ids": Tensor.from_numpy(rng.integers(0, cfg.v, cfg.t), ("t",)),
        "tgt": Tensor.from_numpy(rng.integers(0, cfg.v, P), ("u",)),  # truth for 1..T+K-1
        "wte": f(0.4 * rng.standard_normal((cfg.v, cfg.d)), ("v", "d")),
        "wpe": f(0.1 * rng.standard_normal((P, cfg.d)), ("p", "d")),
        "lng": f(1.0 + 0.1 * rng.standard_normal(cfg.d), ("d",)),
        "lnb": f(0.1 * rng.standard_normal(cfg.d), ("d",)),
        "lnfg": f(1.0 + 0.1 * rng.standard_normal(cfg.d), ("d",)),
        "lnfb": f(0.1 * rng.standard_normal(cfg.d), ("d",)),
        "wq": f(0.4 * rng.standard_normal((cfg.d, cfg.hk)), ("d", "hk")),
        "wk": f(0.4 * rng.standard_normal((cfg.d, cfg.hk)), ("d", "hk")),
        "wv": f(0.4 * rng.standard_normal((cfg.d, cfg.hk)), ("d", "hk")),
        "wo": f(0.4 * rng.standard_normal((cfg.hk, cfg.d)), ("hk", "d")),
        "tau": f(0.8, ()),
        # named, keyed noise fields: the Gumbel draws are DATA (reproducible
        # by the key; recompute replays them exactly because inputs are inputs)
        "gum1": f(-np.log(-np.log(np.random.default_rng(seed + 101).uniform(size=cfg.v))), ("v",)),
        "gum2": f(-np.log(-np.log(np.random.default_rng(seed + 102).uniform(size=cfg.v))), ("v",)),
    }
    ls = lift_step(make_trainer(cfg), **{k: v.layout for k, v in inputs.items()})
    return ZooModel(ls.program, inputs, ls.outputs[0], lambda inp: _np_trainer(cfg, inp), ())
