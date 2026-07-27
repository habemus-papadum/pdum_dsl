"""Capacity-factor MoE (200 §1.9) — routing stays in the subset.

The whole router is STATIC SHAPES: `argtopk` over router logits
(gradient-free), gates by softmax over the CHOSEN logits (the aligned
take — a per-token gather, the same spelling as the embedding gather),
position assignment by exclusive prefix sums over one-hot masks (scan —
the flat (t, c) order decomposed as row totals + within-row prefixes, so
no lattice ever merges), `scatter_add` into the fixed (expert, capacity)
buffer through the DECLARED linearization (slot = choice·CAP + position;
`over=("t", "c")` consumes both routing dims), dense expert compute,
`take` back at the same linearized indices, gate-weighted combine.
Overflow beyond capacity is masked out — dropped pairs point at slot 0
with a zeroed value and a zeroed gate: the standard capacity-factor
semantics, stated honestly. Every index is integer-carrier end to end
(const_like's int face — a Python int literal IS the carrier
declaration); slots are unique by construction, so the scatter's
duplicate-sum law is exercised only by the mask zeros.

Fully dynamic dispatch remains a recorded boundary (a dynamic-shapes
problem, not an indexing problem)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..compute import const_like, contract, iota, pointwise, red, reduce, repeat_like, scan
from ..indexing import argtopk, scatter_add, take
from ..lifting import lift_step
from ..markers import eq, lt, where
from ..tensor import Tensor
from .zoo_common import ZooModel, gelu, np_gelu, np_softmax, softmax


@dataclass(frozen=True)
class MoEConfig:
    t: int = 6  # tokens
    d: int = 4  # model width
    e: int = 3  # experts
    k: int = 2  # choices per token (top-k)
    cap: int = 4  # per-expert capacity
    m: int = 5  # expert hidden width


def make_moe(cfg: MoEConfig):
    E, K, CAP = cfg.e, cfg.k, cfg.cap

    def moe(x, wr, w1, w2):
        logits = contract(x, wr, axis="d")  # (t, e)
        choice = argtopk(logits, dim="e", k=K, k_name="c")  # (t, c) int
        gates = softmax(take(logits, choice, dim="e"), k="c")  # (t, c): the aligned gather

        # one-hot occupancy and the flat-order exclusive prefix (t-major, c-minor)
        rep = repeat_like(choice, logits)  # (t, c, e) int
        onehot = pointwise(where, pointwise(eq, rep, iota(rep, "e")), const_like(rep, 1), const_like(rep, 0))
        row = reduce(red.sum, onehot, "c")  # (t, e): choices per token per expert
        excl_t = scan(red.sum, row, "t") - row  # tokens before t
        cum_c = scan(red.sum, onehot, "c")  # within-token prefix
        excl = repeat_like(excl_t, onehot) + cum_c - onehot  # earlier (t', c') pairs at e
        pos = reduce(red.sum, onehot * excl, "e")  # (t, c) int: my slot at my expert

        keep = pointwise(lt, pos, const_like(pos, CAP))  # capacity mask
        dest = pointwise(
            where, keep, choice * const_like(choice, CAP) + pos, const_like(pos, 0)
        )  # (t, c) int: the declared linearization (dropped pairs park at 0, zero-valued)

        keepf = pointwise(where, keep, const_like(gates, 1.0), const_like(gates, 0.0))
        vals = repeat_like(x, dest) * repeat_like(keepf, repeat_like(x, dest))  # (t, d, c)
        buf = scatter_add(vals, dest, dim="slot", extent=E * CAP, over=("t", "c"))  # (slot, d)

        bufec = buf.split("slot", ee=E, cap=CAP)  # (ee, cap, d): slot = ee*CAP + cap
        h = pointwise(gelu, contract(bufec, w1, axis="d"))  # (ee, cap, m)
        y2 = contract(h, w2, axis="m")  # (ee, cap, d)

        back = take(y2.merge(("ee", "cap"), "slot2"), dest, dim="slot2")  # (t, c, d)
        weight = gates * keepf  # (t, c)
        return reduce(red.sum, back * repeat_like(weight, back), "c")  # (t, d)

    return moe


def _np_moe(cfg: MoEConfig, inp):
    E, K, CAP = cfg.e, cfg.k, cfg.cap
    x, wr, w1, w2 = inp["x"], inp["wr"], inp["w1"], inp["w2"]
    logits = x @ wr
    choice = np.argsort(-logits, axis=1, kind="stable")[:, :K]
    gates = np_softmax(np.take_along_axis(logits, choice, axis=1), axis=1)
    counts = np.zeros(E, dtype=int)
    buf = np.zeros((E, CAP, cfg.d))
    slots = np.full((cfg.t, K), -1)
    for t in range(cfg.t):
        for c in range(K):
            ex = choice[t, c]
            if counts[ex] < CAP:
                buf[ex, counts[ex]] = x[t]
                slots[t, c] = counts[ex]
            counts[ex] += 1
    y2 = np_gelu(np.einsum("ecd,edm->ecm", buf, w1))
    y2 = np.einsum("ecm,emd->ecd", y2, w2)
    out = np.zeros((cfg.t, cfg.d))
    for t in range(cfg.t):
        for c in range(K):
            if slots[t, c] >= 0:
                out[t] += gates[t, c] * y2[choice[t, c], slots[t, c]]
    return out


def moe(cfg: MoEConfig = MoEConfig(), seed: int = 11) -> ZooModel:
    rng = np.random.default_rng(seed)

    def _t(arr, names):
        return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)

    inputs = {
        "x": _t(rng.standard_normal((cfg.t, cfg.d)), ("t", "d")),
        "wr": _t(rng.standard_normal((cfg.d, cfg.e)), ("d", "e")),
        "w1": _t(0.4 * rng.standard_normal((cfg.e, cfg.d, cfg.m)), ("ee", "d", "m")),
        "w2": _t(0.4 * rng.standard_normal((cfg.e, cfg.m, cfg.d)), ("ee", "m", "d")),
    }
    ls = lift_step(make_moe(cfg), **{k: v.layout for k, v in inputs.items()})
    return ZooModel(ls.program, inputs, ls.outputs[0], lambda inp: _np_moe(cfg, inp), ("t", "d"))
