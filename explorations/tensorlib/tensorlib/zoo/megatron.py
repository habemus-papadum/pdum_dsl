"""Megatron-style tensor parallelism as PLACEMENT metadata (L3-lite).

One transformer block, hand-lowered the global-view way (PLACEMENT.md):
attention heads carry a mesh dim `g` (column-parallel QKV, heads sharded),
the MLP width carries the same `g` (column-parallel up, row-parallel down)
— and `g` is BOUND to the machine level. No collective ops appear anywhere:
the two per-block all-reduces Megatron's paper prescribes are simply the
two reduce-over-`g` contractions (attention output projection, MLP down
projection), which the traffic pass reads off the algebra.

`level=None` builds the ERASURE — the identical program minus bindings —
for the denotation-preservation check: placed and erased runs must agree
bit-for-bit, because placement is cost-bearing metadata, never meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..build import Build
from .zoo_common import ZooModel, layernorm, np_gelu, np_layernorm, np_softmax, t_in


@dataclass(frozen=True)
class MegatronConfig:
    t: int = 4  # sequence
    d: int = 6  # model width
    g: int = 2  # mesh extent (tensor-parallel ways)
    hl: int = 1  # heads per device (nh = g * hl)
    hk: int = 3  # head width
    ml: int = 4  # mlp width per device (m = g * ml)
    eps: float = 1e-5


def megatron_block(cfg: MegatronConfig = MegatronConfig(), level: str | None = "gpu", seed: int = 29) -> ZooModel:
    rng = np.random.default_rng(seed)
    T, D, G, HL, HK, ML = cfg.t, cfg.d, cfg.g, cfg.hl, cfg.hk, cfg.ml
    b = Build()
    inputs: dict = {}

    def w(name, *shape, names, bound=False):
        v = t_in(inputs, name, 0.4 * rng.standard_normal(shape), names)
        if bound and level is not None:
            inputs[name] = inputs[name].bind(g=level)
        return v

    x = t_in(inputs, "x", rng.standard_normal((T, D)), ("t", "d"))
    b.input(x)
    for nm, shape, names, bound in (
        ("ln1g", (D,), ("d",), False),
        ("ln1b", (D,), ("d",), False),
        ("wq", (D, G, HL, HK), ("d", "g", "hl", "hk"), True),
        ("wk", (D, G, HL, HK), ("d", "g", "hl", "hk"), True),
        ("wv", (D, G, HL, HK), ("d", "g", "hl", "hk"), True),
        ("wo", (G, HL, HK, D), ("g", "hl", "hk", "d"), True),
        ("ln2g", (D,), ("d",), False),
        ("ln2b", (D,), ("d",), False),
        ("w1", (D, G, ML), ("d", "g", "ml"), True),
        ("b1", (G, ML), ("g", "ml"), True),
        ("w2", (G, ML, D), ("g", "ml", "d"), True),
        ("b2", (D,), ("d",), False),
    ):
        b.input(w(nm, *shape, names=names, bound=bound))

    def bindg(v: str) -> str:
        # broadcasts that INTRODUCE the mesh dim must declare its placement
        return b.emit("bind", (v,), levels={"g": level}) if level is not None else v

    def bcast_g(v: str, reps) -> str:
        out = b.bcast(v, reps)
        if any(n == "g" for n, _ in reps):
            out = bindg(out)
        return out

    def contract(xv, yv, x_missing, y_missing, over, hint):
        xb = bcast_g(xv, x_missing) if x_missing else xv
        yb = bcast_g(yv, y_missing) if y_missing else yv
        return b.red("sum", b.pw("mul", xb, yb), over, hint=hint)

    def const_g(value, dims, hint):
        return bindg(b.const(value, dims, hint=hint))

    td = [("t", (0, T)), ("d", (0, D))]
    heads = [("g", (0, G)), ("hl", (0, HL)), ("hk", (0, HK))]
    tsgh = [("t", (0, T)), ("s", (0, T)), ("g", (0, G)), ("hl", (0, HL))]
    a = layernorm(b, x, "d", (0, D), td, "ln1g", "ln1b", cfg.eps)
    a_s = b.emit("rename", (a,), hint="as", mapping={"t": "s"})
    q = contract(a, "wq", heads, [("t", (0, T))], ("d",), hint="q")
    kk = contract(a_s, "wk", heads, [("s", (0, T))], ("d",), hint="k")
    vv = contract(a_s, "wv", heads, [("s", (0, T))], ("d",), hint="v")
    thgk = [("t", (0, T)), ("g", (0, G)), ("hl", (0, HL)), ("hk", (0, HK))]
    qs = b.pw("mul", q, const_g(1.0 / np.sqrt(HK), thgk, hint="scale"), hint="qs")
    sc = contract(qs, kk, [("s", (0, T))], [("t", (0, T))], ("hk",), hint="sc")
    # causal mask + softmax over s, with placement-declared constants
    it = b.emit("iota", (sc,), hint="it", name="t")
    isv = b.emit("iota", (sc,), hint="is", name="s")
    m = b.pw("le", isv, it, hint="mask")
    sm = b.pw("where", m, sc, const_g(-1e9, tsgh, hint="ninf"), hint="scm")
    mx = b.red("max", sm, ("s",), hint="mx")
    e = b.pw("exp", b.pw("sub", sm, b.bcast(mx, [("s", (0, T))])), hint="e")
    z = b.red("sum", e, ("s",), hint="z")
    pr = b.pw("div", e, b.bcast(z, [("s", (0, T))]), hint="p")
    ctx = contract(pr, vv, [("hk", (0, HK))], [("t", (0, T))], ("s",), hint="ctx")
    o = contract(ctx, "wo", [("d", (0, D))], [("t", (0, T))], ("g", "hl", "hk"), hint="o")  # all-reduce #1
    h = b.pw("add", x, o, hint="hres")
    a2 = layernorm(b, h, "d", (0, D), td, "ln2g", "ln2b", cfg.eps)
    a1 = contract(a2, "w1", [("g", (0, G)), ("ml", (0, ML))], [("t", (0, T))], ("d",), hint="a1")
    a1b = b.pw("add", a1, b.bcast("b1", [("t", (0, T))]), hint="a1b")
    gg = b.pw("zoo.gelu", a1b, hint="gelu")
    m2 = contract(gg, "w2", [("d", (0, D))], [("t", (0, T))], ("g", "ml"), hint="m2")  # all-reduce #2
    m2b = b.pw("add", m2, b.bcast("b2", [("t", (0, T))]), hint="m2b")
    out = b.pw("add", h, m2b, hint="out")

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

    return ZooModel(b.program(), inputs, out, ref, ("t", "d"))
