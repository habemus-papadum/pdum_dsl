"""The tile-tier flagships (320 §10.3) — hand-written tile programs beside
their naive twins.

Each entry is one PER-TILE BODY (the binding law hands a kernel
pre-selected tiles; these regions are that body, so the grid is out of
frame) built op-by-op with the Builder — the honest authoring surface
until a ``@tile`` front-end exists. The naive twin is the erased idiom:
the same denotation with no splits, no stages, no tile-fold. Both run on
``run_region`` today — the tile tier's day-one oracle — and the
differential between them is gated by the DECLARED licenses (gemm's
k-sum re-bracketing; flash's online-softmax lemma), never by an ad-hoc
tolerance. The stencil has no licensed deviation, so its twin must
agree to the bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pdum.dsl.ir import Builder, Region
from pdum.dsl.ops import CORE_OPS
from pdum.dsl.types import f64

from ..dialect import TL_OPS, _minus_dim, check_tier, tensor_type_of_layout
from ..tensor import Tensor
from .zoo_common import np_softmax


@dataclass(frozen=True)
class TileFlagship:
    region: Region  # the tile-tier program (checked at the tile tier)
    naive: Region  # the erased-idiom twin — the denotation
    inputs: dict[str, Tensor]  # positional per param, both regions
    oracle: Callable  # dict[str, np.ndarray] -> np.ndarray

    def numpy_inputs(self) -> dict[str, np.ndarray]:
        return {k: v.to_numpy() for k, v in self.inputs.items()}


def gemm_tile(MI: int = 8, NI: int = 6, K: int = 8, KI: int = 2, seed: int = 13) -> TileFlagship:
    """320 §2's flagship: acc carried through a fold over the k-tiles, both
    operand slices STAGED to "shared", the contraction spelled as
    repeat_like/mul/reduce. The accumulator is registers by ABSENCE — it is
    just the carry."""
    KO = K // KI
    rng = np.random.default_rng(seed)
    A = Tensor.from_numpy(rng.standard_normal((MI, K)), ("mi", "k"))
    B = Tensor.from_numpy(rng.standard_normal((K, NI)), ("k", "ni"))

    ops = {**CORE_OPS, **TL_OPS}
    b = Builder(ops)
    pa = b.param(0, tensor_type_of_layout(A.layout))
    pb = b.param(1, tensor_type_of_layout(B.layout))
    at = b.emit("tl.split", pa, name="k", parts=(("ko", KO), ("ki", KI)))
    bt = b.emit("tl.split", pb, name="k", parts=(("ko", KO), ("ki", KI)))
    acc0 = b.emit("tl.const", value=0.0, dims=(("mi", MI), ("ni", NI)))

    # the step: (acc, a-slice, b-slice) at one ko -> next acc
    sb = Builder(ops)
    p_acc = sb.param(0, acc0.type)
    p_a = sb.param(1, _minus_dim(at.type, "ko"))
    p_b = sb.param(2, _minus_dim(bt.type, "ko"))
    a_s = sb.emit("tl.stage", p_a, level="shared")
    b_s = sb.emit("tl.stage", p_b, level="shared")
    prod = sb.emit("tl.pointwise", sb.emit("tl.repeat_like", a_s, b_s), sb.emit("tl.repeat_like", b_s, a_s), f="mul")
    part = sb.emit("tl.reduce", prod, dims=("ki",), f="sum")
    nxt = sb.emit("tl.pointwise", p_acc, part, f="add")
    step = Region(params=(p_acc, p_a, p_b), body=(sb.emit("core.yield", nxt),))

    fold = b.emit(
        "tl.fold", acc0, at, bt, regions=(step,), dim="ko", state=("acc",), element=("a", "b"), out=("final", 0)
    )
    region = Region(params=(pa, pb), body=(b.emit("core.yield", fold),))

    # the naive twin: contract, unsplit and unstaged
    nb = Builder(ops)
    na = nb.param(0, tensor_type_of_layout(A.layout))
    nnb = nb.param(1, tensor_type_of_layout(B.layout))
    nprod = nb.emit("tl.pointwise", nb.emit("tl.repeat_like", na, nnb), nb.emit("tl.repeat_like", nnb, na), f="mul")
    nred = nb.emit("tl.reduce", nprod, dims=("k",), f="sum")
    naive = Region(params=(na, nnb), body=(nb.emit("core.yield", nred),))

    return TileFlagship(
        region=check_tier(region, "tile"),
        naive=naive,
        inputs={"a": A, "b": B},
        oracle=lambda inp: inp["a"] @ inp["b"],
    )


def stencil_tile(MI: int = 6, NI: int = 8, alpha: float = 0.1, seed: int = 17) -> TileFlagship:
    """The fused stencil chain (K-D) as one tile body: the halo tile stages
    to "shared" once, the 5-point Laplacian is four shifted slices over the
    STAGED copy, and the whole chain (nsum - 4u, u + alpha*lap) fuses as
    pointwise — one load, one tile of work. The twin runs the same chain on
    the unstaged halo. Domains: the halo is [0, MI+2) x [0, NI+2); the
    interior [1, MI+1) x [1, NI+1) is the tile's own patch."""
    rng = np.random.default_rng(seed)
    U = Tensor.from_numpy(rng.standard_normal((MI + 2, NI + 2)), ("x", "y"))

    ops = {**CORE_OPS, **TL_OPS}

    def body(bld, halo):
        # neighbor(dx, dy): u[x+dx, y+dy] over the interior, via shift+slice
        def neighbor(dx, dy):
            sh = bld.emit("tl.shift", halo, deltas=(("x", -dx), ("y", -dy)))
            return bld.emit("tl.slice", sh, ranges=(("x", (1, MI + 1)), ("y", (1, NI + 1))))

        center = neighbor(0, 0)
        ew = bld.emit("tl.pointwise", neighbor(0, 1), neighbor(0, -1), f="add")
        ns = bld.emit("tl.pointwise", neighbor(1, 0), neighbor(-1, 0), f="add")
        nsum = bld.emit("tl.pointwise", ew, ns, f="add")
        four = bld.emit("core.const", type=f64, value=4.0)
        lap = bld.emit("tl.pointwise", nsum, bld.emit("tl.pointwise", four, center, f="mul"), f="sub")
        al = bld.emit("core.const", type=f64, value=alpha)
        return bld.emit("tl.pointwise", center, bld.emit("tl.pointwise", al, lap, f="mul"), f="add")

    b = Builder(ops)
    ph = b.param(0, tensor_type_of_layout(U.layout))
    staged = b.emit("tl.stage", ph, level="shared")
    region = Region(params=(ph,), body=(b.emit("core.yield", body(b, staged)),))

    nb = Builder(ops)
    nh = nb.param(0, tensor_type_of_layout(U.layout))
    naive = Region(params=(nh,), body=(nb.emit("core.yield", body(nb, nh)),))

    def oracle(inp):
        u = inp["u"]
        lap = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1]
        return u[1:-1, 1:-1] + alpha * lap

    return TileFlagship(
        region=check_tier(region, "tile"),
        naive=naive,
        inputs={"u": U},
        oracle=oracle,
    )


def flash_tile(T: int = 6, E: int = 3, OD: int = 2, SI: int = 2, seed: int = 7) -> TileFlagship:
    """Flash attention as a tile body (K-D): the online-softmax state
    (m, den, o) carried through a fold over the s-tiles, K/V slices STAGED
    to "shared" per tile. Two spelling laws worth reading:

    - the causal mask needs ABSOLUTE s coordinates inside the step, and a
      step has no tile coordinate — so the mask is built OUTSIDE as a free
      closed form (iota comparisons over the (t, s) lattice), split along
      s, and rides in as one more element source;
    - q carries no scan dim, and fold binders are exactly states+elements —
      so q rides in as a stride-0 repeat over "so": a broadcast element,
      zero bytes, sliced back to itself each iteration.

    The final normalization needs TWO finals (o and den); the fold returns
    one state, so two fold nodes SHARE the step region (the adjoint
    machinery's re-out precedent, 260) and the division happens after.
    The twin materializes the softmax — same denotation, different program
    (flash's whole point); the deviation is the online-softmax lemma,
    gated by the declared ``flash.online-softmax`` license."""
    SO = T // SI
    rng = np.random.default_rng(seed)
    Q = Tensor.from_numpy(rng.standard_normal((T, E)), ("t", "e"))
    Kv = Tensor.from_numpy(rng.standard_normal((T, E)), ("s", "e"))
    V = Tensor.from_numpy(rng.standard_normal((T, OD)), ("s", "o"))

    ops = {**CORE_OPS, **TL_OPS}

    def scores(bld, q, k):
        # sc[t, si] = sum_e q[t, e] * k[si, e]
        prod = bld.emit("tl.pointwise", bld.emit("tl.repeat_like", q, k), bld.emit("tl.repeat_like", k, q), f="mul")
        return bld.emit("tl.reduce", prod, dims=("e",), f="sum")

    b = Builder(ops)
    pq = b.param(0, tensor_type_of_layout(Q.layout))
    pk = b.param(1, tensor_type_of_layout(Kv.layout))
    pv = b.param(2, tensor_type_of_layout(V.layout))

    # the mask, as a free closed form on the (t, s) lattice: s <= t
    ts = b.emit("tl.repeat", b.emit("tl.select", pq, coords=(("e", 0),)), name="s", extent=T)
    mask = b.emit("tl.pointwise", b.emit("tl.iota", ts, name="s"), b.emit("tl.iota", ts, name="t"), f="le")

    # element sources over the scan dim: K/V/mask split at s, q broadcast
    kt = b.emit("tl.split", pk, name="s", parts=(("so", SO), ("si", SI)))
    vt = b.emit("tl.split", pv, name="s", parts=(("so", SO), ("si", SI)))
    mt = b.emit("tl.split", mask, name="s", parts=(("so", SO), ("si", SI)))
    qr = b.emit("tl.repeat", pq, name="so", extent=SO)

    m0 = b.emit("tl.const", value=-1e30, dims=(("t", T),))  # flashsm's identity state
    den0 = b.emit("tl.const", value=0.0, dims=(("t", T),))
    o0 = b.emit("tl.const", value=0.0, dims=(("t", T), ("o", OD)))

    # the step: (m, den, o | q, k-tile, v-tile, mask-tile) -> next state
    sb = Builder(ops)
    p_m = sb.param(0, m0.type)
    p_den = sb.param(1, den0.type)
    p_o = sb.param(2, o0.type)
    p_q = sb.param(3, _minus_dim(qr.type, "so"))
    p_k = sb.param(4, _minus_dim(kt.type, "so"))
    p_v = sb.param(5, _minus_dim(vt.type, "so"))
    p_mk = sb.param(6, _minus_dim(mt.type, "so"))
    ks = sb.emit("tl.stage", p_k, level="shared")
    vs = sb.emit("tl.stage", p_v, level="shared")
    neg = sb.emit("core.const", type=f64, value=-1e9)
    sm = sb.emit("tl.pointwise", p_mk, scores(sb, p_q, ks), neg, f="where")
    m_new = sb.emit("tl.pointwise", p_m, sb.emit("tl.reduce", sm, dims=("si",), f="max"), f="maximum")
    alpha = sb.emit("tl.pointwise", sb.emit("tl.pointwise", p_m, m_new, f="sub"), f="exp")
    p_w = sb.emit("tl.pointwise", sb.emit("tl.pointwise", sm, sb.emit("tl.repeat_like", m_new, sm), f="sub"), f="exp")
    den_new = sb.emit(
        "tl.pointwise",
        sb.emit("tl.pointwise", p_den, alpha, f="mul"),
        sb.emit("tl.reduce", p_w, dims=("si",), f="sum"),
        f="add",
    )
    pv_prod = sb.emit(
        "tl.pointwise", sb.emit("tl.repeat_like", p_w, vs), sb.emit("tl.repeat_like", vs, p_w), f="mul"
    )
    o_new = sb.emit(
        "tl.pointwise",
        sb.emit("tl.pointwise", p_o, sb.emit("tl.repeat_like", alpha, p_o), f="mul"),
        sb.emit("tl.reduce", pv_prod, dims=("si",), f="sum"),
        f="add",
    )
    step = Region(
        params=(p_m, p_den, p_o, p_q, p_k, p_v, p_mk),
        body=(sb.emit("core.yield", sb.emit("core.tuple", m_new, den_new, o_new)),),
    )

    # two finals from one loop: the fold nodes SHARE the step region
    fold_args = (m0, den0, o0, qr, kt, vt, mt)
    fold_kw = dict(dim="so", state=("m", "den", "o"), element=("q", "k", "v", "mask"))
    f_o = b.emit("tl.fold", *fold_args, regions=(step,), out=("final", 2), **fold_kw)
    f_den = b.emit("tl.fold", *fold_args, regions=(step,), out=("final", 1), **fold_kw)
    out = b.emit("tl.pointwise", f_o, b.emit("tl.repeat_like", f_den, f_o), f="div")
    region = Region(params=(pq, pk, pv), body=(b.emit("core.yield", out),))

    # the twin: materialized causal softmax — same denotation, different program
    nb = Builder(ops)
    nq = nb.param(0, tensor_type_of_layout(Q.layout))
    nk = nb.param(1, tensor_type_of_layout(Kv.layout))
    nv = nb.param(2, tensor_type_of_layout(V.layout))
    sc = scores(nb, nq, nk)  # (t, s)
    nmask = nb.emit("tl.pointwise", nb.emit("tl.iota", sc, name="s"), nb.emit("tl.iota", sc, name="t"), f="le")
    nsm = nb.emit("tl.pointwise", nmask, sc, nb.emit("core.const", type=f64, value=-1e9), f="where")
    nmax = nb.emit("tl.reduce", nsm, dims=("s",), f="max")
    npw = nb.emit("tl.pointwise", nb.emit("tl.pointwise", nsm, nb.emit("tl.repeat_like", nmax, nsm), f="sub"), f="exp")
    nden = nb.emit("tl.reduce", npw, dims=("s",), f="sum")
    npr = nb.emit("tl.pointwise", npw, nb.emit("tl.repeat_like", nden, npw), f="div")
    nprod = nb.emit("tl.pointwise", nb.emit("tl.repeat_like", npr, nv), nb.emit("tl.repeat_like", nv, npr), f="mul")
    nout = nb.emit("tl.reduce", nprod, dims=("s",), f="sum")
    naive = Region(params=(nq, nk, nv), body=(nb.emit("core.yield", nout),))

    def oracle(inp):
        sc = inp["q"] @ inp["k"].T
        tril = np.tril(np.ones((T, T), dtype=bool))
        return np_softmax(np.where(tril, sc, -1e9), axis=1) @ inp["v"]

    return TileFlagship(
        region=check_tier(region, "tile"),
        naive=naive,
        inputs={"q": Q, "k": Kv, "v": V},
        oracle=oracle,
    )
