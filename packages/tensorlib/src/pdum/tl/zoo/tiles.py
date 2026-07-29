"""The tile-tier flagships (320 §10.3) — hand-written tile programs beside
their naive twins.

Each entry is one PER-TILE BODY (the binding law hands a kernel
pre-selected tiles; these regions are that body, so the grid is out of
frame) built op-by-op with the Builder — the honest authoring surface
until a ``@tile`` front-end exists. The naive twin is the erased idiom:
the same denotation with no splits, no stages, no tile-fold. Both run on
``run_region`` today — the tile tier's day-one oracle — and the
differential between them is gated by the DECLARED licenses
(licenses.GEMM_F16_TILES row one: the k-sum re-bracketing), never by an
ad-hoc tolerance.
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
