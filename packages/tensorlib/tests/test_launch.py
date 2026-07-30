"""Grid & launch (340 §7.2–3): the feasibility scorer and the proposal
ladder — the T=512 shared-memory crash becomes a computed decision."""

from pdum.tl.fusion import plan_region
from pdum.tl.launch import TileLevel, TileMachine, feasible, propose
from pdum.tl.zoo.tiles import flash_tile

# the 4090's opt-in shared-memory limit, as triton's OutOfResources
# reported it at the measured crash — a machine-table row, not a probe
_SHARED = TileMachine((TileLevel("shared", 128, 101376),))


def test_the_measured_crash_is_now_a_computed_refusal():
    """Flash at T=512, D=64 required 231KB against 101KB on silicon.
    The scorer decides that without compiling anything."""
    region = flash_tile(T=512, E=64, OD=64, SI=32).region
    ok_full, fp_full = feasible(region, (("t", 512),), _SHARED)
    ok_32, fp_32 = feasible(region, (("t", 32),), _SHARED)
    assert not ok_full and fp_full > _SHARED.tightest()
    assert ok_32 and fp_32 <= _SHARED.tightest()


def test_the_ladder_is_smallest_first_and_all_feasible():
    """Smallest tile above the floor = the analytic default (maximum
    oversubscription); every rung fits; the infeasible top rungs are
    simply absent."""
    region = flash_tile(T=512, E=64, OD=64, SI=32).region
    cands = propose(region, _SHARED, floor=1024)
    tiles = [dict(c)["t"] for c in cands]
    assert tiles == sorted(tiles) and tiles[0] == 16  # 16*64 elems = the floor
    assert all(feasible(region, c, _SHARED)[0] for c in cands)
    assert 512 not in tiles and 256 not in tiles  # the computed cliff


def test_plan_region_attaches_the_analytic_default():
    f = flash_tile(T=128, E=32, OD=32, SI=2)
    bare = plan_region(f.naive)
    (g0,) = bare.groups
    assert g0.launch == ()
    (g,) = plan_region(f.naive, machine=_SHARED).groups
    assert g.template == "flash" and g.launch and g.launch[0][0] == "t"


def test_the_emptiness_engine_is_three_valued_and_sound():
    """340 §4b: interval evaluation of a closed forest — T/F where the
    boxes decide, M anywhere the algebra is not spelled (sound: unproven
    means unpruned, never wrong)."""
    from pdum.tl.fusion import _match_flash
    from pdum.tl.launch import tri_eval

    mask = _match_flash(flash_tile().naive)["mask"]  # le(iota_s, iota_t)
    assert tri_eval(mask, {"t": (0, 0), "s": (1, 5)}) == ("bool", "F")
    assert tri_eval(mask, {"t": (2, 3), "s": (0, 1)}) == ("bool", "T")
    assert tri_eval(mask, {"t": (2, 3), "s": (2, 4)}) == ("bool", "M")


def test_the_bound_fitter_is_exact_or_absent():
    from pdum.tl.launch import fit_affine

    assert fit_affine([5, 5, 5], 0, 8)[1] == 0  # constant
    assert fit_affine([1, 2, 3, 4], 0, 8) == (1, 1, 1)  # affine
    assert fit_affine([0, 0, 1, 2], 0, 8) == (-1, 1, 1)  # clamped window edge
    a, d, c = fit_affine([1, 1, 2, 2, 3, 3], 0, 8)  # ceil: affine over a denominator
    assert [min(8, max(0, (a + d * g) // c)) for g in range(6)] == [1, 1, 2, 2, 3, 3]
    assert fit_affine([0, 3, 1, 2], 0, 8) is None  # no closed form -> no prune


def test_prune_flash_derives_the_causal_bounds():
    """The diagonal bound a hand author writes, computed from the mask:
    hi = pid+1 at BT=SI, and the ceil form when BT < SI."""
    from pdum.tl.fusion import _prune_flash

    naive = flash_tile(T=128, E=32, OD=32, SI=2).naive
    assert _prune_flash(naive, tile=32, si=32) == ((0, 0, 1), (1, 1, 1))
    lo, hi = _prune_flash(naive, tile=16, si=32)
    assert lo == (0, 0, 1) and [min(4, max(0, (hi[0] + hi[1] * g) // hi[2])) for g in range(8)] == [
        1, 1, 2, 2, 3, 3, 4, 4]


def test_a_fully_masked_row_refuses_pruning():
    """The m-chain law: strict-causal masks row 0 entirely — its softmax
    is uniform-by-convention, so pruning would change it. Refuse."""
    import numpy as np

    from pdum.dsl.ir import Builder, Region
    from pdum.dsl.ops import CORE_OPS
    from pdum.dsl.types import f64
    from pdum.tl.dialect import TL_OPS, tensor_type_of_layout
    from pdum.tl.fusion import _prune_flash
    from pdum.tl.tensor import Tensor

    ops = {**CORE_OPS, **TL_OPS}
    rng = np.random.default_rng(3)
    b = Builder(ops)
    q = b.param(0, tensor_type_of_layout(Tensor.from_numpy(rng.standard_normal((64, 32)), ("t", "e")).layout))
    k = b.param(1, tensor_type_of_layout(Tensor.from_numpy(rng.standard_normal((64, 32)), ("s", "e")).layout))
    v = b.param(2, tensor_type_of_layout(Tensor.from_numpy(rng.standard_normal((64, 32)), ("s", "o")).layout))
    prod = b.emit("tl.pointwise", b.emit("tl.repeat_like", q, k), b.emit("tl.repeat_like", k, q), f="mul")
    sc = b.emit("tl.reduce", prod, dims=("e",), f="sum")
    mask = b.emit("tl.pointwise", b.emit("tl.iota", sc, name="s"), b.emit("tl.iota", sc, name="t"), f="lt")
    sm = b.emit("tl.pointwise", mask, sc, b.emit("core.const", type=f64, value=-1e9), f="where")
    mx = b.emit("tl.reduce", sm, dims=("s",), f="max")
    e = b.emit("tl.pointwise", b.emit("tl.pointwise", sm, b.emit("tl.repeat_like", mx, sm), f="sub"), f="exp")
    den = b.emit("tl.reduce", e, dims=("s",), f="sum")
    pr = b.emit("tl.pointwise", e, b.emit("tl.repeat_like", den, e), f="div")
    pv = b.emit("tl.pointwise", b.emit("tl.repeat_like", pr, v), b.emit("tl.repeat_like", v, pr), f="mul")
    out = b.emit("tl.reduce", pv, dims=("s",), f="sum")
    region = Region(params=(q, k, v), body=(b.emit("core.yield", out),))
    assert _prune_flash(region, tile=16, si=16) is None


def test_plan_region_attaches_the_prune():
    f = flash_tile(T=128, E=32, OD=32, SI=2)
    (g,) = plan_region(f.naive, machine=_SHARED).groups
    assert g.prune == (("so", (0, 0, 1), (1, 1, 1)),)
