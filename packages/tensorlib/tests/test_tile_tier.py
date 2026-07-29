"""The tile tier (320): the gate, the stage op, the erasure oracle, and the
flagships against their naive twins under the declared licenses."""

import numpy as np
import pytest

from pdum.dsl.ir import Builder, Region
from pdum.dsl.ops import CORE_OPS
from pdum.tl.chart import chart
from pdum.tl.dialect import TL_OPS, check_tier, run_region, tensor_type_of_layout
from pdum.tl.licenses import FLASH_ONLINE_SOFTMAX, GEMM_F16_TILES
from pdum.tl.tensor import Tensor
from pdum.tl.transforms import erase_stages
from pdum.tl.zoo.tiles import flash_tile, gemm_tile, stencil_tile

OPS = {**CORE_OPS, **TL_OPS}


def _t(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


# --- the gate -----------------------------------------------------------------


def test_tile_tier_admits_the_flagship_vocabulary():
    # construction runs check_tier(region, "tile") — building IS the assertion
    gemm_tile()
    stencil_tile()
    flash_tile()


def test_tile_tier_refuses_data_dependent_addressing():
    """The K-G ops are host citizens at the tile tier (320 §3) even though
    the tensor tier admits them."""
    b = Builder(OPS)
    x = b.param(0, tensor_type_of_layout(_t(np.zeros((4, 3)), ("t", "e")).layout))
    top = b.emit("tl.argtopk", x, dim="e", k=2, k_name="c")
    region = Region(params=(x,), body=(b.emit("core.yield", top),))
    check_tier(region, "tensor")  # legal one tier up
    with pytest.raises(ValueError, match="host citizen"):
        check_tier(region, "tile")


def test_tile_fold_steps_check_at_the_tile_tier():
    """320 §3: a tile-fold's step widens INSIDE the tile discipline — the
    quoted step tier would readmit the K-G ops the tile tier refuses."""
    ops = OPS
    b = Builder(ops)
    init = _t(np.zeros(3), ("e",))
    src = _t(np.zeros((4, 3)), ("ko", "e"))
    p_i = b.param(0, tensor_type_of_layout(init.layout))
    p_s = b.param(1, tensor_type_of_layout(src.layout))

    sb = Builder(ops)
    s_acc = sb.param(0, tensor_type_of_layout(init.layout))
    s_el = sb.param(1, tensor_type_of_layout(init.layout))
    noise = sb.emit("tl.random", s_el, dist="normal", key=7)
    nxt = sb.emit("tl.pointwise", s_acc, noise, f="add")
    step = Region(params=(s_acc, s_el), body=(sb.emit("core.yield", nxt),))

    fold = b.emit("tl.fold", p_i, p_s, regions=(step,), dim="ko", state=("acc",), element=("s",), out=("final", 0))
    region = Region(params=(p_i, p_s), body=(b.emit("core.yield", fold),))
    check_tier(region, "tensor")  # the quoted step tier admits tl.random
    with pytest.raises(ValueError, match="host citizen"):
        check_tier(region, "tile")


# --- stage's reference semantics ----------------------------------------------


def test_stage_reorders_and_charts_ride():
    """stage is the one copying op with a residence: dims reorder per
    `order`, and charts RIDE — chart-stripping stays materialize's separate
    contract (320 §4)."""
    cx = chart(0, 1, axis="x")
    u = _t(np.arange(6.0).reshape(2, 3), ("x", "y")).with_charts(x=cx)
    b = Builder(OPS)
    p = b.param(0, tensor_type_of_layout(u.layout))
    st = b.emit("tl.stage", p, level="shared", order=("y", "x"))
    got = run_region(Region(params=(p,), body=(b.emit("core.yield", st),)), [u])
    np.testing.assert_array_equal(got.to_numpy(order=("y", "x")), u.to_numpy().T)
    assert got.layout.dim("x").chart == cx  # the chart rode the copy

    mat = b.emit("tl.materialize", p, order=("y", "x"))
    got_m = run_region(Region(params=(p,), body=(b.emit("core.yield", mat),)), [u])
    assert got_m.layout.dim("x").chart is None  # materialize strips, by contract


# --- the erasure oracle --------------------------------------------------------


@pytest.mark.parametrize("flagship", [gemm_tile, stencil_tile, flash_tile])
def test_erasure_is_bit_exact(flagship):
    """A stage moves residence and presentation, never values-by-name: the
    erased region's denotation is BIT-exactly the staged one's (320 §6)."""
    f = flagship()
    vals = list(f.inputs.values())
    staged = run_region(f.region, vals)
    erased_region = erase_stages(f.region)
    check_tier(erased_region, "tile")  # erasure never leaves the tier
    erased = run_region(erased_region, vals)
    order = staged.names
    np.testing.assert_array_equal(staged.to_numpy(order=order), erased.to_numpy(order=order))


# --- the flagships against their twins -----------------------------------------


def test_gemm_tile_matches_its_naive_twin_under_the_reassociation_license():
    """The k-sum re-brackets by tile — exactly the deviation the declared
    license covers; its tolerance, not an ad-hoc one, gates the check."""
    f = gemm_tile()
    lic = next(x for x in GEMM_F16_TILES if x.kind == "reassociation")
    vals = list(f.inputs.values())
    tiled = run_region(f.region, vals).to_numpy(order=("mi", "ni"))
    naive = run_region(f.naive, vals).to_numpy(order=("mi", "ni"))
    np.testing.assert_allclose(tiled, naive, rtol=lic.rtol, atol=lic.atol)
    np.testing.assert_allclose(tiled, f.oracle(f.numpy_inputs()), rtol=lic.rtol, atol=lic.atol)


def test_flash_tile_matches_its_naive_twin_under_the_online_softmax_license():
    """The materialized softmax and the s-tiled online form are the same
    denotation re-bracketed with running-max rescale — the declared
    ``flash.online-softmax`` license names the deviation and its bound."""
    f = flash_tile()
    (lic,) = FLASH_ONLINE_SOFTMAX
    vals = list(f.inputs.values())
    tiled = run_region(f.region, vals).to_numpy(order=("t", "o"))
    naive = run_region(f.naive, vals).to_numpy(order=("t", "o"))
    np.testing.assert_allclose(tiled, naive, rtol=lic.rtol, atol=lic.atol)
    np.testing.assert_allclose(tiled, f.oracle(f.numpy_inputs()), rtol=lic.rtol, atol=lic.atol)


def test_stencil_tile_matches_its_naive_twin_bit_exactly():
    """The stencil flagship has NO licensed deviation — no reassociation,
    no demotion — so staged and naive agree to the bit; the numpy oracle
    (its own association order) gets a stated f64-roundoff tolerance."""
    f = stencil_tile()
    vals = list(f.inputs.values())
    tiled = run_region(f.region, vals).to_numpy(order=("x", "y"))
    naive = run_region(f.naive, vals).to_numpy(order=("x", "y"))
    np.testing.assert_array_equal(tiled, naive)
    np.testing.assert_allclose(tiled, f.oracle(f.numpy_inputs()), rtol=1e-13, atol=1e-14)
