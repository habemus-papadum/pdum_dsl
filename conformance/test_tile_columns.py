"""The tile flagships on the framework columns (310 x 320): the same tile
regions the reference oracle serves, evaluated on the torch and jax
substrates — stage is a copy there too (the type carries the chosen
order), so the columns witness the tile tier from independent substrates.
Tolerances quote the DECLARED license where one applies (gemm's k-sum
re-bracketing); the stencil has no licensed deviation and holds at f64
roundoff."""

import numpy as np
import pytest

from pdum.tl.licenses import FLASH_ONLINE_SOFTMAX, GEMM_F16_TILES
from pdum.tl.zoo.tiles import flash_tile, gemm_tile, stencil_tile

FLAGSHIPS = {"gemm": gemm_tile, "stencil": stencil_tile, "flash": flash_tile}
_GEMM = next(x for x in GEMM_F16_TILES if x.kind == "reassociation")
(_FLASH,) = FLASH_ONLINE_SOFTMAX
TOL = {
    "gemm": dict(rtol=_GEMM.rtol, atol=_GEMM.atol),
    "flash": dict(rtol=_FLASH.rtol, atol=_FLASH.atol),
    "stencil": dict(rtol=1e-13, atol=1e-14),
}


def _differential(run, to_numpy, name, device):
    f = FLAGSHIPS[name]()
    vals = [v.to_numpy() for v in f.inputs.values()]  # param order, param dims
    got = to_numpy(run(f.region, vals, device=device))
    np.testing.assert_allclose(got, f.oracle(f.numpy_inputs()), **TOL[name])


@pytest.mark.parametrize("name", sorted(FLAGSHIPS))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tile_flagships_on_torch(name, device):
    torch = pytest.importorskip("torch", reason="the torch reference-runtime group is not installed")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    from torch_evaluator import run_region_torch

    _differential(run_region_torch, lambda t: t.cpu().numpy(), name, device)


@pytest.mark.parametrize("name", sorted(FLAGSHIPS))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tile_flagships_on_jax(name, device):
    jax = pytest.importorskip("jax", reason="the jax reference-runtime group is not installed")
    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("no CUDA device available to jax")
    from jax_evaluator import run_region_jax

    _differential(run_region_jax, np.asarray, name, device)
