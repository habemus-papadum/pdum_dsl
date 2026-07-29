"""The jax BASELINE column agrees with the zoo's numpy denotation (310).

The differential runs the JITTED path — that is what the rig times, and
eager shares the same trace. f64 both sides (x64 on); the tolerance covers
only operation-order drift (XLA fuses and reassociates): rtol=1e-9 holds at
zoo shapes and is asserted as stated. CUDA runs the same battery when a
device is present."""

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="the jax reference-runtime group is not installed")

from jax_zoo import BASELINES  # noqa: E402 — needs the importorskip above
from pdum.tl.zoo import (  # noqa: E402
    flash_attention,
    gated_attention,
    gpt2,
    heat2d,
    llama_block,
    moe,
    qknorm_attention,
    sliding_attention,
    tiled_matmul,
)

ENTRIES = {
    "gpt2": gpt2,
    "llama": llama_block,
    "sliding": sliding_attention,
    "gated": gated_attention,
    "qknorm": qknorm_attention,
    "flash": flash_attention,
    "heat2d": heat2d,
    "moe": moe,
    "gemm": tiled_matmul,
}


def _differential(name, device):
    m = ENTRIES[name]()
    dev = jax.devices("gpu" if device == "cuda" else "cpu")[0]
    inp = {k: jax.device_put(v, dev) for k, v in m.numpy_inputs().items()}
    got = np.asarray(jax.jit(BASELINES[name])(inp))
    np.testing.assert_allclose(got, m.ref(m.numpy_inputs()), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_baseline_matches_numpy_on_cpu(name):
    _differential(name, "cpu")


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_baseline_matches_numpy_on_cuda(name):
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("no CUDA device available to jax")
    _differential(name, "cuda")
