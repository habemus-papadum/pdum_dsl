"""The BASELINE column agrees with the zoo's numpy denotation (310).

Independently-authored idiomatic torch (sdpa, F.layer_norm, index_add) vs
each entry's `ref` on identical inputs — f64 both sides, so the tolerance
covers only operation-order drift (fused kernels reassociate): rtol=1e-9
holds comfortably at zoo shapes and is asserted as stated. CUDA runs the
same battery when a device is present."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the torch reference-runtime group is not installed")

from torch_zoo import BASELINES  # noqa: E402 — needs the importorskip above

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
    inp = {
        k: torch.as_tensor(v).to(device=device) for k, v in m.numpy_inputs().items()
    }
    got = BASELINES[name](inp).cpu().numpy()
    np.testing.assert_allclose(got, m.ref(m.numpy_inputs()), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_baseline_matches_numpy_on_cpu(name):
    _differential(name, "cpu")


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_baseline_matches_numpy_on_cuda(name):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    _differential(name, "cuda")
