"""The torch CHECK column over the zoo (310): every entry's region, evaluated
on the torch substrate at f64, matches its numpy denotation — the same
assertion test_zoo.py makes of the reference interpreter, on a second,
independent substrate. When a CUDA device is present the whole battery runs
there too (whole-chain-on-device: every op of the region computes on the
device column, nothing quietly falls back to host — PR #8's rule)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the torch reference-runtime group is not installed")

from torch_evaluator import Untranslatable, run_named_torch  # noqa: E402 — needs the importorskip above

from pdum.tl.zoo import (  # noqa: E402
    fdtd1d_staggered,
    flash_attention,
    gated_attention,
    gpt2,
    heat2d,
    llama_block,
    moe,
    qknorm_attention,
    sliding_attention,
    tiled_matmul,
    unrolled_trainer,
)

ENTRIES = {
    "gpt2": gpt2,
    "llama": llama_block,
    "sliding": sliding_attention,
    "gated": gated_attention,
    "qknorm": qknorm_attention,
    "flash": flash_attention,
    "flash_naive": lambda: flash_attention(naive=True),
    "heat2d": heat2d,
    "fdtd": fdtd1d_staggered,
    "moe": moe,
    "trainer": unrolled_trainer,
    "gemm": tiled_matmul,
}


def _param_inputs(m):
    """Inputs keyed by claimed name, each array in its param's type-dim order
    (the evaluator's binding convention)."""
    out = {}
    for p in m.region.params:
        name = m.names[id(p)]
        order = tuple(d.name for d in p.type.dims)
        out[name] = m.inputs[name].to_numpy(order=order)
    return out


def _differential(name, device):
    m = ENTRIES[name]()
    try:
        vals = run_named_torch(m.region, _param_inputs(m), m.names, device=device)
    except Untranslatable as exc:
        pytest.skip(f"no torch translation yet: {exc}")
    got = vals[m.out].numpy(order=m.order)
    np.testing.assert_allclose(got, m.ref(m.numpy_inputs()), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_zoo_forward_matches_numpy_on_torch_cpu(name):
    _differential(name, "cpu")


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_zoo_forward_matches_numpy_on_torch_cuda(name):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    _differential(name, "cuda")
