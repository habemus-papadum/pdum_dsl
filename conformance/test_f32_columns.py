"""The f32 face of the CHECK columns (310): dtype is representation, never
semantics (200 §4) — the SAME regions, evaluated at f32, against the f64
numpy oracle under stated tolerances.

First subjects: heat2d (a stencil — well-conditioned diffusion, few steps)
and gemm (short-k dot products). The translated columns are pointwise/
reduce chains — no tensor-core matmul paths — so f32 rounding is the only
drift: rtol 1e-5 / atol 1e-6 holds with margin at zoo shapes. Each test
also asserts the output dtype IS f32 — a column that quietly promotes to
f64 would pass the values while voiding the subject."""

import numpy as np
import pytest

from pdum.tl.zoo import heat2d, tiled_matmul

ENTRIES = {"heat2d": heat2d, "gemm": tiled_matmul}
TOL = dict(rtol=1e-5, atol=1e-6)


def _param_inputs(m):
    out = {}
    for p in m.region.params:
        name = m.names[id(p)]
        order = tuple(d.name for d in p.type.dims)
        out[name] = m.inputs[name].to_numpy(order=order)
    return out


def _assert_f32_column(run_named, name, device, dtype):
    m = ENTRIES[name]()
    vals = run_named(m.region, _param_inputs(m), m.names, device=device, dtype=dtype)
    got = vals[m.out].numpy(order=m.order)
    assert got.dtype == np.float32, f"the column promoted to {got.dtype} — not an f32 subject"
    np.testing.assert_allclose(got, m.ref(m.numpy_inputs()), **TOL)


@pytest.mark.parametrize("name", sorted(ENTRIES))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_f32_column(name, device):
    torch = pytest.importorskip("torch", reason="the torch reference-runtime group is not installed")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device available")
    from torch_evaluator import run_named_torch

    _assert_f32_column(run_named_torch, name, device, torch.float32)


@pytest.mark.parametrize("name", sorted(ENTRIES))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_f32_column(name, device):
    jax = pytest.importorskip("jax", reason="the jax reference-runtime group is not installed")
    if device == "cuda" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("no CUDA device available to jax")
    import jax.numpy as jnp

    from jax_evaluator import run_named_jax

    _assert_f32_column(run_named_jax, name, device, jnp.float32)
