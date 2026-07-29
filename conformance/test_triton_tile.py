"""The Triton pair over the tile flagships (320 §8), CUDA only.

CHECK: the Region -> Triton translator runs each flagship's OWN region on
the device and matches the f64 oracle at stated f32 tolerances (the
translated kernels are load/arith chains — no tensor-core paths, so f32
rounding is the only drift). BASELINE: the hand-written Triton twins run
at realistic sizes with NON-DIVISIBLE edges (masks earn their keep),
tl.dot pinned to ieee f32 so the comparison stays honest.
"""

import numpy as np
import pytest

pytest.importorskip("triton", reason="triton is not installed (rides the torch CUDA group)")
torch = pytest.importorskip("torch", reason="the torch reference-runtime group is not installed")

from pdum.tl.zoo.tiles import flash_tile, gemm_tile, stencil_tile  # noqa: E402

FLAGSHIPS = {"gemm": gemm_tile, "stencil": stencil_tile, "flash": flash_tile}


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")


@pytest.mark.parametrize("name", sorted(FLAGSHIPS))
def test_translated_flagships_run_on_triton(name):
    _require_cuda()
    from triton_tile import compile_tile

    f = FLAGSHIPS[name]()
    run = compile_tile(f.region)
    got = run([v.to_numpy() for v in f.inputs.values()])
    np.testing.assert_allclose(got, f.oracle(f.numpy_inputs()), rtol=1e-5, atol=1e-6)


def test_the_tile_fold_becomes_an_in_kernel_loop():
    """Structural pins on the generated source: the fold is a real loop
    (300's real-break license, landed), flash's two shared-step folds emit
    two loops, and no barrier is ever spelled — synchronization is the
    lowering's inference, never the language's (320 §3)."""
    _require_cuda()
    from triton_tile import compile_tile

    gemm_src = compile_tile(FLAGSHIPS["gemm"]().region).source
    assert gemm_src.count("for ") == 1
    flash_src = compile_tile(FLAGSHIPS["flash"]().region).source
    assert flash_src.count("for ") == 2
    assert "barrier" not in gemm_src and "barrier" not in flash_src


def test_gemm_baseline_matches_numpy():
    """f32 accumulation over K=48 under ieee tl.dot; sizes are deliberate
    non-multiples of the 32-blocks so every mask path runs."""
    _require_cuda()
    from triton_zoo import gemm_triton

    rng = np.random.default_rng(5)
    a, b = rng.standard_normal((90, 48)), rng.standard_normal((48, 70))
    np.testing.assert_allclose(gemm_triton(a, b), a @ b, rtol=1e-4, atol=1e-5)


def test_flash_baseline_matches_numpy():
    """The online-softmax kernel at T=100 (masked tail block) vs the
    materialized softmax in f64."""
    _require_cuda()
    from triton_zoo import flash_triton

    rng = np.random.default_rng(6)
    T, D = 100, 32
    q, k, v = (rng.standard_normal((T, D)) for _ in range(3))
    sc = q @ k.T
    sm = np.where(np.tril(np.ones((T, T), dtype=bool)), sc, -np.inf)
    e = np.exp(sm - sm.max(1, keepdims=True))
    want = (e / e.sum(1, keepdims=True)) @ v
    np.testing.assert_allclose(flash_triton(q, k, v), want, rtol=1e-4, atol=1e-5)


def test_stencil_baseline_matches_numpy():
    _require_cuda()
    from triton_zoo import stencil_triton

    rng = np.random.default_rng(7)
    u = rng.standard_normal((102, 82))
    lap = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1]
    want = u[1:-1, 1:-1] + 0.1 * lap
    np.testing.assert_allclose(stencil_triton(u), want, rtol=1e-4, atol=1e-6)
