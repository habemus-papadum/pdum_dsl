"""The quad+f GOLDEN on the conformance executor — the P8 gate's named
item: the same PSO through the reference interpolator and a REAL
vertex/fragment pipeline, differentially, on the development Mac."""

import numpy as np
import pytest
from pdum.dsl import jit
from pdum.dsl.markers import sqrt  # noqa: F401 — bare in device bodies
from pdum.tl import Tensor, i32, thread_idx  # noqa: F401 — ambient vocabulary: bodies' globals
from pdum.tl.graphics import fragment, pair, position, render, vertex  # noqa: F401
from wgsl_executor import Untranslatable, render_wgpu


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def _require_device():
    pytest.importorskip("wgpu")
    try:
        from pdum.tl.graphics import _device

        _device()
    except Exception as exc:
        pytest.skip(f"no WebGPU device available: {exc}")


def test_the_quad_f_golden_runs_differentially():
    """The gate: the S.4 quad + the compute zoo's f, rendered by the
    reference interpolator AND by the GPU rasterizer, pixel-identical
    (the circle boundary is provably f32-stable: no pixel center lies
    within f32 epsilon of d == r)."""
    _require_device()

    def circle(cy, cx, r):
        @jit()
        def go(y, x):
            d = sqrt((y - cy) * (y - cy) + (x - cx) * (x - cx))
            return 1.0 if d > r else 0.0

        return go

    @vertex
    def quad():
        (vid,) = thread_idx("vertex_id")
        i = i32(vid)
        u = 1.0 if (i == 1 or i == 3 or i == 4) else 0.0
        v = 1.0 if (i == 2 or i == 4 or i == 5) else 0.0
        return position(u * 2.0 - 1.0, v * 2.0 - 1.0)

    @fragment
    def shade(f, varying):
        return f(varying.v * 23.0, varying.u * 39.0)

    pso = pair(quad, shade)
    ref = T(np.zeros((24, 40)), ("y", "x"))
    render(pso, circle(12.0, 20.0, 8.0), target=ref)
    try:
        got = render_wgpu(pso, circle(12.0, 20.0, 8.0), shape=(24, 40))
    except Untranslatable as exc:
        pytest.skip(f"no WGSL translation yet: {exc}")
    assert ref.to_numpy().min() == 0.0 and ref.to_numpy().max() == 1.0  # both colors present
    np.testing.assert_allclose(got, ref.to_numpy(), atol=1e-6)


def _quad():
    @vertex
    def quad():
        (vid,) = thread_idx("vertex_id")
        i = i32(vid)
        u = 1.0 if (i == 1 or i == 3 or i == 4) else 0.0
        v = 1.0 if (i == 2 or i == 4 or i == 5) else 0.0
        return position(u * 2.0 - 1.0, v * 2.0 - 1.0)

    return quad


def test_the_textured_quad_golden_nearest():
    """The gate's other golden: a 16x16 pattern sampled by the fragment
    (nearest, clamp). Texel SELECTION is exact (no pixel maps to a texel
    boundary at these sizes); the residue is the hardware sRGB decode
    unit, which the spec permits to deviate from the exact IEC curve by
    ~0.5/255 in linear light — the tolerance states exactly that."""
    _require_device()
    from pdum.tl.graphics import sample, sampler, upload

    pattern = T((np.arange(256.0).reshape(16, 16) % 16.0) / 16.0, ("y", "x"))
    tex = upload(pattern)
    smp = sampler(filter="nearest", address="clamp")

    @fragment
    def shade(varying):
        return sample(tex, smp, (varying.v, varying.u), lod=0)

    pso = pair(_quad(), shade)
    ref = T(np.zeros((32, 32)), ("y", "x"))
    render(pso, target=ref)
    got = render_wgpu(pso, shape=(32, 32))
    assert ref.to_numpy().std() > 0.0  # the pattern actually landed
    np.testing.assert_allclose(got, ref.to_numpy(), atol=0.5 / 255.0)


def test_the_textured_quad_linear_agrees_within_weight_precision():
    """Bilinear filtering: hardware interpolates with ~8-bit fractional
    weights, so the differential states its tolerance instead of
    pretending exactness (recorded v1 limit)."""
    _require_device()
    from pdum.tl.graphics import sample, sampler, upload

    rng = np.random.default_rng(3)
    tex = upload(T(rng.random((16, 16)), ("y", "x")))
    smp = sampler(filter="linear", address="clamp")

    @fragment
    def shade(varying):
        return sample(tex, smp, (varying.v, varying.u), lod=0)

    pso = pair(_quad(), shade)
    ref = T(np.zeros((24, 40)), ("y", "x"))
    render(pso, target=ref)
    got = render_wgpu(pso, shape=(24, 40))
    np.testing.assert_allclose(got, ref.to_numpy(), atol=1.0 / 64.0)
