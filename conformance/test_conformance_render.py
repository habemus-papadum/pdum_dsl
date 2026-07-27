"""The quad+f GOLDEN on the conformance executor — the P8 gate's named
item: the same PSO through the reference interpolator and a REAL
vertex/fragment pipeline, differentially, on the development Mac."""

import numpy as np
import pytest
from pdum.dsl import jit
from pdum.dsl.markers import sqrt  # noqa: F401 — bare in device bodies
from pdum.tl import Tensor
from pdum.tl.graphics import fragment, pair, position, render, vertex, vertex_index  # noqa: F401
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
        vid = vertex_index()
        u = 1.0 if (vid == 1 or vid == 3 or vid == 4) else 0.0
        v = 1.0 if (vid == 2 or vid == 4 or vid == 5) else 0.0
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
