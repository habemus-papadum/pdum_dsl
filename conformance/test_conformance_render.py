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


def test_vertex_pulling_buffer_quad_differential():
    """STORAGE-BUFFER VERTEX PULLING (stage 5b): positions and a varying
    pulled from a (vertex_id, c) attribute buffer — the device twin
    matches the reference interpolator pixel-for-pixel on grid-aligned
    triangles (no pixel center touches an edge)."""
    _require_device()
    quad = T(
        [[-1.0, -1.0], [0.0, -1.0], [-1.0, 1.0], [0.0, -1.0], [0.0, 1.0], [-1.0, 1.0]],
        ("vertex_id", "c"),
    )

    @vertex
    def mesh(verts):
        x = verts.select(c=0)
        y = verts.select(c=1)
        u = y * 0.5 + 0.5  # noqa: F841 — a claimed varying (the tagless law)
        return position(x, y)

    @fragment
    def shade(varying):
        return varying.u * 2.0 + 0.25

    pso = pair(mesh, shade)
    ref = T(np.zeros((8, 8)), ("y", "x"))
    render(pso, quad, target=ref)
    got = render_wgpu(pso, quad, shape=(8, 8))
    assert ref.to_numpy().max() > 0.0  # the left half is covered
    np.testing.assert_allclose(got, ref.to_numpy(), atol=1e-6)


def test_the_cylinder_renders_on_the_device():
    """The zoo's cylinder through real vertex pulling end to end: the
    rippled mesh buffer pulled in the vertex stage, the angle uniform
    riding the vertex-stage slot channel, the staged value_and_grad AA
    fragment. On this mesh the coverage sets agree EXACTLY (measured:
    zero pixels differ); the stated tolerance is f32 interpolation
    through the steep AA ramp (max observed ~4e-4 vs the f64
    reference)."""
    _require_device()
    from pdum.dsl import value_and_grad
    from pdum.dsl.intrinsics import clamp  # noqa: F401 — inlines by capture-and-call
    from pdum.dsl.markers import sqrt  # noqa: F401 — bare in the body
    from pdum.tl.graphics import fragment as _fragment
    from pdum.tl.zoo.cylinder import MESH_DT, TAU, cylinder_mesh, ripple, spun, stripes

    mesh = cylinder_mesh(16)
    n = mesh.layout.dim("vertex_id").size
    rippled = Tensor.from_numpy(np.zeros(n, dtype=MESH_DT), ("vertex_id",))
    ripple(mesh.field("theta"), mesh.field("h"), rippled.field("theta"), rippled.field("h"))
    g = value_and_grad(stripes(3.0 * TAU), wrt=("u", "v"))

    @_fragment
    def shade(f, varying):
        s, (du, dv) = f(varying.u, varying.v)
        w = sqrt(du * du + dv * dv) * 0.7 + 1e-6
        return clamp(s / w * 0.5 + 0.5, 0.0, 1.0)

    pso = pair(spun(1.2), shade)
    ref = T(np.zeros((32, 48)), ("y", "x"))
    render(pso, rippled, g, target=ref)
    got = render_wgpu(pso, rippled, g, shape=(32, 48))
    assert np.isfinite(got).all()
    assert (got != 0.0).any()  # the cylinder covers pixels
    np.testing.assert_allclose(got, ref.to_numpy(), atol=2e-3)


def test_vertex_pulling_record_quad_differential():
    """The record face of vertex pulling: a real WGSL struct per record
    buffer, fields pulled by NAME at the vertex index — pixel-exact
    against the reference on the aligned quad."""
    _require_device()
    dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("w", "<f8")])
    pts = [(-1.0, -1.0, 0.0), (0.0, -1.0, 1.0), (-1.0, 1.0, 0.5), (0.0, -1.0, 1.0), (0.0, 1.0, 1.0), (-1.0, 1.0, 0.5)]
    quad = Tensor.from_numpy(np.array(pts, dtype=dt), ("vertex_id",))

    @vertex
    def mesh(verts):
        (vid,) = thread_idx("vertex_id")
        p = verts[vid]
        u = p.w * 1.0  # noqa: F841 — a claimed varying
        return position(p.x, p.y)

    @fragment
    def shade(varying):
        return varying.u * 2.0 + 0.25

    pso = pair(mesh, shade)
    ref = T(np.zeros((8, 8)), ("y", "x"))
    render(pso, quad, target=ref)
    got = render_wgpu(pso, quad, shape=(8, 8))
    assert ref.to_numpy().max() > 0.0
    np.testing.assert_allclose(got, ref.to_numpy(), atol=1e-6)
