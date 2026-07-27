"""The spinning cylinder — the graphics zoo's first entry (P8).

Every P8 feature in one demo: a MESH as a RECORD vertex buffer (fields
theta/h over the vertex_id dim — the structured dtype IS the memory
shape, 200 §4), the geometry RIPPLED by a compute kernel over the
FIELD VIEWS (the phase an unmarked capture — a WARM uniform across
frames, the literal doctrine live), the rotation ANGLE as a
vertex-shader uniform converted to the world transform IN the shader,
and an analytic-AA fragment (value_and_grad — fwidth with no 2x2 quad)
shading a pattern PERIODIC across the vertical seam.

Reference-tier render loop: "see it work". Device goldens are the
conformance executor's job; the reference rasterizer has no depth
buffer, so the far side overdraws in draw order (BOUNDARIES.md).
"""

import numpy as np
from pdum.dsl import jit, value_and_grad
from pdum.dsl.intrinsics import clamp  # noqa: F401 — inlines by capture-and-call
from pdum.dsl.markers import cos, sin, sqrt  # noqa: F401 — bare in bodies
from pdum.tl import Tensor, compute, global_idx, thread_idx  # noqa: F401 — bodies' globals
from pdum.tl.graphics import fragment, pair, position, render, vertex

TAU = 6.283185307179586

_PHASE = 0.0  # the ripple phase: an unmarked capture — warm across frames


MESH_DT = np.dtype([("theta", "<f8"), ("h", "<f8")])


def cylinder_mesh(segments: int = 32) -> Tensor:
    """The side surface as a RECORD vertex buffer: two triangles per
    segment; each element a (theta, h) record — fields by NAME, no
    anonymous columns."""
    verts = []
    for s in range(segments):
        t0, t1 = s * TAU / segments, (s + 1) * TAU / segments
        verts += [(t0, -1.0), (t1, -1.0), (t0, 1.0), (t1, -1.0), (t1, 1.0), (t0, 1.0)]
    return Tensor.from_numpy(np.array(verts, dtype=MESH_DT), ("vertex_id",))


@compute
def ripple(theta_in, h_in, theta_out, h_out):
    """The geometry ripple over the record's FIELD VIEWS (the 200 §4
    door): h gains a phase-shifted sine of theta; the phase rides the
    uniform channel. No computed reads, no column tricks — the fields
    have names."""
    (i,) = global_idx("vertex_id")
    theta_out[i] = theta_in[i]
    h_out[i] = h_in[i] + sin(theta_in[i] * 3.0 + _PHASE) * 0.08


def spun(angle: float):
    """The rotation as a UNIFORM: the world transform is built IN the
    vertex shader from the captured angle — never a host-side matrix."""

    @vertex
    def vs(verts):
        (vid,) = thread_idx("vertex_id")
        p = verts[vid]  # the record element: fields by name
        world = p.theta + angle  # rotation -> world space, in-shader
        px = sin(world) * 0.85
        py = p.h * 0.7 + cos(world) * 0.12  # a hint of tilt
        u = p.theta * (1.0 / TAU)  # noqa: F841 — a claimed varying (the tagless law)
        v = p.h * 0.5 + 0.5  # noqa: F841 — a claimed varying
        return position(px, py)

    return vs


def stripes(k: float):
    @jit()
    def g(u, v):
        return sin(u * k)  # periodic in u — continuous across the seam by construction

    return g


def demo_frames(angles=(0.0, 1.2, 2.4), size=(48, 64)):
    """The render loop: per frame, rebind the ripple phase (a warm
    compute relaunch), rebuild the world from the angle uniform, draw
    with the analytic-AA shader. Returns the frames as numpy images."""
    global _PHASE
    mesh = cylinder_mesh()
    g = value_and_grad(stripes(3.0 * TAU), wrt=("u", "v"))

    @fragment
    def shade(f, varying):
        s, (du, dv) = f(varying.u, varying.v)
        w = sqrt(du * du + dv * dv) * 0.7 + 1e-6
        return clamp(s / w * 0.5 + 0.5, 0.0, 1.0)  # the analytic edge, one footprint wide

    n = mesh.layout.dim("vertex_id").size
    frames = []
    for i, angle in enumerate(angles):
        _PHASE = 0.9 * i  # rebinding the unmarked capture: a WARM relaunch, fresh value
        rippled = Tensor.from_numpy(np.zeros(n, dtype=MESH_DT), ("vertex_id",))
        ripple(mesh.field("theta"), mesh.field("h"), rippled.field("theta"), rippled.field("h"))
        img = Tensor.from_numpy(np.zeros(size), ("y", "x"))
        render(pair(spun(angle), shade), rippled, g, target=img)
        frames.append(img.to_numpy())
    return frames
