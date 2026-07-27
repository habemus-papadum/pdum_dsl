"""The spinning cylinder — the graphics zoo's first entry (P8).

Every P8 feature in one demo: a MESH as a vertex ARRAY (per-vertex
attributes theta/h over the vid dim), the geometry RIPPLED by a compute
kernel (theta read at a computed index; the phase an unmarked capture —
a WARM uniform across frames, the literal doctrine live), the rotation
ANGLE as a vertex-shader uniform converted to the world transform IN
the shader, and an analytic-AA fragment (value_and_grad — fwidth with
no 2x2 quad) shading a pattern PERIODIC across the vertical seam.

Reference-tier render loop: "see it work". Device goldens are the
conformance executor's job; the reference rasterizer has no depth
buffer, so the far side overdraws in draw order (BOUNDARIES.md).
"""

import numpy as np
from pdum.dsl import jit, value_and_grad
from pdum.dsl.intrinsics import clamp  # noqa: F401 — inlines by capture-and-call
from pdum.dsl.markers import cos, ge, sin, sqrt, where  # noqa: F401 — bare in bodies
from pdum.tl import Tensor, compute, f32, global_idx, i32, thread_idx  # noqa: F401 — bodies' globals
from pdum.tl.graphics import fragment, pair, position, render, vertex

TAU = 6.283185307179586

_PHASE = 0.0  # the ripple phase: an unmarked capture — warm across frames


def cylinder_mesh(segments: int = 32) -> Tensor:
    """The side surface as a vertex ARRAY: two triangles per segment;
    attributes per vertex — c=0 is theta, c=1 is h."""
    verts = []
    for s in range(segments):
        t0, t1 = s * TAU / segments, (s + 1) * TAU / segments
        verts += [(t0, -1.0), (t1, -1.0), (t0, 1.0), (t1, -1.0), (t1, 1.0), (t0, 1.0)]
    return Tensor.from_numpy(np.asarray(verts, dtype=np.float64), ("vertex_id", "c"))


@compute
def ripple(src, dst):
    """The geometry ripple: h gains a phase-shifted sine of theta —
    theta read at a COMPUTED index (the c=0 column), the phase riding
    the uniform channel."""
    i, c = global_idx("vertex_id", "c")
    theta = src[i32(i), i32(c) * 0.0]  # the theta column, broadcast per vertex row
    dst[i, c] = src[i, c] + where(ge(f32(c), 0.5), sin(theta * 3.0 + _PHASE) * 0.08, 0.0)


def spun(angle: float):
    """The rotation as a UNIFORM: the world transform is built IN the
    vertex shader from the captured angle — never a host-side matrix."""

    @vertex
    def vs(verts):
        theta = verts.select(c=0)
        h = verts.select(c=1)
        world = theta + angle  # rotation -> world space, in-shader
        px = sin(world) * 0.85
        py = h * 0.7 + cos(world) * 0.12  # a hint of tilt
        u = theta * (1.0 / TAU)  # noqa: F841 — a claimed varying (the tagless law)
        v = h * 0.5 + 0.5  # noqa: F841 — a claimed varying
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
        rippled = Tensor.from_numpy(np.zeros((n, 2)), ("vertex_id", "c"))
        ripple(mesh, rippled)
        img = Tensor.from_numpy(np.zeros(size), ("y", "x"))
        render(pair(spun(angle), shade), rippled, g, target=img)
        frames.append(img.to_numpy())
    return frames
