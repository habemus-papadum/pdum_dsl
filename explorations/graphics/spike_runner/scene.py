"""The cylinder scene, built once and driven by two host scalars.

The zoo's ``demo_frames`` is a cold loop by construction: per frame it
allocates a fresh ``rippled`` tensor, launches ``ripple`` (compile-keyed on
the captured ``_PHASE``'s TYPE, so a hit — but a full numpy re-execution),
allocates a fresh target, and calls ``render(pair(spun(angle), shade), ...)``
which re-lowers both stages every time. This module keeps the same scene but
separates what is BUILT from what is DRIVEN.

The two per-frame scalars and where they live:

  phase  a module GLOBAL in ``pdum.tl.zoo.cylinder`` that the ripple kernel
         captures unmarked. Driving it means assigning to another module's
         private global; there is no parameter door.

  angle  a CLOSURE variable of a fresh ``spun(angle)`` vertex shader. Driving
         it means minting a new VertexShader object per frame and trusting
         that it is layout-identical to the compiled one.
"""

from __future__ import annotations

import numpy as np
from pdum.dsl import value_and_grad
from pdum.dsl.intrinsics import clamp  # noqa: F401 — inlines by capture-and-call
from pdum.dsl.markers import sqrt  # noqa: F401 — bare in the fragment body
from pdum.tl import Tensor
from pdum.tl.graphics import fragment, pair
from pdum.tl.zoo import cylinder as cyl
from pdum.tl.zoo.cylinder import MESH_DT, TAU, cylinder_mesh, ripple, spun, stripes

SEGMENTS = 16
SIZE = (64, 96)  # (H, W)


def make_shade():
    g = value_and_grad(stripes(3.0 * TAU), wrt=("u", "v"))

    @fragment
    def shade(f, varying):
        s, (du, dv) = f(varying.u, varying.v)
        w = sqrt(du * du + dv * dv) * 0.7 + 1e-6
        return clamp(s / w * 0.5 + 0.5, 0.0, 1.0)

    return shade, g


def make_buffers(segments: int = SEGMENTS):
    """The mesh and its rippled twin: two record buffers, four field views,
    exactly TWO host ``Buffer`` objects."""
    mesh = cylinder_mesh(segments)
    n = mesh.layout.dim("vertex_id").size
    rippled = Tensor.from_numpy(np.zeros(n, dtype=MESH_DT), ("vertex_id",))
    return mesh, rippled


def set_phase(p: float) -> None:
    """Drive the ripple. Assigning into another module's private global is
    the only door the unmarked capture leaves open."""
    cyl._PHASE = p


def phase_of(i: int) -> float:
    return 0.35 * i


def angle_of(i: int) -> float:
    return 0.09 * i


def make_pso(angle: float):
    shade, g = make_shade()
    return pair(spun(angle), shade), g


def vs_fn_for(angle: float):
    """The frame loop's vertex-shader handle for this angle."""
    return spun(angle).fn


def reference_frame(mesh, angle: float, phase: float, size=None):
    """The numpy reference render of one frame — the golden.

    Runs the ripple through the ordinary kernel launch (numpy executor) into
    a FRESH pair of buffers, so it cannot observe anything the device path
    wrote."""
    from pdum.tl.graphics import render

    size = SIZE if size is None else size
    set_phase(phase)
    n = mesh.layout.dim("vertex_id").size
    rippled = Tensor.from_numpy(np.zeros(n, dtype=MESH_DT), ("vertex_id",))
    ripple(mesh.field("theta"), mesh.field("h"), rippled.field("theta"), rippled.field("h"))
    shade, g = make_shade()
    img = Tensor.from_numpy(np.zeros(size), ("y", "x"))
    render(pair(spun(angle), shade), rippled, g, target=img)
    return img.to_numpy()
