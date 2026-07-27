"""The graphics zoo: the spinning cylinder (P8) — reference tier."""

import numpy as np
from pdum.tl.zoo.cylinder import TAU, cylinder_mesh, demo_frames, stripes


def test_the_spinning_cylinder_renders_a_loop():
    """Mesh -> compute-kernel ripple (warm phase uniform) -> angle
    uniform -> world transform in the vertex shader -> analytic-AA
    periodic shading, three frames."""
    frames = demo_frames()
    assert len(frames) == 3
    for fr in frames:
        assert np.isfinite(fr).all()
        assert 0.0 <= fr.min() and fr.max() <= 1.0
        assert fr.std() > 0.01  # something actually rendered
    assert any(not np.allclose(frames[0], f) for f in frames[1:])  # it spins


def test_the_pattern_is_periodic_across_the_seam():
    """The shader's field is continuous at theta = 0 == TAU by
    construction — the seam can never crack."""
    from pdum.dsl.reference import reference

    g = reference(stripes(3.0 * TAU))
    assert abs(g(0.0, 0.5) - g(1.0, 0.5)) < 1e-9


def test_the_mesh_is_a_closed_side_surface():
    mesh = cylinder_mesh(segments=8)
    arr = mesh.to_numpy(order=("vertex_id", "c"))
    assert arr.shape == (8 * 6, 2)
    assert abs(arr[:, 0].max() - TAU) < 1e-12 and arr[:, 0].min() == 0.0
