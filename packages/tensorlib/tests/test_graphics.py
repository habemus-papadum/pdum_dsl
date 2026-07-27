"""The graphics tier over the reference interpolator (P8, 200 §S.4).

The committed spellings live in test_kernel_spec.py; this battery pins
the machinery around them — the vertex-ARRAY draw above all (owner-
requested): per-vertex attributes are ordinary tensors over the ``vid``
dim, attribute fields are ``.select()`` views of them, and the draw
count IS the vid extent.
"""

import numpy as np
from pdum.tl import Tensor, f32, thread_idx  # noqa: F401 — ambient vocabulary: bodies' globals
from pdum.tl.graphics import fragment, pair, position, render, vertex


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def test_vertex_arrays_drive_the_draw():
    """A REAL vertex array: one right triangle from a (vid, c) attribute
    buffer covering the lower-left half of NDC; the count comes from the
    buffer, the attributes ride as fields, the varyings interpolate."""

    @vertex
    def mesh(verts):
        u = verts.select(c=0)  # claimed varyings: the attribute fields
        v = verts.select(c=1)
        return position(u, v)

    @fragment
    def shade(varying):
        return varying.u * 0.0 + 1.0  # flat white over the covered pixels

    tri = T([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]], ("vertex_id", "c"))
    img = T(np.zeros((8, 8)), ("y", "x"))
    render(pair(mesh, shade), tri, target=img)
    out = img.to_numpy()
    assert out[0, 0] == 1.0  # the lower-left pixel center is inside
    assert out[-1, -1] == 0.0  # the upper-right is outside
    assert out.sum() == 36.0  # exactly the pixel centers with x+y <= 0


def test_interpolated_varyings_are_barycentric():
    """The varying field interpolates linearly across the triangle: a
    gradient from the u attribute lands as the pixel-center u value."""

    @vertex
    def mesh(verts):
        u = verts.select(c=0)
        v = verts.select(c=1)
        return position(u * 2.0 - 1.0, v * 2.0 - 1.0)

    @fragment
    def shade(varying):
        return varying.u

    # two triangles spanning the unit square in (u, v)
    quad = T(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        ("vertex_id", "c"),
    )
    img = T(np.zeros((4, 4)), ("y", "x"))
    render(pair(mesh, shade), quad, target=img)
    want = np.broadcast_to((np.arange(4) + 0.5) / 4.0, (4, 4))
    np.testing.assert_allclose(img.to_numpy(), want, rtol=1e-12)


def test_pairing_refuses_missing_varyings_with_the_names():
    """The other half of subset pairing: a fragment requiring a field no
    vertex shader produces refuses AT PAIR TIME, naming both sides."""
    import pytest

    @vertex
    def lean():
        (vid,) = thread_idx("vertex_id")
        u = f32(vid) * 0.0
        return position(u, u)

    @fragment
    def needs_w(varying):
        return varying.w

    with pytest.raises(ValueError, match=r"pairing refused.*\['w'\].*missing"):
        pair(lean, needs_w)
