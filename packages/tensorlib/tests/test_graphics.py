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


def test_multiple_buffers_and_the_draw_count_consistency():
    """Multiple vertex buffers bind together; the draw count derives from
    their shared vertex_id extent — disagreement refuses at lower time."""
    import pytest

    @vertex
    def mesh(pos, extra):
        x = pos.select(c=0)
        y = pos.select(c=1)
        u = extra * 0.5  # a second, 1-D per-vertex buffer  # noqa: F841
        return position(x, y)

    @fragment
    def shade(varying):
        return varying.u

    tri = T([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]], ("vertex_id", "c"))
    good = T([0.0, 1.0, 2.0], ("vertex_id",))
    img = T(np.zeros((4, 4)), ("y", "x"))
    render(pair(mesh, shade), tri, good, target=img)
    assert np.isfinite(img.to_numpy()).all()
    bad = T([0.0, 1.0], ("vertex_id",))
    with pytest.raises(ValueError, match="disagree on the draw count"):
        render(pair(mesh, shade), tri, bad, target=img)


def test_record_vertex_buffers_fields_by_name():
    """A RECORD vertex buffer: the structured dtype IS the memory shape
    (200 §4); verts[vid] admits the Coordinate and yields the element,
    fields by NAME — no anonymous component column."""
    dt = np.dtype([("x", "<f8"), ("y", "<f8")])
    tri = Tensor.from_numpy(np.array([(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)], dtype=dt), ("vertex_id",))

    @vertex
    def mesh(verts):
        (vid,) = thread_idx("vertex_id")
        p = verts[vid]
        u = p.x * 0.0 + 1.0  # noqa: F841 — a claimed varying (the tagless law)
        return position(p.x, p.y)

    @fragment
    def shade(varying):
        return varying.u

    img = T(np.zeros((8, 8)), ("y", "x"))
    render(pair(mesh, shade), tri, target=img)
    assert img.to_numpy().sum() == 36.0  # the same half-plane as the float-buffer case


def test_record_buffer_refusals():
    import pytest

    dt = np.dtype([("x", "<f8"), ("y", "<f8")])
    tri = Tensor.from_numpy(np.array([(0.0, 0.0)] * 3, dtype=dt), ("vertex_id",))
    img = T(np.zeros((2, 2)), ("y", "x"))

    @fragment
    def white(varying):
        return varying.u

    @vertex
    def by_value(verts):
        p = verts[0]  # not a Coordinate: the subscript law holds for records too
        u = p.x  # noqa: F841
        return position(p.x, p.y)

    with pytest.raises(TypeError, match="ONE Coordinate"):
        render(pair(by_value, white), tri, target=img)

    @vertex
    def wrong_field(verts):
        (vid,) = thread_idx("vertex_id")
        u = verts[vid].z  # noqa: F841
        return position(u, u)

    with pytest.raises(AttributeError, match="no field 'z'"):
        render(pair(wrong_field, white), tri, target=img)

    nested = Tensor.from_numpy(np.zeros(3, dtype=np.dtype([("p", [("x", "<f8")])])), ("vertex_id",))

    @vertex
    def uses(verts):
        (vid,) = thread_idx("vertex_id")
        u = verts[vid].p  # noqa: F841
        return position(u, u)

    with pytest.raises(TypeError, match="recorded boundary"):
        render(pair(uses, white), nested, target=img)
