"""The declared device slot map: f32 carriage, exact-only integer narrowing."""

import struct

import numpy as np
import pytest

from pdum.rt.staging import device_slots, pack_device


def _staging(rows):
    """rows: (name, fmt, value) -> (slots, bytes) with packed offsets."""
    slots, buf, off = [], bytearray(), 0
    for name, fmt, v in rows:
        slots.append((name, off, fmt))
        buf += struct.pack(fmt, v)
        off += struct.calcsize(fmt)
    return slots, bytes(buf)


def test_device_rows_are_offset_ordered_and_f32():
    slots, _ = _staging([("b", "<d", 1.5), ("a", "<q", 3)])
    rows = device_slots(reversed(slots))
    assert rows == (("b", 0, "f32"), ("a", 1, "f32"))


def test_floats_and_small_ints_pack():
    slots, staging = _staging([("x", "<d", 2.25), ("n", "<q", 7), ("f", "<f", -1.0)])
    out = pack_device(slots, staging)
    assert out.dtype == np.float32
    assert list(out) == [2.25, 7.0, -1.0]


def test_integer_beyond_f32_exact_refuses_with_the_quoted_fix():
    slots, staging = _staging([("big", "<q", (1 << 24) + 1)])
    with pytest.raises(ValueError, match=r"exact integer\s+range.*cast at the capture site"):
        pack_device(slots, staging)


def test_boundary_is_exact():
    slots, staging = _staging([("edge", "<q", 1 << 24)])
    assert pack_device(slots, staging)[0] == float(1 << 24)
