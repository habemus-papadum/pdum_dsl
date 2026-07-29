"""The staging plan's DEVICE representation — declared, never invented.

The host staging plan (``abi.slot``: offset + struct fmt per captured
scalar) says nothing about the device side, so every backend to date
independently invented "uniforms arrive as a flat f32 array" and
silently narrowed i64 slots on the way (three inventions — 210,
FAIL-4). This module is the single answer both packers and generators
read.

v1 device map (one representation, DECLARED):

- f32/f64 slots ride f32 — the declared narrowing of the numeric
  policy (f32 computes as f64 on the reference; per-device narrowing
  is declared, never silent). Tolerances are the differential's
  affair.
- i32/u32/bool ride f32 exactly (|v| ≤ 2^24 always holds for i32?
  no — i32 exceeds 2^24; the exactness REFUSAL below applies to every
  integer carrier).
- INTEGER slots whose value lies outside f32's exact range (±2^24)
  REFUSE at pack time — that is the 210 rule made real. The fix is
  quoted: cast at the capture site, or keep the value in range. A
  typed device slot map (i32 slots as real i32) is the recorded next
  rung, taken when a consumer needs the range.
"""

from __future__ import annotations

import struct

import numpy as np

_EXACT = 1 << 24  # f32's exact integer range

_INT_KINDS = "iqIQnN?"  # struct integer/bool format characters


def device_slots(slots) -> tuple:
    """The declared device rows for a host slot list.

    ``slots`` is an iterable of ``(name, offset, fmt)`` (the host plan's
    spelling). Returns ``(name, index, carrier)`` rows in offset order —
    the device buffer is index-addressed, hole-free, f32-carried (v1).
    """
    ordered = sorted(slots, key=lambda s: s[1])
    return tuple((name, i, "f32") for i, (name, _off, _fmt) in enumerate(ordered))


def pack_device(slots, staging: bytes) -> np.ndarray:
    """Host staging bytes → the device slot buffer (f32, v1).

    Integer slots refuse outside f32's exact range — declared
    narrowing, never silent (210)."""
    ordered = sorted(slots, key=lambda s: s[1])
    out = np.empty(len(ordered), dtype=np.float32)
    for i, (name, off, fmt) in enumerate(ordered):
        (v,) = struct.unpack_from(fmt, staging, off)
        if fmt[-1] in _INT_KINDS and not isinstance(v, bool) and abs(int(v)) > _EXACT:
            raise ValueError(
                f"slot {name!r} (fmt {fmt!r}) carries {v}, outside f32's exact integer "
                f"range (±2**24) — the v1 device slot map declares integer→f32 narrowing "
                f"as exact-only (210): cast at the capture site, or keep the value in "
                f"range; a typed i32 slot map is the recorded next rung"
            )
        out[i] = float(v)
    return out
