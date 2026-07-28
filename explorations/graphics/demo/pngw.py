"""A minimal PNG writer — zlib + struct, no new dependencies.

The venv has no Pillow / imageio / pypng (checked), and the brief says don't
add heavy deps. 8-bit grayscale, filter type 0, one IDAT.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)


def write_gray(path, img) -> None:
    """``img``: float array in [0, 1] (clipped), or uint8. Row 0 is the TOP
    row of the written image."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(np.nan_to_num(a.astype(np.float64), nan=0.0), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    h, w = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )
    with open(path, "wb") as fh:
        fh.write(png)
