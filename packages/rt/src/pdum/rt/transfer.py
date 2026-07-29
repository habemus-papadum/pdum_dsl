"""The HOST side of a launch — what leaves a tensor, and what returns.

Every column pays these two acts identically (the Metal spike's seam
audit tagged both [SHARED] line by line), so they live once. Neither
touches a device: a tensor arrives duck-typed and leaves as bytes.

The repack was 88–100% of launch cost on every backend until the affine
law made the export one strided view (210, fixed 2026-07-28) — so
``host_f32`` is cheap now and the remaining copy is the f64→f32
narrowing itself. The adoption-era zero-copy path (page-aligned host
allocations Metal adopts outright) replaces this module's upload half,
not its discipline.
"""

from __future__ import annotations

import numpy as np


def value_names(art) -> tuple[str, ...]:
    """The launch values' names in the launcher's order — tensor params
    then requested taps — spelled the way ``art.writable`` spells them,
    so writeback can ask ``name in art.writable`` and get the truth for
    a tap buffer as well as a parameter."""
    taps = tuple(f"tap:{n}" for n in getattr(art, "requested_taps", ()))
    return tuple(art.tensor_params) + taps


def host_f32(t) -> np.ndarray:
    """A tensor's bytes as the contiguous f32 array a device buffer
    wants, in the tensor's OWN dim order — which is the order every
    generated buffer index is row-major over."""
    order = tuple(d.name for d in t.layout.dims)
    return np.ascontiguousarray(t.to_numpy(order=order), dtype=np.float32)


def writeback(t, raw: np.ndarray) -> None:
    """Device results into the tensor's own buffer, at its own strides.

    ``raw`` is f32 in the tensor's dim order (what ``host_f32`` uploaded,
    read back). This is the reference tier's store discipline without
    the reference tier: the strided ndarray IS the layout (the affine
    law), and a broadcast target refuses because a store into it has no
    single answer."""
    dims = t.layout.dims
    if any(d.stride == 0 for d in dims):
        raise ValueError("writeback: the writable target must be injective (no broadcast dims)")
    origin = t.layout.offset + sum(d.stride * d.start for d in dims)
    view = np.ndarray(
        buffer=t.buffer.data,
        dtype=t.dtype,
        shape=tuple(d.size for d in dims),
        strides=tuple(d.stride for d in dims),
        offset=origin,
    )
    view[...] = raw.reshape(view.shape)
