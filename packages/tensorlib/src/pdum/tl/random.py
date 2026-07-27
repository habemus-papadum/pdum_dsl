"""Randomness (200 §1.8): counter-based, coordinate-indexed, closed-form fields.

``uniform(key, layout)`` is a pure function of (key, lattice coordinates) —
Philox-class bits, element *i* computed directly, no sequential state. It is
a buffer-class citizen exactly like ``iota``: ZERO memory, exact under view
ops (the layout addresses a virtual dense table; views rewrite only the
layout), free in the cost models, materialized only at a boundary that
demands it. Bits are exact — a uniform draw is the rational ``u32 / 2³²``.

Keys are ordinary values; streams derive from site paths and step indices
via ``fold_in(key, data)`` — insertion-stable and refactor-stable where
positional splitting is not. Dropout is an idiom, not an op:
``where(uniform(stream, x.layout) < p, 0, x / (1 - p))``. The recompute
theorem holds by construction: the same (key, coordinates) regenerate the
same bits, so gradients under checkpoint/revolve recompute are exact.

The reference generator is Philox2x32-10 (frozen: device lowerings must
produce bit-identical fields). Normal draws are Box–Muller over two lanes
of the counter space — real-valued, not exact rationals.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from .buffer import Buffer
from .layout import Layout
from .tensor import Tensor

_M = 0xD256D193  # Philox2x32 multiplier
_W = 0x9E3779B9  # Weyl key bump
_MASK = 0xFFFFFFFF


def _philox2x32(key: int, counter: int, rounds: int = 10) -> int:
    """Philox2x32-10: 32 bits from (32-bit key, 64-bit counter). Frozen."""
    x0, x1 = counter & _MASK, (counter >> 32) & _MASK
    k = key & _MASK
    for _ in range(rounds):
        prod = _M * x0
        x0, x1 = ((prod >> 32) ^ x1 ^ k) & _MASK, prod & _MASK
        k = (k + _W) & _MASK
    return x0


def fold_in(key: int, data) -> int:
    """Derive a stream key from a site path or step index — stable across
    processes (strings hash through sha256, never Python's salted hash)."""
    if isinstance(data, str):
        counter = int.from_bytes(hashlib.sha256(data.encode()).digest()[:8], "big")
    elif isinstance(data, int):
        counter = data & (2**64 - 1)
    else:
        raise TypeError(f"fold_in takes a path string or a step index, got {data!r}")
    return _philox2x32(key, counter)


@dataclass(frozen=True, eq=False)
class RandomBuffer(Buffer):
    """No memory: read(loc) = closed-form bits at lattice index loc//scale."""

    key: int = 0
    dist: str = "uniform"
    scale: int = 8

    def read(self, loc: int, dtype) -> object:
        q, r = divmod(loc, self.scale)
        if r:
            raise ValueError(f"misaligned random read at byte {loc} (scale {self.scale})")
        if self.dist == "uniform":
            value: object = Fraction(_philox2x32(self.key, q), 2**32)  # exact: u32 / 2^32
        else:  # normal: Box–Muller over two counter lanes; u1 offset avoids log(0)
            u1 = (_philox2x32(self.key, 2 * q) + 0.5) / 2**32
            u2 = _philox2x32(self.key, 2 * q + 1) / 2**32
            value = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        dt = np.dtype(dtype)
        return dt.type(float(value) if dt.kind == "f" else value)


def _field(dist: str, key: int, t: Tensor | Layout, dtype=None) -> Tensor:
    layout = t.layout if isinstance(t, Tensor) else t
    dt = np.dtype(dtype or np.float64)
    scale = dt.itemsize
    strides, acc = [], scale
    for d in reversed(layout.dims):
        strides.append(acc)
        acc *= max(d.size, 1)
    new_dims = tuple(replace(d, stride=s) for d, s in zip(layout.dims, reversed(strides)))
    offset = -sum(d.stride * d.start for d in new_dims)
    buf = RandomBuffer(nbytes=acc, key=int(key), dist=dist, scale=scale)
    carrier = "rat" if dist == "uniform" else "real"
    return Tensor(buf, dt, Layout(new_dims, offset), carrier=carrier).check()


def uniform(key: int, t: Tensor | Layout, dtype=None) -> Tensor:
    """The exact-rational uniform field over `t`'s lattice: u32 / 2^32."""
    return _field("uniform", key, t, dtype)


def normal(key: int, t: Tensor | Layout, dtype=None) -> Tensor:
    """The standard-normal field over `t`'s lattice (Box–Muller, real)."""
    return _field("normal", key, t, dtype)
