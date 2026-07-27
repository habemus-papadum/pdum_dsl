"""Encodings and boundary descriptors (200 §4).

Facts at the boundary, choices in the interior, carrier semantics
throughout. Dtype is a property of buffers and encodings AT THE BOUNDARY,
recorded in descriptors, never on tensors mid-computation. Every finite bit
pattern decodes to a specific rational — exact decode — so the denotation
stays exact over exactly-known inputs: a bf16 checkpoint is a FACT the
descriptor records, not a semantic property of the program. inf/nan bit
patterns REFUSE at decode (an extended-real carrier is a recorded future
opt-in). The reference executors' float64 interior is a declared oracle
property, never semantics.

The IR cannot mint encoding-bearing values: no astype op exists, and the
one sanctioned door is the explicit exact op ``round_to(encoding)`` —
encode∘decode over the interior value, straight-through AD by default
(zero by declaration). Byte truth enters exactly twice: here, at boundary
descriptors, and at L2's encoding assignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class Encoding:
    """A declared exact decode/encode between bit patterns and values."""

    def nbytes(self, numel: int) -> int:
        raise NotImplementedError

    def decode(self, raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def encode(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def round_trip(self, values: np.ndarray) -> np.ndarray:
        """decode(encode(values)) — what ``round_to`` computes, exactly."""
        return self.decode(self.encode(values))


def _refuse_nonfinite(arr: np.ndarray, who: str) -> None:
    if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
        raise ValueError(
            f"{who}: inf/nan bit patterns refuse at decode (200 §4) — an "
            f"extended-real carrier is a recorded future opt-in; mask with "
            f"finite sentinels instead"
        )


@dataclass(frozen=True)
class NumpyEncoding(Encoding):
    """A numpy dtype as the encoding — including STRUCTURED dtypes (field
    names, offsets, padding: the memory shape of tensors-of-structs).
    Offsets/padding/alignment are encoding facts the interior never sees;
    the logical record type is the interior value type."""

    dtype: np.dtype

    def __post_init__(self):
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    def nbytes(self, numel: int) -> int:
        return numel * self.dtype.itemsize

    def decode(self, raw: np.ndarray) -> np.ndarray:
        if self.dtype.names:  # structured: decode each field, exactly
            out = np.empty(raw.shape, dtype=[(n, "<f8") for n in self.dtype.names])
            for n in self.dtype.names:
                f = np.asarray(raw[n], dtype=np.float64)
                _refuse_nonfinite(f, f"NumpyEncoding[{self.dtype}].{n}")
                out[n] = f
            return out
        out = np.asarray(raw, dtype=np.float64)
        _refuse_nonfinite(out, f"NumpyEncoding[{self.dtype}]")
        return out

    def encode(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values).astype(self.dtype)


@dataclass(frozen=True)
class QuantGroupEncoding(Encoding):
    """int4 nibbles + per-group scales over two buffer regions. Decode is
    EXACT: value = scale[g] · q — a specific rational per bit pattern."""

    group: int  # elements per scale group

    def nbytes(self, numel: int) -> int:
        groups = math.ceil(numel / self.group)
        return math.ceil(numel / 2) + 4 * groups  # nibbles + f32 scales

    def decode(self, raw) -> np.ndarray:
        q, scales = raw  # (int8-held nibble values in [-8, 7], f32 per group)
        q = np.asarray(q, dtype=np.int64)
        scales = np.asarray(scales, dtype=np.float64)
        _refuse_nonfinite(scales, "QuantGroupEncoding.scales")
        if q.min() < -8 or q.max() > 7:
            raise ValueError("QuantGroupEncoding: nibble values out of int4 range [-8, 7]")
        g = np.arange(q.size) // self.group
        return (scales[g] * q.reshape(-1)).reshape(q.shape)

    def encode(self, values: np.ndarray) -> tuple:
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        groups = math.ceil(v.size / self.group)
        pad = np.pad(v, (0, groups * self.group - v.size)).reshape(groups, self.group)
        scales = np.abs(pad).max(axis=1) / 7.0
        scales[scales == 0.0] = 1.0
        q = np.clip(np.round(pad / scales[:, None]), -8, 7).astype(np.int64)
        return q.reshape(-1)[: v.size].reshape(np.shape(values)), scales


@dataclass(frozen=True)
class FormatEncoding(Encoding):
    """A named interchange format with the transfer curve in decode —
    e.g. 8-bit unorm with the sRGB curve. Decode lands in linear light."""

    name: str  # "unorm8" | "unorm8-srgb"

    def nbytes(self, numel: int) -> int:
        return numel

    def decode(self, raw: np.ndarray) -> np.ndarray:
        u = np.asarray(raw, dtype=np.float64) / 255.0
        if self.name == "unorm8":
            return u
        if self.name == "unorm8-srgb":
            return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)
        raise ValueError(f"unknown format {self.name!r}")

    def encode(self, values: np.ndarray) -> np.ndarray:
        v = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
        if self.name == "unorm8-srgb":
            v = np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)
        return np.round(v * 255.0).astype(np.uint8)


@dataclass(frozen=True)
class Descriptor:
    """THE boundary descriptor: Buffer + Layout + Encoding (+ carrier +
    units). What provisioning holds, what cost oracles read for byte
    truth, where dictated encodings live (adopt = interop = precision
    facts, one concept)."""

    buffer: object  # Buffer (rank-1, explicit device) or the raw region(s)
    layout: object  # Layout — structural addressing
    encoding: Encoding
    carrier: str = "real"
    value_units: object = None

    @property
    def nbytes(self) -> int:
        numel = 1
        for d in self.layout.dims:
            numel *= d.size
        return self.encoding.nbytes(numel)


def adopt(raw, encoding: Encoding, names: tuple):
    """The boundary act: decode a dictated-encoding region into the
    interior (the reference's declared-f64 oracle interior). The encoding
    is a FACT recorded by the caller's descriptor; the returned tensor is
    carrier-valued and encoding-free."""
    from .tensor import Tensor

    decoded = encoding.decode(raw)
    return Tensor.from_numpy(np.asarray(decoded, dtype=np.float64), names)
