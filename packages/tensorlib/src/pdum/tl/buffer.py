"""Buffer: rank-1, featureless storage — plus the read seam.

A Buffer is a pointer-like handle plus an address space and a length in
bytes. It has no shape, no strides, no dtype — all of that lives in the
Layout/Tensor layer. DLPack would be the interchange format for real device
buffers; host buffers carry a memoryview so the layout math can be
*exercised* in tests; device buffers are just described (data=None).

`read(loc, dtype)` is the single seam through which values leave a buffer.
A tensor is the composition value(coords) = buffer.read(loc(coords)) — the
layout owns the affine part, the buffer owns the dereference. That
factorization is what makes FunctionalBuffer possible:

`FunctionalBuffer` owns NO memory and declares its read as an exact affine
function of the byte location: read(loc) = const + coeff * (loc // scale),
with rational coefficients, cast to the requesting dtype only at the read.
It is the tight form of `iota` (the identity table, declared instead of
materialized) — and the closure invariant comes free: every view op rewrites
only the layout, never the buffer, and affine ∘ affine = affine, so
iota-ness cannot be destroyed by any layout operation. A compiler recognizes
the closed form by the buffer's type.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass(frozen=True, eq=False)
class Buffer:
    nbytes: int
    device: str = "cpu"
    data: memoryview | None = None  # host bytes when accessible, else None

    def __post_init__(self) -> None:
        if self.nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        if self.data is not None and len(self.data) != self.nbytes:
            raise ValueError(f"data length {len(self.data)} != nbytes {self.nbytes}")

    @classmethod
    def from_bytes(cls, raw, device: str = "cpu") -> "Buffer":
        mv = memoryview(raw).cast("B")
        return cls(nbytes=len(mv), device=device, data=mv)

    @classmethod
    def allocate(cls, nbytes: int, device: str = "cpu") -> "Buffer":
        return cls.from_bytes(bytearray(nbytes), device=device)

    @property
    def is_host(self) -> bool:
        return self.data is not None

    def read(self, loc: int, dtype) -> object:
        """The read seam: one value of `dtype` at byte `loc`."""
        if self.data is None:
            raise RuntimeError(f"buffer on {self.device} is not host-readable")
        return np.frombuffer(self.data, dtype, count=1, offset=loc)[0]

    def __repr__(self) -> str:
        return f"Buffer({self.nbytes}B @ {self.device})"


def host_view(carr) -> memoryview:
    """Best-effort zero-copy byte view of a C-contiguous array; falls back
    to a copy when the buffer protocol refuses the cast (e.g. some
    structured dtypes). The single home of this fallback."""
    try:
        return memoryview(carr).cast("B")
    except ValueError, TypeError:
        return memoryview(bytearray(carr.tobytes()))


@dataclass(frozen=True, eq=False)
class FunctionalBuffer(Buffer):
    """No memory: read(loc) = const + coeff * (loc // scale), exactly.

    Coefficients are exact rationals; the cast to the requesting dtype
    happens only at the read — representation never enters the semantics.
    `nbytes` is the extent the buffer *would* occupy if materialized, so
    footprint/check semantics are identical to the tabulated form."""

    scale: int = 1  # bytes per lattice step (the itemsize of the virtual table)
    coeff: Fraction = Fraction(1)
    const: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.scale <= 0:
            raise ValueError("scale must be a positive byte count")
        object.__setattr__(self, "coeff", Fraction(self.coeff))
        object.__setattr__(self, "const", Fraction(self.const))

    def read(self, loc: int, dtype) -> object:
        q, r = divmod(loc, self.scale)
        if r:
            raise ValueError(f"misaligned functional read at byte {loc} (scale {self.scale})")
        value = self.const + self.coeff * q
        dt = np.dtype(dtype)
        if dt.kind in "iub":
            if value.denominator != 1:
                raise ValueError(f"non-integer value {value} read as {dt}")
            return dt.type(int(value))
        return dt.type(float(value))

    def __repr__(self) -> str:
        return f"FunctionalBuffer({self.nbytes}B, read = {self.const} + {self.coeff}*(loc//{self.scale}))"
