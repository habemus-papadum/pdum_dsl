"""Uniform-buffer layout: turn a closure's ``env_types`` into a WGSL uniform
struct + a byte packer, honoring the WGSL uniform address-space alignment rules.

These rules are the classic footgun (vec3 has size 12 but **align 16**), so this
module is small and heavily unit-tested. Reference: WGSL spec §14.4.x "Memory
Layout" / WebGPU uniform layout constraints.

    type        size  align
    f32/i32/u32  4     4
    vec2<T>      8     8
    vec3<T>     12    16     <- content 12, padded to 16
    vec4<T>     16    16
    struct       -    max(member align); total roundUp to 16 for var<uniform>
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from ...types import BoolType, FloatType, IntType, Type, VecType


def narrow_type(ty: Type) -> Type:
    """Narrow a DSL type to its WGSL width (WGSL uniforms are 32-bit scalars)."""
    if isinstance(ty, FloatType):
        return FloatType(32)
    if isinstance(ty, IntType):
        return IntType(32, ty.signed)
    if isinstance(ty, BoolType):
        return ty
    if isinstance(ty, VecType):
        return VecType(narrow_type(ty.elem), ty.n)
    raise TypeError(f"type {ty!r} is not representable in a uniform buffer")


def _scalar_name(ty: Type) -> str:
    if isinstance(ty, FloatType):
        return "f32"
    if isinstance(ty, IntType):
        return "i32" if ty.signed else "u32"
    if isinstance(ty, BoolType):
        return "u32"  # bool is not host-shareable; stored as u32
    raise TypeError(f"not a scalar: {ty!r}")


def wgsl_type_name(ty: Type) -> str:
    """WGSL type spelling for emission (e.g. ``vec3<f32>``)."""
    if isinstance(ty, VecType):
        return f"vec{ty.n}<{_scalar_name(ty.elem)}>"
    if isinstance(ty, BoolType):
        return "bool"  # in expressions bool is fine; only uniform *storage* is u32
    return _scalar_name(ty)


def align_of(ty: Type) -> int:
    if isinstance(ty, VecType):
        return 8 if ty.n == 2 else 16
    return 4


def size_of(ty: Type) -> int:
    if isinstance(ty, VecType):
        return {2: 8, 3: 12, 4: 16}[ty.n]
    return 4


def _round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


_PACK_FMT = {"f32": "<f", "i32": "<i", "u32": "<I"}


@dataclass
class Field:
    name: str
    dsl_type: Type  # narrowed
    offset: int
    size: int
    align: int

    @property
    def wgsl(self) -> str:
        return wgsl_type_name(self.dsl_type)


@dataclass
class Layout:
    fields: list[Field]
    size: int  # total buffer size (>= 16, multiple of 16)

    def by_name(self) -> dict[str, Field]:
        return {f.name: f for f in self.fields}

    def uniform_types(self) -> dict[str, Type]:
        """Name → narrowed DSL type, for the inference pass."""
        return {f.name: f.dsl_type for f in self.fields}

    def struct_wgsl(self, name: str = "Uniforms") -> str:
        if not self.fields:
            # WGSL forbids empty structs; give a padding member.
            return f"struct {name} {{\n  _pad: vec4<f32>,\n}};"
        lines = [f"  {f.name}: {f.wgsl}," for f in self.fields]
        return f"struct {name} {{\n" + "\n".join(lines) + "\n};"

    def pack(self, env: dict[str, object]) -> bytes:
        """Pack captured values into the uniform buffer bytes (per offsets)."""
        buf = bytearray(self.size)
        for f in self.fields:
            value = env[f.name]
            scalar = _scalar_name(f.dsl_type if not isinstance(f.dsl_type, VecType) else f.dsl_type.elem)
            fmt = _PACK_FMT[scalar]
            if isinstance(f.dsl_type, VecType):
                comps = tuple(value)
                if len(comps) != f.dsl_type.n:
                    raise ValueError(f"{f.name}: expected vec{f.dsl_type.n}, got {len(comps)} comps")
                for i, c in enumerate(comps):
                    struct.pack_into(fmt, buf, f.offset + i * 4, _coerce(scalar, c))
            else:
                struct.pack_into(fmt, buf, f.offset, _coerce(scalar, value))
        return bytes(buf)


def _coerce(scalar: str, v: object) -> object:
    if scalar == "f32":
        return float(v)
    return int(v)


def build_layout(names: tuple[str, ...], types: tuple[Type, ...]) -> Layout:
    """Build a uniform layout from free-variable names + their (DSL) types,
    in ``co_freevars`` order."""
    fields: list[Field] = []
    offset = 0
    for name, ty in zip(names, types):
        nt = narrow_type(ty)
        a = align_of(nt)
        s = size_of(nt)
        offset = _round_up(offset, a)
        fields.append(Field(name=name, dsl_type=nt, offset=offset, size=s, align=a))
        offset += s
    total = _round_up(offset, 16) if offset else 16
    return Layout(fields=fields, size=total)
