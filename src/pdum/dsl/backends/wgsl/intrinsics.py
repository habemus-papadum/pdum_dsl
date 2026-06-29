"""Mapping from DSL intrinsics/builtins to WGSL, and the supported builtin-call set."""

from __future__ import annotations

from ...types import FloatType, Type, VecType, f32

# builtins.<X> attribute name  ->  IR Intrinsic name
INTRINSIC_NAMES = {
    "FragCoord": "frag_coord",
    "resolution": "resolution",
    "time": "time",
}

# IR Intrinsic name -> (WGSL expression, type)
INTRINSIC_WGSL = {
    "frag_coord": ("fragcoord", VecType(f32, 4)),  # @builtin(position) param of fs_main
}


# builtin function name -> (WGSL name, result-type rule)
# result-type rule: a callable (arg_types) -> Type
def _first(arg_types: list[Type]) -> Type:
    return arg_types[0]


def _scalar_f32(_arg_types: list[Type]) -> Type:
    return f32


BUILTIN_CALLS = {
    "sqrt": ("sqrt", _scalar_f32),
    "abs": ("abs", _first),
    "floor": ("floor", _first),
    "fract": ("fract", _first),
    "sin": ("sin", _scalar_f32),
    "cos": ("cos", _scalar_f32),
    "min": ("min", _first),
    "max": ("max", _first),
    "length": ("length", _scalar_f32),  # vec -> f32
    "mix": ("mix", _first),
    "clamp": ("clamp", _first),
}


def is_float(t: Type) -> bool:
    return isinstance(t, FloatType) or (isinstance(t, VecType) and isinstance(t.elem, FloatType))
