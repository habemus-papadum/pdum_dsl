"""Emit WGSL text from typed IR. A pure ``IR -> str`` function — no GPU — so the
whole frontend/middle-end can be unit-tested without a device."""

from __future__ import annotations

from ... import ir
from ...types import FloatType, IntType, Type, VecType
from .intrinsics import BUILTIN_CALLS, INTRINSIC_WGSL
from .layout import Layout, wgsl_type_name

# Full-screen triangle: 3 clip-space verts cover the framebuffer (the standard
# ShaderToy trick). The fragment shader gets pixel coords via @builtin(position).
VS_FULLSCREEN = """\
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0,  3.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0, -1.0),
  );
  return vec4<f32>(p[vi], 0.0, 1.0);
}"""


def emit_module(fn: ir.Function, layout: Layout) -> str:
    parts = [
        layout.struct_wgsl("Uniforms"),
        "@group(0) @binding(0) var<uniform> u: Uniforms;",
        VS_FULLSCREEN,
        _emit_fragment(fn),
    ]
    return "\n\n".join(parts) + "\n"


def _emit_fragment(fn: ir.Function) -> str:
    lines = [
        "@fragment",
        "fn fs_main(@builtin(position) fragcoord: vec4<f32>) -> @location(0) vec4<f32> {",
    ]
    for stmt in fn.body:
        if isinstance(stmt, ir.Let):
            lines.append(f"  let {stmt.name} = {_emit(stmt.value)};")
        elif isinstance(stmt, ir.Return):
            lines.append(f"  return {_emit_color(stmt.value)};")
    lines.append("}")
    return "\n".join(lines)


def _emit_color(node: ir.Node) -> str:
    """Coerce a returned value to the fragment output ``vec4<f32>``."""
    expr = _emit(node)
    t = node.type
    if isinstance(t, VecType):
        if t.n == 4:
            return expr
        if t.n == 3:
            return f"vec4<f32>({expr}, 1.0)"
        if t.n == 2:
            return f"vec4<f32>({expr}, 0.0, 1.0)"
    # scalar -> grayscale
    g = f"f32({expr})" if isinstance(t, IntType) else expr
    return f"vec4<f32>(vec3<f32>({g}), 1.0)"


def _fmt_float(v: float) -> str:
    r = repr(float(v))
    # repr always yields a '.' or 'e' for finite floats, which WGSL accepts.
    return r


def _is_floatish(t: Type | None) -> bool:
    return isinstance(t, FloatType) or (isinstance(t, VecType) and isinstance(t.elem, FloatType))


def _emit_f(node: ir.Node) -> str:
    """Emit a numeric node, widening an integer to ``f32`` (WGSL has no implicit
    int→float conversion)."""
    s = _emit(node)
    return f"f32({s})" if isinstance(node.type, IntType) else s


def _emit(node: ir.Node) -> str:
    if isinstance(node, ir.Lit):
        v = node.value
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, float):
            return _fmt_float(v)
        return str(v)

    if isinstance(node, ir.Name):
        return f"u.{node.name}" if node.scope == "uniform" else node.name

    if isinstance(node, ir.Intrinsic):
        return INTRINSIC_WGSL[node.name][0]

    if isinstance(node, ir.Swizzle):
        return f"{_emit(node.base)}.{node.comps}"

    if isinstance(node, ir.Unary):
        return f"({node.op}{_emit(node.operand)})"

    if isinstance(node, ir.BinOp):
        if node.op == "**":  # WGSL pow operates on floats
            return f"pow({_emit_f(node.left)}, {_emit_f(node.right)})"
        if _is_floatish(node.type):  # widen an int operand to match a float result
            return f"({_emit_f(node.left)} {node.op} {_emit_f(node.right)})"
        return f"({_emit(node.left)} {node.op} {_emit(node.right)})"

    if isinstance(node, ir.Compare):
        if _is_floatish(node.left.type) or _is_floatish(node.right.type):
            return f"({_emit_f(node.left)} {node.op} {_emit_f(node.right)})"
        return f"({_emit(node.left)} {node.op} {_emit(node.right)})"

    if isinstance(node, ir.Select):
        # WGSL select(false_value, true_value, condition)
        return f"select({_emit(node.if_false)}, {_emit(node.if_true)}, {_emit(node.cond)})"

    if isinstance(node, ir.MakeVec):
        elems = ", ".join(_emit_f(e) for e in node.elems)  # vecN<f32> wants floats
        return f"vec{len(node.elems)}<f32>({elems})"

    if isinstance(node, ir.Call):
        wname, _ = BUILTIN_CALLS[node.func]
        args = ", ".join(_emit(a) for a in node.args)
        return f"{wname}({args})"

    raise TypeError(f"cannot emit {node!r}")


def wgsl_scalar_or_vec(ty: Type) -> str:  # convenience re-export
    return wgsl_type_name(ty)


_ = (FloatType, IntType)  # referenced for type narrowing parity / future use
