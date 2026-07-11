"""Type inference over the IR: fills every node's ``type`` given the uniform
(env) types and any argument types. WGSL is statically typed, so emission needs a
type on every subexpression.

The lattice handled here is the M0 subset: f32/i32/bool scalars and float vectors,
with scalar↔vector promotion in binary ops. It is intentionally small and grows
with the language.
"""

from __future__ import annotations

from .. import ir
from ..backends.wgsl.intrinsics import BUILTIN_CALLS, INTRINSIC_WGSL
from ..types import BoolType, FloatType, IntType, Type, VecType, boolean, f32, i32


def infer_function(
    fn: ir.Function,
    uniform_types: dict[str, Type],
    arg_types: dict[str, Type] | None = None,
) -> ir.Function:
    arg_types = arg_types or {}
    locals_: dict[str, Type] = {}

    def ty(node: ir.Node) -> Type:
        t = _infer(node, uniform_types, arg_types, locals_, ty)
        node.type = t
        return t

    for stmt in fn.body:
        if isinstance(stmt, ir.Let):
            t = ty(stmt.value)
            stmt.type = t
            locals_[stmt.name] = t
        elif isinstance(stmt, ir.Return):
            t = ty(stmt.value)
            stmt.type = t
            fn.ret_type = t
        else:
            raise TypeError(f"unexpected statement in body: {stmt!r}")

    fn.locals = locals_
    return fn


def _promote(a: Type, b: Type) -> Type:
    if isinstance(a, VecType):
        return a
    if isinstance(b, VecType):
        return b
    if isinstance(a, FloatType) or isinstance(b, FloatType):
        return f32
    return i32


def _infer(node, uniform_types, arg_types, locals_, ty) -> Type:
    if isinstance(node, ir.Lit):
        v = node.value
        if isinstance(v, bool):
            return boolean
        if isinstance(v, int):
            return i32
        if isinstance(v, float):
            return f32
        raise TypeError(f"unsupported literal {v!r}")

    if isinstance(node, ir.Name):
        if node.scope == "uniform":
            return uniform_types[node.name]
        if node.scope == "arg":
            return arg_types[node.name]
        return locals_[node.name]

    if isinstance(node, ir.Intrinsic):
        return INTRINSIC_WGSL[node.name][1]

    if isinstance(node, ir.Swizzle):
        base = ty(node.base)
        elem = base.elem if isinstance(base, VecType) else base
        n = len(node.comps)
        return elem if n == 1 else VecType(elem, n)

    if isinstance(node, ir.Unary):
        return ty(node.operand)

    if isinstance(node, ir.BinOp):
        return _promote(ty(node.left), ty(node.right))

    if isinstance(node, ir.Compare):
        ty(node.left)
        ty(node.right)
        return boolean

    if isinstance(node, ir.Select):
        ty(node.cond)
        t_true = ty(node.if_true)
        ty(node.if_false)
        return t_true

    if isinstance(node, ir.MakeVec):
        for el in node.elems:
            ty(el)
        return VecType(f32, len(node.elems))  # M0: vectors are float

    if isinstance(node, ir.Call):
        arg_t = [ty(a) for a in node.args]
        _, rule = BUILTIN_CALLS[node.func]
        return rule(arg_t)

    raise TypeError(f"cannot infer type of {node!r}")


# Re-exported for callers that want to recognize bool/int specially.
__all__ = ["infer_function", "BoolType", "IntType"]
