"""The base-language rule pack: the accepted Python subset, as data.

This is a SATELLITE. Each entry maps an ``ast`` node type to a lowering rule;
widening the language means adding an entry here (or in another dialect's
pack) — never editing the kernel driver. The pack's policies are the base
dialect's, chosen deliberately:

- **Strict, like the core** (settled at the ch05 walkthrough): no automatic
  cast insertion. ``i64 + f64`` is a loud, loc-bearing error; write
  ``float(i)``. Literals are honest (``1`` is i64, ``1.0`` is f64) — in a
  float expression, write float literals. A friendlier dialect may choose to
  auto-insert casts in ITS pack; this one does not.
- ``float(x)`` / ``int(x)`` / ``bool(x)`` are the base cast vocabulary,
  lowering to explicit ``core.cast`` (visible in IR, inside the content key).
- Calls resolve: cast builtins -> captured-Handle inlining -> loud
  MissingRule (the overload surface arrives with the Registry, step 8).
"""

from __future__ import annotations

import ast

from ..kernel.lower import MissingRule, fmt
from ..kernel.types import boolean, f64, i64
from ..kernel.valuekind import typeof

_BINOPS = {ast.Add: "core.add", ast.Sub: "core.sub", ast.Mult: "core.mul", ast.Div: "core.div",
           ast.Mod: "core.mod", ast.Pow: "core.pow"}
_PREDS = {ast.Lt: "lt", ast.Gt: "gt", ast.LtE: "le", ast.GtE: "ge", ast.Eq: "eq", ast.NotEq: "ne"}
_CASTS = {"float": f64, "int": i64, "bool": boolean}


def _expr_stmt(ctx, node):
    if isinstance(node.value, ast.Constant):
        return None  # docstrings and stray constants vanish
    raise MissingRule(f"expression statements have no effect here [{fmt(ctx.loc(node))}]")


def _assign(ctx, node):
    if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
        raise MissingRule(f"only single-name assignment is in the base pack [{fmt(ctx.loc(node))}]")
    ctx.locals[node.targets[0].id] = ctx.lower(node.value)
    return None


def _return(ctx, node):
    if node.value is None:
        raise MissingRule(f"bare return is not a value [{fmt(ctx.loc(node))}]")
    return ctx.lower(node.value)


def _constant(ctx, node):
    if isinstance(node.value, (bool, int, float)):
        return ctx.emit("core.const", node=node, type=typeof(node.value), value=node.value)
    raise MissingRule(f"literal {node.value!r} has no base-pack lowering [{fmt(ctx.loc(node))}]")


def _name(ctx, node):
    resolved = ctx.resolve(node.id, node)
    if isinstance(resolved, tuple) and resolved[0] == "callee":
        raise MissingRule(f"a captured kernel is callable, not a value (yet) [{fmt(ctx.loc(node))}]")
    return resolved


def _binop(ctx, node):
    op = _BINOPS.get(type(node.op))
    if op is None:
        raise MissingRule(f"operator {type(node.op).__name__} is not in the base pack [{fmt(ctx.loc(node))}]")
    return ctx.emit(op, ctx.lower(node.left), ctx.lower(node.right), node=node)


def _unary(ctx, node):
    if isinstance(node.op, ast.USub):
        return ctx.emit("core.neg", ctx.lower(node.operand), node=node)
    raise MissingRule(f"unary {type(node.op).__name__} is not in the base pack [{fmt(ctx.loc(node))}]")


def _compare(ctx, node):
    if len(node.ops) != 1:
        raise MissingRule(f"chained comparison is not in the base pack [{fmt(ctx.loc(node))}]")
    pred = _PREDS.get(type(node.ops[0]))
    if pred is None:
        raise MissingRule(f"comparison {type(node.ops[0]).__name__} is not in the base pack [{fmt(ctx.loc(node))}]")
    return ctx.emit("core.cmp", ctx.lower(node.left), ctx.lower(node.comparators[0]), node=node, pred=pred)


def _ifexp(ctx, node):
    from ..kernel.ir import Region

    cond = ctx.lower(node.test)
    then = Region(body=(ctx.emit("core.yield", ctx.lower(node.body), node=node.body),))
    other = Region(body=(ctx.emit("core.yield", ctx.lower(node.orelse), node=node.orelse),))
    return ctx.emit("core.if", cond, regions=(then, other), node=node)


def _call(ctx, node):
    if not isinstance(node.func, ast.Name) or node.keywords:
        raise MissingRule(f"only simple positional calls are in the base pack [{fmt(ctx.loc(node))}]")
    args = tuple(ctx.lower(a) for a in node.args)
    name = node.func.id
    if name in _CASTS and len(args) == 1:
        return ctx.emit("core.cast", args[0], node=node, to=_CASTS[name])
    from ..kernel.lower import NameFateError

    resolved = None
    try:
        resolved = ctx.resolve(name, node)
    except NameFateError:
        pass  # unknown name: fall through to the richer cannot-call message
    if isinstance(resolved, tuple) and resolved[0] == "callee":
        return ctx.inline(resolved[1], resolved[2], args, node)
    raise MissingRule(
        f"cannot call {name!r}: not a cast, not a captured kernel; overloads arrive with the "
        f"Registry (surface B) [{fmt(ctx.loc(node))}]"
    )


LOWER_RULES = {
    ast.Expr: _expr_stmt,
    ast.Assign: _assign,
    ast.Return: _return,
    ast.Constant: _constant,
    ast.Name: _name,
    ast.BinOp: _binop,
    ast.UnaryOp: _unary,
    ast.Compare: _compare,
    ast.IfExp: _ifexp,
    ast.Call: _call,
}
