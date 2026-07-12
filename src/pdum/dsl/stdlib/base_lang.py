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
from ..kernel.types import Record, boolean, f64, i64
from ..kernel.valuekind import typeof

_BINOPS = {
    ast.Add: "core.add",
    ast.Sub: "core.sub",
    ast.Mult: "core.mul",
    ast.Div: "core.div",
    ast.Mod: "core.mod",
    ast.Pow: "core.pow",
}
_PREDS = {ast.Lt: "lt", ast.Gt: "gt", ast.LtE: "le", ast.GtE: "ge", ast.Eq: "eq", ast.NotEq: "ne"}
_CASTS = {"float": f64, "int": i64, "bool": boolean}


def _expr_stmt(ctx, node):
    if isinstance(node.value, ast.Constant):
        return None  # docstrings and stray constants vanish
    raise MissingRule(f"expression statements have no effect here [{fmt(ctx.loc(node))}]")


def _assign(ctx, node):
    if len(node.targets) != 1:
        raise MissingRule(f"chained assignment is not in the base pack [{fmt(ctx.loc(node))}]")
    target = node.targets[0]
    if isinstance(target, ast.Tuple) and all(isinstance(e, ast.Name) for e in target.elts):
        from ..kernel.types import Tuple as TupleType

        value = ctx.lower(node.value)  # ONE evaluation; swap works because it reads old locals first
        if not isinstance(value.type, TupleType) or len(value.type.elems) != len(target.elts):
            raise MissingRule(f"cannot unpack {value.type!r} into {len(target.elts)} names [{fmt(ctx.loc(node))}]")
        for i, elt in enumerate(target.elts):
            ctx.locals[elt.id] = ctx.emit("core.extract", value, node=node, index=i)
        return None
    if not isinstance(target, ast.Name):
        raise MissingRule(f"assignment targets are names or name-tuples in the base pack [{fmt(ctx.loc(node))}]")
    ctx.locals[target.id] = ctx.lower(node.value)
    return None


def _augassign(ctx, node):
    """``x += e`` is pure sugar for ``x = x + e`` — same STRICT typing."""
    op = _BINOPS.get(type(node.op))
    if not isinstance(node.target, ast.Name) or op is None:
        raise MissingRule(f"aug-assign supports named targets and + - * / % ** [{fmt(ctx.loc(node))}]")
    current = ctx.resolve(node.target.id, node)
    if isinstance(current, tuple):  # a captured kernel is callable, never arithmetic
        raise MissingRule(f"{node.target.id!r} is a captured kernel, not a value [{fmt(ctx.loc(node))}]")
    ctx.locals[node.target.id] = ctx.emit(op, current, ctx.lower(node.value), node=node)
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


def _bool_const(ctx, node, value):
    return ctx.emit("core.const", node=node, type=boolean, value=value)


def _shortcircuit(ctx, node, is_and, left, right):
    """``a and b`` = if(a){b}{false}; ``a or b`` = if(a){true}{b}. The
    dominator-placed renderers keep the untaken side LAZY — real
    short-circuit, no new mechanism (the base pack's stated policy: 070)."""
    from ..kernel.ir import Region

    if left.type != boolean or right.type != boolean:
        raise MissingRule(f"strict and/or want bool operands; truthiness is a dialect policy [{fmt(ctx.loc(node))}]")
    branch = ctx.emit("core.yield", right, node=node)
    const = ctx.emit("core.yield", _bool_const(ctx, node, not is_and), node=node)
    taken, other = Region(body=(branch,)), Region(body=(const,))
    regions = (taken, other) if is_and else (other, taken)
    return ctx.emit("core.if", left, regions=regions, node=node)


def _boolop(ctx, node):
    result = ctx.lower(node.values[0])
    for value in node.values[1:]:
        result = _shortcircuit(ctx, node, isinstance(node.op, ast.And), result, ctx.lower(value))
    return result


def _compare(ctx, node):
    """Single AND CHAINED comparisons; ``a < b < c`` evaluates ``b`` once
    (the operand NODE is shared) and folds with the and-circuit."""
    left, result = ctx.lower(node.left), None
    for op, comp in zip(node.ops, node.comparators):
        pred = _PREDS.get(type(op))
        if pred is None:
            raise MissingRule(f"comparison {type(op).__name__} is not in the base pack [{fmt(ctx.loc(node))}]")
        right = ctx.lower(comp)
        pair = ctx.emit("core.cmp", left, right, node=node, pred=pred)
        result = pair if result is None else _shortcircuit(ctx, node, True, result, pair)
        left = right
    return result


def _tuple_expr(ctx, node):
    return ctx.emit("core.tuple", *(ctx.lower(e) for e in node.elts), node=node)


def _subscript(ctx, node):
    """Constant-index access into tuples (and Records by position never —
    fields are names). The ch08 stand-in, promoted into the pack (ch07a: 🔧)."""
    base = ctx.lower(node.value)
    if not (isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int)):
        raise MissingRule(f"only constant integer indices are in the base pack [{fmt(ctx.loc(node))}]")
    return ctx.emit("core.extract", base, node=node, index=node.slice.value)


def _attribute(ctx, node):
    base = ctx.lower(node.value)
    if isinstance(base.type, Record) and node.attr in dict(base.type.fields):
        return ctx.emit("core.field", base, node=node, name=node.attr)
    raise MissingRule(f"attribute {node.attr!r} needs a Record-typed value [{fmt(ctx.loc(node))}]")


def _ifexp(ctx, node):
    from ..kernel.ir import Region

    cond = ctx.lower(node.test)
    then = Region(body=(ctx.emit("core.yield", ctx.lower(node.body), node=node.body),))
    other = Region(body=(ctx.emit("core.yield", ctx.lower(node.orelse), node=node.orelse),))
    return ctx.emit("core.if", cond, regions=(then, other), node=node)


def _call(ctx, node):
    if node.keywords:
        raise MissingRule(f"keyword arguments are not in the base pack [{fmt(ctx.loc(node))}]")
    registry = ctx.rules.get(
        "__registry__"
    )  # planted by Registry._build per build; string keys never collide with ast types
    args = tuple(ctx.lower(a) for a in node.args)
    if isinstance(node.func, ast.Attribute):  # method call: Record overload_method (surface B)
        base = ctx.lower(node.func.value)
        impl = registry and isinstance(base.type, Record) and registry.overloads.get((base.type.name, node.func.attr))
        if impl:
            return ctx.inline(impl, 0, (base, *args), node)
        raise MissingRule(f"no method {node.func.attr!r} registered for {base.type!r} [{fmt(ctx.loc(node))}]")
    if not isinstance(node.func, ast.Name):
        raise MissingRule(f"only name and method calls are in the base pack [{fmt(ctx.loc(node))}]")
    name = node.func.id
    if name in _CASTS and len(args) == 1:
        return ctx.emit("core.cast", args[0], node=node, to=_CASTS[name])
    from ..kernel.lower import NameFateError

    resolved = None
    try:
        resolved = ctx.resolve(name, node)
    except NameFateError:
        pass  # unknown name: overloads below, then the loud message
    if isinstance(resolved, tuple) and resolved[0] == "callee":
        return ctx.inline(resolved[1], resolved[2], args, node)
    if resolved is not None:  # a local/captured VALUE shadowing a battery name: python would
        raise MissingRule(f"{name!r} is a value here, not callable [{fmt(ctx.loc(node))}]")  # raise; so do we
    impl = registry.overloads.get(name) if registry else None
    if isinstance(impl, str):  # an intrinsic: the name IS an op (spelling per target)
        return ctx.emit(impl, *args, node=node)
    if impl is not None:  # a DSL-written battery: capture-free, inlined like a callee
        return ctx.inline(impl, 0, args, node)
    raise MissingRule(
        f"cannot call {name!r}: not a cast, capture, or registered overload (surface B) [{fmt(ctx.loc(node))}]"
    )


LOWER_RULES = {
    ast.Expr: _expr_stmt,
    ast.Assign: _assign,
    ast.AugAssign: _augassign,
    ast.Return: _return,
    ast.Constant: _constant,
    ast.Name: _name,
    ast.BinOp: _binop,
    ast.UnaryOp: _unary,
    ast.BoolOp: _boolop,
    ast.Compare: _compare,
    ast.IfExp: _ifexp,
    ast.Tuple: _tuple_expr,
    ast.Subscript: _subscript,
    ast.Attribute: _attribute,
    ast.Call: _call,
}
