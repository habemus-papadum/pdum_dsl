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


def _lower_block(ctx, stmts):
    """A statement suite inside a branch or loop. The base pack's SINGLE TAIL
    RETURN policy is enforced here: only the function body's last statement
    may return, so any statement that produces a value inside a suite (i.e.
    ``return``) is refused with the policy named."""
    for stmt in stmts:
        if ctx.lower(stmt) is not None:
            raise MissingRule(
                f"single tail return: `return` must be the function body's last statement, "
                f"not inside a branch or loop (core.yield IS the return) [{fmt(ctx.loc(stmt))}]"
            )


def _branch_locals(ctx, stmts, before):
    """Lower a suite against a COPY of the locals; hand back what it bound."""
    ctx.locals = dict(before)
    _lower_block(ctx, stmts)
    return ctx.locals


def _if_stmt(ctx, node):
    """Statement `if`: both suites lower against copies of the locals, then
    the rebound names JOIN — strictly. A name assigned in either suite must
    have the SAME type on both paths (no unification, no fixpoint); a name
    born in only one suite is refused (it may be undefined after the if)."""
    from ..kernel.ir import Region

    cond = ctx.lower(node.test)
    if cond.type != boolean:
        raise MissingRule(
            f"if condition must be bool, got {cond.type!r}; truthiness is a dialect policy [{fmt(ctx.loc(node))}]"
        )
    before = dict(ctx.locals)
    then_l = _branch_locals(ctx, node.body, before)
    else_l = _branch_locals(ctx, node.orelse, before)
    ctx.locals = before
    # Names born in only ONE suite are branch-local temporaries: they DIE with
    # their suite (like loop-locals die with the loop) — a later read is a loud
    # NameFateError, so "undefined afterwards" is impossible by construction.
    # (The first draft refused them outright, which outlawed innocent scratch
    # variables — review-caught.) The join covers names both paths can supply:
    changed = [
        n
        for n in dict.fromkeys((*then_l, *else_l))
        if n in then_l and n in else_l and (then_l.get(n) is not before.get(n) or else_l.get(n) is not before.get(n))
    ]
    if not changed:
        raise MissingRule(f"this `if` binds nothing that survives it — dead code is refused [{fmt(ctx.loc(node))}]")
    for n in changed:
        if then_l[n].type != else_l[n].type:
            raise TypeError(
                f"strict join: {n!r} is {then_l[n].type!r} on the then-path but {else_l[n].type!r} "
                f"on the else-path [{fmt(ctx.loc(node))}]"
            )

    def joined(branch_l):  # ONE yielded value per region (the walker/renderer contract):
        vals = [branch_l[n] for n in changed]  # multi-name joins ride a literal core.tuple
        return vals[0] if len(vals) == 1 else ctx.emit("core.tuple", *vals, node=node)

    then_y = ctx.emit("core.yield", joined(then_l), node=node)
    else_y = ctx.emit("core.yield", joined(else_l), node=node)
    res = ctx.emit("core.if", cond, regions=(Region(body=(then_y,)), Region(body=(else_y,))), node=node)
    if len(changed) == 1:
        ctx.locals[changed[0]] = res
    else:
        for k, n in enumerate(changed):
            ctx.locals[n] = ctx.emit("core.extract", res, node=node, index=k)
    return None


def _assigned_names(stmts):
    out: list = []
    for s in stmts:
        if isinstance(s, ast.Assign):
            for t in s.targets:
                elts = t.elts if isinstance(t, ast.Tuple) else (t,)
                out += [e.id for e in elts if isinstance(e, ast.Name)]
        elif isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name):
            out.append(s.target.id)
        elif isinstance(s, ast.If):
            out += _assigned_names(s.body) + _assigned_names(s.orelse)
        elif isinstance(s, ast.For):
            out += _assigned_names(s.body)
    return out


def _fresh_binder(ctx, type):
    """Region binders come from the KERNEL seam now (Lowerer.binder — the
    uniqueness/determinism invariant is kernel law, 130 §7); this shim stays
    as the pack's local name for it."""
    return ctx.binder(type)


def _for_stmt(ctx, node):
    """Statement `for i in range(lo, hi)` -> `core.for`. Loop-carried values
    are the pre-existing locals the body rebinds; names born in the body die
    with it; the loop variable dies too (Python's leak is refused as
    shadowing instead). Bounded loops only — the GPU-honest subset."""
    from ..kernel.ir import Region

    it = node.iter
    if (
        node.orelse
        or not isinstance(node.target, ast.Name)
        or not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range")
        or it.keywords
        or not 1 <= len(it.args) <= 2
    ):
        raise MissingRule(
            f"the base pack loops over `range(n)` or `range(lo, hi)` with a single name target "
            f"(no step, no iterables, no for-else) [{fmt(ctx.loc(node))}]"
        )
    bounds = [ctx.lower(a) for a in it.args]
    lo = bounds[0] if len(bounds) == 2 else ctx.emit("core.const", node=node, type=i64, value=0)
    hi = bounds[-1]
    if lo.type != i64 or hi.type != i64:
        raise TypeError(f"range bounds are strict i64, got {lo.type!r}..{hi.type!r} [{fmt(ctx.loc(node))}]")
    if node.target.id in ctx.locals or node.target.id in ctx.handle.env:
        raise MissingRule(
            f"loop variable {node.target.id!r} shadows an existing name; the loop variable dies "
            f"with the loop here (Python's leak is not honored) [{fmt(ctx.loc(node))}]"
        )
    carried = [n for n in dict.fromkeys(_assigned_names(node.body)) if n in ctx.locals]
    if not carried:
        # Lower the body ANYWAY (against a scratch scope) before refusing: a
        # body whose real problem is an unsupported construct (`while`, ...)
        # must surface THAT refusal, not a misleading carry complaint
        # (review-caught: the prescan cannot see into unruled constructs).
        scratch = dict(ctx.locals)
        ctx.locals[node.target.id] = _fresh_binder(ctx, i64)
        _lower_block(ctx, node.body)
        ctx.locals = scratch
        raise MissingRule(f"this loop carries nothing — no pre-existing local is rebound [{fmt(ctx.loc(node))}]")
    inits = [ctx.locals[n] for n in carried]
    init = inits[0] if len(inits) == 1 else ctx.emit("core.tuple", *inits, node=node)
    iv, carry = _fresh_binder(ctx, i64), _fresh_binder(ctx, init.type)  # ONE carry (a tuple when several)
    before = dict(ctx.locals)
    ctx.locals[node.target.id] = iv
    if len(carried) == 1:
        ctx.locals[carried[0]] = carry
    else:
        for k, n in enumerate(carried):
            ctx.locals[n] = ctx.emit("core.extract", carry, node=node, index=k)
    _lower_block(ctx, node.body)
    finals = [ctx.locals[n] for n in carried]
    final = finals[0] if len(finals) == 1 else ctx.emit("core.tuple", *finals, node=node)
    if final.type != init.type:
        raise TypeError(
            f"strict loop carry: {carried!r} enter as {init.type!r} but leave an iteration as "
            f"{final.type!r} [{fmt(ctx.loc(node))}]"
        )
    y = ctx.emit("core.yield", final, node=node)
    ctx.locals = before
    res = ctx.emit("core.for", lo, hi, init, regions=(Region(params=(iv, carry), body=(y,)),), node=node)
    if len(carried) == 1:
        ctx.locals[carried[0]] = res
    else:
        for k, n in enumerate(carried):
            ctx.locals[n] = ctx.emit("core.extract", res, node=node, index=k)
    return None


def _refuse_while(ctx, node):
    raise MissingRule(
        f"`while` is not in the base pack: bounded `for i in range(...)` only — every serious "
        f"kernel language draws this line (R11) [{fmt(ctx.loc(node))}]"
    )


def _refuse_break(ctx, node):
    raise MissingRule(
        f"`{'break' if isinstance(node, ast.Break) else 'continue'}` is not in the base pack: "
        f"loops are single-entry single-exit; guard with `if` instead [{fmt(ctx.loc(node))}]"
    )


def _pass(ctx, node):
    return None


def _call(ctx, node):
    if node.keywords:
        raise MissingRule(f"keyword arguments are not in the base pack [{fmt(ctx.loc(node))}]")
    registry = ctx.context.get("registry")  # planted by Registry._build (the 130 §7 context seam)
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
    ast.If: _if_stmt,
    ast.For: _for_stmt,
    ast.While: _refuse_while,
    ast.Break: _refuse_break,
    ast.Continue: _refuse_break,
    ast.Pass: _pass,
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
