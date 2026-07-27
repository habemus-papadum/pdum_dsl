"""THE one derivative table, at its forced altitude — and both engines.

The one-body-language law (200 §S.3 amendment, 220 §14) ratified ONE
forward-seeding derivative engine over ONE table for every per-element
tier. That forces this module DOWN to the language package: ``sqrt`` is a
dsl intrinsic, so its slope is a language fact — pdum.tl imports the table
from here (its ``nodes``/``derivative`` modules re-export; no consumer
rewrites — the schema's no-rewrite guarantee holds through the move).

Three residents:

- **The schema** (Arg/Const/Prim): the tiny frozen tree contract between
  every producer of scalar expression trees and every consumer.
- **THE TABLE** (op → local-slope builders; ``None`` = gradient-free
  position — the carrier discipline). It grows ONLY when a primitive joins
  the core. The at-kink law: one-sided and PARTITIONING — at a tie exactly
  one operand receives the cotangent, first-wins (``maximum`` goes left on
  ``ge``; ``minimum`` mirrors); frozen contract, pinned at test_at_kink.
- **The two engines over it**: ``diff`` — tree-tier (marker partials,
  ``CompositeMarker.partial``, the reducer BPTT engine); ``tangent`` —
  value-tier forward seeding over core IR, powering ``with_respect_to``
  (value-space, a lowering macro) and ``value_and_grad`` (function-space,
  a Derived transform — the pipe build rule's promised sibling).

Value-tier policy: integer/bool carriers are gradient-free; branch and
loop derivatives are NOT yet vetted (200 §1.3) and refuse loudly; a float
primitive with no table entry refuses — the table never guesses.
"""

from __future__ import annotations

from dataclasses import dataclass

from .capture import Handle
from .derived import DerivedValue
from .ir import VerifyError
from .staging import macro, staged
from .types import Record, Scalar


@dataclass(frozen=True)
class Arg:
    """A formal parameter of the marker (by position)."""

    index: int


@dataclass(frozen=True)
class Const:
    """A literal: int/Fraction (exact) or float (value-space)."""

    value: object


@dataclass(frozen=True)
class Prim:
    """Application of a PRIMITIVE marker, referenced by name."""

    op: str
    args: tuple


Node = Arg | Const | Prim


def is_const(n, v) -> bool:
    return isinstance(n, Const) and n.value == v


TABLE = {
    "add": (lambda a, b: Const(1), lambda a, b: Const(1)),
    "sub": (lambda a, b: Const(1), lambda a, b: Const(-1)),
    "neg": (lambda a: Const(-1),),
    "mul": (lambda a, b: b, lambda a, b: a),
    "div": (
        lambda a, b: Prim("div", (Const(1), b)),
        lambda a, b: Prim("neg", (Prim("div", (a, Prim("mul", (b, b)))),)),
    ),
    "exp": (lambda a: Prim("exp", (a,)),),
    "log": (lambda a: Prim("div", (Const(1), a)),),
    "sqrt": (lambda a: Prim("div", (Const(1), Prim("mul", (Const(2), Prim("sqrt", (a,)))))),),
    "sin": (lambda a: Prim("cos", (a,)),),
    "cos": (lambda a: Prim("neg", (Prim("sin", (a,)),)),),
    "tanh": (lambda a: Prim("sub", (Const(1), Prim("mul", (Prim("tanh", (a,)), Prim("tanh", (a,)))))),),
    # the kinks: first-wins — a tie sends the whole cotangent LEFT
    "maximum": (lambda a, b: Prim("ge", (a, b)), lambda a, b: Prim("gt", (b, a))),
    "minimum": (lambda a, b: Prim("le", (a, b)), lambda a, b: Prim("lt", (b, a))),
    "abs": (lambda a: Prim("where", (Prim("ge", (a, Const(0))), Const(1), Const(-1))),),  # tie at 0 -> +1
    "floor": (None,),  # gradient-free BY DECLARATION (zero a.e.; the carrier discipline)
    "where": (
        None,  # the condition is gradient-free
        lambda c, x, y: Prim("where", (c, Const(1), Const(0))),
        lambda c, x, y: Prim("where", (c, Const(0), Const(1))),
    ),
    "eq": (None, None),
    "ne": (None, None),
    "le": (None, None),
    "lt": (None, None),
    "ge": (None, None),
    "gt": (None, None),
}


def diff(node: Node, i: int) -> Node:
    """Tree tier: d(node)/d(Arg(i)), with light zero/one folding."""
    if isinstance(node, Arg):
        return Const(1) if node.index == i else Const(0)
    if isinstance(node, Const):
        return Const(0)
    if node.op not in TABLE:
        raise NotImplementedError(f"primitive {node.op!r} has no entry in the derivative table")
    rules = TABLE[node.op]
    total: Node = Const(0)
    for j, arg in enumerate(node.args):
        rule = rules[j]
        if rule is None:
            continue
        inner = diff(arg, i)
        if is_const(inner, 0):
            continue
        local = rule(*node.args)
        if is_const(local, 0):
            continue
        if is_const(inner, 1):
            term = local
        elif is_const(local, 1):
            term = inner
        else:
            term = Prim("mul", (local, inner))
        total = term if is_const(total, 0) else Prim("add", (total, term))
    return total


# --- the value-tier engine: forward seeding over core IR ---------------------
# The op<->base-name maps derive from the vocabulary's ONE core-owned map
# (markers.CORE_OWNED) plus the table's row set — never a second hand copy.

from .markers import CORE_OWNED as _CORE_OWNED  # noqa: E402 — after TABLE (markers reads it lazily)
from .markers import PREDS as _PRED_SET  # noqa: E402

_TABLE_KEY = {op: base for base, op in _CORE_OWNED.items()} | {
    f"pw.{n}": n for n in TABLE if n not in _CORE_OWNED and n not in _PRED_SET
}
_SPLICE_OPS = {n: op for op, n in _TABLE_KEY.items() if n != "where"}
_ZERO_LEAVES = {"core.param", "core.env", "core.const", "abi.slot"}
_GRAD_FREE = {"core.cmp"}


def _splice(b, tree, t):
    """A table rule's slope tree -> core IR. Leaves are either schema Consts
    (typed like the primal) or the primal's own operand Nodes, spliced as-is;
    predicate trees become masks via ``core.select`` (the strict dialect has
    no bool arithmetic)."""
    if isinstance(tree, Prim):
        if tree.op == "where":
            c, x, y = tree.args
            return b.emit("core.select", _splice_pred(b, c, t), _splice(b, x, t), _splice(b, y, t))
        if tree.op in _PRED_SET:
            one = b.emit("core.const", type=t, value=1.0)
            zero = b.emit("core.const", type=t, value=0.0)
            return b.emit("core.select", _splice_pred(b, tree, t), one, zero)
        return b.emit(_SPLICE_OPS[tree.op], *(_splice(b, a, t) for a in tree.args))
    if isinstance(tree, Const):
        return b.emit("core.const", type=t, value=float(tree.value))
    return tree  # a primal operand Node


def _splice_pred(b, tree, t):
    return b.emit("core.cmp", _splice(b, tree.args[0], t), _splice(b, tree.args[1], t), pred=tree.op)


def _is_one(n) -> bool:
    return n.op == "core.const" and dict(n.attrs).get("value") == 1.0


def tangent(b, node, seed_key: bytes, memo: dict):
    """d(node)/d(seed) as IR over the primal DAG, memoized by content key.
    Returns ``None`` for an exact zero (sparsity is the common case)."""
    if node.key == seed_key:
        return b.emit("core.const", type=node.type, value=1.0)
    if node.key in memo:
        return memo[node.key]
    memo[node.key] = result = _tangent_of(b, node, seed_key, memo)
    return result


def _tangent_of(b, node, seed_key, memo):
    t = node.type
    if isinstance(t, Scalar) and not t.kind.startswith("f"):
        return None  # integer/bool carriers are gradient-free (the carrier discipline)
    if not isinstance(t, Scalar):
        raise VerifyError(
            f"with_respect_to differentiates real scalars today, not {t!r} — the record/tensor "
            f"clause of the type law arrives with the remaining P8 machinery"
        )
    key = _TABLE_KEY.get(node.op)
    if key is not None:
        total = None
        for rule, arg in zip(TABLE[key], node.args):
            if rule is None:
                continue
            dt = tangent(b, arg, seed_key, memo)
            if dt is None:
                continue
            local = _splice(b, rule(*node.args), t)
            term = local if _is_one(dt) else (dt if _is_one(local) else b.emit("core.mul", local, dt))
            total = term if total is None else b.emit("core.add", total, term)
        return total
    if node.op in _ZERO_LEAVES or node.op in _GRAD_FREE:
        return None
    if node.op == "core.cast":
        dt = tangent(b, node.args[0], seed_key, memo)
        return None if dt is None else b.emit("core.cast", dt, to=dict(node.attrs)["to"])
    if node.regions:
        raise VerifyError(
            f"cannot differentiate through {node.op!r}: branch/loop derivatives are not yet "
            f"vetted (200 §1.3) — differentiate straight-line value code"
        )
    raise VerifyError(
        f"primitive {node.op!r} has no entry in the derivative table — it grows only when a primitive joins the core"
    )


@macro
def _with_respect_to(ctx, args, node):
    """The value-space operator, as a LOWERING MACRO: differentiate the
    already-lowered DAG from the wrt value to the differentiated value —
    forward seeding, at lower time, into the same region. Tangent memos live
    in the build context (the 130 §7 seam), keyed by seed."""
    from .lower import fmt

    if len(args) != 2:
        raise VerifyError(f"with_respect_to(value, wrt) takes exactly two values [{fmt(ctx.loc(node))}]")
    v, u = args
    if not (isinstance(u.type, Scalar) and u.type.kind.startswith("f")):
        raise VerifyError(
            f"with_respect_to: the wrt must be castable to real (a float scalar), got {u.type!r} [{fmt(ctx.loc(node))}]"
        )
    memo = ctx.context.setdefault("tangent_memos", {}).setdefault(u.key, {})

    def d_of(x):
        # THE DERIVATIVE TYPE LAW (200 §1.3, the record clause): the result
        # has the SAME TYPE as the value — per-field for records, recursively.
        if isinstance(x.type, Record):
            if x.op != "core.tuple":
                raise VerifyError(
                    f"with_respect_to: a record value differentiates per-field when constructed "
                    f"in-body; a {x.type.name} arriving as {x.op!r} has no per-field view yet "
                    f"[{fmt(ctx.loc(node))}]"
                )
            return ctx.builder.emit("core.tuple", *(d_of(a) for a in x.args), type=x.type)
        if not (isinstance(x.type, Scalar) and x.type.kind.startswith("f")):
            raise VerifyError(
                f"with_respect_to: the value must be castable to real (a float scalar or a "
                f"record of them), got {x.type!r} [{fmt(ctx.loc(node))}]"
            )
        d = tangent(ctx.builder, x, u.key, memo)
        return d if d is not None else ctx.builder.emit("core.const", type=x.type, value=0.0)

    return d_of(v)


class _ValueAndGrad(DerivedValue):
    __slots__ = ()

    def __init__(self, base: Handle, wrt: tuple):
        super().__init__(
            "value_and_grad",
            base.fntype.template,
            (base.fntype,),
            (("wrt", wrt),),
            ("VG", base.fp, wrt),
            (base,),
        )


@staged
def value_and_grad(f, *, wrt):
    """Function-space: ``g = value_and_grad(f, wrt=("y", "x"))`` is a Derived
    transform; ``g(args)`` returns ``(f(args), (df/dy, df/dx))``. With wrt =
    the ambient coordinates this IS dFdx/dFdy/fwidth (200 §S.4)."""
    if not isinstance(f, Handle):
        raise TypeError(f"value_and_grad transforms a jitted Handle, got {type(f).__name__}")
    wrt = (wrt,) if isinstance(wrt, str) else tuple(wrt)
    if not wrt:
        raise TypeError("value_and_grad needs at least one wrt parameter name")
    return _ValueAndGrad(f, wrt)


def build_value_and_grad(vg, rules, ops, arg_types, derived, *, context=None, prefix=()):
    """The transform build rule (the ``build_pipe`` sibling): lower the base
    with the caller's arg types, then run the ONE engine once per wrt
    parameter over the same DAG; yield ``(value, (grads...))``."""
    from .ir import Builder, Loc, Region
    from .lower import Lowerer, check_coherence

    base = vg.captures[0]
    check_coherence(base)
    (wrt,) = [v for k, v in vg.fntype.template.static_params if k == "wrt"]
    code = base.pyfunc.__code__
    names = code.co_varnames[: code.co_argcount]
    if len(arg_types) != len(names):
        raise VerifyError(f"{vg.fntype.template.label} takes {len(names)} args; got types for {len(arg_types)}")
    builder = Builder(ops)
    params = tuple(builder.param(i, t) for i, t in enumerate(arg_types))
    sub = Lowerer(base, rules, ops, derived, prefix=(*prefix, 0), wrap=Loc("<value_and_grad>", 1), context=context)
    sub.params = params
    sub.locals.update(zip(names, params))
    value = sub.run_body()
    grads = []
    for w in wrt:
        if w not in names:
            raise VerifyError(
                f"value_and_grad: wrt name {w!r} is not a parameter of "
                f"{base.fntype.template.label} (parameters: {', '.join(names)})"
            )
        seed = params[names.index(w)]
        if not (isinstance(seed.type, Scalar) and seed.type.kind.startswith("f")):
            raise VerifyError(f"value_and_grad: wrt parameter {w!r} must be a float scalar, got {seed.type!r}")
        d = tangent(builder, value, seed.key, {})
        grads.append(d if d is not None else builder.emit("core.const", type=value.type, value=0.0))
    out = builder.emit("core.tuple", value, builder.emit("core.tuple", *grads))
    return Region(params=params, body=(builder.emit("core.yield", out),))


def install(registry) -> None:
    """with_respect_to (a macro overload) + the value_and_grad build rule."""
    registry.overloads["with_respect_to"] = _with_respect_to
    registry.derived["value_and_grad"] = build_value_and_grad
