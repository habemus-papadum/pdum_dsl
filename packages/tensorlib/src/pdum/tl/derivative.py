"""THE one derivative table (200 §S.2) — op → linearization, None = gradient-free.

The value-tier tangent rules and the marker partial rules are the same
object: this table. It grows ONLY when a primitive joins the core;
everything else derives — ``CompositeMarker.partial(i)`` rewrites the lowered
body through ``diff`` (derivation-under-cache), and the reducer BPTT engine
consumes partials through the same interface.

**The at-kink law.** The table is one-sided and PARTITIONS: at a tie exactly
one operand receives the cotangent — first-wins. ``maximum`` sends the
cotangent left on ``ge`` and right on strict ``gt`` (a tie goes left);
``minimum`` mirrors with ``le``/``lt``. Reduce adjoints derive through the
pairwise combine and inherit the partition (autodiff's first-occurrence
mask). This is frozen contract, pinned at the kink points (test_at_kink).
"""

from __future__ import annotations

from .nodes import Const, Node, Prim
from .nodes import is_const as _is_const

# per-primitive local-slope builders: arg nodes -> Node; None = gradient-free
# position (the carrier discipline at the tree level)
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
    """d(node)/d(Arg(i)), with light zero/one folding to keep trees small."""
    from .nodes import Arg

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
        if _is_const(inner, 0):
            continue
        local = rule(*node.args)
        if _is_const(local, 0):
            continue
        if _is_const(inner, 1):
            term = local
        elif _is_const(local, 1):
            term = inner
        else:
            term = Prim("mul", (local, inner))
        total = term if _is_const(total, 0) else Prim("add", (total, term))
    return total
