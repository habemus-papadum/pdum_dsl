"""Transforms: over (named axis binding), jvp (forward AD), in-kernel ``D``.

Design: ``110_transforms-and-derivatives.md``. The load-bearing choices:

- **over is SIMT-shaped, not SIMD-shaped**: it adds one trailing i64
  coordinate parameter and WEAVES it into every access to a capture whose
  NamedArray type carries the mapped axis. Intermediates stay scalar, each
  lane runs its own branches and trip counts — control flow needs zero new
  machinery, and the lazy-branch guarantee survives (no JAX ``where`` wart).
- **One tangent engine, two doors**: per-op linearization rules synthesize
  tangent nodes alongside the existing primal DAG. ``jvp(f)`` seeds every
  float parameter with a tangent parameter; ``D(x)`` seeds parameter basis
  vectors and hands back one partial per enclosing-kernel param. Zero
  tangents are ``None`` until forced, so untouched slices cost nothing.
- **Loops widen**: a tangent recurrence needs the primal carry, so
  ``core.for`` rebuilds with carry ``(primal, tangent)``; jvp re-points
  primal consumers at lane 0 of the widened loop.
- ``matmul(A, B, *out_indices)`` is the named-contraction stretch: pair
  the operands' unique shared axis name (woven axes excluded FIRST — that
  is how batch matmul composes for free), read the trip count from the
  shape slot (rank-generic), and expand to the element loop.
"""

from __future__ import annotations

import ast

from ..kernel.derived import DerivedValue
from ..kernel.ir import Builder, Region, VerifyError
from ..kernel.lower import MissingRule, fmt, lower_handle
from ..kernel.types import Scalar, Tuple, i64


class _Tangents:
    """Tangent synthesis over an existing DAG. ``None`` means a structural
    zero (forced only at joins/results); ``seed(index, type)`` supplies the
    tangent of integer-indexed function params; loop widenings land in
    ``replace`` so a whole-region caller can re-point primal consumers."""

    def __init__(self, emit, seed, memo):
        self.emit, self.seed, self.memo = emit, seed, memo
        self.replace: dict = {}

    def mat(self, t, node):
        return t if t is not None else _zero_like_e(self.emit, node.type)

    def of(self, n):
        # The memo stores (node, tangent) PAIRS: keying by id() alone let a
        # GC'd node's address be reused by a fresh node, silently serving the
        # dead node's tangent (review-caught, execution-verified). Holding the
        # node reference pins the id for the memo's lifetime.
        hit = self.memo.get(id(n))
        if hit is not None:
            return hit[1]
        t = self._rule(n)
        self.memo[id(n)] = (n, t)
        return t

    def _rule(self, n):  # noqa: C901 — one dispatch table, kept together on purpose
        e, op, a = self.emit, n.op, n.args
        attrs = dict(n.attrs)
        if op == "core.param":
            idx = attrs["index"]
            if not isinstance(idx, int):
                # A loop binder with no pre-seeded tangent: D was applied INSIDE
                # a loop body, where the carry's dependence on the params is
                # invisible without tangent carries. Silence here dropped the
                # recurrence term (review-caught) — refuse instead.
                raise VerifyError(
                    "derivative reached a loop binder without a tangent — apply D/jvp OUTSIDE the "
                    "loop body (tangent carries through in-body derivatives arrive with grad, step 13)"
                )
            return self.seed(idx, n.type)
        if op in ("core.const", "core.env", "abi.slot", "array.buffer", "array.load", "core.cmp"):
            return None  # constants w.r.t. args (loads: data is frozen; index jumps are measure-zero)
        if op in ("core.add", "core.sub"):
            ta, tb = self.of(a[0]), self.of(a[1])
            if ta is None and tb is None:
                return None
            if tb is None:
                return ta
            if ta is None:
                return tb if op == "core.add" else e("core.neg", tb)
            return e(op, ta, tb)
        if op == "core.neg":
            ta = self.of(a[0])
            return None if ta is None else e("core.neg", ta)
        if op == "core.mul":
            ta, tb = self.of(a[0]), self.of(a[1])
            left = None if tb is None else e("core.mul", a[0], tb)
            right = None if ta is None else e("core.mul", ta, a[1])
            if left is None:
                return right
            return left if right is None else e("core.add", left, right)
        if op == "core.div":
            ta, tb = self.of(a[0]), self.of(a[1])
            top = None if ta is None else e("core.div", ta, a[1])
            if tb is None:
                return top
            corr = e("core.div", e("core.mul", a[0], tb), e("core.mul", a[1], a[1]))
            return e("core.neg", corr) if top is None else e("core.sub", top, corr)
        if op == "core.pow":
            if self.of(a[1]) is not None:
                raise VerifyError("jvp: varying exponent — write exp(b*log(a)) explicitly")
            ta = self.of(a[0])
            if ta is None:
                return None
            one = e("core.const", type=a[1].type, value=1.0)
            return e("core.mul", e("core.mul", a[1], e("core.pow", a[0], e("core.sub", a[1], one))), ta)
        if op == "core.cast":
            to = attrs["to"]
            ta = self.of(a[0])
            if ta is None or not (isinstance(to, Scalar) and to.kind[0] == "f"):
                return None  # casts through int kill tangents (documented)
            return ta if ta.type == to else e("core.cast", ta, to=to)
        if op == "core.select":
            ta, tb = self.of(a[1]), self.of(a[2])
            if ta is None and tb is None:
                return None
            return e("core.select", a[0], self.mat(ta, a[1]), self.mat(tb, a[2]))
        if op == "core.tuple":
            ts = [self.of(x) for x in a]
            if all(t is None for t in ts):
                return None
            return e("core.tuple", *(self.mat(t, x) for t, x in zip(ts, a)))
        if op == "core.extract":
            tb = self.of(a[0])
            return None if tb is None else e("core.extract", tb, index=attrs["index"])
        if op == "core.field":
            tb = self.of(a[0])
            return None if tb is None else e("core.field", tb, name=attrs["name"])
        if op == "core.if":
            yt = [r.body[-1].args[0] for r in n.regions]
            ts = [self.of(y) for y in yt]
            if all(t is None for t in ts):
                return None
            regions = tuple(Region(body=(e("core.yield", self.mat(t, y)),)) for t, y in zip(ts, yt))
            return e("core.if", a[0], regions=regions)
        if op == "core.for":
            return self._widen_for(n)
        rule = JVP_RULES.get(op)
        if rule is None:
            raise VerifyError(f"no jvp rule for {op!r} — register one (the transform column is open)")
        ts = [self.of(x) for x in a]
        if all(t is None for t in ts):
            return None
        return rule(e, n, a, [self.mat(t, x) for t, x in zip(ts, a)])

    def _widen_for(self, n):
        """Rebuild the loop with carry (primal, tangent); memo the tangent as
        lane 1 and record lane 0 as the primal replacement."""
        e = self.emit
        lo, hi, init = n.args
        iv, carry = n.regions[0].params
        t_init = self.mat(self.of(init), init)
        init_w = e("core.tuple", init, t_init)
        carry_w = self._param(("jvp", dict(carry.attrs)["index"]), init_w.type)
        sub = {id(carry): e("core.extract", carry_w, index=0)}
        body_final = _substitute(n.regions[0].body[-1].args[0], sub, e)
        inner = _Tangents(e, self.seed, dict(self.memo))
        inner.memo[id(sub[id(carry)])] = (sub[id(carry)], e("core.extract", carry_w, index=1))
        inner.memo[id(iv)] = (iv, None)
        t_final = inner.mat(inner.of(body_final), body_final)
        y = e("core.yield", e("core.tuple", body_final, t_final))
        new = e("core.for", lo, hi, init_w, regions=(Region(params=(iv, carry_w), body=(y,)),))
        self.replace[id(n)] = e("core.extract", new, index=0)
        return e("core.extract", new, index=1)

    def _param(self, index, type):
        from ..kernel.ir import Node

        return Node("core.param", type, attrs=(("index", index),))


def _zero_like_e(emit, t):
    if isinstance(t, Scalar):
        return emit("core.const", type=t, value=0.0 if t.kind[0] == "f" else 0)
    if isinstance(t, Tuple):
        return emit("core.tuple", *(_zero_like_e(emit, e2) for e2 in t.elems))
    raise VerifyError(f"no zero tangent for type {t!r}")


def _substitute(node, sub, emit):
    """Rebuild the slice above replaced nodes (memoized; regions included)."""
    memo: dict = {}

    def walk(n):
        if id(n) in sub:
            return sub[id(n)]
        if id(n) in memo:
            return memo[id(n)]
        args = tuple(walk(a) for a in n.args)
        regions = tuple(Region(r.params, tuple(walk(m) for m in r.body)) for r in n.regions)
        if args == n.args and regions == n.regions:
            memo[id(n)] = n
            return n
        out = emit(n.op, *args, regions=regions, type=n.type, **dict(n.attrs))
        memo[id(n)] = out
        return out

    return walk(node)


# --- the transform column (surface-A-shaped: op -> linearization rule) ---------
def _r_sqrt(e, n, a, t):
    two = e("core.const", type=n.type, value=2.0)
    return e("core.div", t[0], e("core.mul", two, n))


def _r_minmax(pred):
    def rule(e, n, a, t):
        return e("core.select", e("core.cmp", a[0], a[1], pred=pred), t[0], t[1])

    return rule


JVP_RULES = {
    "math.sqrt": _r_sqrt,
    "math.exp": lambda e, n, a, t: e("core.mul", n, t[0]),
    "math.sin": lambda e, n, a, t: e("core.mul", e("math.cos", a[0]), t[0]),
    "math.cos": lambda e, n, a, t: e("core.neg", e("core.mul", e("math.sin", a[0]), t[0])),
    "math.floor": lambda e, n, a, t: None,  # piecewise constant
    "math.abs": lambda e, n, a, t: e(
        "core.select", e("core.cmp", a[0], _zero_like_e(e, a[0].type), pred="ge"), t[0], e("core.neg", t[0])
    ),
    "math.min": _r_minmax("le"),
    "math.max": _r_minmax("ge"),
}


# --- the wrappers (DerivedValue subclasses: ONE protocol, 130 §7) ---------------
class Over(DerivedValue):
    """`over(f, axis="batch")`: the axis-binding transform. v1 lowering is
    the SIMT weave (the lane arrives as a trailing i64 coordinate); the
    map-loop IR form arrives with the tensor step (130 §4.3) without
    changing this identity."""

    __slots__ = ("base", "axis")

    def __init__(self, base, axis: str):
        self.base, self.axis = base, axis
        super().__init__("over", base.fntype.template, (base.fntype,), (("axis", axis),), ("O", axis, base.fp), (base,))


class Jvp(DerivedValue):
    __slots__ = ("base",)

    def __init__(self, base):
        self.base = base
        super().__init__("jvp", base.fntype.template, (base.fntype,), (), ("J", base.fp), (base,))


def over(f, *, axis: str):
    """This kernel, over that axis: `over(f, axis="batch")(*args, b)` runs
    lane `b`. Captures carrying the named axis are woven; everything else
    broadcasts. Composes: `over(over(g, axis="x"), axis="y")(*args, bx, by)`
    — lanes trail in application order, outermost LAST."""
    return Over(f, axis)


def jvp(f):
    """Forward mode: `jvp(f)(*args, *tangents)` -> (primal, directional)."""
    return Jvp(f)


def build_over(w: Over, rules, ops, arg_types, derived, *, context=None, prefix=()):
    """Composition IS lower_handle re-entry (130 §7): a Derived base
    dispatches to its own build rule with the merged context intact — no
    special composition mechanism exists or is needed."""
    context = dict(context or {})
    if not arg_types or arg_types[-1] != i64:
        raise VerifyError(f"over adds a trailing i64 lane coordinate: call g(*args, b); got {arg_types!r}")
    woven = dict(context.get("woven") or {})
    if w.axis in woven:
        raise VerifyError(f"over axis {w.axis!r} is already mapped by an enclosing over — axes nest, never repeat")
    lane = Builder(ops).param(len(arg_types) - 1, i64)
    woven[w.axis] = lane
    hits = context.setdefault("woven_hits", [])
    context["woven"] = woven
    inner = lower_handle(
        w.base, rules, ops, arg_types=arg_types[:-1], derived=derived, context=context, prefix=(*prefix, 0)
    )
    if not any(w.axis in entry for entry in hits):
        raise VerifyError(
            f"over axis {w.axis!r}: no capture carries it — name the axis on the data (Named/xarray), or drop the over"
        )
    return Region(params=(*inner.params, lane), body=inner.body)


def build_jvp(w: Jvp, rules, ops, arg_types, derived, *, context=None, prefix=()):
    n = len(arg_types) // 2
    if len(arg_types) != 2 * n or arg_types[:n] != arg_types[n:]:
        raise VerifyError(f"jvp doubles the args: call jf(*args, *tangents) with matching types; got {arg_types!r}")
    for t in arg_types[:n]:
        if not (isinstance(t, Scalar) and t.kind[0] == "f"):
            raise VerifyError(f"jvp differentiates w.r.t. float args; got {t!r}")
    inner = lower_handle(
        w.base, rules, ops, arg_types=arg_types[:n], derived=derived, context=dict(context or {}), prefix=(*prefix, 0)
    )
    b = Builder(ops)
    result = inner.body[-1].args[0]
    tparams = tuple(b.param(n + i, t) for i, t in enumerate(arg_types[n:]))
    eng = _Tangents(b.emit, lambda k, t: tparams[k] if k < len(tparams) else None, {})
    tangent = eng.mat(eng.of(result), result)
    out = b.emit("core.tuple", result, tangent)
    if eng.replace:  # widened loops: re-point EVERY consumer (primal AND the old
        out = _substitute(out, eng.replace, b.emit)  # nodes inside tangent products)
    return Region(params=(*inner.params, *tparams), body=(b.emit("core.yield", out),))


# --- the in-kernel doors: D and matmul ------------------------------------------
def _d_operator(ctx, node):
    argc = len(ctx.root.params)  # the kernel's own params, however deep the inlining (the root seam)
    if argc == 0:
        raise MissingRule(f"D needs the enclosing kernel's params (none in this build) [{fmt(ctx.loc(node))}]")
    if node.keywords or len(node.args) != 1:
        raise MissingRule(f"D takes exactly one positional value [{fmt(ctx.loc(node))}]")
    x = ctx.lower(node.args[0])
    parts = []
    for j in range(argc):
        memo = ctx.context.setdefault("tangents", {}).setdefault(j, {})

        def seed(k, t, j=j, node=node):
            if k != j:
                return None
            if not (isinstance(t, Scalar) and t.kind[0] == "f"):
                raise MissingRule(f"D differentiates w.r.t. float params; param {k} is {t!r} [{fmt(ctx.loc(node))}]")
            return ctx.emit("core.const", node=node, type=t, value=1.0)

        eng = _Tangents(lambda op, *a, **kw: ctx.emit(op, *a, node=node, **kw), seed, memo)
        part = eng.mat(eng.of(x), x)
        if eng.replace:  # keep the D-slice on the widened loop (the kernel's own
            part = _substitute(part, eng.replace, lambda op, *a, **kw: ctx.emit(op, *a, node=node, **kw))
        parts.append(part)  # primal consumers still run the original — documented residual
    return ctx.emit("core.tuple", *parts, node=node)  # ALWAYS a tuple: uniform contract, sugar-safe


def _matmul(ctx, node):
    from .arrays import NamedArray, _linear_index
    from .base_lang import _fresh_binder

    if node.keywords or len(node.args) < 3:
        raise MissingRule(f"matmul(A, B, *out_indices) — positional only [{fmt(ctx.loc(node))}]")
    A, B = ctx.lower(node.args[0]), ctx.lower(node.args[1])
    idx = [ctx.lower(a) for a in node.args[2:]]
    for m in (A, B):
        if not isinstance(m.type, NamedArray):
            raise MissingRule(f"matmul pairs axes BY NAME; got {m.type!r} — name your arrays [{fmt(ctx.loc(node))}]")
    woven = ctx.context.get("woven") or {}
    adims = [d for d in A.type.dims if d not in woven]
    bdims = [d for d in B.type.dims if d not in woven]
    shared = [d for d in adims if d in bdims]
    if len(shared) != 1:
        raise MissingRule(
            f"matmul needs exactly ONE shared axis name to contract; {A.type.dims} vs {B.type.dims} "
            f"share {shared!r} [{fmt(ctx.loc(node))}]"
        )
    inner = shared[0]
    outs = [d for d in adims if d != inner] + [d for d in bdims if d != inner]
    # Duplicate OUT names are impossible: a name on both sides would be a
    # second SHARED name, refused above (dominance; stage-1 audit killed the
    # dead guard that used to sit here).
    if len(idx) != len(outs):
        raise MissingRule(
            f"matmul: {len(outs)} output indices for axes {outs!r}, got {len(idx)} [{fmt(ctx.loc(node))}]"
        )
    from .arrays import ShapedArray

    if isinstance(A.type, ShapedArray) and isinstance(B.type, ShapedArray):
        ea, eb = A.type.shape[A.type.dims.index(inner)], B.type.shape[B.type.dims.index(inner)]
        if ea != eb:
            raise MissingRule(f"matmul: contracted axis {inner!r} has extents {ea} vs {eb} [{fmt(ctx.loc(node))}]")
    # Rank-generic operands: extents are runtime staging values — mismatch is UB
    # like every unchecked index (100's no-bounds-check posture), trip count from A.
    given = dict(zip(outs, idx))
    pos = 1 + A.type.dims.index(inner)
    if A.op == "core.env":
        hi = ctx.emit("core.env", node=node, type=i64, slot=(*dict(A.attrs)["slot"], pos))
    else:  # an argument array: the trip count reads its plan slot via array.dim
        hi = ctx.emit("array.dim", node=node, src=("arg", dict(A.attrs)["index"]), sub=pos)
    zero = ctx.emit("core.const", node=node, type=A.type.dtype, value=0.0)
    iv, acc = _fresh_binder(ctx, i64), _fresh_binder(ctx, A.type.dtype)

    def load(m, dims):
        used = [d for d in dims if d in woven]
        if used:
            hits = ctx.context.get("woven_hits")
            if hits is not None:
                hits.append(tuple(used))
        order = [woven[d] if d in woven else (iv if d == inner else given[d]) for d in dims]
        return _linear_index(ctx, node, m, order)

    term = ctx.emit("core.mul", load(A, A.type.dims), load(B, B.type.dims), node=node)
    y = ctx.emit("core.yield", ctx.emit("core.add", acc, term, node=node), node=node)
    lo = ctx.emit("core.const", node=node, type=i64, value=0)
    return ctx.emit("core.for", lo, hi, zero, regions=(Region(params=(iv, acc), body=(y,)),), node=node)


def make_call_rule(prev):
    def _call(ctx, node):
        f = node.func
        if isinstance(f, ast.Name) and f.id in ("D", "matmul"):
            shadowed = f.id in ctx.locals or f.id in getattr(ctx.handle, "env", {})
            if not shadowed:
                return _d_operator(ctx, node) if f.id == "D" else _matmul(ctx, node)
        return prev(ctx, node)

    return _call


def install(registry) -> None:
    registry.derived["over"] = build_over
    registry.derived["jvp"] = build_jvp
    registry.lower_rules[ast.Call] = make_call_rule(registry.lower_rules[ast.Call])
