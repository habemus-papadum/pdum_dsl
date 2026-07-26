"""The tl dialect over the dsl core (240 C4.1) — promoted from the C3 spike.

ONE IR, many dialects (the pivot's objective): tensor-typed SSA lives on
the dsl ``Node``/``Region`` infrastructure as the ``tl.*`` op family, with
the layout lattice AS the type — so tl's alignment law is an ordinary
type rule that refuses at emission with source locations, and content
keys (``region.key``) come from the core for free.

Design items resolved here (the spike's frictions):

- **The per-kind yield protocol.** Every region ends in ``core.yield``
  (Julia's rule, at the IR level); WHAT a body yields is the KIND's
  declaration, never user ceremony: a ``step`` yields its returned
  value, a ``compute`` kernel yields the final ordering token (its
  effect) — and a kernel ``return`` refuses with the pinned voice.
- **Object-based intrinsic recognition.** ``thread_idx`` and the S.1
  vocabulary (``pointwise``, ``const_like``) are recognized by OBJECT
  IDENTITY from the function's own globals — never by name string.
- **The op-selection pattern, written once.** Operator/comparison rules
  are built by ONE factory (``_typed_rule``): lower the children, and
  TENSOR-TYPEDNESS selects the dialect path (a table row), else the
  base value pack serves the scalar subtree unchanged. A dialect
  extends by rows, not by wrapping rules.
- **The two-pass mechanism (owner-ruled).** Pass 1
  (``check_fold_step_supported``) walks a region and refuses anything
  the machinery cannot handle, WITH the reason; pass 2 does the work.

Deliberately deferred, recorded in 240: charts/labels/levels and dtype
join TensorType with the step-tier migration; store index checking and
fn-valued arguments join with the kernel switch (C4.2); reduce/scan and
the layout family follow as slices. Schedules (store-all vs revolve)
are EVALUATION STRATEGIES over the same regions — never IR (the C3b
finding); an L4 certified descent may bake one later.
"""

from __future__ import annotations

import ast as pyast
from dataclasses import dataclass

from pdum.dsl.capture import make_handle
from pdum.dsl.derivative import TABLE, Const, Prim
from pdum.dsl.ir import Builder, Region
from pdum.dsl.lower import Lowerer, check_coherence
from pdum.dsl.ops import CORE_OPS, PURE, OpDef
from pdum.dsl.registry import DEFAULT
from pdum.dsl.types import Type, f64
from pdum.dsl.value import LOWER_RULES, _assign, _binop, _call, _compare

from .compute import const_like
from .compute import iota as _eager_iota
from .compute import pointwise as _eager_pw
from .ir import Token, _store, pw_marker
from .lifting import _Intrinsic
from .markers import Marker
from .tensor import Tensor

# --- types -------------------------------------------------------------------


@dataclass(frozen=True)
class TensorType(Type):
    """A tensor-typed SSA value: the layout lattice IS the type."""

    dims: tuple  # ((name, start, stop), ...)


@dataclass(frozen=True)
class TokenType(Type):
    """The ordering token — stores consume and produce it (dataflow)."""


def tensor_type(t: Tensor) -> TensorType:
    return TensorType(tuple((d.name, d.start, d.stop) for d in t.layout.dims))


def _minus_dim(tt: TensorType, name: str) -> TensorType:
    return TensorType(tuple(d for d in tt.dims if d[0] != name))


# --- the ops, typed by rules (tl's alignment law AS type rules) --------------


def _r_pointwise(args, attrs, regions):
    ts = [a for a in args if isinstance(a, TensorType)]
    if not ts:
        raise TypeError("tl.pointwise wants at least one tensor operand")
    if any(t != ts[0] for t in ts):
        raise TypeError(f"tl.pointwise wants ALIGNED operands, got {ts[0]!r} vs a mismatch")
    return ts[0]  # scalars broadcast (the const-lift discipline)


def _r_iota(args, attrs, regions):
    (t,) = args
    if not isinstance(t, TensorType):
        raise TypeError("tl.iota wants the lattice source tensor")
    if attrs["name"] not in [d[0] for d in t.dims]:
        raise TypeError(f"tl.iota: the lattice has no dim {attrs['name']!r}")
    return t


def _r_store(args, attrs, regions):
    tok, dst, val = args
    if not isinstance(tok, TokenType):
        raise TypeError("tl.store threads the ordering token first")
    if dst != val:
        raise TypeError(f"tl.store wants the value aligned to the target: {dst!r} vs {val!r}")
    return TokenType()


def _r_fold(args, attrs, regions):
    init, src = args
    (step,) = regions
    if not (isinstance(init, TensorType) and isinstance(src, TensorType)):
        raise TypeError("tl.fold wants (init state, element source) tensors")
    elem = _minus_dim(src, attrs["dim"])
    s_t, m_t = (p.type for p in step.params)
    if s_t != init:
        raise TypeError(f"tl.fold: the state binder is {s_t!r} but the init state is {init!r}")
    if m_t != elem:
        raise TypeError(f"tl.fold: the element binder is {m_t!r} but slicing {attrs['dim']!r} gives {elem!r}")
    if step.body[-1].args[0].type != init:
        raise TypeError("tl.fold: the step must yield the state type")
    return init


TL_OPS = {
    "tl.token": OpDef("tl.token", lambda a, at, r: TokenType(), PURE),
    "tl.uniform": OpDef("tl.uniform", None, PURE),  # a per-launch scalar slot (type passed explicitly)
    "tl.iota": OpDef("tl.iota", _r_iota, PURE),
    "tl.pointwise": OpDef("tl.pointwise", _r_pointwise, PURE),
    "tl.store": OpDef("tl.store", _r_store, PURE),  # the effect rides the token
    "tl.fold": OpDef("tl.fold", _r_fold, PURE, nregions=1),
}


# --- the rule pack: ONE typed-rule factory; dialects extend by rows ----------

_BIN_MARKER = {pyast.Add: "add", pyast.Sub: "sub", pyast.Mult: "mul", pyast.Div: "div"}
_CMP_MARKER = {pyast.Lt: "lt", pyast.Gt: "gt", pyast.LtE: "le", pyast.GtE: "ge"}


def _typed_rule(table, base_rule, pick):
    """The op-selection pattern, once: lower the children; a tensor operand
    selects the dialect row from ``table``; otherwise the base value pack
    serves the node unchanged."""

    def rule(ctx, node):
        f = table.get(type(pick(node)))
        if f is not None:
            operands = [ctx.lower(a) for a in _operands_of(node)]
            if any(isinstance(o.type, TensorType) for o in operands):
                return ctx.emit("tl.pointwise", *operands, node=node, f=f)
        return base_rule(ctx, node)

    return rule


def _operands_of(node):
    if isinstance(node, pyast.BinOp):
        return (node.left, node.right)
    return (node.left, node.comparators[0])


def _pick_binop(node):
    return node.op


def _pick_cmp(node):
    return node.ops[0] if len(node.ops) == 1 else None


def _globals_of(ctx):
    return ctx.handle.pyfunc.__globals__


def _tl_call(ctx, node):
    """Call resolution, OBJECT-IDENTITY-FIRST: the tl intrinsics and the S.1
    vocabulary are recognized as the objects the body's globals actually
    hold (direct-name imports — the package attr ``compute`` is the
    DECORATOR, the P5 shadowing lesson); markers via the registry;
    everything else is the base pack."""
    if isinstance(node.func, pyast.Name):
        obj = _globals_of(ctx).get(node.func.id)
        if isinstance(obj, _Intrinsic) and obj.name == "thread_idx":
            lattice = ctx.root.params[-1]  # the writable target (S.3 convention)
            names = [c.value for c in node.args]
            return tuple(ctx.emit("tl.iota", lattice, node=node, name=n) for n in names)  # ALWAYS a tuple
        if obj is _eager_pw:  # the S.1 STEP-tier spelling
            marker = _globals_of(ctx).get(node.args[0].id) or ctx.context["registry"].overloads.get(node.args[0].id)
            rest = [ctx.lower(a) for a in node.args[1:]]
            return ctx.emit("tl.pointwise", *rest, node=node, f=marker.name)
        if obj is const_like:  # scalar broadcast: the const IS the operand
            return ctx.lower(node.args[1])
        impl = ctx.context["registry"].overloads.get(node.func.id)
        if isinstance(impl, Marker):
            args = [ctx.lower(a) for a in node.args]
            if any(isinstance(a.type, TensorType) for a in args):
                return ctx.emit("tl.pointwise", *args, node=node, f=impl.name)
    return _call(ctx, node)


def _tl_assign(ctx, node):
    tgt = node.targets[0]
    if isinstance(tgt, pyast.Subscript):  # the store (full aligned lattice; indices join C4.2)
        target, value = ctx.lower(tgt.value), ctx.lower(node.value)
        tok = ctx.context.get("tl.token") or ctx.emit("tl.token", node=node)
        ctx.context["tl.token"] = ctx.emit("tl.store", tok, target, value, node=node)
        return None
    value = _tl_call(ctx, node.value) if isinstance(node.value, pyast.Call) else ctx.lower(node.value)
    if isinstance(tgt, pyast.Tuple) and isinstance(value, tuple):  # y, x = thread_idx(...)
        for e, v in zip(tgt.elts, value):
            ctx.locals[e.id] = v
        return None
    if isinstance(tgt, pyast.Name):
        ctx.locals[tgt.id] = value
        return None
    return _assign(ctx, node)


TL_RULES = {
    **LOWER_RULES,
    pyast.Assign: _tl_assign,
    pyast.BinOp: _typed_rule(_BIN_MARKER, _binop, _pick_binop),
    pyast.Compare: _typed_rule(_CMP_MARKER, _compare, _pick_cmp),
    pyast.Call: _tl_call,
}


# --- the per-kind yield protocol + the body driver ---------------------------


def lower_body(fn, arg_types: tuple, *, kind: str, registry=None) -> Region:
    """Lower a body through the DSL Lowerer with the tl pack. WHAT the
    region yields is the KIND's declaration (the per-kind yield protocol):
    a ``step`` yields its returned value; a ``compute`` kernel yields the
    final ordering token — and a kernel ``return`` refuses."""
    handle = make_handle(fn, "device")
    check_coherence(handle)
    ctx = Lowerer(handle, TL_RULES, {**CORE_OPS, **TL_OPS}, {}, context={"registry": registry or DEFAULT})
    params = tuple(ctx.builder.param(i, t) for i, t in enumerate(arg_types))
    ctx.params = params
    names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    ctx.locals.update(zip(names, params))
    if kind == "step":
        return Region(params=params, body=(ctx.builder.emit("core.yield", ctx.run_body()),))
    if kind != "compute":
        raise ValueError(f"unknown body kind {kind!r} — kinds declare their yields here")
    tree = next(n for n in pyast.parse(handle.snapshot.text).body if isinstance(n, pyast.FunctionDef))
    for stmt in tree.body:
        if isinstance(stmt, pyast.Return):
            raise ValueError("kernels return nothing — stores into writable arguments are the effect")
        ctx.lower(stmt)
    tok = ctx.context.get("tl.token")
    if tok is None:
        raise ValueError(f"{fn.__qualname__} stores nothing — a kernel's effect is its stores")
    return Region(params=params, body=(ctx.builder.emit("core.yield", tok),))


# --- pass 1 of the two-pass mechanism ----------------------------------------

_FOLD_STEP_SUPPORTED = {"tl.pointwise", "core.const", "core.param", "core.yield"}


def walk_region(region: Region):
    seen: set[int] = set()

    def go(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for a in n.args:
            yield from go(a)
        yield n

    for node in region.body:
        yield from go(node)


def check_fold_step_supported(step: Region) -> None:
    """PASS 1 (owner-ruled): is this step a shape the fold machinery
    handles? Anything else refuses NOW, with the reason — never
    mid-derivation."""
    for n in walk_region(step):
        if n.op not in _FOLD_STEP_SUPPORTED:
            raise TypeError(
                f"tl.fold step contains {n.op!r}, which the fold adjoint does not support yet "
                f"(supported: {', '.join(sorted(_FOLD_STEP_SUPPORTED))}) — the adjoint derives "
                f"per-op through the one derivative table, and {n.op!r} has no region rule here"
            )
        if n.op == "tl.pointwise":
            f = dict(n.attrs)["f"]
            if f not in TABLE:
                raise TypeError(
                    f"tl.fold step applies marker {f!r}, which has no row in the derivative "
                    f"table — the table grows only when a primitive joins the core"
                )


# --- the VJP: region-in, region-out ------------------------------------------


def _splice_tl(b, tree):
    """A table slope tree -> dialect IR: Prim -> tl.pointwise, schema Const
    -> a scalar core.const; a dsl Node leaf (a primal operand) rides as-is."""
    if isinstance(tree, Prim):
        return b.emit("tl.pointwise", *(_splice_tl(b, a) for a in tree.args), f=tree.op)
    if isinstance(tree, Const):
        return b.emit("core.const", type=f64, value=float(tree.value))
    return tree


def derive_step_vjp(step: Region, ops=None) -> Region:
    """The adjoint of a fold step, AS A REGION: params (state, element,
    upstream state-adjoint); recomputes the forward inside itself and
    yields d(state). The element's adjoint must be gradient-free (the
    dropout-mask discipline) — asserted."""
    check_fold_step_supported(step)  # pass 1, always
    b = Builder(ops or {**CORE_OPS, **TL_OPS})
    s_t, m_t = (p.type for p in step.params)
    p_s, p_m, p_ds = b.param(("v", 0), s_t), b.param(("v", 1), m_t), b.param(("v", 2), s_t)
    sub_env = {id(step.params[0]): p_s, id(step.params[1]): p_m}
    order: list = []

    def sub(n):
        if id(n) in sub_env:
            return sub_env[id(n)]
        if n.op == "core.const":
            made = b.emit("core.const", type=n.type, value=dict(n.attrs)["value"])
        else:
            made = b.emit(n.op, *(sub(a) for a in n.args), **dict(n.attrs))
        sub_env[id(n)] = made
        order.append(made)
        return made

    out = sub(step.body[-1].args[0])
    adj: dict[int, object] = {id(out): p_ds}
    for node in reversed(order):
        a = adj.get(id(node))
        if a is None or node.op != "tl.pointwise":
            continue
        rules = TABLE[dict(node.attrs)["f"]]
        for rule, operand in zip(rules, node.args):
            if rule is None or operand.op == "core.const":
                continue
            term = b.emit("tl.pointwise", _splice_tl(b, rule(*node.args)), a, f="mul")
            prev = adj.get(id(operand))
            adj[id(operand)] = term if prev is None else b.emit("tl.pointwise", prev, term, f="add")
    if adj.get(id(p_m)) is not None:
        raise TypeError("derive_step_vjp: the element adjoint must be gradient-free in this slice")
    ds = adj.get(id(p_s))
    if ds is None:
        raise TypeError("derive_step_vjp: the state never reaches the output")
    return Region(params=(p_s, p_m, p_ds), body=(b.emit("core.yield", ds),))


# --- the evaluation column (ir.run's successor for this dialect) -------------


def run_region(region: Region, values: list, uniforms: dict | None = None):
    """Evaluate a dialect region over tl Tensors — fields slice at ABSOLUTE
    coordinates, so closed-form random fields regenerate exactly. Stores
    write through the target's buffer (the ONE effect, token-ordered);
    ``uniforms`` binds per-launch scalar slots (unmarked captures are DATA
    — the literal doctrine, 240 C4.2b)."""
    memo: dict[int, object] = {}
    by_param = {id(p): v for p, v in zip(region.params, values)}

    def ev(n):
        if id(n) not in memo:
            memo[id(n)] = _ev(n)
        return memo[id(n)]

    def _ev(n):
        attrs = dict(n.attrs)
        if id(n) in by_param:
            return by_param[id(n)]
        if n.op == "core.const":
            return attrs["value"]
        if n.op == "core.yield":
            return ev(n.args[0])
        if n.op == "tl.token":
            return Token()
        if n.op == "tl.uniform":
            return (uniforms or {})[attrs["name"]]
        if n.op == "tl.iota":
            return _eager_iota(ev(n.args[0]), attrs["name"])
        if n.op == "tl.pointwise":
            ops_v = [ev(a) for a in n.args]
            ref = next(v for v in ops_v if isinstance(v, Tensor))
            ops_v = [v if isinstance(v, Tensor) else const_like(ref, float(v)) for v in ops_v]
            return _eager_pw(pw_marker(attrs["f"]), *ops_v)
        if n.op == "tl.store":
            tok, dst, val = n.args
            return _store(ev(tok), ev(dst), ev(val))
        if n.op == "tl.fold":
            init, src = ev(n.args[0]), ev(n.args[1])
            dim = attrs["dim"]
            lo, hi = next((d[1], d[2]) for d in n.args[1].type.dims if d[0] == dim)
            s = init
            for q in range(lo, hi):
                s = run_region(n.regions[0], [s, src.select(**{dim: q})])
            return s
        raise AssertionError(f"run_region: unexpected op {n.op!r}")

    return ev(region.body[-1])


# --- schedules: EVALUATION STRATEGIES over the same regions (never IR) -------


def fold_grad(step: Region, vjp: Region, init, src, dim: str, *, slots: int | None = None):
    """d(sum(final state))/d(init) — store-all when ``slots`` is None, else
    segment-checkpoint revolve: recompute each segment from its checkpoint
    during the backward, RE-SELECTING elements at absolute coordinates
    (the recompute theorem's mechanism; bit-identical by construction)."""
    src_dims = [(d.name, d.start, d.stop) for d in src.layout.dims]
    lo, hi = next((s, e) for (name, s, e) in src_dims if name == dim)
    stride = hi - lo if slots is None else -(-(hi - lo) // slots)
    checkpoints = {lo: init}
    s = init
    for q in range(lo, hi):
        if q != lo and (q - lo) % stride == 0:
            checkpoints[q] = s
        s = run_region(step, [s, src.select(**{dim: q})])
    ds = const_like(s, 1.0)
    starts = sorted(checkpoints)
    for seg in reversed(range(len(starts))):
        base = starts[seg]
        end = starts[seg + 1] if seg + 1 < len(starts) else hi
        states = [checkpoints[base]]
        for q in range(base, end):  # the recomputation the theorem is about
            states.append(run_region(step, [states[-1], src.select(**{dim: q})]))
        for q in reversed(range(base, end)):
            ds = run_region(vjp, [states[q - base], src.select(**{dim: q}), ds])
    return ds
