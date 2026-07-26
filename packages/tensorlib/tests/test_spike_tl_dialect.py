"""C3a SPIKE (240): one slice of the tl IR as a dsl DIALECT — side path.

Self-contained ON PURPOSE: the dialect (TensorType/TokenType +
tl.iota/tl.pointwise/tl.store as OpDefs with real type rules), a tl rule
pack layered over the dsl value pack, and an evaluation visitor all live
in THIS file; nothing under src/ changes. The owner reviews this file as
the C3a decision gate.

What it demonstrates:
- tensor-typed SSA over dsl Node/Region: the layout lattice IS the type,
  so tl's alignment law becomes an ordinary dsl TYPE RULE (misalignment
  refuses at emission with source locations — the dsl machinery doing
  tl's checking);
- the dialect-region question in miniature: ONE lowerer, the tl pack
  layered over the value pack, with TENSOR-TYPEDNESS selecting the path
  (type-directed region detection, no annotations) — the base pack
  serves scalar subtrees unchanged;
- the differential: the same kernel body lowered by today's
  _KernelLowerer + ir.run and by the dsl Lowerer + this evaluator agree
  bit-for-bit;
- fp-keyed IR: two lowerings of the same body produce content-identical
  Regions (region.key), the cache-efficiency premise.

Spike scope cuts (C4 design items, recorded in 240): TensorType carries
the dim lattice only (charts/labels/levels punted); stores assume the
full aligned lattice (index checking punted); kernels need a no-return
body driver (driven manually here, the build_pipe precedent); fn-valued
args and reduce/fold are OUT (fold is C3b).
"""

import ast as pyast
from dataclasses import dataclass

import numpy as np
import pytest
from pdum.dsl.capture import make_handle
from pdum.dsl.ir import Region
from pdum.dsl.lower import Lowerer, check_coherence
from pdum.dsl.markers import pw
from pdum.dsl.ops import CORE_OPS, PURE, OpDef
from pdum.dsl.registry import DEFAULT
from pdum.dsl.types import Type
from pdum.dsl.value import LOWER_RULES, _assign, _binop, _call
from pdum.tl import Tensor, compute, thread_idx
from pdum.tl.markers import Marker, maximum, tanh, where  # noqa: F401 — bare in bodies, BOTH engines

# --- the dialect: types ------------------------------------------------------


@dataclass(frozen=True)
class TensorType(Type):
    """A tensor-typed SSA value: the layout lattice IS the type."""

    dims: tuple  # ((name, start, stop), ...)


@dataclass(frozen=True)
class TokenType(Type):
    """The ordering token — stores consume and produce it (dataflow)."""


def _tt(t: Tensor) -> TensorType:
    return TensorType(tuple((d.name, d.start, d.stop) for d in t.layout.dims))


# --- the dialect: ops, typed by rules (tl's alignment law AS a type rule) ----


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


TL_OPS = {
    "tl.token": OpDef("tl.token", lambda a, at, r: TokenType(), PURE),
    "tl.iota": OpDef("tl.iota", _r_iota, PURE),
    "tl.pointwise": OpDef("tl.pointwise", _r_pointwise, PURE),
    "tl.store": OpDef("tl.store", _r_store, PURE),  # the effect rides the token
}

_BIN_MARKER = {pyast.Add: "add", pyast.Sub: "sub", pyast.Mult: "mul", pyast.Div: "div"}


# --- the tl rule pack: layered over the value pack, type-directed ------------


def _tl_call(ctx, node):
    if isinstance(node.func, pyast.Name):
        if node.func.id == "pointwise":  # the S.1 STEP-tier spelling
            marker = ctx.context.get("registry").overloads.get(node.args[0].id)
            rest = [ctx.lower(a) for a in node.args[1:]]
            return ctx.emit("tl.pointwise", *rest, node=node, f=marker.name)
        if node.func.id == "const_like":  # scalar broadcast: the const IS the operand
            return ctx.lower(node.args[1])
        if node.func.id == "thread_idx":
            lattice = ctx.root.params[-1]  # the writable target (S.3 convention)
            names = [c.value for c in node.args]
            out = tuple(ctx.emit("tl.iota", lattice, node=node, name=n) for n in names)
            return out if len(out) > 1 else out[0]
        impl = ctx.context.get("registry").overloads.get(node.func.id)
        if isinstance(impl, Marker):
            args = [ctx.lower(a) for a in node.args]
            if any(isinstance(a.type, TensorType) for a in args):
                return ctx.emit("tl.pointwise", *args, node=node, f=impl.name)
    return _call(ctx, node)  # everything else: the base value pack


def _tl_binop(ctx, node):
    lhs, rhs = ctx.lower(node.left), ctx.lower(node.right)
    if isinstance(lhs.type, TensorType) or isinstance(rhs.type, TensorType):
        f = _BIN_MARKER.get(type(node.op))
        if f is None:
            raise TypeError(f"operator {type(node.op).__name__} has no tensor pointwise")
        return ctx.emit("tl.pointwise", lhs, rhs, node=node, f=f)
    return _binop(ctx, node)


def _tl_assign(ctx, node):
    tgt = node.targets[0]
    if isinstance(tgt, pyast.Subscript):  # the store (full aligned lattice, spike scope)
        target, value = ctx.lower(tgt.value), ctx.lower(node.value)
        tok = ctx.context.get("tl.token") or ctx.emit("tl.token", node=node)
        ctx.context["tl.token"] = ctx.emit("tl.store", tok, target, value, node=node)
        return None
    value = ctx.lower(node.value) if not isinstance(node.value, pyast.Call) else _tl_call(ctx, node.value)
    if isinstance(tgt, pyast.Tuple) and isinstance(value, tuple):  # y, x = thread_idx(...)
        for e, v in zip(tgt.elts, value):
            ctx.locals[e.id] = v
        return None
    if isinstance(tgt, pyast.Name):
        ctx.locals[tgt.id] = value
        return None
    return _assign(ctx, node)


TL_RULES = {**LOWER_RULES, pyast.Assign: _tl_assign, pyast.BinOp: _tl_binop, pyast.Call: _tl_call}


def lower_kernel_body(fn, arg_tts):
    """Drive the DSL Lowerer over a no-return kernel body (the build_pipe
    manual-driving precedent); yield the final ordering token."""
    handle = make_handle(fn, "device")
    check_coherence(handle)
    ops = {**CORE_OPS, **TL_OPS}
    ctx = Lowerer(handle, TL_RULES, ops, {}, context={"registry": DEFAULT})
    params = tuple(ctx.builder.param(i, t) for i, t in enumerate(arg_tts))
    ctx.params = params
    names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    ctx.locals.update(zip(names, params))
    tree = next(n for n in pyast.parse(handle.snapshot.text).body if isinstance(n, pyast.FunctionDef))
    for stmt in tree.body:
        ctx.lower(stmt)
    tok = ctx.context.get("tl.token")
    assert tok is not None, "a kernel's effect is its stores"
    return Region(params=params, body=(ctx.builder.emit("core.yield", tok),))


# --- evaluation: the visitor (ir.run as a COLUMN over the dialect) -----------


def evaluate(region: Region, arrays: list) -> None:
    memo: dict[int, object] = {}

    def ev(n):
        if id(n) not in memo:
            memo[id(n)] = _ev(n)
        return memo[id(n)]

    def _ev(n):
        attrs = dict(n.attrs)
        if n.op == "core.param":
            return arrays[attrs["index"]]
        if n.op == "core.const":
            return attrs["value"]
        if n.op == "tl.token":
            return None
        if n.op == "tl.iota":
            dims = n.type.dims
            axis = [d[0] for d in dims].index(attrs["name"])
            shape = tuple(stop - start for (_, start, stop) in dims)
            ar = np.arange(dims[axis][1], dims[axis][2], dtype=np.float64)
            return np.broadcast_to(ar.reshape([-1 if i == axis else 1 for i in range(len(dims))]), shape)
        if n.op == "tl.pointwise":
            return getattr(pw, attrs["f"]).fn(*[ev(a) for a in n.args])
        if n.op == "tl.store":
            tok, dst, val = n.args
            ev(tok)  # the ordering edge
            ev(dst)[...] = ev(val)
            return None
        if n.op == "core.yield":
            return ev(n.args[0])
        raise AssertionError(f"spike evaluator: unexpected op {n.op!r}")

    for node in region.body:
        ev(node)


# --- the kernel under differential -------------------------------------------


@compute
def spike_kernel(img):
    y, x = thread_idx("y", "x")
    v = maximum(y, x) * 0.25 + (y - x) * 0.5
    img[y, x] = v * v + 1.5


def test_the_dialect_differential_is_bit_identical():
    """The same body, two machines: today's _KernelLowerer + ir.run vs the
    dsl Lowerer with the tl pack + the spike evaluator — identical bytes."""
    a = Tensor.from_numpy(np.zeros((5, 7)), ("y", "x"))
    spike_kernel(a)  # the incumbent tl machinery
    b = np.zeros((5, 7))
    region = lower_kernel_body(spike_kernel.fn, (_tt(a),))
    evaluate(region, [b])
    np.testing.assert_array_equal(a.to_numpy(), b)


def test_alignment_is_a_type_rule_and_refuses_with_locations():
    """tl's alignment law enforced by the dsl TYPE machinery: a misaligned
    store refuses AT EMISSION, with the source point in the message."""

    def bad(img, other):
        y, x = thread_idx("y", "x")
        img[y, x] = other * 1.0  # a different lattice: misaligned

    img = Tensor.from_numpy(np.zeros((4, 4)), ("y", "x"))
    other = Tensor.from_numpy(np.zeros((3, 3)), ("y", "x"))
    with pytest.raises(TypeError, match=r"aligned to the target.*\[.*test_spike_tl_dialect"):
        lower_kernel_body(bad, (_tt(img), _tt(other)))


def test_lowering_is_content_keyed():
    """fp-keyed IR: two independent lowerings of the same body produce
    content-identical Regions — the cache-efficiency premise holds."""
    a = Tensor.from_numpy(np.zeros((5, 7)), ("y", "x"))
    r1 = lower_kernel_body(spike_kernel.fn, (_tt(a),))
    r2 = lower_kernel_body(spike_kernel.fn, (_tt(a),))
    assert r1.key == r2.key


# ============================================================================
# C3b — fold, the hard part (240: owner-ruled to stay IN the spike).
#
# The TWO-PASS mechanism (owner-proposed): pass 1 walks the step region and
# refuses — with the reason — anything the fold machinery cannot handle;
# pass 2 does the work. The VJP derivation is REGION-IN/REGION-OUT (the
# generated-function model): it consumes the step region and emits the
# adjoint region over the same dialect, slopes spliced from THE one table.
# Schedules (store-all vs revolve) are EVALUATION STRATEGIES over the same
# two regions — the b3 finding for the schedule-in-IR question.
# ============================================================================

from pdum.dsl.derivative import TABLE, Const, Prim  # noqa: E402 — the C3b section's own imports
from pdum.dsl.ir import Builder  # noqa: E402 — the C3b section's own imports
from pdum.dsl.types import f64  # noqa: E402 — the C3b section's own imports
from pdum.dsl.value import _compare  # noqa: E402 — the C3b section's own imports
from pdum.tl.compute import const_like, pointwise  # noqa: E402 — the C3b section's own imports
from pdum.tl.compute import pointwise as _eager_pw  # noqa: E402 — the C3b section's own imports


def _minus_dim(tt: TensorType, name: str) -> TensorType:
    return TensorType(tuple(d for d in tt.dims if d[0] != name))


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


TL_OPS["tl.fold"] = OpDef("tl.fold", _r_fold, PURE, nregions=1)

_CMP_MARKER = {pyast.Lt: "lt", pyast.Gt: "gt", pyast.LtE: "le", pyast.GtE: "ge"}


def _tl_compare(ctx, node):
    if len(node.ops) == 1 and type(node.ops[0]) in _CMP_MARKER:
        lhs, rhs = ctx.lower(node.left), ctx.lower(node.comparators[0])
        if isinstance(lhs.type, TensorType) or isinstance(rhs.type, TensorType):
            return ctx.emit("tl.pointwise", lhs, rhs, node=node, f=_CMP_MARKER[type(node.ops[0])])
    return _compare(ctx, node)


TL_RULES[pyast.Compare] = _tl_compare


# --- pass 1: the supported-shape check (refuse with the reason) --------------

_FOLD_STEP_SUPPORTED = {"tl.pointwise", "core.const", "core.param", "core.yield"}


def _walk(region: Region):
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
    """PASS 1 of the two-pass mechanism: is this step a shape we handle?
    Anything else refuses NOW, with the reason — never mid-derivation."""
    for n in _walk(step):
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


# --- lowering a step function (it HAS a return, unlike a kernel) -------------


def lower_step(fn, state_tt: TensorType, elem_tt: TensorType) -> Region:
    handle = make_handle(fn, "device")
    check_coherence(handle)
    ops = {**CORE_OPS, **TL_OPS}
    ctx = Lowerer(handle, TL_RULES, ops, {}, context={"registry": DEFAULT})
    binders = (ctx.builder.param(("st", 0), state_tt), ctx.builder.param(("st", 1), elem_tt))
    ctx.params = binders
    names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    ctx.locals.update(zip(names, binders))
    result = ctx.run_body()
    return Region(params=binders, body=(ctx.builder.emit("core.yield", result),))


# --- pass 2a: the VJP derivation — region-in, region-out ---------------------


def _splice_tl(b, tree, args_env):
    """A table slope tree -> dialect IR: Prim -> tl.pointwise, schema Const ->
    a scalar core.const; a dsl Node leaf (a primal operand) splices as-is."""
    if isinstance(tree, Prim):
        return b.emit("tl.pointwise", *(_splice_tl(b, a, args_env) for a in tree.args), f=tree.op)
    if isinstance(tree, Const):
        return b.emit("core.const", type=f64, value=float(tree.value))
    return tree  # a primal operand node (already substituted)


def derive_step_vjp(step: Region, ops) -> Region:
    """The adjoint of the step, AS A REGION: params (state, element,
    upstream state-adjoint); recomputes the forward inside itself (no saved
    intermediates — spike scope) and yields d(state). The element's adjoint
    must be gradient-free here (the dropout mask discipline) — asserted."""
    check_fold_step_supported(step)  # pass 1, always
    b = Builder(ops)
    s_t, m_t = (p.type for p in step.params)
    p_s, p_m, p_ds = b.param(("v", 0), s_t), b.param(("v", 1), m_t), b.param(("v", 2), s_t)
    sub_env = {id(step.params[0]): p_s, id(step.params[1]): p_m}
    order: list = []

    def sub(n):
        if id(n) in sub_env:
            return sub_env[id(n)]
        made = (
            b.emit(n.op, *(sub(a) for a in n.args), **dict(n.attrs))
            if n.op != "core.const"
            else b.emit("core.const", type=n.type, value=dict(n.attrs)["value"])
        )
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
            slope = _splice_tl(b, rule(*node.args), sub_env)
            term = b.emit("tl.pointwise", slope, a, f="mul")
            prev = adj.get(id(operand))
            adj[id(operand)] = term if prev is None else b.emit("tl.pointwise", prev, term, f="add")
    assert adj.get(id(p_m)) is None, "the element adjoint must be gradient-free in this spike"
    ds = adj.get(id(p_s))
    assert ds is not None, "the state must reach the output"
    return Region(params=(p_s, p_m, p_ds), body=(b.emit("core.yield", ds),))


# --- evaluation over TENSORS (fields slice at ABSOLUTE coordinates) ----------


def tensor_eval(region: Region, values: list):
    memo: dict[int, object] = {}
    by_param = {id(p): v for p, v in zip(region.params, values)}

    def ev(n):
        if id(n) in memo:
            return memo[id(n)]
        memo[id(n)] = r = _ev(n)
        return r

    def _ev(n):
        attrs = dict(n.attrs)
        if id(n) in by_param:
            return by_param[id(n)]
        if n.op == "core.const":
            return attrs["value"]
        if n.op == "core.yield":
            return ev(n.args[0])
        if n.op == "tl.pointwise":
            ops_v = [ev(a) for a in n.args]
            ref = next(v for v in ops_v if isinstance(v, Tensor))
            ops_v = [v if isinstance(v, Tensor) else const_like(ref, float(v)) for v in ops_v]
            return _eager_pw(getattr(pw, attrs["f"]), *ops_v)
        if n.op == "tl.fold":
            init, src = ev(n.args[0]), ev(n.args[1])
            dim = attrs["dim"]
            (lo, hi) = next((d[1], d[2]) for d in n.args[1].type.dims if d[0] == dim)
            s = init
            for q in range(lo, hi):
                m = src.select(**{dim: q})  # ABSOLUTE coordinate: fields regenerate
                s = tensor_eval(n.regions[0], [s, m])
            return s
        raise AssertionError(f"tensor_eval: unexpected op {n.op!r}")

    return ev(region.body[-1])


# --- pass 2b: the two SCHEDULES, as evaluation strategies --------------------


def fold_grad_store_all(step: Region, vjp: Region, init, src, dim: str, lo: int, hi: int):
    states = [init]
    for q in range(lo, hi):
        states.append(tensor_eval(step, [states[-1], src.select(**{dim: q})]))
    ds = const_like(states[-1], 1.0)  # d(sum(s_final))/d(s_final)
    for q in reversed(range(lo, hi)):
        ds = tensor_eval(vjp, [states[q - lo], src.select(**{dim: q}), ds])
    return ds


def fold_grad_revolve(step: Region, vjp: Region, init, src, dim: str, lo: int, hi: int, slots: int):
    """Segment-checkpoint revolve: keep ``slots`` checkpoints, RECOMPUTE each
    segment's states from its checkpoint during the backward — re-selecting
    the elements at their ABSOLUTE coordinates (the field regenerates)."""
    total = hi - lo
    stride = -(-total // slots)
    checkpoints = {lo: init}
    s = init
    for q in range(lo, hi):
        if q != lo and (q - lo) % stride == 0:
            checkpoints[q] = s
        s = tensor_eval(step, [s, src.select(**{dim: q})])
    ds = const_like(s, 1.0)
    starts = sorted(checkpoints)
    for seg in reversed(range(len(starts))):
        base = starts[seg]
        end = starts[seg + 1] if seg + 1 < len(starts) else hi
        states = [checkpoints[base]]
        for q in range(base, end):  # the recomputation the theorem is ABOUT
            states.append(tensor_eval(step, [states[-1], src.select(**{dim: q})]))
        for q in reversed(range(base, end)):
            ds = tensor_eval(vjp, [states[q - base], src.select(**{dim: q}), ds])
    return ds


# --- the C3b differentials ---------------------------------------------------


def _theorem_step(s, m):
    # ONE source, two engines — spelled in the ratified S.1 STEP style
    # (step bodies are the assemblage tier: pointwise is spelled; the
    # kernel tier's bare-marker style is a DIFFERENT dialect leaf):
    kept = pointwise(where, m < 0.4, const_like(s, 0.0), s)
    return s * 0.8 + pointwise(tanh, kept) * 0.3


def _theorem_setup():
    from pdum.tl import fold_in, uniform

    N, TM = 4, 6
    rng = np.random.default_rng(2)
    s0 = Tensor.from_numpy(rng.standard_normal(N), ("x",))
    lattice = Tensor.from_numpy(np.zeros((TM, N)), ("tm", "x"))
    mask = uniform(fold_in(5, "train.drop"), lattice.layout)  # zero-memory FIELD
    return s0, mask, TM


def test_b1_fold_forward_differential():
    """The fold forward through the dialect ≡ the tl fold under ir.run —
    the same single-source step function, two engines, identical bytes."""
    from pdum.tl.ir import Instr, Program, run
    from pdum.tl.lifting import lift_step

    s0, mask, TM = _theorem_setup()
    step = lower_step(_theorem_step, _tt(s0), _minus_dim(_tt(Tensor.from_numpy(np.zeros((TM, 4)), ("tm", "x"))), "tm"))
    check_fold_step_supported(step)  # pass 1
    b = Builder({**CORE_OPS, **TL_OPS})
    p_s, p_src = b.param(0, _tt(s0)), b.param(1, TensorType(tuple((d.name, d.start, d.stop) for d in mask.layout.dims)))
    fold = b.emit("tl.fold", p_s, p_src, regions=(step,), dim="tm")
    region = Region(params=(p_s, p_src), body=(b.emit("core.yield", fold),))
    got = tensor_eval(region, [s0, mask])
    # the incumbent: lift the SAME step, fold under ir.run
    ls = lift_step(_theorem_step, s=s0.layout, m=Tensor.from_numpy(np.zeros(4), ("x",)).layout)
    fold_params = {
        "step": ls.program,
        "dim": "tm",
        "state": ("s",),
        "element": ("m",),
        "carry": {"s": ls.outputs[0]},
        "out": ("final", ls.outputs[0]),
    }
    prog = Program(
        (
            Instr("s0", "input", (), {}),
            Instr("mask", "input", (), {}),
            Instr("sf", "fold", ("s0", "mask"), fold_params),
        )
    )
    want = run(prog, {"s0": s0, "mask": mask})["sf"]
    np.testing.assert_array_equal(got.to_numpy(order=("x",)), want.to_numpy(order=("x",)))


def test_b2_store_all_adjoint_differential():
    """The region-derived VJP under store-all ≡ tl's autodiff gradient —
    same step, same field, gradient wrt the initial state."""
    from pdum.tl.autodiff import grad
    from pdum.tl.ir import Instr, Program, run
    from pdum.tl.lifting import lift_step

    s0, mask, TM = _theorem_setup()
    elem_tt = TensorType(tuple((d.name, d.start, d.stop) for d in s0.layout.dims))
    step = lower_step(_theorem_step, _tt(s0), elem_tt)
    vjp = derive_step_vjp(step, {**CORE_OPS, **TL_OPS})
    got = fold_grad_store_all(step, vjp, s0, mask, "tm", 0, TM)
    ls = lift_step(_theorem_step, s=s0.layout, m=Tensor.from_numpy(np.zeros(4), ("x",)).layout)
    fold_params = {
        "step": ls.program,
        "dim": "tm",
        "state": ("s",),
        "element": ("m",),
        "carry": {"s": ls.outputs[0]},
        "out": ("final", ls.outputs[0]),
    }
    prog = Program(
        (
            Instr("s0", "input", (), {}),
            Instr("mask", "input", (), {}),
            Instr("sf", "fold", ("s0", "mask"), fold_params),
            Instr("zloss", "reduce", ("sf",), {"f": "sum", "dims": ("x",)}),
        )
    )
    joint, grads = grad(prog, "zloss", {"s0": s0, "mask": mask}, fold_segments=1)
    want = run(joint, {"s0": s0, "mask": mask})[grads["s0"]]
    np.testing.assert_allclose(got.to_numpy(order=("x",)), want.to_numpy(order=("x",)), rtol=1e-12)


def test_b3_revolve_is_bit_identical_with_the_field_on():
    """THE THEOREM through the dialect: revolve (recompute from checkpoints,
    re-selecting the zero-memory field at absolute coordinates) and
    store-all produce BIT-IDENTICAL gradients — dropout on, no mask stored."""
    from pdum.tl.random import RandomBuffer

    s0, mask, TM = _theorem_setup()
    assert isinstance(mask.buffer, RandomBuffer) and mask.buffer.data is None  # zero bytes
    elem_tt = TensorType(tuple((d.name, d.start, d.stop) for d in s0.layout.dims))
    step = lower_step(_theorem_step, _tt(s0), elem_tt)
    vjp = derive_step_vjp(step, {**CORE_OPS, **TL_OPS})
    g_all = fold_grad_store_all(step, vjp, s0, mask, "tm", 0, TM)
    g_rev = fold_grad_revolve(step, vjp, s0, mask, "tm", 0, TM, slots=2)
    np.testing.assert_array_equal(g_rev.to_numpy(order=("x",)), g_all.to_numpy(order=("x",)))
    assert not np.array_equal(g_all.to_numpy(order=("x",)), np.zeros(4))


def test_b_pass1_refuses_unsupported_shapes_with_the_reason():
    """The two-pass mechanism's first pass: an unsupported op in a step
    refuses NAMING the op, the supported set, and why."""
    b = Builder({**CORE_OPS, **TL_OPS})
    tt = TensorType((("x", 0, 4),))
    p = b.param(("st", 0), tt)
    rogue = b.emit("tl.iota", p, name="x")
    bad = Region(params=(p, b.param(("st", 1), tt)), body=(b.emit("core.yield", rogue),))
    with pytest.raises(TypeError, match=r"contains 'tl.iota'.*does not support yet.*no region rule"):
        check_fold_step_supported(bad)
