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
from pdum.tl.markers import Marker, maximum  # noqa: F401 — bare in the kernel body

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
