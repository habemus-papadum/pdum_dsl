"""Tensor-typed lifting — the shared expression syntax's tensor half (S.2).

A fold step is a plain function over tensor-typed parameters; lifting lowers
it to a step Program by TYPE-DIRECTED inspection of the AST:

- unannotated parameters are TENSOR-typed: they become ``input`` instrs, and
  every operation over them emits IR (operators → pointwise markers, method
  calls → layout ops, layout shadows tracked through ``ir.infer_instr``);
- parameters annotated ``n: Literal[int]`` are STRUCTURAL (200 §1.5): bound
  to build-time values and evaluated on the host — ``n - 1`` in a slice
  extent is ordinary arithmetic at lift time;
- sub-expressions with no tensor in them (captured charts, Fractions,
  helper-function results) evaluate on the host; a helper called on tensor
  arguments inlines — its body lowers under the same rules.

Structural slots (slice/pad extents, shift deltas, method keywords) accept
only host values: a tensor reaching one refuses, naming the annotation fix.
Straight-line enforced at lowering, exactly like marker bodies.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from pdum.dsl.types import LiteralAnnotation

from .build import Build
from .ir import Instr, Program, infer_instr
from .layout import Layout
from .producer import _captured, _fn_ast
from .tensor import Tensor

_STRUCTURAL_SLOT = (
    "{what} is a STRUCTURAL slot: a runtime tensor cannot shape the lattice — "
    "pass a build-time value (annotate the parameter: `n: Literal[int]`, 200 §1.5)"
)

_CMP = {ast.Eq: "eq", ast.NotEq: "ne", ast.LtE: "le", ast.Lt: "lt", ast.GtE: "ge", ast.Gt: "gt"}
_BIN = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}

# method name -> (op, packer(args, kwargs) -> params); every packer input is
# host-evaluated before packing (the structural-slot discipline)
_METHODS = {
    "slice": ("slice", lambda a, kw: {"ranges": kw}),
    "select": ("select", lambda a, kw: {"coords": kw}),
    "shift": ("shift", lambda a, kw: {"deltas": kw}),
    "rename": ("rename", lambda a, kw: {"mapping": kw}),
    "repeat": ("repeat", lambda a, kw: {"name": a[0], "extent": a[1], **kw}),
    "flip": ("flip", lambda a, kw: {"name": a[0]}),
    "split": ("split", lambda a, kw: {"name": a[0], "parts": kw}),
    "merge": ("merge", lambda a, kw: {"parts": tuple(a[0]), "name": a[1], **kw}),
    "diagonal": ("diagonal", lambda a, kw: {"parts": tuple(a[0]), "name": a[1], **kw}),
    "window": ("window", lambda a, kw: dict(zip(("name", "k_name", "k", "dilation"), a)) | kw),
    "decimate": ("decimate", lambda a, kw: dict(zip(("name", "factor", "phase"), a)) | kw),
    "pad": ("pad", lambda a, kw: {"fill": a[0] if a else kw.pop("fill"), "extents": kw}),
    "stencil": ("stencil", lambda a, kw: dict(zip(("name", "k", "k_name", "fill", "dilation"), a)) | kw),
    "strip_charts": ("strip_charts", lambda a, kw: {}),
    "with_charts": ("with_charts", lambda a, kw: {"charts": kw}),
    "with_labels": ("with_labels", lambda a, kw: {"labels": kw}),
    "bind": ("bind", lambda a, kw: {"levels": kw}),
    "simplify": ("simplify", lambda a, kw: {}),
    "with_value_units": ("with_value_units", lambda a, kw: {"value_units": a[0]}),
}


@dataclass(frozen=True)
class _T:
    """A tensor-typed intermediate: its SSA var and its layout shadow."""

    var: str
    shadow: object


@dataclass(frozen=True)
class _Intrinsic:
    """An S.1 vocabulary function: meaningful only inside a lowered body."""

    name: str

    def __call__(self, *args, **kwargs):
        raise TypeError(
            f"{self.name} is assemblage vocabulary — it lowers by inspection "
            f"inside a unit or step body; there is nothing to call"
        )


contract = _Intrinsic("contract")
iota_of = _Intrinsic("iota_of")
const_like = _Intrinsic("const_like")
reduce_over = _Intrinsic("reduce_over")


def _holds_tensor(v) -> bool:
    if isinstance(v, _T):
        return True
    if isinstance(v, (tuple, list)):
        return any(_holds_tensor(x) for x in v)
    if isinstance(v, dict):
        return any(_holds_tensor(x) for x in v.values())
    return False


@dataclass(frozen=True)
class LiftedStep:
    program: Program
    inputs: tuple[str, ...]  # tensor parameter names, in signature order
    outputs: tuple[str, ...]  # SSA vars of the returned tensors, in order


def lift_step(fn, **bindings) -> LiftedStep:
    """Lift ``fn`` to a step Program. Bind every tensor parameter to a
    Layout (or Tensor, whose layout is taken) and every ``Literal``-annotated
    parameter to a build-time value."""
    tree = _fn_ast(fn)
    anns = getattr(fn, "__annotations__", {})
    lifter = _Lifter(_captured(fn))
    params = [a.arg for a in tree.args.args]
    inputs = []
    for p in params:
        if p not in bindings:
            raise ValueError(f"parameter {p!r} is unbound — lift_step binds every parameter by name")
        v = bindings.pop(p)
        ann = anns.get(p)
        if isinstance(ann, str):  # `from __future__ import annotations` in the def site
            ann = eval(ann, fn.__globals__)  # noqa: S307 — the def site's own namespace
        if isinstance(ann, LiteralAnnotation):
            if not isinstance(v, ann.base):
                raise ValueError(f"parameter {p!r} is Literal[{ann.base.__name__}]; got {v!r}")
            lifter.env[p] = v
            continue
        if isinstance(v, Tensor):
            v = v.layout
        if not isinstance(v, Layout):
            raise ValueError(
                f"parameter {p!r} is tensor-typed (unannotated) but received {v!r} — "
                f"structural parameters declare themselves: annotate `{p}: Literal[{type(v).__name__}]`"
            )
        lifter.b.input(p)
        lifter.shadows[p] = v
        lifter.env[p] = _T(p, v)
        inputs.append(p)
    if bindings:
        raise ValueError(f"unknown parameters bound: {sorted(bindings)}")
    outs = lifter.run_body(tree)
    return LiftedStep(lifter.b.program(), tuple(inputs), outs)


class _Lifter:
    def __init__(self, env: dict):
        self.env = env
        self.b = Build()
        self.shadows: dict[str, object] = {}

    # ---- emission --------------------------------------------------------

    def adopt(self, v):
        """Hook: convert a captured value at name resolution (the unit
        lowerer turns captured Params into named inputs here)."""
        return v

    def child(self, env: dict) -> "_Lifter":
        """A same-kind lowerer over ``env`` sharing this one's program and
        name space — how helpers inline without losing the subclass."""
        inner = type(self)(env)
        inner.b, inner.shadows = self.b, self.shadows
        return inner

    def emit(self, op: str, operands: tuple[str, ...], hint: str, **params) -> _T:
        var = self.b.emit(op, operands, hint=hint, **params)
        self.shadows[var] = infer_instr(self.b.instrs[-1], self.shadows)
        return _T(var, self.shadows[var])

    def rebind(self, t: _T, name: str) -> _T:
        """Rename the JUST-emitted instr's var to the Python binding name
        (deduped through the same Namer) — nothing references it yet."""
        last = self.b.instrs[-1]
        if last.var != t.var or name in self.b.names:
            return t  # a re-bound existing var, or a taken name: keep the hint
        fresh = self.b.names.derive(name)
        self.b.instrs[-1] = Instr(fresh, last.op, last.operands, dict(last.params))
        self.shadows[fresh] = self.shadows.pop(t.var)
        return _T(fresh, self.shadows[fresh])

    def const_like(self, value, t: _T) -> _T:
        """A structural scalar broadcast over a tensor operand's lattice —
        charts/labels/placement restamped so alignment holds."""
        dims = tuple((d.name, (d.start, d.stop)) for d in t.shadow.dims)
        out = self.emit("const", (), "c", value=float(value), dims=dims)
        charts = {d.name: d.chart for d in t.shadow.dims if d.labels is None}
        labels = {d.name: d.labels for d in t.shadow.dims if d.labels is not None}
        if any(c is not None for c in charts.values()):
            out = self.emit("with_charts", (out.var,), "c", charts=charts)
        if labels:
            out = self.emit("with_labels", (out.var,), "c", labels=labels)
        levels = {d.name: d.level for d in t.shadow.dims}
        if any(lv is not None for lv in levels.values()):
            out = self.emit("bind", (out.var,), "c", levels=levels)
        return out

    def pointwise(self, f: str, *operands, hint: str | None = None) -> _T:
        ts = [o for o in operands if isinstance(o, _T)]
        ops = tuple(o.var if isinstance(o, _T) else self.const_like(o, ts[0]).var for o in operands)
        return self.emit("pointwise", ops, hint or f, f=f)

    # ---- the body --------------------------------------------------------

    def run_body(self, tree) -> tuple[str, ...]:
        if isinstance(tree, ast.Lambda):
            return self.returns(self.value(tree.body))
        for stmt in tree.body[:-1]:
            self.statement(stmt)
        last = tree.body[-1]
        if not isinstance(last, ast.Return) or last.value is None:
            raise ValueError("a step body must end in `return <tensor(s)>`")
        return self.returns(self.value(last.value))

    def returns(self, value) -> tuple[str, ...]:
        vals = value if isinstance(value, tuple) else (value,)
        if not all(isinstance(v, _T) for v in vals):
            raise ValueError(f"a step must return tensors, got {vals!r}")
        return tuple(v.var for v in vals)

    def statement(self, stmt) -> None:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], self.value(stmt.value)
            if isinstance(target, ast.Name):
                # SSA names come from Python binding names (S.1): the RHS's
                # final emission takes the binding name as its hint
                if isinstance(value, _T):
                    value = self.rebind(value, target.id)
                self.env[target.id] = value
                return
            if isinstance(target, ast.Tuple) and all(isinstance(e, ast.Name) for e in target.elts):
                if not isinstance(value, tuple) or len(value) != len(target.elts):
                    raise ValueError(f"cannot destructure {value!r} into {len(target.elts)} names")
                for e, p in zip(target.elts, value):
                    self.env[e.id] = p
                return
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            return  # a docstring
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            self.value(stmt.value)  # an effectful statement call (tap sites)
            return
        if isinstance(stmt, ast.FunctionDef):  # a local helper: bind it for later inlining
            raise ValueError("define step helpers OUTSIDE the step body; calls inline them")
        raise ValueError(
            f"step bodies are straight-line: a {type(stmt).__name__} statement cannot be "
            f"lowered — bounded control flow exists only in the value language (S.2)"
        )

    # ---- expressions -----------------------------------------------------

    def value(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise ValueError(f"unknown name {node.id!r} in a step body")
            return self.adopt(self.env[node.id])
        if isinstance(node, ast.Tuple):
            return tuple(self.value(e) for e in node.elts)
        if isinstance(node, ast.Dict):
            return {self.value(k): self.value(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.BinOp):
            lhs, rhs = self.value(node.left), self.value(node.right)
            op = _BIN.get(type(node.op))
            if isinstance(lhs, _T) or isinstance(rhs, _T):
                if op is None:
                    raise ValueError(f"operator {type(node.op).__name__} has no pointwise primitive")
                return self.pointwise(op, lhs, rhs)
            return _HOST_BIN[type(node.op)](lhs, rhs)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = self.value(node.operand)
            return self.pointwise("neg", v) if isinstance(v, _T) else -v
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            lhs, rhs = self.value(node.left), self.value(node.comparators[0])
            if isinstance(lhs, _T) or isinstance(rhs, _T):
                return self.pointwise(_CMP[type(node.ops[0])], lhs, rhs)
            return _HOST_CMP[type(node.ops[0])](lhs, rhs)
        if isinstance(node, ast.Attribute):
            base = self.value(node.value)
            if isinstance(base, _T):
                raise ValueError(f"tensors have no attribute access in step bodies (.{node.attr})")
            return getattr(base, node.attr)
        if isinstance(node, ast.Subscript):
            base, idx = self.value(node.value), self.value(node.slice)
            if isinstance(base, _T):
                raise ValueError("tensor subscripts do not exist here — use .slice()/.select()")
            return base[idx]
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, (ast.IfExp, ast.BoolOp, ast.GeneratorExp, ast.ListComp)):
            raise ValueError(
                "step bodies are straight-line: if/and/or cannot be lowered — "
                "use where(cond, a, b); the branch is data flow here"
            )
        raise ValueError(f"cannot lower a {type(node).__name__} in a step body")

    def kwargs_of(self, node: ast.Call) -> dict:
        out: dict = {}
        for kw in node.keywords:
            if kw.arg is None:  # a **splat: the dict is a host value
                out.update(self.value(kw.value))
            else:
                out[kw.arg] = self.value(kw.value)
        return out

    def call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):  # a method: tensor layout op, or host method
            base = self.value(node.func.value)
            args = [self.value(a) for a in node.args]
            kwargs = self.kwargs_of(node)
            if isinstance(base, _T):
                return self.tensor_method(base, node.func.attr, args, kwargs)
            return getattr(base, node.func.attr)(*args, **kwargs)
        target = self.value(node.func)
        args = [self.value(a) for a in node.args]
        kwargs = self.kwargs_of(node)
        if isinstance(target, _Intrinsic):
            return getattr(self, f"_i_{target.name}")(*args, **kwargs)
        prim = getattr(target, "op", None)
        if prim is not None:  # a primitive over tensors -> pointwise; over hosts -> refuse
            if not any(isinstance(a, _T) for a in args):
                raise ValueError(f"{prim}() over build-time values is host arithmetic — spell it plainly")
            if kwargs:
                raise ValueError(f"{prim} takes positional arguments")
            return self.pointwise(prim, *args)
        if callable(target):
            if any(isinstance(a, _T) for a in args) or any(isinstance(v, _T) for v in kwargs.values()):
                return self.inline(target, args, kwargs)
            return target(*args, **kwargs)  # fully structural: build-time evaluation
        raise ValueError(f"cannot call {target!r} in a step body")

    # ---- the S.1 method vocabulary --------------------------------------

    def tensor_method(self, base: _T, name: str, args: list, kwargs: dict):
        if name in ("mean", "sum", "max", "min"):  # reduce by dim name(s)
            dims = args[0] if args else kwargs.get("dims")
            names = (dims,) if isinstance(dims, str) else tuple(dims)
            return self.emit("reduce", (base.var,), name, f=name, dims=names)
        if name in ("sqrt", "exp", "log", "tanh"):
            return self.pointwise(name, base, hint=name)
        if name == "extent":  # a STRUCTURAL read: the dim's (start, stop)
            d = base.shadow.dim(args[0])
            return (d.start, d.stop)
        if name == "repeat_like":
            return self._repeat_like(base, args[0], **kwargs)
        if name not in _METHODS:
            raise ValueError(f"tensors have no method {name!r} in step bodies")
        if any(_holds_tensor(v) for v in list(args) + list(kwargs.values())):
            raise ValueError(_STRUCTURAL_SLOT.format(what=f".{name}(...)"))
        op, pack = _METHODS[name]
        return self.emit(op, (base.var,), name, **pack(args, kwargs))

    def rep_dim(self, t: _T, d) -> _T:
        """Broadcast one dim onto ``t``, carrying the source dim's chart,
        labels, and placement (the declaration stays complete)."""
        out = self.emit(
            "repeat", (t.var,), "rep", name=d.name, extent=(d.start, d.stop), chart=d.chart, labels=d.labels
        )
        if d.level is not None:
            out = self.emit("bind", (out.var,), "rep", levels={d.name: d.level})
        return out

    def _repeat_like(self, base: _T, x, but=None, dim=None) -> _T:
        """Broadcast ``base`` toward ``x``'s dims: with ``dim=`` add exactly
        those dims (from x's extents); otherwise add every dim of x that
        ``base`` lacks, except any named in ``but``."""
        if not isinstance(x, _T):
            raise ValueError("repeat_like takes the tensor to align with")
        src = {d.name: d for d in x.shadow.dims}
        if dim is not None:
            names = (dim,) if isinstance(dim, str) else tuple(dim)
        else:
            have = {d.name for d in base.shadow.dims}
            excl = {but} if isinstance(but, str) else set(but or ())
            names = tuple(n for n in src if n not in have and n not in excl)
        out = base
        for n in names:
            out = self.rep_dim(out, src[n])
        return out

    # ---- the S.1 function vocabulary (intrinsics) -----------------------

    def _i_iota_of(self, t, dim):
        if not isinstance(t, _T):
            raise ValueError("iota_of takes a tensor and a dim name")
        return self.emit("iota", (t.var,), "iota", name=dim)

    def _i_const_like(self, t, value):
        return self.const_like(value, t)

    def _i_contract(self, a, b_, axis=None):
        """Named-axis contraction: sum over the UNIQUE shared axis, or the
        axis/axes named to break a genuine ambiguity; non-contracted dims
        ride. Matmul as declaration — repeat + mul + reduce (D5)."""
        if not (isinstance(a, _T) and isinstance(b_, _T)):
            raise ValueError("contract takes two tensors")
        da = {d.name: d for d in a.shadow.dims}
        db = {d.name: d for d in b_.shadow.dims}
        shared = [n for n in da if n in db]
        if axis is None:
            if len(shared) != 1:
                what = "no shared axis" if not shared else f"shared axes {sorted(shared)}"
                raise ValueError(
                    f"contract: {what} between operands (a: {sorted(da)}, b: {sorted(db)}) — "
                    f"a unique shared axis contracts implicitly; name the contraction: "
                    f"contract(a, b, axis=...)"
                )
            axes = (shared[0],)
        else:
            axes = (axis,) if isinstance(axis, str) else tuple(axis)
            for ax in axes:
                if ax not in shared:
                    raise ValueError(f"contract axis {ax!r} is not shared (a: {sorted(da)}, b: {sorted(db)})")
        xb = a
        for n, d in db.items():
            if n not in da:
                xb = self.rep_dim(xb, d)
        yb = b_
        for n, d in da.items():
            if n not in db:
                yb = self.rep_dim(yb, d)
        m = self.pointwise("mul", xb, yb, hint="mm")
        return self.emit("reduce", (m.var,), "mm", f="sum", dims=axes)

    def _i_reduce_over(self, f, operands, dims):
        """The two-operand reduce form (S.1 committed vocabulary): a named
        (composite) reducer over aligned operand tensors."""
        ts = operands if isinstance(operands, tuple) else (operands,)
        if not all(isinstance(t, _T) for t in ts):
            raise ValueError("reduce_over takes tensor operands")
        names = (dims,) if isinstance(dims, str) else tuple(dims)
        fname = f if isinstance(f, str) else f.name
        return self.emit("reduce", tuple(t.var for t in ts), "red", f=fname, dims=names)

    def inline(self, fn, args, kwargs) -> object:
        """A helper called on tensor arguments: lower its body here, its
        parameters bound to the caller's values — inlining by lowering.
        Defaults and keyword-only parameters bind through the real
        signature (S.1 helpers read like ordinary Python)."""
        import inspect

        tree = _fn_ast(fn)
        inner = self.child(_captured(fn))
        try:
            ba = inspect.signature(fn).bind(*args, **kwargs)
            ba.apply_defaults()
        except TypeError as exc:
            raise ValueError(f"{fn.__name__}(...): {exc}") from None
        inner.env.update(ba.arguments)
        outs = inner.run_body(tree)
        return _T(outs[0], self.shadows[outs[0]]) if len(outs) == 1 else tuple(
            _T(o, self.shadows[o]) for o in outs
        )


_HOST_BIN = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}
_HOST_CMP = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.LtE: lambda a, b: a <= b,
    ast.Lt: lambda a, b: a < b,
    ast.GtE: lambda a, b: a >= b,
    ast.Gt: lambda a, b: a > b,
}
