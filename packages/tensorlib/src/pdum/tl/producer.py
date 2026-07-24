"""The AST producer — the shared expression syntax's scalar half (200 §S.2).

Lowers a plain Python function (a def or a lambda) onto the Node schema
(nodes.py): parameters become Args, captured numbers and literals become
Consts, calls to primitive names become Prims. Nothing is executed — the
body is lowered by inspection, which is exactly what the marker-granularity
gate demands: one marker is one named, inspectable body tree over
primitives, never an opaque call.

Straight-line is enforced AT LOWERING (the tracer refused at trace time via
``__bool__``): assignments and a final return, no control flow — ``where``
is the branch, data flow. Comparisons lower directly (``L.m >= R.m`` is
``ge``); records construct, destructure by attribute, and return
(record-typed reducer state); tuples subscript by constant and destructure.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
from fractions import Fraction
from functools import lru_cache

from .nodes import Arg, Const, Node, Prim

_STRAIGHT_LINE = (
    "marker bodies are straight-line: {what} cannot be lowered — "
    "use where(cond, a, b); the branch is data flow here"
)

_CMP = {ast.Eq: "eq", ast.NotEq: "ne", ast.LtE: "le", ast.Lt: "lt", ast.GtE: "ge", ast.Gt: "gt"}
_BIN = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}


def is_record(cls) -> bool:
    return isinstance(cls, type) and (dataclasses.is_dataclass(cls) or hasattr(cls, "_fields"))


def record_fields(cls) -> tuple[str, ...]:
    if dataclasses.is_dataclass(cls):
        return tuple(f.name for f in dataclasses.fields(cls))
    return tuple(cls._fields)


# ---- bindings: how a parameter's NAME maps onto Arg indices ----------------


def scalars(n: int, base: int = 0) -> tuple:
    return tuple(Arg(base + i) for i in range(n))


def tuple_binding(n: int, base: int = 0) -> tuple:
    return (tuple(scalars(n, base)),)


def record_binding(cls, n: int, base: int = 0) -> tuple:
    return (_RecordV(cls, scalars(n, base)),)


@dataclasses.dataclass(frozen=True)
class _RecordV:
    """A record-typed intermediate: a class + one value per field."""

    cls: type
    parts: tuple

    def field(self, name: str):
        fields = record_fields(self.cls)
        if name not in fields:
            raise ValueError(f"{self.cls.__name__} has no field {name!r} (fields: {', '.join(fields)})")
        return self.parts[fields.index(name)]


# ---- source recovery -------------------------------------------------------


@lru_cache(maxsize=None)
def _file_ast(filename: str):
    with open(filename) as f:
        return ast.parse(f.read())


def _fn_ast(fn) -> ast.FunctionDef | ast.Lambda:
    code = fn.__code__
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if len(defs) == 1 and fn.__name__ != "<lambda>":
            return defs[0]
        lambdas = [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]
        if len(lambdas) == 1:
            return lambdas[0]
    except (OSError, SyntaxError):
        pass
    try:  # a lambda inside a larger statement: find it by line in the file's AST
        lambdas = [
            n
            for n in ast.walk(_file_ast(code.co_filename))
            if isinstance(n, ast.Lambda) and n.lineno == code.co_firstlineno
        ]
    except (OSError, SyntaxError):
        lambdas = []
    if len(lambdas) == 1:
        return lambdas[0]
    raise ValueError(
        f"cannot recover a single unambiguous body for {fn!r} "
        f"(line {code.co_firstlineno} of {code.co_filename}) — define it as a plain `def`"
    )


def _captured(fn) -> dict:
    env = dict(vars(__import__("builtins")))
    env.update(fn.__globals__)
    if fn.__closure__:
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                env[name] = cell.cell_contents
            except ValueError:
                pass
    return env


# ---- the lowering visitor --------------------------------------------------


class _Lowerer:
    def __init__(self, env: dict):
        self.env = env  # name -> Node | tuple | _RecordV | captured Python value

    def refuse(self, node, what: str):
        raise ValueError(_STRAIGHT_LINE.format(what=what))

    def value(self, node) -> object:
        """Lower an expression to a Node, a tuple of values, or a _RecordV."""
        if isinstance(node, ast.Constant):
            return _const(node.value)
        if isinstance(node, ast.Name):
            if node.id not in self.env:
                raise ValueError(f"unknown name {node.id!r} in a marker body")
            return _lift(self.env[node.id], node.id)
        if isinstance(node, ast.BinOp):
            op = _BIN.get(type(node.op))
            if op is None:
                raise ValueError(f"operator {type(node.op).__name__} has no primitive; spell the call")
            return Prim(op, (self.scalar(node.left), self.scalar(node.right)))
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return Prim("neg", (self.scalar(node.operand),))
            if isinstance(node.op, ast.UAdd):
                return self.scalar(node.operand)
            self.refuse(node, f"operator {type(node.op).__name__}")
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ValueError("chained comparisons do not lower; split them")
            op = _CMP.get(type(node.ops[0]))
            if op is None:
                self.refuse(node, f"comparison {type(node.ops[0]).__name__}")
            return Prim(op, (self.scalar(node.left), self.scalar(node.comparators[0])))
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, ast.Attribute):
            base = self.value(node.value)
            if isinstance(base, _RecordV):
                return base.field(node.attr)
            raise ValueError(f"attribute access needs a record-typed value, got {base!r}")
        if isinstance(node, ast.Subscript):
            base = self.value(node.value)
            idx = node.slice
            if isinstance(base, tuple) and isinstance(idx, ast.Constant) and isinstance(idx.value, int):
                return base[idx.value]
            raise ValueError("subscripts lower only as tuple[<int literal>]")
        if isinstance(node, ast.Tuple):
            return tuple(self.value(e) for e in node.elts)
        if isinstance(node, (ast.IfExp, ast.BoolOp)):
            self.refuse(node, "if/and/or")
        self.refuse(node, f"{type(node).__name__}")

    def scalar(self, node) -> Node:
        v = self.value(node)
        if isinstance(v, (Arg, Const, Prim)):
            return v
        raise ValueError(f"a scalar is required here, got {v!r}")

    def call(self, node: ast.Call):
        target = None
        if isinstance(node.func, ast.Name):
            target = self.env.get(node.func.id, node.func.id)
        elif isinstance(node.func, ast.Attribute):
            base = self.env.get(node.func.value.id) if isinstance(node.func.value, ast.Name) else None
            target = getattr(base, node.func.attr, None) if base is not None else None
        if is_record(target):
            fields = record_fields(target)
            parts: dict = {}
            for f, a in zip(fields, node.args):
                parts[f] = self.value(a)
            for kw in node.keywords:
                parts[kw.arg] = self.value(kw.value)
            if set(parts) != set(fields):
                raise ValueError(f"{target.__name__}(...) must bind every field ({', '.join(fields)})")
            return _RecordV(target, tuple(parts[f] for f in fields))
        op, arity = getattr(target, "op", None), getattr(target, "arity", None)
        if op is None and isinstance(target, str):
            from .derivative import TABLE

            if target in TABLE:
                op, arity = target, len(TABLE[target])
        if op is None:
            raise ValueError(
                f"call to {ast.unparse(node.func)!r} does not lower: marker bodies "
                f"call primitives (exp, maximum, where, ...) or construct records — "
                f"an opaque call would break derived differentiation"
            )
        if node.keywords or len(node.args) != arity:
            raise ValueError(f"{op} takes {arity} positional arguments")
        return Prim(op, tuple(self.scalar(a) for a in node.args))


def _const(v) -> Const:
    if isinstance(v, bool) or not isinstance(v, (int, float, Fraction)):
        raise ValueError(f"cannot lower literal {v!r} into a marker body")
    return Const(v)


def _lift(v, name: str):
    if isinstance(v, (Arg, Const, Prim, _RecordV, tuple)):
        return v
    if isinstance(v, bool):
        raise ValueError(f"cannot lower captured {name}={v!r} into a marker body")
    if isinstance(v, (int, float, Fraction)):
        return Const(v)
    if is_record(v) or getattr(v, "op", None) is not None:
        return v  # a record class / primitive sentinel: meaningful only in calls
    raise ValueError(f"captured {name!r} is {type(v).__name__}-valued — only numbers lower into a body")


def lower(fn, bindings: tuple) -> tuple:
    """Lower ``fn`` with positional parameters bound to ``bindings`` (each a
    Node, a tuple, or a _RecordV). Returns the tuple of returned values —
    one per returned component (a returned record yields its fields)."""
    tree = _fn_ast(fn)
    params = [a.arg for a in tree.args.args]
    if len(params) != len(bindings):
        raise ValueError(f"{fn!r} takes {len(params)} parameters; {len(bindings)} bound")
    if tree.args.vararg or tree.args.kwonlyargs or tree.args.kwarg or tree.args.defaults:
        raise ValueError("marker bodies take plain positional parameters only")
    lo = _Lowerer(_captured(fn))
    lo.env.update(zip(params, bindings))
    if isinstance(tree, ast.Lambda):
        return _returns(lo.value(tree.body))
    for stmt in tree.body[:-1]:
        _statement(lo, stmt)
    last = tree.body[-1]
    if not isinstance(last, ast.Return) or last.value is None:
        raise ValueError("a marker body must end in `return <expression>`")
    return _returns(lo.value(last.value))


def _statement(lo: _Lowerer, stmt) -> None:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target, value = stmt.targets[0], lo.value(stmt.value)
        if isinstance(target, ast.Name):
            lo.env[target.id] = value
            return
        if isinstance(target, ast.Tuple) and all(isinstance(e, ast.Name) for e in target.elts):
            parts = value.parts if isinstance(value, _RecordV) else value
            if not isinstance(parts, tuple) or len(parts) != len(target.elts):
                raise ValueError(f"cannot destructure {value!r} into {len(target.elts)} names")
            for e, p in zip(target.elts, parts):
                lo.env[e.id] = p
            return
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        return  # a docstring
    if isinstance(stmt, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.Match)):
        lo.refuse(stmt, f"a {type(stmt).__name__.lower()} statement")
    lo.refuse(stmt, f"a {type(stmt).__name__} statement")


def _returns(value) -> tuple:
    if isinstance(value, _RecordV):
        value = value.parts
    if not isinstance(value, tuple):
        value = (value,)
    for v in value:
        if isinstance(v, _RecordV):
            raise ValueError("nested records do not lower as reducer state yet — return scalars")
        if not isinstance(v, (Arg, Const, Prim)):
            raise ValueError(f"a marker body must return scalars, got {v!r}")
    return value
