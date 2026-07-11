"""Lower a jitted function's source AST into untyped IR.

Captures come from the code object (``co_freevars`` → uniforms, the first
``co_argcount`` ``co_varnames`` → args); everything else assigned is a local. The
accepted Python subset is deliberately small (the "define the syntax very
specifically" seam): assignments, tuple-unpack, returns, arithmetic/compare,
ternary, vector tuples, attribute swizzles, ``builtins.*`` intrinsics, and a fixed
set of builtin calls.
"""

from __future__ import annotations

import ast
import textwrap
from types import CodeType

from .. import ir
from ..backends.wgsl.intrinsics import INTRINSIC_NAMES

_BINOP = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**", ast.Mod: "%"}
_CMP = {ast.Lt: "<", ast.Gt: ">", ast.LtE: "<=", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
_SWIZZLE = set("xyzw")


class LoweringError(Exception):
    pass


class Lowerer:
    def __init__(self, code: CodeType):
        self.freevars = set(code.co_freevars)
        self.argnames = set(code.co_varnames[: code.co_argcount])
        self._tmp = 0

    def _fresh(self) -> str:
        self._tmp += 1
        return f"_t{self._tmp}"

    def lower_function(self, source: str) -> ir.Function:
        if not source:
            raise LoweringError("no source available to lower")
        mod = ast.parse(textwrap.dedent(source))
        fn = next((n for n in mod.body if isinstance(n, ast.FunctionDef)), None)
        if fn is None:
            raise LoweringError("expected a function definition")
        body: list[ir.Node] = []
        for stmt in fn.body:
            self._lower_stmt(stmt, body)
        params = [a.arg for a in fn.args.args]
        return ir.Function(name=fn.name, params=params, body=body)

    # --- statements --------------------------------------------------------

    def _lower_stmt(self, stmt: ast.stmt, out: list[ir.Node]) -> None:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                raise LoweringError("chained assignment not supported")
            target = stmt.targets[0]
            value = self._lower_expr(stmt.value)
            if isinstance(target, ast.Name):
                out.append(ir.Let(target.id, value))
            elif isinstance(target, ast.Tuple):
                tmp = self._fresh()
                out.append(ir.Let(tmp, value))
                for i, elt in enumerate(target.elts):
                    if not isinstance(elt, ast.Name):
                        raise LoweringError("nested unpack not supported")
                    out.append(ir.Let(elt.id, ir.Swizzle(ir.Name(tmp, "local"), "xyzw"[i])))
            else:
                raise LoweringError(f"unsupported assignment target {ast.dump(target)}")
        elif isinstance(stmt, ast.Return):
            if stmt.value is None:
                raise LoweringError("bare return not supported")
            out.append(ir.Return(self._lower_expr(stmt.value)))
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            return  # docstring / stray constant
        elif isinstance(stmt, ast.Pass):
            return
        else:
            raise LoweringError(f"unsupported statement: {type(stmt).__name__}")

    # --- expressions -------------------------------------------------------

    def _lower_expr(self, e: ast.expr) -> ir.Node:
        if isinstance(e, ast.Constant):
            return ir.Lit(e.value)
        if isinstance(e, ast.Name):
            return self._lower_name(e.id)
        if isinstance(e, ast.Attribute):
            return self._lower_attr(e)
        if isinstance(e, ast.BinOp):
            op = _BINOP.get(type(e.op))
            if op is None:
                raise LoweringError(f"unsupported binary op {type(e.op).__name__}")
            return ir.BinOp(op, self._lower_expr(e.left), self._lower_expr(e.right))
        if isinstance(e, ast.UnaryOp):
            if isinstance(e.op, ast.USub):
                return ir.Unary("-", self._lower_expr(e.operand))
            if isinstance(e.op, ast.UAdd):
                return self._lower_expr(e.operand)
            if isinstance(e.op, ast.Not):
                return ir.Unary("!", self._lower_expr(e.operand))
            raise LoweringError(f"unsupported unary op {type(e.op).__name__}")
        if isinstance(e, ast.Compare):
            if len(e.ops) != 1:
                raise LoweringError("chained comparison not supported")
            op = _CMP.get(type(e.ops[0]))
            if op is None:
                raise LoweringError(f"unsupported comparison {type(e.ops[0]).__name__}")
            return ir.Compare(op, self._lower_expr(e.left), self._lower_expr(e.comparators[0]))
        if isinstance(e, ast.IfExp):
            return ir.Select(
                self._lower_expr(e.test),
                self._lower_expr(e.body),
                self._lower_expr(e.orelse),
            )
        if isinstance(e, ast.Tuple):
            return ir.MakeVec([self._lower_expr(x) for x in e.elts])
        if isinstance(e, ast.Call):
            return self._lower_call(e)
        raise LoweringError(f"unsupported expression: {type(e).__name__}")

    def _lower_name(self, name: str) -> ir.Name:
        if name in self.freevars:
            scope = "uniform"
        elif name in self.argnames:
            scope = "arg"
        else:
            scope = "local"
        return ir.Name(name, scope)

    def _lower_attr(self, e: ast.Attribute) -> ir.Node:
        v = e.value
        if isinstance(v, ast.Name) and v.id == "builtins":
            iname = INTRINSIC_NAMES.get(e.attr)
            if iname is None:
                raise LoweringError(f"unknown builtin intrinsic builtins.{e.attr}")
            return ir.Intrinsic(iname)
        base = self._lower_expr(v)
        if set(e.attr) <= _SWIZZLE and 1 <= len(e.attr) <= 4:
            return ir.Swizzle(base, e.attr)
        raise LoweringError(f"unsupported attribute .{e.attr}")

    def _lower_call(self, e: ast.Call) -> ir.Node:
        # A call is either a builtin (sqrt, length, ...) or a device-function call
        # to be resolved + inlined by the inliner pass. Both lower to ir.Call here.
        if not isinstance(e.func, ast.Name):
            raise LoweringError("only simple name calls are supported")
        if e.keywords:
            raise LoweringError("keyword arguments not supported in shader calls")
        return ir.Call(e.func.id, [self._lower_expr(a) for a in e.args])
