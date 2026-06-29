"""Monomorphic inlining: flatten an entry shader + all the device functions it
calls into a single function, merging every captured uniform into one namespace.

Because the design is fully monomorphizing (every call site sees exactly one
``FnType``), each device call has a statically-known target and is inlined — the
functional composition collapses into the "pedantic" kernel, exactly the Julia
behavior. A device function is reached either as a captured free variable or as an
argument to a higher-order entry (``shader(img)``); both are resolved here.

Device functions are required to be a single ``return`` expression (M0.4); this
makes inlining pure substitution and keeps nesting (img → weave) trivial. Multi-
statement device fns (let-hoisting) are a later extension.

The merged uniforms are collected fresh each call, so the *values* track the
current (rebuilt-every-frame) closures while the *structure* — names/types/order —
stays stable, which is what lets the cached pipeline be reused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import ir
from ..frontend.ast_lower import Lowerer
from ..jit import Handle, Program
from ..types import Type, typeof


class InlineError(Exception):
    pass


@dataclass
class Flattened:
    fn: ir.Function  # entry with all device calls inlined
    names: list[str]  # merged uniform order (== layout field order)
    types: dict[str, Type]  # merged name -> DSL type
    values: dict[str, object] = field(default_factory=dict)  # merged name -> current value


class _Inliner:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.types: dict[str, Type] = {}
        self.values: dict[str, object] = {}
        self._ctr = 0

    def _add_uniform(self, merged: str, value: object) -> None:
        if merged not in self.types:
            self.names.append(merged)
            self.types[merged] = typeof(value)
            self.values[merged] = value

    def _prefix(self, base: str) -> str:
        self._ctr += 1
        return f"{base}{self._ctr}_"

    def _scan_freevars(self, handle: Handle, prefix: str):
        """Split a handle's captures into device-fn resolutions and scalar-uniform
        renames, registering the scalars into the merged namespace."""
        resolver: dict[str, Handle] = {}
        rename: dict[str, str] = {}
        for fv in handle.pyfunc.__code__.co_freevars:
            if fv not in handle.env:
                continue
            val = handle.env[fv]
            if isinstance(val, Handle):
                resolver[fv] = val
            else:
                merged = prefix + fv
                self._add_uniform(merged, val)
                rename[fv] = merged
        return resolver, rename

    def process_entry(self, program: Program) -> ir.Function:
        entry = program.entry
        fn = Lowerer(entry.pyfunc.__code__).lower_function(entry.source)
        resolver, rename = self._scan_freevars(entry, prefix="")
        # Bind higher-order params (e.g. shader(img)) to their device-handle args.
        params = entry.pyfunc.__code__.co_varnames[: entry.pyfunc.__code__.co_argcount]
        for i, pname in enumerate(params):
            if i < len(program.args) and isinstance(program.args[i], Handle):
                resolver[pname] = program.args[i]
        fn.body = [self._xform_stmt(s, resolver, rename, {}) for s in fn.body]
        return fn

    def _inline(self, dev: Handle, arg_nodes: list[ir.Node]) -> ir.Node:
        fn = Lowerer(dev.pyfunc.__code__).lower_function(dev.source)
        if len(fn.body) != 1 or not isinstance(fn.body[0], ir.Return):
            raise InlineError(f"device function {dev.pyfunc.__name__!r} must be a single return expression")
        resolver, rename = self._scan_freevars(dev, prefix=self._prefix(dev.pyfunc.__name__))
        pmap = dict(zip(fn.params, arg_nodes))
        return self._xform_expr(fn.body[0].value, resolver, rename, pmap)

    # --- transform ---------------------------------------------------------

    def _xform_stmt(self, stmt, resolver, rename, pmap):
        if isinstance(stmt, ir.Let):
            return ir.Let(stmt.name, self._xform_expr(stmt.value, resolver, rename, pmap))
        if isinstance(stmt, ir.Return):
            return ir.Return(self._xform_expr(stmt.value, resolver, rename, pmap))
        raise InlineError(f"unexpected statement {stmt!r}")

    def _xform_expr(self, e, resolver, rename, pmap):
        x = lambda n: self._xform_expr(n, resolver, rename, pmap)  # noqa: E731

        if isinstance(e, (ir.Lit, ir.Intrinsic)):
            return e
        if isinstance(e, ir.Name):
            if e.scope == "uniform":
                return ir.Name(rename[e.name], "uniform")
            if e.scope == "arg" and e.name in pmap:
                return pmap[e.name]
            return e
        if isinstance(e, ir.Swizzle):
            return ir.Swizzle(x(e.base), e.comps)
        if isinstance(e, ir.Unary):
            return ir.Unary(e.op, x(e.operand))
        if isinstance(e, ir.BinOp):
            return ir.BinOp(e.op, x(e.left), x(e.right))
        if isinstance(e, ir.Compare):
            return ir.Compare(e.op, x(e.left), x(e.right))
        if isinstance(e, ir.Select):
            return ir.Select(x(e.cond), x(e.if_true), x(e.if_false))
        if isinstance(e, ir.MakeVec):
            return ir.MakeVec([x(el) for el in e.elems])
        if isinstance(e, ir.Call):
            dev = resolver.get(e.func)
            args = [x(a) for a in e.args]
            if isinstance(dev, Handle):
                return self._inline(dev, args)
            return ir.Call(e.func, args)  # builtin — validated at emit
        raise InlineError(f"cannot inline {e!r}")


def flatten(program: Program | Handle) -> Flattened:
    """Inline an entry (Program or bare fragment Handle) into one flat function +
    merged uniform set."""
    if isinstance(program, Handle):
        program = Program(program, ())
    inl = _Inliner()
    fn = inl.process_entry(program)
    return Flattened(fn=fn, names=inl.names, types=inl.types, values=inl.values)
