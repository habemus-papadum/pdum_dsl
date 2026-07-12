"""Phase B's front half: decoration-time source -> typed core IR, one fused pass.

The driver owns orchestration and services; the *language* lives outside as
``lower_ast`` rules — a dict mapping ``ast`` node types to
``fn(ctx, ast_node) -> Node | None`` (None for statements that only bind).
Widening the accepted Python subset is a registration, never a driver edit.

What the driver does, in order (architecture §4.3): **snapshot coherence**
(the decoration-time text must still compile to code value-equivalent to the
template — stale source cannot lower; checked via a recursive code
fingerprint because a closure body must be recompiled inside a synthetic
wrapper to reproduce its freevars); **parse + rebase** (AST lines become
absolute ``Loc``s via the snapshot's first line); **name fates** (param /
local / captured value -> ``core.env`` with a *path* attr / captured Handle
-> inlined callee / anything else -> loud); **fused typing+lowering** (types
come from the Builder's op rules as nodes are emitted — a type error carries
its source points).

Inlining is pure and multi-statement (no M0 single-return restriction):
a callee's body lowers with params bound to the caller's argument nodes,
env paths prefixed by the callee's slot, and every emitted loc wrapped in
``CallLoc`` — provenance chains compose through nesting. ``Derived``
templates (pipelines today, transforms later) lower through registered
**build rules** instead of source.

Env slots are **paths** into the capture tree: root capture *i* is
``(i,)``; an inlined callee at root slot *i* contributes its capture *j* as
``(i, j)`` — exactly the ``EnvLeaf`` recursion marshaling flattens (ch08).

Book: ``docs/book/ch07-source-to-ir.ipynb``.
"""

from __future__ import annotations

import ast as pyast
import textwrap
from types import CodeType

from .capture import Handle
from .ir import Builder, CallLoc, Loc, Node, Region, VerifyError
from .types import Derived


class NoSourceError(RuntimeError):
    """Phase B needs source; this handle has none (REPL?). The per-backend
    raw_kernel escape hatch arrives with the backends."""


class StaleSourceError(VerifyError):
    """The snapshot no longer compiles to the captured code object."""


class MissingRule(VerifyError):
    """No lower_ast rule for this syntax — widening is a registration."""


class NameFateError(VerifyError):
    """A name with no sanctioned fate (param/local/capture)."""


def _code_fp(code: CodeType) -> tuple:
    consts = tuple(_code_fp(c) if isinstance(c, CodeType) else c for c in code.co_consts)
    return (code.co_code, consts, code.co_names, code.co_varnames, code.co_freevars, code.co_argcount)


def check_coherence(handle: Handle) -> None:
    snap = handle.snapshot
    if snap is None:
        raise NoSourceError(f"{handle.fntype.template.label}: no source available to lower")
    wrapper = "def __outer__(" + ", ".join(handle.freevars) + "):\n" + textwrap.indent(snap.text, "    ")
    try:
        outer = compile(wrapper, snap.filename, "exec")
    except SyntaxError as exc:
        raise StaleSourceError(f"{snap.qualname}: snapshot no longer parses ({exc})") from None
    name = handle.pyfunc.__code__.co_name
    inner = next((c for c in outer.co_consts[0].co_consts if isinstance(c, CodeType) and c.co_name == name), None)
    if inner is None or _code_fp(inner) != _code_fp(handle.pyfunc.__code__):
        raise StaleSourceError(f"{snap.qualname}: source drifted from the captured code object")


class Lowerer:
    """Per-function lowering context: services for rules, nesting for inlining."""

    def __init__(self, handle: Handle, rules: dict, ops: dict, derived: dict, prefix=(), wrap=None):
        self.handle, self.rules, self.ops, self.derived = handle, rules, ops, derived
        self.prefix, self.wrap = prefix, wrap
        self.builder = Builder(ops)
        self.locals: dict[str, Node] = {}
        self._bound = list(handle.env)  # bound freevar names, capture order (= env paths)

    def loc(self, node) -> Loc | CallLoc:
        snap = self.handle.snapshot
        raw = Loc(snap.filename, snap.firstlineno + node.lineno - 1, node.col_offset)
        return CallLoc(raw, self.wrap) if self.wrap is not None else raw

    def emit(self, op: str, *args, node=None, **kw) -> Node:
        return self.builder.emit(op, *args, loc=self.loc(node) if node is not None else None, **kw)

    def lower(self, node) -> Node | None:
        rule = self.rules.get(type(node))
        if rule is None:
            raise MissingRule(f"no lower_ast rule for {type(node).__name__} [{fmt(self.loc(node))}]")
        return rule(self, node)

    def resolve(self, name: str, node):
        if name in self.locals:
            return self.locals[name]
        if name in self._bound:
            idx = self._bound.index(name)
            value = self.handle.env[name]
            if isinstance(value, Handle):
                return ("callee", value, idx)
            return self.emit("core.env", node=node, type=self.handle.table.typeof(value), slot=self.prefix + (idx,))
        raise NameFateError(
            f"{name!r} is neither a parameter, a local, nor a capture [{fmt(self.loc(node))}]; "
            f"globals have no sanctioned fate yet — capture it"
        )

    def inline(self, callee: Handle, idx: int, args: tuple, call_node) -> Node:
        check_coherence(callee)
        sub = Lowerer(callee, self.rules, self.ops, self.derived, self.prefix + (idx,), self.loc(call_node))
        names = callee.pyfunc.__code__.co_varnames[: callee.pyfunc.__code__.co_argcount]
        if len(names) != len(args):
            raise VerifyError(f"{callee.fntype.template.label} takes {len(names)} args, got {len(args)}")
        sub.locals.update(zip(names, args))
        return sub.run_body()

    def run_body(self) -> Node:
        fn = next(n for n in pyast.parse(self.handle.snapshot.text).body if isinstance(n, pyast.FunctionDef))
        for stmt in fn.body:
            result = self.lower(stmt)
            if result is not None:
                return result
        raise VerifyError(f"{self.handle.fntype.template.label}: body never returns")


def fmt(p) -> str:
    from .ir import format_loc

    return format_loc(p)


def lower_handle(handle, rules: dict, ops: dict, *, arg_types=(), derived: dict | None = None) -> Region:
    """Source (or a Derived build rule) -> a verified, typed core Region."""
    derived = derived or {}
    template = handle.fntype.template
    if isinstance(template, Derived):
        build = derived.get(template.tag)
        if build is None:
            raise MissingRule(f"no build rule for Derived template {template.tag!r}")
        return build(handle, rules, ops, arg_types, derived)
    check_coherence(handle)
    ctx = Lowerer(handle, rules, ops, derived)
    code = handle.pyfunc.__code__
    names = code.co_varnames[: code.co_argcount]
    if len(arg_types) != len(names):
        raise VerifyError(f"{handle.fntype.template.label} takes {len(names)} args; got types for {len(arg_types)}")
    params = tuple(ctx.builder.param(i, t) for i, t in enumerate(arg_types))
    ctx.locals.update(zip(names, params))
    result = ctx.run_body()
    return Region(params=params, body=(ctx.builder.emit("core.yield", result),))
