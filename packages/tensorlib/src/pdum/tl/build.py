"""A thin SSA builder: NAME MANAGEMENT for writing IR by hand, nothing more.

Deliberately not a frontend — no tracing, no operator overloading, no
expression trees (those live behind the marker DSL's stability boundary).
This exists so multi-layer zoo models can be written in topological order
without hand-numbering SSA variables. Name assignment is the CORE's
(pdum.dsl.naming, P3): explicit inputs claim (collisions refuse), hints
derive. Build itself lives until P5, when makers replace the zoo builders.
"""

from __future__ import annotations

from pdum.dsl.naming import Namer

from .ir import Instr, Program


class Build:
    def __init__(self) -> None:
        self.instrs: list[Instr] = []
        self.names = Namer()

    def input(self, name: str) -> str:
        self.names.claim(name)
        self.instrs.append(Instr(name, "input", (), {}))
        return name

    def emit(self, op: str, operands=(), hint: str | None = None, **params) -> str:
        var = self.names.derive(hint or op)
        self.instrs.append(Instr(var, op, tuple(operands), params))
        return var

    # thin sugar for the common ops
    def pw(self, f: str, *operands: str, hint: str | None = None) -> str:
        return self.emit("pointwise", operands, hint=hint or f, f=f)

    def red(self, f: str, x: str, dims, hint: str | None = None) -> str:
        return self.emit("reduce", (x,), hint=hint or f, f=f, dims=tuple(dims))

    def repeat(self, x: str, name: str, extent, hint: str | None = None) -> str:
        return self.emit("repeat", (x,), hint=hint or "rep", name=name, extent=extent)

    def bcast(self, x: str, reps, hint: str | None = None) -> str:
        """Repeat over each (name, extent) in reps — broadcast by declaration."""
        for name, extent in reps:
            x = self.repeat(x, name, extent, hint=hint)
        return x

    def const(self, value, dims, hint: str = "c", **kw) -> str:
        return self.emit("const", (), hint=hint, value=value, dims=tuple(dims), **kw)

    def program(self) -> Program:
        return Program(tuple(self.instrs))
