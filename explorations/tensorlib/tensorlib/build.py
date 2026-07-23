"""A thin SSA builder: NAME MANAGEMENT for writing IR by hand, nothing more.

Deliberately not a frontend — no tracing, no operator overloading, no
expression trees (those live behind the marker DSL's stability boundary).
This exists so multi-layer zoo models can be written in topological order
without hand-numbering SSA variables.
"""

from __future__ import annotations

from .ir import Instr, Program


class Build:
    def __init__(self) -> None:
        self.instrs: list[Instr] = []
        self.taken: set[str] = set()

    def _name(self, hint: str) -> str:
        if hint not in self.taken:
            self.taken.add(hint)
            return hint
        i = 1
        while f"{hint}{i}" in self.taken:
            i += 1
        name = f"{hint}{i}"
        self.taken.add(name)
        return name

    def input(self, name: str) -> str:
        if name in self.taken:
            raise ValueError(f"input {name!r} already defined")
        self.taken.add(name)
        self.instrs.append(Instr(name, "input", (), {}))
        return name

    def emit(self, op: str, operands=(), hint: str | None = None, **params) -> str:
        var = self._name(hint or op)
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
