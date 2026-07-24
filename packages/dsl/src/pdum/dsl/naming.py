"""Name assignment — the core mechanism the naming law builds on (200 §1.6).

Names are contract surface: parameters, taps, derived gradients, and SSA
variables are all addressed by name, so name assignment happens in ONE
mechanism, here in the core, never ad hoc in a satellite. At P3 this is the
piece the SSA builders need — deterministic hint dedup and explicit-claim
collision refusal; the full naming law (scope paths, level-first hierarchy,
derived suffixes) lands with the scope at P5 on top of this.

Two doors, two disciplines:

- ``claim(name)`` — an EXPLICIT name (an input, a declared leaf). A collision
  refuses with a designed message: contract names are never auto-suffixed
  (200 §1.6 — "never an auto-suffix").
- ``derive(hint)`` — an INTERNAL name from a hint (SSA temporaries). The bare
  hint if free, else ``hint1``, ``hint2``, … — deterministic from claim
  order, carrying no contract weight.
"""

from __future__ import annotations


class NameCollision(ValueError):
    """An explicit name was claimed twice."""


class Namer:
    """One name space: explicit claims refuse collisions; derived hints dedup."""

    def __init__(self, taken=()) -> None:
        self.taken: set[str] = set(taken)

    def __contains__(self, name: str) -> bool:
        return name in self.taken

    def claim(self, name: str) -> str:
        if name in self.taken:
            raise NameCollision(
                f"name {name!r} is already declared — contract names are never "
                f"auto-suffixed; declare it once, or address a different path"
            )
        self.taken.add(name)
        return name

    def derive(self, hint: str) -> str:
        if hint not in self.taken:
            self.taken.add(hint)
            return hint
        i = 1
        while f"{hint}{i}" in self.taken:
            i += 1
        name = f"{hint}{i}"
        self.taken.add(name)
        return name
