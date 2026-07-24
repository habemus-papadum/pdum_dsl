"""The Node schema — the declared stability boundary (200 §3.2).

Three tiny frozen dataclasses (Arg / Const / Prim) are the contract between
every producer of scalar expression trees and every consumer of them. This
module imports NOTHING — not from pdum.dsl, not from the rest of pdum.tl.

- consumers (compute.py / autodiff.py / signatures.py / opcount.py): numpy
  evaluation, symbolic partial derivatives, carrier/unit signature
  propagation — all walk Nodes and never care where they came from;
- producers: the shared-syntax AST producer (producer.py) — since P4 the
  ONE producer of scalar bodies (the operator-overloading tracer preceded
  it and is gone). Swapping producers can never force a rewrite of
  consumers — that is the no-rewrite guarantee, held by the schema rather
  than by promise.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Arg:
    """A formal parameter of the marker (by position)."""

    index: int


@dataclass(frozen=True)
class Const:
    """A literal: int/Fraction (exact) or float (value-space)."""

    value: object


@dataclass(frozen=True)
class Prim:
    """Application of a PRIMITIVE marker, referenced by name."""

    op: str
    args: tuple


Node = Arg | Const | Prim


def is_const(n, v) -> bool:
    return isinstance(n, Const) and n.value == v
