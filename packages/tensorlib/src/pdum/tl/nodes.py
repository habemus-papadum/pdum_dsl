"""The Node schema — the declared stability boundary (200 §3.2).

Three tiny frozen dataclasses (Arg / Const / Prim) are the contract between
every producer of scalar expression trees and every consumer of them. The
one-body-language law (200 §S.3 amendment) moved the schema's HOME to the
language package — ``pdum.dsl.derivative`` — because the one derivative
table builds these trees and ``sqrt``'s slope is a language fact; this
module re-exports it unchanged, so no consumer rewrites (the no-rewrite
guarantee, held by the schema rather than by promise). It still imports
nothing from the rest of pdum.tl.

- consumers (compute.py / autodiff.py / signatures.py / opcount.py): numpy
  evaluation, symbolic partial derivatives, carrier/unit signature
  propagation — all walk Nodes and never care where they came from;
- producers: the shared-syntax AST producer (producer.py) — the ONE
  producer of scalar bodies.
"""

from __future__ import annotations

from pdum.dsl.derivative import Arg, Const, Node, Prim, is_const

__all__ = ["Arg", "Const", "Node", "Prim", "is_const"]
