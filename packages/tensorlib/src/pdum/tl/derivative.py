"""THE one derivative table (200 §S.2) — re-exported from its forced altitude.

The value-tier tangent rules and the marker partial rules are the same
object: ONE table. The one-body-language law (200 §S.3 amendment, 220 §14)
moved its home to the language package — ``pdum.dsl.derivative`` — because
the value tier differentiates too and dsl cannot import tensorlib; this
module re-exports it so every tl consumer (``CompositeMarker.partial``, the
reducer BPTT engine, autodiff) reads the same rows unchanged.

The at-kink law rides with the table: one-sided and PARTITIONING — at a tie
exactly one operand receives the cotangent, first-wins (``maximum`` sends
the cotangent left on ``ge``; ``minimum`` mirrors). Frozen contract, pinned
at the kink points (test_at_kink).
"""

from __future__ import annotations

from pdum.dsl.derivative import TABLE, diff

__all__ = ["TABLE", "diff"]
