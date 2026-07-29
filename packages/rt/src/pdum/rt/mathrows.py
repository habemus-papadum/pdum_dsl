"""The target numeric contract — per-target math rows, with proofs.

The category FAIL-0 named (210): a target's math library deviates from
the reference's function on part of its domain, the translation row is
faithful, the runtime never touches arithmetic — so neither box owns
the fix. These rows do. Each row substitutes a spelling for a marker
and carries WHY and the freeness evidence (the proof that applying it
never changes a finite answer), so an artifact can state exactly what
it substituted (LaunchContract.math).

Row one, measured: Metal's f32 ``tanh`` returns NaN for |x| ≥ 44.3614
(exactly log(FLT_MAX)/2 — an exp(2x) implementation; MTLMathModeSafe
does not help). ``tanh(clamp(x, -20, 20))`` is BITWISE-identical to
the unclamped call wherever both are finite and finite over
[-200, 200] (spike_metal/mathrows.py). The row applies to every
column v1: it is free where the library is already correct, and the
wgpu path reaches the same Metal library on this hardware. Per-device
conditionality arrives with a second math-library data point (the
CUDA column's survey duty: exp, sinh/cosh, pow; fast-math flags are
LICENSES, never defaults).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    name: str  # the marker it substitutes
    spelling: str  # format string over the operand, e.g. "tanh(clamp({0}, ...))"
    why: str
    freeness: str  # the evidence that applying it never changes a finite answer


ROWS: dict[str, Row] = {
    "tanh": Row(
        name="tanh",
        spelling="tanh(clamp({0}, -20.0, 20.0))",
        why="Metal f32 tanh is NaN for |x| >= log(FLT_MAX)/2 (exp-based)",
        freeness="bitwise-identical to unclamped tanh wherever both are finite; "
        "tanh(20.0f) == 1.0f exactly (spike_metal/mathrows.py, rerunnable)",
    ),
}


def row(name: str) -> Row | None:
    return ROWS.get(name)
