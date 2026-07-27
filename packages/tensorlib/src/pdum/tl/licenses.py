"""Descent licenses — the SCHEMA STUB (200 §4 interior / §8 runway).

A license is a DECLARATION that a lowering may replace the reference
denotation with a cheaper equivalent, with the claim stated honestly:
what kind of deviation, over what equivalence, within what tolerance, on
what input domain. The reference NEVER licenses itself — it is the
denotation; licenses are consumed at DESCENT (L4), where the license set
joins the registry key (two programs lowered under different license
sets are different artifacts). The numeric tier monitors divergence and
never certifies it.

This module is deliberately data-only: the schema and one worked
declaration (the GEMM the spec's worked check names — f16 tiles with f32
accumulators). Enforcement, monitoring, and the certified-rewrite
checker arrive with L4; nothing here executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# the taxonomy is CLOSED (200 §4): a new kind is a spec amendment
KINDS = ("none", "reassociation", "precision-demotion")


@dataclass(frozen=True)
class License:
    """One declared deviation, registry-key material."""

    name: str  # e.g. "gemm.f16tile.f32acc" — joins the registry key at descent
    kind: str  # one of KINDS
    equivalence: str  # the claim, stated over the CARRIER denotation (never bytes)
    rtol: float  # tolerance of the claim ...
    atol: float  # ... on the stated domain
    input_domain: str  # where the claim holds (families, ranges, exclusions)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(
                f"license kind must be one of {KINDS}, got {self.kind!r} — the taxonomy is closed (200 §4)"
            )


# the worked declaration (200 §4's worked check): a real-carrier
# contraction lowered to f16 tiles + f32 accumulators. Reassociation is
# implied by tiling (the k-sum re-brackets); the demotion is the tiles.
GEMM_F16_TILES = (
    License(
        name="gemm.k-reassoc",
        kind="reassociation",
        equivalence="sum over k re-bracketed by tile (ko, ki) — exact over reals, reordered over floats",
        rtol=1e-6,
        atol=1e-7,
        input_domain="finite operands; no catastrophic cancellation family (adversarial tails are L4 gates)",
    ),
    License(
        name="gemm.f16tile.f32acc",
        kind="precision-demotion",
        equivalence="operand tiles read at f16, accumulated at f32; result within tolerance of the f64 interior",
        rtol=1e-3,
        atol=1e-4,
        input_domain="|x| in [2^-14, 2^15] (f16 normal range); weights/activations per boundary descriptors",
        notes="the writable argument's descriptor encodes the RESULT precision — byte-blind capacity "
        "mistakes are impossible by construction (200 §4)",
    ),
)


@dataclass(frozen=True)
class Descent:
    """A lowering's identity contribution: the program plus its license
    set. Stub — L4's registry consumes it."""

    program: str  # the program's identity (registry key material)
    licenses: tuple[License, ...] = field(default_factory=tuple)

    def key(self) -> tuple:
        return (self.program, tuple(sorted(lic.name for lic in self.licenses)))
