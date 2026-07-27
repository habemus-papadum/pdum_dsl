# Design documents

**`200_the-spec.md` is the specification** — the single reference for the
system and the migration (P0–P9). `210_backend-notes.md` carries the
distilled knowledge from the purged backends (numeric policy, runtime
learnings, instrumentation methodology), written for the L4-era backend
builders. `220_principles.md` is the living canon of mechanisms;
`250_coordinate-algebra.md` is normative for the indexing surface
(Frame/Coordinate/Displacement/Slice) across all tiers. The syntax
tour is EXECUTABLE — `packages/tensorlib/notebooks/20_syntax_tour.ipynb`,
run in CI and under pytest, with committed-future cells skip-tagged —
so the tour can never rot.

Everything numbered 010–195 is **history** (`history/`): the original
architecture and plan, the step-by-step build of the first system, the
critical-assessment charter and report, and the integration decision
documents that produced 200. Read 200; read history only for archaeology.
