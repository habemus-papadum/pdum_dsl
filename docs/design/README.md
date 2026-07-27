# Design documents

**`200_the-spec.md` is the specification** — the single reference for the
system and the migration (P0–P9). `210_backend-notes.md` carries the
distilled knowledge from the purged backends (numeric policy, runtime
learnings, instrumentation methodology), written for the L4-era backend
builders. `220_principles.md` is the living canon of mechanisms;
`250_coordinate-algebra.md` is normative for the indexing surface
(Frame/Coordinate/Displacement/Slice and the §10 value door) across all
tiers. `260_l4-brief.md` is the onboarding brief for the L4 work — the
reading order, the ruled-and-closed decisions, the excavation-first
plan, and the traps already sprung once. `270_owners-guidance.md` is
the owner's own guidance for the efficiency era — it rides on top of
260 and wins where they differ. The syntax tour (230) is an
EXECUTABLE NOTEBOOK — `230_syntax_tour.ipynb`, run in CI and under
pytest with committed-future cells skip-tagged (so it can never rot),
and rendered into the documentation site with the rest of the
notebooks.

Everything numbered 010–195 is **history** (`history/`): the original
architecture and plan, the step-by-step build of the first system, the
critical-assessment charter and report, and the integration decision
documents that produced 200. Read 200; read history only for archaeology.
