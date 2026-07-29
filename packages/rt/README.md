# pdum.rt

Runtimes and backend emitters — the third workspace package (design
`docs/design/283_runtimes-package.md`). One expression generator under
four lexical rules; the launch contract (thread sizing specializes;
guard, per-stage bindings, staging device representation, math rows as
negotiated clauses); the device registry with features-at-creation;
device compilation through the content door. Columns: `wgsl/`, `msl/`
(evidence-backed by the graphics campaign), `cuda/` (the L4 team's —
the name exists, the column refuses toward its arrival).

Depends on `pdum.dsl` only: regions are walked structurally (ops are
strings, attrs are data), which is what lets `pdum.tl` import rt
without a cycle.
