# The pdum workspace

Two packages, one lockstep version (see `docs/design/200_the-spec.md` — THE
specification and the reference for execution):

- **`pdum.dsl`** (`packages/dsl`) — type-keyed DSL compiler infrastructure:
  reflection capture, the two-tier type-keyed cache, the Node/Region IR with
  rewrite and lowering machinery, marshaling, the registry and its extension
  surfaces, the events seam, the value language (device functions over
  `is_bits` value types), the fuse pipe, and the reference evaluator (the
  always-spelled oracle: `reference(f)(...)`).
- **`pdum.tl`** (`packages/tensorlib`) — the assemblage tensor language:
  the exact layout algebra, carriers, the compute primitives, the
  Program/Instr IR, reverse-mode AD with derived adjoints, the transforms,
  the cost semantics, placement, and the model zoo. (Promoted from
  `explorations/` at migration P2.)

Design history (010–195) lives in `docs/design/history/`; git history is the
archive. Distilled backend knowledge: `docs/design/210_backend-notes.md`.
