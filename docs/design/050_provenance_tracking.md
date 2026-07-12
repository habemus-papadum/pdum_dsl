# 050 — Provenance tracking (source locations through the pipeline)

**Status:** schema committed 2026-07-12 (step 5 era); policies land at their
chapters. Companion to `010_proposed-architecture.md` (ledger entry same
date). Implemented in `kernel/ir.py` (the algebra, builder default, error
rendering) and `kernel/rewrite.py` (driver integration).

## The topic and its calibration

Two consumers walk backwards from artifacts to source: **diagnostics** (a
missing cast should name the source points involved — the operator site and
both operands' origins) and **performance attribution** (a hot generated
kernel should name the source that built it). The unusual calibration that
shapes everything: the primary consumer is an **advanced AI programmer**. It
needs a *correct starting region*, not IDE-grade precision — it will write
its own throwaway probes and take its own measurements from there. Full
fidelity (the LLVM/DWARF road: line tables maintained through every
optimization, "line 0" conventions, a permanent bug class) is explicitly
rejected as intractable-for-the-value.

**The written contract: starting region, not ground truth.**

## Prior art consulted

- **MLIR** (the steal): `Location` as a small recursive algebra —
  file/line/col leaves, `CallSiteLoc` (inlining chains), `FusedLoc`
  (merges), unknown — with the rewriter defaulting new ops to the replaced
  op's location. That default is why locations survive hundreds of passes
  with zero per-pass effort.
- **JAX**: per-equation source info with *user-frame filtering* — the
  starting-region philosophy in production.
- **Source maps (JS)**: generated line → original loc as a side table by the
  artifact — the model for our text-rendering backends.
- **LLVM/DWARF**: the cautionary tale (see above). **tinygrad**: ignores
  locs, leans on its rewrite debugger — the road declined.

## The schema (committed — the hard-to-refactor half)

1. **The algebra** (`ir.py`): `Loc(file, line, col)`;
   `CallLoc(callee, caller)` — non-negotiable here because monomorphic
   inlining is this system's entire execution story ("wave.py:5, inlined
   from art.py:40"); `FusedLoc(locs)` for merges; `None` is unknown, legal
   everywhere. All provenance is `compare=False` — **never** in structural
   identity, never in content keys ("where code came from is not what it
   is"), and the anti-pattern gate polices provenance types too (no
   `object`-typed fields reachable from `Node`).
2. **The driver default** (`rewrite.py` + `Builder.default_loc`): while a
   rule's replacement function runs, the Builder's default loc is the
   replaced node's — so **freshly emitted nodes inherit provenance
   automatically**, and rule authors never think about it. Crucially,
   **survivor nodes keep their own story**: a rule returning an existing
   node (`x+0 → x`) does not restamp it — object identity and sharing are
   preserved (stamping survivors would have broken DAG sharing; caught in
   design). Rules wanting richer provenance (const-fold → `FusedLoc` of both
   operands) may set it explicitly.
3. **Errors render the chain** (`Builder.emit`): a type-rule failure carries
   `[site; operand; operand]` points via `format_loc` — the strict-core
   missing-cast error becomes a multi-point starting region.

Measured while deciding: a 400-firing rewrite pass runs in ~1.4 ms;
loc-stamping via the builder default costs nothing extra (no allocation —
the loc rides the normal emit path).

## Policies (the incremental half, at their chapters)

- **Lowering (ch07):** every emitted node stamped from its AST node, rebased
  to absolute file/line via the `SourceSnapshot`; inlining (device calls,
  the combinator build rule) wraps inlinee provenance in `CallLoc`.
- **Backends (ch09/ch10):** renderers return a side table
  `generated_line → provenance` stored on the artifact (source-map-lite),
  plus optional `// ← art.py:12` comments behind a flag. Profiler output
  attributes through a join. **Kernel-level attribution is already free**:
  `FastRecord → template → SourceSnapshot`, and pipeline `Derived`
  identities name their stages.
- **Diagnostics polish** (ongoing): JAX-style user-frame emphasis in
  rendered chains as they get deep.

## Anti-goals (documented)

No DWARF. No mandatory-loc invariant (`None` legal everywhere). No fidelity
promises through aggressive rewrites beyond the inherit/fuse defaults. No
exactness claims for post-fusion profiling attribution. Precision beyond the
starting region is the consumer's job, by design.

## Why the split commitment is safe

The **schema** (what `Node.loc` can hold; the driver default; the
outside-identity rule) touches every producer ever written — that is the
part committed now and meant to be lived with. The **policies** (what gets
stamped where, table formats, message rendering) are ordinary incremental
work: adding a renderer side table at ch10 changes no schema, and improving
error prose never breaks a cache key, because provenance can never touch
identity.
