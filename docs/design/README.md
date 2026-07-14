# docs/design — the numbered canon

Read in numeric order. The scheme (adopted 2026-07-12; this folder is part of
the mkdocs site):

- **`010_proposed-architecture.md`** — the **master document**. Always kept
  current; every note below is referenced from its "Companion documents"
  list. Where documents disagree, 010 wins.
- **`020_implementation-plan.md`** — the step sequence, gates, and the book
  meta-pattern.
- **`022_closure_specialization.md`**, **`024_dsl_caching_layer.md`** — the
  pre-M0 **evidence analyses**: why numba cannot reuse closure
  specializations, and the caching-layer hazard checklist the kernel must
  satisfy. Older than the redesign, still normative.
- **`030+` — topic notes**, numbered in the order of the book material they
  serve (gaps left for insertion):
  - `030_deep-learning-notes.md` — the differentiable-programming satellite.
  - `040_combinators-notes.md` — pipelines, roles, composition rules, the
    bracket config contract (§3c), DPS/outputs.
  - `050_provenance_tracking.md` — source locations through the pipeline:
    the MLIR-lite algebra, the inherit-default, starting-region contract.
  - `070_backends-notes.md` — the backend detour synthesis: bridge
    ride-vs-own verdicts (own CUDA via cuda.core; own Metal; wgpu-py
    settled), the shader-family dialect layering, the compute invocation
    surface (explicit DPS, config schema, ping-pong chaining), the
    graphics draw surface, packaging/CI. Raw surveys: research/R12–R17.
  - `060_rendering-notes.md` — rich static notebook widgets: the
    fragment/style composability contract, CSS-only interactivity, the
    jsdom dev loop.
  - `080_backend-organization.md` — families vs targets vs cells (the
    sparse matrix), three-tier backend resolution, why the fused demos live
    at `dsl.demo`, the namespace-package + entry-point contribution contract.
  - `090_core-and-extensions.md` — the punning charter: core+extensions at
    the dialect AND runtime layers (vendor namespaces, capability flags,
    artifact protocols, the runtime's do/refuse list), stdlib minimalism,
    the buffer/tensor-interop contract step 11 consumes, and the
    multi-device testing ladder (fake-runtime conformance → probe-gated →
    cross-device).
  - `100_arrays-and-axes.md` — arrays & axes (step 11): the type algebra
    (rank-generic / ShapedArray / NamedArray), the pedantic indexing
    decision (mandatory `isel` on named arrays), marshaling through the
    leaves channel, statement policy (strict joins, single tail return,
    bounded loops), the C target, and the recorded scope cuts.
  - `110_transforms-and-derivatives.md` — transforms (step 12): the spike
    finding (SIMT weaving, not SIMD widening — control flow needs zero
    transform machinery), named-first vmap, the one tangent engine behind
    `jvp` and the in-kernel `D` (analytic shader derivatives), and named
    contraction with batching for free.
  - `120_events-and-instrumentation.md` — **implemented** (§0 is the
    as-built reference; §§1–11 the original proposal, kept as rationale):
    the kernel event seam (`emit`/`span`/`forbid`) generalizing the ad-hoc
    counters and `no_compile()`, the recorder satellite (`events.record()`
    / `expect()`, interned structured tracebacks, per-name sampling, the
    per-thread phase tree), the `Memo` primitive, the traced-dispatch twin
    that made the miss path observable, the budget-policy change (caps are
    tripwires; 1150 → 1500), and the deletion of `bench.py`'s
    live-cache-entry monkeypatch.
  - `130_tensors-tiles-and-over.md` — **proposed** (the fork point): axes
    as IR lowered late — `over` (vmap renamed) emits map-loop IR instead
    of erasing the axis; arrays become IR values (DPS results, args,
    elementwise tiles); the named tensor dialect (`map`/`reduce`/
    `contract` + comprehensions); memory spaces, layouts, effect tokens,
    precision kinds; the tiled-GEMM syntax exercise; the transform seams
    (context, composition, DerivedValue) under 120's tripwire policy; the
    four-stage plan.

Also here, unnumbered: `research/` — the frozen 20-agent research corpus
(R/V/P/J) behind 010, plus later targeted surveys appended in the same
R-series (R10/R11: construct-level language surfaces, feeding the book's
ch07a lay-of-the-land interlude). Records, not living documents; paths inside them may predate
later renames. Historical motivation material lives at the repo-root
`archive/`, outside the site.
