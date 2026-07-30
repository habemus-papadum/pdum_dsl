# 330 — The fusion pass: a recognized-schedule registry in a measured loop

**Status: DRAFT — owner-ratified direction (2026-07-30); implementation
not started.** K-A's "kernels as grouping annotations" becomes concrete:
the pass that partitions an assemblage region into fusion groups and
stamps each with a kind (compute or tile). 320 built the target language
and its certification; this records how groups are chosen, why the pass
refuses rather than degrades, and the law that everything — analysis
included — lives in a hot loop behind content keys.

## §1 Three sub-problems, three characters

Fusion partitioning (minimize materialized bytes subject to lowerable
groups) is NP-hard — there is no optimal pass to write, and the design
embraces that instead of hiding it. The problem decomposes:

- **Legality is CRISP.** In a pure, name-based IR with token-ordered
  effects there is no dependence analysis: the SSA dataflow is the
  dependence structure. A group is legal iff its subgraph re-expresses at
  a target kind and passes the tier gate. Two targets exist, and the
  choice is STRUCTURAL, not heuristic: reuse (repeat_like's stride-0
  dims feeding a reduce = contraction; overlapping shift/slice
  footprints = stencil) selects the tile kind; its absence selects the
  compute kind, where coalescing is the whole game.
- **Profitability is COMPUTABLE.** Fusion's value in the memory-bound
  regime is the bytes it stops materializing: L1 (memory.py) gives exact
  bytes, the traffic model (320 §5) gives exact granules, the machine
  table gives capacities. The fan-out question — recompute a shared
  producer in both groups or materialize once — is the checkpoint
  min-cut (transforms.py), the same mathematical object with the same
  exact capacities. White-box layouts are the edge: frameworks autotune
  where we compute.
- **Selection is MEASURED.** Only here does the machine get a vote, and
  the loop (§5) is built for it.

## §2 The registry: strong-or-refuse (the ledger philosophy, applied to performance)

The pass is NOT a general optimizer that degrades gracefully. It is a
CLOSED REGISTRY of recognized schedules — templates known to lower to
strong code — and a loud refusal for everything else. V1 templates,
covering the zoo:

1. **Map chain** — pointwise/layout/iota subgraphs → one map kernel
   (compute kind; tile when a staged operand pays).
2. **Contraction group** — the rl/mul/reduce idiom plus pointwise
   prologue and epilogue → the GEMM tile template with fusion hooks.
   The epilogue (bias, activation) rides the accumulator out of
   registers and never touches memory — the same structure a
   cuBLAS-epilogue or hand-Triton author writes.
3. **Row normalization** — reduce-max/sub/exp/reduce-sum/div chains →
   the online-softmax template; contraction + normalization +
   contraction composes to the flash template (the measured 50x flash
   gap was STRUCTURAL — templates capture exactly the structure).
4. **Stencil group** — shift/slice/pad neighborhoods + pointwise → the
   stencil tile template.
5. **Everything else** — take/scatter_add/argtopk groups (K-G),
   unrecognized reductions, exotic folds — **REFUSES**: flagged
   uncompiled, served by the reference or a framework column, visible
   in the plan report. Never silent mediocrity.

Every template is a quadruple: matcher (over the ~20-op vocabulary),
generator (emits a TILE-TIER REGION — never source), license set, and
certification hook. Because generators emit tile regions, every fusion
product runs through `certify` against its own unfused subgraph —
proved by content key where normalization reaches, license-gated
differential where it does not. The pass may be aggressive precisely
because its outputs are never trusted, only certified.

**Confidence is a report, not a hope**: green = certified and measured;
yellow = certified, analytic cost bound only; red = refused
(uncompiled). The plan carries the colors.

## §3 Plans are data

A fusion PLAN is an inspectable object: the partition (node -> group),
a template assignment per group, and per-group parameters (tile sizes,
staged operands). Plans are diffable, cacheable, and comparable — never
compiler-internal state. The v1 PLANNER is deterministic
anchor-and-absorb: grow groups outward from contraction/reduction/
stencil anchors, absorbing pointwise and layout producers/consumers;
each absorption is gated by L1's exact byte savings under the machine
table's capacity — absorb iff a materialization disappears and the
group still fits.

## §4 Everything lives in a hot loop: the analysis cache

The owner's law, recorded as ratified: ANALYSIS IS A CACHED, COMPOSABLE
CITIZEN — the content-door discipline extended from artifacts to facts.

- An analysis result is keyed by (the content keys of its inputs, the
  analysis's own fingerprint, its parameters). First run computes —
  slow is fine; every later run in the hot loop is a lookup and SKIPS.
- An analysis's inputs may include OTHER ANALYSES' keys, so analyses
  compose into a DAG: a second analysis starts from a cached first, a
  refinement starts from the coarse pass it refines, and a tree of
  partial analyses is a valid, resumable state — the prefix never
  recomputes.
- **There is no invalidation problem, by construction**: regions are
  immutable and content-addressed, so a changed program is a DIFFERENT
  key. Staleness cannot be spelled. (Query-architecture compilers spend
  most of their machinery on invalidation; immutability deletes it.)
- **A measurement is an analysis whose evaluator is the machine**:
  keyed by (artifact key, machine identity), attached forever — the
  performance ledger and the analysis cache are ONE mechanism. Nothing
  keyed identically is ever measured twice.
- The warmth law extends: analyses emit through the events seam
  (`analysis.miss` / `analysis.hit`), and a warm loop asserts ZERO
  misses — the same `events.forbid` discipline that pins zero
  recompiles today.

## §5 The loop (and the runtime interface it implies)

propose -> certify -> measure -> revise. The planner proposes a plan;
each (group, template, parameters) compiles to an artifact keyed by
content + license set + machine table; the rig's discipline
(verify-before-time, warmth pins) measures; measurements attach to
artifact keys; the planner revises — tile-size sweeps are a small
discrete search per template. The loop is incremental and resumable at
every point, with a valid partial plan in hand.

The implication for the runtime negotiation, stated plainly: the
interface should traffic in PLANS and KEYED ARTIFACTS WITH ATTACHED
MEASUREMENTS — never a magic `compile()` that answers everything at
once. Compilation is a resumable conversation with the machine, and
every party (the planner, the runtime, the human) reads and writes the
same cached, keyed facts.

## §6 Honest limits

For the recognized set, the hand-written-framework bar is reachable
because STRUCTURE dominates and tile sizes are a small measured search.
Not immediately matched: exotic within-template schedules (persistent
kernels, split-K, warp specialization — within-template refinements
later) and vendor-fused libraries (cuDNN attention can beat generic
Triton). The BASELINE columns keep those gaps permanently measured;
the confidence report marks them, never hides them.

Lineage, for the record: XLA's anchor-grow fusion with an exact cost
model instead of heuristics; Halide/TVM's compute/schedule separation
in 320's form (program / plan / machine table); legality made trivial
by the pure IR; autotuning replaced by the keyed measurement ledger;
Salsa-style query caching made invalidation-free by immutability. The
egglog satellite remains the growth path if template matching ever
needs canonicalization beyond certify's normalizer.

## §7 Sequence

1. This document.
2. The analysis-cache seam (§4): keys, events, the one Memo discipline
   extended — small, and everything after rides it.
3. Templates 1–2 (map chain; contraction + epilogue) end to end:
   matcher -> tile region -> certify -> Triton -> measure. GEMM with a
   fused epilogue is the highest-value case and certifiable today.
4. Template 3–4 (normalization/flash; stencil), the refusal report,
   confidence colors.
5. The planner loop against the zoo; the tile-level perf rig grows the
   fused columns.

### §7.5 The partitioner — LANDED, with the backward census

pdum/tl/partition.py: anchor-and-absorb over whole-model regions.
Claims run most-specific-first over the anchors; absorption walks
epilogues downstream while the root has one consumer; interior members
may not leak (one output per group — the translator's law); residue
splits per externally-consumed root, promoted to a fixpoint. Free
views (consts, renames, repeat_like broadcasts, discovered the hard
way: the adjoint SHARES the forward's broadcast nodes) belong to no
group — they ride every claim that references them, duplicated at
rebuild, and boundaries punch through them to real values. Every
carve rebuilds CANONICALLY (params renumbered by first use), so
repeated layers collapse onto one content key: gpt2 at 12 layers
plans 234 carves onto 16 distinct kernels, and certification is paid
per kernel, never per layer. The carved plan executes BIT-equal to
the whole model (pinned).

Measured (this box): partition 2.9 ms and mapping 49 ms cold / 27 ms
warm for the 1166-node 12-layer gpt2; forward coverage 72.9% by
interior node count (15 contraction+epilogue shapes, the attention
softmax as row-normalization — riders ride, strict flash declines).

The backward census IS the roadmap, priced by frequency (toy joint,
682 interior): 48× reduce-of-computed-product (chart-wrapped adjoint
chains — needs the contract-over-computed-operands row, not a matcher
tweak), 9× mean-reducer chains (layernorm fwd+bwd — the row-statistics
row), 8× multi-dim contractions (wo and its grads), the softmax
adjoint composition, and scatter_add (embedding grad). Joint coverage
16.1% until those rows exist — refused loudly, named exactly.
