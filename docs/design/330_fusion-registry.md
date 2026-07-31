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

### §7.6 The backward rows — RATIFIED 2026-07-30 (owner conversation)

The census classes collapse under one new POWER plus one new row, a
relaxation, and one honest refusal. Profiled under the actual claiming
law (not raw cone-walking), the five classes and their rulings:

**A. Computed operands — upstream absorption (the ruling).** The
mirror of epilogue absorption: a contraction's operand prologue joins
the claim through the EXACT vocabulary (pointwise, views, charts,
consts, iota) while every consumer lies inside the claim — a fork is
a boundary, because someone else needs that value materialized anyway.
Measured on the joint region: absorbed prologues are SHALLOW (1–2
pointwise for 53 of 72 sum-of-mul sites, max 12); the scary deep cones
were an artifact of the adjoint sharing forward nodes, and shared
means forked, and forked means boundary. Prologue recompute per
k-tile is BIT-exact — stronger than a license: pointwise per-element
work has no order to reassociate. Epilogues needed licenses because
they moved across reductions; prologues move into the staging of
operands and touch nothing ordered. Cost is closed-form: rowsum-shaped
sites recompute NOTHING (each element used once); gemm-shaped sites
recompute an operand tile once per program on the other grid axis —
traffic saved vs FLOPs added, both sides priced by the existing
traffic model. Under v1's single-dim launches the multiplier is 1, so
absorption is unconditionally sound today; the scorer inherits the
decision when launches grow a second dim. The contraction row also
accepts the ROWSUM shape (same-space operands, no broadcast pair —
the softmax adjoint's rowsum(dP*P)), which lowers as mul+tl.sum and
never touches the dot path.

**B. Row statistics.** The zoo spells layernorm TWO-PASS (mu =
mean(x); sd = sqrt(mean(xc*xc) + eps)) — the stable form, structurally
softmax's two-sweep shape. The row generalizes row-normalization
rather than duplicating it. The translator gains f=mean as fold-sum
with a divide-by-N finalize — N static, one deterministic scalar op;
numerics sit inside the EXISTING reassociation class every reducing
certificate already prices. No new license category anywhere in §7.6.

**C. Multi-dim contractions** (wo over (nh, hk) and its grads): fold
over one contracted dim, reduce the rest inside the step — what a
hand author writes. Reduction-order change is the already-priced
class. Flatten-to-single-k is a measured upgrade, not v1.

**D. The softmax adjoint composes** — rowsum(dP*P) is a computed-
operand rowsum, the outer P*(dP - ...) is the epilogue rule that
exists. No new template; verified by re-census after A lands.

**E. scatter_add is REFUSED red** (take's adjoint, the embedding
gradient): colliding writes across programs are the same act as
split-K — cross-program reduction, 340 §6's family. Re-entry: a
sort-and-segment spelling, or an atomics row with a measured
collision model. One site; it costs almost no coverage.

**Vocabulary ruling:** tl.with_charts / tl.strip_charts / tl.simplify
are FREE views in the partitioner (metadata never blocks a claim,
duplicated at rebuild) and legal anywhere in a carved kernel — the
translator already passes them through.

**The horizon, named and not smuggled in:** flash-backward proper
(recomputing attention inside the backward kernel instead of
materializing P) is cross-GROUP rematerialization — a memory-vs-FLOPs
law v1 does not have. Until it exists the backward materializes the
t×s attention matrix, and that is the honest cost.

### §7.7 Size-capped certification — RATIFIED 2026-07-30

The certificate is about the PROGRAM, not the size. Certification runs
differential families through the python reference; at T>=4096 a
flash region materializes T^2 softmax through that reference and the
OOM kills the session (observed twice). Ruling: above a cap, certify a
SHRUNK TWIN — the same match and the same generator over the region
with extents clamped — and attach that certificate to the full-size
kernel. The families still exercise every structural failure mode the
template declares (the mask still masks, the rescale still rescales);
what they no longer exercise is size itself, which the certificate
never priced anyway. The cap is a plan-level constant, not a machine
fact.

### §7.6–§7.7 LANDED (2026-07-30, three commits), measured

The census classes closed as ratified. Toy forward: 72.9% -> 98.2%
coverage (only the embedding gather red). Toy joint: 16.1% -> 97.9%.
The 12-layer joint (4367 nodes, 2365 interior): partition 13.3 ms,
mapping 6.3 s cold / 192 ms warm, 897 carves onto 59 distinct
kernels, 98.4% coverage. Red, named: the softmax max-orphans (the
adjoint shares the forward softmax's interior, so the row-norm claim
leaks and its pieces claim separately — sum and div claim, max has no
row), the scans, and scatter_add (E's refusal). Landed along the way:
the PLAIN shape (bias gradients — a sum with no product, the
degenerate rowsum); certificates as FACTS (certify.certificate rides
the cache keyed by the twin pair — 12-layer warm mapping 202 ms ->
30 ms); the epilogue walk no longer crosses into a repeat_like of its
root (broadcast-into-elsewhere is another anchor's product). On
silicon, four §7.6 proofs: prologue-inside-the-sweep (ieee dot),
softmax adjoint as rowsum+epilogue, layernorm with the '/ N' finalize,
and the (nh,hk) two-dim contraction — tripwire silent throughout.

### §7.8 Recompute-vs-materialize — RATIFIED 2026-07-31 (owner conversation)

The flash-backward law, stated generally: **the statistics stay, the
map travels.** Sort any re-derivation cone by three kinds — map work
(recompute-exact, §7.6's class), TILE-LOCAL reductions (dims inside
the tile, e.g. the e-contraction: recomputable, bit-exact at matched
block shape, the priced reassociation class otherwise), and SWEPT-DIM
reductions (m, den — row-global, unrecomputable tile-locally BY
CONSTRUCTION). The rulings:

1. **Artifacts** are fold-carried finals surfaced as extra stores —
   the ONLY multi-output form. Zero extra FLOPs (the sweep already
   computed them), O(rows) extra bytes. den is already surfaced for
   the division; m is carried and merely never stored.
2. **The remat law**: an outside consumer of a claim's interior X is
   legal iff X re-derives from {params, artifacts} through map work
   and tile-local reductions; the cone COPIES into the consumer
   (content-addressing dedups); swept-dim reductions never travel —
   they are what artifacts exist for. §7.6's fork law survives as the
   special case where nothing travels (pure-map forks — gelu — travel
   with no artifacts at all: activation checkpointing falls out).
3. **Capacity forces, the ledger picks**: materializing t^2 above the
   capacity wall is a COMPUTED refusal (forced remat — the T=512
   shared-memory refusal's sibling); below the wall, remat-vs-
   materialize is a measured tie-break. No new machine columns.
4. **No new template, no atomics**: the backward kernels are §7.6
   contractions whose prologues re-derive P per tile; dQ folds s and
   grids t, dK/dV the transpose — separate carves, every output owned
   by one program; cross-program reduction stays refused with its
   §6 wording. v1 recomputes P per consumer (3–4x vs FA's 2x); the
   D = rowsum(dO*O) identity and accumulator distribution are
   ALGEBRAIC merges — re-entries, entered on a measured gap.
5. **Certificates** run against the naive joint under the flash
   license domain — the saved statistics are the ONLINE (m, den),
   whose deviation from naive is already the priced class.

Flagship: the bare flash_tile joint (riders are the queued, orthogonal
matcher extension). Bounds transfer (340 §4b, roles swapped) rides
behind.

### §7.8 LANDED (2026-07-31), with three laws the flagship forced

The flagship (bare flash_tile joint, T=8 toy): the partitioner
DISCOVERS the FlashAttention-backward structure — flash-with-artifacts
(o, m, den from ONE sweep; 3 stores, pinned on silicon) plus nine
contraction carves, four of which re-derive P per tile (dV, D, and
the dS pair), one red scan. Every carve certifies; the carved plan
executes on the reference to the naive joint at 1e-9; dV runs on the
4090 with exp and the score-sum INSIDE its sweep — P never touches
memory. Implementation surfaced three laws, all in the partitioner's
docstrings: the PUNCH-THROUGH law (free views forward values — the
fork and leak checks follow consumers through shared frees; a chart's
membership in two claims proves nothing); the ROOT EXEMPTION
(consumption through a claim's own root is the export, never a leak);
tl.repeat is FREE (repeat_like's literal-extent twin — classing it as
compute materialized a (t,o,s) broadcast). Translator additions:
multi-store tuple yields, _reduce_at (a reduce re-emitted at ambient
composed coordinates — how tile-local reductions travel), and lazy
composite-marker expansion (autodiff's f.dN slopes; eager operand
emission left dead loads in the kernel).

**The measured board WAITS on one emission capability, named:**
_reduce_at currently lowers as mul+tl.sum, whose 3-D intermediate
outgrows the register budget past toy tiles — the dot-at-ambient
emission (tl.dot at composed coordinates, the tensor-core form every
hand FA kernel uses) is the priced next step, and the remat-vs-
materialize tie-break rig lands with it. Bounds transfer (340 §4b,
roles swapped) rides behind. Both are follow-ups, not blockers: the
LAW is landed, certified, and pinned end to end.

### §7.8 AMENDMENTS (2026-07-31, ratified as options 1–2 of the sequence)

**Dot-at-ambient (the emission gap closes).** _reduce_at gains the dot
fast path: a two-operand product-sum whose cores are 2D re-emits per
tile as an ieee tl.dot at the ambient composed coordinates — kept tile
widths read off the coordinates' own arange, sub-16 or odd shapes fall
back to mul+sum (toys unchanged). With _uncore (operand cores peel
charts, broadcasts, AND projection markers — autodiff's mul.d0/mul.d1
slopes are bare Args, so the node IS one of its operands, typed at the
joint space but never work), the backward kernels emit the FA inner
loop: two ieee dots with the exp between them. propose() ladders
EVERY output dim (dV grids its big axis; tuple yields use the output's
dims; leading-dim candidates first so old picks keep winning).

**Root-travel (the law completes).** A carve whose root serves ONLY
other claims, and whose every reduction — travel copies included, the
copy must be legal where it finally RESTS — is tile-local for each
consumer's sweep, dissolves: members join the travel set, consumers
re-derive per tile. Runs to fixpoint, pure-map pieces first (a rowsum
must not ride into a piece that could otherwise dissolve whole). Two
guards carry the ratified rulings: a region-yielded root stays
materialized; and the TRAFFIC GATE (ruling 3's analytic default) —
dissolve only when the root's round trip exceeds every eater
re-reading the cone's boundaries, so at toy sizes materialization
honestly wins and nothing dissolves. What the gate discovers at real
sizes: dP and dS dissolve into dQ/dK/dV (full FA-2 structure — only
the O(t) statistics and D cross memory), D's rowsum-over-s is held
back from dQ's s-sweep AUTOMATICALLY (FA's precomputed per-row D,
derived not designed), and in the FORWARD the score contractions
dissolve INTO the softmax claims with their mask forests riding — the
flash direction, discovered by arithmetic. The one refusal left in
the flash joint: the max-adjoint's tl.scan (argmax tie-breaking),
reference-served, erased only by softmax shift-invariance — an
algebraic rewrite, future.

**The first §7.8 board (T=512, E=64, OD=64, causal, f32, 4090 laptop,
measured picks over the ladder, every candidate verified against the
reference before timing).** The FA-shaped kernels: flash forward with
artifact stores and causal prune 201 us (sdpa fwd: 128 — 1.6x); dV
186, dQ 305, dK 310 — the three gemm-shaped backward kernels total
~800 us against sdpa's whole backward ~390. The itemized remainder:
the two row-statistics kernels (D and the den-grad rowsum) cost 591
and 1888 us — their rank-1-output emission has no dot form yet and
spills; and two (t,s) pieces stay off the GPU (the scan-tainted
max-adjoint, reference-served red; the rowsum-blocked dS piece,
launch-infeasible). GPU total 3.5 ms vs sdpa 0.52 ms — the 6.7x is
carried almost entirely by the named gaps, not by the FA kernels.
Learned along the way and fixed in the bench: prune affines are
TILE-specific (a plan's prune must never ride another candidate's
launch), and a measured pick must reject numerically-wrong candidates
before comparing speed.
