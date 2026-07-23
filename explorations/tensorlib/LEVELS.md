# LEVELS — the representation ladder × the machine tree

The plan-we-don't-have-to-stick-to, written down 2026-07-23 after the
machine-modeling design conversation. REPRESENTATIONS.md holds the detailed
memory-level notes (its Levels 0–3 are this document's L0–L3); this is the
umbrella: the full ladder down to lanes, the organizing principles, the
model zoo, and the order of attack.

## The two artifacts (the anti-chaos principle)

**1. The machine description is DATA, not representation.** A machine is a
tree: cluster → node → GPU → SM → warp → lane. Each level has a count, a
memory (capacity, bandwidth, latency), and links between siblings. The IR
never says "warp"; it binds dims to *named machine levels*, and the machine
tree gives those names meaning. Swapping the tree retargets the model.

**2. The representation ladder is a sequence of NAMED LEVELS over ONE IR.**
Each level is a well-formedness predicate — "program P is at level ≤ k" —
the same untyped-IR + WF-predicate idiom used for the program IR and the
signature pass. Lowerings are functions L_k → L_{k+1}; every level preserves
the L0 denotation; every level adds a cost semantics (a second denotation
into a resource monoid — ops_count was the first). One IR, one denotation
for Lean, cost models as monoid homomorphisms.

**Corollary — distribution and tiling are the same move.** Sharding across
a GPU mesh, blocking into SMs, vectorizing across lanes: all are "split a
semantic dim, bind one part to a machine level, place the buffer in that
level's memory," at different tree depths. Collectives-are-alignment-fixes
applies at every tier: an all-gather across nodes and a warp shuffle are
the same alignment repair on different links.

## The ladder

Named by the question each level answers:

- **L0 — Denotation ("what").** Value SSA + layout algebra + AD + marker
  DSL + `fold` (tensor-state scan). Cost: ops_count. Exists.
- **L1 — Footprint ("how much").** DAG separated from schedule; exact byte
  sizes; liveness; views-as-aliases; recomputation expressible. Artifacts:
  peak-memory simulator, requested-gradients DCE, min-cut checkpointing,
  revolve on chain segments. (REPRESENTATIONS.md §Level 1, order of attack
  unchanged.)
- **L2 — Storage ("which bytes").** Bufferization: value→buffer assignment,
  reuse, in-place. Our layout algebra is already the exact alias theory.
- **L3 — Placement ("where").** The machine tree appears. Machine-bound
  dims (split + bind), buffer placement in a level's memory, explicit
  `copy` as the only byte-mover, collectives derived from alignment
  diagnosis, traffic-per-link cost semantics. Device meshes AND SMEM
  staging both live here — same mechanism, different depth.
- **L4 — Kernels ("who").** Partition the DAG into kernels; a kernel
  boundary forces materialization to the parent memory. Inside: tiled
  loops over machine-bound dims. Costs: parent↔child traffic, per-kernel
  register/SMEM footprint (occupancy proxy), launch counts. Objective at
  this level: minimize parent-memory traffic subject to child-memory
  capacity (the flash-attention derivation; Hong–Kung red-blue pebbling is
  the lower-bound theory — the same pebbling formalism as checkpointing).
- **L5 — Schedule ("when").** Time enters: streams, events, async copies,
  double buffering / software pipelining, warp specialization as
  producer/consumer roles. Cost: a timeline simulator giving roofline-style
  wall time — taking a SET of programs (concurrent workloads are in
  scope). Wall time only becomes a meaningful objective here; occupancy is
  a constraint/modifier, never the objective.
- **L6 — Microkernel ("exactly how").** Lanes, warp collectives,
  tensor-core fragments, coalescing, bank conflicts, swizzles.

Two structural facts about the bottom of the ladder:

- **L6 cost questions are layout queries.** Coalescing = "is the
  lane-bound dim's byte stride ≤ element size within a 128B segment"; bank
  conflict degree = multiplicity of `(addr/4) mod 32` over the lane dim —
  exact, static, computable from Layout today. CuTe's layout algebra is
  essentially ours with nested strides; XOR swizzles are the one known
  extension beyond affine.
- **Warp cooperation is reduce/scan over a lane-bound dim.** A shuffle
  tree is `reduce(sum, dim=lane)`; tensor-core `mma` is a contraction
  whose operands are lane-bound with published fragment layouts. Lowering
  must PRESERVE reduce/scan structure on machine-bound dims (not erase it
  into loops) so the backend pattern-matches instead of reverse-engineers.

## Surface discipline (split/merge are not physical)

Split/merge are computational choices, not physics — so **L0 surface
programs never split**. Splits are introduced by lowering (L3+), and the
parts they create obey:

- Machine-bound dims carry NO charts, NO units, no labels: they are
  addresses. The physics stays on the unsplit semantic dim.
- Axis identity survives: `x_blk` and `x_lane` both carry axis tag `x`;
  their strides compose back through the split arithmetic to x's chart.
  ("Charts live on semantic dims; splits factor through them" — a sentence
  to eventually state in Lean.)

## Assurance tiers (human-as-compiler + Lean)

Lowering is done by a human first. Equivalence assurance comes in three
tiers, cheapest first:

1. **Numeric spot-check** (exists): run both programs on random small
   inputs, compare. Probabilistic translation validation, free today.
2. **Algebraic normalization** (Python, soon): layout-op compositions
   normalize to canonical affine form — layout-level equivalence is
   decidable integer arithmetic. Compute-level reorderings check as
   applications of declared licenses (associativity etc.).
3. **Certified rewrite rules** (Lean, the destination): Lean proves each
   REWRITE RULE once (split∘merge = id under divisibility, guard/slice
   commutation, reassociation via declared associativity, fusion = elision
   of identity materializations — nearly free in a pure IR). Python checks
   that a lowering is a chain of certified rules. No monolithic
   "prove these two programs equal" obligation ever exists.

## The model zoo (test corpus for every level)

A spanning set by mechanism, not an exhaustive gallery crawl:

- **GPT-2** (MHA, LayerNorm, GELU, learned positions) — baseline canon,
  forward + backward.
- **Llama-style block** (RMSNorm, RoPE, GQA, SwiGLU) — GQA is `repeat` of
  KV heads; RoPE exercises paired rotation.
- **Norm/window variations** (QK-norm, post-norm vs pre-norm, sandwich
  norm, sliding-window attention) — sliding windows are GUARDS (masks are
  free); norm placement stress-tests AD.
- **Gated attention** (sigmoid gate on attention output, Qwen3-Next
  style).
- **Online-softmax attention** (flash accumulator: running
  (max, denom, weighted-sum) as a `defreducer` with the associative
  rescaling combine) — the flagship linking the marker DSL to fusion:
  L0 states the algorithm, L4 shows why the fused form wins.
- **Linear-attention / SSM block** (Mamba-2/DeltaNet-style gated linear
  recurrence) — via the tensor-state `fold` (below).
- **Physics:** 2D heat (explicit Euler, guards as BCs), 1D/2D FDTD on a
  **Yee staggered grid** (half-integer charts are exactly what exact
  rational charts are for), upwind advection. Exercise guards, charts,
  and stencil-chain fusion in ways transformers don't.

**Recorded boundaries (deliberate exclusions):** MoE routing / top-k
(data-dependent gather), KV-cache decode (mutation), dynamic shapes.

## Tensor-state scan (`fold`) — the gap two domains hit

Scalar composite reducers carry k SCALARS per lattice site. Mamba-2 /
DeltaNet carry a d×d MATRIX per step; PDE time-stepping carries whole
FIELDS. Same wall, one answer: **scan over programs**. The `fold` op:
state = named tensors with a fixed-layout carry contract, element = per-step
slices along the scan dim, step = an IR *Program* (programs as first-class
step functions — the IR's one structured combinator, still no branching).
`out=(emit, v)` stacks a per-step value along the dim; `out=(final, v)`
returns a final carry. The adjoint is DERIVED: differentiate the step
program (grad applied to a scalarized VJP wrapper), then fold the VJP
program in reversed time over the stored state trajectory — BPTT with
per-step recompute, generated by self-application of the existing AD.
Sequential by definition; an associative tensor COMBINE (the Mamba-2
chunking license) is a later declaration the compiler may exploit, exactly
parallel to `associative` on scalar reducers.

## Order of attack

1. ~~This document.~~
2. ~~**Tensor-state `fold`**~~ — landed (op + run/infer + derived BPTT
   adjoint + signatures + opcount; GLA and FDTD validated).
3. ~~**Model zoo**~~ — landed (`tensorlib.zoo`): gpt2, llama_block (RoPE/
   GQA/SwiGLU), sliding/gated/qknorm attention, flash reducer (backward
   DERIVED, matches naive analytically), heat2d, charted staggered FDTD.
4. **L1**: ~~peak-memory simulator~~ (memory.py) → ~~requested-gradients
   DCE~~ and ~~min-cut checkpointing~~ (transforms.py: exact-byte node
   capacities; closed forms free, views never saved, contractions banned
   from recompute; lazy just-in-time recompute placement; zoo GPT-2:
   boundary 47%, peak 76%, with DCE 64% of the joint) → next: revolve on
   chain segments; FDTD adjoint time-stepping as the fold-checkpointing
   use case.
5. **L3-lite**: machine tree + mesh placement + collectives-by-diagnosis +
   traffic costs; Megatron-style sharded GPT-2 as flagship. (L2
   bufferization lags deliberately; needed for exact reuse, not for
   placement reasoning.)
6. **L4**: manual fusion + tiling with the rewrite checker; flash
   attention and a fused stencil chain as flagships.
7. **L5/L6**: timeline/pipelining simulator; lane-level layout queries and
   warp-collective lowering.
8. **Lean package** starts when step 6's rewrites exist; hardware
   calibration is a parallel thread, not on the critical path.
