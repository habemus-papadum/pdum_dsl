# Beyond linear SSA — the representation ladder (pre-discussion notes)

Recorded from the memory-optimization pre-discussion (2026-07-22). Nothing
here is implemented; this is the map for the levels above the linear SSA IR.
(2026-07-23: LEVELS.md is now the umbrella — the full ladder to L6, the
machine tree, surface discipline, zoo, and order of attack. This document's
Levels 0–3 are LEVELS.md's L0–L3; the memory-level detail stays here.)
Scope note: this is about *intermediary* memory — activations kept for the
backward pass, buffer lifetimes, placement — NOT kernel-level concerns like
the naive matmul's materialized m×k×n product (that is fusion's job, one
level down, deliberately ignored here).

## What linear SSA lacks — exactly four things

1. **It conflates the dependency DAG with the schedule.** The linear order
   is one arbitrary topological sort; peak memory is a property of *which*
   sort you pick, and picking it is the optimization (min-register
   topological ordering ≈ pebbling; hard in general).
2. **SSA variables are values, not storage.** No notion of two values
   sharing a buffer, or of a value dying (no liveness, no `free`).
3. **Pure SSA cannot express recomputation.** Checkpointing computes the
   same value twice; that breaks name=value unless the representation
   allows duplicate computations of one semantic value.
4. **No sizes or costs.** Though here we are unusually rich: every value's
   exact byte size falls out of layout × dtype, and view-ops are statically
   identifiable as zero-cost aliases.

## The ladder

- **Level 0 (now)**: linear SSA, pure values. The linear order doubles as
  both DAG and canonical schedule. Right home for semantics and AD.
- **Level 1 — DAG + schedule + sizes**: same instruction set, plus (a)
  per-value byte sizes from layouts, (b) liveness / last-use (`free`
  events), (c) view-instructions marked alias-only, (d) the schedule as a
  separate, optimizable object. This is where checkpointing and scheduling
  live. First artifact should be *measurement*: a peak-memory simulator
  (walk the schedule, track live bytes, views free, report the high-water
  mark and its live set) — small, and it makes every later optimization
  falsifiable. It is also the seed of the Lean machine model.
- **Level 2 — bufferized IR**: explicit buffers (alloc/dealloc/offset),
  value→buffer assignment, in-place writes. The value-semantics →
  memory-semantics split is the hard-won MLIR/linalg lesson (bufferization
  as a pass). Our advantage: the white-box layout algebra IS the alias
  theory a bufferizer needs — `overlaps`/footprints/injectivity are already
  exact.
- **Level 3 — placed IR**: device meshes and sharding (below).

## Prior art map

**Checkpointing / rematerialization.** Theory anchor: Griewank & Walther's
`revolve` — provably optimal binomial checkpointing for *chain-shaped*
programs (O(log n) memory for n steps): directly applicable to transformer
layer stacks and PDE time-stepping. General DAGs: NP-hard; practice is
Chen et al. 2016 (√n heuristic), **Checkmate** (MILP over schedule ×
rematerialization; near-optimal at real sizes), Rockmate/Rotor (DP on
mostly-sequential graphs), and — most immediately implementable for us —
the **min-cut formulation** (AOTAutograd's partitioner): choose the
saved-for-backward set as a minimum cut between forward inputs and backward
consumers, edge weights = tensor bytes. With exact sizes from layouts, this
is a clean small first optimizer.

**Buffer reuse.** Given a fixed schedule: classical liveness + interval-graph
coloring (poly-time for straight-line code), then offset assignment (strip
packing; XLA/TVM-style memory planners). The unifying theory for both
questions is **pebbling**: this whole problem is black pebbling of the
forward+backward DAG with recomputation allowed (Sethi; PSPACE-complete in
general — hence structure-exploiting heuristics). The red-blue variant
(Hong–Kung) is the I/O-lower-bound theory that becomes relevant at the
kernel level later; nice that one formalism spans both levels.

**Sharding / parallelism.** GSPMD-style per-dim sharding annotations,
Megatron tensor parallelism, ZeRO state sharding, Alpa's ILP+DP automated
search, DTensor named sharding.

## Where this design has something distinctive to say

1. **Some tensors are free to save.** Standard checkpointers treat every
   tensor as an opaque blob. We know *structurally* that iota,
   masks-held-as-guards, constants, and views of already-saved tensors cost
   zero bytes — FunctionalBuffers and guards are closed forms, not memory.
   The rematerialization cost model reads this off the representation
   (attention masks, positional data, broadcast operands drop out of the
   activation budget entirely).
2. **"Which gradients do I care about" is DCE on the joint DAG.** Prune
   backward nodes not reaching requested gradients; the saved-activation
   set shrinks automatically and *correctly* (frozen early layers still
   save exactly what downstream-flowing gradients need — the DAG encodes
   this with no special cases). Run DCE before any memory planning.
3. **Sharding is the layout algebra wearing a mesh.** A sharded tensor is
   `split` of a dim into (mesh-axis, local) where the mesh axis is a
   *labeled* dim bound to devices; a sharding spec is a dim-name →
   mesh-axis binding; and **collectives are alignment fixes**: an
   all-gather / reduce-scatter is exactly what the misalignment diagnosis
   prescribes when operands disagree about which mesh axis a dim lives on.
   D17 (diagnose, never silently repair) extends verbatim to distributed
   placement, with a communication cost attached to each recipe.

## Order of attack (when we get there)

1. ~~Level-1 annotations + the peak-memory simulator~~ (memory.py).
2. ~~DCE for requested-gradients pruning~~ (transforms.dce).
3. ~~Min-cut saved-set selection~~ (transforms.checkpoint — sizes exact;
   iota/const free, views uncuttable, reduce/scan/fold banned from
   recompute by default; recompute placed just-in-time in the backward).
4. Schedule search / revolve for chain-shaped segments (the "functions as
   linear sequences" organizational layer gives the segments).
5. Liveness coloring + offset assignment (Level 2).
6. Mesh-labeled dims + collective insertion via alignment diagnosis
   (Level 3) — likely its own design conversation.
