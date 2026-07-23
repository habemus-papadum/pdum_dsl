# PLACEMENT — the L3-lite design note

Decisions from the design conversation (2026-07-23), recorded before code:

- **D-A — binding lives on `Dim`.** A dim is machine-bound via a `level`
  field (like charts/labels: one more per-dim annotation over the unchanged
  lattice). Everything the alignment machinery compares stays in one place,
  which matters because alignment IS the collective-inference engine.
- **D-B — no new semantic ops.** reduce/repeat/split/merge applied to
  machine-bound dims ARE the collectives; only the COST semantics interprets
  them as communication. reduce over a bound dim = all-reduce; merge of a
  bound part = all-gather; repeat-then-bind from replicated = free
  distribution; rebinding = the composite of its parts. "Collectives are
  alignment fixes" is thereby literal: the misalignment diagnosis's existing
  recipes acquire costs when dims are bound.
- **D-C — one mesh level now, depth-ready schema.** The `Machine` is a
  tuple of levels (cluster → … → lane capable); v1 exercises a single
  "gpu" mesh. Nothing in the binding, alignment, or traffic machinery
  assumes depth-1 — the same move applies at any tree depth later (L4/L6).
- **D-D — global view.** One program describes the global computation
  (GSPMD's logical view). No per-device SPMD extraction: the reference
  interpreter runs the placed program UNCHANGED, computing global values.

**The erasure invariant** (the L3 well-formedness contract): forgetting
bindings (and re-merging split dims) yields the L0 program with the
IDENTICAL denotation. Placement is cost-bearing metadata, never meaning.
Corollary: `run` needs no changes, and assurance tier 1 is one comparison.

**Surface discipline**: machine-bound dims are addresses — chartless,
unlabeled (enforced at `Dim` construction and in `bind`). The physics stays
on semantic dims; axis identity survives splits (LEVELS.md).

**Traffic model (v1 formulas, per participating device, p = mesh extent):**
- all-reduce (reduce over a bound dim): 2·(p−1)/p × result-local bytes
  (ring). The result drops the bound dim and is replicated — exactly the
  value semantics of reduce.
- all-gather (merge of a bound part): (p−1)/p × merged-global bytes.
- distribute (repeat + bind from a replicated source) and shard-view
  (split + bind of data every device already holds): 0 bytes — closed-form
  placement, the mesh analogue of "masks are free."
- α-β time and link topologies are cost-model refinements, not IR.

**Loud refusals (D17 at L3):** lattice surgery on a bound dim (slice/
shift/select/decimate/window/stencil/flip), scan or fold along a bound
dim, and bound levels absent from the machine (or mesh extents exceeding
the level count) all raise in the traffic pass rather than guessing.

**Placed backward (landed after v1):** gradients carry their primal's
placement, by the same construction that carries charts — every cotangent
contribution is restamped, and `bind` joined the restamp; backward
`repeat`s that re-create reduced mesh dims re-declare their binding. The
consequences are the point:

- backward collectives are just adjoint reduces: the Megatron block's
  joint program shows 6 all-reduces — the forward pair plus one per
  broadcast chain (q, k, v, mlp-up). The reference is UNFUSED: Megatron's
  f/g conjugate operators fuse attention's three input-gradient reductions
  into one; collective fusion is a recorded later optimization, not a
  modeling error.
- data parallelism falls out: bind the batch dim and the weight-gradient
  reduction over batch IS the gradient all-reduce (repeat† = reduce).
- sharded weights get sharded gradients, replicated weights get
  replicated gradients — no ZeRO-style resharding is modeled yet.

**Out of scope (recorded, not forgotten):** collective fusion (above),
overlap/latency hiding (L5), sub-GPU tiers (L4), uneven sharding
(divisibility, as with `split`), optimizer-state/ZeRO (no optimizer),
per-device program extraction.

**Flagship validation:** a Megatron-style tensor-parallel transformer
block (zoo/megatron.py): heads and MLP width carry a mesh dim `g` bound to
"gpu"; the traffic pass must report EXACTLY two all-reduces per block
(after the attention output projection and the MLP down projection), the
forward must match an independent numpy denotation, and the per-device
peak (peak_memory with `machine=`) must drop accordingly.
