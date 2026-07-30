# 340: Grid & launch — the data-parallel quotient

Status: RATIFIED 2026-07-30 (owner conversation; three rulings recorded
in §2, §5, §6). Sequence in §7. Companion to 320 (tile tier) and 330
(fusion registry); consumes both.

The translated tile kernels run on a (1,) grid — one program, one SM.
The measured wall: at T=128 the pass-fused flash kernel is 2.5× behind
the hand gridded twin (53.4 vs 21.4 µs); at T=512 it refuses to compile
outright (231KB shared required vs 101KB). Nothing yet carves the
output space across programs. This document is the carving law.

## §1 The theory ledger

The same discipline as 330: say precisely what has a sound answer,
what is measured instead of modeled, and what is refused loudly.

**Sound (analytic, no measurement):**

- *Legality is syntactic.* An axis is griddable iff no fold carries
  state along it and no reduce consumes it — folds and reduces name
  their dims, so this is a one-pass walk. Flash grids over t, never s;
  gemm grids over (m, n), never k.
- *Correctness is license-free.* Gridding is a pure partition: every
  output element is computed by exactly one program running the same
  IR-level arithmetic. Unlike fusion (which reassociates under
  licenses), the partition itself introduces no deviation. Machine
  caveat, measured at landing: Triton lays out reductions by BLOCK
  SHAPE, so a reducing body's within-row sums reassociate when BT
  changes (flash grid-4 vs grid-1: 4.8e-7; identical oracle
  distance). The law therefore splits: reduce-free bodies are
  BIT-equal gridded vs ungridded (stencil, proven); reducing bodies
  must stay within their certificate's license tolerances — no NEW
  license, because block-shape reassociation is the class the
  certificate already prices.
- *Tile shape has a real theory.* Hong–Kung: a contraction at fast-
  memory capacity S moves Ω(W/√S) bytes, met by √S-square tiles.
  Stencils are governed by halo-to-interior surface/volume; row
  normalization wants rows-that-fit; map chains have no reuse, so
  tile shape is traffic-irrelevant (only the granularity floor
  matters). Each registry row carries its own closed form, a function
  of CAPACITY alone — the 320 §5 machine table, no new columns.
- *Feasibility is analytic.* Per-program staged footprint ≤ per-level
  capacity, computed by the traffic model before Triton ever sees the
  kernel. The T=512 crash becomes a computed decision: BT=512
  infeasible, BT=32 feasible and traffic-optimal. (The scorer is a
  model; triton's OutOfResources stays as the backstop tripwire.)

**Measured, never modeled (ledger facts under the 330 §4 law):**

- The granularity floor — minimum per-program work before fixed
  overhead amortizes. A per-device scalar, learned once by the rig,
  stored as an analysis fact. Never a hardcoded architectural datum.
- Tie-breaks among feasible tile shapes, num_warps/num_stages:
  autotune-LITE — the closed-form optimum plus its pow2 neighbors
  (3–5 candidates, not Triton's thousand-config search), measured
  once, cached forever by (kernel content, machine, launch).

**No theory — refuse loudly, re-entry conditions in §6:** split-K and
cross-program reductions; persistent kernels; grid rasterization
order for L2 locality (swizzling).

## §2 The launch is a plan artifact, not IR (ruling 1)

The binding law stays untouched: a tile region remains a per-tile body
with NO tile coordinate in the language. The launch lives on the plan:
the Group grows a `launch` field — an ordered tuple of
`(output dim, tile extent)` pairs; the program count is DERIVED
(product of cdivs), never chosen. The translator grows a prelude —
`tl.program_id` → tile coordinates → the same request-driven
coordinate composition, with launched dims' starts becoming affine in
the program id instead of constants. The language never does pointer
math; the launch layer is exactly the one place where tile coordinates
become addresses. Cost accepted knowingly: exotic schedules cannot be
*expressed* in the language until a §6 re-entry.

Request-driven emission makes this small AND makes stencils correct
for free: neighbor reads compose ABSOLUTE coordinates down to the
param load, so a pid-shifted slab reads its own halo with no halo
machinery — the domain guards already spell the boundary.

## §3 The partition law

At the IR level, per-element arithmetic is invariant under the
partition — grid axes are the independent axes, and slicing them
touches no reduction order. At the MACHINE level one caveat survives
(§1): Triton's reduction trees follow block shape, so shrinking BT
reassociates a reducing body's sums at the ulp level. The law as
pinned by the battery: **reduce-free kernels are bit-equal gridded vs
ungridded** (`assert_array_equal`, ragged tails included); **reducing
kernels stay within their certificate's tolerances**, gridded-vs-
ungridded agreement measured and asserted at license scale. Tile-SIZE
choices that reassociate the fold axis (si=8 vs si=32) belong to
fusion and its licenses, not to the grid.

## §4 Tile shape per registry row

Closed forms, capacity-relative, one per template (v1):

| template | grid dims | tile closed form |
|---|---|---|
| map-chain | all output dims | granularity floor only |
| contraction-epilogue | (m, n) | √S-square (three resident blocks) |
| row-normalization / flash | rows (t) | rows-that-fit: staged k/v + carried (m, den, o) ≤ S |
| stencil | interior (x, y) | square by surface/volume |

Candidates = the closed form ± pow2 neighbors, measured through the
ledger; the pick is a fact, the color turns green by measurement.

## §4b Mask-derived bounds (landed with the prune commit)

A closed mask does more than cost nothing to evaluate: it PROVES tiles
inert. Three separable pieces, so causal is one instance and not the
design:

1. **The emptiness engine** (launch.py `tri_eval`): three-valued
   interval evaluation of any closed forest over coordinate boxes —
   T/F where the algebra decides, M anywhere unspelled. Sound by
   construction: unproven means unpruned, never wrong.
2. **Template-declared inertness** (the template knows its step, as it
   knows its adversarial families): flash declares mask-false tiles
   inert — fill enters max as a no-op and exp underflows to exact
   zero, so the skip is BIT-exact (pinned). The legality edge is the
   m-chain law: a row with NO live tile has uniform-softmax
   semantics and refuses pruning outright (strict-causal's row 0,
   pinned).
3. **The bounds artifact**: per-program [lo, hi) as floor-affine
   coefficients — clamp((a + d·pid)//c) — fitted EXACTLY from the
   emptiness table (clamps and tile-ratio ceil patterns land on one
   line; a failed fit means no prune). Emitted as scalar clamps on
   the fold's range: causal at BT=SI becomes
   `range(0, tl.minimum(SO, 1 + pid))` — the bound a hand author
   writes, computed from the mask the user wrote. The analysis is a
   cached fact (`fusion.prune-flash` through the 330 §4 seam).

**The wave law (measured, 76-SM 4090).** Pruning converts FLOPs to
wall-clock only past ONE WAVE of programs: below ~1 wave every program
owns an SM and the longest (unprunable) program sets the clock —
measured neutral at 64 programs. At 1.7 waves: 1.40×; at 3.4 waves:
1.92× — approaching causal's 2× asymptote. Real workloads carry
batch/heads grid dims, so waves come free at scale. Re-entries: a
skip BITMAP for non-contiguous inert sets (stripes); balanced pairing
(program g also takes G−1−g) to fix one-wave imbalance; both enter
only on a measured gap.

## §5 The machine stays a table (ruling 2)

No SM counts. The quantities the theory consumes are per-level
capacity (tile shape), per-program resource limits (feasibility), and
the granularity floor (minimum work) — the first two are the existing
machine table, the third is a measured ledger fact. Processor count
enters only in the ratio programs-vs-P, and the hardware scheduler
absorbs it as long as we OVERSUBSCRIBE: derive the grid as
cdiv(extent, tile) and let placement be the machine's problem. This is
why hand Triton kernels are portable — our own triton_zoo baselines
launch cdiv grids and never query the device — and we inherit that
portability. Across device families: the machine is still a table; v1
adds no columns.

## §6 Refused, with re-entry conditions (ruling 3)

- **Split-K / cross-program reduction**: refused red. Re-entry: a zoo
  entry measurably launch-starved on its independent axes. When it
  enters, the device's saturation program count arrives as a MEASURED
  table row, never a compile-time constant. The refusal wording is
  deliberate: partitioning a *carried* axis is the same act at every
  level of the machine hierarchy — grid programs at the bottom, mesh
  collectives at the top (placement.py already prices the latter). The
  grid is the top of 320's derived hierarchy, and stage generalizes
  across it; nothing here commits the distributed design, but nothing
  contradicts it either.
- **Persistent kernels**: refused; re-entry = measured launch-overhead
  dominance on a real entry.
- **Rasterization/swizzle for L2**: refused; re-entry = an L2 row in
  the machine table plus a measured locality gap.

## §7 Sequence

1. **Launch artifact + gridded translation + the partition law.**
   `launch` on Group; translator prelude (pid decomposition, shifted
   starts, ragged-tail guards); battery: bit-equality gridded vs
   ungridded (divisible and ragged), structural pin on the prelude.
2. **Feasibility scorer.** pdum/tl/launch.py: the tile-tier machine
   table (320 §5's three columns) + per-program footprint; the T=512
   case becomes a computed refusal/re-tile in a test.
3. **Closed-form proposals + measured pick.** propose_launch per
   registry row (§4 table); candidates measured through the ledger;
   green by fact. plan_region attaches launches to Groups.
4. **The board, re-measured.** Fused+gridded flash vs hand gridded at
   T=128 and T=512 (the crash size) — the 2.5× should collapse toward
   the 1.4× emission gap; report with the remaining asymmetries named.
