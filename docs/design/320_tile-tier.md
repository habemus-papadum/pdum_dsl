# 320 — The tile tier: a sister kind, one dialect

**Status: DRAFT — positions converged with the owner (2026-07-29);
implementation not started.** This records the design conversation's
outcome so the tiling era opens against written law. Supersedes nothing;
K-A…K-G (LEVELS.md) remain the parent conversation, 260/270 the arc.

## The central claim

"Tile" is a new kernel KIND in the existing tier stratification —
exactly the move that made vertex/fragment/compute distinct disciplines
over one Region dialect (290). There is no second IR and no second
syntax tree: a tile kernel is a region whose admitted vocabulary
differs from compute's, gated by `check_tier` like every kind. The two
kernel languages never call each other; the K-A fusion pass partitions
an assemblage region into groups and stamps each group's kind (compute
for per-element work, tile where reuse pays), and the assemblage level
stays blind to the choice. Interop lives at the host level, where it
already lives.

One decision, three consequences a separate language could not buy:

- **A day-one oracle.** A tile kernel ERASED of placement and staging
  is an ordinary tensor-tier region — `run_region` executes it today
  (megatron's `level=None` erasure is the precedent that stripping
  placement preserves denotation bit-exactly).
- **Equivalence is region-vs-region**, never cross-language (§6).
- **The reference-runtime pattern extends unchanged** (310): Triton is
  the tile tier's pair the way torch/jax are the tensor tier's (§8).

## §1 The binding law

The launch declares grid axes; the body sees each param PRE-SELECTED at
the ambient tile coordinate (the compute tier's ambient law, one level
up). The author never computes a block offset; there are no pointers
anywhere — the layout algebra IS the addressing. Open clause, with a
lean: some kernels (flash's causal early-exit) want to know WHICH tile
they are — extend `tl.coord`'s credential to the tile lattice rather
than invent a new door.

## §2 The body, by flagship

```python
@tile(grid=("mo", "no"))                  # grid dims consumed ambiently
def gemm(a, b, c):                        # a:(mi,k)  b:(k,ni)  c:(mi,ni) writable
    acc = tl.const(0.0, mi=MI, ni=NI)     # the carry — registers, spelled by ABSENCE
    for kt in tiles("k", KI):             # bounded fold over k-tiles, acc carried
        As = stage(a.slice(k=kt), level="shared")   # the one copying op (§4)
        Bs = stage(b.slice(k=kt), level="shared")
        acc = acc + contract(As, Bs, axis="k")
    c.store(acc)                          # the token-ordered effect
```

"An assembly program with small tensors": explicit movement, explicit
loop, tensor ops between. Every line except `stage` is existing
dialect — `slice` is layout, `contract` is pointwise+reduce, the loop
is `tl.fold` with a carried state, `store` is the effect.

## §3 Vocabulary

Admitted: STRUCTURAL, SCALAR, POINTWISE, reduce/scan, the affine layout
ops (slice/select/shift/pad/window/stencil/split/merge), `tl.iota`
(masks and positions are free closed forms), the tile-fold, `stage`,
EFFECT. Refused at v1, each with a ledger row (290's coverage law):
take/scatter_add/argtopk/argsort/random — data-dependent addressing is
K-G's conversation. The store-free statement-if law carries unchanged.

Two consciously widened laws:

- **The tile-fold step vocabulary widens** beyond the current
  `{pointwise, const, param, yield}` to admit slice/stage/contract —
  the k-loop is the flagship's spine.
- **300's parked real-break license lands here**: tile-tier loops are
  real loops at lowering, not flat masked forms (the license was always
  "values identical, cost lower"; the host renderer remains its proof).

Masks are `pad`, not predicates: Triton's `mask=`/`other=` idiom is our
slice-beyond-domain + `pad(fill)` — guarded layouts already carry the
semantics, so non-divisible boundary tiles cost no new syntax.

**No barrier op exists.** A stage's result being consumed IS the
synchronization structure; `__syncthreads` is inferred from region
dataflow at lowering (Triton proves barrier inference is viable; the
token/dataflow discipline makes it natural here).

## §4 stage — the one copying op, generalized

`stage(x, order=?, level=?)` unifies placement movement and physical
reordering, because a stage IS a relayout with a destination level and
the cost of any stage is determined by the (source layout, destination
layout) pair:

- omit `level` → today's `tl.materialize`: the same-level relayout
  ("which dims vary fast against the buffer index" — the generalized
  transpose). Materialize becomes the degenerate stage; there is ONE
  copying op in the language.
- omit `order` → keep the source order, move residence only.
- **Charts ride.** A stage preserves charts (same physics, different
  residence); chart-stripping remains materialize-the-degenerate-case's
  separate, explicit contract.
- **The destination layout may pad/skew** — the classic +1 skew that
  kills bank conflicts is a destination-stride choice the layout
  algebra already spells, and the traffic model (§5) shows why you did
  it.

Levels are NAMES, referenced explicitly (`"shared"`), resolved against
the machine table (§5). Relative levels were considered and REJECTED
(recorded): placement-by-absence (unstaged = registers/SSA; unstaged
reads = global through the view) gives relativity's elegance where it
was genuine, and names keep level-skipping — GEMM's whole art —
explicit and honest. Re-entry condition: program portability across
machine shapes becoming a real, measured need.

## §5 The machine is a table; the hierarchy is derived

**Nothing models the machine in the language.** What remains of "the
machine" is a three-column table consumed ONLY by cost analysis:

    { level name → granule size, capacity, bandwidth }

The hierarchy is assumed SEQUENTIAL for now (the real tree of resources
is out of scope), and the dimension hierarchy is DERIVED, never
declared: sorting a dense storage's dims by stride recovers the nesting
(stride-0 riders fall out); the region itself records split provenance
(the ops are the history — `(mo, mi)` came from `m` because the split
is in the IR). Split-as-first-class-object was considered and REJECTED:
strides + region history already carry both the order and the
provenance; modeling it twice was the mistake. Backward compatibility
is law: a user ignorant of the hierarchy writes valid, possibly slow
programs — the hierarchy is an analysis view, never a semantic
obligation.

**The traffic model — L1's sibling.** `memory.py` counts CAPACITY
(live bytes under a schedule); this model counts TRAFFIC: for a stage,
the number of distinct granules touched on each side —
`|{ ⌊byte/G⌋ }|` over the footprint — a function of exactly (source
strides, destination strides, granule). Layouts are affine, so this is
COMPUTED, not profiled (the white-box property that makes L1 exact).
V1 enumerates the footprint — tiles are small, exact by construction;
closed forms for contiguous-inner-run cases later. Two corollaries for
free:

- **Coalescing** is this same count at the global level's granule — a
  coalesced access covers its lines densely — so coalescing is reasoned
  about STRUCTURALLY, without modeling threads (which keeps the thread
  deferral in §9 safe). The `core.vec` coalescing-era row (210, 4th
  lexical rule) is this model's consumer at spelling time.
- **Bank conflicts** are the identical calculation with a modular
  granule — same machinery, second row of the table, when wanted.

K-C's objective — parent-memory traffic under child capacity — is now
two computable numbers over shadows: traffic from this model, capacity
from L1.

## §6 Equivalence: erasure first, licenses for the rest

- **Erasure is the certificate for everything except reassociation.**
  Strip stages/levels (identity by the erasure law), inline the
  tile-fold back to a reduce, cancel split/merge via region provenance:
  if normalization reaches the naive region's content key, equivalence
  is PROVED, not sampled.
- **What does not erase is exactly the two license kinds already
  declared** (licenses.py, the closed taxonomy): the k-sum
  re-bracketing is a `reassociation` license — the reducer's declared
  associativity (mdsl) is precisely the algebraic fact that licenses
  it — and mixed precision is a `precision-demotion` license. The
  worked `GEMM_F16_TILES` declaration is this design's first consumer.
- **The operational gate**: differential against the erased oracle
  under ADVERSARIAL input families (260's law — never random draws
  alone: −inf masks, cancellation, non-divisible tails), tolerances
  quoted FROM the license, license set joining the artifact key.

## §7 Precision is a clause, not a layer

The tile syntax needs exactly one precision affordance: a declared
accumulator encoding on contract/reduce — a parameter of the op, never
the type (`round_to`'s law generalizes). V1 runs f32/f32 (310's dtype
axis is the receiving rig); f16-tiles/f32-acc arrives by consuming the
existing worked license, not by building a precision layer first.

## §8 Triton — the tile tier's reference pair

Per 310's pattern, one level down: hand-written Triton twins of the
flagships are the BASELINE (and the apprenticeship — where Triton's
real performance juice lives becomes measured knowledge, not folklore);
a Region→Triton translator is the CHECK. The mapping is close —
`stage`→`tl.load`, `contract`→`tl.dot`, `pad`→mask+other,
tile-fold→`for range` — because both languages are block-scoped tensor
programs. We adopt Triton's BLOCK SEMANTICS and refuse its POINTER
STYLE: our params arrive as tiles through the binding law, never as
base pointers. The pair mounts behind pdum.rt's door like every column
(283), and the rig's f32 board (310) is where the flagships' bars
already stand.

## §9 Deferred, each with its re-entry condition

- **Threads / warp cooperatives**: below the abstraction line (Triton
  is the existence proof that tile-level languages reach tensor-core
  performance without warp syntax). Re-enter iff a flagship measurably
  cannot reach its bar through tile-level lowering; arrives as a third
  kind at descent (the cuda.coop probe), never as fusion-pass target
  syntax.
- **Relative levels** (§4) and **split-as-object** (§5): rejected with
  rationale recorded above.
- **Data-dependent tiles** (gather/scatter in tile bodies): K-G.
- **Double buffering needs no deferral** — it is a split of the
  SEQUENTIAL stratum (`ko → (ko2, 2)`) with the stage pipelined over
  the 2-stratum: software pipelining is tiling of the loop dim, no new
  vocabulary. Recorded here so nobody invents an op for it.

## §10 Sequence

1. This document.
2. The tier gate + the erasure oracle (tile kernels run on the
   reference interpreter from day one).
3. Hand-write the K-D flagships — tiled GEMM, flash, the fused stencil
   chain — against their zoo twins.
4. The equivalence harness (§6): normalize-and-compare + adversarial
   differentials quoting licenses.
5. The Triton pair (§8), joining the rig and the f32 board.
6. Only then K-A's fusion/decision pass, once the language it emits is
   real.
