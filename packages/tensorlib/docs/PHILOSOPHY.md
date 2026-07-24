# PHILOSOPHY — why this library is the way it is

Written 2026-07-24, once the shape had emerged: L0 denotation through
L3-lite placement, with the ladder mapped to L6. PROVENANCE.md names the
sources; this file names the convictions.

## Why we are doing this

Tensor computation is run through stacks nobody can see through: framework
semantics defined by implementation, memory behavior discovered by OOM,
communication discovered by profiler, numerics discovered by divergence.
This project is a **white-box modeling lab**: a small, exact, fully
inspectable account of tensor computation from the mathematics down to the
machine — built to *understand and predict* rather than to run fast. The
reference layer is deliberately slow; its job is to be a denotational
semantics later layers must match, an honest ruler for every cost model,
and eventually the thing a proof assistant can hold. Speed is a property
we intend to *model* precisely before we ever attempt to *have* it.

## The load-bearing convictions

**Exactness before convenience.** Coordinates, strides, charts, units,
guards — all exact (integers and rationals; floats are confined to value
space, where approximation is honest). Exactness is what makes every
later claim checkable: layouts compose by arithmetic, not convention;
alignment is decidable, not heuristic; the Lean story is possible at all.

**Meaning first, cost as separate semantics.** There is ONE IR and one
denotation. Ops count, peak memory, communication traffic — each is an
extra semantics laid over the same program (a Counter monoid, a max-plus
walk, a per-link byte tally), never a change to what the program means.
The ladder (LEVELS.md) is a stack of well-formedness predicates and
erasures, not a stack of dialects.

**Erasure invariants everywhere.** Charts label the lattice without
touching addresses. Units annotate values without touching arithmetic.
Placement binds dims without touching denotation. In every case the
metadata can be forgotten and the program means the same thing — which is
precisely what makes each layer's correctness a one-line theorem and each
layer's cost model free to be opinionated.

**Diagnosis, never surgery (D17).** The library refuses to guess.
Misaligned operands, dimensioned exponentials, cross-placement
combination, un-modeled traffic — all produce a refusal that *quotes the
fix* and leaves applying it to the caller. Every automatic repair we
declined is a place where a silent wrong answer cannot exist.

**Names first, order never.** Dims are named; presentation order is a
property of export, not of meaning. This one choice quietly powers the
rest: heads and rotary pairs are *born* as weight dims instead of split
into existence; broadcasts are declarations; sharding is a name bound to
a machine level; `permute` does not exist because it never needed to.

**Transform where it is safe; commit late.** All rewriting — AD,
checkpointing, schedules, placement — happens in pure value semantics,
where substitution is sound and recomputation is just duplication.
Storage (buffers, in-place, offsets) is deliberately LAST, adopted only
when rewriting stops. We took this lesson from others (see PROVENANCE)
and it has repaid us at every level.

**Measure → transform → re-measure.** No optimization lands without a
simulator that can falsify it. The peak-memory simulator preceded the
checkpointers; the traffic pass preceded any distributed optimization;
the curve (√T, then log T) was *measured*, not asserted. A cost model we
cannot check against the reference is an opinion, not a model.

**Strategy and correctness factor.** The fold adjoint is one certified
construction; store-all, uniform segmenting, and binomial revolve are
*schedules* over it. A wrong schedule wastes memory or compute — it
cannot produce a wrong gradient. We look for this factoring everywhere:
it is what keeps search spaces safe to explore.

**Derive, don't enumerate.** Composite markers get partials by tree
rewriting; composite reducers get backwards by BPTT-as-IR; folds get
adjoints by differentiating their own step program; flash attention's
backward exists because the online-softmax combine was *declared*, not
because anyone wrote it. Hand-maintained rule tables are where semantic
rot begins; we let them stop growing.

**Honesty as a deliverable.** CONCERNS.md is a first-class artifact: every
known coarseness, tie-caveat, unfused collective, and process-level
registry is written down next to the code that has it. The zoo pins every
model to an independent numpy denotation. When a win only appears under
`out=final`, the notebook says so. The documentation is a lab ledger, not
marketing.

## What is genuinely ours vs. gratefully borrowed

Most individual ideas here have clear ancestry (PROVENANCE.md credits
them). What we believe is distinctive is the **combination and its
discipline**: one exact layout object carrying strides, guards, charts,
units, and placement together; collectives with *no ops at all*, read off
the same algebra by a cost pass; adjoints derived by self-application all
the way up to program-valued scan steps; masks, positions, and broadcasts
that are *structurally* free across every cost model because they are
closed forms in the representation; and a verification story where every
layer's theorem is an erasure or a factored schedule rather than a
monolithic compiler proof. The aesthetic we are chasing: a small
vocabulary, composed relentlessly, with nothing load-bearing hidden.
