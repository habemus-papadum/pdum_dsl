# Lean 4 modeling diary

Working notes, not a plan of record. The contract for this document: every
time the Python side grows a feature, this file gets a dated entry saying how
that feature would be modeled (or why it wouldn't be), at whatever level of
detail is honest. Lean snippets are *sketches* — none have been checked by
Lean yet. When we start the actual Lean package, this diary becomes its
design rationale.

Guiding taste: pragmatic before beautiful. Prefer shallow embeddings and
`omega`-sized proofs first; postpone dependently-typed IRs until something
actually needs to quantify over programs. Use Mathlib aggressively.

---

## 2026-07-22 — initial layout of the whole cake

### The layer map (Python artifact → Lean artifact)

| Python | Lean |
|---|---|
| dim names, D5 order-free | an index *type* `δ` (DecidableEq, Fintype) — order never exists |
| raw lattice domains [start, stop) | `Box δ := (lo hi : δ → ℤ)`; coords are subtype functions |
| abstract tensor (COMPUTE.md §8 layer 1) | `Tensor δ b α := Coord b → α` — Mathlib `Matrix` generalized |
| Layout (strides, offset) | `Layout δ` structure + `loc : Coord b → ℤ` |
| Buffer / read seam | `Buffer α := ℤ → α` (element-granular; bytes abstracted away) |
| FunctionalBuffer | a *definable* affine `Buffer ℚ` |
| guards + fill | totalized denotation: `if guards hold then buffer (loc c) else fill` |
| Chart / Quantity / units | `Qty (d : Dims) := ℚ` type family; `Dims := String →₀ ℤ` |
| categorical labels | `Fin n ↪ Name` (an injection; the nominal rung) |
| markers (pw/red) | typeclass instances / bundled structures with proofs |
| carriers | the `α` in `Tensor … α` + instance constraints per marker |
| programs (linear sequences) | shallow: composed Lean functions; deep (later): inductive IR |
| machine model | cost semantics over the deep IR (vision) |

Three pleasant surprises fell out of writing this table:

1. **Lean is the canonical quotient.** Python keeps a presentation order and
   needs `canonical()`; in Lean, dims are a type, so order never existed.
   The D5 decision was secretly "make the Python data structure a
   presentation of the Lean one."
2. **Alignment dissolves into typing.** `pointwise` on abstract tensors is
   just `fun c => f (A c) (B c)` — it *only typechecks* when both operands
   share `δ` and `b`. Python's `alignment()` diagnosis is exactly the
   construction of that shared type; the recipes (flip/shift/slice/repeat)
   are the transport maps. "Alignment is a proof obligation" is literal.
3. **Layout ops are linear maps.** Every view op induces an `ℝ`-linear map
   `Tensor b₁ ℝ →ₗ[ℝ] Tensor b₂ ℝ` (repeat, slice-as-restriction, window,
   decimate — all linear; pointwise(mul) bilinear; reduce(add) linear). The
   AD adjoint table in COMPUTE.md §7 is then Mathlib's `LinearMap.adjoint`
   on finite-dimensional inner-product spaces — repeat†=reduce-sum and
   slice†=zero-pad become *computations*, not new theory.

### Sketches for what exists today

Coordinates and abstract tensors (unchecked):

```lean
structure Box (δ : Type) where
  lo hi : δ → ℤ

def Coord (b : Box δ) := ∀ d, {i : ℤ // b.lo d ≤ i ∧ i < b.hi d}

def ATensor (b : Box δ) (α : Type) := Coord b → α
```

Concern noted immediately: `Fin n` would be more ergonomic than ℤ-subtypes,
but raw coordinates (D3) are load-bearing in the Python design, so we bite
the ℤ bullet and expect `omega` to eat the interval arithmetic. If it gets
painful, an iso `Coord b ≃ ∀ d, Fin (size d)` is a one-time lemma.

Layout, denotation, guards:

```lean
structure Layout (δ : Type) [Fintype δ] where
  stride : δ → ℤ
  offset : ℤ

def loc (L : Layout δ) (c : Coord b) : ℤ :=
  L.offset + ∑ d, L.stride d * (c d).val

structure Guard (δ) where
  coeff : δ → ℤ   -- finitely many nonzero
  lo hi : ℤ

def Guard.ok (g : Guard δ) (c : Coord b) : Prop :=
  g.lo ≤ (∑ d, g.coeff d * (c d).val) ∧ (∑ d, g.coeff d * (c d).val) < g.hi

def denote (L : Layout δ) (gs : List (Guard δ)) (buf : ℤ → α) (fill : α) :
    ATensor b α :=
  fun c => if ∀ g ∈ gs, g.ok c then buf (loc L c) else fill
```

Modeling decision: **buffers are element-granular functions ℤ → α**. Byte
strides, itemsize, dtype decoding — all representation, all skipped. The
semantics only needs locations to be an ℤ with affine structure. If we ever
want byte-level fidelity (aliasing across dtypes, field selection into
structured dtypes), that is a refinement layer `decode : Bytes → α` we can
add without disturbing anything above it. field() therefore denotes as an
offset shift composed with a different decode — fine, later.

Units without partiality — the type-family trick:

```lean
abbrev Dims := String →₀ ℤ            -- dimension exponent vectors
abbrev Qty (d : Dims) := ℚ            -- magnitude in base units

-- addition is only defined within Qty d: dimension errors are type errors
structure Chart (d : Dims) where
  origin : Qty d
  step   : Qty d
  step_ne : step ≠ 0
```

This makes the Python runtime check ("cannot add 1 um and 1 s") a *static*
impossibility in the model. The exactness discipline (D8) means everything
here is ℚ — no floats exist anywhere in the semantic layer, so no numerical
analysis exists anywhere in the proofs. This was the point of D8 all along.

The axis invariant becomes an actual theorem statement:

```lean
def physPos (axis : String) (charts : …) (c : Coord b) : Qty d :=
  ∑ dims tagged axis, (chart _).origin + (c _).val • (chart _).step

theorem select_preserves_physPos : … -- the "glued labels" theorem, for real
```

Compute primitives (shallow):

```lean
def pointwise (f : α → β → γ) (A : ATensor b α) (B : ATensor b β) :
    ATensor b γ := fun c => f (A c) (B c)

def reduce [CommMonoid α] (A : ATensor b α) (r : Finset δ) :
    ATensor (b.drop r) α := fun c => ∑ over the dropped sub-box, A (c ⊕ _)

-- markers-as-declarations ↦ typeclass instances: red.sum is AddCommMonoid,
-- red.max is a (linearly ordered) semilattice; associativity is not a
-- flag, it is an instance argument.
```

`scan` is `List.scanl` along one ℤ-interval (which, unlike an abstract
Fintype, is ordered — scan is the one primitive that *uses* the order of a
dim, worth remembering). The Blelloch equivalence (`scanl f = parallel scan`
given associativity) is a classic verified-algorithms exercise and slots in
whenever the machine model needs it.

`iota` and the closure invariant: iota is `fun c => (c d).val` — the
identity. The tightness theorem is an induction over op sequences:

```lean
theorem iota_stays_affine (ops : List ViewOp) :
    ∃ a : ℤ, ∃ w : δ' → ℤ, ⟦apply ops iotaT⟧ = fun c => a + ∑ d, w d * (c d).val
```

which is the Lean form of "FunctionalBuffer ∘ any layout chain is affine."
The Python invariant test (`W.buffer is I.buffer`) is the operational shadow
of this statement.

Carriers: the abstract tensor's `α`, constrained per marker — `exp` demands
`ℝ` (or `RCLike`), comparisons land in `Bool`/`Prop`, `div` picks its
meaning from the instance. The coercion chain ℤ →+* ℚ →+* ℝ →+* ℂ is stock
Mathlib. The "carrier rat represented as float64" tensor denotes an
ℚ-tensor, full stop — representation does not exist here.

### Theorem shopping list (deliberately ordered)

1. **Warm-ups that pin the core**: `loc` affine; footprint containment
   (`∀ c ∈ box, loc c ∈ [lo, hi)`); `simplify` soundness (a guard vacuous
   over the box can be dropped — interval arithmetic, `omega`).
2. **Op simulation lemmas** — for each view op, denotation commutes with the
   abstract definition: `⟦window T⟧ (x, k) = ⟦T⟧ (x + k)`,
   `⟦decimate T f p⟧ j = ⟦T⟧ (f*j + p)`, split/merge round-trip, flip
   involution. Each Python test is a quarry for the statement; most proofs
   should be `simp`+`ring`+`omega`.
3. **The guarded-footprint λ-lemma** — when guard coefficients are
   proportional to strides, the footprint honoring the guard is exact. This
   is the one subtle argument the adversarial review only verified
   empirically (300 random chains); Lean settles it forever. High
   value-per-line.
4. **Axis invariance** — select-compensation preserves `physPos`; shift and
   flip glue labels to data. The design's central "theorem, not convention"
   claim, made checkable.
5. **Adjoint pairs** — `⟨repeat x, y⟩ = ⟨x, reduceSum y⟩`, slice/pad, the
   window/overlap-add pair. Sets up AD.
6. **The milestone with a flag on it**: the matmul program
   (repeat·mul·reduce) denotes `Matrix.mul`. Connecting our normal form to
   Mathlib's matrix algebra is the moment the model has teeth.
7. Then and only then: the deep embedding.

### The vision layers (future, shape only)

**Shallow now, deep later.** Program *transformations* (fusion, reordering,
AD-as-transformation, cost) need programs as data — an inductive IR. Typed
deeply (IR indexed by boxes) it's honest but expensive; untyped with a
well-formedness predicate is the pragmatic middle. Decision deferred until
target 6 above is done; premature deep embedding is how Lean projects die.
Until then, transformations are stated as equational lemmas about composed
functions (`pointwise f ∘ pointwise g = pointwise (f ∘ g)` and friends) —
genuinely useful already.

**AD.** Two-stage plan matching COMPUTE.md §7: (a) leaf lemmas — per-marker
derivatives, mostly stock Mathlib (`Real.exp` has a derivative; bilinearity
of mul is free); (b) the reverse-mode transformation on the deep IR, proved
against `fderiv` composition. The linear-map realization above means the
layout-op half of the tape needs no calculus at all — just adjoints of
linear maps on `EuclideanSpace`. Fan-out/accumulation is `LinearMap.add`.
Max-reduce subgradient: state the a.e./tie-caveat version only; do not
litigate subdifferentials.

**Machine model.** Deep IR + a cost interpretation: memory levels with
capacities and bandwidths (ℕ-valued resources), execution hierarchy, tensor
cores as special-cased cost rules for matmul-shaped nodes, precision =
bytes-per-element entering *only* cost, never denotation (the carrier/dtype
split, again). Claims have the shape "denotation-preserving ∧ cost(S) ≤
cost(S')" — performance is model-relative and provable; accuracy is
empirical and deliberately unprovable here. This is the genuinely novel
research; everything before it is established technique.

### Bridging Python ↔ Lean without full verification

The real correctness contract today is Lean-model ↔ Python-tests, and that
bridge can be mechanical without being formal: have Python emit concrete
test vectors (random layout chains, their op sequences, item()-level
expectations) as Lean `example` statements over ℚ/ℤ, discharged by
`decide`/`native_decide` in CI. Cheap, brutal, and it catches divergence in
either direction the day it happens. Worth doing as soon as the Lean package
exists — long before any interesting theorem is proved.

### Concerns / issues (diary-honest)

- ℤ-interval coordinates everywhere: `omega` should cope, but sums over
  `Finset.Ico` are clunkier than `Fin n` sums. Mitigation: the one-time iso.
- Names: Lean wants index types, Python wants strings. The mapping is
  bookkeeping (a `δ := {x, y, z}`-style enum per example, or `String`
  subtypes); do not try to make Lean kwargs happen.
- Guarded denotation totalizes with `fill` — good for pointwise semantics,
  but it means "the real region" is a *proposition* about coordinates, and
  theorems must quantify over it. Fine, just noisy.
- Structured dtypes/field() sit below the element-granular buffer
  abstraction. Punt until something needs it (the complex-as-two-fields
  story will eventually).
- Categorical labels have almost no proof content yet (an injection and a
  refusal list). That is honest: nominal data has no algebra — the model
  should be equally silent.
- `Fraction` growth (CONCERNS #12) has no Lean analogue — ℚ is ℚ. The
  fixed-point normalization pass is a *representation* concern; it will
  appear in the machine model, not the semantics.
- Mathlib is a heavy dependency with real build times; accept it (the
  alternative — reinventing `Finsupp`, `omega`, linear algebra — is worse).
- Biggest modeling risk: the deep-IR typing decision. Wrong choice = months.
  Hence: postponed, with the shallow layer generating the requirements.

## 2026-07-22 (later) — the IR and reverse-mode AD landed in Python

(1) *Denotation*: `ir.Program` is the deep embedding this diary deferred —
linear SSA, no branching. Its denotation is a fold of per-instruction
denotations over an environment; `ir.run` is that fold over the reference
layer, `ir.infer` is the same fold over layouts only (a second, abstract
interpretation — Lean will recognize this as two algebras over one syntax,
begging for a generic fold). `materialize` denotes the identity (its whole
content is representation).

(2) *Theorems touched*: the AD transformation is now a concrete function on
programs, so "AD correctness" has a precise statement: for differentiable
programs P and scalar target t, ⟦grad(P, t)⟧ computes ∇⟦P⟧ — provable
against `fderiv` composition given the marker-derivative leaf lemmas. The
adjoint table is pinned per-op by finite-difference tests; each row is a
lemma statement (the linear-map/adjoint realization from the initial entry
applies verbatim). The seed contract (VJP) matches `LinearMap.adjoint`
applied to the output functional.

(3) *Vision moved*: the deep-IR typing question now has data — the Python
IR is *untyped with runtime/inference-time checking* (shadows), and the AD
transform only ever needed `infer`'s layouts, not a dependent index. That
argues for the untyped-IR + well-formedness-predicate route in Lean, with
`infer` as the shape-checking function whose success is the WF proof.
Checkpointing/scheduling (REPRESENTATIONS.md) will want programs-as-data
too; same embedding serves.

## 2026-07-24 — placement is a forgetful functor

L3-lite landed the way the formal story wanted it: `bind` adds structure
that ERASURE forgets, and the whole correctness contract is one statement —
the placed program's denotation IS its erasure's (tested bit-exact on the
Megatron block). No collective ops exist to axiomatize: an all-reduce is
`reduce` over a machine-bound dim, so its meaning was fixed back when
reduce was, and only the COST semantics (traffic — the fourth resource
semantics, a per-level byte Counter with an alpha-beta time collapse) knows
communication happened. The Lean shape: placement is extra data on dims, a
WF-predicate (bound ⇒ chartless, level exists, extent ≤ count), and every
placed theorem factors through erasure — nothing about values needs
re-proving. Alignment's new placement clause is the interesting typing
judgment: cross-placement combination is ill-formed until an explicit
(cost-bearing) fix, which is D17 promoted to a distributed-typing rule.

## 2026-07-23 (late night) — the curve exists

Segmented fold adjoints (`fold_segments=K`) produced the first MEASURED
memory/recompute tradeoff: FDTD T=12 peaks 2680→1816 bytes with the
minimum at K≈√T and ops rising monotonically — Chen's √T heuristic
reproduced from our own primitives. Lean-wise nothing new is needed: the
segmented adjoint is the same reverse-fold lemma applied per segment plus
one seam identity (segment j's final reverse carry = the cotangent of
segment j-1's end state), an associativity-of-composition fact about
folds. Binomial revolve will be a SCHEDULE over the same certified pieces
— strategy and correctness stay factored, which is the point.

## 2026-07-23 (night) — DCE and the min cut

(1) *Denotation.* Both transformations have one-line soundness statements
against the L0 denotation: dce — restricting a program to the backward
closure of kept vars preserves their denotation (induction on the pruned
suffix); checkpoint — duplicating pure instructions under fresh names and
re-pointing operands preserves every ORIGINAL var's denotation (purity =
substitution lemma). The interesting obligation is the flow argument: the
min cut guarantees every backward-read value is derivable from saved ∪
inputs through recompute-allowed ops — a reachability invariant, provable
against the network construction once, independent of Ford–Fulkerson.

(2) *Theorems touched.* The recompute-duplication lemma is the first place
name≠value becomes formal: the natural statement is an equivalence relation
on SSA vars ("denote the same value") that transformations may coarsen —
value numbering as a QUOTIENT, which is pleasingly the same
canonical-quotient shape as layouts-up-to-relabeling.

(3) *Vision moved.* The L1 arc is now measure → transform → re-measure with
falsifiable numbers (GPT-2: boundary 47%, peak 76%, 64% with DCE), which
is exactly the loop every later level should replicate: L3 will measure
traffic and transform placement; L4 will measure per-kernel footprint and
transform fusion. The pebbling connection is now concrete: peak_memory is
the pebble count of a schedule; checkpoint trades pebbles for recompute
moves; revolve is the optimal chain strategy — all three shapes of one
game the Lean machine model should state once.

## 2026-07-23 (evening) — the zoo, and memory as max-plus

(1) *Denotation.* The zoo is the denotation CORPUS: nine programs whose
specs are independent numpy functions. For Lean this is the extensional
test bed every certified rewrite gets exercised against before proof —
and the flash-vs-naive pair is the first pre-stated equivalence THEOREM:
online-softmax reduce = softmax-then-contract, an induction over the
associative rescaling combine (its identity relies on exp underflow at
-1e30 in floats but is EXACT in the real-number denotation with -inf).

(2) *Theorems touched.* peak_memory realizes the predicted second resource
semantics: where ops_count maps programs into (Counter, +), peak memory
maps schedules into (bytes, max) over a liveness walk — max-plus, not
plus, so it is a semantics of SCHEDULES rather than programs; the program
only bounds it below. The correctness statement pairs with the alias
theorem: a layout op's denotation factors through its operand's buffer
(views allocate nothing), which our layout algebra makes decidable.

(3) *Vision moved.* Staggered-grid FDTD forced the chart discipline
through the fold adjoint: the reverse pass must PRESERVE the primal's
charts (a chart-aware step misaligns on stripped inputs), which is the
gradients-carry-their-primal's-labeling invariant showing up as an
implementation constraint, not just a stance. The half-integer Yee charts
(Fraction(1,2) origins) with explicit with_charts recharting are exactly
the "discretization honesty made syntax" the physics thesis wanted.

## 2026-07-23 (later) — LEVELS.md and the fold combinator

(1) *Denotation.* `fold` is the first STRUCTURED combinator in the IR:
denotationally `List.foldl step init elems` with per-step emission — the
single most proof-friendly control structure there is (every property is
one induction over the step list). This is much better than unrolling:
theorems about time-stepped programs (FDTD, SSM recurrences) become
statements about one induction scheme instead of n-instruction syntactic
blobs. The carry contract (step preserves the state layout, checked at
run/infer) is precisely the invariant the induction carries.

(2) *Theorems touched.* The fold adjoint is DERIVED by self-application:
grad of a scalarized wrapper around the step gives the VJP program, and the
adjoint is the reverse fold of that program over the stored trajectory. So
its correctness theorem is MODULAR: "if grad is correct on the step
program, the reverse fold computes the fold's derivative" — an induction
over steps that never inspects the step's internals. This is the
compositional shape we hoped AD proofs would take: one lemma per
combinator, leaf obligations discharged by the existing per-marker story.

(3) *Vision moved.* LEVELS.md fixes the formal architecture for the whole
machine-modeling arc: levels are WF-predicates over ONE IR, lowerings are
denotation-preserving functions between them, cost models are monoid
homomorphisms out of per-level resource semantics, and equivalence
assurance is stratified (numeric check → decidable layout normalization →
Lean-certified rewrite rules, with Python verifying rule-chain
applications). Lean never faces "prove these two programs equal" — only
"prove this rewrite rule once." The machine tree is data, so no machine
detail ever enters the logic; machine-bound dims are chartless by
discipline, which keeps the chart/unit theorems untouched by lowering.

## 2026-07-23 — signatures, the SSM backward pass, and counting

(1) *Denotation.* The signature pass is a TYPING JUDGMENT laid over the
untyped IR — exactly the WF-predicate idiom chosen for programs, now for
values: `VInfo = (carrier?, unit?)` is an abstract domain, `infer_signatures`
is monotone forward abstract interpretation, and `None` is ⊤. Soundness is
one statement: if the pass assigns (c, u) to a var, the reference denotation
lands in carrier c with unit u. The composite-reducer case is a small
fixed-point over the state tuple — finite lattice (five carriers), so
termination is by height, a two-line Lean argument.

(2) *Theorems touched.* The composite-scan adjoint turns on two facts worth
proving abstractly, neither Python-specific: (a) the cotangent of a fold
s_t = C(s_{t-1}, l_t) is the reversed-time LINEAR recurrence
ŝ_t = (∂C/∂left)ᵀ ŝ_{t+1} + Pᵀ ȳ_t — BPTT-as-lemma, an induction over the
fold; (b) the boundary needs no special case because init is the monoid
IDENTITY: C(init, r) = r implies ∂C/∂right = I at t = 0, so the uniform
formula is exact. (b) upgrades `init` from a convenience to a stated
obligation alongside associativity — the reducer's typeclass instance now
carries (assoc, identity, and later: combine differentiable). The generated
matrix-linrec carrier is itself an instance of the same structure, which is
pleasingly self-referential: the adjoint of a verified scan is another
verified scan.

(3) *Vision moved.* opcount.py is the first COST SEMANTICS: a second
denotation of the same program into a commutative monoid (Counters under +),
with cost models as monoid homomorphisms applied afterward. That is exactly
the machine-model shape planned for the memory/execution hierarchy — peak
memory will be another such semantics (max-plus rather than plus), so the
Lean machine model should be designed once as "program → resource monoid"
with ops-count as its simplest instance. MAC fusion previews the pattern:
a cost-preserving-up-to-model rewrite, stated and proved per pattern.

## 2026-07-22 (later still) — the marker DSL landed

(1) *Denotation*: a composite marker body is a scalar expression tree
(Arg/Const/Prim) denoting a function `α^arity → α` by structural recursion —
in Lean, a tiny inductive with an `eval` fixpoint; the easiest denotation in
the whole project. Composite reducers denote fold/scan over a product state
type with a declared-associative combine.

(2) *Theorems touched*: two lemma families become stateable and small:
`eval (diff body i) = deriv (eval body) i` — symbolic-differentiation
soundness, an induction over the tree using Mathlib's `deriv` rules per
primitive (this DISCHARGES the per-marker leaf obligations from the AD plan
for every composite at once; only true primitives remain axioms/lemmas) —
and per-reducer associativity (e.g. the linrec pair combine), which is
exactly the CommMonoid-style instance obligation predicted earlier; proving
it unlocks the verified Blelloch equivalence for the machine model.

(3) *Vision moved*: the marker IR is the scalar sublanguage the compiler's
kernel bodies need, and it is frontend-agnostic by construction — the Lean
model can define its own well-formed-tree predicate without caring whether
Python traced the tree or pdum.dsl lowered it. The no-rewrite seam
(producers vs consumers of a dumb schema) is the same shape as the
untyped-IR + WF-predicate decision for programs; the two should share
idioms.

### Update protocol

When a Python feature lands, add a dated entry here answering three
questions: (1) what is its denotation (or: why has it none)? (2) which
existing theorems does it touch? (3) does it move anything from the vision
layers into reach? Keep entries short; this is a diary, not a spec.
