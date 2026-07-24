# The compute layer — vision restatement and initial assessment

Restating the proposed computational model over the tensorlib layout algebra,
with assessment interleaved. Nothing here is implemented; this is the
design conversation before code.

## 1. The thesis

**Three computational primitives, over the existing layout algebra, suffice
for most of modern deep learning and much of PDE physics:**

1. `pointwise(f, A, B, ...)` — the map. (Renamed from `element_wise`;
   rationale below.) Precondition: all operands **100% aligned** — same dim
   names, identical `[start, stop)` domains, equal charts/labels. Output:
   `C[i, j, ...] = f(A[i, j, ...], B[i, j, ...], ...)`, carrying the shared
   layout labeling.
2. `reduce(f, zero, A, dims)` — fold the named dims with `f : (acc, elem) ->
   acc` starting from `zero`; reduced dims drop from the output.
3. `scan(f, A, dim)` — the inclusive prefix reduce: every intermediate
   accumulator is kept, so the dim survives (cumsum = scan(add); the SSM /
   linear-recurrence shape once pair-state markers exist). A reverse scan is
   `flip ∘ scan ∘ flip` — layout ops, free. Adopted as a first-class
   primitive because associativity (declared by the marker) gives the
   O(log n)-depth parallel evaluation, and because its adjoint is again a
   scan (§7).

`f` is a **marker**, not a function: an opaque primitive tag (`add`, `mul`,
`max`, `exp`, ...) with, later, a tiny DSL for composite markers.

These three, plus the `iota` generator (§2c) and the layout ops (slice / select / shift / repeat / flip /
split / merge / diagonal / window / stencil / pad / decimate / field /
recenter / with_charts / with_labels), are the **entire user surface**.
There is no user-facing relayout, no materialize, no explicit memory
placement. Programs are **linear sequences** of these operations, organized
into functions that are themselves linear sequences, composed with the
pdum.dsl pipeline machinery. Lower layers (not user-facing) may exist for
tiling, shared-memory placement, register mapping.

**Assessment: coherent, and strongly precedented.** This is the map/reduce
decomposition of array programming — the normal form underlying einsum,
Halide's pure-function + reduction-domain model, tensor comprehensions, and
Dex. The known result from that lineage: structural views + map + associative
reduce cover dense multilinear algebra plus pointwise nonlinearity, which is
the overwhelming bulk of DL compute. What is *unusual* here — and it is the
load-bearing novelty — is how strong the view layer is: window/stencil with
exact guards, dilation, decimation, blocking, broadcasting, physical charts.
Operations that other systems hard-code as special compute ops (conv,
pooling, masking, striding, padding) become *layout* here, shrinking the
compute vocabulary to two verbs. The pieces also interlock with everything
already built: `alignment()` (D17) is literally the precondition checker for
`pointwise`; stride-0 `repeat` is the broadcast; guards are the masks.

### Naming

`element_wise` → **`pointwise`**. It says what the alignment precondition
means (one value per shared point), it is standard in the PDE world, and it
avoids `map`'s collision with Python. (`zip_with` is the honest FP name for
the n-ary form; keep it in mind if `pointwise` grates.)

## 2. Markers are algebraic declarations, not callbacks

The most consequential design point in the proposal, worth making explicit:
a marker should carry **declared algebra**, not code:

- identity element (the `zero` of reduce is really the marker's monoid unit);
- associativity / commutativity — the license for parallel and tiled
  reduction, *declared* rather than inferred from a lambda (inference from
  code is undecidable; declaration is free);
- unit signature (mul: units multiply; add: units must match — `value_units`
  finally gets its consumer);
- derivative markers (`d(mul) = ...`) — the autodiff layer's leaf table (§7);
- cost/precision annotations later, for the machine model (§9).

The initial marker set is small: `add`, `mul`, `max`, `min`, `sub`, `div`,
`exp`, `log`, `where` (ternary select), comparisons. Note `where` and
comparison markers are just n-ary pointwise `f`s — no new machinery.

### Normalization

Mean/variance normalization needs `1/N` — and **N is static**: it is the
numel of the reduced dims, known exactly from the layout. So normalization is
a pointwise scale by an exact rational constant; a `mean` (and `var`) marker
is pure sugar. No new primitive needed. Related and important: **constants
are already layout-native** — a one-element buffer plus stride-0 `repeat`s
IS a constant field of any shape. Scalars, ±∞ masks fills, eps — all covered.

### The workhorse composition

`reduce` output has fewer dims; most normalizations need it *back*:

    m  = reduce(max, -inf, S, dims=("v",))
    S' = pointwise(sub, S, repeat(m, "v", ...))     # broadcast-back
    e  = pointwise(exp, S')
    Z  = reduce(add, 0, e, dims=("v",))
    P  = pointwise(div, e, repeat(Z, "v", ...))

reduce → repeat → pointwise is softmax, layernorm, mean-centering, attention
normalization. `repeat` is the adjoint of sum-reduce (§7) — the pattern is
not an accident.

## 2b. Carriers — semantics never mentions precision

A tensor's value type is three orthogonal fields:

- **carrier** — the algebraic object the values *approximate*:
  `bool | int | rat | real | complex`, with the coercion chain
  bool → int → rat → real → complex (ℕ deferred; finite fields someday);
- **unit** — `value_units`, already present;
- **dtype** — the machine *representation*, demoted to exactly what the
  machine model (§9) says precision is: a footprint/cost resource, never
  semantics.

Carriers are inferred from the dtype by default (float→real, int→int,
bool→bool) and declared explicitly where inference cannot know better:
`"rat"` is never inferred — it arises from iota and chart-derived data,
where the values are *exactly* rational even when represented in float64.
Marker signatures over carriers (`exp : real → real` — exp leaves ℚ, so
`exp(iota)` demands an explicit embedding; comparisons produce `bool`;
`div` differs per carrier) are the future type-checking surface; today the
carrier is threaded metadata, like value_units before it. The Lean layer
denotes by carrier — a float32 tensor denotes an ℝ-tensor; proofs never see
bits. Complex is the showcase: one carrier ℂ, several representations
(planar re/im fields vs interleaved — and a struct of same-typed fields is a
categorical dim in disguise), freely chosen by a compiler without touching
semantics.

## 2c. Tight iota — the closure invariant

`iota` is NOT materialized. A tensor is the composition
`value(coords) = buffer.read(loc(coords))`; iota is the case where `read` is
itself affine, so its buffer is a `FunctionalBuffer` — no memory, declaring
`read(loc) = const + coeff·(loc // scale)` with **exact rational
coefficients**, cast to the machine dtype only at the read (the carrier/
representation split §2b, executed in code).

**The closure invariant: iota-ness is a buffer property, and layout ops
cannot destroy it.** Every view op rewrites only the layout; affine ∘ affine
= affine; so shift, slice, split, decimate, flip, window, pad, guards — all
preserve the closed form *by construction* (`window` on iota reads `x + k`,
the tap-position field; `decimate` reads `factor·j + phase`; a guarded iota
is affine + linear guard + fill, still white-box). A compiler recognizes
the closed form by the buffer's type and never touches memory for it. The
lattice face is the ℤ identity (carrier int); the physical face is the
chart's exact ℚ-affine applied on top (carrier rat, unit recorded).

## 3. The reference semantics

A deliberately inefficient correctness layer: check `aligned(...)`, convert
operands `to_numpy()`, apply the numpy function the marker names, wrap the
result with the surviving dims' charts. If markers are numpy ufuncs, this
layer is ~50 lines. It is also the **denotational semantics** the Lean layer
(§9) and the compiler are each obligated to match — one artifact, three uses.

## 4. Canon walkthroughs

### Matmul (the normal form)

A: (m, k), B: (k, n):

    A3 = A.repeat("n", n_ext, chart=B.j_chart)   # stride-0: no data
    B3 = B.repeat("m", m_ext, chart=A.i_chart)
    P  = pointwise(mul, A3, B3)                  # dims {m, k, n} — aligned
    C  = reduce(add, 0, P, dims=("k",))

Note D5 paying off: A3's dims arrive as (m, k, n) and B3's as (k, n, m) —
names-first alignment makes the order question unaskable. Every einsum
(batched matmul, QKᵀ, attention-value contraction) is this same three-step.

### ResNet — fully expressible, no gaps

- Conv2d: `window` on H and W (+ `dilation=` for dilated variants), repeat
  input over out-channels and weights over batch/space, pointwise mul,
  reduce (C, kh, kw). Same-padding via `pad` fill 0 — the guards.
- Strided conv: window then `decimate`. Pooling: window + max/mean reduce.
- BatchNorm: reduce over (N, H, W) + broadcast-back + pointwise; ReLU:
  pointwise; residual add: pointwise (alignment enforced!); FC: matmul;
  global average pool: mean reduce. **Verdict: ResNet ✓ end to end.**

### GPT-2 — expressible modulo one thing, and the mask question has a
### better answer than expected

- LayerNorm, GELU, projections, attention contractions: all §2 patterns ✓.
- Positional embeddings: a *slice* of the table (positions are contiguous —
  no gather needed) ✓.
- **Causal attention: the mask is a GUARD.** `M[i,j] = (j ≤ i)` is the
  linear-form condition `i − j ≥ 0` — and linear-form guards with a fill
  value are *exactly* our guarded-layout family. A guarded view of the score
  tensor with guard `i − j ∈ [0, ∞)` and fill `−inf` masks causally with
  zero data movement, and softmax reads through it. Banded / sliding-window
  attention (|i−j| ≤ w) is two guards; dilated-strided patterns are guards
  after decimate. Today guards are only constructed by pad/stencil; the
  proposal: **expose a general guard constructor** (`t.guard(form, bounds,
  fill)`) — no new family, just surfacing existing machinery. A compiler
  sees the mask *structurally* (it can skip fully-masked tiles — the thing
  FlashAttention special-cases, given here by representation). Block-sparse
  masks are the piecewise family — known deferral. Alternatively any mask is
  just a constant 0/1 input tensor and a `where`/mul — the user's
  observation that masks are multiplied is correct; guards are the
  aristocratic version.
- **Token embedding is the real gap**: a gather. The classical bridge is
  one-hot × table matmul — expressible, but *constructing* one-hot from
  token ids needs to compare ids against coordinates. Which surfaces the
  actual missing primitive (§5). Cross-entropy's target-logit pick: same
  one-hot story. Sampling/argmax at inference: outside (needs RNG /
  index-producing reduce). **Verdict: GPT-2 training forward+loss ✓ given
  iota (or one-hot inputs prepared outside the language).**

### Physics

- Explicit stencil methods (heat, wave, advection FTCS, FDTD on staggered
  Yee grids — where half-step chart origins shine): stencil views +
  pointwise, often no reduce at all ✓.
- Boundary conditions: Dirichlet/constant ✓ (fill IS the BC); reflecting and
  periodic ✗ — the piecewise family, known and accepted limitation.
- Multigrid: restriction = window+decimate conv ✓; prolongation =
  zero-stuffing conv — expressible via the split/mask/merge pattern in §7.
- FFT: expressible as a log(n)-step program of split + twiddle-constant
  pointwise + merge (the classic reshape-butterfly decomposition) —
  a beautiful stress test for the surface language.
- Implicit methods: linear solves as iterative programs (CG = matvec + dots
  + AXPYs, all in-language) — but the *convergence test* is data-dependent
  control flow, which the linear-sequence program model does not have.
  Fixed-iteration loops are fine (program-level unrolling/pipelining);
  tolerance-driven loops need a scalar-conditioned loop construct someday.
- Semi-Lagrangian advection, particle methods: gather ✗ (named exclusion).

## 5. What is genuinely missing (short list, in priority order)

1. **`iota` — coordinate materialization.** A generator whose *values* are
   its own coordinates (lattice ints, or the chart's physical labels — the
   chart made into data). Layout-native, gradient-free, and it unlocks:
   one-hot construction (→ embeddings, cross-entropy), masks built
   in-language, positional encodings, distance/Green's-function kernels, and
   the decimate-adjoint in §7. This is the third primitive, and it is tiny.
   (It is also philosophically pleasing: the bridge from label-space to
   value-space.)
2. **General guard construction** — surface `guard(form, bounds, fill)`
   (§4). Existing machinery, new constructor.
3. **`scan`** — ADOPTED as the third primitive (§1). The reference layer
   covers ufunc-style markers; SSM-shaped scans over composite pair-state
   operators await the marker DSL.
4. **Gather/scatter** — deliberately excluded, correctly in my view; the
   one-hot bridge covers training-time uses at the correctness layer (and a
   compiler can strength-reduce onehot-matmul back into gather later —
   moving the difficulty from semantics to optimization, where it belongs).
   KV-cache updates and MoE routing are scatter-shaped: autoregressive
   *inference* stays out of scope until this is revisited.
5. **Data-dependent control flow** (convergence loops, sampling): a
   program-model question, not a primitive question. Defer.

## 6. The compiler question (previewed only)

One paragraph of position: the linear-sequence-of-views + two-primitives form
is close to a polyhedral normal form — iteration domains are boxes, access
maps are affine, masks are linear guards, reuse structure is explicit in
window/repeat, and reduction algebra is *declared* by markers. That is
precisely the information a scheduler wants, delivered exactly (ℚ/ℤ, no
approximation). The central compilation obligation is already visible: the
reference semantics materializes repeats and windows; a real backend must
treat stride-0 and overlapping views as *virtual* — i.e., fusion is not an
optimization here, it is the difference between O(mkn) memory and O(mn).
That the lineage (Halide/TVM/Triton/tensor comprehensions) compiles this
form well is the evidence the thesis is plausible; the open research is real
but not speculative. Deferred until the details arrive.

## 7. Autodiff — will reverse mode stay inside the language?

**Yes — with two footnotes, and one pleasing theorem-shape.** The
loss-is-a-full-reduction setup makes reverse mode a mechanical transform of
the (control-flow-free, SSA-like) program — the tape is the program itself.
The VJP of every surface op is again surface ops:

| primal | adjoint (cotangent side) |
|---|---|
| `pointwise(f, ...)` | `pointwise(∂f_i, operands..., dC)` per operand — from the marker's derivative table |
| `reduce(add, dims)` | `repeat(dC, dims)` — pure layout |
| `scan(add, dim)` | reverse scan of the cotangent: `flip ∘ scan(add) ∘ flip` — scans are self-class under AD |
| `repeat` (broadcast) | `reduce(add)` over the repeated dim — the adjoint pair |
| `slice` | `pad` with fill 0 — guards earn their keep |
| `pad` | `slice` |
| `shift` / `flip` / `rename` / `merge` / `split` | themselves (relabelings are self-adjoint up to inverse) |
| `select` | `pad`-like zero-extension (slice adjoint, thin case) |
| `reduce(max)` | equality-mask trick: `pointwise(eq, A, repeat(maxA)) * repeat(dC)` — standard tie caveat |
| `window`/`stencil` (conv) | overlap-add: per-tap `select` + `shift` + `pad` + accumulate — expressible today as a kernel-size-unrolled sequence; a general unimodular **shear view** would make it a single view + reduce (the one candidate addition AD suggests for the layout family) |
| `decimate` | zero-stuffing: `repeat` phases + iota-built phase mask + `merge` — expressible, **needs iota** |

Footnotes: (1) fan-out — a tensor consumed twice gets its cotangents
*added*; program-level bookkeeping, pointwise add. (2) the marker table is
where truth bottoms out: the transformation is generic and provable once,
per-marker derivatives are leaf declarations. No language expansion is
required beyond iota (already wanted) and optionally the shear view for
non-unrolled conv adjoints. This is unusually clean because the layout ops
were *chosen* to be an algebra closed under exactly these dualities.

## 8. The Lean4 program

Layered as proposed, and the layering matches what exists:

1. **Abstract tensor**: a function from (named, finite) coordinates to a
   field element — Mathlib's `Matrix` generalized; index type = dependent
   function over a name-keyed family. Operations get mathematical
   definitions (`∀ i k, window(X)[i, k] = X[i + k]`, etc.). No layouts.
2. **Layout soundness**: our guarded layout denotes an abstract tensor
   (`resolve`-based); prove each view op *commutes with denotation* — a
   simulation lemma per op. The arithmetic is integer-linear (`omega`-
   friendly) and exact-rational — **the D8 exactness discipline was,
   unknowingly, a Lean-compatibility decision**: no floats anywhere in the
   semantics means no numerical analysis anywhere in the proofs. The
   existing 130-test suite is a quarry of lemma statements (footprint
   containment, guard-rewrite correctness, label-sum invariance, adjoint
   dualities).
3. **AD correctness**: prove the reverse-mode program transformation
   correct *given* correct marker derivatives (leaf lemmas over the field —
   formal derivative rules, no analysis needed for polynomial/semiring
   markers; postulate the transcendental ones).
4. **Program transformations**: programs as lists, denotational semantics
   into layer 1, prove reorder/fusion rewrites denotation-preserving.
5. **Machine model**: cost semantics over memory hierarchy (bounded
   capacities, throughputs), execution hierarchy (thread/warp/block),
   tensor cores; precision as *footprint only* — bit-width as a resource,
   no IEEE semantics, no approximation-bound theorems. This is the right
   scoping: performance claims are model-relative and provable; accuracy
   claims are empirical and explicitly out of scope.

Feasibility: layers 1–2 are bread-and-butter Lean; 3–4 are established
territory (verified compiler transformations); 5 is the research-flavored
part — cost semantics exist in the literature (work/span, roofline-style)
but a memory-hierarchy + tensor-core model with proofs about schedules is
genuinely novel and genuinely valuable. Nothing in it looks infeasible;
everything in it looks like work.

## 9. Verdict

**Coherent, and worth pursuing without major revision.** The unusual
property of this design is how hard the pieces lock together:

- exact ℚ/ℤ semantics ↔ Lean formalizability;
- white-box guards ↔ masks-as-views ↔ compiler tile-skipping;
- marker-declared algebra ↔ parallel reduction ↔ AD leaf table;
- `alignment()` diagnosis ↔ `pointwise` precondition;
- repeat/reduce, slice/pad, window/overlap-add adjoint pairs ↔ AD closure.

Recommended additions (small): `pointwise` as the name; markers as algebraic
declarations; **iota**; the general guard constructor; `scan` and
gather/scatter as *named* deferrals with their consequences written down
(SSMs; embeddings-by-one-hot; no autoregressive inference yet).

Open questions carried forward: the compiler (efficiency capture — the
declared central research question); the program model's eventual need for
data-dependent loops; shear views for uniform conv adjoints; block-sparse
masks (piecewise family); the machine model's fidelity/tractability tradeoff.
