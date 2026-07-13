# 110 — Transforms & derivatives: vmap, jvp, and the in-kernel `D`

**Status:** canon (2026-07-12, step 12 opening). Companions: 100 (arrays &
axes — the named-axis machinery this builds on), 030 (deep-learning notes),
020 step-12 row. User-directed additions this step: the analytic
shader-derivative operator `D` (dual numbers, "only when necessary"),
named-first vmap (`axis="name"`), the batching-ignorance demo arc, and the
named-contraction (matmul) stretch goal.

## 1. The spike: vmap over `if`/`for` — and why the hard case dissolves

The architecture priced transforms at ~180 lines per region op per column
and demanded a spike before commitment (>350 lines ⇒ re-hear). The spike's
finding is structural, not incremental:

**Our vmap is SIMT-shaped, not SIMD-shaped.** JAX's vmap *widens values*:
every intermediate grows a batch dimension, and control flow is where that
hurts — a batched predicate means lanes disagree, `core.if` must become
execute-both-and-select (breaking the lazy-branch guarantee: guard-then-
divide would evaluate the guarded division in untaken lanes — JAX's
documented `where` wart), and batched trip counts need max-trip masking.

But this system already made the other choice at step 9: **the compute
family's params ARE coordinates**. The batch axis does not belong inside
values; it belongs in the *execution domain*. So `vmap(f, axis="batch")`
adds one coordinate parameter and *weaves* it into every access to a
capture that carries the named axis. Consequences:

- Intermediates STAY scalar — each lane runs its own branch, its own trip
  count. `if`/`for` need **zero** new machinery. The per-region-op tax is
  ~0 lines, far under the re-hear threshold; the 180-line price was for
  value-widening, which we do not do.
- The lazy-branch guarantee survives vmap untouched (each lane takes one
  path — no select-blending, no `where` wart).
- What is LOST, recorded honestly: cross-lane operations (reductions over
  the mapped axis, JAX's `psum`) have no home in a woven representation —
  they arrive, if ever, with GPU collectives (the mapped coordinate is
  already launch-domain-shaped, which is where collectives live anyway).
- Execution v1: the woven kernel is an ordinary scalar kernel with one
  extra trailing arg (`g(*args, b)`); the book drives it with a host loop
  and notes that mapping `b` onto a launch domain is the compute family's
  existing move, not new design.

## 2. Named-first vmap (the surface)

`vmap(f, axis="batch")`, mirroring the step-11 pedantry:

- Captures whose NamedArray type carries `"batch"` are woven; everything
  else broadcasts (unchanged). If NOTHING carries the axis, the build
  refuses loudly.
- Inside the body the woven name is invisible: `isel(batch=…)` refuses
  ("that axis is mapped away here"). This name-scoping is what will make
  nested `vmap("y") ∘ vmap("x")` legible and order-insensitive — but
  transform COMPOSITION is not wired in v1 (a loud refusal, §5): it
  arrives with step 13's transform driver, where the woven maps merge.
- Anonymous arrays never weave (axes without names cannot be selected by
  name); positional `in_axes` is the escape hatch that arrives on demand,
  not the default.
- Identity: `Derived("vmap", base, (("axis", name),))` — the axis name is
  cache-key material; rebuilding closures under vmap stays zero-recompile.
- Mechanism: a lowering-context door (`rules["__woven__"]`), consumed at
  the ONE bottleneck every named access already flows through
  (`_linear_index`). Weaving is a registration, not a driver edit.

## 3. jvp: one tangent engine, two doors

Forward-mode AD as a per-op rule column: every differentiable op gets a
2-line linearization (`mul`: `aḃ + ȧb`; `sqrt`: `ṫ/(2√a)`), registered in
a table the way spellings are (custom ops bring their own jvp rule —
surface A grows a column, not a mechanism).

The engine synthesizes tangent nodes alongside the EXISTING primal DAG
(memoized per (node, seed) in the lowering context, so repeated requests
share tangents). Control flow:

- `core.if` → a PARALLEL `core.if` on the same condition yielding the
  branch tangents — laziness pairs with the primal's.
- `core.for` → the loop is WIDENED: carry becomes `(primal, tangent)`,
  body rebuilt with both chains, and primal consumers of the old loop are
  re-pointed at `extract 0` of the widened one (the tangent recurrence
  needs the primal carry per-iteration; a parallel loop cannot). Honest
  cost note for `D`: each partial direction widens separately, and the
  kernel's OWN primal consumers keep the original loop — `D` of a
  loop-computed value executes the recurrence once per partial plus once
  for the primal. Fusing all directions into one widened carry is the
  recorded optimization (§5); `D` *inside* a loop body refuses loudly
  (the carry's tangent does not exist there until tangent carries land).
- Captures (`core.env`, array loads) are constants w.r.t. args → tangent 0.
- `pow` with a varying exponent refuses (write `exp(b*log(a))`); casts
  through int kill tangents (documented).

Two doors into the same engine:

- **`jvp(f)`** — the whole-region transform: `jf(*args, *tangents)` →
  `(primal, tangent)`. `Derived("jvp", base)`. The grad precursor (step 13
  = transpose of this) and the finite-difference-checked oracle.
- **`D(x)`** — the in-kernel operator (user-directed; GLSL's dFdx idea
  done analytically): partials of any intermediate w.r.t. the ENCLOSING
  KERNEL'S params, seeded with basis vectors, one partial per param,
  positionally: `di, dj = D(x)`. Structured `x` differentiates
  structurally (a tuple's tangent is a tuple — the "same layout, ×N
  directions" intuition, which is just more SSA registers until a result
  crosses the ABI). Kernels without `D` compile to identical artifacts —
  the tangent slice exists only where demanded. Compute shaders have no
  quads, hence no GLSL-style `dpdx` — analytic `D` is not a convenience
  there; it is the only derivative in town, and it is exact.
- Family sugar (`ddx`/`ddy`/`fwidth`) lives in `demo.graphics` as one-line
  DSL batteries over `D` — GL vocabulary stays out of the stdlib (090).

## 4. Named contraction (the stretch): matmul without a rules engine

`matmul(A, B, i, j)` as a lowering special form (same door as `isel`):
pair the operands' UNIQUE shared axis name (after excluding woven axes —
which is how batching composes for free), refuse zero or multiple
candidates loudly, and expand to the element loop: `for k in
range(shape[inner]): acc += A[..k..] * B[..k..]`, with the trip count read
from the shape SLOT — rank-generic, so a new inner extent is a cache HIT.
The "rules engine" is one type-rule-shaped pairing function; einsum
notation is explicitly NOT built. Output axis names for whole-array
results await array results themselves (100 §6 cut); v1's output naming is
the launch domain's, as with every element kernel.

Batched matmul is then literally `vmap(cell, axis="batch")` over rank-3
named captures — the demo arc: batch-ignorant element kernel → refusal
without vmap (the unaccounted axis is caught, not silently broadcast) →
woven kernel matches `np.matmul` per (b, i, j).

## 5. Deferred, with reasons

**Transform composition** (`vmap∘vmap`, `jvp∘vmap`, …): refused loudly in
v1; arrives with step 13's transform driver (woven-map merging, duplicate-
axis refusal). Positional `in_axes`/`out_axes` (on demand); collectives
over the mapped axis (GPU-shaped); `D(x, wrt=local)` (that is step 13's
partial-eval question); second-order `D∘D` (mechanically plausible with
duals, held until wanted); `D` inside a loop body (needs tangent carries —
refused loudly, not silently wrong); multi-directional loop fusion (one
widened carry for all of `D`'s partials instead of one loop per
direction); vmap over scalar ARGUMENTS (blocked on array args, 100 §6);
jvp tangents for array captures (blocked on the same); calling a CAPTURED
transformed kernel in-body (first-class kernel values — the capture
*summarizes* correctly via its ValueKind, the call refuses at legalize).
