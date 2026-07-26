# 240 — The cleanup pivot

**Status: TRANSIENT working document (P8.5).** This is the pivot's home:
education first, then the plan, then the ledger of what lands. Sections
graduate into 200/210/220 as work completes; the document retires to
`history/` when it is empty. Nothing in here changes ratified syntax —
the skipped spec tests remain the forward contract.

**Why now.** The `value_and_grad`-in-kernel investigation exposed that
the kernel tier stages host code by *convention* (any callable applied
to zero tensors executes at lower time, unconditionally), patches the
consequences with an `isinstance` special case, and runs on a second
lowering machine with a narrower honesty envelope than the dsl core.
Every remaining P8 rung (ambient triple, buffer reads, fn-arg inlining,
vertex/fragment) builds on exactly this machinery. Restructure first,
build once.

---

## Part I — The machinery as it is

### I.1 The two pipelines

**The dsl tier** (`pdum.dsl.registry.Registry.dispatch`) — what happens
when you call a jitted handle or `reference(f)(...)`:

*Per-call (the hit path, every loop iteration).* Fingerprint the
arguments — **types, never values**. Build the key:
`(target.fp, arg fingerprints, backend.fp, cache generation)`. Probe
tier 1 (`SpecializationCache`). Check **guards** — one identity triple
per captured closure cell, so a rebound capture misses loudly instead
of serving stale bytes. Then run the precompiled `FastRecord`:
*extract* the capture/argument leaves → *pack* them into a reused
staging buffer → *launch* the artifact. No parsing, no typeof, no IR.

*The miss path (first time, or a real change).* `typeof` the args →
**lower** (`lower_handle`): first the *coherence check* — the captured
source snapshot must still compile to code value-equal to the captured
code object (recursively, through captured helpers) — then AST → typed
`Region` via the rule packs, with callee inlining (env paths prefixed,
provenance chained) and Derived build rules for transforms → **rewrite
stages**: backend-gated decompositions, env normalization,
parameter legalization → probe tier 2 (`ArtifactCache`) on the
**content key** `(region.key, backend.fp)` — identical IR never renders
twice, even from different templates.

*The compile step.* Only on a tier-2 miss: **render** the Region to
source through the backend's spelling table (`code_for_op`), compile
it, then build the plan, the extractor, the guard tuple, and install
the `FastRecord` under a per-key future.

**The tl kernel tier** (`pdum.tl.kernel`) — what happens when you call
a `@compute` kernel:

*Per-call.* Key = `(kernel code fp, arg fingerprints, tap names)`;
probe one `Memo`. On a hit: the overlap refusal, then the **rebind
channel** — every fn-argument's current value is written into
`_ARG_BINDINGS` under its marker names — then `ir.run` interprets the
Program over numpy.

*The miss path.* `_compile`: AST → a tl `Program` via
`_KernelLowerer` (thread iotas over the writable lattice, stores as
token-threaded instructions, claims registered by the naming law,
fn-arg calls emitted as launch-rebindable markers). There is no
separate compile step: the Program *is* the artifact and `ir.run` is
the executor.

**The asymmetry, in one table.** This is the honesty gap the pivot
closes:

| | dsl tier | tl kernel tier |
|---|---|---|
| identity | types never values | types never values |
| two-tier cache (spec + content) | yes | no (one Memo) |
| coherence check (stale source) | yes, recursive | **no** |
| capture guards / value keying | yes (cell triples) | **no** — a captured float or edited helper serves stale silently |
| events seam | yes | yes (kernel.miss, spans) |
| backend column (render/spell) | yes | no (interpreter only) |
| host evaluation during lowering | rule-pack policy, explicit refusals | **unconditional** for tensor-free calls |

(The assemblage tier solved value-keying once already — `_canon` folds
captured values into unit fingerprints. The kernel tier never got that
fix.)

### I.2 The engines and IRs — the inventory

There are **two IRs** and **three body-lowering machines** today:

1. **The dsl IR** (`pdum/dsl/ir.py`): typed `Node`/`Region`, ops
   namespaced by dialect, structured control flow as regions
   (`core.if`, `core.for`, `core.call`), memoized content keys, a
   rewrite driver, verify, and source-rendering backends. Lowered into
   by the **dsl Lowerer** (`lower.py` + the `value.py` rule pack).
   Current op namespaces — the concrete dialects: **`core.*`**
   (operators, structure, control), **`abi.*`** (marshaling slots),
   **`pw.*`** (the function primitives, numpy-spelled on the
   reference); `wgsl.*` is reserved for the device era.

2. **The tl IR** (`pdum/tl/ir.py`): a flat SSA `Program` of `Instr`s.
   Leaves: `input const iota random token`. Compute:
   `pointwise reduce scan fold materialize round_to repeat_like store
   with_value_units`. Plus the layout-op family (`slice select shift
   rename repeat flip split merge diagonal window decimate pad stencil
   …`) as zero-cost view instructions. Tensor-typed throughout, with
   layout *shadows* inferred per instruction; `ir.run` is a numpy
   interpreter; `autodiff.py` derives adjoint programs;
   `signatures.py`/`opcount.py`/`memory.py` are static consumers.
   Lowered into by the **tl `_Lifter`** (`lifting.py`) for step/unit
   bodies and its subclass **`_KernelLowerer`** for kernels.

3. **The tree producer** (`pdum/tl/producer.py`): scalar marker bodies
   → `Arg/Const/Prim` trees (the schema now homed in
   `pdum.dsl.derivative`), consumed by marker partials and signatures.
   Tiny and schema-stable.

**Already unified** (this month's work): the marker vocabulary and its
numpy semantics (`pdum.dsl.markers`, tl re-exports), the one derivative
table + tree schema (`pdum.dsl.derivative`), the naming engine (tl's
`_Emit` uses `pdum.dsl.naming.Namer`), the events seam, the cache
classes (`Memo` is `pdum.dsl.cache`). The one-body-language law unified
the *syntax*. What is **not** unified: the IR, the lowering engine, the
call-resolution policy, the cache/guard discipline, and the
inline-vs-host-evaluate decision.

### I.3 Where the seams leak (the agreed assessment)

1. **Door 4 is a convention.** `lifting.py`'s call resolution ends in:
   callable + zero tensor args → `target(*args, **kwargs)` at lower
   time, unconditionally. Anything importable is a staging combinator
   by accident. Built for structural math (`n - 1` in a slice extent);
   walked through by `value_and_grad`.
2. **The kernel key is blind** to captured values and captured helper
   code (no guards, no coherence check, no `_canon`). First-invocation
   values can bake into artifacts that later invocations warm-hit.
3. **Staging is not a protocol.** The rebind path special-cases
   `_ValueAndGrad` by `isinstance`; the second staged transform copies
   the special case or exposes the gap.
4. **The fn-arg boundary is where the two machines meet awkwardly**:
   per-element `np.vectorize(reference(g))` dispatch (sanctioned
   oracle, but N markers per flat output), and the destructure pattern
   is trusted rather than checked against a result type (FnType's
   result slot is reserved, empty).
5. **Duplication**: two AST walkers, two call-resolution policies, two
   inlining mechanisms, two straight-line enforcements, two cache
   disciplines — one language.

### I.4 The presumptive direction (owner-flagged, unvalidated)

The tensor tier genuinely needs its own IR *content*: tensor-typed
values with layout shadows, the primitive set, the adjoint table, cost
oracles, `fold`/revolve. The presumption to validate: that content can
be an **extension** of the dsl methodology rather than a second
methodology — tl ops as a dialect (`tl.pointwise`, `tl.reduce`,
`tl.store`, …) registered via `defop` with type rules that carry
Layout shadows, over the same `Node`/`Region` infrastructure, with
`ir.run` becoming an evaluation column and the rewrite driver replacing
bespoke program surgery. **This is presumptive until the spike (C3)
proves the ergonomics.** The producer/tree tier likely stays as-is
(tiny, schema-stable, already homed in dsl).

---

## Part II — The cleanup path

Each step ends green and committed, march discipline unchanged. Steps
C1–C2 are worth doing even if C3 fails its gate.

- **C0 — Freeze the safety net.** Name the invariant battery that
  DEFINES behavior-preservation for this pivot: the differentials
  (iota-unification, two-consumers, eager-vs-lowered), the zoo gate,
  the recompute theorem, the at-kink pins, the refusal-contract
  battery, the key-discipline pins. Record the list in this document;
  every C-step keeps it green.
- **C1 — Close the honesty holes in place** (no restructuring):
  (a) door 4 becomes explicit — host evaluation at lower time is legal
  for *structural* results (numbers, strings, tuples thereof) and for
  *registered staged citizens*; anything else refuses loudly, naming
  the two fixes; (b) the kernel key learns captured-value and
  captured-helper fingerprints (the `_canon` precedent) or grows dsl
  guards; (c) restaging becomes a declared protocol on staged citizens,
  deleting the `isinstance`.
- **C2 — One staging mechanism.** Staged transforms in kernel bodies go
  through the same machinery as the dsl tier's macros/build rules:
  fp-keyed IR-in/IR-out (the "generated function" model), registered,
  never convention-recognized. `value_and_grad` becomes the first
  ordinary citizen of that mechanism instead of its special case.
- **C3 — The spike, in two stages (decision gates; the hard part stays
  IN the spike — owner-ruled).** *C3a:* express one easy slice of the
  tl IR as a dsl dialect — `tl.pointwise` + `tl.iota` + `tl.store` with
  Layout-shadow type rules, enough to lower a small kernel through the
  *dsl* Lowerer with a tl rule pack — differential-tested against
  today's `ir.run`. *C3b:* the hard part, NOT deferred — `fold` (and
  the revolve schedule) as a region-carrying dialect op, plus the
  adjoint derivation over it, differential-tested against today's
  autodiff on the recompute-theorem case. Owner reviews after EACH
  stage. If either stage disappoints, C4–C5 are re-planned and C1–C2
  stand.
- **C4 — Migrate primitive-by-primitive** (only after the C3 gate):
  pointwise/iota/store → reduce/scan → repeat_like/layout family →
  fold/random, each slice differential-tested; the static consumers
  (autodiff, signatures, opcount, memory) retarget the dialect ops —
  they key on op names and the one table, so this is mostly mechanical.
- **C5 — One pipeline.** The kernel tier moves onto the dsl dispatch
  path proper: coherence check, guards, two-tier caches, backend
  column — every "no" in the I.1 table becomes "yes". fn-arg inlining
  replaces per-element oracle dispatch here (it was already a P8 rung;
  it lands where it belongs). **DO-NOT-FORGET (owner-acked):**
  (B) `tl.uniform` unifies into the `abi.*` dialect here — it is
  `abi.slot`'s kernel-tier cousin, same concept, two tiers; merge at
  C5, do not let them coexist past it.
- **C6 — Graduate.** Fold the surviving content of this document into
  200 (S.2/S.3 amendments), 220 (a principle entry if one crystallized),
  210; retire the file to `history/`. **DO-NOT-FORGET (owner-acked):**
  (A) the token/store ORDERING mechanism is a promotion candidate out
  of tl — tile barriers (L4 brief) and any effectful dialect consume
  the same tokens; promote to a shared/core dialect WHEN L4 asks, and
  record the pointer in §8's brief at graduation so L4 cannot miss it.
  Organizing principle (owner-acked): op DEFINITION SITES own all
  their columns — type rule, evaluator row, adjoint row, spelling
  rows.

**Owner rulings** (received; supersede the open questions):

1. **Structural math stays implicit.** Numbers, tuples, and structural
   facts (charts, extents) host-evaluate without annotation. The new
   refusal targets exactly the smell: a host call returning a FUNCTION
   CITIZEN (fp-carrying) must come from a declared staged transform.
2. **Staged citizens are functional and composable** — a decorator
   (`@staged`), not a class hierarchy. Sequences of transformations
   must compose from smaller transformations: recipes chain, so
   `t2(t1(f))` restages through both. Undecorated inlining stays free;
   decorated helpers are ALSO the freedom to annotate which dialect a
   function's body expects (a door C2/C3 may use).
3. **One IR, many dialects, is the lean** — and the pivot's MAIN
   OBJECTIVE is restated: find the right formulation for declaring and
   detecting which dialect region a body is in. The dialect space is
   plausibly a tree — a control-flow-free scalar core (the tiny
   Arg/Const/Prim tree may be its most foundational expression), the
   value language adding bounded control flow, tensor-typed
   straight-line above, stores/ambient at the kernel leaf. The tree
   producer's fate is decided BY that staging, not before it.
4. **The hard part stays in the spike** — C3 is split (C3a easy slice,
   C3b fold/revolve in regions), never deferred; de-risking is worth
   the extra time.
5. **The pivot runs as one arc**, C0 → C6, step by step, then the march
   resumes at ambient derivatives.

---

## Part III — De-risking: the journey, and the chaos budget

**The journey we must not lose:** P8 graphics — ambient triple
(`block_idx`/`grid_layout`/`global_thread_idx`), buffer reads v1,
vertex/fragment with the reference interpolator, tagless varyings,
subset pairing, MRT — closing with the translation-only **WebGPU
conformance executor**; then P9 (indexing family) and the L4 handoff.
The pivot is **P8.5, inserted before the remaining rungs because they
all build on the kernel lowerer** — its output is a better foundation,
not a detour. The skipped spec tests in `test_kernel_spec.py` are the
contract that survives the pivot untouched: when the pivot ends, the
same spellings un-skip on better machinery.

**Chaos rules** (all existing law, restated for the pivot):

1. Every step green + committed; the C0 battery defines
   behavior-preserving.
2. Spike-before-commit for the IR question; C1–C2 are valuable even if
   C3 fails.
3. No dual-running beyond within-step scaffolding (the 200 not-doing
   list binds the pivot too); a superseded mechanism is deleted in the
   step that supersedes it.
4. No new features, no backends, no graphics, no spec-syntax changes
   inside the pivot. New syntax rulings go through 200 as always.
5. LOC budgets stay tripwires; raises are conscious and recorded.
6. This document is the pivot's ledger: each C-step appends what
   landed, what was deleted, and what graduated — so at any moment the
   distance to "done" is readable here.

## Ledger

**C0 — the frozen safety net (landed).** The invariant battery that
DEFINES behavior-preserving for every pivot step. All of it green at
freeze (589 passed, 17 skipped); any pivot step that reddens a line of
this list is wrong by definition:

- *The differentials* — `test_kernel.py`:
  `test_the_s3_example_runs_on_the_reference_evaluator`,
  `test_the_iota_unification_differential`,
  `test_the_two_consumers_differential`; the eager-vs-lowered
  differentials in `test_lifting.py` / `test_scope_assemblage.py`.
- *The zoo gate* — `test_zoo.py` (every pin) + `test_transforms.py`.
- *The recompute theorem* — `test_random.py::`
  `test_the_recompute_theorem_revolve_equals_store_all_with_dropout_on`
  and the whole random-field battery.
- *The derivative contract* — `test_at_kink.py` (incl. the one-home
  identity pins), `packages/dsl/tests/test_derivative.py`, the marker
  granularity gate (`test_marker_granularity.py`).
- *The refusal contracts* — both `test_refusal_contract.py` batteries
  (dsl + tl): messages pinned by wording.
- *Key discipline + compile-once* — `test_kernel.py::`
  `test_key_discipline_shape_miss_value_hit_launch_never_keys_fn_swap_miss`,
  `::test_compile_once_thesis_for_function_valued_arguments`; the dsl
  cache/runtime batteries (`test_cache.py`, `test_runtime.py`,
  `test_traced_dispatch.py`).
- *The kernel dialect* — the LIVE tests of `test_kernel.py` and
  `test_kernel_spec.py` (claiming, invalidation, config bracket,
  in-kernel staging, analytic AA) and the SKIPPED spec tests as frozen
  spellings (may not be respelled inside the pivot).
- *Static consumers* — `test_opcount.py`, `test_memory.py`,
  `test_signatures.py`, `test_autodiff.py`, `test_fold.py`.

**C1 — the honesty holes, closed in place (landed).**
(a) *Door 4 is explicit*: the host-evaluation branch in
`lifting._Lifter.call` still evaluates structural results implicitly
(ratified), but a result that is a FUNCTION CITIZEN (fp-carrying) now
requires the callable to be declared `@staged`
(`pdum.dsl.staging`, exported from `pdum.dsl`) — otherwise a loud
refusal naming both fixes. Declared staged calls are recorded as
replayable recipes (`staged_recipes`, shared down the inline chain).
(b) *Restaging is a protocol, not an isinstance*:
`_KernelLowerer._resolve_staged` walks the recipe chain from a staged
result to the ONE kernel parameter underneath and builds a replay
closure; the `_ValueAndGrad` special case is deleted;
`value_and_grad` is now `@staged` — the mechanism's first ordinary
citizen. Composition works and is pinned
(`test_staged_transforms_compose_and_restage`): `t2(t1(f))` restages
through both on a warm hit. A staged call referencing zero or multiple
parameters refuses (relaxable later, recorded here).
(c) *The kernel key is no longer blind*: `_env_fp` fingerprints the
captured environment the body can see (referenced names only —
markers by name, fn-citizens by fp, scalars by value, user helpers by
recursive code+env fp, `pdum.*` library callables by qualified name,
opaque objects by type). A rebound global or an edited helper is now a
MISS with fresh values, never a stale artifact — pinned by
`test_rebound_captured_global_misses_never_stale` and
`test_edited_captured_helper_misses_never_stale`. All C0 battery
lines stayed green throughout (593 passed at land).

**C2 — one staging family, declaration-first (landed).**
`pdum/dsl/staging.py` is now the family's home and doctrine: two
decorators, one contract. `@macro` declares a value-space lowering
macro (IR-in/IR-out at the call site — `with_respect_to` now carries
the decorator instead of a hand-set attribute); `@staged` declares a
function-space staged transform (citizen-in/citizen-out, build-rule
IR, replayable recipes — `value_and_grad`). Recognition in the tl
lifter is DECLARATION-FIRST: a declared staged call host-evaluates,
is VALIDATED to return a function citizen (a declared transform
returning structural data refuses — pinned), and records its recipe;
the result-type sniff survives only as the refusal backstop for
undeclared calls. Both decorators export from `pdum.dsl`. *Deliberate
scope cut, recorded:* staged calls inside dsl-tier `@jit` bodies
(staged-citizen locals in the value language) are NOT opened here —
that door belongs to the dialect formulation (C3), where "which
region am I in" gets its real answer. 594 passed at land.

**C3a — the spike, AWAITING OWNER REVIEW (landed as a side path).**
One self-contained file — `packages/tensorlib/tests/test_spike_tl_dialect.py`
(~170 counted lines, nothing under src/ touched) — expresses the easy
slice (`tl.pointwise`/`tl.iota`/`tl.store` + TensorType/TokenType) as
a dsl dialect. All three pins pass first-run:
(1) *the differential is bit-identical* — the same kernel body through
today's `_KernelLowerer` + `ir.run` and through the dsl `Lowerer` with
a tl rule pack + a 40-line evaluation visitor;
(2) *alignment became an ordinary type rule* — a misaligned store
refuses AT EMISSION with the source location in the message, for free
(the dsl Builder's loc plumbing; tl's lifter hand-builds these);
(3) *content keys came free* — two lowerings of one body are
`region.key`-identical (the cache-efficiency premise).
The dialect-region question got its first empirical answer: ONE
lowerer, packs LAYER (three rule overrides delegating to the value
pack), and TENSOR-TYPEDNESS selects the path — no annotations needed
for this slice; scalar subtrees (const lifting, host math) flowed
through the base pack unchanged.
*Honest frictions, for the owner's review:* (i) kernels have no
`return`, so the spike drives the Lowerer manually (`run_body` demands
a return — C4 needs a no-return body driver); (ii) the pack had to
override `Assign`/`BinOp`/`Call` wholesale and delegate — a
first-class dialect-layering seam (per-node dispatch by operand type)
would shrink packs; (iii) `thread_idx` is recognized by NAME in the
pack — should become intrinsic-object recognition when real.
*Scope cuts carried to C4:* charts/labels/levels not yet in
TensorType; store index checking; fn-valued args; reduce. **fold is
C3b — the hard part, next, pending the owner's C3a verdict.**
597 passed at land.

**C3b — fold/revolve through the dialect, AWAITING OWNER REVIEW.**
Appended to the same spike file, in the owner-proposed TWO-PASS shape:
`check_fold_step_supported` (pass 1) walks the step region and refuses
unsupported ops WITH the reason and the supported set — never
mid-derivation (pinned); pass 2 does the work. Results, all green:
- *b1* — `tl.fold` as a region-carrying op (the type rule computes the
  element type by REMOVING the folded dim from the source's layout
  type and checks binders and yield against it); forward differential
  vs tl's `ir.run` fold is bit-identical, from ONE single-source step
  function lowered by both engines.
- *b2* — the VJP is REGION-IN/REGION-OUT (the generated-function
  model made literal): `derive_step_vjp` consumes the step region,
  recomputes the forward inside the adjoint region, splices slopes
  from THE one table (`Prim` trees → `tl.pointwise` chains), and
  asserts the element's adjoint is gradient-free (the mask
  discipline). Store-all gradient vs tl `autodiff.grad` agrees to
  1e-12 (different engines, different summation order — exact
  equality is reserved for within-engine).
- *b3* — **the recompute theorem holds through the dialect**: revolve
  (checkpoint, recompute segments, RE-SELECT the zero-memory random
  field at ABSOLUTE coordinates) and store-all produce BIT-IDENTICAL
  gradients with dropout on. Coordinate provenance survived the
  region formulation — the risk the owner kept inside the spike, and
  it held.
*The surfaced design answer (predicted as a fork; resolved by
evidence):* schedules lived naturally as EVALUATION STRATEGIES over
the same two regions (step + vjp) — store-all and revolve share all
IR; nothing pushed the schedule into the IR. Evidence for
schedule-as-execution-column; baking a schedule into IR remains
available to the L4 certified-descent era without conflict.
*Owner guidance appended (post-C4.3b):* the fold boundary IS the
natural checkpointing/activation-recompute site — its region
structure already declares everything a future engine needs (step
boundary, carried state, elements re-derivable at absolute
coordinates), and buffer reuse/ping-pong reasoning hangs off the
token/store ordering and the overlap contract. WHETHER to checkpoint
is a machine-characteristics decision (L4); at most, HINT helpers
(config-tier data, never identity) may be added WHEN such an engine
exists to read them — none before.
*A dialect-hierarchy finding:* the single-source step exposed that
STEP bodies (S.1 tier: `pointwise(where, ...)` spelled) and KERNEL
bodies (scalar tier: bare markers) are DIFFERENT dialect leaves
sharing the same ops — the pack needed two vocabulary entries
(`pointwise`, `const_like`), concrete input for the C4 region-
formulation design. 601 passed at land. **The full spike (C3a + C3b)
now awaits the owner's verdict before C4 begins.**

**C4.1 — the dialect foundation lands in src/ (landed; verdict was
"go").** `pdum/tl/dialect.py` is the spike promoted with the design
items resolved: the **per-kind yield protocol** (`lower_body(fn,
arg_types, kind=...)` — a `step` yields its returned value, a
`compute` kernel yields its final token, and a kernel `return`
refuses with the pinned voice — kinds declare their yields, users add
no ceremony); **object-based intrinsic recognition** (`thread_idx`
and the S.1 vocabulary recognized by OBJECT IDENTITY from the body's
own globals, never name strings — and the P5 shadowing lesson bit
again on the way in: `pdum.tl.compute` the package attr is the
DECORATOR, so direct-name imports only); **the op-selection pattern
written once** (`_typed_rule` — one factory builds the operator and
comparison rules; a dialect extends by table rows, not by wrapping);
the two-pass check, the region VJP, `run_region` (the evaluation
column, reusing `ir._store` and eager iota), and `fold_grad` with
store-all and revolve as ONE code path parameterized by `slots`.
The spike file was promoted to `tests/test_dialect.py` — eight pins,
kept verbatim where possible, now testing src/. Nothing deleted yet
(the foundation supersedes nothing until C4.2 flips the kernel).
602 passed at land. **Next: C4.2 — the kernel switch** (@compute
lowers through the dialect; claiming/taps, fn-arg recipes, the env
fingerprint, and launch migrate onto Regions; `_KernelLowerer`
retires; kernel keys become region content keys).

**C4.2 — the kernel switch (landed).** Every `@compute` kernel now
lowers through the ONE dsl Lowerer with the KERNEL RULE PACK — a
layer over the tl dialect pack, itself a layer over the base value
pack — onto `tl.*` Regions executed by `run_region`.
**`_KernelLowerer` is deleted** (the whole class and its `_Lifter`
kernel machinery — superseded and removed in the same step, per the
chaos rules); `ir.Program`/`ir.run` no longer serve the kernel tier.
What migrated onto Regions, behavior-identical (the full kernel
battery — 20 tests — plus the live spec tests passed unchanged,
except one introspection line reading region ops instead of Program
instrs): tagless claiming with honest invalidation (claims flow
through inlined helpers via the SHARED build context — the 130 §7
seam again), config-bracket taps as appended region params +
token-threaded stores, fn-valued arguments with per-launch rebind and
tuple-result destructuring, staged recipes with replay, the C1/C2
staging door (kernel-side, declaration-first), the env fingerprint,
the overlap refusal, and every pinned refusal voice. The iota
unification is now LITERAL: thread coordinates ARE `tl.iota` nodes.
One new mechanism the switch demanded: **the kernel capture shim**
(`_kernel_handle`) — the same snapshot/coherence surface as a dsl
Handle, but closure values stay RAW, because kernel bodies close over
helpers, markers, and staged transforms: compile-time CITIZENS keyed
by the env fingerprint, never typed env slots. Host-scalar freevars
and globals const-lift through `_k_name` (module constants are kernel
vocabulary). Name resolution is one order everywhere: locals →
closure freevars (raw) → the body's globals. 602 passed at land, C0
battery green throughout. *Remaining C4 slices:* the step/assemblage
tier (reduce/scan, the layout family, charts/labels/levels + dtype in
TensorType, autodiff/signatures/opcount retarget), then C5 (dispatch
alignment: coherence + guards + two-tier caches for kernels), C6
graduation.

**C4.2b — THE LITERAL DOCTRINE (owner-ruled, landed).** The C4.2
scalar treatment was convicted as an anti-pattern and replaced.
The rule: **untagged is data, always** — an unmarked scalar reaching
a body (captured global, closure value) is runtime data: it becomes a
per-launch UNIFORM SLOT (`tl.uniform`, re-read from the environment
at every launch), warm on change, its value never entering identity
(the env fingerprint keys its name and TYPE only). A value becomes a
compile-time constant only through a DECLARED door, of which there
are three with one meaning: the operator-definition annotation
(`n: Literal[int]` — everything passed there is structural; the
already-settled §1.5 door), the call site (`f(literal(k), ...)`), and
the value's definition site (`C = literal(0.79...)` — new
`pdum.dsl.literal`/`LiteralValue`). Declared literals bake and
value-key — recompiling on change is then CHOSEN. Source-text
constants are code, not captures: inside the code fingerprint,
unaffected. Machinery (fold/reduce) gets NO scalar special cases:
operators declare their structural slots and pass-1 CHECKS that
incoming IR is compile-time constant, refusing with the reason.
Pinned both ways: rebinding an unmarked global is a warm hit with the
fresh value under `forbid("kernel.miss")`; rebinding a
`literal(...)`-wrapped one is a refused miss under forbid and
computes the new value on recompile. Budget: owner raised the total
cap 3000 → 3400 for the migration ("don't bang your head");
re-measure at C6. 603 passed at land.

**C4.3a — the step-tier op families, bridged (landed).** The dialect
now covers the WHOLE tl op inventory through two migration bridges
with one source of truth: type rules for
reduce/scan/materialize/round_to/repeat_like/random/with_value_units
and the entire layout-method family are `_r_bridge(base)` — build the
instruction, ask the incumbent `infer_instr`; evaluation is the
extracted `ir.eval_instr` (the run loop refactored into a
per-instruction evaluator both engines share — a pure refactor under
the C0 battery). `TensorType` gained the full Layout as a
NON-IDENTITY shadow payload (identity stays the dims lattice —
alignment is by NAME, order-free, per tl's law; charts/strides ride
for inference; dtype still deferred). The pack gained the S.1
spellings (`reduce`/`scan`/`repeat_like` by object identity), the
frozen layout-method family (reusing `lifting._METHODS` packers and
the STRUCTURAL-slot refusal verbatim), and `lower_body(host=...)`
for structural/kw bindings. **Flagship pin:** the zoo's `layernorm` —
verbatim, untouched — lowers through the dialect and matches its own
eager execution bit-for-bit; plus a shift/slice/pad chain
differential and the structural-slot refusal. NOT yet switched: every
consumer (lift_step, assemblage, autodiff, the zoo) still runs the
incumbent Program world — C4.3b+ moves them. 606 passed at land.

**C4.3b — the general region VJP (landed).** The adjoint walker
factored out of the fold-specific derivation into ONE engine
(`_substitute` + `_pullback`), and grew two op families: **reduce**
(sum: repeat back; mean: repeat back over the static reduced numel)
and **repeat_like** (reduce-sum over the added dims; the like operand
is layout-reference only, per doctrine — it receives no adjoint).
`derive_vjp(region)` is the general form — params + upstream seed in,
a tuple of per-param adjoints out, exact zeros materialized aligned
to their params; `derive_step_vjp` became a thin role assignment over
the same walker (element-gradient-free still checked). Pass 1
(`check_vjp_supported`) refuses per-op with the arriving-slice reason
— a max-reduce refuses toward the first-occurrence-mask slice, never
a silent wrong gradient (pinned). **Flagship pin:** the zoo's
layernorm differentiated through the dialect — gradients wrt x, g,
AND b match the incumbent autodiff engine to 1e-12. `run_region`
gained core.tuple/extract. DO-NOT-FORGET items (A: token promotion at
L4; B: tl.uniform→abi at C5) recorded on their steps above,
owner-acked. 608 passed at land. *Next slices:* first-occurrence-mask
reducers + scan + layout-op adjoints, then the lift_step/assemblage
switch (deleting the incumbent lowering as superseded).

**C4.3c — the migration view (landed; ROUTE CHANGE, recorded
honestly).** The plan said "refactor the incumbent's adjoint rules
against an emitter protocol." Reading the max/min partition chain
(closures over the Program builder, ~60 intricate lines) showed that
transcription or extraction both risk the most delicate knowledge in
the codebase for no consumer's benefit yet. The cheaper, safer route
per the pivot's own bridge philosophy: **`export_program(region,
names)`** — a dialect region rendered as an incumbent `Program`
(generic op mapping, consts materialized at use sites with the
incumbent's chart/label/placement restamp discipline, multi-output
tuples → output lists) — so every Program consumer serves regions
UNCHANGED: autodiff with its max/scan/layout adjoints, signatures,
opcount, memory. Adjoint knowledge stays SINGLE-COPY (the incumbent's,
battle-tested); the region-native `derive_vjp` remains for its live
uses and grows opportunistically; the view dies when the last
consumer retargets. **Crown pin:** the incumbent autodiff — the
partition law with a TIE in the reduced dim, plus shift/slice/pad
adjoints — runs over an exported region bit-identical to the
incumbent-lifted path. Plus round-trip and two-output (FDTD-shaped)
export pins. 611 passed at land. *Next:* move the plain-helper inline
branch + staging door from the kernel pack into the shared dialect
pack, then switch `lift_step` onto lower_body+export behind its
existing API (message-parity work: the step tier's pinned refusal
voices), then assemblage, deleting `_Lifter` when both are off it.

**C4.3d — the step switch (landed).** `lift_step` now lowers through
the ONE dsl Lowerer with the tl pack and renders back through the
migration view — same signature, same refusal voices, every consumer
untouched: the whole zoo (heat2d included), the recompute theorem,
fold/transforms, memory counts, and the naming-law pin (binding names
become SSA names — recorded during lowering in a names ledger the
exporter consumes) all green through the new path. What the flip
forced into the SHARED pack, each an incumbent semantic honored:
helper inlining with kwargs/kw-only/defaults and host-first argument
binding (strings, extents with host math like `max(delta, 0)`,
**splat dicts); the staging door and citizen refusals (moved from the
kernel pack — one door, both tiers); the full recognized S.1 set by
object identity (`iota`/`extent`/`contract` joined; multi-operand
record-state reduces spread); a guard refusing to inline tensor-
library machinery ("lowers by recognition, not inlining");
**constants are HOST values in tl bodies** (the incumbent semantics —
they lift only on meeting tensors; scalar stores broadcast via a
bridged const); unary minus on tensors; lambdas via the incumbent
extractor. The kernel pack slimmed to its genuine layer (ambient
iota-recording + fn-args) over the shared pack. `_Lifter`'s step
entry point is superseded; the class survives ONLY for the assemblage
unit lowerer — deleted when that switches. 611 passed at land.
