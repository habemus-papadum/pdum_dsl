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
- **C3 — The spike (decision gate).** On a side path, express ONE slice
  of the tl IR as a dsl dialect — `tl.pointwise` + `tl.iota` +
  `tl.store` with Layout-shadow type rules, enough to lower a small
  kernel through the *dsl* Lowerer with a tl rule pack — and
  differential-test it against today's `ir.run`. Owner reviews
  ergonomics, LOC, and the fold/region question before any commitment.
  If the spike disappoints, C4–C5 are re-planned and C1–C2 stand.
- **C4 — Migrate primitive-by-primitive** (only after the C3 gate):
  pointwise/iota/store → reduce/scan → repeat_like/layout family →
  fold/random, each slice differential-tested; the static consumers
  (autodiff, signatures, opcount, memory) retarget the dialect ops —
  they key on op names and the one table, so this is mostly mechanical.
- **C5 — One pipeline.** The kernel tier moves onto the dsl dispatch
  path proper: coherence check, guards, two-tier caches, backend
  column — every "no" in the I.1 table becomes "yes". fn-arg inlining
  replaces per-element oracle dispatch here (it was already a P8 rung;
  it lands where it belongs).
- **C6 — Graduate.** Fold the surviving content of this document into
  200 (S.2/S.3 amendments), 220 (a principle entry if one crystallized),
  210; retire the file to `history/`.

**Open questions for the owner** (answer before or during C1/C2):

1. Does structural math stay *implicitly* host-evaluable (numbers and
   tuples always legal — my lean), or annotated?
2. The staged-citizen spelling: registration (`staged(value_and_grad)`)
   vs a protocol attribute vs a base class?
3. Does the tree producer stay a separate tiny machine (my lean: yes)?
4. `fold`/revolve in a region-based IR is the hardest design question
   of C4 — does it block the spike (no: the spike deliberately excludes
   it) and who designs it?
5. Timing: C1 immediately, or the whole pivot as one arc?

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

- *(empty — appended as C-steps land)*
