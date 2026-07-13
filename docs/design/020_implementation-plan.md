# Implementation plan: steps, notebooks, and gates

**Status:** proposed. Companion to [`proposed-architecture.md`](010_proposed-architecture.md)
(the design) and [`docs/desiderata.md`](../desiderata.md) (the wants).

## The meta-pattern

Work proceeds in **sequential steps**. Within a step, work may fan out to
parallel agents (noted per step); steps themselves never overlap. Every step
ends the same way:

1. Implementation + tests + that step's CI gates green.
2. A **self-review pass** (`/code-review`) with findings resolved.
3. A **chapter notebook** written and executing cleanly in the notebook
   harness.
4. **The step gate: a user walkthrough of the notebook.** The step is not done
   until you've run it, poked at the internals, and signed off. Feedback folds
   in before the next step starts — including *terminology corrections*, which
   land in the glossary (below).

The notebooks are not demos of the API. Each one exposes the **internals**
built in that step — printed IR at each stage, cache-key anatomy, hexdumps of
packed staging buffers, miss-counter readouts, rendered source text — with
"things to notice" prompts, links to the source files (`src/pdum/dsl/...`),
and a closing "what we can't do yet" section pointing at the next chapter
(the same implemented-vs-planned honesty discipline the M0 docs used).

**The book.** Chapters live in `docs/book/chNN-<slug>.ipynb` and are kept
green forever — they are simultaneously the step-acceptance artifact and the
permanent bottom-up teaching track ("how the machine works, layer by layer").
**Lay-of-the-land interludes** (`chNNa-<slug>.ipynb`, decided at the 2026-07-12
walkthrough): `ch07a-lay-of-the-land` baselines the *language surface* after
seven chapters — the supported list printed live from `LOWER_RULES`, refusals
shown via their real errors, the inference story, the cross-library matrix
(research R10/R11). After steps that materially widen the base pack
(notably 10 and 11), a short new interlude records the **deltas** — what
just became possible — rather than restating the baseline; interlude cells
run against the current rule pack, so a stale claim shows up as a changed
output, and the next interlude names it.
`docs/book/GLOSSARY.md` is a living terminology file; every chapter's new
terms (Handle, FnType, specialization cache, leaf, slot, aspect, surface, …) get an
entry, and walkthrough feedback edits it. A future **top-down tutorial track**
("make art in five minutes", progressive disclosure) is explicitly planned as
separate later docs work that deep-links into book chapters — the book serves
the systems-minded audience; it is not the only door.

**Notebook harness.** Extend `scripts/test_notebooks.sh` to execute
`docs/book/`. GPU-dependent cells are tagged (`gpu`) and skipped where no
adapter exists; CI runs the CPU chapters, the pre-release script runs
everything locally (Metal available).

## Gate phase-in (from architecture §6)

| CI gate | Activates at |
|---|---|
| Kernel line budget + per-file caps + PR delta report | Step 0 |
| Fingerprint-soundness fuzz | Step 1 |
| Perturbation key test (synthetic artifacts) | Step 3 → re-armed with real backends at Step 9 |
| Anti-pattern grep (no `object` field reachable from `Node` except `attrs`) | Step 4 |
| Golden printed IR per stage | Step 5 |
| Thesis test (`compiles == 1` over N calls) + `no_compile` mode | Step 8 (Python) / Step 9 (WGSL) |
| Hit-path microbench (alarm 5 µs / fail 10 µs) + flatten allocation budget | Step 9 |
| Backend-seam differential (WGSL image ≈ Python image) | Step 9 |
| Extension-locality test (new fn/method/statement ⇒ zero kernel diffs) | basic at Step 9, full at Step 10 |
| Attr lint (only `LiteralType`-originated `core.const`) | Step 10 |

---

## Phase I — the thesis without a compiler (steps 0–3)

The caching thesis is provable with **dummy artifacts** before any IR exists.
This de-risks the identity of the project immediately and gives the book its
foundation chapters.

### Step 0 — scaffolding + chapter 0

**Builds:** package skeleton (`src/pdum/dsl/kernel/`, `stdlib/`, `backends/`,
`tools/`), the line-count gate (tokenized counter + caps, tinygrad's `sz.py`
shape), notebook harness extension, pytest layout, `docs/book/GLOSSARY.md`
seeded with the ~15 core terms from the architecture doc.
**Notebook `ch00-thesis`:** no new code to inspect — states the thesis and the
map of the book, then *demonstrates the thesis with the frozen reference
asset* (`pdum.dsl_reference` disk demo, `compiles=1` over moving captures),
and names what M0 got wrong that the rebuild fixes (per-frame flatten, scalar-
only captures, coupling). Ends with the book's table of contents.
**Exit:** harness runs ch00 in CI; caps enforced on an empty kernel.

### Step 1 — types (`kernel/types.py`, fingerprint half of `valuekind.py`)

**Builds:** the frozen `Type` lattice (Scalar/Vec/Array/Record/FnType/
LiteralType), `TemplateId` (Base/Derived), structural eq/hash/serialization,
`typeof` for scalars/tuples, int range-bucketing, structural fingerprints +
the soundness fuzz. (~135 LOC kernel.)
**Notebook `ch01-types-are-values`:** construct types by hand; use them as
dict keys; `typeof(5)` vs `typeof(2**63)` vs `typeof(2**70)` (the summary-
function idea, §13 of the architecture); fingerprint vs full typeof; run a
mini soundness fuzz live and *break it deliberately* (enrich a type, forget
the fingerprint, watch the fuzzer object).
**Exit gate:** fuzz green; walkthrough.

### Step 2 — capture (`kernel/capture.py`)

**Builds:** `safe_cell`, `SourceSnapshot` memo, `make_handle`, `Handle`,
env fingerprints, `FnType` assembly. Phase A cannot fail. (~85 LOC.)
**Notebook `ch02-what-a-closure-is`:** `closure(5)`/`closure(6)` share an
`FnType`; `closure(3.0)` doesn't; **code-object value equality** shown raw
(`compile()` the same source twice → `==` but not `is`; edit one token →
unequal) — the live-coding invalidation story before any cache exists; empty
self-referential cells; snapshot coherence (mutate the source file, watch the
phase-B-style check refuse).
**Exit:** walkthrough; glossary entries for Handle/FnType/TemplateId settled.

### Step 3 — the cache (`kernel/cache.py`)

**Builds:** specialization tier + artifact tier, per-key futures (reentrant,
forward-declared recursion slots), generation, guards mechanism (synthetic
guards for now), LRU + superseded-template retirement, per-tier miss counters,
`no_compile` mode. (~105 LOC.)
**Notebook `ch03-one-compile-per-signature`:** wire the cache to a *dummy*
`compile_fn` that returns a counter-stamped object: `closure(5)`/`closure(6)`
→ one compile; edit-and-redefine → natural miss; `bump_generation()`; a
two-thread race compiling once; the perturbation test run live (mutate each
key component, read which counter moved); evict and watch retirement.
**Exit:** perturbation + race tests in CI; walkthrough. **The thesis is now
proven on the new kernel — before any compiler exists.**

### Step 3b — pipelines as values (inserted 2026-07-11; `docs/design/040_combinators-notes.md`)

**Builds:** the blessed combinator **satellite** (`src/pdum/dsl/combinators.py`
— zero kernel edits, the first extension-locality proof): internalized
plumbum semantics (`@op` stage constructors, `|` composes inert, `>` applies
once), Roles v1 over `Handle.kind`, the composition-rule registry with
`IncompatibleRoles` explanations, `f[config]` configured-stage syntax
(recorded conservatively as static until execution), materializer terminals
(stubbed), flattened `Derived("pipe", …)` identities.
**Notebook `ch04-pipelines-are-values`:** definition ≠ application; identity
stability across rebuilds; flattened associativity; role refusals read aloud;
the thesis for pipelines via a dummy dispatcher (300 rebuilt applications,
one compile, under `no_compile`).
**Exit:** walkthrough (taste-heavy surface — Role naming settles here).
**Downstream deltas:** chapter numbers after ch03 shift by one; the lowering
step gains the Derived/combinator **build rule** (dress rehearsal for
transforms); the marshaling step's contract is **bidirectional**
(PackPlan + ResultPlan, per combinators-notes §3b); from the CPU-backend
chapter on, **combinator style is the house style for examples**.

---

## Phase II — the compiler (steps 4–8)

### Step 4 — the IR (`kernel/ir.py`, `kernel/ops.py`, `kernel/printer.py`)

**Builds:** `Node`/`Region`, memoized content hash, `Builder`, structural
verifier, `OpDef` + traits + the ~30-op core table, MLIR-flavored printer.
(~220 LOC.)
**Notebook `ch05-programs-are-values`:** build the disk-shader body *by hand*
with `Builder`; print it; hash it; rebuild it in a different order → same
hash; the **no-value invariant** demonstrated by attempting to smuggle a
capture value into a node (there is no field for it — show `core.env(slot=0)`
vs `core.const` and why the difference *is* the caching thesis); the three
region ops and why there are exactly three.
**Exit:** anti-pattern grep gate on; golden print of the hand-built program.

### Step 5 — rewriting (`kernel/rewrite.py`)

**Builds:** `Pat`/`RuleSet`, the single driver (post-order, fixpoint,
rebuild-on-change), `Stage` legality (always-on), match logging. (~150 LOC.)
**Notebook `ch06-everything-is-a-rule`:** write `x+0→x` and const-folding as
rules in the notebook; watch fixpoint converge with the match log on; compose
rule sets; violate stage legality deliberately and read the error (op + loc);
peek at rule dispatch indexing (why 900 rules stay fast in tinygrad).
**Exit:** golden-IR-per-stage gate on.

### Step 6 — lowering (`kernel/lower.py` + first `stdlib/lower_rules/` pack)

**Builds:** snapshot coherence check, `classify_names` (the closed fate
taxonomy + dependency tags→guards), the fused typing+lowering driver
dispatching on `lower_ast` rules; rules for the slice subset (float arith,
compare, `IfExp`, tuple→vec, assignment, swizzle attribute, intrinsic hook);
`MissingRule`/`NoSourceError` diagnostics. (~135 kernel + first satellite
rule pack; the satellite bucket's separate line-count starts here.)
**Notebook `ch07-source-to-ir`:** decorate the disk shader; walk
`classify_names` output (every name's fate); lower and print typed IR;
captures appear as `core.env` slots; `loc` round-trip (error in the shader →
caret on the user's line); unsupported syntax → the exact error text; a
`lower_ast` rule added *live in the notebook* to widen the language (preview
of extension locality).
**Exit:** walkthrough — this chapter is where the user's mental model of
"what the frontend does" gets locked; expect glossary churn.

### Step 7 — marshaling (`kernel/pack.py`, `valuekind.py` completed, legalize stage)

**Builds:** `Leaf` vocabulary, `leaf_types`/`flatten` for scalar/tuple/Handle
kinds, `LeafPath`/`SlotSpec`/`PackPlan` + the generic packer,
`build_extractor`, the `legalize_params` stage emitting `abi.slot` ops, a toy
`PhysicalDest` for testing. (~150 LOC.)
**Notebook `ch08-one-value-n-parameters`:** legalize the lowered shader and
print the ABI-stage IR (`abi.slot` visible); build a `PackPlan` from types
alone; pack values and **hexdump the staging buffer**; change a capture value
→ repack, same plan, same artifact key; nested-closure leaves; where a future
unit conversion would sit (`SlotSpec.convert`).
**Exit:** walkthrough; leaf/slot/plan terminology settled.

### Step 8 — the Python backend and the hot path (`backends/python.py`, `kernel/api.py`)

**Builds:** Python-source renderer + exec runtime, `Backend`/`Runtime`
records, `Registry` v1 (`kernel/registry.py` — enough for backends +
lower_ast + intrinsic tables), `FastRecord` assembly, guards armed with real
dependency tags, `Handle.__call__` hot path, `@jit`. (~220 LOC — completes
the kernel budget.)
**Notebook `ch09-end-to-end-on-cpu`:** the payoff chapter. The disk shader:
source → typed IR → legalized IR → **rendered Python source (read it!)** →
executed → image displayed inline. Then the loop: 300 frames of moving
captures under `no_compile`, `compiles == 1`, counters printed. Then the
autopsy: the `FastRecord` fields one by one; time the hit path; trip a guard
(rebind a folded global) and watch refuse-or-recompile.
**Exit:** thesis test + `no_compile` gates on; full kernel line-budget report
in the notebook itself; walkthrough.

---

## Phase III — the seam proven (steps 9–10)

### Step 9 — the WGSL backend (`backends/wgsl/`) — **completes M1, the vertical slice**

**Revised at the 2026-07-12 backend detour (070_backends-notes, R12): the
step leads with COMPUTE, fragment follows as a thin variant in the same
step, sharing one runtime module.** `@workgroup_size` is pipeline-creation-
time (block-in-artifact-key confirmed); compute exercises every marshaling
contract with the fewest moving parts and is the path M0 did NOT prove
(the ch08 staging ABI already ran live against a WGSL uniform block on the
M3 — session probe). Fragment then ports M0's `GpuProgram`/`Drawer`/
`OffscreenTarget` shape onto `FastRecord` (target format in
`Backend.params_key`; per-frame = pack → `write_buffer` → draw).
**Builds:** WGSL renderer (~150), uniform-layout planner (M0's `layout.py`
generalized to leaves→`UniformSlot`), wgpu compute + offscreen + window
runtime (~200), `FragCoord` intrinsic registered *from the backend package*;
the `[grid, block, smem, stream]` config schema (070 §3) with `block`
value-specialized; multi-destination `out=` through ResultPlan.
**Parallel within step:** (a) renderer+planner, (b) wgpu runtime+targets,
(c) differential-test harness — three agents, then integrate.
**Notebook `ch10-the-gpu-and-the-seam`:** same shader, second backend: WGSL
text side-by-side with the Python source; the uniform layout table (offsets,
the vec3 align-16 footgun); compute dispatch + offscreen render;
**differential image compare** vs ch09; the live loop with `compiles == 1`;
hit-path microbench readout (alarm/fail thresholds); [gpu-tagged cells].
**Exit:** all M1 gates green (microbench, differential, thesis-on-GPU,
perturbation re-armed); walkthrough. *The architecture's day-1 claim is now
fact or falsified.*

### Step 10 — the five surfaces & batteries (`stdlib/`, registry completed)

**Base-pack table stakes (from the ch07a lay-of-the-land matrix,
2026-07-12; research R10/R11):** four registration-sized widenings ride
along with this step's stdlib work — aug-assign, `and`/`or` (base pack
picks short-circuit via `core.if`, stated as dialect policy), chained
comparisons, and tuples (literal/unpack; the `Tuple` type exists). Early
return, `if`/`for` *statements*, and the strict-join policy are step 11
machinery, not registrations. The matrix also fixed two policy stances:
**single tail return — settled at the ch07a walkthrough (2026-07-12,
user decision; 010 ledger):** one `return`, at the tail, `core.yield` is
the return, no unification machinery ever; and globals stay a loud
`NameFateError` — never a silent freeze.

**Builds:** `@overload`/`@overload_method`/`@overload_attribute` with
target-token MRO, the call-typing resolution order, registry layering
(`extend`), `Dialect` bundling, ~10 batteries (sqrt, abs, min/max, clamp,
mix, smoothstep, length, dot, a `Color` record + `to_oklab`) ported to
Python + WGSL targets; the full extension-locality CI test.
**Parallel within step:** batteries fan out across agents once the first two
(sqrt, Color) validate the surfaces.
**Notebook `ch11-the-five-surfaces`:** add `sinh` live in the notebook —
defop, overload, WGSL table entry, decomposition fallback — and use it in a
shader without restarting; the `Color` record end-to-end (captured Color →
three uniform slots, method call inlined); MRO selection shown
(`@overload(..., target=WGSL)` beating Generic); the extension-locality test
output: **zero kernel diffs**.
**Exit gate:** battery portability count vs the numba 2:1 floor (architecture
risk #4); walkthrough.

---

### Step 10b — the bench satellite (inserted 2026-07-12, ch10 walkthrough)

**Why here:** step 11's ray-march spike carries a go/no-go verdict that must
not rest on un-decomposed wall-clock, and ch15's four-targets chapter is a
differential PERF story. **Builds:** `bench` satellite (BenchmarkTools-style
adaptive sampling: warmup, tune evals/sample, run to confidence, min/median),
timing hooks at the `FastRecord.launch` seam + per-phase dispatch marks,
wgpu `timestamp-query` GPU events (feature present on the M3), a TIMELINE
widget in the viz satellite (static-HTML contract). **Notebook:** "Measuring
the machine" — decompose ch10's ~2 ms/frame into host / bridge / GPU /
readback. **Exit:** the step-9 microbench thresholds become real gates;
walkthrough.

### The post-10b pause (2026-07-12): the punning charter

Before step 11, a user-driven design pause produced
`090_core-and-extensions.md` (canon): stdlib minimalism + the squatting
test (`Color` and the 2D helpers moved to `dsl.demo.graphics`,
explicit-import by design); core+extensions conventions at the dialect
layer (vendor op namespaces, capability flags) and the runtime layer
(do/refuse charter, artifact capability protocols, `record.artifact` as
the escape hatch); the buffer/tensor-interop contract (device axis in
Array types, OWNED/ADOPTED leaves, zero-copy both directions, async
readback) **which step 11 consumes on arrival**; and the multi-device
testing ladder (fake-runtime conformance suite → probe-gated device tests
→ cross-device differential). Runtime abstraction *in code* is deferred to
step 14 — rule of three: extract from wgpu + cuda.core + Metal, bench as
first generic consumer. A user-provided Linux+CUDA box is available:
primary mode at step 14 is a handoff document + parallel agent running on
the box; direct SSH for short ABI-validation bursts (optionally a cheap
CUDA-C compile-check spike during step 11's C-backend work).

---

## Phase IV — width (steps 11–15; each still ends in a chapter + walkthrough)

| Step | Builds | Notebook | Risk retired |
|---|---|---|---|
| 11 — arrays, `core.for`, C backend | ndarray ValueKind (Buffer/Shape/Stride leaves), `core.for` lowering, shaped-kind opt-in (§13), C renderer + cc/ctypes runtime; **ray-march spike** | `ch12-data-and-loops`: color-table shader; rank-generic vs shape-in-type caching behavior shown with counters; generated C inspected; ray-march verdict documented | region-op sufficiency; planner vocabulary; C proves the seam generalizes beyond GPU |
| 12 — vmap + jvp | **vmap-over-if/for spike first** (>350 lines ⇒ re-hear per architecture); `Derived` ids, batch/jvp columns, transform driver | `ch13-transforms-are-rules`: vmap a scalar kernel, print base vs transformed IR; `grad`-precursor jvp checked against finite differences via the Python backend | transform taxation (~180 lines/region op claim tested) |
| 13 — grad | partial-eval + transpose (~450 lines, own step by design), `custom_vjp` hatch | `ch14-differentiating-a-shader`: grad of a smooth SDF shader; a design-optimization mini-loop (gradient-descend a shape parameter, live) | AD architecture |
| 14 — CUDA + Metal backends (revised 2026-07-12, 070/R13/R14: OWN both stacks) | `backends/cuda/` on **cuda.core** (opt-in-cache off; escalation: once-per-FastRecord pointer table into staging via `cuLaunchKernel`); `backends/metal/` on **ctypes-objc/PyObjC** (`setBytes`=staging, `setBuffer`=leaves); cupy/MLX demoted to optional fallback runtimes/allocators; CUDA developed design-for-skip + Modal burst (**parallel agents**, one per backend); `raw_kernel` escape hatch | `ch15-four-targets-one-ir` [platform-gated cells]: same kernel on every available target, differentially compared | multi-backend claim at N=4–5 |
| 15 — units | `Dim`/`Quantity`, unit aspect column, `Affine` pack converters | `ch16-dimensions-and-units`: mm→inch knob tweak with **zero recompiles** (pack-tier miss only, counters shown); dimension error naming both locs | two-tier law under a real domain |

**Later, unscheduled** (each becomes a step+chapter when pulled forward):
t-string mini-language (`ch17-einops-in-a-tstring`), disk cache, an appendix
chapter implementing a §12 solver satellite (egglog units *or* Z3 bounds — the
hackability claim executed), notebook/anywidget live canvas, the top-down
tutorial track, and the **differentiable-programming satellite** (desiderata
§2.1, working notes `docs/design/030_deep-learning-notes.md`: Equinox-shaped labeled closures + optax-shaped gradient chains over a
tensor dialect delegated to an MLX-class backend — pulls forward only after
M4/grad and a tensor backend exist).

---

## Working agreements

- **Step gate is the user walkthrough.** No step starts while the previous
  chapter awaits sign-off; feedback (including naming) lands before moving on.
- **Parallel agents inside a step, never across steps.** Typical split:
  implementation / test-writing / notebook drafting, or per-component fan-out
  where noted. Every step ends with a single integrated review.
- **Chapters stay green forever.** A later step that breaks an earlier
  chapter's output is a regression (CI runs the whole book); if behavior
  *legitimately* changes, the chapter is updated in the same PR — the book is
  documentation of the system as it is, never as it was.
- **LOC ledger in every step's PR:** kernel used / cap, satellite buckets,
  per-file caps — the architecture's budgets surfaced continuously, not
  audited after the fact.
- **Deviations from the architecture doc** discovered during implementation
  get a dated note in `proposed-architecture.md` §10's ledger, not silent
  drift.
