# 280 — Graphics & runtimes: the design exercise (DRAFT)

**Status: DRAFT — an exploration, not canon.** Nothing here is ratified;
this document is the working surface of the graphics/runtime campaign
(worktree `graphics-design`). It is written hypothetical-syntax-first
in the tour's spirit: we write the programs we wish users could write,
and spikes under `explorations/graphics/` keep the prose honest. Rides
on 200 §S.4, 210, 260, and 270 (which wins where they differ).

## The fuzzy set, named

Four questions, one campaign:

1. **The kernel syntax charter.** Not everything the tensor layer
   offers belongs inside a compute/vertex/fragment body. Rule the
   surface per tier — in, out, or deferred — with rationale. Seed
   position (owner): in-kernel syntax is (a) indexed reads and writes
   into tensor buffers, (b) scalar expression-building. Control flow:
   see the position below.
2. **Runtime vs backend.** A *runtime* owns devices, host↔device
   transfer, launch, streams/events, profiling, presentation. A
   *backend* takes IR to executable code, which a runtime launches.
   Usually 1:1 — but 270's thesis (no magic compile function; the
   program stays runnable at every point in the chain) means the
   backend notion must fit multiple incremental paths, not one
   `compile()`. Definitions to be fitted to the seams that already
   exist (the executor column, the `Backend` registry record,
   `Buffer.device`, the staging ABI).
3. **The render-loop syntax.** What allocation, device use, and a real
   frame loop read like — the WebGPU windowed-canvas integration is
   the forcing use case. Written twice: today's re-upload world and
   the committed-residency world (the tour's skip-tagged cell); the
   syntax must be the bridge.
4. **The Metal twin.** The same program with the runtime/backend
   swapped. The diff between the two programs defines what "backend"
   and "runtime" mean here (unified memory kills half the transfer
   API).

## The control-flow position (owner, this campaign's opening)

The select-normal-form is DELIBERATE, not a limitation: strict
both-branches semantics keeps typing simple and autodiff clean, and
the target programs are essentially straight-line — divergence is
meant to recombine quickly (min/max-shaped), so both-sides evaluation
is cheap in practice. The analyses never see control flow. The door
held open: EMISSION may reconstruct real `if` from select where the
arms are pure (the splicer already guarantees purity), so the IR's
simplicity need not cost machine-level optimality — that recovery is a
backend peephole, never an IR feature.

**Spike verdict (`spike_controlflow`, measured on the M3/Metal with
GPU timestamp queries): the position HOLDS for branches and FAILS for
loops.** Select-vs-`if` parity is exactly 1.000× for cheap arms in
both divergence regimes (the hardware compiler if-converts; a
methodology control with ~24-op arms shows 0.533× coherent, so the rig
sees differences when they exist). "Recombines quickly" is true for a
branch — one arm, one join — and FALSE for a data-dependent trip
count: in a 64-step SDF raymarcher the flat done-mask form costs
**2.2–4.0×** vs a real `break` (ratio tracks 64/mean-steps almost
exactly; a deep-march control where 85% of threads saturate falls to
1.09×, pinning the interpretation as skipped work). The
if-reconstruction pass exists (~135 LOC, exclusive-use fixpoint over
the region DAG, arms sink into real WGSL `if`), is **bitwise-equal**
to the flat translation on all three subjects including nested
selects and shared-subexpression hoisting — and is worth 0–1% on
every shape the splicer can currently produce, because pointwise
kernels are store-bandwidth bound (store floor 0.098 ms at 2048²;
90-op arms add 4 µs). So: keep select-normal-form for branches; don't
ship reconstruction until an ALU-bound consumer exists; the REAL
design question is **what a bounded loop with early exit looks like
in our IR and whether it can lower to a real `break`** — today a
`for` in a `@jit` body isn't flattened-and-slow, it drops to the
per-element oracle path (`_liftable` rejects nested regions except
`core.if`), which no device translates. That's where the 3× lives,
and the raymarcher is the canonical graphics workload that hits it.

Pass-relevant IR facts the spike surfaced: `args` is not dataflow
(`tl.iota` carries the lattice tensor it never reads — the op table,
not the arg list, defines dataflow for any pass); `walk_region` does
not descend into `n.regions` (moot while the splicer flattens, bites
when bounded control flow becomes translatable); reconstruction is
RECOVERY not inversion (it will branch on selects the user never
wrote as `if` — a policy choice); fragment-stage sinking needs a
uniformity check (implicit derivatives) that nothing enforces today.

Still open on this position: the masked-cotangent sharp edge under
the non-trapping policy (the 0·NaN pin, below).

## Constraints carried from the survey (the flaws that block clean design)

Named once, discharged by the design — not optional side-notes:

- `graphics.py` (reference semantics) owns the wgpu device singleton;
  the conformance executor imports it back. Dependency inversion.
- `upload()`/`sampler()` return wgpu Python objects registered in the
  global type table — a WebGPU type in the public API.
- Three expression-walker copies (`render.py:emit_dominated` unused;
  `_translate._expr` and `_Gen.expr` near-duplicates).
- `render()` vs `render_wgpu()` have different call shapes; no
  encodable exists though the spec names it as the deliverable.
- `WGPU_FP` declared as the executor column's cache-key value, never
  used — executor swap without rekeying.
- Silent f32 narrowing of i64 uniform slots and int consts (210 says
  narrowing is declared, never silent).
- `Untranslatable` → green-with-skips: coverage regressions are
  invisible.

Added by `spike_runner` (measured, not read):

- The graphics tier has no artifact and no lowering cache — every
  render re-runs AST lowering for both stages (0.937 ms, 7× a warm
  frame); `render_wgpu`'s apparent cheapness is the driver's shader
  cache (distinct source: 8–9 ms per pipeline).
- A compiled kernel artifact cannot be obtained without launching it
  (`_invoke` keys+compiles+launches; the artifact never escapes).
- The staging pack loop exists three times (graphics, wgsl_executor,
  kernel) and each backend invents the device layout the plan doesn't
  describe — one staging-plan object is the missing seam.
- Device paths *destroy* layout at the buffer boundary
  (`ascontiguousarray`/re-interleave): two views of one buffer become
  two unrelated device buffers. `residency.py` proves one resident
  buffer can serve compute field views AND the render record,
  bit-identically, with zero semantic change — blocked only by a
  missing encoding descriptor (f64 host / f32 device narrowing).
- `_device()` requests no features, so timestamp queries (210's honest
  timer) are unreachable without monkeypatching the singleton.
- `Tensor.to_numpy` is a per-element reference-tier materializer
  (`itertools.product` + `.item()`, tensor.py:184-200) sitting on the
  hot path of every device launch — 88–100% of real launch cost on
  both backends (`spike_metal` Part 3).
- The conformance kernel battery has no `where`/select subject, and
  `tanh` blows up (NaN) at |x|≥44.36 on Metal-compiled f32 — invisible
  only because the battery's subject runs at 5×7.
- MRT on device is broken, not just missing (`render_wgpu` on a
  `_BoundPSO` → AttributeError); the slot channel binds as storage,
  not uniform (210 contradiction; needs stride-16 plan emission);
  `_ARG_BINDINGS` is a process-global two in-flight PSOs would
  clobber; `_lower_fragment`'s resolution-shaped `varying_types` is a
  false dependency (dummy (2,2) → bit-identical output at every
  resolution).

## Workstreams

- **A. The charter** (design): one table over four tiers —
  assemblage / compute / vertex / fragment — every construct ruled.
  Draft table below.
- **B. Runtime/backend vocabulary + seam design** (design): interface
  sketches mapped onto existing seams; ends with the us-vs-L4 split
  proposal.
- **B (draft frame). Runtime vs backend — hypotheses under test.**
  The working definitions (owner): a RUNTIME owns devices,
  host↔device transfer, launch, streams/events, profiling,
  presentation; a BACKEND takes IR to executable code, which a
  runtime launches. Usually 1:1 per target — but under 270 (no magic
  compile; the program stays runnable at every point) the backend is
  not one `compile()`: it is a family of translators over regions,
  entered at different points of the transformation chain, all
  landing on the same launch contract. Hypothesized carve, to be
  tested by `spike_metal`'s seam ledger:
  - *Backend-owned:* source emission (op rows, literal spelling,
    binding declarations, workgroup attributes), narrowing per the
    declared type map.
  - *Runtime-owned:* device acquisition + feature requests, buffer
    creation/adoption (incl. unified-memory zero-copy), queue/submit,
    sync, readback, timestamp/event profiling, the canvas/swap chain.
  - *Shared, target-neutral (the seam itself):* the region walk, the
    marker tables, the staging/uniform PLAN (one object both packer
    and backend read — H3), launch geometry arithmetic, the artifact
    +cache key discipline (executor fp actually keyed — the `WGPU_FP`
    bug), residency registry (compile-time input), and the
    encodable/launch contract (`(values, staging) -> effect`;
    `bind()`/`encode(pass)` on the render side).
  Known seams already in the tree that the design should grow rather
  than replace: the kernel executor column, `pdum.dsl.registry.
  Backend` (value tier only today), `Buffer.device` + DLPack intent,
  `abi.slot` physical staging.

  **Spike verdict (`spike_metal`, DONE): the carve held physically and
  the executor column IS the seam** — the entire test-side diff
  between "run on WebGPU" and "run on Metal" is ONE IDENTIFIER;
  staging, uniform capture, splicing, the overlap refusal, and
  writeback reuse `kernel.py:1082-1113` unmodified; WGSL and MSL
  outputs are bitwise-equal on all 10 translatable subjects (caveat:
  both columns reach the same Metal compiler on this machine — the
  first honest cross-vendor test is CUDA/Vulkan). 100% of emitted
  statements convert under three lexical rules; no operator row
  differs. But the boundary is **a negotiated contract, not a line**
  — five carve failures, each with evidence in the spike:
  1. Workgroup size is backend-owned in WGSL (shader text) and
     runtime-owned in Metal (a dispatch argument) — 210's rule is
     WebGPU-shaped. A backend returning only `str` cannot express
     Metal; it must return (source, launch-contract).
     **RULED (owner, in-session): THREAD SIZING (threads per group)
     is a SPECIALIZATION parameter** — it keys the artifact and
     triggers recompilation when it changes, whether or not the
     target embeds it in source text (WGSL: source; Metal: a
     `dispatchThreadgroups:` argument; CUDA: blockDim, where launch
     bounds make it compile-relevant too). **BLOCK SIZING (number of
     groups) is launcher data everywhere and never specializes.**
     The launch contract carries the thread size as data; where it
     lands — source text or dispatch call — is a per-target detail.
  2. The bounds guard is emitted source whose NECESSITY is a runtime
     choice (whole-workgroup dispatch vs `dispatchThreads:` exact
     grids — guard-vs-exact proven bitwise equal).
  3. The binding table is co-owned: `layout="auto"` pruning is a
     runtime rule encoded in backend code; Metal has no bind-group
     object at all. Bindings must be returned as DATA.
  4. The staging plan describes host bytes only — the DEVICE
     representation belongs to neither box, so every backend invents
     it (three inventions now) and both silently narrow i64→f32.
     The hole is in the SHARED tier.
  5. Executor identity never enters the content key: WGSL and MSL
     artifacts share an identical `region.key` — they would collide
     if keyed today, and un-keyed they recompile every call. Backend
     identity must enter the key before any second backend ships.
  Plus a category the definitions don't have: **the target numeric
  contract** (FAIL-0). Metal's `tanh` is NaN for |x|≥44.36 (exp-based;
  `MTLMathModeSafe` doesn't help); a free clamp row fixes it,
  bitwise-identical where both are finite — but no box owns per-target
  math-library deviations. 210 has the section; the mechanism is
  missing. Also: unified-memory adoption is real (`MTLBuffer` over
  page-aligned numpy; kernel stores visible with zero
  readback), 5.5–10.8× over the wgpu protocol on raw benchmarks — and
  through the real `launch()` it deflates to noise because
  `Tensor.to_numpy` (a per-element reference-tier materializer) makes
  the host repack 88–100% of every device launch. Residency (H4) is
  not just a render-loop concern; it is the whole launch cost.

  Proposed user-facing spelling (per no-magic and the bracket
  precedent; `wgpu_artifact`/`metal_artifact` conversion functions
  stay test-side — that shape is the "magic compile" 270 refuses):

  ```python
  from pdum.tl.runtime import metal, webgpu
  kernel[on(metal)](src, dst)            # per-launch selection
  art = kernel.artifact(src, dst)        # the missing public door (H2)
  art.on(metal).launch((src, dst))       # 270's incremental chain
  ```

  `on(...)` keys the artifact cache (`region.key`, backend fp, runtime
  fp); it names a backend+runtime PAIR (separable in principle — the
  same MSL serves a metal-cpp harness; the same runtime launches a
  precompiled `.metallib`); and it never implies a transfer — under
  adoption a tensor's memory is simultaneously host and device, which
  is the moment `Buffer.device`-as-string needs a real registry.

- **C. The render loop** (design + spike `spike_runner`, DONE): the
  encodable + a frame runner owning device/pass/submit/canvas per the
  210 doctrine (shared encode path, timestamp queries, no
  sync-readback timing). Spike verdict: the factoring works — one
  pipeline served 8 angles and a 4× attachment, bit-identical to
  `render_wgpu`, warm frame 0.135 ms vs 5.66 ms today. The encodable's
  shape, learned by building it: `compile_pso(pso, inputs, args,
  target, residency)` once; per frame `bind(**slots)` (queue-touching)
  strictly separate from `encode(pass)` (encoder-touching; the pass is
  begun by the HOST — encode-into-encoder was tried and is wrong: it
  forbids two PSOs per pass and pulls frame-level clear/load decisions
  into the per-draw object). The artifact key is (pso, input layouts,
  target format) — target format keys (the 1→4 channel expansion is
  baked into WGSL), resolution must NOT key (proven false dependency).
  Residency is a compile-time input, not an encode-time one — strided
  binding indices are baked into the shader; that is the zero-copy
  mechanism, and it needs only an encoding descriptor (f64→f32
  elementwise narrowing preserves indices), not L2's full machinery.
  **Corrections from the dual-backend windowed demo**
  (`explorations/graphics/demo/FINDINGS.md`, which ran this shape
  against a second, structurally different API): (1) `encode(pass)`
  is NOT the portable primitive — Metal needs the full pass
  descriptor (attachment/load/clear) BEFORE an encoder exists, so the
  primitive is `encode(command_buffer_or_encoder, target, clear=...)`;
  target and load-op are inseparable from pass creation. (2) The
  binding table is PER-STAGE data — Metal's vertex and fragment
  stages index two separate buffer tables (the same logical resource
  gets two indices), so one shared index space is a WebGPU-shaped
  assumption. (3) Presentation needs a seat the sketch lacks:
  in-flight synchronization (semaphore + rotating slot buffers) — on
  unified memory a slot refresh IS a host-memory write and an
  in-flight frame can see torn uniforms. (4) Render-stage rows are
  100% WGSL→MSL portable under ONE added lexical rule (vector type
  spelling — four rules total), so the shared-emitter design extends
  to render unchanged; the remaining diff is shell (per-stage
  bindings, varying-struct spelling, stage qualifiers, and the
  surface-expansion axis — confirmed on both targets). (5) The
  capture fingerprint discriminates `float` from `np.float64` — a
  numpy-derived uniform silently keys a different artifact and
  recompiles every frame with no warning; the closure-swap/warmth
  guard should graduate to library-tier in the cleanup.
  **(6) OWNER CORRECTION (2026-07-29): `bind(name=value)` is NOT the
  user surface** — that spelling is other frameworks' set-uniform
  ceremony, exactly what the capture doctrine exists to end. The
  per-frame USER act is rebinding the captured environment (an
  ordinary Python assignment to the global a body reads, or minting
  the fresh closure) and handing the program the new function object;
  the machinery re-extracts captures through the existing abi.slot
  plan and repacks — names never appear at the call site. The demo
  already spells it right (`engine.update(bind_mouse(mouse))`); the
  spike's `bind(**slots)` survives only as the INTERNAL seam name for
  the queue/memory-touching half (update ≠ encode), never as API.
- **D. The Metal twin** (design + spike `spike_metal`, DONE): MSL
  emission + a PyObjC Metal runtime; the seam's existence proof
  delivered (see B's verdict above). ~40 lines of glue; zero-copy
  unified-memory adoption proven; JIT MSL needs no Xcode
  (`newLibraryWithSource:`).

## A. The kernel syntax charter — draft table

Status per cell: **IN** (admitted), **OUT** (should refuse with the
ruled voice), **out\*** (effectively out today but by ACCIDENT —
incidental failure or silent acceptance, needs the ruled refusal),
**door** (a recorded future opening). "Today" verified by
`explorations/graphics/probe_kernel_bleed.py` where marked (†).

| construct family | assemblage/step | @compute | @vertex | @fragment |
|---|---|---|---|---|
| ambient (`thread_idx`/`global_idx` over the kind's lattice) | n/a | IN | IN (`vertex_id`) | IN (pixel = fwidth's wrt) |
| indexed tensor read (`t[i]`, ambient or computed-i32) | host subscript law | IN | IN (record buffer by ONE Coordinate) | via varyings only |
| indexed store `t[i] = v` (bijective; never scatter — ruled P9) | n/a | IN | OUT (return `position`) | OUT (return color0; claims are taps) |
| scalar expressions (markers, arithmetic, `f32`/`i32` doors) | IN | IN | IN | IN |
| records (construct, fields by name) | IN | IN | IN (buffers, varyings) | IN (taps) |
| uniforms as captures (the literal doctrine) | IN | IN | IN | IN |
| layout ops (flip/shift/pad/select/rename/… ×18) | IN (their home) | **out\*** † — `flip` lowers AND runs; shift/pad/select fail on incidental grounds (containment/alignment), not a tier rule | out\* | out\* |
| reductions (`red.*`) | IN | OUT † (ruled voice exists: "a host citizen here"); K-G reduce-by-index is the door | OUT | OUT |
| fold / scan / take / scatter_add | IN | OUT (ruled) | OUT | OUT |
| statement `if` / `IfExp` / `BoolOp` in bodies | OUT † (straight-line, frozen msgs) | OUT † | IN — `IfExp`/`BoolOp` as `where`/max/min (committed S.4 spelling) | OUT (inherits kernel rules) |
| `if` in jit device fns (value tier) | IN → `core.if`, spliced to select | IN via splice (pure arms) | IN via splice | IN via splice |
| `while` / data-dependent trip counts | OUT | OUT (the L4 door — tiling language) | OUT | OUT |
| `for range(k)` | OUT (unit-tier unroll is a registry door) | IN (compile-time unroll only) | IN (same) | IN (same) |
| device-fn args (fn-valued params) | IN | IN via splice; oracle-class = reference-only (named edge) | IN | IN (same edge) |
| `sample` (textures) | OUT | OUT (recorded future work) | OUT | IN (v1 narrow) |

**The charter's three proposed rulings** (to be owner-ratified):

1. **The kernel face is: indexed reads, indexed stores, scalar
   expressions.** Everything tensor-shaped (layout, reductions, fold,
   the indexing family) is a host act. The out\* cells become REAL
   refusals in the reductions' voice ("layout is a host citizen here —
   apply it at the call site, pass the view in") — today `flip`'s
   silent acceptance and `shift`'s incidental containment error are
   two different accidents where one ruled refusal belongs.
2. **Admission ⊆ translatability.** What a tier admits should be what
   a device can carry (or be explicitly marked reference-only, like
   the oracle class). A construct that lowers on reference but is
   `Untranslatable` on device — kernel-body `flip` today — widens the
   reference/device gap silently: user code "works" until it meets
   hardware. The conformance battery's honest-skip is not enough; the
   tier boundary itself should refuse.
3. **Select-normal-form is the law; branch recovery is a backend
   peephole** (the control-flow position above). The vertex tier's
   `IfExp` spelling and the kernel tier's refusal are today
   INCONSISTENT surfaces over the same law — unify per the
   statement-`if` open question below.

## Rulings (owner, 2026-07-28, via the 282 comment pass)

- **Kernel rulebook (282 §1): ratified as proposed — and the
  tier-stratification implementation is CLAIMED BY THE L4 TEAM**
  (their PR #7 comment); we consume it after their merge + our rebase.
- **Control flow (282 §2): mask law and expression-`if`-everywhere
  ratified; part (iii) AMENDED by the owner** — statement `if` in
  kernel bodies is admitted when its arms are store-free (statically
  adjudicable at lowering: a store is an assignment to a buffer
  subscript; arms' locals join via select). Only an `if` containing a
  store refuses, speaking the law: effects may not branch. Lands with
  the L4 stratification.
- **Depth (282 §3): IN for v2**, implemented with the encodable.
- **Code home (282 §4): the third workspace package** (runtime +
  emitters; name proposed at implementation).
- **Bounded loop (282 §5): option (a) — handed WHOLE to the L4 team**
  with the spike as exhibit; our demos avoid raymarching until it
  lands (supersedes the "we design the surface" leaning).
- **Coverage ledger (282 §6): (b) ratified** — the
  not-yet-translatable set becomes a versioned, machine-checked
  ledger; never-legal constructs refuse via the stratification.
- **Selection spelling (282 §7): explicit dataclasses** — a generator
  (backend) dataclass and a runtime dataclass (empty for now), a pair
  dataclass holding both; user-facing names are pre-assembled
  instances.
- **`Buffer.device` (282 §8) and frame-sweep tolerance (282 §9):
  directions agreed** as proposed. **282 §10–12: postponements
  agreed.**
- **Thread-sizing supersession (PR #7, resolved on-thread):** the
  ruling OVERRULES the kernel tier's "geometry is validated launcher
  data, never identity" policy for the THREADS half; block sizing
  stays launcher data. kernel.py's policy comments and the frozen
  geometry refusal get repinned deliberately in the same change.

## Open questions (for the owner)

- Where the runtime/runner tier lives (`runtime/`? — "never
  `backends/`" was ruled for the conformance executor, so a new home
  is an owner decision).
- Whether depth buffering enters graphics v2 (a reference-semantics
  change, not a device feature).
- Whether statement-`if` becomes admissible surface syntax in kernel
  bodies as sugar for the select form (the vertex tier's
  `IfExp`/`BoolOp` spellings already are), or stays refused to keep
  bodies visually straight-line.
- The masked-cotangent (0·NaN) pin: a day-one test wherever gradients
  cross `where` under the non-trapping float policy.
- Frame-sweep conformance tolerance: the cylinder golden's atol=2e-3
  is calibrated to its one angle — sweeping angles puts 2/6144 pixels
  at 5.1e-3 (steepest AA-ramp pixels; `render_wgpu` misses the same
  pixels identically). A loop battery needs a scene-independent
  tolerance statement or a ramp-pixel exclusion.
- **The bounded-loop door** (the spike-sharpened L4 question): what a
  fixed-bound loop with early exit looks like in the IR — a
  `core.for` with a declared exit predicate? a `tl.march`-style
  primitive? — and whether its reference semantics (run to the bound,
  mask after exit) can keep the AD/analysis story while lowering to a
  real `break`. Worth 2.2–4.0× on the canonical graphics workload;
  belongs in the K-A…K-G conversation with this spike as its exhibit.
- If-reconstruction policy, when it ships: branch only on selects
  recovered from user-written `if`, or on any select with exclusive
  arms? (Recovery cannot tell them apart; a marker-lowered `where`
  would branch too.)
- **The target numeric contract's mechanism**: where per-target
  math-library deviations live (Metal's exp-based `tanh`; `exp`,
  `sinh/cosh`, `pow` are the usual company) — declared rows per
  backend? part of the descent-license schema? 210 states the policy
  ("differentials state their tolerances") but nothing owns the fix
  rows or their bitwise-equivalence proofs.
- Whether `on(...)` names a backend+runtime PAIR or one fused thing —
  the spike shows they're separable; the spelling shouldn't bake in
  1:1 before a consumer forces the question.
- `Buffer.device` under adoption: unified memory makes "which device
  is this on" a question the string label cannot answer (the memory is
  simultaneously host and device). What replaces it — a residency set?
  an epoch/ownership record (the L2 handshake arriving early)?
