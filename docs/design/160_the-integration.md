# 160 — The integration: one workspace, tensorlib as the assemblage language

**Status: PROPOSED — for owner review before any code changes.** Produced by a planning workflow from the owner's post-150 decisions. Prior canon: 010–150; 150 is the critical-assessment report this document acts on, 140 its charter. Tensorlib lives at `explorations/tensorlib/` until M3 below. Nothing in this document is executed until the owner ratifies it; §7 lists the decisions only the owner can make.

---

## 1. The decisions

Six decisions from the post-150 walkthrough, recorded as canon. Each names what it reverses.

**D1 — Tensorlib syntax IS the assemblage syntax.** Its layout algebra (affine map + box + guards + charts + units + placement) and four compute primitives (pointwise/reduce/scan/fold) are adopted wholesale — they are far more developed than any array machinery in pdum.dsl. Its caching mechanism, `Build` name-manager, process-level registries (`COMPOSITE_MARKERS`/`COMPOSITE_REDUCERS`, mdsl.py:240-241), and the mdsl traced-lambda frontend are NOT adopted — they were a weak reimplementation of what pdum.dsl does properly. No pdum-invented tensor verbs, no translation frontend: assemblage programs are written in tensorlib language directly; host Python provides control flow and composition; at most a light lazy-compile annotation at top-level entry points (§4 S.1). **Reverses:** 130 stages 2–3 (pdum tensor dialect, comprehension frontend, second AD) — cancelled; the 150 direction memo's installments 1–2 as written, including the surface→Program emission-seam artifact the memo ranked "single highest-priority" (150:1464-1478) — D1 goes past the memo: there is no frontend to seam; and tensorlib's own frontend stack (README.md:188-250, "The `Build` name-manager (not a frontend)").

**D2 — The repo becomes a uv workspace of logically-organized packages; mutual cannibalization.** Neither codebase survives as "the core piece": (a) a core-infrastructure package extracted from pdum.dsl kernel machinery — two-tier type-keyed caching, type dispatch (typeof/ValueKind/KindTable), name resolution and name-fate analysis, AST/reflection capture, control-flow (straight-line) detection; pipeline machinery folds in or stands beside it (undecided, §2b combinators row). (b) Tensorlib converted onto that core and promoted out of explorations/. (c) The zoo becomes part of the tensor package and MUST keep working — zoo programs + numpy-pinned denotations are the acceptance gate (§2d). "As-is" means the denotational and naming contract, not source text: every builder takes `b: Build` as first parameter today (zoo_common.py:52-107), and Build dies, so re-authoring is forced; the gate substitutes bit-equal denotations plus name identity for textual identity (amendment from review, recorded so no future reader takes M6 as violating D2c). (d) pdum.dsl stdlib DIES — Named/arrays.py, transforms.py `over`/`jvp` as pdum concepts, batteries' array half; possible residue: in-kernel ambient derivatives (fwidth) at the shader tier. This moots the 150 repair rulings on `over`/jvp/`D`/matmul/Pipeline (HV §2.4c-e, 150:244-252; F3/F12/F22/F31/F36) — their principles survive by construction (binding is representation-level `bind`/`Dim.level`; contraction structure is never erased; frozen-primitives/derived-composites is the only AD polarity left). (e) The scalar statement language (if/for/statement lowering, base_lang.py) SURVIVES as the device-function syntax. (f) The remainder archives, dsl_reference-style. **Reverses:** 040/050-era stdlib canon; F11/F15's flagship loop gate (dies with the stdlib; the no-extent-loops principle is absorbed structurally).

**D3 — The syntax stack** (§4): (1) tensorlib assemblage at top — straight-line, host Python composes; (2) ONE shared Python-expression syntax for scalar functions AND straight-line tensor fold steps, typed lifting; replaces the mdsl markers and is the fold-step authoring story; the invariant is straight-line/no-branching, not pointwise-only; (3) @compute kernels: `thread_idx(...)` ambient intrinsic (NOT positional params), explicit stores into argument buffers, launch config at invocation, function-valued arguments with FnType-in-key semantics; (4) tile DSL (stage/barrier + capacity/race WF certificates); (5) warp DSL (straight-line lane intrinsics); (6) vendor escapes (090 punning) + external oracles (raw CUDA C / Numba as test fixtures) — no CUDA-clone language. Vocabularies + WF predicates over ONE frontend machine, never separate grammars. Iota unification: ambient coordinates ARE the launch-domain iotas; the descent rewrite iota→thread_idx never materializes them. Straight-line symmetry: branchless at top and bottom; control flow confined to scalar kernels and the host. Invocation concerns (blocks, shared memory, streams, pipelining) never appear in user programs — they become visible only in transformation steps. **Reverses:** 090 §3's "params ARE thread coordinates" core-profile clause (c.py:290-299, wgsl.py:63-72) and F4/F25/F37's per-family coordinate divergence at the root; 040's one-`|` pipeline (`|` survives only as fuse-inline device-function composition).

**D4 — Async is DROPPED.** Explicitly rejected by the owner. **Reverses:** the 090 §5 async-readback path; any async pipeline variant. Readback stays explicit and synchronous.

**D5 — The objective:** this integration clears the runway for tensorlib L4 (the kernel language, LEVELS.md ladder) and L2 (bufferization). Integration first, done carefully; L4 and L2 proceed immediately after. Tensorlib's K-F ordering holds: bufferization consumes kernel boundaries (LEVELS.md:216-218). The 150 L4 design brief (K-A..K-F answers, 150:1501-1517) carries intact into §8.

**D6 — Precision (REVISED mid-planning; the retraction is on record, §3).** Precision FACTS at the boundary, precision CHOICES in the interior, CARRIER semantics throughout. Mid-planning the owner retracted an earlier carrier-only-surface stance: loaded weights *dictate* encodings — a bf16 checkpoint is a fact the program cannot not-know — so precision cannot vanish from the user's world entirely. **Reverses:** tensorlib's status quo (carrier + dtype both user-facing on `Tensor`, tensor.py:42-48); 130's GEMM `astype(f16)` (stays convicted, 150:303); the owner's own interim carrier-only stance (retracted); and demotes the 150 two-surface model (A1 + breadth flag 3, 150:1425) to recorded fallback with a written criterion (§3e).

These decisions resolve 150 arbitration item f.1 at the root; the full f.1–f.10 disposition is §6. The 150 installment-0 silent-corruption fixes (F6/F23 alias refusal — the report's gravest verified flaw, execution-verified twice; F29) remain valid regardless and are sequenced first (M0).

---

## 2. Target architecture

### 2a. Workspace layout — Option A (root-preserving incremental)

The workspace stanza already exists with an empty glob, designed so adding a member "is a pure add" (pyproject.toml:45-50); members auto-discover under `packages/*`, root deps on members go through `[tool.uv.sources] { workspace = true }` (pyproject.toml:52-53). Option A exploits this with zero release churn: `_versioning.py` hardcodes `INIT_PY = src/pdum/dsl/__init__.py` (_versioning.py:32) and release.yml builds only the root dist (release.yml:137-149), so new members stay **unpublished workspace-internal packages** until the owner deliberately ports the deliberately-absent constraint-repin machinery (_versioning.py docstring, lines 15-22) and extends release.yml — owner-only under AGENTS.md:14-33.

- **`packages/core` → `pdum.core`** (no `pdum/__init__.py` — the PEP 420 namespace rule, pyproject.toml:40-43, replicated in every member). Contents: the kernel engine verbatim — types/valuekind/capture/cache/registry/ir/ops/printer/rewrite/lower/pack/derived/events/api — plus `pdum.core.render` (backends/_emit.py) and `pdum.core.scalar` (base_lang rule pack + surfaces + surviving batteries; D2e). The `kernel/__init__.py` BUILTINS pack-aspect registration (kernel/__init__.py:23-28) survives the move or `extend()` mints tables that cannot marshal. Depends on stdlib-Python only.
- **`packages/tensor` → `pdum.tensor`**: tensorlib promoted and converted — layout stack unchanged; IR/compute/autodiff/transforms/memory/placement/signatures/opcount repointed onto core naming/caching/capture; `pdum.tensor.zoo` inside it (zoo already lives at tensorlib/zoo; in-package avoids an inter-member dep needing the absent repin machinery).
- **`pdum.core.backends`** (subpackage, not a member yet): c.py minus the GRID family, its one dying import (`from ..stdlib.transforms import Over`, c.py:290-291) severed at extraction. Vendor backends become members later per 090's rule of three.
- **Root dist `habemus-papadum-dsl`**: shrinks to `pdum.dsl_reference` (frozen, unchanged) + the D2f archive of today's pdum.dsl remainder + docs/book. Root's dependence on members for book execution goes in `[dependency-groups]`, **not** `[project.dependencies]` — a published root wheel naming unpublished members would be uninstallable from PyPI (amendment from review; clearly right, adopted).

Kernel extraction is import-clean: `kernel/` imports no satellite (registry.py:7's law holds by grep); the only executable kernel-adjacent coupling to a dying module is c.py:290-291.

### 2b. Disposition table

**pdum.dsl side** (paths relative to `src/pdum/dsl/`):

| Module | Disposition | Destination / re-homed content |
|---|---|---|
| kernel/{types,valuekind,capture,cache,derived,events,api,printer,ops,rewrite,lower}.py | CORE | `pdum.core` verbatim. `Vec` (types.py:58) migrates to the shader tier later; `Array` kept until tensorlib's tensor type replaces it |
| kernel/registry.py | CORE, two edits | Fix F33 unknown-kind silent default-routing (`backend_for` falls through to `default_backend`, registry.py:121-124) at extraction — never port the hole. `Backend.param_types` retires **at M8, not extraction** (amended per review: the GRID family and demos consume it until @compute replaces them, c.py:290-299, wgsl.py:63-72) |
| kernel/ir.py | CORE, one edit | Relax the bodies-end-in-yield verify rule (ir.py:163-166) for @compute store bodies — prepared at M2, exercised at M8 |
| kernel/pack.py | CORE | Reword the ndarray/ch12-naming refusal texts (pack.py:288-292) |
| backends/_emit.py | CORE | `pdum.core.render` |
| backends/c.py | SURVIVES, **scheduled** | Moves at M2 with GRID intact and the Over import severed (duck-typed locally); GRID retires at M8. Keep: trunc div/mod policy (c.py:57,136-137), u64/inf-nan constant refusals (c.py:127-135), artifact-carries-contract header (c.py:259-264). (Amendment: prior draft left this move unscheduled; review caught the M6–M8 GRID import-crash window) |
| stdlib/base_lang.py | SURVIVES | `pdum.core.scalar` — pure rule-pack data (LOWER_RULES); inherits F14's designed-refusals obligation |
| stdlib/surfaces.py | SURVIVES | Beside core; `_invalidate` both-tier bump (surfaces.py:30-37) is load-bearing |
| stdlib/batteries.py | SURVIVES | Device-function stdlib under 090 §2 minimalism; hardcoded `_PY`/`_WGSL` names (batteries.py:28-29) die with the demos; wiring moves to entry points (registry.py:216-224) |
| stdlib/arrays.py | DIES | Re-home: numpy ValueKind adapters + contiguity/dtype refusals (arrays.py:91-101) → tensor boundary capture; Shaped-drift value check (arrays.py:190-199); erase-refinement-before-emit so named/positional twins share content keys (arrays.py:275-281) → tensorlib named dims; `array.buffer/dim/load` op pattern → tensor buffer ABI template; isel refusal texts |
| stdlib/transforms.py | DIES | Re-home: `_Tangents` + `D`/ctx.root seam parked for the fwidth residue; `JVP_RULES` rows merge into the one derivative table (§4 S.2); `woven_hits` refusal as error-quality benchmark. matmul fully superseded; its frozen 3.0 extent pin (test_refusal_contract.py:116-137) is **deleted, never ported**, with the tensorlib extent refusal pinned in its place (f.2) |
| combinators.py | SURVIVES partially | `Stage`/`Pipeline` + `build_pipe` fusion-by-inlining (combinators.py:180-204, 254-278) are exactly D3.3's `twill\|weave\|zoom` need. The module globals `_ROLES/_RULES/_DISPATCHER` — process-level registries of exactly the kind D1 rejects in tensorlib — fold into Registry **at M2** (their own docstring calls them a staged seed, combinators.py:22-24) |
| viz.py, bench.py, events.py (recorder) | SURVIVE | Import re-pointing scheduled (viz imports cache internals, viz.py:24-30) — rides M8 with the shim deletion |
| demo/ | ARCHIVE at M8 | wgsl.py's runtime (device mgmt, ComputeProgram tiering, timestamp queries, uniform-plan, wgsl.py:172-321) is the **seed of the real @compute wgsl backend** — promoted, not stripped; graphics.py's ddx/ddy/fwidth (graphics.py:45-55) is the residue shape |
| dsl_reference/ | UNCHANGED | Frozen, runnable, shipped |

**tensorlib side** (relative to `explorations/tensorlib/tensorlib/`):

| Module | Disposition | Notes |
|---|---|---|
| layout.py, guarded.py, chart.py, units.py | SURVIVE unchanged | Pure data + algebra; zero imports from Build/mdsl |
| buffer.py | SURVIVES | Already boundary-shaped: `read(loc, dtype)` is the one seam where dtype meets values (buffer.py:57-61) |
| tensor.py | SURVIVES, D6 re-roled | Retreats to the boundary as the adoption descriptor (§3b); `field()` becomes a descriptor act |
| dtypes.py | SPLITS | `CARRIERS` stays semantic; encodings → boundary descriptors; `round_to(encoding)` added |
| ir.py, compute.py, signatures.py, opcount.py | SURVIVE | Composite lookup repoints from process dicts to core-cache-backed registries; name→definition resolution unchanged (ir.py:50-64); `const/iota` dtype params become carrier params; alignment gate unchanged (compute.py:167-170) |
| autodiff.py, transforms.py | SURVIVE | AD semantics intact. The two sibling name managers — `autodiff._Builder` (autodiff.py:92-116) and `.rc` suffixing (transforms.py:30-33) — are reconciled explicitly at M4: fold into core naming or consciously keep internal; either way the derived-name conventions (`name.d{i}`, `.rc`, `%hint`) are pinned as contracts, since programs and grad maps reference them |
| memory.py, placement.py | SURVIVE, D6 edit | `_ITEM = 8` (memory.py:37, placement.py:33) → descriptor+annotation-fed dtype-exact sizes — an L4 precondition (150 A3; CONCERNS #25) |
| build.py | DIES → **frozen fixture** | Replaced by D3.2 + core naming. Replacement MUST reproduce: hint dedup (build.py:19-28), dotted weight prefixes, nested step-program construction, `input`-name identity. Build itself is archived importable in `packages/tensor/tests/fixtures` (dsl_reference in miniature, ~60 lines) so the M6 builder-identity pin runs live-vs-frozen — Program serialization does not exist (CONCERNS #22) and a deleted Build would leave the gate uncheckable (amendment from review; adopted) |
| mdsl.py | DIES IN PARTS | Dies: the Sym tracer + defmarker/defreducer entry points (mdsl.py:84-166, 382-425; archived as fixture like Build) and the process registries + `node_digest` (mdsl.py:240-261). **Survives relocated: the `Arg/Const/Prim` Node schema** (mdsl.py:50-72) — five consumers walk it (compute.py:144-152, signatures.py:128-135, opcount.py:46-52, autodiff's Jacobians, reducer fields); mdsl.py:1-16 designed it as the stability boundary with pluggable producers, and D3.2 is a new *producer*, never a consumer rewrite. Survives: symbolic `diff`/`_D` (mdsl.py:175-233) and the CompositeReducer BPTT machinery (mdsl.py:300-379) — AD semantics, not frontend |
| zoo/* | SURVIVES | `pdum.tensor.zoo`; builders re-authored, denotation-identical (§2d) |
| notebooks + root .md docs | SURVIVE, governed | Enter nav under a **pre-integration snapshot banner** until revised — M3 must not publish canon that M5–M7 then contradict (DESIGN.md:55 "Tensor = Buffer + Layout + DType" is reversed by §3; README.md:189 describes Build, deleted at M6). Doc-revision line items ride M5/M6/M7 (amendment from review; adopted) |

### 2c. Caching and identity across tiers

One mechanism (`pdum.core.cache`), three keyspaces.

**Assemblage tier (new client).** Building a Program is the compile step. Tier 1 keys `(fp_head, arg_fp, backend_fp, generation)` with `fp_head = ("H", code, env_fp)` (capture.py:96). Tier 2 is content-addressed on the built Program in canonical form, replacing `node_digest` + idempotent `_register_marker` wholesale, and supports **derivation-under-cache**: partials, component markers, adjoint scanners are cache entries computed on demand from cache entries (mdsl.py:298-343). The CONCERNS #22 items deferred "until pdum.dsl integration" (registries-as-libraries, reducer content addressing, fresh-process deserialization, CONCERNS.md:185-199) become ArtifactCache clients — this integration is their deadline. F17's naming law becomes a contract of core name resolution: `grad` returns a name-keyed map (autodiff.py:867-876) and zoo tests index `"L0.wq"` (test_zoo.py:53,62), so names are part of the cached artifact's ABI, pinned by a rebuild-stability test. **Open soundness caveat (§7 O1, high):** "types and identity only, never captured values" is unsound for builders whose captured scalars are *structural* — extents baked into slice/pad/fold params (physics.py:36-42; megatron.py:38-40) fingerprint identically across values under int range-bucketing (valuekind.py soundness law), so same-code-different-N would silently reuse the wrong Program. The owner must pick a rule before M6; §7 O1 gives the options.

**Kernel tier (unchanged thesis).** The two-tier scheme survives verbatim, gated on the existing pins: out/domain never in identity (test_grid.py:38-39); rank-generic warm hits; live-knob zero-recompile; fresh-closure idiom + mutated-capture re-extract (load-bearing, currently unpinned — pinned at M2); cross-Handle guard re-key. D3.3 function-valued arguments are F8/A4 applied here: the argument Handle's FnType (incl. capture types) enters `arg_fp` — different pipeline shape = new artifact; same shape, different captured values = hit + uniform rewrite (mechanism present: capture.py:128-138; pack leaf-walking of FnType args, pack.py:88,180-188).

**Descent tier (per the 150 L4 brief).** `chunk_fp` on the named-op Program in canonical form (F21 — trivially satisfiable since contraction structure is never erased). Registry key = (normalized chunk skeleton, boundary contract incl. saved-set demand + layout classes, license set, capability set, rules-generation) (150:1517); value = chain + authored region + artifact + assurance tier; chain mandatory and content-addressed (F7); trust at tier ≥2 only (A8). **Honesty note (amendment, adopted):** both "canonical form" keys presuppose a Program-normalization pass that exists nowhere (`canonical()` is layout-only; 150:1721, AV A5). Until it lands, both tiers are honest **private caches** — per-project hits only; the normalization pass is a named L4-runway prerequisite of the registry's cross-model payoff (§8).

### 2d. The zoo-compat gate

The gate = the existing test files, moved into CI (they run **zero times** today — no `explorations/` reference exists in .github/, scripts/, or pyproject; the conftest is a `sys.path.insert` hack, explorations/tensorlib/conftest.py:1-4) and kept green through conversion, plus two new pins:

1. Forward denotations vs numpy at rtol 1e-9 / atol 1e-12 (test_zoo.py:31-36).
2. Gradients vs finite differences, indexed by input names `"x"`, `"L0.wq"` (test_zoo.py:51-63) — name identity is load-bearing; numeric tests alone cannot catch grad-map key drift.
3. flash == naive: forward rtol 1e-9 AND derived backward rtol 1e-6 — composite-reducer BPTT, no hand rule (test_zoo.py:66-79).
4. FDTD gradients carry staggered charts (test_zoo.py:82-91).
5. Placement erasure bit-exact: megatron placed vs `level=None` at rtol 0/atol 0; exactly two gpu all-reduces of 192 bytes; erased program communicates nothing (test_placement.py:48-68).
6. Cost oracles stable (test_memory/test_transforms/test_opcount), modulo the deliberate D6 size change landing as its own re-derived diff.
7. **NEW — builder-identity pin:** each re-authored builder runs beside its frozen Build-authored predecessor (the archived fixture, §2b) under `ir.run` on identical inputs; outputs bit-equal; `m.inputs`/`m.out`/grad-map keys identical. Interior hint-derived names may drift only if `grad`'s mapping stays coherent.
8. **NEW — refusal transfer:** the tensorlib shared-axis extent refusal pinned in the joint battery in place of the deleted stdlib 3.0 pin (f.2).

Any conversion step that cannot keep 1–6 green is wrong (150:1527: the reference layer is never a quarry to strip).

### 2e. Calling-convention matrix skeleton

Rows = caller, columns = callee. **I** inline, **L** launch, **C** compose, **R** refuse. Graphics rows deliberately absent pending f.8 — recorded deferral, not silence. Host-calling any Handle scalar-wise remains available as debug/oracle only (150:274 n.1, now the rule).

| caller ↓ \ callee → | Host | Device fn | @compute | Tile | Warp | Assemblage |
|---|---|---|---|---|---|---|
| **Host Python** | — | L (oracle) | **L**: launch config at invocation; day-one writable-arg alias refusal; fn-valued args FnType-keyed | L (bench only) | R | L(many); optional `@assemblage` |
| **Device fn** | R | I | I(body), kind-checked | R | R | R |
| **@compute** | R | I (incl. passed-in fns) | I(body), kind-checked | R | I (lane intrinsics) | R |
| **Tile** | R | I (epilogues) | I(body) | I | I (mma fragments) | R |
| **Warp** | R | I (scalar only) | R | R | I | R |
| **Assemblage** | — | C(marker): declared, never a callback | L(select) via certified lowerings | L(select) | R | C: fold takes a Program |

Invariants from 150:258-287: launches from non-host callers refuse (no dynamic parallelism); kind validated at dispatch AND cross-family inline — a body using the `thread_idx` ambient inlined into a non-kernel context refuses (V4 made real; the inline check lands at M8, the dispatch check at M2); the four composition semantics never share syntax (`|` fuse-inline; sequencing = host + fold; PSO rides f.8; rewrite chains get their own form). Two recorded deviations from the 150 matrix (amendment, adopted): Tile→@compute is I(body) here vs 150:270's R — justified iff M8's kind check maps the ambient contract into the tile context, else it reverts to R; and the n-ary/pytree composition slot dissolves into host-Python assemblage composition now that its satellite dies with the stdlib.

### 2f. 090 punning and the events seam

090 carries over nearly whole: vendor namespaces, capability flags at build, `code_for_op` key-presence as the capability bit, capability-gated `debug.print`, `record.artifact`, rule-of-three for runtime abstraction. Three amendments: (i) 090 §3's coords-as-params core profile is revised — `thread_idx(...)` is the ambient core profile; vendor addressing beyond it enters as vendor intrinsics exactly as 090 prescribes. (ii) 090 §5's buffer contract merges with D6: the adopt descriptor is where dictated encodings live — 090 §5 and Layout-as-adoption-descriptor (150 §7.2) are one concept. (iii) The §5 async-readback path is dropped per D4. "No tensor semantics, no autograd — interop partners" now reads: *foreign* tensor libraries are interop partners; `pdum.tensor` is in-house.

The events seam moves verbatim into `pdum.core` (kernel/events.py:1-24). New emission points at `pdum.tensor`'s compile-ish seams (Program build, grad/adjoint derivation, descent certification) make assemblage-tier cache discipline assertable the way kernel-tier is today — `forbid`/`no_compile` pin "this training loop builds zero Programs".

---

## 3. The precision model

**The retraction, on record.** Mid-planning the owner retracted the interim carrier-only-surface stance. The forcing observation: loaded weights dictate precision — a checkpoint arrives with encodings (f32/bf16/int4+scales) as facts drawn from the data; an audio callback hands you f32 buffers; a swapchain has a format. A model with no user-visible precision anywhere cannot state these facts honestly (they had "nowhere to live"). The revised model: **facts at the boundary, choices in the interior, carrier semantics throughout.** The 150 two-surface model (A1 + flag 3) is the recorded fallback (§3e).

### 3a. The semantic contract

Semantics are carrier-valued end to end (`bool/int/rat/real/complex`, dtypes.py:31; COMPUTE.md:101-125 already demoted dtype to a footprint/cost resource). No compute dtype on the user surface. Dtype is a property of buffers/encodings at the boundary, recorded in load/adopt/out descriptors, never on tensors mid-computation. **Exact decode:** every finite float bit pattern is a specific rational; int4+scale decodes to `scale[g]·q`, exactly rational — so the denotation stays exact over exactly-known inputs; the file fact does not poison the semantics. This is already tensorlib law at two sites: `FunctionalBuffer` ("representation never enters the semantics", buffer.py:79-84) and physical iota (exact rational, cast only at the read, compute.py:282-286); D6 generalizes it to all reads through `Buffer.read(loc, dtype)` (buffer.py:57-61), dtype supplied by the descriptor. **Explicit rounding:** when rounding IS the semantics (QAT, stochastic rounding, a model defined as its bf16 computation), it is an explicit exact op `round_to(encoding)` — a well-defined function on carriers whose result is again exact. **The discipline:** every precision appearance is a boundary fact, a descent choice (§3c), or an explicit `round_to`; mid-program astype-as-semantics stays convicted and the integrated IR has no op for it. The float64 reference executor (compute.py:235,270; zoo_common.py:47-49) is the *oracle's own interior representation* — a declared oracle property with tolerance framing, lowering number zero.

Two holes, resolved as amendments (adopted): **inf/nan** bit patterns are valid boundary bits but elements of no carrier — default stance: refuse at decode (matching the C backend's constant refusals, c.py:127-135), with an opt-in extended-real carrier as a recorded future door; test with a nan-bearing checkpoint. **Entailed out-rounding:** writing real-carrier results into an f32 out buffer rounds without an explicit op — the boundary contract is written `denotation = encode_out ∘ carrier-function ∘ decode_in`, with encode_out's rounding part of the descriptor's declared tolerance; the one implicit rounding in the model, declared, never silent.

### 3b. The boundary descriptor

**Adoption descriptor = Buffer handle + Layout + Encoding (+ carrier + units)** — field-for-field today's `Tensor(buffer, dtype, layout, value_units, carrier)` (tensor.py:42-48), extending the 150 ruling that the adoption descriptor "should BE tensorlib's Layout" (150:1613). The migration is a **re-roling**, not an invention: the dtype-carrying Tensor retreats to the boundary (load/adopt/out and the reference executor); interior program values carry carrier+units+layout shadows only — already how `infer` works (fabricated-stride shadows, ir.py:336-345; no interior pass consults dtype except through the 8-byte convention). `Encoding` outgrows np.dtype (dtypes.py:12-21 covers np.dtype + ml_dtypes only): a small hierarchy — NumpyEncoding, QuantGroupEncoding (int4 nibbles + per-group scale tensor, a composite over two buffer regions), FormatEncoding (bgra8unorm-srgb with the sRGB curve in decode) — each declaring its exact decode/encode. Flows: checkpoint load (safetensors metadata → per-weight descriptors), foreign adoption (the 150 audio-f32 gap — c.py:324-327's hardcoded result dtype forcing alloc+copy — resolves as a descriptor fact, no new concept), and `out=` (descriptor + writable flag + the F6/F23 alias-refusal obligation).

**Depth correction (amendment from review; adopted).** The prior draft claimed "interior values are never Tensors (already true)". False at runtime: `Tensor` is the reference executor's universal value type, minted with a dtype at every instruction (`_tensor_like`, compute.py:122-130; ir.py routes const/iota/shadows through it). The actual shape of M7: Tensor keeps an encoding-ish field as a **declared oracle-internal property** (consistent with the float64-forcing framing); the user-surface change is constructor/API policy — `dense`/`from_numpy`/`field` are boundary acts producing descriptors; **IR programs cannot mint encoding-bearing values**, enforced at the IR/signature layer (lowering never *consults* the field), not by deleting the field. The M7 edit list extends beyond attention.py:57 to the machinery's own dtype sites: autodiff's `dtype="int64"` consts (autodiff.py:571,750) and `const_like(dtype=)` (autodiff.py:175-178), and `ir.run`'s `p.get("dtype", np.float64)` (ir.py:149). Small work either way; the doc's depth claim now matches the code.

**Stride units (amendment; §7 O6).** Strides are bytes, so `dense()` builds strides from `dtype.itemsize` (tensor.py:82-87; DESIGN.md D4 chose bytes deliberately, DESIGN.md:206-211) — removing dtype from the surface still requires an element size to build any dense layout. Recommendation: semantic layouts in element units with byte elaboration deferred to the boundary/lowering step where the encoding is known (`strides_in_elements`, tensor.py:114-121, is the existing escape hatch); `field()` on structured dtypes becomes a boundary-descriptor operation. Consequence for L2 (recorded in §8): interior shadow layouts (8-byte-fabricated, ir.py:336-345) are relative-nesting-only, never byte-authoritative — bufferization re-materializes interior layouts from assigned encodings, and the capacity WF consumes the re-derived layouts, or the A3 fix is illusory.

### 3c. The interior

Precision enters at exactly three lowering points: **(1) Descent (L4): precision-demotion licenses** — taxonomy {none, reassociation, precision-demotion}, equivalence over the carrier denotation, tolerances + input domain in the declaration, license set in the widened registry key (F8/A4); the numeric tier monitors divergence, never certifies it (150:303-305). Per D6(iv) the schema is settled before L4 — M9 lands a concrete license-schema stub plus one worked declaration (the A1 GEMM f16-tile/f32-accumulate license), so L4 starts against a datatype, not a paragraph (amendment; adopted). **(2) L2 storage assignment** — materialized intermediates get encodings chosen at bufferization, recorded on the assignment. **(3) Machine-tree byte predicates** — consumers, not choosers. The capacity fix is forced by construction: capacity reads lowering annotations plus boundary facts, never user types — there are none to read. Check against the A1 GEMM: weights bf16, activations f32 (facts); surface program is a real-carrier contraction; descent stages f16 tiles + f32 accumulators under a license; result encodes per the out descriptor; `astype(f16)` appears nowhere; the A3 4× overcount is impossible.

**`round_to`'s AD rule** is a declared policy, not a derivative (true derivative zero a.e.; QAT wants straight-through): registered as an explicit primitive-table entry, documented like the tie caveat, decided before any QAT sample is promised (§7 O12).

### 3d. Impact

Surface amputation plus one additive subsystem: dtypes.py splits (CARRIERS semantic; `as_dtype`/`bfloat16` seed Encoding; `carrier_of` becomes descriptor-side inference); tensor.py re-roles (structural change near zero — the change is role and rule); buffer.py nearly ready; compute.py keeps float64 forcing as oracle property; ir/signatures trivial (`const(carrier=)`); memory/placement gain descriptor-fed sizes (the one real build, required by A3 regardless). Tests: dtype tokens concentrate in fixture construction (test_compute 24, test_regressions 16, test_ops 14 — verified counts), still legal as boundary acts. **The zoo is precision-clean**: no astype anywhere (only autodiff.py:894, oracle-internal), one IR-level dtype use (attention.py:57, already a carrier declaration in disguise) — this migration threatens zero denotations.

### 3e. Fallback

The two-surface model (150 A1 + flag 3: family element-dtype parameter + descent license) is the recorded fallback. **Criterion:** before L4, write the mixed-precision/QAT sample (master vs bf16 weights, loss scaling) in boundary-facts terms. Fall back iff a required program's *meaning* — not cost — depends on an interior encoding that is neither a boundary fact nor expressible as explicit `round_to`. Cost or convenience pressure alone does not trigger it; the license machinery is model-independent either way, so the fallback is a surface change, not an architecture change. Falling back is a written owner decision at M7, never a silent switch.

---

## 4. The syntax stack

Tags: **[exists]** running today; **[planned]** mechanism exists, wiring absent; **[proposed]** new machinery this plan commits to.

### S.1 Assemblage

**[exists]** today, Build-authored (zoo_common.py:65-71). **[proposed]** the same Program in the shared syntax:

```python
def rmsnorm(x, g, *, feat="e", eps=1e-5):          # straight-line; host Python composes
    ms = (x * x).mean(feat)
    sd = (ms + eps).sqrt()
    xn = x / sd.repeat(feat, x.extent(feat))       # broadcast stays a DECLARATION
    return xn * g.repeat_like(x, but=feat)
```

Ops are tensorlib's own; methods are sugar over `emit`; the alignment refusal is unchanged (compute.py:167-170). SSA names come from Python binding names via core name-fate analysis, replacing hint dedup (build.py:19-28); input names are declared and `grad`'s map stays keyed on them (autodiff.py:867-876); dotted prefixes (`L{i}.wq`) come from host-level declaration. **Lift rule, normative** (amendment; adopted): Python *numbers* lift to consts aligned to the tensor operand (dims + charts inherited) — the one implicit lift, const-only; tensor–tensor misalignment always refuses; `repeat` stays explicit; pinned by a test that `x / mean(x)` (missing dim) refuses. **Vocabulary completeness** (amendment; adopted): two zoo builders exceed the method list — megatron binds *interior* expressions (`b.emit("bind", ...)`, megatron.py:67-83; gate item 5 runs through it) and attention derives iota from an interior instr plus a 2-operand composite reduce (attention.py:52-58, 140-142). The committed S.2 vocabulary therefore includes `.bind(level=...)`, `iota_of(t, dim)`, and the 2-operand reduce form; `Program`/`Instr` remain public hand-constructible data regardless, and the emit-level naming contract (hint dedup, dotted prefixes) is a named core-naming deliverable so hand-emits are not left nameless.

**Entry point:** ONE optional annotation, `@assemblage` — exactly a pdum Handle (Phase-A capture, `("H", code, env_fp)`, two-tier cache, lowering at first call); the product is a Program + input layouts. Without it every call site hand-manages lowering and forfeits warm hits and the fresh-closure idiom. Subject to the §7 O1 identity decision for structural captures.

### S.2 The ONE shared expression syntax

Type-directed: scalar-typed values lower to the scalar core; tensor-typed values lift arithmetic pointwise plus view methods (`shift/slice/pad/rename/with_charts/repeat/bind`) plus reduce/contract. Invariant: straight-line/no-branching, enforced by core control-flow detection at lowering — replacing `Sym.__bool__`'s trace-time raise (mdsl.py:92-97). Bounded `if`/`for` exist only in the scalar statement language (base_lang.py:198-245, 271-309), never over tensor-typed values.

**(1) Scalar helper — one definition, two consumers** (pointwise marker in assemblage AND device function inlined into @compute); **[exists]** as traced lambda (zoo_common.py:17), **[proposed]** same body AST-captured:

```python
def gelu(x):
    return 0.5 * x * (1 + tanh(GELU_C * (x + 0.044715 * x*x*x)))
```

Lowering target unchanged: the `Arg/Const/Prim` Node schema survives as the marker-body IR; the AST producer replaces the tracer exactly as mdsl designed for (mdsl.py:11-15). Captured constants become Consts with FnType-in-key semantics. **The two-consumers property gets a gate** (amendment; adopted): M8 adds a differential — gelu lowered as a pointwise marker under `ir.run` vs inlined as a device function into a @compute kernel, compared numerically — since nothing else pins that the shared syntax's two lowering paths agree.

**(2) Reduction combine.** **[exists]** attention.py:20-32; **[proposed]** with record state and shared subexpressions (the traced lambda re-expands `maximum(L[0], R[0])` five times, attention.py:26-28):

```python
def flashsm_combine(L, R):                     # State = (m, den, o)
    m = maximum(L.m, R.m)
    sl, sr = exp(L.m - m), exp(R.m - m)
    return State(m, L.den*sl + R.den*sr, L.o*sl + R.o*sr)
```

**D3 boundary, recorded** (amendment; adopted): a reducer is not one expression — it is a structured declaration (state/element/lift/combine/init/project + declared associativity, mdsl.py:400-425). The `defreducer`-shaped *declaration API* survives; only its lambda bodies switch producer. One uniform syntax does not lower all three artifact kinds; the plan says so rather than discovering it at M5. Same for the acceptance criterion: the D3.2 syntax must reproduce marker-body granularity — declared, traceable, differentiable bodies and declared combines — or the migration silently loses derived composites (flash's derived backward, test_zoo.py:66-79) and the f.4 resolution. This is an explicit gate of the M5 design, not an aspiration.

**(3) Tensor-typed fold step** (FDTD; today a nested Build, physics.py:94-114). **[proposed]**:

```python
def fdtd_step(E, H):
    dE = (E.shift(x=-1).slice(x=(0, N-1)) - E.slice(x=(0, N-1))).with_charts(x=h_chart)
    H1 = H + c * dE                            # c lifts to a const inheriting dims AND charts
    dH = (H1.slice(x=(1, N-1)) - H1.shift(x=1).slice(x=(1, N-1))).with_charts(x=e_chart)
    E1 = E + c * dH.pad(x=(0, N), fill=0.0)
    return E1, H1                              # carry; layout preserved — checked (ir.py:224-228)
```

Step input names = param names per the fold contract (ir.py:162-175); the chart-inheriting lift dissolves physics.py's rechart boilerplate; nesting is just a function. Note the captured `N` in slice/pad — the §7 O1 counterexample lives in this very sample.

**The merged derivative table.** mdsl `_D` (mdsl.py:175-203) and pdum `JVP_RULES` (transforms.py:217-228) are the same object — op → linearization rule, `None` = gradient-free position — and merge into ONE table in the core transform column. `CompositeMarker.partial(i)` is reimplemented as forward-tangent application over the lowered body with basis seed, DCE, registered `name.d{i}` — derivation-under-cache. The reducer BPTT machinery consumes partials through the same interface, untouched. **At-kink (f.5), forced here:** both pairwise tables are one-sided and partition (mdsl.py:190-191; transforms.py:210-227), but tensorlib's reduce-max backward gives every tied element the full cotangent — `[1,1,0]` is pinned (test_autodiff.py:377-387). Deriving reduce adjoints through the pairwise combine flips the pin to `[1,0,0]`. RECOMMENDATION: adopt the partition law as table law, re-pin first-wins, record the semantics change now — before convex-optimization consumers exist. Keeping the overcount means a hand adjoint contradicting the derivation machine forever. Owner sign-off required; the re-pin lands as its own commit with a canon paragraph so the decision is visible independent of the frontend swap (amendment; adopted).

### S.3 @compute kernels **[proposed]**

```python
@compute
def my_shader(f, img):
    y, x = thread_idx("y", "x")                # ambient intrinsic, NOT positional params (G1)
    img[y, x] = f(y, x)                        # explicit store into an ARGUMENT buffer (G2)

f = twill(4, 3) | weave | zoom(center=(20, 50), r=20, scale=5)
my_shader(f, img, launch=grid(blocks=ceil_div(img.shape, 16), threads=(16, 16)))
```

- `f` **[exists]**: Stage/Pipeline runs today (combinators.py:180-204, 254-278). Cache key **[exists — the caching half]**: `zoom(center=(30,60),...)` → same FnType → warm hit, new center rides the uniform channel; swapping `weave` → new artifact (capture.py:96; registry.py:134). The lowering half **[proposed]**: inlining through an FnType-typed param (refused today, base_lang.py:383-384), arg-rooted env-path ABI slots (pack.py:271 legalizes env-rooted only), designed on passing the Handle VALUE to `_build` alongside arg types — dispatch has it in `args` — rather than snapshot recovery from FnType alone (snapshots are WeakKeyDictionary code-keyed, capture.py:59, and can be GC'd). Guard policy for argument Handles is a §7 decision (O11): captures are frozen at construction and fp recomputed per call (capture.py:88-96), so per-call fp without cell guards is arguably sound — decided, not drifted.
- `thread_idx` **[proposed]**: retires coords-as-derived-params (wgsl.py:63-72) and `Backend.param_types` — at M8, with the GRID family.
- Store **[exists, orphaned]**: `core.store` is in the op table (ops.py:149) but nothing produces it; Region is documented pure (ir.py:110-113); two stores in a use-keyed DAG have no ordering edge. **This is the largest genuine design gap in the stack, not a wiring gap** — the effect representation (ordered body statements vs token threading) is a design sub-deliverable gating M8, owner-reviewed before any store lowering lands (§7 O2). Day-one contract: writable-argument/read-capture overlap refusal (`shares_memory` over leaves, the ping-pong message) — the F6/F23 seam re-created deliberately, refused deliberately; in-place only ever as an L2-certified rewrite.
- Launch config **[proposed]**: invocation-only, riding the leaves channel ("never touches any key" is already law, registry.py:126-129). Commitment (amendment; adopted): threads-per-block is a value-specialized bracket — artifact re-render on change, no identity change; blocks/streams are pure launcher data; residual cost recorded: first call at a new block size recompiles.

**Iota unification:** the same picture is expressible today as `pointwise(f_marker, iota(lay,"y"), iota(lay,"x"))` (compute.py:277-311), iota being a zero-byte exact FunctionalBuffer. The descent rewrite iota→thread_idx is a rewrite-driver Stage whose WF predicate is "no iota reaches a materialization boundary"; a @compute kernel is the fused form of a pointwise over coordinate fields — checkable on small domains against `ir.run` (the M8 gate's differential).

### S.4 Vertex/fragment sketch **[proposed — contingent on f.8]**

Reserved shape only; f.8 stays an owner item with a deadline (§7 O4). Consistent with F26 (host owns pass/submit/swapchain) and the 150 probe sketch (150:440-477): `@vertex`/`@fragment` share the ambient contract; `fwidth` is the wrt-ambient residue (graphics.py:45-55); `draw_pair` is the third composition semantics, never punned on `|`; the deliverable is an encodable bundle. The varyings interface check needs result types FnType does not carry (types.py:177-189; 150:1698) — hence the M2 decision to reserve an optional result-type slot in FnType while the cache-key vocabulary is still being moved (amendment; adopted — useful immediately for fold-step diagnostics).

### S.5 Tile and warp sketches **[proposed — the 150 L4 brief governs]**

The authored descent is NOT IR: it is the value of a certified-lowerings registry entry; the kernel boundary in the Program stays an erasure-preserving annotation (K-A, 150:1505). Tile vocabulary: stage/barrier + accum, tile loops as split+bind one level down (K-B); WF certificates checked on the RESULT, never derived from the chain — capacity (Σ staged bytes ≤ `Level.capacity`, placement.py:38-50, **dtype-exact**, which is why D6 interior descriptors precede L4), race-freedom (checker-owned tokens), convexity. Whether `mma` pattern-matches mul→reduce or requires annotation is the open f.6 half, answered inside K-A/K-D. Warp: straight-line post-unroll, uniform control, lane-complete; shuffle/ballot/mma fragments. Below: vendor punning + external oracles as test fixtures only. Both warp and the external-oracle fixtures are **recorded deferrals to L4** in M9's registry — deliberate, with reasons, not the silent-deferral shape the 150 breadth check convicted (amendment; adopted).

### S.6 Invocation concerns appear ONLY in transformation steps

Blocks, shared memory, streams, pipelining never appear in S.1–S.5 user programs; they become visible exactly where a transformation step introduces them (K-C manual-directives-first, LEVELS.md:203-206):

```python
prog = flash_attention().program                   # zero launch facts
d = descend(prog, kernel=annotate(prog, [...]))    # K-A annotation
d = d.split("s", 64).bind("s.outer", "sm")
d = d.stage("k", at="shared", double_buffer=True)  # smem + pipelining become visible HERE
d = d.launch(threads=(128,))                       # step-level, value-tier
art = d.certify()                                  # chain (F7), WF certs (F9), key (F8)
assert bitexact(run(prog, env), art.run(env))      # erasure gate — zoo denotation is the oracle
```

Two existing laws keep this cheap: launcher data rides the leaves channel and never keys (registry.py:126-129), and placement proved the erase-the-annotation test pattern (megatron `level=None`, bit-exact). The one current violation — workgroup size baked into rendered text and fp (wgsl.py:49) — is scheduled out at M8. **Caveat (amendment; adopted):** a `stream=` token is struck from the sample until stream/overlap semantics is designed — it is an M9 open-registry item (§7 O13), sharpened by D4's async rejection.

---

## 5. The migration plan

Sequential steps (inherits 020's meta-pattern); every step ends with the repo green and importable. Gate suites: **P** root pytest, **T** tensorlib tests, **Z** the zoo gate (§2d), **N** notebook harness, **B** loc_budget + ruff + mkdocs. Standing constraints: versions/release owner-only (AGENTS.md:14-33); all members unpublished; `src/pdum/dsl/__init__.py` remains the version mirror throughout (_versioning.py:32; test_example.py::test_version stays green).

**M0 — Installment-0 batch (S).** The 150 silent-wrongness fixes owing nothing to the fork: (1) `out=`-alias refusal (F6/F23): `np.shares_memory` over buffer leaves in `make_grid_launcher` (today dtype/contiguity/rank only, c.py:315-338); (2) Named-as-`out=` designed refusal (CS §7.4); (3) F14 record-value designed refusals; (4) stale-doc repairs: placement docstring, zoo/LEVELS KV wording, workhorse-status paragraph (F32/F34, description not adjudication), **and the F24 wgsl/ch10 stale-promise wording** (amendment: previously dropped; ch09/ch10 stay live teaching material until M8). Deferred with reasons: F33 → M2 (fix at extraction, never port the hole); F28 → M5 (pinning now would prejudge the derivative-table merge); F29 → M6 (pin, matmul, and stdlib die together; tensorlib extent refusal pinned same commit). **GATE:** P green; both 150 alias reproducers refuse.

**M1 — the gate exists before the migration (S).** Tensorlib enters CI unchanged, in place: `uv run pytest explorations/tensorlib/tests -q` in ci.yml; baseline runtimes recorded; coverage thresholds re-set consciously. Notebooks: **take the deferral branch** — an explicit note, harness wiring deferred to M6 when the surface is stable (amendment: wiring them at M1 would turn M5's gate red when the APIs they teach die). **GATE:** CI red if any T/Z test fails.

**M2 — workspace scaffolding + core extraction (L).** `packages/core` born; kernel modules git-mv verbatim; `src/pdum/dsl/kernel/` becomes a thin re-export shim (re-export, never duplicate — isinstance identity must hold) **which survives until M8** (amendment: the prior draft deleted it at M6, stranding demo/, combinators, viz/bench/recorder, root `__init__`, and the archived remainder, all of which import `pdum.dsl.kernel` until M8 — e.g. wgsl.py:38-43, combinators.py:35-36, viz.py:24-30). Extraction edits: F33 fix (registry.py:121-124 refuses unknown kinds); pack refusal rewording; yield-rule relaxation prepared; **c.py moves to `pdum.core.backends` with GRID intact and the Over import severed** (duck-type the over-chain unwrap locally, deleting c.py:290-291's import — otherwise every grid build between M6 and M8 raises ImportError while CI's import-only probe stays green); **combinators' module globals fold into Registry** (as the disposition table claims — previously unscheduled); FnType gains its reserved optional result-type slot decision; member pyproject created bearing the root's exact current `X.Y.Z+dev` string (discover_version_files auto-enrolls every `packages/*/pyproject.toml` as a lockstep published file, _versioning.py:66-74; `scripts/_versioning.py current` must still pass). Same commit: loc_budget constants **split** — KERNEL points at the new home, a new SATELLITE_BASE stays at `src/pdum/dsl` until M6 (satellite lookups resolve relative to KERNEL.parent, loc_budget.py:113, and a declared-but-missing satellite is itself a breach, loc_budget.py:116-117); `--cov` widened; mkdocstrings paths; uv.lock regenerated; ci.yml:47's C-probe repointed. **GATE:** P green through the shim; identity pins green; **new pin:** fresh-closure + mutated-capture re-extract; B green with new buckets.

**M3 — tensor promotion (M).** git mv `explorations/tensorlib/tensorlib` → `packages/tensor/src/pdum/tensor` (zoo inside); tests → `packages/tensor/tests`; sys.path conftest deleted; ci.yml/coverage repointed; root gains the member dep **in `[dependency-groups]`** (amendment, §2a); tensorlib docs + notebooks enter nav **under a pre-integration snapshot banner** (amendment, §2b); 140/150/160 enter nav; 100/110/130 labeled superseded. **GATE:** T+Z green at new paths in CI; `import pdum.tensor`; B green.

**M4 — tensorlib onto core: caching, registries, naming (M).** Process registries die: `COMPOSITE_*` + `node_digest` + `_register_marker` → core-cache-backed registries with derivation-under-cache; lookups repointed in ir/signatures/opcount/autodiff/compute; the Node schema relocated to its own module as the declared stability boundary; events-seam emission at Program-build/grad seams. The three name managers reconciled explicitly (§2b). **The joint refusal battery is created here** (amendment; adopted — 150 §7.8's one-voice contract was invoked but never scheduled): the shared refusal-shape helper (what happened / principle / quoted fix / loc) + a battery seeded with tensorlib's quote-the-fix pins and the M0 refusals; every later step's refusals (M6 extent, M8 alias/kind) extend it. **Fixture capture begins:** golden Node-tree fixtures for the zoo marker set checked in (M5's gate needs them, see below). **GATE:** T+Z green; idempotence pin: re-registering the same marker yields one entry.

**M5 — shared expression syntax: the mdsl frontend dies (L).** AST/reflection producer of Nodes replaces the Sym tracer; core straight-line detection replaces `Sym.__bool__`; `defmarker` retired; `defreducer` survives as the structured declaration API with producer-swapped lambdas (§4 S.2's D3 boundary); record-typed reducer state; tensor-typed lifting lowers fold steps to step Programs. The expression rule pack is **tensor-side and imports only `pdum.core`** (amendment: base_lang lives in stdlib until M6, and importing it from packages/tensor would create a distribution cycle with M3's root→tensor dep). Merged derivative table lands; `CompositeMarker.partial` reimplemented; **at-kink decision executed as its own commit with owner sign-off** (§4 S.2). The Sym tracer and defmarker/defreducer entry points are **archived as importable frozen fixtures**, not deleted (amendment: the producer-equivalence gate needs the tracer it retires). Notebooks 07-08 dispositioned (frozen or re-authored). **GATE:** Z green with zoo markers/reducers/fold-steps re-authored — denotation-identical incl. flash derived backward and FDTD charts; producer-equivalence: traced Node tree == AST-produced tree over the zoo marker set (against the frozen tracer fixture); the re-pinned kink test.

**M6 — Build dies; @assemblage; stdlib dies; archive move (L).** (a) Zoo builders re-authored in S.1/S.2 (incl. the bind/iota_of vocabulary); `@assemblage` lands (subject to the O1 identity decision, which must be made before this step); Build **archived as fixture** (§2b), deleted from the live surface; F17 naming law pinned; `const(carrier=)` rename rides this same change so denotation tests run once against the combined edit. (b) stdlib splits: base_lang/surfaces/batteries → `pdum.core.scalar`; arrays/transforms die with the re-homing checklist executed; monolithic `install()` → per-package entry-point installs — and the archived demo package registers via the same mechanism for its two remaining steps (batteries spells onto demo backends by name until M8, batteries.py:28-29, 91-95). (c) Archive move, dsl_reference pattern; `src/pdum/dsl/__init__.py` remains the version-bearing shell (owner confirms at this step: shell vs owner-run re-point, §7 O9); **the kernel shim is NOT deleted here** (amendment, see M2); book ch12-14 archived (build_chapters split; frozen nav), ch00-11b repointed, **ch09/ch10 + demo/ NOT archived** — they hold the shader story until M8. Same commit: matmul + 3.0 pin deleted, tensorlib extent refusal pinned (gate 8); SATELLITE_BASE moves and caps redraw; tensorlib doc revisions for Build/mdsl sections (README/DESIGN historical notes); notebooks 09-13 dispositioned; harness wiring for surviving tensorlib notebooks lands now. test_grid.py notes: it imports stdlib arrays/transforms (verified) — its over-dependent cases archive here, its identity pins (test_grid.py:38-39) move against the surviving GRID family until M8. **GATE:** full zoo gate 1-8 incl. the builder-identity pin (live-vs-frozen-fixture, in-process — no serialization needed); P green minus archived tests; N and B green, verified specifically for ch09/ch10.

**M7 — precision: boundary-facts (M).** The §3 design lands before any L4 work: dtypes split; Encoding module; descriptors; `dense/from_numpy/field` re-roled as boundary acts with the IR-never-consults enforcement; `Buffer.read` descriptor-fed; machinery dtype sites converted (autodiff.py:175-178, 571, 750; ir.py:149); `round_to` + its AD-rule decision; `_ITEM = 8` → descriptor+annotation-fed sizes; the retraction paragraph enters canon (§3); the QAT fallback sample written and evaluated. **The Encoding-on-buffer-leaf interface is specified as an early M7 artifact and named as an input to M8(vi)** (amendment: M8's wgsl rebuild consumes element encodings; M8(i-v) stay parallel to M7, M8(vi) prefers M7-first or ships against a transitional np.dtype-backed shim). **GATE:** Z green (zero denotation changes expected); cost-oracle values re-derived as their own reviewable diff; QAT sample passes or the fallback triggers as a written owner decision.

**M8 — @compute: thread_idx, stores, function-valued args (L).** In order: (i) the effect-ordering representation — decided, owner-reviewed, landed (the O2 prerequisite); (ii) the store path: subscript-store lowering → `core.store`, verify relaxation exercised, renderer spelling, unit-result dispatch, **day-one writable-arg/read-capture overlap refusal**; (iii) `thread_idx` + the iota→thread_idx descent Stage with the never-materialize WF check; `Backend.param_types` and the GRID family retire together; **cross-family inline kind checks land here** (a body referencing the thread_idx ambient outside a kernel context refuses, pinned per R cell of the §2e matrix; the two-concept-split canon paragraph — Handle = authoring/ABI contract, registry route = execution family — is written into this doc's canon by this step); (iv) function-valued-argument lowering (Handle-value-to-`_build`, arg-rooted ABI slots, the O11 guard decision executed); (v) launch config at invocation with the G4 commitment recorded; (vi) the wgsl backend rebuilt from the demo seed; demo/ + ch09/ch10 archive; **the kernel shim deletes here** with viz/bench/recorder and root `__init__` repointed in the same commit; **test_notebooks.sh's GPU probe repoints** at the new backend's `is_available()` in the same commit demo/ archives (the probe hard-fails on ImportError today). Also: the epoch/ownership handshake for device-resident adopted buffers named as an open decision inside G5 (150:1670); perf-gate disposition executed — port a dispatch-overhead + speedup gate to the @compute path on bench.py, or record deliberate retirement until L4 cost work (the GRID ≥10x gate dies here, 150:1732). **GATE:** the §S.3 example runs on c and wgsl; iota-unification differential; **two-consumers differential** (gelu as marker vs inlined device function); key-discipline pins (shape miss / value hit; launch never keys); alias refusals at the new seam; compiles==1 thesis test for fn-valued args; N green with replacement chapters.

**M9 — runway handoff (S).** The tiled-matmul zoo entry (K-D names it missing); the L4 handoff memo (§8) incl. the **license-schema stub + the worked A1 GEMM license declaration**; the L2 blocker list; the open-items registry, expanded per review: f.8 graphics (owner; **decision is a named precondition of starting L4 tile work** — 150:1489-1491's before-or-alongside rule), the f.6 mma half, RNG + executor seams, **warp DSL (deferred to L4, vocabulary reserved in S.5), external-oracle fixtures (deferred to L4 validation), stream/overlap semantics, device-resident persistent state / buffer donation, Program-normalization (A5), the GA-operators door** (per-type operator extension, surfaces.py its intended home, 150:1400/1714), and adversarial input families for L4 flagships (-inf masks, cancellation, non-divisible tails, A8 — one -inf-mask attention case seeded into gate Z at M6). The ring/window boundary sample is specified with **both** instances — KV-decode and the audio delay-line (BC flag 5) — and the someone-else's-loop canon paragraph (constraint, fresh-closure idiom, fold-vs-encode-bundle split, 150:1387) enters this doc's canon. **GATE:** Z green incl. tiled-matmul; 160 merged; L4 may start once f.8 is decided.

### What we are NOT doing

No async (D4) — no async readback, no async pipeline; streams only as M8+ transformation-step values once designed. No CUDA-clone language. No 130 stages 2-3, no emission-seam artifact. No over/jvp as pdum concepts; no matmul/3.0-pin/Build/Sym-tracer/process-registry ports (frozen fixtures are test-only). No carrier-only surface (retracted) and no two-surface model now (fallback only). No graphics scheduling in this plan — f.8 is an explicit owner decision with a deadline. No publishing of members, no release.yml/_versioning edits beyond the M6 coordination (owner-run); no Option-B big-bang; no speculative constraint-repin port. No new tensor verbs, no translation frontend (D1).

---

## 6. Arbitration disposition (150:1567-1580 under D1–D6)

| # | Item | Disposition | Lands |
|---|---|---|---|
| f.1 | IR fork | RESOLVED at root: no pdum tensor dialect, no frontend, no second AD; 130 stages 2-3 cancelled; tensorlib grad+dce is the one AD | M4–M6 |
| f.2 | matmul extent UB | RESOLVED by supersession: pin + matmul die; tensorlib extent refusal pinned in their place | M6 |
| f.3 | pipeline 1-vs-3 | RESOLVED: `\|` = fuse-inline only; sequencing = host + fold; async variant dead (D4); PSO rides f.8 | M8 |
| f.4 | rule-table polarity | RESOLVED: frozen-primitives/derived-composites is the only polarity left | M5 |
| f.5 | at-kink | DISSOLVED as collision; residue: partition-law re-pin [1,0,0], owner sign-off, own commit | M5 |
| f.6 | contract vs fusion | HALF RESOLVED (tensorlib op set, no contract primitive); mma-selection half open inside K-A/K-D, incl. F35's saved-for-backward miss | M9 registry |
| f.7 | precision | RESOLVED + extended by D6: boundary-facts; retraction recorded; two-surface = fallback with written criterion | M7 |
| f.8 | graphics | OPEN — owner, with a deadline: decide before L4 tile work. §S.4 reserves the shape; F24/F26 carried as constraints | M9 precondition |
| f.9 | KV exclusion | ABSORBED into the L2 runway; interim = alias refusal + ring/window sample (both instances); wording fix at M0 | M0/M9 |
| f.10 | workhorse status | RESOLVED by supersession; `ir.run`/scalar-dispatch oracle status stated descriptively | M0 |

---

## 7. Risks and open questions for the owner

Ranked by severity; each states what resolves it. Duplicates across the review streams are merged.

**O1 (high) — Assemblage-tier cache identity vs structural captures.** `("H", code, env_fp)` with range-bucketed int fingerprints means same code + different captured extents (fold/slice/pad params, dim counts — physics.py:36-42, megatron.py:38-40, and the S.2 fdtd_step sample itself) fingerprints identically and silently reuses the wrong Program — the worst failure class by valuekind's own soundness law. Options: (i) mandate operand-derived extents (`x.extent(d)`) and refuse structural use of captured ints at lowering; (ii) promote structural captures into the type (Shaped/LiteralType precedent, arrays.py:41-47); (iii) value-key the `@assemblage` tier, forfeiting warm hits across extent changes (probably correct — building a Program is cheap relative to a wrong one). **Resolved by:** owner picks before M6; the choice is recorded in §2c and S.1.

**O2 (high) — Effect ordering for @compute stores.** Region is documented pure (ir.py:110-113); nothing produces `core.store`; two stores have no ordering edge. Candidate representations: program-ordered body statements vs explicit token threading. L2 bufferization will consume the same mechanism (K-F), so this outlives @compute. **Resolved by:** the M8(i) design sub-deliverable, owner-reviewed before any store lowering lands.

**O3 (high) — D3.2 differentiability contract.** Derived composites (the f.4 resolution; flash's derived backward) depend on marker-body granularity — declared, traceable, differentiable bodies and declared reducer combines. The shared syntax must reproduce this or the migration silently loses it; the reducer additionally needs the structured declaration API to survive (§4 S.2). **Resolved by:** the acceptance criteria written into M5's design review; the flash derived-backward gate is the instrument.

**O4 (high) — f.8 graphics deadline.** D1-D6 are silent on the founding domain; 150 requires the graphics installment before-or-alongside the tile step and calls silent deferral a charter violation (150:1489-1491, 1578). **Resolved by:** the owner deciding schedule-or-defer-with-reasons before L4 tile work starts (the M9 precondition), plus the M2 FnType result-slot reservation so the decision is not foreclosed by the extraction.

**O5 (high) — The F17 naming law's exact scope.** Input-name identity is hard (grad maps, `m.inputs`); fold param-name = step-input-name is hard (ir.py:162-175); interior binding-derived names are soft; anonymous temporaries need a deterministic scheme; hand-emitted Instrs (megatron bind, attention iota) need the emit-level naming contract once hint dedup dies. **Resolved by:** the naming-law spec written and pinned (rebuild-stability test) before M6(a).

**O6 (high) — Stride units.** §3b recommends element-unit semantic layouts with byte elaboration at the boundary/lowering, reversing DESIGN.md D4's bytes-in-representation; L2 then re-materializes interior layouts from assigned encodings. **Resolved by:** owner ratifies the recommendation (or keeps bytes and accepts descriptor-supplied itemsize at every construction site) at M7 design review.

**O7 (high) — Tensorlib canon and notebook governance.** M3 publishes docs and 14 notebooks that M5-M7 partially reverse; four notebooks teach Build/defmarker (07-09, 11). The plan's answer — snapshot banner at M3, revision line items at M5/M6/M7, notebook disposition at M5/M6 — costs real re-authoring work not yet budgeted. **Resolved by:** owner accepts the budget or downgrades the affected notebooks to the frozen pattern permanently.

**O8 (med) — The at-kink re-pin.** [1,1,0] → [1,0,0] is a user-visible semantics change to a pinned behavior. **Resolved by:** sign-off on the partition law before the M5 merge; the alternative (keep the overcount, exempt reduce-max from derivation forever) is recorded as rejected-by-default.

**O9 (med) — The M6 version-mirror choice.** Keep `src/pdum/dsl/__init__.py` as the version-bearing shell (plan default, zero owner-run work) vs an owner-run change re-pointing INIT_PY and release.yml to a renamed archive. **Resolved by:** owner decision at M6(c).

**O10 (med) — G4 launch-key residue.** The plan commits threads-per-block as a value-specialized bracket (recompile on new block size, no identity change). **Resolved by:** confirming this cost is acceptable, at M8(v).

**O11 (med) — Argument-Handle guard policy.** Per-call fp recomputation with frozen captures (capture.py:88-96) vs extending `_guards` to arg-rooted paths with measured dispatch cost. **Resolved by:** an explicit decision in M8(iv)'s design, documented either way.

**O12 (med) — Precision edge policies.** inf/nan decode stance (default refuse, opt-in extended carrier) and `round_to`'s AD rule (straight-through vs zero). **Resolved by:** both declared in M7, before any checkpoint/QAT sample is promised.

**O13 (med) — Stream/overlap semantics.** Named at transformation-step level but designed nowhere; D4 makes "overlap without an async surface" a real question. **Resolved by:** the M9 registry entry with L4/L5 ownership; no `stream=` token in any sample until then.

**O14 (med) — Device-resident state.** M8's G5 walks into the adopted-GPU-buffer mutation/epoch-ownership question (150:1670), and training loops are host-bandwidth-bound without device-resident persistent state (150:1710) — "really the first L2 requirement". **Resolved by:** the named open decision in M8 G5 and the M9 L2 blocker entry.

**O15 (med) — Program normalization (A5).** Until the pass exists, both content-addressed tiers are private caches (§2c). **Resolved by:** scheduling the pass in the L4 runway or permanently advertising the caches as private.

**O16 (low) — Perf gating.** The only performance instruments (GRID ≥10x, bench thresholds) die at M8. **Resolved by:** the M8 disposition (port to @compute on bench.py, or recorded retirement until L4).

**O17 (low) — The operators door.** Per-type operator extension (GA product, quaternions) has no registration surface; surfaces.py is its intended home. **Resolved by:** the M9 registry entry; no work now.

**O18 (low) — CI budget.** Tensorlib tests + notebooks have unknown CI runtime; coverage thresholds (80/70) will jolt. **Resolved by:** M1's baselines and conscious threshold re-sets at M2/M3.

---

## 8. The runway

**What L4 inherits on completion.** The 150 L4 brief intact (K-A..K-F, 150:1501-1517), sharpened by D3: the tile DSL is a vocabulary + WF certificates over the one frontend machine; kernel boundary = erasure-preserving annotation (the `Dim.level` template, layout.py:95-107) + certified-lowerings registry entry; `chunk_fp` at named-op level (F21, trivially satisfiable); the widened registry key (F8/A4: license set, saved-set demand, layout classes, rules-generation); chain-as-mandatory-stored-artifact (F7); WF certificates checked on the result (F9: race/capacity/convexity), with capacity **dtype-exact** from M7's descriptor-fed sizes (150 A3 — the 8-byte era ends at M7); assurance tiers with trust at ≥2 (A8) and adversarial input families named for flagship gates; F19/F20/F9 as tile-DSL constraints (derived axis names, blocked-k as declared reduce, checker-owned tokens); the license-schema stub + worked GEMM license from M9; the tiled-matmul zoo entry beside the existing flash naive/fused pair (attention.py:122-149) and heat/FDTD steps (K-D complete); manual fusion directives living exactly where D3 puts invocation concerns (K-C); `peak_memory(local=True)` as per-kernel footprint and materialization-boundary bytes as traffic (K-E). Open inside the brief: the f.6 mma half (pattern-match vs annotation, incl. F35's saved-for-backward miss), Program normalization (O15), stream semantics (O13), warp vocabulary and external-oracle fixtures (recorded M9 deferrals), and the f.8 decision as the start precondition.

**What L2 inherits.** Its inputs already exist: `overlaps/footprint/injectivity` as the exact alias theory (tensor.py:123-148; REPRESENTATIONS.md:40-45), writes-through-views gated on injectivity (DESIGN D6), materialize-elision when nesting holds (CONCERNS #21), and K-F's ordering (bufferization consumes kernel boundaries, after fusion decisions). The blocker list, carried explicitly: value numbering for `.rc` name≠value (CONCERNS #27); chart-denominator normalization for codegen (CONCERNS #12); interior-encoding assignment from M7 (§3c point 2), with the O6 consequence that interior shadow layouts are relative-nesting-only and L2 re-derives byte layouts from assigned encodings; the ring/window boundary sample with KV-decode AND audio delay-line instances (f.9/F16, BC flag 5), mutation confined to the boundary seam, the erasure obligation (same surface program, bufferization reproduces the row-write) as near-term work; device-resident persistent state / buffer donation and the epoch/ownership handshake (O14); and the effect-ordering mechanism from M8(i), which bufferization consumes directly.

**What the whole system inherits.** One frontend machine with six vocabularies over it; one AD with one derivative table and a derivation-under-cache discipline; one cache mechanism with three keyspaces and an honest statement of what is private until normalization lands; one refusal voice with a joint battery; one precision doctrine — facts at the boundary, choices at lowering, carriers throughout — with its retraction recorded and its fallback priced; and a zoo whose denotations gated every step of the way here and gate every step from here (150:1527). Integration first, done carefully. Then L4 and L2 proceed immediately.

