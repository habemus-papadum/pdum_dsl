# 150 — Critical assessment report (run of the 140 charter)

**Status:** as-run report of the team-of-agents assessment chartered by 140. Findings below survived adversarial verification (3 candidates were refuted and dropped; findings that failed verification are dropped, not softened). Collisions with decided positions in either stream are flagged for human arbitration per 140 section 8 — nothing here is self-ratifying.


## 1. Concept inventory — code

# Concept inventory — CODE half (pdum.dsl), charter §4 Q1

**Global placement fact (read first).** Everything pdum.dsl lowers to is one IR: frozen `Node`/`Region` over scalar core ops (`ir.py:89-124`, `ops.py:130-154`). In ladder (axis-two) terms the *entire* IR is L0 denotation of scalar expressions: there is no representation of footprint, storage, placement, machine levels, or kernels. Machine concepts — launch domain, lane, buffers — live outside the IR, in `Backend` records, launchers, and the `out=` calling convention (`registry.py:55-70,126-137`, `c.py:315-338`). Axis-one is real in the code; axis-two exists only at its bottom rung. Every placement below inherits this.

**`@jit` / Handle.** Claims: phase-A capture, compile-free; calling the Handle enters two-tier dispatch (`api.py:24-35`). Verified: `make_handle` reflects `(FnType, env)` with no parse/IR/compile (`capture.py:141-174`); `Handle.__call__` dispatches through DEFAULT (`capture.py:110-119`); fingerprint `("H", code, env_fp)` precomputed (`capture.py:96`); untypeable captures fail loudly at the def site (`valuekind.py:135-142`, `test_capture.py:119`). **§2.4(a) mapping:** the conflation is confirmed in source — the *same* Handle is a host-dispatchable kernel (`capture.py:110`) and an inlinable device function when captured (`lower.py:135-137,143-159`; `base_lang.py:381-382`). Role is assigned by context, CuTe-style, but there is **no explicit launch-boundary marker**: the boundary is implicitly "called from Python vs called inside DSL source", and `kind` is a string capture never validates (`capture.py:150-152`). Axis-one: spans L0 and L1 by context. Axis-two: L0. Misfit → F2.

**capture / typeof / fingerprint (KindTable).** Claims: types-only identity; fingerprint soundness law (`fingerprint(a)==fingerprint(b)` ⇒ `typeof` equal) fuzz-enforced (`valuekind.py:1-37`). Verified: MRO dispatch, loud on miss (`valuekind.py:126-142`); `Literal` is the one value-in-type opt-in (`types.py:116-131`); `Base`/`Derived` template identity gives live-coding invalidation (`types.py:136-174`). Two-tier cache: spec key `(fp, arg_fps, backend_fp, generation)` (`registry.py:134`, `cache.py:174-175`); artifact tier content-addressed on `Node.key` sha256 + backend fp (`ir.py:102-108`, `registry.py:190-193`); identity guards against capture drift (`cache.py:62-70`, `registry.py:83-92`); `no_compile()` (`cache.py:73-75`, `test_runtime.py:47`). Axis-one: the shared frontend machine for every level (§3.1 step 4's prerequisite — real). Axis-two: keys are computed on L0 IR; the §7.7 structural-skeleton sketch has no code yet. Clean fit; the best-verified part of the system.

**DerivedValue.** Claims: the ONE wrapper protocol for anything FnType-shaped that isn't source-backed (`derived.py:1-13`). Verified: fntype/fp/captures/kind, dispatchable (`derived.py:21-43`); MRO-covered ValueKind (`derived.py:45-59`). Default `kind` = first capture's role (`derived.py:32-34`) but Pipeline overrides to *last* stage (`combinators.py:207-209`) — role propagation is per-subclass convention, not law. Axis-one: the L2→L1 bridge (composites dispatch like kernels). Axis-two: L0; a Derived template is a build-rule pointer (`lower.py:187-192`). Fit is good; it is also the seam F5 says the grid family bypasses.

**Named/Shaped arrays + isel.** Claims: rank-generic `Array` summary; `Shaped` puts shape in the type (specialize per shape); `Named` puts axis names in the type, erased by codegen — "pedantry at zero machine-code cost" (`arrays.py:1-26`). Verified: kinds (`arrays.py:104-161`); positional indexing on named arrays refused with the fix quoted (`arrays.py:284-291`); `isel` keywords mandatory, every axis exactly once (`arrays.py:302-332`, `test_arrays.py:77,83`); names erased on the emitted node so named and positional twins share one artifact (`arrays.py:276-281`, `test_arrays.py:187`). v1: captures + arguments, read-only, C-contiguous, scalar results refused (`arrays.py:212-216`). Axis-one: type-system freight usable at L0/L1; names answer §7.4's "names live in types" prior by construction. Axis-two: an *erasure invariant* exactly in tensorlib's sense — the one place the code already implements a ladder principle. Strong fit.

**over.** Claims: axis-binding transform, SIMT-shaped: one trailing i64 lane coordinate woven into every capture carrying the named axis (`transforms.py:1-9,232-243,253-259`). Verified: trailing-i64 contract refused otherwise (`transforms.py:271-272`, `test_refusal_contract.py:23`); nesting composes, duplicate axis refused (`transforms.py:274-275`, `test_transforms.py:286,308`); refuses when no capture carries the axis (`transforms.py:283-286`); composition is `lower_handle` re-entry with a shared context (`transforms.py:266-287`). **§2.4(c) mapping:** `over` is a *syntax-level shadow* of split+bind: it binds the axis to a parameter position; the actual machine binding happens later, per-registry — the C grid backend unwraps `Over` chains by `isinstance` to turn lanes into domain loop coordinates (`c.py:290-299`), the scalar family leaves the lane a host-passed argument (`test_grid.py:163-165`). The IR itself records no binding: after `build_over`, a lane param is indistinguishable from a data param. So the binding lives in the *dispatch registry*, not the representation — evidence for reabsorption by the general §2.2 mechanism. Axis-one: L2 transform. Axis-two: claims a split+bind (ladder L3/L4 move) but is representationally L0. The 16×/10× gate holds (`test_grid.py:148-173`).

**jvp.** Claims: forward mode, `jvp(f)(*args, *tangents) -> (primal, directional)` (`transforms.py:261-263`). Verified: arg doubling and float-only refusals (`transforms.py:290-296`); tangent synthesis over the primal DAG with structural-zero `None`s (`transforms.py:34-147`); loops widen carries to (primal, tangent) (`transforms.py:149-167`, `test_transforms.py:101`); finite-difference validation (`test_transforms.py:22`, `test_jvp_rules.py:98`). **§2.4(e) mapping:** `JVP_RULES` is a hand table of 8 math ops (`transforms.py:217-228`); core-op linearization is a *closed if-chain* (`transforms.py:59-147`). The "transform column is open" claim (`transforms.py:141-143`) is narrower than stated → F6. Kinks pick a side (abs/min/max as select, `transforms.py:210-227`) — today's subgradient-at-kinks behavior is an unlabeled accident, per §7.5. Axis-one: L2 transform. Axis-two: a pure L0→L0 program transformation — the one enrichment pass (§3.1 step 2) that exists in this half.

**D.** Claims: in-kernel forward derivative; one partial per enclosing-kernel param, always a tuple (`transforms.py:312-335`). Verified: differentiates w.r.t. the *root* kernel's params however deep the inlining (`lower.py:88-90,98-99`); refuses zero-param kernels, non-float params, in-loop application (`transforms.py:315,325-327,66-72`; `test_refusal_contract.py:37,55`; `test_transforms.py:272`); fwidth sugar validated (`test_transforms.py:123`); zero cost when unused (`test_transforms.py:142`). Axis-one: L0 (value-centric syntax inside a kernel). Axis-two: L0. Misfits: no `wrt` selection (all params, always) — the §7.5 candidate replacement is untouched by code — and D is *family-incompatible*: C-grid coordinates are i64 (`c.py:299`) while wgsl compute coordinates are f64 (`wgsl.py:63-72`), so the screen-derivative idiom refuses on one compute family and works on the other → F4 (equal-citizens violation).

**matmul.** Claims: named contraction — pair the unique shared axis name, woven axes excluded first so batch composes free (`transforms.py:18-22,338-396`). Verified: refusals for unnamed arrays, ≠1 shared axis, index count, Shaped extent mismatch (`transforms.py:346-357,364-372`; `test_refusal_contract.py:64-116`); numpy-matched incl. batching and argument arrays (`test_transforms.py:361`, `test_array_args.py:67`). **§2.4(d) mapping:** matmul is a *lowering-time macro*: it expands directly to a scalar `core.for` accumulation of `mul`/`add` (`transforms.py:381-396`). No contraction/reduce node ever exists in the IR → the contraction structure both streams agreed to preserve until target selection is erased at birth → F3. Axis-one: L0 syntax door (a call name intercepted in `make_call_rule`, `transforms.py:399-408`, shadowing-checked). Axis-two: claims a contraction (an L4-selectable structure), delivers L0 loop denotation.

**Pipeline.** Claims: inert first-class composition; identity a flattened `Derived("pipe",…)`; fusion at lowering (`combinators.py:1-26,180-204`). Verified: roles + composition rules with loud `IncompatibleRoles` (`combinators.py:39-98`); flattened associativity (`test_combinators.py:68`); fusion build rule inlines stages in sequence (`combinators.py:254-278`); executes through the same two-tier path (`stdlib/__init__.py:46-47`, `test_runtime.py:110`). Constraint: pipes thread exactly ONE value and every stage takes exactly one argument (`combinators.py:262-263,271-272`) — the "load-bearing value proposition" operator is unary-only today. Axis-one: L2 composition lowering into a single L1 artifact; the §7.3 dispatch-sequencing sense (frames, ping-pong) has zero code — `_param_types` explicitly refuses pipelines on wgsl (`wgsl.py:66-71`). Axis-two: fusion-as-inlining at L0; no recorded command sequence exists.

**The grid family.** Claims: params are integer domain coordinates; the domain loop lives IN the artifact; one dispatch fills the out array (`c.py:283-299`). Verified: `param_types` derives `(i64,)*n` unwrapping over-chains (`c.py:290-299`); positional args refused when params are derived (`registry.py:167-169`); domain rides `out=` as shape-tuple (allocate) or array (adopt), dtype/rank/contiguity checked against a contract carried on the artifact (`c.py:260-264,315-338`; `test_grid.py:67-91`); domain never enters identity (`test_grid.py:38-39`). **§2.4(b) mapping:** kernel *kind* is a registry property, confirmed — the same `@jit()` Handle is a per-cell scalar kernel through one registry and a whole-domain kernel through another (`test_grid.py:21-24,159-167`); `install_grid` is a registry act (`c.py:385-389`); routes map kind→backend with kind-scoped backends never claiming default (`registry.py:112-124`). The Handle's own `kind` string stays "device" throughout → F2. Axis-one: the first real L1 citizen. Axis-two: the grid loop is the one machine-binding in the system and it lives in rendered text (`c.py:208-222`), not representation.

**registry / dialect / extend.** Claims: one explicit object owns kinds, rules, derived builders, backends, caches; satellites register in; the kernel never imports a satellite (`registry.py:1-34`). Verified: `extend()` layers with copied registrations, per-record `code_for_op` copies, FRESH caches (`registry.py:204-214`); five surfaces as thin sugar (`surfaces.py:40-129`); registration invalidates both tiers (`surfaces.py:30-37`); entry-point discovery (`registry.py:216-224`); batteries wire DEFAULT at import (`stdlib/__init__.py:28-52`, `api.py:8-14`). One inversion found: `backends/c.py` imports `stdlib.transforms.Over` (`c.py:291`) — a backend enumerating satellite transform classes to derive the domain contract → F5. Axis-one: the calling-convention switchboard — the §10 matrix will largely be a description of `routes` + `param_types` + `out=`. Axis-two: nothing; the registry never sees a level.

**Refusal posture.** Claims: refuse loudly and early, message quotes the fix. Verified pervasively: `MissingRule`/`NameFateError`/`VerifyError` all carry source locs (`lower.py:53-58,126,139-141`); `while`/`break` refused with the principle named (`base_lang.py:340-351`); truthiness, promotion, unbounded loops, array results, u64 C constants, inf/nan all refused not approximated (`base_lang.py:119,206-209`; `arrays.py:212-216`; `c.py:129-134`); frozen contract tests (`test_refusal_contract.py:23-165`). The strict-core rule (no promotion, explicit casts — `ops.py:9-26`) is the same voice. Axis-one/two: the posture is level-independent and is the code's most consistent identity. One residual: refusals are three exception types with no shared formatting contract — a joint-system voice audit (§7.8) has raw material but no single seam.

**§2.4(f) the attention sample.** The flagship batched-attention kernel is written with two raw `for s in range(S)` loops over the sequence extent, `S` a captured Python int (`test_grid.py:116-128`) — a direct, live violation of the §2.1 no-extent-loops principle in the code that also passes the step-14 gate. The per-lane scalar dispatch of the same kernel is the gate's baseline (`test_grid.py:163-165`) and the default family (`python.py:182-185`) → F1.

## 1b. Concept inventory — canon and the taught model

# CANON concept inventory (docs 010/020/090/100/110/120/130 + taught model `scripts/book/build_chapters.py`)

All citations read this session. "Taught" = the book builder, the de facto UX contract (020 "The book", 020:28-45).

## Per-concept: what canon claims

**`@jit` / Handle / capture (phase A/B).** Canon: decoration is pure reflection, cannot fail (010 §4.1); phase B lowers once per type signature; a Handle is a first-class DSL closure `(FnType, env, env_fp, snapshot, kind, registry)` (010 §2.4). Axis-1 placement: a `@jit` kernel is simultaneously device-function (inlined via call rules, 010 §3-B) and host-visible (dispatchable) — the conflation charter §2.4 names. Canon never adjudicates it; `kind` is a Handle field *and* grid-kernel-ness is a registry install (`c.install_grid(grid, default=True)`, build_chapters.py:4033-4034) — kind lives in two places.

**Type lattice / typeof / summary dial.** Types are structural summaries chosen by ValueKinds, not Python classes (010 §13); `LiteralType` is the ONE value-in-type opt-in (010 §2.1); static-by-default polarity is the product (010 §14.2). `Array(dtype, ndim, layout, byteorder, writeable, device)` rank-generic default; `ShapedArray`, `NamedArray` are satellite refinements erased at emission — named and positional twins share one artifact (100 §1; 010 ledger step 11, 010:920-928). Ladder placement: this is the specialization/caching axis, orthogonal to tensorlib's levels; charter §7.7's structural-skeleton fingerprint is the planned bridge, not in canon yet.

**Two-tier cache / FastRecord / guards.** Specialization tier keyed `(template_fp, env_fp, arg_fp, backend_fp, generation)`; artifact tier content-addressed on `Node.key` (010 §4.4). Hit path = key + guards + one probe + extract + pack + launch, measured 2.4 µs (010 ledger step 10b, 010:842-866). Guards refuse-or-recompile, never stale (010 §4.4). This is the §3.4 "registry of certified lowerings" substrate canon already claims by shape.

**Registry / five surfaces / punning.** Everything attaches through exactly five surfaces; zero-kernel-diff is CI-enforced (010 §0, §3). 090: stdlib minimalism + squatting test (090 §2); vendor op namespaces = visible portability opt-out (090 §3); runtime do/refuse list — no schedulers, no tensor semantics, no shape inference, "explicit DPS `out=` forever" (090 §4).

**Arrays / named axes / isel.** Named arrays index by name, MANDATORY; positional refused (100 §2). Names are types-not-values, gone by codegen (100 §1). Loops bounded, strict joins, single tail return, `while`/`break` refused by policy (100 §4). Adoption zero-copy, C-contiguous only (100 §3).

**`over` (né vmap).** SIMT-shaped: adds one lane coordinate, weaves it into named-capture accesses; intermediates stay scalar; control flow needs zero machinery; cross-lane collectives "have no home in a woven representation" — recorded loss (110 §1). Composition = `lower_handle` re-entry with merged context (010 ledger step 13, 010:996-1000). 130 §2-4.3 upgrades the claim: weaving is *premature lowering*; `over` should emit `for {kind:"map", axis}` IR — but that is the **next installment** (020 step-14 row, 020:324: "remaining: … over-emits-map-IR"), while the ledger simultaneously says "`over` needed no v2 break" on the grid (010:1020-1023). Both true (contract vs representation) but the axis is erased today — exactly the split+bind shadow charter §2.4 interrogates.

**`jvp` / `D`.** One tangent engine, per-op linearization rules registered as a surface-A column (110 §3) — the hand `JVP_RULES` table charter §2.4 collides with tensorlib's "derive, don't enumerate". `D(x)` = partials of any intermediate w.r.t. the ENCLOSING KERNEL'S params (110 §3) — precisely the design charter §7.5 doubts. Dangling deferral: `D(x, wrt=local)` is deferred to "step 13's partial-eval question" (110 §5), but the resequenced plan moved grad to step 15 / 130 stage 3 (020 Phase IV table; 130 §8.3) and neither carries the `wrt` question forward.

**`matmul` / named contraction.** Pairing the unique shared axis name, woven axes excluded first; einsum deliberately not built (110 §4). 130 §1 says it "needs to graduate from cell to tile"; `tensor.contract` generalizes it (130 §4.2, planned).

**Pipeline / combinators.** `|` composes inert values, `>` applies; flattened `Derived("pipe")` identities (020 step 3b). 020:144-145 declares "from the CPU-backend chapter on, combinator style is the house style for examples" — **not honored**: grep shows pipeline usage only in ch03/ch04 (build_chapters.py:744-1102); ch12-ch14 teach closure-factory (`make_attn`, `make_gemm`) + direct dispatch exclusively. The charter §5 load-bearing idiom is untested in the taught model beyond toy stages.

**Grid family.** `backends.c.grid`: params are integer domain coordinates, domain loop compiles into the artifact, `out=` shape is the launch domain; 16× gate met (010 ledger step 14, 010:1008-1031; build_chapters.py:4105-4117). Zero kernel lines — kind-by-registry, not kind-by-kernel.

**Tensor dialect / tile family (planned).** `tensor.map/reduce/contract/slice` as satellite ops defaulting to `core.for` decompositions; named comprehensions; memory spaces, tokens for `stage`/`barrier`; the §5 GEMM centerpiece (130 §4-5, all [planned], stages 2-4 of 130 §8). Governing principle: "Named contract/reduce/map are the working representation until target selection" (130 pre-§1) — canon's own statement of the no-unlowering convergence.

**Events / refusals.** Counts exact and always-on, detail sampled and armed (120 §2); `forbid`/`expect` as CI performance assertions (120 §6.4); caps are tripwires, satellites must ask for seams (010 §6.1). Refusal texts frozen by contract tests (charter §9 names `test_refusal_contract.py`).

## Canon↔taught↔charter divergences (summary)

1. **The step-14 attention gate mandates the violation**: 020:324 specifies "softmax as loops"; the taught flagship writes `for s in range(S)` twice (build_chapters.py:4078-4083). This is not sample sloppiness — the plan *requires* the raw-extent-loop form, contradicting both charter §2.1's level-0 loop principle and 130's own working-representation principle. The comprehension replacement exists in canon (130 §4.2) but is unscheduled relative to the gate.
2. **130's PROPOSED status is stale**: stages 1 and half of 2 landed and are canonized in the 010 ledger (010:984-1035); 110's header says its deferrals are "resolved" by 130 §7. A reader cannot tell landed from planned inside 130.
3. **The workhorse has no status**: per-element host dispatch is the ch14 baseline (build_chapters.py:4108-4113) and the ray-march verdict calls it "the wrong granularity" (010:940-942), but no canon doc assigns it first-class/debug-only/removed — charter §1's explicit obligation is unmet in canon.
4. **Collectives held in three positions**: deferred woven-axis ops (110 §1, §5), named tile ops `reduce/scan` (130 §6.1), and tensorlib/charter §2.2's "collectives are not ops — read off the algebra by a cost pass". Unadjudicated.
5. **Backend parity drift in 130**: `contract` "≡ tensor cores", "maps 1:1 onto `mma.sync`" (130 §5); stage 4 is "CUDA + tiles … Metal follows" (130 §8.4); WebGPU/subgroup limits absent from the tile plan — against charter §2.1 equal-citizens and 010's own three-target diagram (010 §0).
6. **Kernel kind two-homed**: `kind` on Handle (010 §2.4, `@jit(kind="gpu.tile")` in 130 §5) vs kind-by-registry-install (grid, build_chapters.py:4033). Canon states both, adjudicates neither (charter §2.4 obligation).
7. Minor: 020's "combinator house style" vs taught practice (finding below); `D`-wrt deferral orphaned by resequencing (digest above, not a finding for space).


## 1c. Concept inventory — tensorlib

# Tensorlib inventory (critical-assessment run, tensorlib half)

All paths relative to repo root `/Users/nehal/src/pdum_dsl/.claude/worktrees/bmap-exploration/explorations/tensorlib/` unless prefixed.

## (a) Concept inventory

**Layout (core family).** Claims: an affine map + box domain — `loc(coords) = offset + sum(stride_d * i_d)`, raw coordinates in half-open domains — is the "semantic ground truth of the whole library" via the `get_loc` contract (layout.py:1-19; DESIGN.md §2, §8.1). Named dims (identity by name, D5), byte strides with zero=repeat and negative=flip (layout.py:90), no permute ever — order-free identity is `canonical()`, export order a parameter of materialization (DESIGN.md §4; README.md:29-31). White-box closed family: a compiler composes/simplifies layouts, never reverse-engineers (DESIGN.md §2 "Layouts are white-box"). Analyses: footprint, injectivity, overlaps, `alignment(*t)` — a diagnosis quoting the fixing primitive per item, never applied automatically (D17, DESIGN.md; README.md:25-31). Extensions on the same object: linear-form guards + fill (guarded.py; README.md:33-45), exact ℚ charts/units (D7-D13, chart.py, units.py), categorical labels (D18), axis identity (D14), and machine binding `Dim.level` (layout.py:95; PLACEMENT.md D-A). One dataclass carries strides+guards+charts+units+placement together — PHILOSOPHY.md:87-99 claims this combination is the distinctive contribution.

**Tensor.** Buffer + Layout + DType (DESIGN.md §1; tensor.py:42); every op returns a new tensor sharing the buffer (DESIGN.md §3). Carries `carrier` (bool/int/rat/real/complex — "the algebraic object values approximate"; dtype demoted to representation/cost, "never semantics") and `value_units` (tensor.py:282-289; COMPUTE.md §2b). `alignment`/`aligned` implement the pointwise precondition (tensor.py:358-363).

**iota / FunctionalBuffer.** Coordinates as data with no memory: an exact-rational read function, closed under every view op by construction ("layout ops cannot destroy iota-ness") — masks, positions, one-hots become structurally free across all cost models (README.md:66-70; COMPUTE.md §2c; CONCERNS #18).

**Markers / mdsl.** Compute `f`s are declared markers, not callbacks (README.md:70-71). mdsl.py's Node schema (Arg/Const/Prim) is the declared stability boundary, importing nothing from pdum.dsl — "swapping producers can never force a rewrite of consumers" (mdsl.py:1-17). `defmarker` traces a plain lambda; partials are DERIVED by tree rewriting so "the hand-maintained gradient table stops growing" (mdsl.py:18-25). `defreducer` declares structured-state reducers (lift/associative combine/init/project); associativity is declared, not verified (mdsl.py:26-33). Content-addressed names `m_<digest>` make the registry "a cache keyed by structure, which is the main repo's build-in-a-loop philosophy" (CONCERNS #22). signatures.py propagates carrier/unit facts; `exp` of micrometers refuses; `grad` infers `target_unit` (README.md:80-82; CONCERNS #16).

**Program / IR.** Linear SSA — (var, op, operands, params), no branching, no phi; markers by name so programs stay data (ir.py:1-30). Leaves input/const/iota; layout ops as thin adapters through ONE dispatch table shared by `run` and `infer` (ir.py:399-403); `materialize` is the one copying op (ir.py:19-23; CONCERNS #21); `fold` is the single structured combinator — tensor-state scan whose step is itself a Program with a fixed-layout carry contract (ir.py:159-181; LEVELS.md:143-159). `infer` propagates layouts only (ir.py:348).

**grad.** Reverse-mode AD as program transformation: the generated gradient is ordinary IR, "run it, inspect it, differentiate it again" (autodiff.py:1-9). Scalar targets seed 1; non-scalar require explicit seed — "silent ones-seeding is a footgun we refuse" (autodiff.py:12-15). Gradients carry the primal's charts/labels/placement by restamping every contribution (autodiff.py:16-24, 168-172, 295-313). Layout-op adjoints per the COMPUTE.md §7 table (repeat†=reduce, slice†=pad, decimate†=zero-stuffing...), each validated by finite differences (README.md:122-133). Primitive pointwise partials are a hand-written if/elif chain over ~15 markers with a loud refusal for unknowns (autodiff.py:225-289); composite markers derive (autodiff.py:274-283); composite reducers get BPTT-as-generated-IR (autodiff.py:42-45, 330-400); fold adjoints derive by self-application, with `fold_segments=K` (Chen uniform) and `fold_slots=S` (binomial revolve) as schedules over the same certified pieces (autodiff.py fold_rule docstring ~403-420; CONCERNS #27) — "strategy and correctness factor" (PHILOSOPHY.md:67-71).

**Transforms (L1 optimizers).** `dce(prog, keep)`: requested-gradients pruning by reachability — "freezing a layer is just a smaller keep-set" (transforms.py:1-9). `checkpoint`: min-cut saved-set with exact byte capacities; iota/const cost 0, views cost ∞, reduce/scan/fold recompute-banned (a POLICY, not a cost model — CONCERNS #27); recompute duplicates break name=value (`v.rc`) (transforms.py:10-33).

**Memory / opcount (cost models).** Each cost model is a second denotation over the unchanged program (PHILOSOPHY.md:27-32; LEVELS.md:17-22). `peak_memory`: layout ops are zero-byte aliases, closed forms free, THE SCHEDULE IS AN ARGUMENT; deliberately coarse (uniform 8-byte itemsize) (memory.py; CONCERNS #25; README.md:96-104). `ops_count`: exact per-primitive Counters; MAC is a recognized FUSION (mul consumed solely by reduce-sum), cost weights a separate machine property (opcount.py; CONCERNS #23).

**Placement (L3-lite).** Machine is DATA (tree of levels); the IR binds dims to level names only (placement.py:1-8; LEVELS.md:11-16). NO collective ops: `traffic` reads all-reduce/all-gather off reduce/merge applied to bound dims; distribute/shard-view free (placement.py:9-18; PLACEMENT.md D-B). Erasure invariant: unbinding yields the identical denotation, bit-exact on the Megatron flagship (PLACEMENT.md "erasure invariant"; zoo/megatron.py). Unmodeled cases refuse loudly (placement.py:19-22). Placed backward landed: gradients carry bindings; Megatron joint shows 6 unfused vs Megatron's 4 fused all-reduces; data parallelism falls out as repeat†=reduce (PLACEMENT.md "Placed backward"; CONCERNS #28). NOTE: placement.py:21-22 still says "Forward programs only in v1 — gradients do not yet carry bindings," contradicted by autodiff.py:168-172/295-313 — stale docstring.

**Zoo.** L0 surface programs each pinned to an independent pure-numpy denotation; parameters are not special ("weights and activations are both just inputs"); spanning by mechanism: gpt2, llama_block (RoPE without splits, GQA as repeat), attention variants incl. the flash defreducer with DERIVED backward, heat2d, charted staggered FDTD, megatron_block (zoo/__init__.py:1-21). Recorded boundaries: MoE routing/top-k, KV-cache decode (mutation), dynamic shapes (zoo/__init__.py:19-21; LEVELS.md:140-141).

## (b) Ladder status and the L4 questions

Per LEVELS.md:33-70 and order-of-attack:160-188: **L0 (denotation)** exists — IR + AD + mdsl + fold + zoo. **L1 (footprint)** landed — peak_memory, dce, min-cut checkpoint, fold_segments, fold_slots/revolve. **L2 (storage/bufferization)** deliberately deferred — "needed for exact reuse, not for placement reasoning" (LEVELS.md:180-182). **L3-lite (placement)** landed per PLACEMENT.md, incl. placed backward. **L4 (kernels)** queued — the stream is paused here; the assessment report is the design brief that reopens it (charter §8). **L5 (schedule)/L6 (microkernel)** unstarted; structural claims only (L6 cost questions are layout queries; warp cooperation is reduce/scan over a lane-bound dim — LEVELS.md:71-83).

**K-A..K-F verbatim-condensed (LEVELS.md:190-217):** K-A — what is a kernel in the IR: grouping annotation (erasure-preserving, like placement) vs region op with a body (like fold); Lean leans annotation — "kernels change cost, not meaning." K-B — tiling = the same split+bind move one tier down (sm/warp/lane); does anything NEW appear, or is L4 = L3's mechanism + a per-level capacity constraint? K-C — legality ≈ convex instruction sets (no dependency in-and-out); objective: minimize parent-memory traffic subject to child capacity (red-blue pebbling); occupancy a constraint/modifier; manual fusion directives first, search later. K-D — flagships: flash attention (measured HBM-traffic win), fused stencil chain, tiled matmul as canonical capacity-constrained case. K-E — cost plumbing: per-kernel footprint = peak_memory with local=level sizes; per-kernel traffic = the materialization boundary's bytes; how do they compose into the L5 timeline? K-F — kernel boundaries define what materializes; bufferization consumes them — which forces L2's ordering (after fusion decisions, before/alongside the runtime plan).

## (c) Convergences and collisions with pdum.dsl canon

**Convergences (charter §8's four, verified):** (1) D17 diagnosis-never-surgery (PHILOSOPHY.md:42-46; DESIGN.md D17) ≈ pdum.dsl's frozen refusal contract ("loud-refusal guards ARE the design's error-message contract" — tests/test_refusal_contract.py:1-4). Verified — though mechanisms diverge: tensorlib's alignment recipes are human-facing strings (CONCERNS #7), pdum.dsl's messages are test-frozen; one-voice UX (charter §7.8) is open. (2) "Lowering must PRESERVE reduce/scan structure on machine-bound dims" (LEVELS.md:79-83) ≈ 130's "Named contract/reduce/map are the working representation until target selection" (docs/design/130:28-30). Verified as principle — but see collision F1 below: the streams disagree on whether *contraction* is stated or recognized. (3) Content-addressed marker registries (CONCERNS #22, explicitly citing "the main repo's build-in-a-loop philosophy") ≈ the type-keyed/content-addressed cache (docs/design/010:104,725-746; 024:165-175). Verified, and it is a *declared* adoption, not independent. (4) "Names first, order never" (PHILOSOPHY.md:48-52; DESIGN D5) ≈ mandatory name-indexing, positional refused (docs/design/100:45; stdlib/arrays.py:289). Verified at the surface — but see collision F3: pdum.dsl's `NamedArray` type identity is order-carrying. Additional declared convergences: COMPUTE.md:35-37 plans program composition "with the pdum.dsl pipeline machinery" (`Pipeline` exists — src/pdum/dsl/combinators.py:180), and mdsl's zero-import Node schema is built for pdum.dsl frontends "without any rewrite" (mdsl.py:5-16; README.md:84-85).

**Collisions (detailed in findings):** F1 contract-stated vs MAC-recognized; F2 rule-table growth polarity (JVP_RULES open registration vs tensorlib's frozen-primitive + derived-composite discipline — noting tensorlib *also* has a hand table, autodiff.py:225-271, so the honest adjudication is growth policy, not existence); F3 order-carrying `NamedArray` type vs `canonical()` order-free identity (cache-splitting risk the charter §7.7 fingerprint sketch is supposed to fix); F4 the mutation triangle (grid `out` surface vs pure-values-storage-at-L2 vs both streams' KV-cache deferral); F5 tensorlib's sufficiency thesis and zoo are ML+PDE-scoped by construction, colliding with breadth-is-identity; F6 stale placement.py docstring inside frozen evidence. Naming frictions worth one line each: 130 says "map," tensorlib says "pointwise" (COMPUTE.md §1 Naming); `over` (bind axis to launch domain) and `bind`/`Dim.level` (bind dim to machine level) are the same split+bind move under two names (charter §2.4; PLACEMENT.md D-A) — the joint system should have one binding concept; tensorlib `fold` has no pdum.dsl counterpart yet (combinators.py has only Pipeline), so the name is free but the loop-semantics gap 130 §1 records ("no way to say this loop's iterations are independent") is exactly what fold/reduce/scan markers answer — the integration seam, not a collision.

## 1d. Behavior spec — what the tests pin

# BEHAVIOR-SPEC INVENTORY (joint system)

## Stream 1: pdum.dsl (`tests/`)

### Frozen refusal messages — `test_refusal_contract.py`
Docstring (:1-4): the battery exists so "a refactor that turns a designed refusal into an AttributeError fails HERE, by name." Pinned match-strings (all `MissingRule`/`VerifyError`/generic `Exception`):
- `over` lane not i64 → "trailing i64 lane" (:33)
- `D` with no float params → "none in this build" (:42); >1 positional → "exactly one positional" (:51); integer param → "float params" (:58)
- `matmul`: keyword arg → "positional only" (:72); un-Named operands → "BY NAME" (:84); two shared names → "ONE shared axis" (:100, with note the duplicate-output-name guard is unreachable/deleted); wrong index count → "output indices" (:112)
- Derived value with no build rule → "no build rule for Derived template 'over'" (:152)
- no return → "body never returns" (:161); inline arity → "takes 2 args, got 1" (:174)
- **Pinned NON-refusal**: matmul extent mismatch between Named operands returns 3.0 — "documented UB for mismatch, which is the 100 posture" (:129-137). A silent wrong answer is frozen as spec.

### Grid family contract — `test_grid.py`
- One dispatch fills the whole domain; a new domain shape re-specializes but **out never enters the key** (:38-39 `no_compile` comment).
- Allocation follows the kernel's kind (i64 kernel → int64 out, :77-80); out from shape tuple (:42-49).
- Launcher enforces the artifact's carried contract: `VerifyError` on dtype, non-C-contiguous, rank (:86-91); kernel without a domain contract (a `jvp` result) → "no domain contract" (:107-113). Docstring: "every case below was silent corruption" (:68-69).
- `over` = one more coordinate; nested `over` = two (:52-64, :93-105).
- Batched attention oracle vs numpy (:137-145); **the step-14 gate: one grid dispatch ≥10× per-lane scalar dispatch**, retry-once noise policy (:148-173). Note: charter §2.4 calls it a "16× gate".
- Per-lane scalar dispatch is the gate's baseline and the cell-by-cell oracle (:159-166) — the charter-§1 workhorse, exercised but its status unpinned.

### Argument arrays — `test_array_args.py`
- Arg arrays mirror captures: positional, Named+isel, mixed with captures, C backend (:25-64). **Rank-generic cache hits pinned**: new shape (:32-33) and bigger matmul operands with the same names (:76-77) hit under `no_compile` — extents/trip counts are staging values, not identity.
- `jvp` with an array arg **refuses** ("float args", `VerifyError`, :91-92) — array tangents are a documented 110/130 deferral, refusal not silence.

### JVP table — `test_jvp_rules.py`
- All 17 rule-exercising kernels (mul/div/pow-const/sqrt/exp/sin/cos/abs±/min/max/floor/tuple/int-cast/branch) run through `jvp` and match central finite differences to 1e-4, primal bit-equal (:97-112). Rationale: "line coverage lies about a rule table" (:1-7).
- `POINT = (1.7, 2.3)` is deliberately "away from kinks" (:94) — **at-kink subgradient choice is NOT pinned**.
- Varying exponent `a**b` refuses (:115-121).

### Dispatch erasure — `test_traced_dispatch.py`
- Anti-drift gate (120 §5): dark == untraced == traced across a matrix including Derived wrappers (`over`, `jvp`) (:90-99); phase set exactly `{dispatch.probe, extract, pack, launch}` (:16, :99); miss path emits `spec.miss, spec.compile, lower, rewrite, render, artifact.compile` with nesting (`lower` deeper than `spec.compile`) (:136-141); `out=` flows identically when traced (:144-156).

### Perf gates — `test_bench.py`
- Warm hit path: alarm printed >5µs, **assert fails ≥40µs** (docstring says "fail 10 µs" with "4x CI margin") (:39-49).
- `instrument` phases sum to total within 50µs and must never touch the record's seams post-120 (:52-67); timeline HTML is static, no `<script>` (:70-74); GPU timeline spans exactly `[encode+submit, gpu, readback]`, readback > 0 (:90-97).

## Stream 2: tensorlib (`explorations/tensorlib/tests/`)

### Adjoint validations — `test_autodiff.py`
- Every adjoint rule validated against finite differences via one `check()` harness (:14-22): pointwise, slice/pad, flip/shift/rename, repeat/select, split/merge, window, stencil (fill cotangent discarded :178-191; taps fully outside source → zero :353-360; dilation :194-200), decimate (untouched phase gets zero, :215-218), diagonal (disjoint → zero grad, :363-374), reduce mean/max, scan, matmul, softmax-CE with analytic checks (:295-345).
- Contracts: non-scalar target without seed → `ValueError` (:276-277); seed name collision → "collides" (:412-413); unreachable inputs get `None`, not zero (:283-292); unknown differentiable marker → `NotImplementedError`, "not silent zero" (:390-405); `Instr` params snapshotted, mappingproxy refuses mutation (:416-425); second-order differentiation works by re-grad of the joint program (:428-441).
- **Pinned caveat**: reduce-max tie gives the FULL cotangent to every tied element ([1,1,0], :377-387).
- Gradients carry primal metadata: charts (:449-466), split/select chart compensation accumulates (:469-489), labels (:492-503), value units incl. `target_unit` quotient (:506-524).

### Compute reference — `test_compute.py`
- Alignment refusal **quotes the fix verbatim**: "slice(x=(1, 4))" — "the D17 recipe" (:12-17).
- Presentation order carries no meaning (D5, :31-35); guarded operands read as filled (:39-43).
- The canon as programs: matmul = repeat·mul·reduce, conv1d = window·mul·reduce, padded conv via stencil guard, softmax, causal mask via iota (:85-136).
- Iota closure invariant: `FunctionalBuffer` with `data is None`; layout ops rewrite layouts, never buffers (:179-207); physical iota is exact rationals cast at read (:210-221); carrier inference/threading (:228-254); reverse scan = flip∘scan∘flip (:283-289).

### Placement / erasure — `test_placement.py`
- `bind` is metadata riding through views and compute; "values = erasure" (:19-27). Machine-bound dims must be chartless and label-free (refusals, :30-34). Placement mismatch refusal quotes the `bind(g='gpu')` recipe (:37-45).
- **Erasure invariants**: placed megatron == erased, rtol=0 atol=0, forward (:48-55) and all gradients (:139-148); erased program communicates nothing (:67-68); gradients carry their primal's placement (:151-158).
- Traffic pass: exactly two all_reduces of 192 bytes (:58-65); merge of a bound part = all_gather (:78-87); free distribution costs nothing (:90-99); refusals — slice of machine-bound dim `NotImplementedError`, missing level `KeyError`, extent exceeds mesh `ValueError` (:102-111); α-β time model (:114-119); training-step backward = 6 collectives (unfused; fusion "a recorded later optimization", :161-177); data-parallel grad sync falls out with no special case (:180-206).

### Skim of the rest
- `test_zoo.py`: every zoo entry pinned to its numpy denotation rtol 1e-9 (:31-36); **flash == naive, forward AND backward, the backward DERIVED from the declared combine — "no hand rule anywhere"** (:66-79); FDTD grads keep staggered charts (:82-91).
- `test_mdsl.py`: Python control flow in markers refused (`TypeError` "where", :50-52); composites differentiate automatically via tree rewriting (:95); defmarker refuses primitive names (:357); multi-operand scan misalignment refused "aligned" (:225).
- `test_fold.py`: carry drift refused "state layout" (:302); final must be a carry (:306-307); segmented and revolve adjoints match store-all bit-for-bit with divisibility refusals "divide" (:348-440); revolve knob exclusivity "not both" (:485-494).
- `test_signatures.py`: unit mismatch / exp-of-dimensioned refuse statically and at runtime (`SignatureError`, :66-92); zero is unit-polymorphic (:94); content-addressed defmarker dedupes (:143).
- `test_transforms.py` (tensorlib): DCE keeps exactly the requested keep-set; banned ops read by backward are saved; DCE∘checkpoint compose (:30-133). `test_memory.py`: schedule moves the peak, topological refusal (:55-71), gpt2 backward > forward (:84). `test_regressions.py`: bool coordinates refused (:74-78), alignment diagnoses disjoint domains and flipped operands (:13, :180). `test_guarded.py`: every layout op composes through guards; one fill per tensor (:185-188).

## Cross-stream coverage gaps (behavior relied on, unpinned)
The joint seam (our syntax → tensorlib Program) has zero tests; kink semantics diverge between streams; refusal voice/taxonomy is per-stream; out-aliasing and per-step parameter mutation are unpinned in both; perf-gate numbers drift between docs and asserts. Detailed in findings.

## 2–3. Hierarchy verdict and the calling-convention matrix

# Hierarchy verdict — charter §2 (deliverables 2 and 3)

## Headline

Both axes are affirmed as structure; the joint system is convicted at their seam. Axis one's three levels stand: the device-function/kernel conflation is ruled a **feature**, contingent on a validated launch boundary that does not yet exist. Axis two's ladder is affirmed by construction in tensorlib but is **false as a description of the joint system** — two IRs exist and canon 130 plans a third path; the fork, not any single concept, is the gravest fact. The synthesis survives every probe: the tiling language is a calling convention plus vocabulary, not a language. `over` is reabsorbed as surface syntax for split+bind. The JVP table stays under tensorlib's growth polarity. The step-14 sample and its gate are convicted; their principle is affirmed.

---

## §2.1 Axis one — languages and calling conventions

### Level 0 — device functions: AFFIRMED, three amendments

The level exists, as a **contextual role** of `@jit` Handles: a captured Handle resolves as a callee and inlines (lower.py:135-137, 143-159 — read this session), the same object dispatches from host (capture.py:110-119). Affirmed as a level of the hierarchy.

- **Amendment 1 (universality is currently scalar-only).** "One device-function syntax usable unchanged inside any kernel type" holds today only over the scalar-expression core. Probe A: a fragment kernel cannot return a color (tuple results refuse on WGSL; in-kernel record construction refused); Probe E: records are capture-only, whole-record expressions crash with internals-speak; and `D` is family-incompatible in shipped code (grid coords i64 at c.py:299 — read — vs the shader family's float coords, CODE inventory F4). The **shared struct system named in §2.1 is the unbuilt prerequisite of universality**, not an accessory. Universality: affirmed as norm, in breach.
- **Amendment 2 (no-extent-loops needs an enforcement home).** The principle is AFFIRMED — Probe C shows every descent artifact (chunk fingerprint, rewrite chain, target selection) is impossible from the loop form; Probe E's audio IIR shows the pathology (O(N²) per-lane recompute) admitted rather than refused. But nothing enforces it: the flagship writes `for s in range(S)` twice (test_grid.py:120-123 — read) under a gate that mandates that form (020:324, canon inventory). Enforcement must be staged behind the comprehension/reduce spelling (130 §4.2) with an explicit debug-grade escape; refusing before the replacement exists would strand the only spelling that works (finding V5).
- **Amendment 3 (device functions are not a declared type).** No marker distinguishes a device function from a kernel at the def site; role is assigned entirely by context. Kept (see §2.4a) — but the launch boundary must become the one explicit, validated marker (§2.4a/b).

### Level 1 — host-visible kernels: AFFIRMED, three amendments

Real kernel kinds, not invented ones: per-cell scalar and whole-domain grid compute both exist for the same Handle (test_grid.py:20-24, 148-173 — read), fragment exists in the demo/wgsl kinds (batteries.py:29-30, wgsl.py:459-460 — read), vertex and tile are [proposed]/[planned] respectively. Affirmed.

- **Amendment 1 (kind is a calling convention, jointly owned, currently unvalidated).** Routing is registry data (registry.py:107, 112-124, 133 — read); the Handle's `kind` is an unvalidated free string (capture.py:149-152 — read: "capture is agnostic and does not validate it"), stays `"device"` under grid dispatch, and **unknown kinds silently route to the default backend** (registry.py:122). Ruling in §2.4b; finding V4.
- **Amendment 2 (equal citizens: affirmed as norm, in breach).** The 130 tile plan is CUDA-first with WebGPU absent (Probe C §5 parity table; canon divergence #5); `D`/fwidth already refuses on one compute family and works on another (F4); and the founding domain — vertex/render — is the only roster domain with **zero scheduled installment** (Probe A: no graphics step in 020 Phase IV). Parity divergences must be declared capability-gated refusals (Probe C's refusal sketches endorsed), never discovered.
- **Amendment 3 (tiling kernels: "distinct kind" survives; "distinct language" rejected).** Per the §2.3 ruling: the tile surface rides the same frontend machine (capture, typeof, Literal-keyed specialization — Probe C §1c) and must emit into the same laddered representation; what is distinct is the calling convention (tile-coordinate params + `out=` domain, the grid contract one level down) and vocabulary. Whether `kind="gpu.tile"` sits on the Handle or the registry resolves per §2.4b.

### Level 2 — the assemblage layer: AFFIRMED (existence, by construction); surface = qualified dissolution

Existence was answered by tensorlib's construction (charter §2.1 already retired the question). The **surface** ruling, on Probe D's evidence: **dissolution AFFIRMED for the defining side** — the transformer model needed no new binding form (closure factories over Named weights, contract-by-name, `over` for batch; Probe D P1), and the using side's identity discipline (type-only keys, live-knob captured floats) is pinned end-to-end (Probe D P7). **AMENDED for the training step**: two operators are missing, both at existing seams, neither a new language — (i) a reverse-mode surface operator whose lowering is tensorlib `grad` + `dce` (no reverse operator exists in pdum.dsl at all; transforms.py:261-263 read — only jvp/D); (ii) n-ary/pytree composition — the unary pipe (combinators.py:262-263, 271-272 — read) cannot express gradient clipping or Adam's multi-tensor DAG (Probe D P5/P6; finding V8). One-invocation-many-dispatches remains a representation-level truth with **no executor** (everything tensor-tier runs on `ir.run`; Probe D F8): the assemblage level exists in representation, not yet at runtime.

---

## §2.2 Axis two — the representation ladder: AFFIRMED for tensorlib; REJECTED as description of the joint system, ADOPTED as mandate

- **One IR, one denotation: AFFIRMED by construction** in tensorlib — L0–L1–L3-lite landed with erasure bit-exact including gradients (LEVELS.md:17-23 — read; test_placement.py:48-55 and the autodiff restamping at tl/autodiff.py:168-172, 308-313 — read). **REJECTED as a description of the joint system**: pdum.dsl's entire Node/Region IR is the ladder's bottom rung (L0 scalar denotation; footprint/storage/placement/kernels have no representation — CODE inventory, confirmed by the lowering path read this session, lower.py:176-199), tensorlib's Program is a second, unconnected IR, and canon 130 plans a tensor dialect emitting **Node IR with its own grad** — a third path (Probe D F1). Ruling: the ladder stance stands as the *mandate*; 130's tensor dialect must be re-scoped as a frontend emitting tensorlib's Program, one AD, behind a pdum.dsl surface operator. This collides with a decided position → **escalated to the human per charter §8** (finding V1).
- **Machine description is data: AFFIRMED** (LEVELS.md:9-15, placement.py:1-8 — read; bindings ride views and gradients, verified).
- **Distribution and tiling are the same move: AFFIRMED.** Probe C K-B: split+bind is unchanged one tier down; the three new things at L4 (capacity WF-predicate, ordering/tokens, materialization-boundary placement) are predicates and decisions over existing machinery, not a new representation.
- **Collectives are not ops: AFFIRMED at the representation level**, and this ruling adjudicates the three-position collective question (canon divergence #4): tensorlib's traffic-pass position wins for the IR (placement.py:9-18 — read; collectives read off reduce/merge on bound dims); 130 §6.1's named `reduce`/`scan` survive as *surface verbs* that must lower to structure-preserved reduce/scan on machine-bound dims (LEVELS.md:79-83 — read) so backends pattern-match; 110's "collectives have no home in a woven representation" deferral **closes** — the missing home was an artifact of the premature weave (finding V3): once binding is represented, cross-lane ops have their home.
- **Erasure invariants everywhere: AFFIRMED** — and pdum.dsl already practices two: Named-axis erasure (named/positional twins share one artifact, CODE inventory arrays.py:276-281) and domain-never-in-identity (test_grid.py:38-39 — read). The one blemish inside frozen evidence: placement.py:19-22 still claims "gradients do not yet carry bindings," contradicted by tl/autodiff.py:168-172 (both read) — stale docstring, low severity, already inventoried.

## §2.3 The synthesis: AFFIRMED, two amendments

**Calling conventions decouple from representation — already true in code**: the same `@jit()` Handle is a per-cell kernel through one registry and a whole-domain kernel through another (test_grid.py:159-167 — read; `install_grid` is a registry act, c.py:385-389 — read); routes are data (registry.py:107,114). **The tiling language is not a separate language** — Probe C authored both flagship descents through the one frontend machine into one representation in ≤8 named rewrites each, with tile sizes as Literal-typed specialization keys; what drives split/bind/schedule is the §3 descent pipeline (ruled by the architecture judge, not here). Affirmed.

- **Amendment 1 (the seam is the missing artifact).** The synthesis presupposes surface→laddered-IR emission; zero code path exists (tl build layer "deliberately not a frontend"; the declared adapter seam is marker-body granularity — Probe D F1). The synthesis is affirmed as design; the emission seam is the single highest-priority artifact (finding V1).
- **Amendment 2 (count the surfaces and the compositions).** Surfaces affirmed: the scalar kernel/device language [exists], the tile vocabulary [planned], the assemblage surface = ordinary Python + operators [qualified dissolution, §2.1 L2]. Compositions are where punning threatens: fuse-inline (`|`), dispatch sequencing, PSO pairing, and rewrite chains are **four distinct semantics** and must not share syntax (Probe C §8, Probe A; finding V8).

## §2.4 Mapping obligations — individual rulings

**(a) `@jit` dual role — FEATURE, AFFIRMED with CuTe's refinement made mandatory.** Roles by context are confirmed in source (inline: lower.py:135-159; dispatch: capture.py:110-119) and cost nothing; a declared host-visibility property would double every helper. But CuTe's lesson stands: the **launch boundary must be explicit and validated**. It already is explicit on the using side — the launch marker *is* registry dispatch (`h(...)`/`registry.dispatch`); what is missing is the declared side: nothing validates that a body's assumed ambient contract (fragment coords, tile vocabulary) matches the context it is inlined into or dispatched from. Amendment: keep one decorator; validate kind at dispatch and at cross-family inline (finding V4).

**(b) kind on kernel vs registry — REGISTRY WINS, AMENDED into a two-concept split.** The same Handle running as scalar and as grid (test_grid.py:159-167) proves the *execution family* cannot be a Handle property; but D's family incompatibility proves bodies *do* assume family contracts — both homes carry real information. Ruling: `kind` on the Handle = the **authoring/ABI contract** (what the body assumes), drawn from a validated vocabulary; the registry's routes = the **execution family** (kind→backend, a deployment fact). Concrete misfits convicted: kind is a free string never validated (capture.py:149-152), and absent kinds silently fall to the default backend (registry.py:122) — unknown kind at dispatch must refuse. Closes canon divergence #6 (finding V4).

**(c) `over` — REABSORBED. The standalone-transform status is REJECTED; the operator survives as surface syntax.** Evidence: the binding lives entirely in dispatch machinery — after `build_over` a lane param is indistinguishable from a data param, and the C grid backend reconstructs the domain by isinstance-unwrapping the satellite `Over` class (c.py:290-299 — read; a kernel←satellite inversion). The operator's own docstring already concedes the ruling: "the map-loop IR form arrives with the tensor step (130 §4.3) without changing this identity" (transforms.py:232-236 — read). Ruling: `over` becomes the surface verb for split+bind at launch-domain depth, emitting map/bind structure; the SIMT weave becomes one *lowering* of it; `over` and tensorlib's `bind`/`Dim.level` unify into **one binding concept** at different tree depths (LEVELS.md:25-30). The 16× gate, the weave mechanics, and the trailing-lane contract all survive the reabsorption (finding V3).

**(d) jvp / D / matmul / Pipeline — levels named.**
- `jvp`: axis-1 L2 transform, axis-2 a pure L0→L0 enrichment pass. AFFIRMED at that placement; function-centric forward lens.
- `D`: axis-1 L0 value-centric syntax, axis-2 L0. AMENDED: no `wrt` selection (all params, always — transforms.py:312-335 read) and family-incompatible (F4); the §7.5 `D(x, wrt=…)` direction is endorsed, and the charter's ambient-parameter rule (forward duals, no tape) is affirmed as D's semantics.
- `matmul`: **claims an L4-selectable contraction, delivers L0 loop denotation — MISFIT CONVICTED.** It expands to a `core.for` mul/add accumulation at lowering (transforms.py:381-396 — read); the contraction never exists as IR, violating the no-unlowering constraint both streams independently declared (130:28-30; LEVELS.md:79-83) and making §3.4's structure-keyed registry impossible (Probe C §4). Its *name-pairing front* (transforms.py:349-357) is right and survives; the expansion must become the default decomposition of a preserved contraction structure (finding V2).
- `Pipeline`: axis-1 L2 composition lowering to a single L1 artifact; axis-2 fusion-as-inlining at L0. AMENDED: demoted from "the load-bearing idiom" to the value-fusion composition along one thread — strictly unary (combinators.py:262-272 — read), load-bearing in probes only along the residual trunk (Probe D §3) and epilogues (Probe C §8); the dispatch-sequencing sense has zero code and is a separate concept (finding V8).

**(e) JVP rule table vs "derive, don't enumerate" — ADJUDICATED: the table stays, under tensorlib's growth polarity.** The honest comparison is growth *policy*, not existence — tensorlib also holds a hand chain over ~15 primitive markers (tl/autodiff.py:225-289, tensorlib inventory). A closed, frozen table over scalar primitives is defensible **because composites differentiate through it rather than joining it** — which is exactly tensorlib's discipline (frozen primitives, derived composites). Two convictions ride the ruling: (i) the "transform column is open" framing (transforms.py:204 comment — read; JVP_RULES is 8 entries plus a closed core-op if-chain) overstates and must be narrowed or the registration seam made real; (ii) at-kink behavior (abs/min/max pick a side) is an unlabeled accident — the validation point is deliberately "away from kinks" (test_jvp_rules.py:94, behavior-spec inventory). Ruling: declare the growth law ("primitive rule set frozen; new differentiable ops are composites or markers with derived partials — never new rows") and pin the at-kink subgradient choice as a commitment with tests, or change it (finding V7).

**(f) the step-14 attention sample — PRINCIPLE RIGHT; SAMPLE AND GATE CONVICTED.** The raw `for s in range(S)` loops (test_grid.py:120-123 — read) are not sample sloppiness: the plan mandates the form (020:324 "softmax as loops", canon divergence #1), so the gate institutionalizes the founding-example violation. Probe C proves the principle load-bearing (fingerprint, chain, and target selection are all impossible from the loop form); Probe E's audio IIR shows the same pathology as admission-instead-of-refusal. Ruling: rewrite the flagship in named reduce/comprehension form when 130 §4.2 lands, re-scope the gate to that form, then stage the extent-loop refusal with a debug-grade escape (finding V5). **The workhorse's status, ruled** (charter §1 obligation, unmet in canon — divergence #3): per-lane host dispatch is **debug-grade oracle** — retained as the differential-testing baseline and legitimate for scalar workloads; per-lane dispatch of lane-bearing (over/grid-contract) kernels outside test infrastructure is the convicted pattern (warn now, refuse when the grid family generalizes). It is the default family today (demo/simple_shader/python.py:182 — read); the status must enter canon (finding V6).

---

### The calling-convention matrix

Levels affirmed (rows = caller, columns = callee). Legend: **I** = inline (body fused into caller's artifact), **L** = launch (host-boundary dispatch through a registry), **C** = compose (non-call composition with its own semantics), **R** = refuse (loud, designed message), — = not applicable. Tags: [e] exists (test cited), [p] planned (020/130/LEVELS), [x] proposed (finding-bearing). The one `@jit` object fills the "device fn", "scalar kernel", and "grid kernel" rows identically — role is contextual (§2.4a); the rows are kept separate because their *callee* columns differ.

| caller ↓ \ callee → | Host Python | Device fn | Scalar kernel | Grid kernel | Vertex | Fragment | Tile kernel | Assemblage program |
|---|---|---|---|---|---|---|---|---|
| **Host Python** | — | L [e]¹ | L [e] (capture.py:110-119) | L [e] `out=`=domain (test_grid.py:29-39) | R→C(pair) [x]² | R→C(pair) [x]² | L [p] `out=`=tile domain (130 §5) | L(many) [p]³ |
| **Device fn** | R [e]⁴ | I [e] (lower.py:143-159) | I [e]⁵ | I(body) [e]⁵ | R [x]⁶ | R [x]⁶ | R [x]⁶ | R [e]⁴ |
| **Scalar kernel** | R [e]⁴ | I [e] | I [e]⁵ | I(body) [e]⁵ | R [x]⁶ | R [x]⁶ | R [x]⁶ | R [e]⁴ |
| **Grid kernel** | R [e]⁴ | I [e] | I [e]⁵ | I(body) [e]⁵ | R [x]⁶ | R [x]⁶ | R [x]⁶ | R [e]⁴ |
| **Vertex** | R | I [x] | R⁷ | R⁷ | R | C(PSO) [x]⁸ | R⁷ | R |
| **Fragment** | R | I [e] (batteries wgsl spellings) | R⁷ | R⁷ | R | R | R⁷ | R |
| **Tile kernel** [p] | R | I [p] (130 §5 epilogues) | I(body) [x]⁹ | R⁷ | R | R | I [x] | R |
| **Assemblage program** | — | C(marker) [e-tl]¹⁰ | L(select) [p]¹¹ | L(select) [p]¹¹ | L(encode) [x]² | L(encode) [x]² | L(select) [p]¹¹ | C [e-tl]¹² |

Rationales (non-obvious cells):
1. Calling any Handle from Python **is** a dispatch; the per-cell scalar family is the route (demo/simple_shader/python.py:182) — ruled debug-grade for lane-bearing work (§2.4f), so this cell is L with a status, not an R.
2. A render kernel alone has no host-callable semantics; the deliverable is a vertex+fragment pair bound into a PSO and **encoded into a pass we do not own** (Probe A: the committed `draw(target)`-owns-the-pass design is convicted; the artifact must be an encodable).
3. One invocation → many dispatches is the assemblage contract; today only the reference `ir.run` exists — the executor is the missing runtime (Probe D F8).
4. Untypeable captures fail loudly at the def site (valuekind.py:135-142 per CODE inventory; mechanism at capture.py:157-158) — host objects and Programs without a ValueKind never enter a kernel silently.
5. Kernel-ness is contextual/registry-owned: a captured Handle always inlines its **body**; the callee's domain contract belongs to the *dispatching registry*, never the body (c.py:385-389) — which is why cross-"kind" inline of scalar bodies is sound.
6. The callee's body assumes a family ambient contract (fragment coords, tile vocabulary) absent in scalar context. Today this is **unchecked** — kind is never validated at inline or dispatch (capture.py:149-152; registry.py:122) — finding V4 makes these R cells real.
7. All launches from non-host callers refuse: no dynamic parallelism in the subset; the launch boundary is host-only and is the explicit marker §2.4a demands.
8. Vertex→fragment is not a call: varyings are the vertex return record; pairing is PSO composition — a third composition semantics that must not pun on `|` (Probe A; finding V8).
9. A tile kernel may inline a scalar kernel's body as an epilogue element (130 §5's element-wise epilogues; L0 universality).
10. Compute `f`s enter programs as **declared markers, never callbacks** (tl/mdsl.py:1-25); an actual call would reintroduce the callback wall the schema exists to prevent.
11. The assemblage layer launches kernels via structure-keyed selection — §3.4's registry of certified lowerings, the artifact-cache shape that already exists (content-addressed tier, CODE inventory registry.py:190-193).
12. Programs are data: `fold` takes a Program as its step function (tl/ir.py:159-181); assemblage-level composition is program composition, no call semantics involved.

**Outside the matrix (compositions, not calls):** `over`, `jvp`, `D`, `|`, PSO-pairing, and rewrite chains are transforms/compositions. `Over`/`Jvp`/`Pipeline` wrap Handles into new dispatchable DerivedValues and inline at lowering via build rules (transforms.py:266-308; combinators.py:254-278). Four composition semantics must remain syntactically distinct: fuse-inline (`|`, unary, [e]), n-ary/pytree composition ([x], finding V8), dispatch sequencing ([x], §7.3), PSO pairing ([x], Probe A). Rewrite chains (Probe C) are program-rewrites and get their own form — never `|`.

## 4. Architecture verdict (red team)

# Architecture red team — charter §3 (progressive lowering through authored DSLs)

## Headline

§3 is **sustained as amended**. Probe C's descents prove the pipeline shape works: both flagships fit ≤8 named rewrites, one non-layout rewrite each, licenses statable. But the red team convicts the architecture on four counts beyond the pre-registered amendments: (1) the license vocabulary has no precision class, and the centerpiece GEMM itself needs one; (2) certification covers only equivalence — races, capacity, and convexity are well-formedness obligations with no home in a rewrite chain; (3) the §3.4 registry key is unsound as stated (license set, AD saved-set demand, rule versions all missing); (4) cross-model reuse presupposes an L0 normalization pass that exists nowhere and is scheduled nowhere. Amendment one: sustained with three conditions. Amendment two: sustained and generalized — the whole pipeline is a revisit loop, not a waterfall.

---

## 1. Attack lines beyond the pre-registered amendments

### A1. The license vocabulary cannot certify the architecture's own centerpiece (precision)

§3.2's equivalence relation is two-valued: bit-exact, or "equal-modulo-declared-associativity" (charter 140:264-267). LEVELS' certified-rule list is the same set: split∘merge=id, guard/slice commutation, reassociation, fusion-as-elision (LEVELS.md:109-114). **No rule in either list covers precision.** Yet 130 §5's GEMM — the artifact this architecture exists to produce — stages f16 operands, accumulates in f32, and casts the result (`acc = zeros(("m","n"), f32)` … `return acc.astype(f16)`, 130:186,193; "Accumulate-in-f32-emit-f16 is then just types doing their job," 130:174-176). An f16-mma descent from an f64/f32 L0 denotation is not equal to it under *any* associativity license — it is a different function on floats, and the delta is unbounded under cancellation. Every tensor-core descent uses reduced precision; therefore **every real descent falls outside the certifiable set as §3.2 defines it**.

The fix already exists in tensorlib and neither §3.2 nor LEVELS cites it: `Tensor.carrier` — "the algebraic object values approximate" with dtype demoted to representation/cost (tensor.py:48-53). The honest formulation: chain equivalence is stated over the **carrier (ℝ) denotation**; precision demotion is a third license class (declared, like associativity) that changes the float function but not the carrier function; the numeric tier *monitors* float divergence with a pinned tolerance and input domain, and never certifies it. **Ruling: §3.2 amended — license taxonomy = {none (bit-exact), reassociation, precision-demotion}, equivalence stated over carrier denotation, tolerances part of the license declaration.**

### A2. Certification certifies meaning; the descent's hard bugs are not meaning bugs

A rewrite chain witnesses "same denotation." But the failure modes that actually kill tiled kernels — a missing barrier, a shared-memory overflow, a non-convex fusion group — are not denotation changes; they are **well-formedness violations of the lowered form**. §3.2 has no obligation class for them. 130's own token discipline demonstrates the gap: 130:169-173 declares "a missing dependency is a *type* error (missing token), not a race," yet 130 §5's GEMM discards every token (`stage(...)`, `stage(...)`, bare `barrier()`, 130:188-190) — the design's flagship does not satisfy the design's own safety mechanism. If tokens instead become implicit (Probe C §1c's correction, which I affirm as UX), then race-freedom moves from the type system to the checker — and the checker, as chartered, only checks equivalence steps. A racy kernel can pass tier-1 numeric spot-checks nondeterministically.

**Ruling: §3.2 amended — a certified descent = (equivalence chain) + (per-level WF certificate): race-freedom (token/barrier discipline on the elaborated form), capacity (Σ staged bytes ≤ level capacity, K-B's predicate), convexity (K-C's no-dependency-in-and-out). The WF predicates are checked on the result, not derived from the chain; they are the second half of the certificate, and K-C's "legality by construction" (Probe C §3) covers only the third of the three.**

### A3. The cost oracle that steers partition cannot see the thing L4 optimizes

§3.1 step 2 is "measure→transform→re-measure against a cost simulator," and K-B's capacity WF is `peak_memory(..., local=level)` ≤ capacity. But the simulator is deliberately dtype-blind: "uniform 8-byte itemsize (the shadow convention; dtype-exact sizes later)" (memory.py:21). At L1 that coarseness was a virtue; at L4 it is a correctness bug in the WF predicate itself: an f16 staging plan measured at 8 bytes/element is over-counted 4×, so valid tilings are refused — and precision staging is half the point of the descent (A1). **Ruling: amendment, cheap but mandatory — dtype-exact sizes are promoted from "later" to a precondition of L4; the capacity WF must never run on the shadow convention.**

### A4. The §3.4 registry key is unsound as stated

§3.4 defines a certified descent as the pair "(content-addressed fingerprint of the IR chunk → verified lower-level implementation)" (140:290-292). Four independent under-determinations:

1. **License set.** A chain using reassociation + f16 demotion is valid only for users who granted those licenses. Two projects with different tolerance policies sharing one registry entry is exactly the "silent wrong answer" §1 outranks everything by.
2. **AD demand.** The forward-only attention chunk and the training chunk have the *same forward fingerprint* but different boundary contracts — flash's certified form must export the logsumexp for the derived backward (the whole point of Amendment 2; flash-vs-naive gradients: test_zoo.py:66-79). The saved-set demand is part of the chunk's interface, hence of its identity.
3. **Boundary layout classes.** The §7.7 structural-skeleton sketch covers operand skeletons, but the *stored implementation* is validated for specific layout classes at the boundary; they must enter the key, not just the guard.
4. **Rule revocation.** Content-addressing gives immutability, not revocability. If a certified rewrite rule is later found unsound, every registry entry whose chain cites it must die. pdum.dsl's spec tier has exactly this mechanism — `generation` in the key (registry.py:133-134) — but the artifact tier is keyed only `(region.key, backend.fp)` (registry.py:186-191), and the lowerings registry as sketched inherits the artifact tier's shape.

**Ruling: §3.4 amended — the key is (normalized chunk skeleton, boundary contract incl. saved-set demand and layout classes, license set, capability set, rules-generation), and the stored value is (chain with rule citations, authored region, artifact, assurance tier). The payoff claim shrinks from "across models and users" to "within a license-and-demand equivalence class" — still large, no longer false.**

### A5. Cross-model reuse presupposes a normalization pass that exists nowhere

"Every model whose attention chunk **normalizes** to the same fingerprint" (140:294-295) presupposes Program-level canonicalization. Tensorlib has `canonical()` for *layouts* only; `ir.py` contains no normalization pass (grep this session: no canonical/normalize hits in ir.py), and no LEVELS order-of-attack step schedules one (LEVELS.md:160-188). Two honest spellings of the same attention (contract vs repeat·mul·reduce; scale folded differently) fingerprint apart, and the registry silently degrades to a per-project cache — no error, just no payoff. Worse, on the pdum.dsl side nothing today even *produces* the named chunk to fingerprint: `matmul` erases contraction to a `core.for` mul/add at lowering (transforms.py:338-396). Normalization is itself a rewrite system with confluence and cost questions — an unpriced load-bearing component. **Ruling: §3.4 conditionally sustained — the registry payoff is contingent on (a) 130 stage 2's named forms landing and (b) a scheduled Program-normalization design; until both, §3.4 must be advertised as a private cache.**

### A6. The waterfall is wrong beyond AD — Amendment 2 is a special case

§3.1's order (enrich → partition → descend) breaks for AD (Amendment 2). It breaks the same way for the *other* enrichment passes. Checkpointing's min-cut runs with banned-op node capacities over the pre-partition program ("banned ops (default: reduce/scan/fold — the contractions and recurrences whose recompute would double real FLOPs)," tensorlib transforms.py:24-27); the flash rewrite replaces softmax·matmul with one reducer and changes the saved-set — the min-cut computed at step 2 is stale after step 4. Placement traffic interacts with partition identically (a fusion group spanning a machine-bound dim moves a collective inside a kernel boundary). The general law: **cost-bearing enrichment does not commute with structure-changing descent.** §3.1 step 2's own "measure→transform→re-measure" is the germ of the answer. **Ruling: §3.1 amended — the pipeline is a descend-and-revisit loop with declared invalidation edges (fusion invalidates checkpoint plans and traffic plans; placement invalidates partition candidates); one-pass is the special case when no edge fires.**

### A7. The chain has no home — and the fallback direction is the hard one

§3.2 hedges: elaboration produces a chain "**or the checker reconstructs such a chain** as a witness" (140:260-262). In current code the hedge resolves to the bad side by default: build rules elaborate straight to a final `Region` with zero step provenance (`build_pipe` inlines stages, combinators.py:254-278; `build_over` re-enters `lower_handle`, transforms.py:266-287), and the artifact tier stores only compiled programs keyed `(region.key, backend.fp)` (registry.py:186-191). Chain reconstruction is search — precisely the undecidable direction Amendment 1 exists to avoid. **Ruling: §3.2 amended — the chain is a mandatory, content-addressed *output* of elaboration (the second keyspace), stored beside the artifact; "checker reconstructs" is demoted to a migration tool for legacy descents, never the steady state.**

### A8. Tier-1-only certification overdraws the "trustworthy agent compiler" claim

Today the only operative witness is tier-1: "run both programs on random small inputs, compare" (LEVELS.md:103-104); tier-2 normalization is "Python, soon," tier-3 Lean the destination (LEVELS.md:105-114). Tier-1 on licensed rewrites is tolerance-bounded (flash backward: rtol 1e-6, test_zoo.py:79) and non-compositional — chained licensed rewrites have no error-budget algebra — and "random small inputs" misses the adversarial domain (softmax with −inf masks, cancellation, non-divisible tail tiles that only appear at real extents). Meanwhile the one associativity license the flagship leans on is "declared, not verified — property-test it" (mdsl.py:293). §3.4's "certification is what makes an agent-authored compiler trustworthy" (140:299-301) is therefore an overdraft against tier-3. **Ruling: amendment — assurance tier and input-domain coverage are recorded fields of every registry entry; the trust claim of §3.4 attaches only at tier ≥2, and flagship gates must pin adversarial input families, not only random draws.**

---

## 2. Rulings on the pre-registered amendments (Probe C as primary evidence)

**Amendment one — rewrite chains, not monolithic proofs: SUSTAINED, with three conditions.** The affirmative evidence is strong and concrete: both flagship descents decompose into ≤8 rewrites drawn from the rule set LEVELS.md:109-114 already anticipates, with exactly one non-layout rewrite each (GEMM: k-reassociation; attention: online-softmax, licensed by the declared combine that already produces a derived backward equal to naive — test_zoo.py:66-79). The monolithic alternative is achieved even by tensorlib only numerically. Conditions of sustainment: (i) the chain becomes a mandatory stored artifact (A7); (ii) the license taxonomy gains the precision class, with equivalence restated over the carrier denotation (A1); (iii) the certificate gains the WF half — races, capacity, convexity (A2). Without (i) the amendment silently reverts to reconstruction-as-search; without (ii) no real tensor-core descent is certifiable at all; without (iii) "certified" kernels can deadlock, overflow shared memory, or race while provably meaning the right thing.

**Amendment two — AD and partitioning do not commute: SUSTAINED AND GENERALIZED.** Probe C's sharpening is affirmed on its evidence: combine-introducing (saved-set-changing) rewrites must precede `grad` — the flash backward derives from the declared combine with no hand rule (test_zoo.py:66-79) — while split/bind/place rewrites commute with it — placed-backward gradients carry bindings by restamping and pass bit-exact erasure (test_placement.py:139-158 per behavior-spec inventory; PLACEMENT.md placed-backward). The generalization (A6) is the red team's addition: AD is one instance of cost-bearing-pass × structure-changing-rewrite non-commutation; checkpointing and placement traffic are the others, and the pipeline must be a revisit loop with declared invalidation edges. One residual sharpening adopted from Probe C: the naive→flash move must become a *registered named rewrite* whose license is the combine, so the fusion decision stops leaking into L0 authorship (today the zoo author chooses the flash form by hand).

---

## 3. Overall verdict on §3

**SUSTAINED AS AMENDED.** The four-step pipeline (§3.1) survives with its ordering re-stated as a revisit loop (A6). Amendment one survives with the chain-artifact, precision-license, and WF-certificate conditions (A1/A2/A7). Amendment two survives generalized. §3.4 — the payoff — takes the heaviest damage and survives only in reduced form: the registry is real and its shape does match the artifact cache, but its key must widen (A4), its reach shrinks to license-and-demand equivalence classes, and its cross-model promise is contingent on a normalization pass nobody has scheduled (A5) plus named forms that today's `matmul` erases at birth (transforms.py:338-396). No attack line rejected the architecture outright: every failure mode found has a right-level amendment inside the existing machinery (carrier semantics, generation keys, token WF, measure-re-measure), which is itself evidence that the joint system's parts are the correct parts, assembled with four missing contracts.

## 5. Syntax portfolio — Probe A: render loop

# Probe A — vertex + fragment shaders inside someone else's render loop

## Headline

Capture-as-uniforms is the probe's clear winner: closure factories + two-tier cache deliver the per-draw uniform-copy contract today, measured (test_backend_wgsl.py:53-60). Everything else on the vertex side is [proposed]: vertex arrays, varyings, MRT, instancing, and the window/draw surface appear in no plan step — the founding domain is the only roster domain with zero scheduled installment (020 Phase IV has no graphics step). Worse, the committed `draw(target)` design (070 §4) owns the render pass and submit, which is architecturally incompatible with the foreign-render-loop requirement; the deliverable must become an encodable (render bundle / draw-into-pass callable). The shading language itself fails its first test: a fragment kernel cannot return a color (tuple results refuse on WGSL; in-kernel record construction refused), and `D`/`fwidth` — the probe's value-centric derivative — refuses on any fragment signature richer than bare pixel coords. Pipe-as-fusion does not fit vertex→fragment; a third "pair-into-PSO" semantics is needed.

## 1. The aspirational program (all constructs tagged)

Workload: instanced quads with a procedural nebula/ring fragment shader (fbm noise, antialiased SDF ring), two render targets, drawn inside a host app's wgpu render loop where an imgui overlay encodes into the same pass.

### Defining side

```python
import numpy as np
from dataclasses import dataclass
from pdum.dsl import jit                          # [exists] api.py:24-35; capture.py:141-174
from pdum.dsl.stdlib.surfaces import record       # [exists] surfaces.py:77-116
from pdum.dsl.demo.graphics import Color, fwidth  # [exists] graphics.py:45-77 (demo vocabulary)
from pdum.dsl.families.render import (            # [proposed] — no such package; nearest existing
    vertex, fragment, draw_pair, adopt_vertex_buffer,   # thing is demo.simple_shader (wgsl.py)
)

# -- structured value types ---------------------------------------------------
@record                                # [exists] surfaces.py:77-116; test_surfaces.py:142-158
class Spectral4:                       # spectral color, 4 wavelength samples
    s550: float; s600: float; s650: float; s700: float
    def luminance(self): ...           # methods inline as overloads [exists] surfaces.py:107-113
# LIMIT [exists-as-refusal]: fields beyond float/int/bool refuse (surfaces.py:89-91) —
# no vec fields, no nesting. And records CANNOT be constructed in-kernel: a record class
# name resolves as a captured value -> "value here, not callable" (base_lang.py:383-384;
# no constructor rule anywhere in base_lang.py:358-392). Records are read-only freight. -> F4

@record                                # [proposed] the varyings record (vertex->fragment interface)
class V:
    clip: vec4                         # [proposed] @builtin(position); Vec exists as a TYPE
                                       #   (types.py:47; pack.py:95-96 "shader yields vec4")
                                       #   but no lowering surface produces it — dormant
    uv: vec2                           # [proposed] interpolated varying
    glow: float                        # the only field type expressible today

@dataclass(frozen=True)                # uniforms are ordinary Python objects [exists]
class Camera:
    zoom: float; cx: float; cy: float  # -> 3 uniform slots via @record (test_backend_wgsl.py:84-91)

# -- vertex shader ------------------------------------------------------------
def make_quad_vertex(cam: Camera, quads):      # quads = BACKEND-ALLOCATED, adopted (below)
    @jit(kind="render.vertex")                 # [proposed] kind; kind-by-registry mechanism
                                               #   [exists]: registry.py:112-124, wgsl.py:447-462
    def vtx(vid, iid):                         # vertex/instance coords DERIVED by the family —
                                               #   the param_types mechanism [exists]:
                                               #   wgsl.py:63-72 (f64 coords), c.py:290-299 (i64)
        corner = quads.isel(instance=iid, vertex=vid, attr=0)   # named-axis indexing [exists
                                               #   for numpy captures: arrays.py:302-332] /
                                               #   [proposed for adopted GPU buffers] -> F6
        p = ((corner - cam.cx) * cam.zoom, ...)  # capture arithmetic [exists]
        return V(clip=vec4(...), uv=..., glow=...)  # [proposed] record construction + composite
                                               #   result; today: scalar results only on this
                                               #   family (wgsl.py:98-100 renders `-> f32`) -> F4
    return vtx

# -- procedural fragment shader -----------------------------------------------
def make_nebula_frag(cam: Camera, palette: Color):   # cam SHARED with vertex -> F5
    @jit(kind="render.fragment")               # [proposed]
    def frag(v: V):                            # varyings arrive as the record [proposed] -> F7
        n = fbm2(v.uv)                         # [proposed] battery; sin/cos/floor spellings
                                               #   [exist] (c.py:350-358, JVP transforms.py:217-228);
                                               #   hash-noise u32 bit ops: 070:39-52 numeric
                                               #   policy legislates wrap [planned]
        ring = abs(length2(v.uv) - 0.6) - 0.05 # [exists] graphics.py:30-36 + batteries
        w = fwidth(ring)                       # [exists mechanism] D + sugar (graphics.py:45-55;
                                               #   test_transforms.py:123-139) BUT refuses HERE:
                                               #   v is a Record param and D seeds ALL params,
                                               #   float-only (transforms.py:313-335,326-327) -> F3
        edge = 1.0 - smoothstep(-w, w, ring)   # [exists] taught demo build_chapters.py:3809-3825
        col = palette.scaled(n)                # [exists] method -> tuple (graphics.py:66-67)
        return Mrt(color=Color(...), glow_id=v.glow)  # [proposed] MRT as record/tuple result;
                                               #   multi-destination out= is [planned] 070:99,112-113
    return frag

# -- pairing: NOT function composition ----------------------------------------
pso = draw_pair(vtx, frag,                     # [proposed] a third composition semantics: the
        targets=("rgba8unorm", "r32float"),    #   rasterizer sits between the stages; pairs into
        depth="less-equal", blend="premul")    #   a pipeline-state object, checks the varyings
                                               #   interface. Config brackets [exist, inert]:
                                               #   combinators.py:118-127; semantics-tag registry
                                               #   [exists]: combinators.py:67-71 -> F7
```

### Using side (the host owns the loop, the swap chain, the pass)

```python
quads = adopt_vertex_buffer(                   # [proposed] adoption of a foreign wgpu buffer;
    host_app.quad_buffer,                      #   090 §5 OWNED/ADOPTED + zero-copy [planned]
    layout=Layout(("instance","vertex","attr"), strides=..., offset=...))
                                               #   the descriptor is exactly tensorlib's
                                               #   affine+box Layout — the joint seam -> F6

vtx  = make_quad_vertex(camera, quads)         # [exists] phase-A capture, compile-free
frag = make_nebula_frag(camera, palette)
pso  = draw_pair(vtx, frag, ...)               # [proposed]

def on_frame(pass_enc, t, viewport):           # HOST calls us with ITS pass encoder;
    vtx2, frag2 = make_quad_vertex(Camera(zoom(t), 0, 0), quads), ...
                                               # per-frame closure REBUILD is the taught uniform
                                               #   idiom [exists]: 010:566; zero recompiles across
                                               #   value changes: test_backend_wgsl.py:53-60
    bundle = pso.record(instances=host_app.live_count, viewport=viewport)
                                               # [proposed] WebGPU render bundle, re-recorded only
                                               #   on (format, buffer-set) change; per-frame cost =
                                               #   guards + extract + pack + write_buffer [exists
                                               #   mechanism: registry.py:126-137; 070:143-147]
    pass_enc.execute_bundles([bundle])         # [proposed] — TODAY the artifact owns encoder,
                                               #   submit, AND readback (wgsl.py:381-399), and the
                                               #   committed draw(target) design still owns pass +
                                               #   submit (070:145-146) -> F2

host_app.add_layer(on_frame)                   # imgui encodes into the same pass after us: only
                                               #   possible if we never begin/end the pass -> F2
```

## 2. Named-axes stance

- **Vertex arrays: names, at the seam.** The adopted buffer's axes are named (`instance`, `vertex`, `attr`) and live in the type, exactly the `Named` pattern (arrays.py:54-62,129-141). `instance` and `vertex` are machine-bound dims in LEVELS terms — bound to the draw command's two launch dimensions — keeping axis *identity* with no charts, per the LEVELS surface discipline the charter §7.4 prior endorses.
- **Names stop at the shader interface.** Varyings are record *fields*, not axes; intermediates are scalars. A spectral value is a record on the shader side but tensorlib's own doctrine ("a struct of same-typed fields is a categorical dim in disguise", COMPUTE.md:122-125) says the same value is a categorical-dim tensor at the assemblage level — the joint system needs that to be one concept viewed twice, not a dual universe.
- **Ambient coordinates should be nameable.** Pixel/vertex coords are family-derived params with positional identity only (wgsl.py:63-72); `fwidth`'s hardcoded `d[0]/d[1]` (graphics.py:45-55) is the cost. Naming the ambient axes (`x`, `y`) and letting `D(v, wrt=ambient)` target them by name resolves F3 in the name-generic way (names as one more specialization axis).

## 3. Where capture + pipeline fit, and where not

**Capture fit: total.** Uniforms-as-Python-objects is not aspirational — it is the shipped, measured contract: closure factory + typed capture + guards + per-draw `write_buffer` (registry.py:126-137; wgsl.py:253-255; 010:684-693). Per-frame closure rebuild costs a probe + pack, no compile (test_backend_wgsl.py:53-60). The probe's core-mechanic bullet (closing over runtime values, specialized on demand) passes on evidence, with the one caveat that closing over *GPU-allocated* arrays is [proposed] (F6).

**Pipeline misfit: vertex→fragment is not `|`-as-fusion.** `build_pipe` inlines stage bodies into one artifact (combinators.py:254-278) and the planned `orchestrate` records dispatch sequences (070:125-134); a draw pair is *neither* — the rasterizer interpolates between the stages, so composition means pairing into a PSO plus an interface check. The role/semantics-tag machinery anticipates a third tag (combinators.py:67-78 even names "fragment" as a future role), but the interface check has no substrate: `FnType` is (template, env_types) with no result type (types.py:176-189), so the varyings contract is unknowable at composition time (F7). Also note pipelines are explicitly refused on the wgsl family today (wgsl.py:66-71). Frame sequencing, the other pipeline candidate, is out of scope by construction here: the host owns the loop — Probe A never sequences dispatches at all, which is evidence for the charter §7.3 split being real.

## 4. Non-preclusion checklist (what the syntax must at least not forbid)

Depth state, blending, cull, topology, target formats: fixed-function state belongs in the pair/stage config bracket ([exists, inert]: combinators.py:118-127; format-in-key precedent: registry.py:30-31, 070:148-150). Texture sampling: fragment-family feature per 070:27-28 [planned], untouched here — nothing in the probe precludes it. Offscreen readback (rows-of-floats, wgsl.py:398-402) should be explicitly labeled a debug oracle, not a render path. `@workgroup_size`/pipeline-creation-time constants generalize to specialization-time PSO state [exists precedent: wgsl.py:25-28].

## 5. Findings (detail in the findings array)

F1 graphics-scheduling (high) · F2 render-loop-ownership (high) · F3 derivative-ambient-param (high) · F4 fragment-color-results (high) · F5 uniform-sharing (medium) · F6 vertex-array-adoption (medium) · F7 raster-composition-semantics (medium) · F8 instancing-as-binding (medium-low).

## 5b. Syntax portfolio — Probe B: PDE ping-pong

# Probe B — PDE by operator splitting, ping-ponged buffers (heat2d + staggered FDTD)

## Headline

Compile-exactly-once is TRUE and was verified live: a heat2d ping-pong loop over the C grid family, written as two closure-factory Handles with swapped captured buffers, compiled once (spec misses = 1, guard_misses = 0) and matched numpy — but the enabling mechanism (cross-Handle record sharing, guards referencing only the first Handle's cells) is unpinned by any test. Refusal-first is violated at the write seam: `out=` aliasing a captured read buffer silently corrupts (verified: all-zeros vs expected shift, no error). The incumbent stance (mutation is storage-level; syntax stays pure) is SUSTAINED in tensorlib (fold + deferred L2) but VACUOUS at the joint surface: pdum.dsl has no fold, no dispatch sequencing, no path to the tensorlib representation — the user sees the swap, keys handles by buffer id, and does the alias analysis by hand. Pipeline fit: zero. Named axes stop dead at the launch boundary (Named as `out=` raises raw TypeError).

## 1. The programs

### B1 — heat2d, today's subset [all constructs verified by execution this session; script at scratchpad/probe_b_pingpong.py]

Defining side:

```python
def make_step(U_np, nx, ny, a):                    # [exists] closure factory — test_grid.py:29-33
    u = Named(U_np, ("x", "y"))                    # [exists] axis names in the type — arrays.py:54-62, 77-86
    @jit()                                          # [exists] phase-A capture — capture per test_grid.py:21-24
    def cell(i, j):                                 # params ARE domain coords (grid family) — c.py:290-299
        up = u.isel(x=i - 1, y=j) if i > 0 else 0.0        # [exists] BC as branch+ghost — branch join c.py:179-185
        dn = u.isel(x=i + 1, y=j) if i < nx - 1 else 0.0   # [exists] isel keywords mandatory — arrays.py:19-22
        lf = u.isel(x=i, y=j - 1) if j > 0 else 0.0
        rt = u.isel(x=i, y=j + 1) if j < ny - 1 else 0.0
        return u.isel(x=i, y=j) + a * (up + dn + lf + rt - 4.0 * u.isel(x=i, y=j))
    return cell
```

Using side (the host owns the time loop AND the swap AND the alias analysis):

```python
g = DEFAULT.extend(); c.install_grid(g, default=True)   # [exists] kind-by-registry — c.py:385-389
step_AB = make_step(A, N, M, alpha)                     # reads A   [exists]
step_BA = make_step(B, N, M, alpha)                     # reads B   [exists]
g.dispatch(step_AB, (), B)                              # [exists] out= adoption — c.py:315-338, test_grid.py:67-91
with no_compile():                                      # [exists] the assertion — cache.py:73-75
    src, dst, step = A, B, {id(A): step_AB, id(B): step_BA}
    for t in range(1, T):                               # [exists] raw HOST Python loop — no DSL construct
        src, dst = dst, src
        g.dispatch(step[id(src)], (), dst)              # [exists-UNPINNED] cross-Handle cache hit — finding 4
```

Verified: spec misses = 1 for the entire loop; guard_misses = 0; result matches the numpy reference (physics.py:67-74's scheme). Also verified: a per-step scalar knob via factory-per-iteration (`make_step(A, t*dt)` each step) stays at 1 compile — captured floats fingerprint by type (valuekind int bucketing at valuekind.py:47; cache key registry.py:134), and extract reads the *current* handle's captures (registry.py:136). The live-knob polarity works exactly as designed.

What is ugly on the using side: the dispatch site names no data. Reads are ambient closure state; the loop must key Handles by `id(src)`. The kernel never "declares I read U and write U_next" — reads-by-closure is the ONLY option on the grid family because any positional arg is refused when params are derived (registry.py:167-169). Finding 6.

### B2 — staggered FDTD by operator splitting, today [exists, with two honest boundaries]

Two sub-step kernels, two dispatches per host step — this IS operator splitting and it fits the grid family:

```python
h_step = make_h_step(E, H, c_)     # H' = H + c(E[i+1]-E[i]): reads H at i ONLY -> in-place out=H is safe
e_step = make_e_step(E, H, c_)     # E' = E + c(H'[i]-H'[i-1]): reads E at i ONLY -> in-place out=E is safe
for t in range(T):                  # [exists] host loop
    g.dispatch(h_step, (), H)       # [exists] BUT safety is the USER's unchecked hand analysis — finding 1
    g.dispatch(e_step, (), E)
```

Boundary 1: the in-place safety argument (writes at i, reads at i only for the carried field) lives in the user's head; the launcher checks dtype/contiguity/rank (c.py:326-335) and nothing else — heat2d written the same way (neighbor reads of the carried field) is silently wrong. Verified: right-shift kernel with `out=` its own captured input produced all-zeros, no error. Boundary 2: the Yee half-integer staggering is inexpressible — tensorlib carries it as exact charts with alignment refusing unherded combinations (physics.py:78-114, D17), while on the pdum.dsl side staggering is an index convention in comments. Multi-field single-dispatch updates are excluded: scalar cell results only (c.py:204-205, designed refusal) — finding 7.

### B3 — the aspirational joint form (what the seam should buy)

```python
def make_heat_step(alpha):                       # [exists] capture
    @jit()
    def step(u):                                 # array-valued data operand, per family contract [planned — 130 §4.1:97-99,111-118]
        ghost = lambda **d: u.shift(**d, fill=0.0)   # guarded read, BC = the fill [proposed — finding 5;
                                                     #   exists in tensorlib: physics.py:36-42, guarded.py]
        lap = ghost(x=1) + ghost(x=-1) + ghost(y=1) + ghost(y=-1) - 4.0 * u
                                                 # whole-array pointwise/comprehensions [planned — 130 §4.2:130-141]
        return u + alpha * lap                   # DPS array result; named out shape IS the launch domain
                                                 #   [planned — 130 §4.1:100-110; 020:324]
    return step

u_T = host.fold(make_heat_step(0.1), u0, dim="tm", steps=T)
    # [proposed — finding 2] dispatch-level fold with a carry contract.
    # The op EXISTS one seam away: tensorlib fold (ir.py:159-181), used by both zoo
    # PDEs incl. two-state carry {E,H} (physics.py:53-64,116-127). Ping-pong buffers
    # are then L2 bufferization's job [planned-deferred — LEVELS.md:44,180-182;
    # REPRESENTATIONS.md:40-45], and the user never sees the swap.
```

The FDTD form composes `h_step` then `e_step` inside the fold body with state `{"E","H"}` — exactly tensorlib's fold carry contract; nothing new is invented, only the pdum.dsl mirror of it is [proposed].

## 2. Named-axes stance (charter §5)

Where names exist: on captured/argument data end-to-end through indexing and contraction ([exists] — arrays.py:54-62; test_array_args.py:36-42,67-77), and they are name-generic by specialization: the factory takes the dims tuple, names ride the type, each name-set is one more specialization axis — no dual universes ([exists] mechanically). Where names STOP: the launch boundary, completely. Domain coords are positional row-major `c0..ck` (c.py:208-217); `over` lanes trail outermost-LAST by convention (transforms.py:253-257); the out array is nameless, and passing `Named` as `out=` dies with raw `TypeError: 'Named' object is not iterable` (verified; c.py:332 `tuple(spec)`). Canon's own tests need `moveaxis`/`transpose` to interpret grid results (test_grid.py:105,145). Stance: names MUST cross the launch boundary — the write side is where transposition is silent corruption (raw pointer writes, c.py:326-327), i.e. exactly where "the bug names exist to kill" (arrays.py:19-22) lands hardest. The planned seat exists: "the out array's named shape IS the launch domain" (130 §4.1:104-106). Until then: a designed refusal for Named-out, and name-bound coordinate params as the design goal. Machine-bound time dim `tm` carries a name but no chart, per LEVELS surface discipline (LEVELS.md:85-96) — consistent with tensorlib's fold requiring the scan dim chartless (ir.py:247).

## 3. Capture + pipeline fit

Capture: the probe's strongest confirmation. Buffers, extents, coefficients, and the per-step knob all enter by closure; the factory idiom delivers ping-pong and live knobs at exactly one compile (verified). This is the load-bearing value proposition working as advertised. Its dark side: on the grid family capture is the ONLY data channel (registry.py:167-169), so dataflow is invisible at the dispatch site (finding 6).

Pipeline: fit NOWHERE in this probe, and I did not force it. `|` threads exactly one value through unary stages fused into one artifact (combinators.py:262-272); Probe B needs to sequence DISPATCHES — two sub-steps with different out buffers, T repetitions, a two-field carry. Verdict on charter §7.3: value-level fusion pipeline and dispatch-level sequencing are TWO concepts; the second has no syntax, no plan (020:324 remaining list, 130 §4.3 cover neither), and is exactly the host-fold seat of finding 2. Sharing the `|` spelling is defensible only if the carry contract is explicit.

## 4. Incumbent stance + compile-once verdicts

State stance (mutation = storage-level, L2 late): SUSTAINED as representation — tensorlib's fold keeps both zoo PDEs pure and L2 is correctly deferred (LEVELS.md:180-182) — but VACUOUS at today's joint surface: no pdum.dsl construct emits or consumes a fold, so the user sees the swap and is the bufferizer, without the alias theory (which sits unused on the other side of the seam: footprints/overlaps are exact in tensorlib, DESIGN.md §2/216). The stance survives only if the seam lands (finding 2) and the raw form is kept in the subset by an alias refusal (finding 1). Boundary drawn: `out=` overlapping any read buffer is OUTSIDE the subset — refuse with "out aliases captured array 'u' — write into a second buffer (ping-pong), or wait for certified in-place"; safe in-place returns later as an L2-certified rewrite (buffer-elision), never a user gesture.

Compile-exactly-once: TRUE, verified, and it is the cache's finest hour — out never enters the key (test_grid.py:38-39), array captures fingerprint rank/dtype-generically (arrays.py:110-115), trip counts stage (test_array_args.py:76-77). But the ping-pong case rests on cross-Handle record sharing whose guards reference only the first Handle's cells forever (registry.py:83-92; cache.py:62-70) — correct today because extract re-reads current captures, unpinned by any test, and it pins the first Handle's buffers alive via the guard tuple (finding 4).

## 5c. Syntax portfolio — Probe C: the descent

# Probe C — the descent: tiled GEMM and fused attention through §3

## Headline

Both descents (tiled GEMM, fused attention) were authored as rewrite chains through §3.1. Verdict: the progressive-lowering architecture SURVIVES — each flagship decomposes into ≤8 named rewrites from the rule set LEVELS.md:109-114 already anticipates, with exactly one non-layout rewrite per flagship, each carrying a nameable license. Amendment 1 (chains, not proofs) sustained. Amendment 2 (AD × partition) sustained and sharpened into a commutation rule: combine-introducing rewrites run before `grad`; split/bind/place rewrites commute with it (placed-backward is the evidence). K-A resolves as annotation-in-IR + region-in-the-lowering-registry; K-B: split+bind unchanged, three genuinely new things at L4 (capacity WF, ordering, boundary materialization). But operationalizing convicts current artifacts: 130 §5's centerpiece GEMM violates 130's own no-unlowering principle and its own token discipline; today's `matmul` erases contraction at birth, breaking §3.4's certified-lowering registry; partition has no surface in either stream; WebGPU is absent from the tile plan.

---

## 1. Program 1 — tiled GEMM

### 1a. L0 authoring (joint system: our syntax → tensorlib Program)

```python
import numpy as np
from pdum.dsl import jit, Named                    # [exists] capture/Named: tests/test_grid.py:140-144

def make_mm(A, B, bias):                           # closure-factory idiom [exists: tests/test_grid.py:116-128]
    @jit()
    def mm():
        C = contract(A, B)                         # [planned] tensor.contract, 130:130-137; the Program-level
                                                   #   equivalent exists: zoo contract = repeat·mul·reduce
                                                   #   (zoo/attention.py:131, tests/test_compute.py per its canon)
        return relu(C + bias)                      # [planned] elementwise on tensors, 130:119-121
    return mm
```

What exists TODAY instead: `matmul(A, B, m, n)` as a scalar-cell macro that expands directly to a `core.for` mul/add accumulation at lowering — the contraction never exists as IR (src/pdum/dsl/stdlib/transforms.py:381-396). Finding 1 hangs on this.

### 1b. Partition (§3.1 step 3)

```python
chunk = kernels(prog, {"gemm0": ("Ar", "Br", "mul0", "sum0", "add1", "relu1")})   # [proposed → Finding 5]
```

No surface for this step exists in either stream: tensorlib's ir.py has no grouping construct (ir.py:92-119), pdum.dsl has nothing between whole-Handle and nothing. The keep-set shape (like DCE's `keep`) is the natural API; refusal for non-convex sets is the K-C diagnosis.

### 1c. The authored descent — the tiling DSL, corrected form

130 §5's sketch (130:182-198) is the baseline [planned]; the probe's corrected form repairs three defects found by operationalizing it (Findings 2, 3, 4):

```python
from pdum.dsl.tile import tiles, stage, contract, zeros_like   # [planned] vocabulary: 130:159-179, 180-206

def make_gemm_tiled(A, B, bias, *, TM=Literal(128), TN=Literal(128), TK=Literal(32)):
    # tile sizes are Literal-typed captures — in the specialization key, so retargeting
    # tile sizes recompiles, same source. Mechanism [exists]: LiteralType,
    # src/pdum/dsl/kernel/types.py:116-131. Its use here [proposed → Finding 8]
    # (130:176-178 instead assigns this job to the 040 §3c config bracket).
    @jit(kind="gpu.tile")                          # [planned] 130:184; kind-on-Handle vs kind-by-registry
                                                   #   (c.py:385-389) unresolved — CODE-inventory F2, inherited
    def gemm(tm, tn):                              # params are tile coords — the grid family's contract one
                                                   #   level down [exists analogue: backends/c.py:283-299]
        k = A.shared_axis(B)                       # [proposed → Finding 4] derive the contraction axis from
                                                   #   the captured types — the logic exists in matmul today
                                                   #   (transforms.py:349-357); 130 §5 hardcodes "k"
        acc = sum(                                 # [planned corrected → Finding 2] reduce-kind comprehension
            contract(                              #   (130:138-141) — NOT `for kb in tiles("k"): acc = acc + …`
                stage(A.tile(k=kb, at=tm), pad="conflict-free"),   # [planned] tile.stage + padded layout
                stage(B.tile(k=kb, at=tn)))        #   (the prime-size bank-conflict answer as a LAYOUT
            for kb in tiles(k, TK))                #   ATTRIBUTE, 130:159-163)
        # barrier/token threading is IMPLICIT, inserted by elaboration per shared
        # allocation [proposed → Finding 3]; 130:170-174 declares explicit tokens
        # but 130:189-190's own GEMM discards them.
        rowmax = reduce(acc, axis=A.free_axis(), op=max)   # [planned] warp-level named reduce, 130:212-215;
                                                   #   in ladder terms: reduce over a lane-bound dim,
                                                   #   LEVELS.md:79-83 (here as an epilogue statistic)
        return relu(acc + bias.tile(at=tn)).astype(f16)    # [planned] epilogue + precision cast, 130:174-176, 192-193
    return gemm
```

Sparsity: `A24 = A.with_format("2:4")` — a layout `format` refinement [planned: 130:162-163]; `contract` selects sparse mma only when the format is present AND the capability flag is up, else refuses (message in §5 below).

Escape hatch when the two-tier vocabulary fails: `cuda.shfl_down(x, 8)` / `metal.simd_sum(x)` — vendor-namespace ops [planned: 130:216-221].

### 1d. Using side

```python
g  = make_gemm_tiled(A, B, bias)                       # Handle [exists mechanism]
C  = tiles.dispatch(g, (), out=(M, N))                 # DPS out= IS the launch domain — the grid contract
                                                       #   generalized [exists analogue: tests/test_grid.py:144,167;
                                                       #   generalization planned: 130:103-110]
g2 = make_gemm_tiled(A, B, bias, TM=Literal(64))       # tile retarget = new specialization, same source
                                                       #   [proposed; key mechanism exists: types.py:116-131]
gb = over(make_gemm_tiled(A3, B3, bias), axis="batch") # batch = one more grid axis [exists for grid:
                                                       #   tests/test_grid.py:103-105; planned for tile: 130:197]
```

### 1e. The rewrite chain and witness (§3.2)

```
chunk (named form, the no-unlowering line):  C[m,n] = reduce(+, k) mul(A[m,k], B[k,n]); epilogue relu(·+bias)

R1  split(m, TM)                license: none — split∘merge=id under divisibility, else pad-to-tile-with-guards
                                [tier-3 rule named at LEVELS.md:109-114; guards exist: tensorlib guarded.py per DESIGN]
R2  split(n, TN)                same
R3  split(k, TK)                license: REASSOCIATIVITY of + (declared) — reduce(k) → reduce(k_blk)∘reduce(k_in)
R4  bind(m_blk→"sm.y", n_blk→"sm.x"), bind(k_in inner per backend)
                                erasure-preserving [mechanism exists one tier up: layout.py:443-457 bind,
                                PLACEMENT.md:24-27 erasure invariant]
R5  place(A_tile,B_tile → level "sm" memory), pad="conflict-free"   [planned: 130:159-168]
R6  materialization boundary: stage = explicit copy global→shared   [op exists: ir.py:19-23 materialize;
                                the PLACEMENT of it is L4's new content]
R7  reorder k_blk outermost     under R3's license
R8  fuse epilogue into consumer = elision of materialization        [tier-3 rule: LEVELS.md:112-113]

WITNESS: the chain itself — each step a named rule; tier 1 (numeric spot-check vs the L0 denotation)
always-on [exists: tensorlib run + zoo-oracle pattern, tests/test_zoo.py]; tier 2 (layout normalization,
decidable integer arithmetic) for R1/R2/R4-R6; tier 3 (Lean-certified rules) the destination
[LEVELS.md:98-114]. Exactly ONE non-layout rewrite (R3) — its license is one declaration.

CAPACITY WF (K-B's new predicate): peak_memory(kernel instrs, local="sm") ≤ Machine.level("sm").capacity
[pieces exist: memory.py:41-48,60-66 local=; placement.py Level.capacity field; the composition is [planned]]
```

---

## 2. Program 2 — fused attention (matmul + streaming softmax)

### 2a. L0 authoring

```python
def make_attn(Q, K, V, scale):
    @jit()
    def attn():
        sc = contract(Q, K) * scale            # [planned] 130:130-137
        w  = softmax(sc, axis="s")             # [exists at Program level: zoo softmax helper feeding
                                               #   zoo/attention.py:137; our-syntax form planned]
        return contract(w, V)
    return attn
```

Today's actual flagship writes the same chunk as two raw `for s in range(S)` loops over a captured extent (tests/test_grid.py:116-128) — the §2.4(f) violation. Probe C's verdict on that sample: the principle is right and the sample must be rewritten in the named form above; every downstream step of this probe (fingerprint, chain, parity selection) is IMPOSSIBLE from the loop form.

### 2b. The rewrite chain

```
chunk: sc = contract(Q,K)·scale; w = softmax(sc, s); out = contract(w, V)

R1  split(t, BT); bind(t_blk→"sm")                       [layout class]
R2  REWRITE  softmax(s)∘contract(s) → reduce(flashsm, s)
        license: the DECLARED associative combine — the online-softmax lemma carried by the
        defreducer [exists: zoo/attention.py:20-32; associativity declared-not-verified,
        mdsl.py:24-30 — property-test now, Lean typeclass later]
R3  split(s, BS)   reduce(flashsm) over s → sequential folding of per-block partial states
        under R2's SAME combine (streaming; no new license)
R4  stage(K.tile, V.tile → "sm"); boundary materialization
R5  epilogue: flashsm.project (o/den) at the end          [exists: attention.py:31]

WITNESS: R2 is the ONE non-layout rewrite; its certificate is the combine's associativity.
Numeric tier exists TODAY: flash == naive, forward AND backward, rtol 1e-6
[exists: tests/test_zoo.py:66-79 — "no hand rule anywhere"].
```

### 2c. The tiled kernel

```python
def make_flash(Q, K, V, scale, *, BT=Literal(64), BS=Literal(64)):
    @jit(kind="gpu.tile")
    def attn(tb):
        q = stage(Q.tile(at=tb))
        out = reduce(flashsm)(                     # [proposed] reducer-valued tile reduce: 130 §6.1's
            (contract(q, stage(K.tile(sb))) * scale,   # reduce(op=max) covers only scalar monoids —
             stage(V.tile(sb)))                    #   flash needs STRUCTURED state (m, l, o); the
            for sb in tiles("s", BS))              #   generalization is exactly tensorlib's defreducer
        return out                                 #   [exists: mdsl.py:24-30] → Finding 7
    return attn

b = over(make_flash(Q, K, V, s), axis="batch")     # using side: same as GEMM; heads = one more over
out = tiles.dispatch(b, (), out=(T, Dv, B))        # [exists analogue: tests/test_grid.py:143-145]
```

Inside the combine, the per-row running max lowers to a warp/subgroup max over the lane-bound dim [planned: 130:212-215; LEVELS.md:79-83] — the warp-level-primitive requirement, discharged without the user writing a shuffle.

---

## 3. K-A..K-F answered from the syntax side

**K-A (annotation vs region op): BOTH, at different homes — and the reconciliation is §3.4's registry.** The *author's* artifact is a region (a tile kernel is a `def` with a body). The *shared IR* records only a grouping annotation (kernel-id over instructions) plus a content-addressed reference into the registry of certified lowerings, whose stored value is (chain, authored region, compiled artifact). Evidence for annotation-in-IR: tensorlib's one region op, `fold`, already exacts a recurring toll on every pass — memory.py must recursively simulate its body (memory.py:104-112) and checkpointing bans it from recompute as POLICY (CONCERNS #27); making kernels region ops multiplies that toll across all L1/L3 passes, whereas annotation leaves `run`/`infer`/`grad`/`peak_memory`/`traffic` untouched and keeps the erasure theorem one line (drop kernel-ids → same denotation, same shape as PLACEMENT.md:24-27). Evidence that the region belongs OUTSIDE the IR: pdum.dsl already keeps kind out of the kernel (install_grid is a registry act, c.py:385-389) and stores compiled bodies in a content-addressed artifact tier. Lean's lean ("kernels change cost, not meaning," LEVELS.md:194-197) is CONFIRMED from the syntax side.

**K-B (does anything NEW appear?): the split+bind move is unchanged; three new things appear, none of them a new representation.** (1) *Capacity as well-formedness*: L3 checks mesh extent ≤ level count (placement.py traffic pass); L4 needs Σbytes(staged) ≤ level capacity — a new WF predicate over existing machinery (memory.py `local=` + Level.capacity). (2) *Ordering*: barriers/tokens are the first intra-program ordering construct on the ladder — nothing at L0–L3 has one except `fold`'s sequential carry, which is the precedent to generalize (Finding 3). (3) *The materialization boundary*: `materialize` exists (ir.py:19-23); L4 newly makes its placement a decision variable. So L4 = L3's mechanism + capacity WF + ordering + boundary placement. The corollary "distribution and tiling are the same move" (LEVELS.md:25-30) is AFFIRMED by construction: R4/R5 above reuse `bind` verbatim.

**K-C (legality/objective): legality by construction, not by check.** If the only door to a descent is the tiling DSL, and its elaboration *is* a chain of certified rewrites, convex-set legality is implied rule-by-rule; the checker verifies chain application, never re-derives legality. The objective (parent-traffic under child capacity) never appears in source — it lives in the gates (K-E). The syntax's obligation is the D17 diagnosis when a grouping is illegal: refuse quoting the dependency that exits and re-enters the proposed kernel. Manual-first matches §3.1 step 4 exactly.

**K-D (flagships): delivered above** — tiled GEMM (canonical capacity-constrained case) and flash attention (traffic win; the fused stencil chain belongs to Probe B). Addendum: state each flagship's win as a CI *traffic assertion* (modeled HBM bytes fused < naive) in the events.expect style — both streams already own the pieces.

**K-E (cost plumbing): the kernel annotation is the missing scope delimiter.** Per-kernel footprint = `peak_memory(instrs-of-kernel, local=level)` [exists: memory.py:60-66]; per-kernel traffic = bytes at the R6 boundary; both key off the K-A grouping. Composition into L5: each kernel yields (traffic, footprint, launches) — the same event shape the traffic pass already emits (placement.py Collective records); a timeline simulator consumes the sequence. Nothing of this appears in kernel source; it appears in gate assertions.

**K-F (L2 ordering): confirmed from the syntax side** — kernel boundaries define what materializes; bufferization consumes them; so L2 runs after partition/fusion. One caveat: the DPS `out=` contract (130:103-110, exists for grid at c.py:315-338) is a slice of L2 already committed at the host ABI boundary. Acceptable because it is confined to the host seam — but it should be *named* as such in the L2 design, not discovered later.

---

## 4. No-unlowering: verdict

The constraint is AFFIRMED and is load-bearing three times over in this probe: (a) the chunk fingerprint (§3.4) must be computed on the named form or reuse dies (Finding 1); (b) target selection — mma vs simdgroup vs FMA-decomposition — pattern-matches `contract` only if it still exists (130:28-31; LEVELS.md:79-83, independently converged); (c) R2's license is only statable against named softmax/reduce structure. The probe found the constraint violated in both streams' current artifacts: `matmul` erases at birth (transforms.py:381-396) and 130 §5's own k-loop re-erases the reduction the chain needs (Finding 2). The step-14 attention sample (test_grid.py:116-128) is the same violation at L0; the principle wins, the samples must move.

---

## 5. Backend parity (equal citizens)

| Construct | CUDA | Metal | WebGPU |
|---|---|---|---|
| `contract` f16 tile | `mma.sync` [planned: 130:203-204] | simdgroup matrix ops [planned: "Metal follows," 130:270] | **no base-spec matrix op**; subgroup-matrix is an experimental extension; the honest default is the §4.2 decomposition gate → FMA loops over workgroup memory (runs, slower) [planned mechanism: 130:130-137] — **absent from the plan → Finding 6** |
| `stage` (shared mem) | `__shared__` | `threadgroup` | `var<workgroup>` — parity holds |
| `barrier` | `__syncthreads` | `threadgroup_barrier` | `workgroupBarrier` — parity holds |
| warp reduce/scan | shfl/redux | `simd_sum`/`simd_max` | `subgroupAdd`/`subgroupMax` — optional feature, capability-gated |
| f8 / 2:4 sparsity | native | no | no — capability flags from day one (130:288-290), refusal elsewhere |

Refusal sketches [proposed, D17 voice]:
- `VerifyError: contract(f8e4m3) needs a matrix unit with f8 support — this device (wgsl) reports none; astype(f16) first, or route to the decomposed family (runs without matrix units)`
- `VerifyError: format="2:4" has no lowering on backends.wgsl — drop the format (dense contract runs) or route to backends.cuda`

Boundary drawn (exclusion as success): structured sparsity and f8 are INSIDE the subset only where a capability flag is up; on other targets they refuse — they are never silently densified/upcast. Prior family-parity scar tissue supports the rule: the D/fwidth operator is already family-incompatible today (grid coords i64 at c.py:299 vs the shader family's f32 fragcoord, dsl_reference/backends/wgsl/intrinsics.py:16) — parity divergences must be declared, not discovered.

---

## 6. Rulings on the §3 amendments (preliminary)

**Amendment 1 — rewrite chains, not monolithic proofs: SUSTAINED.** Both flagship descents fit in ≤8 named rewrites drawn from a small closed set the LEVELS assurance tiers already anticipate (LEVELS.md:109-114); each flagship needs exactly ONE non-layout rewrite (R3 reassociation; R2 online-softmax), each with a one-declaration license. The monolithic alternative would have to prove flash == naive outright — which even tensorlib achieves only numerically today (test_zoo.py:66-79). Gap the ruling attaches: the chain has NO HOME — `lower_handle`/build rules elaborate straight to a final Region with no step provenance (transforms.py:266-287, combinators.py:254-278), and the artifact tier stores only compiled programs. The chain must become a stored, content-addressed companion artifact (Finding 1's second keyspace); without it "the checker reconstructs a chain" becomes the default, which is the hard direction.

**Amendment 2 — AD and partitioning do not commute: SUSTAINED, and sharpened into a commutation rule.** Evidence both ways: (i) the flash backward is DERIVED from the declared combine with no hand rule and equals naive's gradients (test_zoo.py:66-79) — fusion-aware backward without differentiating tiled code; (ii) placed backward shows layout-class rewrites DO commute with grad (gradients carry bindings by restamping; PLACEMENT.md "Placed backward," autodiff restamp). The rule the probe proposes: **rewrites that change the saved-set (combine-introducing, R2-class) must run before `grad`; pure split/bind/place rewrites commute with it.** Pipeline order becomes: author L0 → licensed structure rewrites → AD → memory passes → partition → split/bind/place descent. What must be declared at L0: the associative combine (`defreducer`) — nothing else was needed by either flagship. Residual (Finding 7): today the flash *form* is chosen by the L0 author (attention.py:136-142), i.e. the fusion decision leaks into step 1; the naive→flash move should be a registered named rewrite whose license IS the combine.

---

## 7. Named-axes stance (charter §5 / §7.4)

- **Where names exist:** semantic dims at L0 (m,n,k,t,s,e); tile selection (`A.tile(k=kb, at=tm)`), contraction and reduce axes in the tile DSL; machine LEVEL names ("sm","warp","lane") from the Machine tree (data, not IR — LEVELS.md:11-16). Split parts keep axis identity while dropping charts/labels — the LEVELS surface discipline (LEVELS.md:85-96), already enforced at Dim construction (layout.py:95,107).
- **Where names stop:** the artifact ABI (erased into strides — the existing erasure invariant); vendor intrinsics (`cuda.shfl_down` takes counts, not names); published mma fragment layouts (positional by hardware spec — names re-enter only as documentation).
- **No dual universes:** tile kernels REQUIRE named operands (extending matmul's "pairs axes BY NAME" posture, transforms.py:348); a nameless array is named once at the seam (`Named(arr, names)`) — one adapter, not a second universe.
- **Name-genericity:** the probe's hard stance — a concrete axis-name literal inside a kernel body is a defect (Finding 4). Names are derived from captured operand types (matmul already does this, transforms.py:349-357); since names live in types, name-generic kernels specialize per name through the existing FnType machinery with zero new mechanism.

---

## 8. Capture + pipeline: where the style fit and where it did not

**Fit:** the closure-factory (`make_gemm_tiled(A, B, bias, TM=…)`) is the natural authoring unit at EVERY level of the descent — L0 model, tile kernel, even the chain's parameters; it is the same idiom the flagship test already uses (test_grid.py:116-128). Tile sizes as Literal captures ride the identical mechanism. `over` composes at host level unchanged (test_grid.py:103-105).

**Did not fit:** (a) **the descent itself is not a pipeline** — a rewrite chain composes program-rewrites, not values; reusing `|` would pun two unrelated compositions, so chains need their own (tiny) form; (b) **the kernel body barely pipes** — `contract` is binary and Pipeline is strictly unary (combinators.py:262-263, 270-272), so pipe helps only in the epilogue (`acc | add(bias) | relu | astype(f16)` would work); (c) **partition has no surface at all** (Finding 5) — the one place this probe had to invent syntax from nothing; (d) dispatch-level pipelines (§7.3) never appeared: each chunk became one kernel; sequencing belongs to Probes B/D.


## 5d. Syntax portfolio — Probe D: transformer

# Probe D — transformer end-to-end (joint system)

## Headline

The model's defining side dissolves cleanly into ordinary Python + capture: closure-factories over Named weights, contract-by-name, `over` for batch — no new model-definition surface needed. The training step does NOT dissolve: gradient clipping, Adam, and multi-tensor state need n-ary composition the unary Pipeline cannot express, and reverse mode in our syntax has no operator at all. The charter's wrt/freezing = DCE-keep-set mapping holds, but only as a two-call idiom (`grad` + `dce`) — `wrt` alone prunes nothing. KV-cache decode is INSIDE the subset today with zero new language (preallocated buffer + `out=` row-adoption + rank-generic prefix-slice recapture); the recorded exclusion is representation-level only. The gravest structural fact: canon 130 builds a second tensor IR and a second AD inside pdum.dsl while the charter mandates fronting tensorlib's Program — the joint seam is an unadjudicated fork, not a gap.

---

## 1. The programs

Repo root: `/Users/nehal/src/pdum_dsl/.claude/worktrees/bmap-exploration`. `tl/` = `explorations/tensorlib/tensorlib/`.

### P1 — model definition (our syntax fronting the tensorlib representation)

```python
# ---------- defining side ----------
def make_block(cfg, ln1g, ln1b, wq, wk, wv, wo, ln2g, ln2b, w1, b1, w2, b2):
    # weights are Named captures                      [exists] arrays.py:76-86; tests/test_array_args.py:36-41
    @jit(kind="tensor")                               # kind= string [exists] capture.py:148-152 (unvalidated);
                                                      # a "tensor" family [proposed] — no such family exists
    def block(x):                                     # x: Named array ARGUMENT (t,d)
        a  = layernorm(x, ln1g, ln1b, cfg.eps)        # [proposed] surface composite; L0 pattern [exists] zoo/gpt2.py:64 (Build-emitted)
        q  = contract(a, wq, axis="d")                # [planned] tensor.contract, 130:130-137
        k  = contract(a.rename(t="s"), wk, axis="d")  # rename in OUR syntax [proposed]; tensorlib op [exists] tl/ir.py:130
        v  = contract(a.rename(t="s"), wv, axis="d")
        sc = contract(q * cfg.scale, k, axis="hk")    # scalar-cell matmul-by-name [exists] tests/test_grid.py:116-128;
                                                      # tensor-tier contract [planned] 130:130-137
        p  = softmax(sc, axis="s", mask=causal("t", "s"))
                                                      # [proposed] surface; causal-softmax L0 program [exists] zoo/gpt2.py:72
        cx = contract(p, v, axis="s")
        o  = contract(cx, wo, axis=("nh", "hk"))      # multi-axis contract [proposed]; today refused: "ONE shared axis"
                                                      # tests/test_refusal_contract.py (:100 pin); zoo does it as one contract [exists] gpt2.py:74
        h  = x + o
        m  = gelu(contract(layernorm(h, ln2g, ln2b, cfg.eps), w1, axis="d") + b1)
        return h + contract(m, w2, axis="m") + b2     # array RESULT [planned] 130:257 "array results (DPS)";
                                                      # today refused: arrays.py:212-213 "array RESULTS are not in v1"
    return block

model = pipeline_of(make_block(cfg, **layer_params(i)) for i in range(L))
                                                      # layer stacking via | [exists mechanism] combinators.py:180-226,
                                                      # BUT only if each block threads ONE value — holds here (x alone)
batched = over(model, axis="batch")                   # [exists] tests/test_grid.py:137-145 (over of a composed handle)

# ---------- emission ----------
# block EMITS a tensorlib Program (Instr stream: contract→repeat·mul·reduce etc.)
#                                                     [proposed] — NO emission path exists. tl/build.py:1-7 is
#                                                     "deliberately not a frontend"; the declared adapter seam is
#                                                     marker-BODY granularity only (tl/mdsl.py:1-16); 130's tensor
#                                                     dialect emits pdum.dsl Node IR, not Program (130:130-134). F1.
```

The L0 target of this emission already exists and is pinned: the zoo GPT-2 Program with weights-as-inputs (`zoo/gpt2.py:30-109`), matched to numpy (digest-pinned `tests/test_zoo.py:31-36`).

### P2 — init / RNG vs the type-keyed cache

```python
rng = np.random.default_rng(cfg.seed)                          # [exists] host values; zoo/gpt2.py:30-40
params = {"L0.wq": Named(0.02*rng.standard_normal((D,H,K)), ("d","nh","hk")), ...}
```

No collision exists: seeds and initial values are host-side *values*; every kind fingerprints by TYPE (floats: valuekind.py:157-167,211; ints: valuekind.py:181-189; arrays: arrays.py:104-115 — fingerprint == typeof, value-free). Two runs with different seeds share every specialization. **Verdict: init needs nothing.** Device-side RNG (dropout, per-step noise) has no story in either stream — see F7; boundary refusal sketched in §5.

### P3 — loss

```python
loss = mean(cross_entropy(logits, targets, axis="v"), axis="t")   # [proposed] surface composites
```

L0 substance [exists]: loss-as-appended-instrs (`tests/…/test_transforms.py:20-27` tensorlib), softmax-CE adjoints validated analytically (inventory: tl tests/test_autodiff.py:295-345). Integer `targets` as a select/one-hot via iota is structurally free [exists] (iota closure, tl/ir.py:11-14) — but token-embedding *gather* is a recorded boundary (zoo/gpt2.py:3-4).

### P4 — reverse mode, wrt, freezing (the mapping, verified)

```python
# ---------- what exists (tensorlib, verified this session) ----------
jp, grads = grad(prog, "loss", input_layouts)          # [exists] tl/autodiff.py:119-128; returns joint Program + var→gradvar
trainable = [grads[k] for k in params if not k.startswith("L0.")]   # freeze layer 0
train = dce(jp, (*trainable, "loss"))                  # [exists] tl/transforms.py:48-59;
                                                       # pinned: tests/test_transforms.py:48-58 (weight-grad work GONE)

# ---------- our syntax front ----------
dloss = grad(loss_fn, wrt=capture_set)                 # [proposed] — pdum.dsl has NO reverse-mode operator
                                                       # (only jvp/D: stdlib/transforms.py:259-263); grad is
                                                       # [planned] 130 stage 3 (130:263-266) — but against the
                                                       # pdum.dsl tensor dialect, NOT tensorlib grad. F1.
```

**Captures-become-input-leaves**: the binder-seam move requires a deterministic capture-path→leaf-name law. tensorlib identifies inputs by string name (`tl/ir.py:289-296`: run keyed on `input` var names); zoo manages layer namespacing by hand-written prefixes (`zoo/gpt2.py:48-63`, `"L{i}."`); pdum.dsl identifies captures positionally with stage-index prefixes at lowering (`combinators.py:254-258`). No law connects them — F2. The prefix mechanism in `build_pipe` is the existing right-level seed.

**The wrt subtlety (verified)**: `grad(..., wrt=names)` computes cotangents for ALL forward vars and only *filters the returned dict* (`tl/autodiff.py:867-876`). Pruning is a separate `dce` the caller must know to run. The charter's "freezing is just a smaller keep-set" is true — but the keep-set lives in `dce`, not in `wrt` — F5. Checkpointing and revolve compose downstream [exists]: `checkpoint` (tl/transforms.py:117-232), `fold_segments`/`fold_slots` (tl/autodiff.py:126-139).

**Per-step updates × cache identity (verified)**: rebuilding the loss closure each step with fresh param arrays keeps `env_fp` stable (array fingerprint is type-only, arrays.py:110-111) → cache hit; guards are per-rebuild fresh cells and pass (registry.py:83-92); drift only fires on stale handles (tests/test_runtime.py:93-107). The 300-frame pin (tests/test_runtime.py:47-54) proves the loop shape for scalars; an array-capture twin of that pin does not exist (gap noted in F6's proposal).

### P5 — gradient clipping (global norm)

```python
# L0 substance [exists]: per-tensor sumsq = pointwise mul + reduce over all dims; scalar add;
# rsqrt; broadcast-scale each grad by repeat (tl/ir.py ops; Build sugar zoo/build.py:43-59)
gn     = sqrt(sum(sumsq(g) for g in grads))            # n-ary over a *list of tensors*
scale  = minimum(1.0, cfg.clip / gn)
gclip  = [g * scale for g in grads]                    # [proposed] as OUR surface — see F4:
                                                       # pipeline threads exactly ONE value
                                                       # (combinators.py:262-263, 271-272), so the
                                                       # grads-list → scalar → grads-list DAG cannot be
                                                       # written in the load-bearing idiom at all.
```

### P6 — SGD + Adam: the training loop in the candidate styles

Moment buffers are STATE. Three styles written; verdicts inline.

**Style A — functional threading (tensorlib-native; runs TODAY, reference-slow)**

```python
step, names = build_train_step(cfg)     # ONE Program: inputs = params ∪ moments(m,v) ∪ batch ∪ {lr, t}
                                        # outputs = new-params ∪ new-moments ∪ loss
state = init_state(rng)
for i, batch in enumerate(data):
    env = run(step, {**state, **batch, "lr": sched(i), "t": i+1})   # [exists] tl/ir.py:289-292
    state = {k: env[names[k]] for k in state}                        # host rebind, pure
```

Adam's update (`m̂ = m/(1-β₁ᵗ)` etc.) is pointwise IR [exists at L0]. Cost honesty: every step re-feeds params + 2× moments across the host boundary; no device-resident state exists in either stream (L2 deferred, inventory LEVELS L2) — F6.

**Style B — closure-rebuild (pdum.dsl-native; the live-knob pattern)**

```python
for i, batch in enumerate(data):
    step = make_step(params, moments, sched(i))   # rebuild = reflection only [exists] capture.py:1-11
    params, moments, loss = step(batch)           # same types → cache hit [exists mechanic]
                                                  # tests/test_runtime.py:47-54 (1 compile / 299 hits);
                                                  # multi-ARRAY results [proposed] — refused today
                                                  # (arrays.py:212-213); single scalar returns only
```

**Style C — in-place via `out=` adoption**

```python
step(batch, out=flat_params)                      # single-out adoption [exists] tests/test_grid.py:42-49,67-91
                                                  # (dtype/rank/contiguity contract-checked);
                                                  # MULTI-out and out-aliasing rules [proposed] — unpinned, F3
```

**Decisive argument**: A is the semantics (and the certification oracle); B is the *surface* the joint system should promote — it is exactly the pattern the cache was designed for — with A as its lowering; C is a storage-level optimization (buffer donation) that must arrive via L2 bufferization, never as syntax. This is the charter §7.1 incumbent affirmed, with the new fact that B's identity story is already proven for scalars and mechanically sound for arrays.

### P7 — LR schedule as live-knob

Covered by Style B: `sched(i)` is a captured float → `_ConstKind(f64)` (valuekind.py:211) → never in any key; pinned end-to-end by tests/test_runtime.py:47-54. **[exists]; the strongest concept-to-workload fit in the whole probe.** Flip side: every captured float is a live knob — `eps`, `scale` can never constant-fold without opting into `Literal`; acceptable, noted, not a finding.

### P8 — inference + KV-cache (PRESSED)

```python
K = np.zeros((S_MAX, H, Dk)); V = np.zeros((S_MAX, H, Dv))     # host-owned, preallocated [exists]
y = None
for t, x_t in enumerate(stream):
    proj_k, proj_v = make_proj(x_t, wk, wv)                     # per-token projection kernels [exists shape]
    proj_k(out=K[t]); proj_v(out=V[t])                          # ROW WRITE by adoption: K[t] is a
                                                                # C-contiguous view [exists mechanism]
                                                                # tests/test_grid.py:42-49,67-91; arrays.py:99-100
    att = make_decode_attn(q_of(x_t), K[:t+1], V[:t+1], t+1)    # prefix-slice recapture each token:
                                                                # C-contiguous [exists] arrays.py:99-100;
                                                                # rank-generic → NO recompile as t grows
                                                                # [exists] tests/test_array_args.py:30-33,76-77;
                                                                # captured int t+1 fingerprints by TYPE
                                                                # [exists] valuekind.py:181-189
    y = att(...)                                                # one dispatch per token
```

**Verdict (the pressed answer): KV-cache decode is INSIDE the subset**, expressible today in the pdum.dsl stream with zero new language. The two streams' recorded exclusions (zoo/__init__.py:19-21 "KV-cache decode (mutation), dynamic shapes") survive only as a *representation-level* rule: the per-token Program stays pure; the cache is a host-owned buffer mutated exclusively at the dispatch boundary via `out=`. Dynamic shape dissolves because extents are staging values, not identity — the rank-generic cache polarity is precisely the decode-loop feature. Two real gaps: (a) out-aliasing (out= overlapping a captured view of the same buffer) is unpinned and would be silent corruption if mishandled — F3; (b) at tensor tier the whole decode step should be ONE dispatch, which needs the executor seam — F8.

---

## 2. Named-axes stance (charter §5)

**Where names exist**: (1) the model surface — weights and activations are Named; contraction pairs by name ([exists] transforms.py:346-357); positional indexing on named arrays refused ([exists] inventory arrays.py:284-291). (2) Gradient identity — `grads` is keyed by named leaves; the gradient of `wq` carries `("d","nh","hk")` and (verified for placement/charts) its primal's metadata by restamping ([exists] tl/autodiff.py:153-173). (3) The batch/launch axis — `over(model, axis="batch")`. (4) Optimizer state — moments inherit their parameter's names (a moment is layout-congruent with its param by construction).

**Where names stop**: inside generated scalar bodies — erased at emission so named and positional twins share artifacts ([exists] inventory arrays.py:276-281); and at machine binding, where dims keep axis *identity* but must be chartless/label-free ([exists] tensorlib refusals, inventory test_placement.py:30-34). The optimizer is the probe's best evidence for name-genericity: Adam is written ONCE, generic over whatever names its param carries, because names arrive from the captured array's type — one more specialization axis, exactly the §7.4 prior. No dual universes appeared: the nameless bridge is `Named()`/`.rename` at the call site, never a function twin.

## 3. Where capture + pipeline fit, and where they did not

**Fit**: layer factories capturing weights (P1); the per-step rebuild loop (P6-B, P7) — capture-as-live-knob is the probe's cleanest win; per-token decode recapture (P8); `|`-stacking of blocks, because a transformer trunk genuinely threads one value (combinators.py:262-263's unary constraint is satisfied by residual-stream architecture — a lucky, ML-specific fit).

**Did not fit** (reported, not silently switched): (a) gradient clipping and Adam — list-of-tensors → scalar → list-of-tensors DAGs; pipe is unary and Stage arity is one (combinators.py:262-272) — F4. (b) reverse mode — `grad` wants the *program*, not a value; no Handle-level operator exists to compose with `|` at all. (c) The Q/K/V fan-out inside a block — three consumers of one value is a DAG, not a pipe; expressible only because ordinary Python assignment carries it (which is the dissolution answer working as intended). The honest summary: capture is load-bearing everywhere; pipeline is load-bearing only along the residual trunk.

## 4. The assemblage-surface verdict (with the programs as evidence)

**Dissolution holds for the defining side, amended for the step.** Evidence: P1 needed no new binding forms — ordinary Python + capture + `Named` + contract-by-name covers the model; P6-B/P7/P8 show the using side is ordinary host Python whose *identity discipline* (types-only keys, live-knob values) already exists and is pinned. The amendment: the training step demands exactly two capabilities with no owner today — (i) n-ary/pytree-valued composition (P5, P6's multi-output step), (ii) a reverse-mode operator on our surface whose lowering is `tensorlib.grad + dce` (P4). Neither requires a new *language*; both require new *operators* on the existing Derived/registry seams. One-invocation-many-dispatches is currently a claim with no executor (F8): everything tensor-tier in this probe runs only on `ir.run`.

## 5. Boundary ledger (refusal messages sketched)

- In-Program mutation: "programs are pure values; in-place update is a dispatch-boundary concern — write through out= at the call site or thread the new value (storage/L2 owns reuse)."
- In-kernel RNG: "rand() is not an op: randomness is data — generate noise/masks on the host and pass them as inputs (a counter-based marker is the future opt-in)."
- out-aliasing: "out= must be disjoint from every captured and argument buffer view in this dispatch; a row you do not read (K[t] vs K[:t]) is disjoint."
- n-ary pipe (today, [exists]): "pipe threads exactly one value; got N arg types" (combinators.py:263).
- MoE routing / top-k, token-embedding gather: remain excluded (data-dependent gather), boundary inherited and endorsed — zoo/gpt2.py:3-4, zoo/__init__.py:19-21.

## 5e. Syntax portfolio — Probe E: breadth sketches

# Probe E — breadth sketches (audio, convex, geometric algebra, discrete sim)

## Headline

Non-preclusion verdict: **2 pass today, 1 passes in the wrong clothes, 1 is precluded in its workhorse form.** Discrete simulation (cellular) and convex proximal steps run end-to-end on the C grid family today — verified against numpy oracles, steady-state loops all cache hits; named axes killed both transposes in ISTA (Aᵀr = `matmul(A, r, j)`, zero transpose). Geometric algebra works fully in tuple+function spelling with zero kernel changes, but the typed-struct level is half-built: records are capture-only, whole-record expressions crash with internals-speak (not designed refusals), and surface-C registration is global-by-accident — broken under `extend()` layering. Audio is the casualty: the IIR filter (audio's workhorse) is expressible **only** as an O(N²) per-lane extent-loop recompute — the founding-principle pathology, admitted rather than refused — while tensorlib already holds the exact answer (`scan` + `defreducer` linrec). The someone-else's-loop constraint unifies all four domains and one idiom serves it: fresh-closure-per-iteration, verified warm.

## 0. The someone-else's-loop constraint (named once)

Render loops, audio callbacks, physics ticks, training steps are **one concept**: a host iteration we do not own hands us per-iteration values (time, buffers, state) and we contribute dispatches into it. The framework's contract for it, verified this session:

- **Per-iteration values ride fresh closures, not rebound cells.** Rebuilding the closure each iteration (`make_voice(phase, ...)` per callback) is a pure tier-1 hit: same code object ⇒ same `fp`, types-only key, guards on the *old* closure's cells still pass, `extract` reads the *new* closure's values every dispatch (`registry.py:134-137`; verified under `no_compile()` in all three loop probes). Rebinding a long-lived closure's cell instead trips the identity guard (`cache.py:62-70`) and rebuilds — the anti-pattern.
- **In-place-mutated numpy captures are legal state**: identity guard passes, buffer re-extracted per dispatch (`registry.py:83-92,137`).
- **The foreign loop's buffers enter via `out=` adoption** with dtype/contiguity/rank checked against the artifact contract (`c.py:324-335`) — correct refusals, but see the f32 gap (finding E6).
- **The grid family refuses the argument channel entirely** (`registry.py:167-169`, verified: `VerifyError: backends.c.grid derives the params; pass the launch domain via out=`), so per-step arrays cannot ride args on the family that matters — captures are the only state channel there. This makes the fresh-closure idiom load-bearing, not stylistic; no canon doc names it (the taught model's `make_attn` factories are it, unnamed).

This idiom should be canonized once, for all four domains.

## 1. Audio / music synthesis — oscillator + filter voice in a foreign callback

### Defining side

```python
# --- oscillator excitation: per-sample, parallel — fits the grid family today
def make_excite(phase0, dphase, gain):
    @jit()                                        # [exists] capture.py:141-174
    def sample(i):                                # [exists] grid: params are domain coords (c.py:290-299)
        return gain * sin(phase0 + float(i) * dphase)   # [exists] sin battery (batteries.py:50), verified
    return sample

# --- one-pole IIR:  y[i] = a*y[i-1] + x[i]
# (A) the ONLY expressible spelling today — verified working, O(N^2) per block:
def make_iir_bad(x, a):                           # x: Named 1-d capture
    @jit()
    def cell(i):
        y = 0.0
        for s in range(0, i + 1):                 # [exists] param-dependent bound, verified — and the pathology
            y = a * y + x.isel(i=s)               # [exists] isel with computed index (arrays.py:305-329)
        return y
    return cell

# (B) the right-level spelling — the joint-system seam:
onepole = defreducer("onepole", state=2, element=2,          # [exists in tensorlib] mdsl.py:400-413 —
    lift=lambda a, x: (a, x),                                #   the docstring's own `linrec` IS this filter
    combine=lambda l, r: (l[0]*r[0], r[0]*l[1] + r[1]),      #   declared associative => parallelizable
    init=(1.0, 0.0), project=lambda A, B: B)
def make_voice(phase0, dphase, gain, a):
    ex = make_excite(phase0, dphase, gain)
    return scan(onepole(a), ex, axis="i")         # [proposed] pdum.dsl door to tensorlib scan (ir.py:68) — finding E1

poly = over(make_voice(...), axis="voice")        # [exists] over composes; polyphony = one more coordinate (test_grid.py:52-64)
```

### Using side

```python
def audio_callback(buf_f32, nframes, when):       # foreign loop: theirs, not ours
    voice = make_excite(phase[0], dphase, 0.2)    # [exists] fresh closure => warm hit (verified under no_compile)
    g.dispatch(voice, (), out=buf_f32)            # [exists refusal] adopt refuses f32 vs f64 kernel (c.py:326-327) — finding E6
    # today: out=buf64 then buf_f32[:] = buf64 — an allocation+copy in the callback
    phase[0] += nframes * dphase                  # [exists] host state threading
    mix = voices_out.sum(axis=-1)                 # [exists-as-host-escape] mixdown reduction is numpy — no reduce primitive
```

**Used unchanged:** jit/capture, grid family, batteries (`sin`), over, fresh-closure idiom, `out=` adoption. **Bent:** IIR forced into the O(N²) spelling (E1); f32 buffers forced through a host conversion copy (E6); mixdown/reductions escape to host numpy. **Refused (correctly):** f32-adopt dtype mismatch, non-contiguous out. Latency is fine: 2.4 µs warm dispatch against a 2.7 ms budget (128 frames @ 48 kHz); the *allocation* in the copy path is what real-time audio forbids.

## 2. Convex optimization — proximal/subgradient ISTA step

### Defining side (verified end-to-end: matches numpy ISTA oracle to 1e-10; 200-iteration loop all cache hits)

```python
def make_residual(A, x, b):        # A: Named ("row","col"); x: ("col",); b: ("row",)
    @jit()
    def cell(i):                                    # [exists] grid over rows
        return matmul(A, x, i) - b.isel(row=i)      # [exists] contraction pairs "col" by name (transforms.py:338-396)
    return cell

def make_prox_step(A, r, x, alpha, lam):
    @jit()
    def cell(j):                                    # [exists] grid over cols
        g = matmul(A, r, j)      # [exists] A^T r with ZERO transpose — shared axis is "row"; names did it
        z = x.isel(col=j) - alpha * g
        t = alpha * lam
        return (z - t) if z > t else ((z + t) if z < -t else 0.0)   # [exists] soft-threshold via ifexp
    return cell

# subgradient consumer: forward mode at kinks — verified: abs'(0)=+1, relu'(0)=+1
jf = jvp(f_scalar)                # [exists] scalar-only (transforms.py:290-296)
# jvp over the vector step: [planned-refused] "jvp differentiates w.r.t. float args" (test_array_args.py:91-92; 110/130 deferral)
```

### Using side

```python
for it in range(200):                                             # host loop — ours this time, same idiom
    g.dispatch(make_residual(An, Named(x,("col",)), bn), (), r)   # [exists] dispatch 1 (materializes r)
    g.dispatch(make_prox_step(An, rn, xn, alpha, lam), (), x_new) # [exists] dispatch 2
    x = x_new                                                     # ping-pong by rebinding host names
    fval = 0.5*(r**2).sum() + lam*np.abs(x).sum()                 # [exists-as-host-escape] objective = numpy reduction
```

**Used unchanged:** grid, matmul-by-name (its two contractions are the probe's best moment — breadth evidence *for* names-first), ifexp, fresh-closure loop, jvp for scalar directional derivatives. **Bent:** the two-stage step is a dispatch *sequence* with different domains (M rows vs N cols) — `Pipeline` cannot express it (unary, one artifact, value-level; `combinators.py:262-272` per inventory); objective values and dot products are host-numpy escapes. **Refused:** vector-level jvp (documented deferral — correct posture). **Kink adjudication (charter §7.5):** today's rules pick a deterministic side (`transforms.py:222-227`: abs uses pred "ge", min "le", max "ge") — every pick is a *valid* subgradient element, so the convex consumer is served by committing, not changing: pin at-kink values in tests (today `POINT = (1.7, 2.3)` is deliberately "away from kinks", test_jvp_rules.py:94) and document "jvp returns a fixed subgradient selection at kinks" (finding E2 in findings list = record crash; kink commitment folded into proposal of finding E5? — no: kink is finding-worthy on its own, see findings).

## 3. Geometric algebra — multivector struct + operator, zero kernel changes

Three spellings tested (Cl(2,0) rotor sandwich R·v·R̃, 90° rotation — both working spellings produce the correct rotation):

```python
# (A) tuple + surface-B vocabulary — [exists], VERIFIED, zero kernel changes:
def gp(u, v):   # capture-free @overload (surfaces.py:61-74); 4-tuples, constant indices [exists]
    return (u[0]*v[0] + u[1]*v[1] + u[2]*v[2] - u[3]*v[3], ...)
overload(ext, "gp")(gp); overload(ext, "rev")(rev)
@jit()
def rotate(x, y):
    v = (0.0, x, y, 0.0)
    out = gp(gp(R, v), rev(R))        # [exists] chaining works — tuples all the way down
    return (out[1], out[2])

# (B) @record struct — [exists capture-only], half a struct system:
@dataclass(frozen=True)
class MV2: s: float; e1: float; e2: float; e12: float
record(DEFAULT, MV2)                  # [exists] surface C (surfaces.py:77-116)
a.gp(b)[0]                            # [exists] method inlines, returns TUPLE (verified: -8.75)
a.gp(b).gp(a)                         # [refused] "no method 'gp' registered for (f64, f64, f64, f64)" — identity lost
MV2(x, 0.0, 0.0, 0.0)                 # [refused] no in-DSL construction ("cannot call 'MV2'")
a * b                                 # [broken] passes strict typing (ops.py:51-54 checks equality, not scalarity),
                                      #   dies at pack: "no slot for env path (0,) ... run NORMALIZE_ENV first" — finding E2
record(DEFAULT.extend(), MV2)         # [broken] @jit decoration then fails: "no ValueKind registered for 'MV2'"
                                      #   (valuekind.py:140; capture.py:141 types against global BUILTINS) — finding E3

# (C) operator spelling — [proposed], VERIFIED feasible with zero kernel diffs:
ext.lower_rules[ast.BinOp] = mv_binop(prev)   # rule chaining, the arrays.py:338-339 pattern
ext.specializations.bump_generation()          # manual — raw pokes skip _invalidate (footgun)
out = (R * v) * rev(R)                # works (verified) — but this is rule surgery, not a surface — finding E4
```

**090-surface verdict (the brief's direct question):** the five surfaces (`surfaces.py:1-14`, `registry.py:104-110`) support GA *vocabulary* (A) with zero kernel changes — confirmed by construction; they do **not** cover per-type operators (no door consults operand types before `_binop` emits core ops, `base_lang.py:95-99`), and surface C's kind registration is global-by-accident (`registry.py:99` defaults every `Registry()` to the shared `BUILTINS`; `extend()` — the documented layering route — is precisely the one that breaks). GA is the domain that makes "struct + operator" load-bearing rather than sugar; it is not precluded, but its natural spelling needs findings E2-E5 fixed.

## 4. Discrete simulation — one cellular tick (integer/branch-heavy)

### Defining side (verified end-to-end: Game of Life tick matches numpy oracle; tick 2 a pure cache hit)

```python
def make_life(board, H, W):            # board: Named ("y","x") capture; H, W: captured ints (staged i64)
    @jit()
    def cell(y, x):                                     # [exists] 2-d grid domain
        n = 0
        for dy in range(-1, 2):                         # [exists] negative bounds, nested loops, carries
            for dx in range(-1, 2):
                yy = (y + dy + H) % H                   # [exists] wrap boundary by hand (core.mod)
                xx = (x + dx + W) % W
                n = n + int(board.isel(y=yy, x=xx))     # [exists] computed indices; int() cast
        n = n - int(board.isel(y=y, x=x))
        alive = board.isel(y=y, x=x) > 0.5              # [exists] compare; strict bool ops
        return 1.0 if (n == 3 or (alive and n == 2)) else 0.0   # [exists] short-circuit + ifexp
    return cell
```

### Using side

```python
for t in range(T):                                       # ping-pong = fresh closures, roles swapped
    g.dispatch(make_life(Named(cur,("y","x")), H, W), (), nxt)   # [exists] out= adopts the write buffer
    cur, nxt = nxt, cur                                  # [exists] host swap; all iterations warm (verified)
```

**Used unchanged:** everything — this is the probe that fit best. Integer arithmetic, branch-heavy selects, mod-wrap, param-dependent trip counts (`range(0, i+1)` verified) all [exists]. **Bent:** boundary handling is manual index arithmetic; OOB reads are silent UB (no bounds checks anywhere in `_linear_index`, arrays.py:234-281 — consistent with the pinned matmul-extent-UB posture, test_refusal_contract.py:129-137, but tensorlib's guard+fill is the evidence the joint system can do better); neighbor-offset tables can't be indexed by loop variable (constant indices only, base_lang.py:157-158) — unrolled loops suffice here. **Refused / boundary verdict:** scatter-shaped discrete work — contact resolution, particle binning, compaction — is **outside the subset**: grid kernels write only their own cell (one `out`, no atomics, no dynamic gather). Boundary refusal sketch: *"grid kernels write out[coords] only — scattered writes (binning, contacts) need an atomic/sort dialect that does not exist; keep particle→cell assignment on the host or wait for the binning family."* This is a success per charter §1, not a failure.

## 5. Named-axes stance (charter §5)

- **Audio:** names live on data axes (wavetable position, voice) and stop at the sample coordinate — a machine-bound dim with axis identity and no chart, exactly LEVELS' surface discipline. `over(..., axis="voice")` is the polyphony spelling; names never appear inside the per-sample body.
- **Convex:** the strongest names-first evidence outside ML in the whole assessment: `matmul(A, x, i)` contracts "col", `matmul(A, r, j)` contracts "row" — **both Gram-step contractions with zero transposes**, because identity-by-name makes Aᵀ a non-operation (DESIGN D5's conviction landing in a domain tensorlib never targeted).
- **GA:** the counter-case that keeps names honest — multivector components are **struct fields, not named dims**. Forcing `e12` into a length-4 named axis would be tensor-shaped clothes on an algebra; the record system (once whole) is the right home. Names and structs are complementary, not competing.
- **Discrete sim:** domain coordinates are the axes; the mild dissonance is that grid *params* are anonymous positional (`cell(y, x)` binds by order to the out-array's axes) while every array is named — the known over/split+bind gap surfacing as "the launch domain has no names."

No dual-universe pressure appeared in any sketch; no name-genericity pressure either (all four use concrete domain names in leaf programs).

## 6. Capture + pipeline fit (charter §5)

**Capture is the star of all four probes**: rotors, boards, matrices, gains, phases — everything rides closures; the fresh-closure-per-iteration idiom (verified warm in three separate loops) is the foreign-loop answer and deserves canon. **Pipeline fit zero of four probes.** Audio (excite → filter), convex (residual → prox), discrete sim (tick → tick) all want to sequence *dispatches* with materialization between stages and different domains per stage; `Pipeline` composes unary value-level stages into one artifact. This is charter §7.3's two-senses question answered from breadth: the senses are distinct, the dispatch-level one is what every foreign-loop domain needs, and today it is spelled as bare host statements (which — to be fair — read fine). The value-level pipe found no natural use because every probe kernel is multi-argument and pipes thread exactly one value.

## 7. Where the sketches bind to the §2/§3 hypotheses

- **§2.1 no-extent-loops:** audio's IIR shows the principle is *right* but unenforced — the extent-loop spelling is not just style debt, it is an asymptotic-complexity trap (O(N²) per block) the subset currently admits. The principle needs its primitive (scan) before it can be enforced.
- **§2.2 one-IR / collectives-read-off-algebra:** the associative `defreducer` combine (linrec) is precisely how audio's filter becomes parallelizable without a new op — breadth confirms the tensorlib stance from a non-ML domain.
- **§2.3 synthesis:** nothing in the four domains needs a new *language* — they need three doors (scan, operators, record construction) and one canonized idiom into the existing one. The dissolution verdict for the syntax layer holds at breadth: ordinary Python + capture + grid + a scan door covers all four.
- **Breadth-is-identity check:** no concept used here is ML-scoped; conversely, two concepts the ML probes lean on (Pipeline-as-fusion, jvp-over-arrays) contributed nothing at breadth — consistent with keeping them satellite, not kernel.

## 6. Verified flaw list

1. **[HIGH / CONFIRMED] rewrite-vocabulary-contraction-shaped** — The certified-rewrite rule classes enumerated for tier-3 assurance (LEVELS.md:109-114) and the charter's chain vocabulary (140:258-260) cannot certify the scheduled L4 fused-stencil-chain flagship (LEVELS.md:183-184, K-D 207-210): overlapped-split/halo-recompute is a distinct class — overlapped tile covers are outside split∘merge=id (disjoint by construction), halo recompute increases ops so it is not elision-of-materialization, and the only recompute machinery (L1 checkpointing) carries an acknowledged name=value certification gap (CONCERNS.md:254-256). No tensorlib doc or charter amendment names the class, and Probe C's rewrite-chain evidence is scoped to GEMM+attention only (140:445-448). Softening: the L4 question list is explicitly pre-decision (LEVELS.md:191-192), so this is an unacknowledged missing rule class on a committed flagship path, not a violated decision.
   - Evidence: LEVELS.md:109-114 (rule list), LEVELS.md:183-184 and 207-210 (fused stencil chain is a scheduled L4 flagship); architecture verdict A1-A8 (silent on the class)
   - Verifier: Trigger: run the committed K-D fused-stencil-chain flagship (LEVELS.md:183-184, 207-210) through tier-3 certification — none of the four named rule classes can produce the overlapped tile cover (split is disjoint: split∘merge=id) or the op-count-increasing halo recompute (fusion rule only elides materializations), and the only existing recompute mechanism (L1 checkpointing) has an acknowledged name=value certification hole (CONCERNS.md:254-256). Mitigation: LEVELS.md:191-192 marks L4 rules as undecided, so this is a missing class in a planned path, not a contradiction in decided canon. — "LEVELS.md:110-114: "Lean proves each REWRITE RULE once (split∘merge = id under divisibility, guard/slice commutation, reassociation via declared associativity, fusion = elision of identity materializations — nearly free in a pure IR)." LEVELS.md:183-184: "L4: manual fusion + tiling with the rewrite "
   - Proposal: Add the overlapped-split/halo-recompute class to the certified-rule roadmap and make the stencil flagship the L4 brief's non-contraction acceptance test; it serves PDE, discrete sim, and graphics convolution.
2. **[HIGH / CONFIRMED] graphics-scheduling-drift** — In the plan of record (020), every remaining scheduled installment (Phase IV steps 11-17) and every unscheduled backlog item builds tensor/transform/tile/backend or units machinery — none builds graphics machinery (vertex shaders, PSO pairing, MRT, foreign-render-loop encode), which after step 9's fragment-only landing exists only as an unassigned roster entry (070:14,28) — while step 16 delivers the tile family CUDA+Metal-first; so the ML/tensor-ward drift that "breadth is identity" (140:58-68) guards against is occurring through scheduling (graphics reduced to demo skin in steps 11/15) rather than through any single concept.
   - Evidence: 020_implementation-plan.md:317-327 (Phase IV steps 11-17: no vertex/fragment/PSO/render step), 020:347-354 (unscheduled list also lacks one), 020:326 (step 16 = CUDA+Metal carrying the tile family + GEMM chapter); charter 140:59-68
   - Verifier: Trigger is the plan itself: the final phase's step table (11-17) and the unscheduled backlog contain no vertex/fragment/PSO/render-interop step while step 16 explicitly carries the tile family on CUDA+Metal; the only domain satellite in the backlog is the deep-learning one. Refinement: fragment shading did land at step 9 and shader-flavored demos persist (steps 11, 15), so the drift is in forward-scheduled *machinery*, not total absence of graphics. — ""| 16 — CUDA + Metal backends (was 14; revised 2026-07-12, 070/R13/R14: OWN both stacks; 130 adds the TILE family: `stage`/`barrier`/tokens, mma-backed `contract`, the GEMM chapter)" (020:326); "**Later, unscheduled** (each becomes a step+chapter when pulled forward): t-string mini-language (`ch17-e"
   - Proposal: Direction memo slots a graphics installment (vertex/fragment in the validated kind vocabulary, PSO pairing as the third composition semantics, encode-into-foreign-pass deliverable) before or alongside the tile step; all parity divergences declared as capability-gated refusals.
3. **[HIGH / CONFIRMED] pipeline-two-concepts** — The shipped pipeline is fuse-only and unary (one threaded value, one-arg stages: combinators.py:262-263,271-272), while dispatch-level sequencing — which probes A/B/E report needing — has no implementation and no 020 plan step (040 §5 parks it "far later"; 020 Phase IV and 130 §4.3 omit it); however, syntax and mechanism for it ARE decided in 040 §2.3/§2.6 (same `|`, role rules returning an "orchestrate" semantics tag already named at combinators.py:69-70, demotion visible via the execution-mode lattice), so the finding's real content is (a) a scheduling gap and (b) a collision between the proposal's "two operators, never one" stance and 040's decided one-operator design — which charter §8 requires escalating to the human rather than resolving silently.
   - Evidence: Unary constraints verified: combinators.py:262-263,271-272; Probes A/B/E each report pipeline fit zero; Probe D needs n-ary; 020:324 and 130 §4.3 cover no sequencing; semantics-tag registry combinators.py:67-71
   - Verifier: Trigger is concrete: any probe needing multi-kernel dispatch sequencing (frames, ping-pong sub-steps) finds only a fuse-only unary pipe (combinators.py:262-272) with orchestration parked "far later" and absent from 020's Phase IV steps — but the claim's "no syntax" and "never one operator" parts overreach: 040 §2.3/§2.6 already decided one `|` operator with role-driven, visible fuse-vs-orchestrate semantics on the exact registry the proposal cites, so the two-vs-one-concept part is a §8 collision to escalate to the human, not a fresh gap. — "combinators.py:262-263 'raise VerifyError(f"pipe threads exactly one value; got {len(arg_types)} arg types")' + 040_combinators-notes.md:248-249 'Effect ordering for orchestrated pipelines (render-graph scheduling) — belongs to the runtime layer that owns passes, far later.' + combinators.py:69-70 '"
   - Proposal: Keep | = fuse (widen to n-ary via roles for Probe D's optimizer); add dispatch sequencing as the pdum.dsl mirror of tensorlib fold with an explicit carry contract; add PSO pairing as a third semantics tag on the existing (op, roles)->tag registry; never one operator with implicit fuse-or-record choice.
4. **[HIGH / CONFIRMED] derivative-ambient-params** — D differentiates w.r.t. all enclosing-kernel params with no wrt selection (transforms.py:313-335) and refuses whenever the differentiated value's dataflow reaches a non-float-scalar param (refusal is dependency-triggered, not signature-triggered: unused non-float params pass). Because coordinate typing is per-family — grid coords i64 (c.py:299) vs wgsl coords f64 (wgsl.py:72) — D/fwidth work in the wgsl family but refuse in the grid family even after explicit float() casts, violating equal-citizens; fwidth additionally hardcodes d[0]/d[1] (graphics.py:45-55), and no test covers D in a grid kernel.
   - Evidence: transforms.py:313-335 and float-only seed at :326-327 (re-read); c.py:299 i64 vs wgsl.py _param_types (f64,)*argcount (re-read); Probe A F3; test_transforms.py:142 zero-cost-unused
   - Verifier: Empirically triggered: in a C-grid kernel, `D(float(i)*0.5 + float(j)*0.25)` raises `MissingRule: D differentiates w.r.t. float params; param 0 is i64` while the identical body in the f64-param family returns 0.75 — and graphics.py:43-44's own comment claims analytic D is the compute-shader derivative story, contradicting the grid family's i64 coords. — "transforms.py:326-327: "if not (isinstance(t, Scalar) and t.kind[0] == 'f'): raise MissingRule(f'D differentiates w.r.t. float params; param {k} is {t!r} ...')"; c.py:299: "return (i64,) * (t.pyfunc.__code__.co_argcount + lanes)"; wgsl.py:72: "return (f64,) * pyfunc.__code__.co_argcount""
   - Proposal: Families declare named ambient axes; replace with D(x, wrt=axis_name); unify per-family coordinate typing; fwidth's hardcoded d[0]/d[1] (graphics.py:45-55) falls out. Defer wrt-as-structure — no probe demanded it.
5. **[HIGH / CONFIRMED] joint-seam-fork** — Canon 130 (status PROPOSED, never amended) plans a pdum.dsl-owned tensor dialect (§4.2, staged in §8 stage 2) and a pdum.dsl-owned reverse-mode Grad (§8 stage 3) [planned], while charter 140 §5 mandates probes front tensorlib's Node/Program schema; the two streams share zero imports and zero tests (mdsl.py's adapter is docstring-only), and pdum.dsl's existing matmul already erases contraction structure at lowering (transforms.py:393-396 [exists]), contradicting both 130's own preserve-until-target-selection principle and LEVELS'. This is an unadjudicated representation fork that charter §8 governance requires escalating to the human before 130 stage-2 code lands.
   - Evidence: 130 §4-5 (tensor dialect, planned); mdsl.py:1-17 zero-import Node schema built for frontends; behavior-spec inventory (joint seam zero tests); Probe D headline; matmul erases contraction at birth, transforms.py:381-396
   - Verifier: Trigger: 130 §8 stage 2 landing as written would build tensor.map/reduce/contract and later Grad on pdum.dsl's own IR with zero connective tissue to tensorlib's Program/grad — two representations and two ADs for the same tensor semantics, and neither 130 nor 140 rescinds or holds the other's plan; the only mitigation is that charter §8/deliverable-8 already provides the escalation channel the proposal invokes, so this is a confirmed collision-to-escalate, not a violation of a decided rule. — "130 §4.2: "`tensor.map` / `tensor.reduce` / `tensor.contract` / `tensor.slice`: satellite ops (surface A) whose *default decomposition* is the corresponding `core.for` form"; 130 §8 stage 3: "Grad ... adjoints of `contract`/`reduce`/`map` are textbook"; vs charter §5: "it writes *our* syntax frontin"
   - Proposal: Escalate to human per charter §8 governance before 130 stage-2 code lands; decide that the tensor dialect emits/mirrors tensorlib's Program so named contract/reduce/map remain the one working representation until target selection.
6. **[HIGH / CONFIRMED] state-aliasing-silent-corruption** — In the C grid family, passing out= an ndarray that aliases a captured read array is accepted and silently produces wrong results (verified by execution on a shift kernel); make_grid_launcher (src/pdum/dsl/backends/c.py:315-338) checks dtype, C-contiguity, and rank only, with no overlap check against the captured buffer leaves — a refusal-first violation at the write seam, ironic given the same function's comments cite "silent corruption" as the reason the dtype/rank checks exist. The proposed fix (launcher-level overlap refusal quoting the ping-pong workaround; in-place only ever as a certified L2 rewrite) matches the incumbent stance that mutation is a storage-level phenomenon.
   - Evidence: Probe B verified execution (all-zeros vs expected shift, no error); make_grid_launcher checks dtype/contiguity/rank only, no overlap check — c.py:315-338 (re-read this session)
   - Verifier: Reproduced this session: a grid shift kernel `t.isel(x=(i+1)%8)` dispatched with out= the captured array itself returns [1..7, 1.] instead of [1..7, 0.] with no error — silent corruption from reading an already-overwritten cell; fresh-out dispatch is correct. The exact wrong values differ from the probe's "all-zeros" (that depends on the access pattern), but the mechanism and silent-wrong-answer trigger are confirmed. Not guarded or decided elsewhere: repo-wide grep finds no shares_memory/overlap check, and docs (design/100 line 148 "needs store legality + aliasing rules"; research/R6 lines 231-232, 338-339) list capture aliasing as an open hole, not a resolution. — "c.py:326-335 validates only dtype/contiguity/rank: `if spec.dtype != want: ... raise VerifyError`; `if not spec.flags["C_CONTIGUOUS"]: raise VerifyError`; `if out.ndim != artifact.grid_rank: raise VerifyError` — then `out = spec  # adopt` and `artifact.grid_call(staging, buffers, out, out.shape)` wi"
   - Proposal: Launcher overlap check of out against captured array leaves; refuse with "out aliases captured array 'u' — write into a second buffer (ping-pong), or wait for certified in-place"; safe in-place returns only as an L2-certified rewrite, never a user gesture.
7. **[HIGH / CONFIRMED] chain-provenance-homeless** — No component in either stream is charged with emitting or storing a rewrite chain: pdum.dsl build rules elaborate directly to a final Region with no step provenance, the artifact tier stores only compiled programs keyed by (region.key, backend.fp), tensorlib has no chain/witness structure, and the only decided plan language (LEVELS.md assurance tier 3 and order-of-attack step 6) places the burden on a checker — so §3.2's "or the checker reconstructs a chain" hedge is the de facto default for the unbuilt L4 path, which for compute-level rewrites is the search-shaped direction Amendment 1 exists to avoid.
   - Evidence: src/pdum/dsl/combinators.py:254-278 (build_pipe inlines); src/pdum/dsl/stdlib/transforms.py:266-287 (build_over re-enters lower_handle); src/pdum/dsl/kernel/registry.py:186-191
   - Verifier: Trigger: the moment L4's tiling DSL is built on the existing frontend machine (build rules return bare Regions — combinators.py:254-278, transforms.py:266-287 — and FastRecord/ArtifactCache have no chain slot, registry.py:195-202, cache.py:106-127), §3.2's primary path has no producer: no document or code assigns chain emission to any component, while LEVELS.md twice words the obligation checker-side, so reconstruction becomes the default by omission. Softening: nothing runs this path today (L4 and the checker are both unbuilt), and for layout-only deltas reconstruction is decidable normalization, not search (LEVELS.md tier 2) — the search burden applies to the compute-level rewrites (fusion, reorder-under-license) that are exactly Probe C's subject. — "registry.py:190-191: "artifact = self.artifacts.get_or_compile(  # content-addressed: identical IR compiles once / (region.key, backend.fp)"; LEVELS.md:112-113: "Python checks that a lowering is a chain of certified rules."; LEVELS.md:183: "**L4**: manual fusion + tiling with the rewrite checker"; c"
   - Proposal: Make the chain a mandatory content-addressed output of tile-DSL elaboration, stored as a companion artifact beside the compiled program; demote checker-side reconstruction to a migration tool, never the steady state.
8. **[HIGH / CONFIRMED] registry-key-unsoundness** — §3.4's sketched certified-lowerings key (chunk fingerprint → implementation) is unsound as written on two axes it omits outright — the granted license set (directly contradicting §3.2's license-relative equivalence) and a rules-revocation/generation axis (required because, unlike the artifact cache whose shape it invokes, its values are authored rather than derived from the key) — while the other two claimed omissions are weaker: the AD saved-set circularity is real (chosen lowering determines the saved set, unknown at fingerprint time) but already registered as a first-class open question in §3.3, and boundary layout classes are plausibly covered by §7.7's fingerprint-skeleton sketch. §3.4 is a design sketch under attack per charter, not shipped code; the proposed key extension (license set + boundary/saved-set contract + rules-generation, mirroring the spec tier's generation) is the right-level fix.
   - Evidence: docs/design/140_critical-assessment-charter.md:289-301; src/pdum/dsl/kernel/registry.py:186-191 (artifact key), 133-134 (generation exists only in spec tier); explorations/tensorlib/tests/test_zoo.py:66-79 (flash backward demand)
   - Verifier: Concrete trigger: two chunks with identical normalized IR but different granted license sets share the §3.4 fingerprint, so a lookup serves an implementation certified only equal-modulo-reassociation to a caller who granted no license — the key cannot express the charter's own license-relative equivalence; and unlike the artifact cache §3.4 claims to mirror (value derived from key, so content-addressing self-invalidates), certified-lowering values are authored, so rule revocation strands stale entries under unchanged keys with no generation axis. — ""A certified descent is a pair: (content-addressed fingerprint of the IR chunk → verified lower-level implementation)." (140_critical-assessment-charter.md:289-291) vs "**Equivalence is license-relative**: bit-exact where no license is used; equal-modulo-declared-associativity where reassociation wa"
   - Proposal: Key = (normalized chunk skeleton, boundary contract incl. saved-set demand + layout classes, license set, capability set, rules-generation); value = (chain with rule citations, authored region, artifact, assurance tier); add a rules-generation bump for revocation, mirroring the spec tier's generation mechanism.
9. **[HIGH / CONFIRMED] wf-obligations-missing** — The certification framework (140 §3.2/§3.4) states obligations only for meaning-preservation; race-freedom, shared-memory capacity, and fusion-group convexity — well-formedness properties of the lowered form — have no obligation class there (capacity/convexity appear only as undecided L4 questions, LEVELS K-C). Race-freedom does have a planned non-chain mechanism (token typing, 130:169-173, [planned] per 130 §8 step 4), but 130's own flagship GEMM (130:188-192) binds no token from stage, passes none to barrier, and consumes none in contract, contradicting the same doc's "misordered kernel is a type error (missing token)" guarantee unless an unstated implicit-threading elaboration is assumed.
   - Evidence: docs/design/130_tensors-tiles-and-over.md:169-173 vs 188-190; docs/design/140_critical-assessment-charter.md:249-274 (equivalence-only); explorations/tensorlib/LEVELS.md:203-206 (K-C)
   - Verifier: Trigger is textual and concrete: charter §3.2 (140:249-274) enumerates only equivalence obligations (its rewrite list even includes pad-to-tile-with-guards and bind-to-level with no capacity/sync certificate), capacity/convexity exist only as undecided queued questions (LEVELS K-C under "nothing here is decided"), and 130's flagship GEMM discards the very tokens whose presence is the doc's stated race-freedom guarantee, so the "type error, not a race" claim cannot fire on the centerpiece as written. — "130:170-173: "`barrier()` → `tile.barrier` (token → token); cooperative ops consume tokens. **Purity preserved as dataflow**: ... a misordered kernel is a *type* error (missing token), not a race." vs 130:188-191: "a = stage(A.tile(m=tm, k=kb), pad=\"conflict-free\") ... barrier() ... acc = acc + co"
   - Proposal: A certified descent = equivalence chain + per-level WF certificate: race-freedom (token/barrier discipline checked on the elaborated form), capacity (dtype-exact staged bytes <= level capacity), convexity (K-C check); WF predicates are checked on the result, not derived from the chain.
10. **[HIGH / CONFIRMED] license-vocabulary-precision** — Charter §3.2's equivalence taxonomy is stated as exactly two license classes (bit-exact; equal-modulo-declared-associativity), and neither covers the precision demotion (f16 staging, f32 accumulate, f16 emit) present in 130 §5's flagship GEMM — so as written the chain-certification scheme cannot certify its own centerpiece descent. The fix direction is partly pre-decided: tensorlib COMPUTE.md §2b already splits carrier (semantics, what Lean denotes) from dtype (cost), so the finding is a §8 decided-position collision — §3.2 must restate equivalence over the carrier denotation with dtype changes as cost (or add an explicit precision-demotion license), with the numeric tier monitoring float divergence under declared tolerance rather than certifying it.
   - Evidence: docs/design/140_critical-assessment-charter.md:264-267; explorations/tensorlib/LEVELS.md:109-114; docs/design/130_tensors-tiles-and-over.md:174-176,186,193
   - Verifier: Concrete trigger: certifying the 130 §5 GEMM descent under §3.2 — the f16 stage reads and astype(f16) emit are neither bit-exact against the carrier-real L0 denotation (tensorlib's decided oracle semantics, COMPUTE.md 2b; also the numeric tier's comparison target, charter :269-270) nor a declared reassociation, and LEVELS.md:109-114's certified-rule list contains no precision rule; the escape "precision is authored at L0 so no rewrite occurs" is foreclosed by tensorlib's own 'semantics never mentions precision' stance. — "140_critical-assessment-charter.md:264-266: "Equivalence is license-relative: bit-exact where no license is used; equal-modulo-declared-associativity where reassociation was claimed." — vs 130_tensors-tiles-and-over.md:186,193: "acc = zeros((\"m\", \"n\"), f32)" ... "return acc.astype(f16)  # precis"
   - Proposal: Add a third license class 'precision demotion'; restate chain equivalence over the carrier (real) denotation using tensorlib's Tensor.carrier (tensor.py:48-53), with dtype as cost; numeric tier monitors float divergence under a declared tolerance + input domain, never certifies it.
11. **[HIGH / CONFIRMED] step14-gate-pins-violation** — The flagship attention sample writes raw `for s in range(S)` extent loops (tests/test_grid.py:119-123) and the ch14 chapter — the taught UX contract — frames the step-14 gate as "softmax as explicit loops" (build_chapters.py:4070), canonizing the form the charter's affirmed no-extent-loops principle (140 §2.1:92-97) forbids; however the gate itself only tests dispatch economics, and the written plan pins the replacement, not the anti-pattern: 020:324 lists "comprehensions" as remaining and 130 §8 stage 2 gates on "softmax via comprehensions matches numpy" — so the fix is a rewrite of sample+chapter prose when 130 §4.2's spelling lands, not a plan change, and severity drops from "plan institutionalizes violation" to "flagship artifacts lag the plan and teach the forbidden form in the interim."
   - Evidence: tests/test_grid.py:116-128 (read: two 'for s in range(S)' over a captured extent); 020:324 'softmax as loops' gate spec (canon inventory divergence #1); Probe C §2a/§4 (every descent step impossible from the loop form); Probe E audio (extent-loop-only spelling yields O(N^2) pathology, admitted not refused).
   - Verifier: Trigger is quotable: the flagship sample (test_grid.py:116-128, reproduced verbatim in ch14 at build_chapters.py:4074-4086) writes raw extent loops, and the taught chapter explicitly narrates the gate as "softmax as explicit loops" — canonizing the anti-pattern in the de facto UX contract. One citation in the candidate is wrong and must be corrected: 020:324's gate spec does NOT say "softmax as loops" — it says "softmax comprehensions vs numpy" and lists comprehensions among "remaining" work, and 130 §8 stage 2 likewise gates on "softmax via comprehensions matches numpy"; the gate test itself (test_grid.py:148-173) measures only the dispatch-count ratio, not spelling. So "the plan itself pins the anti-pattern" is overstated — the plan (020:324, 130 §8.2) already schedules the reduce/comprehension replacement; what pins the loop form is the chapter prose and the frozen test sample, not the plan. The proposal (rewrite flagship in named form once 130 §4.2's spelling lands; stage refusal only after the replacement exists) stands, consistent with ch14's own note (build_chapters.py:4127-4130) that tensor.map/DPS were deliberately deferred. — "tests/test_grid.py:119-123: "den = 0.0 / for s in range(S): / den = den + exp(matmul(Q, K, t, s) * scale)"; build_chapters.py:4069-4070 (ch14, the taught contract): "The stage exit gate (020 step 14): write attention for ONE example — softmax as explicit loops, `matmul` pairing axes by name"; charte"
   - Proposal: Principle affirmed, artifacts convicted: schedule 130 §4.2's reduce/comprehension spelling before or with the next gate revision, rewrite the flagship in named form, re-scope the gate to it; then stage extent-loop refusal at L0 with an explicit debug-grade escape. Do not refuse before the replacement spelling exists.
12. **[HIGH / CONFIRMED] over-binding-not-represented** — over's split+bind is realized only at lowering/dispatch time: build_over returns a Region whose lane is an unmarked trailing i64 param (transforms.py:276,287; Region has no binding metadata, kernel/ir.py:111), so the C grid backend recovers the launch domain by isinstance-unwrapping the stdlib satellite class Over (backends/c.py:290-298) — a backend-imports-satellite layering inversion (in backends/, not kernel/ proper). The map-loop-IR emission that would fix representation is already [planned] (020 step-14 remaining, transforms.py:233-236 docstring); the finding's novel content is the concrete inversion as evidence plus the open charter-§2.4 adjudication that over should be unified with tensorlib's dim-to-level bind (placement.py:1-8, LEVELS.md:25-30) so backends read structure, never satellite classes. Trailing-lane contract, weave lowering, and the 16x gate are unaffected either way.
   - Evidence: src/pdum/dsl/backends/c.py:290-299 (read: imports stdlib.transforms.Over, isinstance unwrap); transforms.py:232-236 (read: docstring already promises 'map-loop IR form arrives... without changing this identity'); 020:324 over-emits-map-IR as remaining work (canon inventory); tensorlib bind/Dim.level as the general mechanism (placement.py:1-8, LEVELS.md:25-30, read).
   - Verifier: Trigger is present-tense and structural: after build_over the Region carries no lane/binding metadata (ir.py:111-117 Region = params+body only), so the C grid backend must isinstance-unwrap the surface Over satellite (backends/c.py:290-298) to recover domain arity — a backend reading a surface class instead of IR structure. Not refuted as already-decided: 020 step 14 lists "over-emits-map-IR" only as remaining [planned] work, and charter §2.4 leaves the reabsorption-into-general-bind question explicitly open. — "backends/c.py:291-296: "from ..stdlib.transforms import Over ... while not hasattr(t, \"pyfunc\"):  # unwrap over-chains: each adds one coordinate / if isinstance(t, Over): lanes, t = lanes + 1, t.captures[0]"; transforms.py:287: "return Region(params=(*inner.params, lane), body=inner.body)" (lane i"
   - Proposal: Reabsorb: over emits map/bind structure in the representation (the already-planned step); unify over and tensorlib bind into one split+bind concept at different machine-tree depths; backends read structure, never satellite classes. The trailing-lane contract, the weave-as-one-lowering, and the 16x gate all survive.
13. **[HIGH / CONFIRMED] joint-ir-fork** — The joint system holds two disconnected program representations (pdum.dsl's scalar core Region IR, lower.py:176-203; tensorlib's Program/Instr IR whose only builder is explicitly "not a frontend", build.py:1-7), connected only by a future-tense marker-body adapter note (mdsl.py:12-14); canon 130 — written before the charter's joint-system decision — plans a third path (tensor.map/reduce/contract satellite ops in pdum.dsl's own IR, 130:130-137, with its own stage-3 reverse-mode grad, 130:263-266) that collides with the charter §5 mandate to front tensorlib's representation and duplicates tensorlib's existing AD; the collision is a stale-canon fork that no document adjudicates, and per charter §8 must be escalated to the human before 130 stages 2-3 land.
   - Evidence: pdum.dsl lowering path is scalar-L0 only (src/pdum/dsl/kernel/lower.py:176-199); tensorlib Program is unconnected (tl/build.py:1-7 'deliberately not a frontend', mdsl.py:1-16 marker-body seam only, per Probe D F1); 130:130-134 and 130:263-266 plan Node-IR tensor dialect + stage-3 grad (canon inventory); charter §5 'the joint system is the target'.
   - Verifier: Trigger verified by direct inspection: zero cross-imports between src/pdum/dsl and explorations/tensorlib (grep both directions, this session), no tensor.* ops in src (130 stage 2 unlanded), lower_handle emits only scalar core Region (lower.py:176-203), and 130:130-137 + 263-266 plan an IR-native tensor dialect plus a second grad while tensorlib's grad already exists — no document adjudicates between them; charter §10's "frontend→Node-schema integration plan" deliverable commissions the adjudication but is not one. — "build.py:3-4 "Deliberately not a frontend — no tracing, no operator overloading"; mdsl.py:12-14 "and, once the main repo's frontend stabilizes, an adapter mapping its lowered AST onto the same Nodes"; 130:263-264 "3. **Grad** (old step 13, resequenced AFTER tensors deliberately) — adjoints of `contr"
   - Proposal: Adjudicate the fork now, before stage-2/3 of 130 lands: re-scope 130's tensor dialect as a frontend emitting tensorlib Program (markers/instrs), one AD (tensorlib grad + dce) behind a pdum.dsl surface operator; the emission seam is the highest-priority artifact. Collision with a decided position — escalate to the human per charter §8.
14. **[HIGH / CONFIRMED] record-values-crash-not-refuse** — Whole-record values in positions other than field/method access crash with internal-invariant VerifyErrors instead of designed refusals: _arith (ops.py:51-54) checks only type equality, so arithmetic on two equal Record types type-checks and then fails at pack-time legalization (pack.py:284-286) with "no slot for env path (0,): ... run NORMALIZE_ENV first"; returning a captured record fails identically; and Record-with-scalar arithmetic is refused with cast advice ("insert an explicit core.cast") that cannot apply to records — all in the first path a record-using author would hit, violating the refusal-first corollary (charter 140 §1) that the codebase itself honors one branch earlier for captured-kernel values.
   - Evidence: ops.py:51-54 (_arith refuses only mismatched types; equal Records pass); empirically verified (scratchpad probe_e2.py): VerifyError 'no slot for env path (0,)' for both a*b and return-a; TypeError 'insert an explicit core.cast' for a+x; refusal-first corollary charter 140:47-48.
   - Verifier: Trigger reproduced empirically this session with real closures over demo.graphics.Color: multiplying two same-typed captured records, or returning a captured record, dies in legalize_params with a message addressed to compiler internals ("run NORMALIZE_ENV first"), not a user refusal; and the Record+f64 refusal's fix advice ("insert an explicit core.cast") is inapplicable to records. No design doc decides this boundary, and pack.py's neighboring FnType branch proves designed refusals are the house style for exactly this slot. — "ops.py:52-53: "if args[0] != args[1]: raise TypeError(f\"core arithmetic is strict: {args[0]!r} vs {args[1]!r} — insert an explicit core.cast\")" — equal Record types pass. Reproduced (Color captures, this session): a*b and return-a both raise VerifyError "no slot for env path (0,): a composite capt"
   - Proposal: Make _arith require Scalar (or Vec) operands with a designed message naming the fixes (extract fields; register an operator when that surface exists), and give whole-record return/use a designed refusal in the arrays.py:213 style; the current crash is a silent-wrongness-class violation of the refusal contract in the path every record-curious user hits first.
15. **[HIGH / CONFIRMED] scan-primitive-missing** — Audio's workhorse (IIR/first-order linear recurrence) is expressible in pdum.dsl today only as an O(N^2) per-lane extent-loop recompute — a param-dependent `for s in range(0, i+1)` that base_lang's for-rule accepts (only i64-typing of bounds is checked, base_lang.py:290-294) and that runs correctly on the C grid — violating the charter's level-0 no-extent-loops principle (140:92-97) without refusal or debug-gating; pdum.dsl has no scan/fold combinator (combinators.py is Pipeline-only), while tensorlib already holds the right-level primitive family (scan and fold in _COMPUTE_OPS, ir.py:68; fold gap analysis LEVELS.md:143-159) and its defreducer docstring example `linrec` (mdsl.py:408-412) is exactly the associative combine for this filter. Note one precision: this is per-lane in-kernel recompute under a single dispatch, i.e. a violation of the §2.1 no-extent-loops principle rather than of the founding per-pixel host-dispatch example itself.
   - Evidence: Verified running: for s in range(0, i+1) one-pole IIR on the C grid (scratchpad probe_e.py, correct output); param-dependent bounds accepted by base_lang.py:271-299; no scan/fold combinator in pdum.dsl (combinators.py has Pipeline only); tensorlib scan/fold in _COMPUTE_OPS explorations/tensorlib/tensorlib/ir.py:68, defreducer with linrec example mdsl.py:400-413, fold gap analysis LEVELS.md:143-159; charter 140:90-97 no-extent-loops principle.
   - Verifier: Trigger reproduced this session: the param-dependent extent loop `for s in range(0, i+1)` one-pole IIR compiled and ran correctly on the C grid (fresh verify script, `accepted: True, correct: True`); base_lang.py:290-294 lowers any i64 expr as a bound with no work-growth guard, combinators.py holds only Pipeline, while tensorlib ir.py:68 lists scan/fold in _COMPUTE_OPS and mdsl.py:408-412's linrec defreducer is literally the first-order filter combine — the pdum.dsl surface admits the O(N^2) form the charter's no-extent-loops principle (140:92-97) says should be inexpressible or debug-confined. — "base_lang.py:275 "Bounded loops only — the GPU-honest subset."; charter 140:95-97 "extent iteration belongs to higher-level primitives (map/reduce/contract) that a scheduler can own"; mdsl.py:408-412 "linrec = defreducer(\"linrec\", state=2, element=2, lift=lambda a, b: (a, b), combine=lambda l, r: "
   - Proposal: Promote scan/fold to the shared surface as the extent-iteration primitive family (a pdum.dsl door emitting tensorlib scan with a declared combine), forced by audio not ML; until it lands, param-dependent extent loops that scale work superlinearly should be flagged or confined to a debug-grade path — the subset currently admits the naive form the charter's founding example forbids.
16. **[HIGH / CONFIRMED] kv-cache-boundary** — A no-recompile KV-cache decode loop is expressible today with zero new language — preallocated C-contiguous host cache, out=cache[t] row adoption (checked for dtype/rank/contiguity only, out never in the cache key: c.py:315-336, test_grid.py:38,86-91), per-token prefix-slice recapture cache[:t] (C-contiguous, rank-generic fingerprint: arrays.py:99-141, test_arrays.py:30-34), and runtime loop bounds from staged captured ints (base_lang.py:272-294, valuekind.py:181-189, test_array_args.py:76-77) — so tensorlib's recorded KV-cache exclusion (zoo/__init__.py:20, "mutation") is representation-level, not workload-level; but neither stream pins any aliasing rule for out= overlapping a captured/argument view, and an overlapping dispatch corrupts silently; the end-to-end composition is also not yet pinned by any single test.
   - Evidence: tests/test_grid.py:42-49,67-91 (out adoption + contract checks: dtype/rank/contiguity, no overlap check); src/pdum/dsl/stdlib/arrays.py:99-100 (C-contiguous captures; prefix slices qualify); tests/test_array_args.py:30-33,76-77 (extents are staging values, pinned cache hits); src/pdum/dsl/kernel/valuekind.py:181-189 (captured int t fingerprints by type); explorations/tensorlib/tensorlib/zoo/__init__.py:19-21 (recorded exclusion 'KV-cache decode (mutation)')
   - Verifier: Trigger: g.dispatch(kernel capturing cache[:t+1] or full cache, (), out=cache[t]) passes every launcher check and does raw C pointer writes into memory also readable through the captured view — silent order-dependent corruption; the canonical disjoint decode (read cache[:t], write cache[t]) works and compiles once (rank-generic capture hit test_arrays.py:30-34, runtime core.for bound base_lang.py:272-294, staged trip count test_array_args.py:76-77), so the zoo's mutation-based exclusion is representation-level only. Caveat: no existing test composes the full decode loop; that composition is inferred from individually pinned mechanisms. — "backends/c.py:325-330: "if spec.dtype != want: raise VerifyError(...) / if not spec.flags[\"C_CONTIGUOUS\"]: raise VerifyError(...) / out = spec  # adopt: the array's shape IS the domain" — dtype, contiguity, rank (334-335) are the ONLY checks; no overlap test against captured buffers exists anywher"
   - Proposal: Reclassify the boundary: programs stay pure; mutation is confined to the dispatch seam (out=). Pin two tests: (1) the decode loop compiles once under no_compile as t grows; (2) the aliasing rule — refuse or document out= overlapping a captured view, with message 'out= must be disjoint from every captured and argument buffer view in this dispatch'. Update zoo/LEVELS recorded-boundary text to say the exclusion is representation-level, not workload-level.
17. **[HIGH / CONFIRMED] capture-leaf-naming** — No deterministic capture→leaf naming law exists anywhere in either stream: tensorlib Programs and grad identify inputs/gradients by string var name, pdum.dsl's IR and marshaling identify captures purely by position (prefix-tuple + freevar index) even though freevar names survive in Handle.env, and the zoo namespaces layers by hand-written string prefixes — so the binder-seam move (captures become input leaves) and cross-seam gradient identity are undefined; the charter acknowledges this as an open Probe D obligation, and the proposed law (leaf name = freevar-name chain with stage/layer index prefixes, pinned by a rebuild-stability test) is feasible because names are already retained at capture time.
   - Evidence: explorations/tensorlib/tensorlib/ir.py:289-296 (run keyed on input var names); explorations/tensorlib/tensorlib/autodiff.py:867-876 (grads keyed by var names); explorations/tensorlib/tensorlib/zoo/gpt2.py:48-63 (hand 'L{i}.' prefixes); src/pdum/dsl/combinators.py:254-258 (env paths prefixed by stage index — the only existing mechanism)
   - Verifier: Trigger: any frontend emission of a Program from a captured closure must invent string input names with no defined law — tensorlib run() and grad() are name-keyed (autodiff.py:867-876) while pdum.dsl's IR capture identity is positional tuples (lower.py:137, capture.py:90-94), and the zoo's "L{i}." prefixes are hand convention; the charter registers the mapping as an untested Probe D obligation, and no doc in either stream defines it. — "ir.py:294-297 'if ins.var not in inputs: raise KeyError(f"missing input {ins.var!r}")'; lower.py:137 'slot=self.prefix + (idx,)'; gpt2.py:48 'p = f"L{i}."'; charter 140 line 503-505 'captured tensors become `input` leaves of the emitted Program (deterministic capture→leaf naming — the binder-seam mo"
   - Proposal: Define the law at the right level: leaf name = the capture path (freevar name chain through nested Handles/Derived wrappers, prefixed by stage/layer index exactly as build_pipe already prefixes env paths). Make it part of the emission contract of F1's chosen seam, and pin it with a test that a rebuilt closure maps the same capture to the same leaf name.
18. **[HIGH / CONFIRMED] joint-seam-fork** — The joint system's central seam is a real, currently unadjudicated fork: canon 130 stages 2-3 (PROPOSED, unlanded — no tensor.* ops in src/) plan a second tensor IR (tensor.map/reduce/contract/slice satellite ops, 130 §4.2) and a second reverse-mode AD (stage 3 "Grad", 130 §8) inside pdum.dsl, while charter 140 §5 mandates probes front tensorlib's representation; the only declared emission seam from pdum.dsl syntax into tensorlib is the marker-body Node schema (mdsl.py:1-16, README:84-85, build.py:3 "Deliberately not a frontend"), with no Program/Instr-level path existing or planned in 020/130/LEVELS — Program-level emission appears only as an untested hypothesis inside the charter itself (Probe D, charter:502-505), and the charter's own deliverable 8 mis-scopes the integration plan to "Node-schema" granularity. The fork is scheduled for adjudication by this very run (mitigating urgency but not the finding); resolution collides with decided positions in both streams and so escalates to the human per §8.
   - Evidence: 130:130-141 (tensor dialect emits pdum.dsl satellite ops), 130:257-266 (stage 3 'Grad' = adjoints in the pdum.dsl dialect); explorations/tensorlib/tensorlib/build.py:1-7 ('Deliberately not a frontend'); explorations/tensorlib/tensorlib/mdsl.py:1-16 (declared adapter seam is marker-body Nodes only); explorations/tensorlib/README.md:85 claims 'without any rewrite' but only the mdsl schema backs it
   - Verifier: Trigger confirmed: 130 stage 3 plans reverse-mode AD (adjoints of contract/reduce/map) over the pdum.dsl tensor dialect while tensorlib already holds reverse-mode AD over Program, and every declared adapter seam (mdsl.py:1-16, README:84-85) is scoped to the Node schema — marker bodies — never Instr/Program; I verified no Program-level emission path exists in code (grep: no tensor.* ops in src/tests) or in 020/130/LEVELS. One mitigation the finding should carry: the fork is process-acknowledged — 140 was written after 130 (130 is PROPOSED 2026-07-13, stages 2-3 unlanded), and 140 §10 deliverable 8 requires a "frontend→Node-schema integration plan" — but even that deliverable names Node-schema granularity while Probe D (charter:502-505) assumes Program-level emission ("captured tensors become `input` leaves of the emitted Program"), so the substantive which-IR-wins question is decided nowhere; charter §5's paraphrase "the Node/Program schema is explicitly designed for pluggable frontends" (charter:333-334) over-claims the README, which backs only the Node half. — "130 §8 stage 3: "**Grad** (old step 13, resequenced AFTER tensors deliberately) — adjoints of `contract`/`reduce`/`map` are textbook ... the jvp column extends; transpose + partial eval"; 130 §4.2: "`tensor.map` / `tensor.reduce` / `tensor.contract` / `tensor.slice`: satellite ops (surface A)"; READ"
   - Proposal: Adjudicate before 130 stage 3 lands a second AD: either (a) the tensor dialect IS the frontend — its op set is made schema-identical to tensorlib's Program ops and lowering emits Programs, with tensorlib grad/dce/checkpoint as the enrichment passes — or (b) tensorlib is demoted to oracle-only and its AD/L1/L3 machinery is ported, not fronted. Escalate to human per charter §8 (collides with decided positions in both streams).
19. **[HIGH / CONFIRMED] name-genericity** — The 130 §5 flagship tile-DSL sketch (all [planned]) hardcodes concrete axis names in the kernel body (zeros(("m","n")), tiles("k"), contract(axis="k"), .tile(m=tm,k=kb)), so make_gemm as written works only for operands named m/k/n — violating charter 140 §7.4(b)'s hard name-genericity requirement in the exact sketch Probe C's installment will build on; the axis="k" literal is redundant by 130 §4.2's own unique-shared-axis rule (implemented literal-free in src/pdum/dsl/stdlib/transforms.py:349-358), while zeros/tiles/.tile-kwarg names have no stated derivation story. Bounding caveats: names are ordinary Python strings so parameterization is not precluded (the flaw is the modeled style plus the missing derivation story, not an impossibility), and 130 predates 140, making this a §8-governance collision to escalate rather than a shipped-code bug.
   - Evidence: 130:186-192 (name literals in the GEMM body); charter 140:589-596 ('a function must never work only for one concrete name'); the derivation logic that makes literals unnecessary already exists — matmul pairs the unique shared axis from the operand types with no name literal (src/pdum/dsl/stdlib/transforms.py:349-357).
   - Verifier: Trigger: call make_gemm with operands whose axes are not literally named m/k/n — the body's zeros(("m","n")), tiles("k"), contract(axis="k"), and .tile(m=tm,k=kb) all fail, contradicting 140 §7.4(b)'s hard requirement; the contradiction is even internal to 130, whose §4.2 already states the literal-free unique-shared-axis derivation (and transforms.py:349-358 implements it) while §5's flagship sketch writes axis="k" anyway. Charter §2.4 (step-14 attention item) establishes that flagship samples are judged against principles, closing the 'just an example' escape. — "130:186-192: 'acc = zeros(("m", "n"), f32) ... for kb in tiles("k"): a = stage(A.tile(m=tm, k=kb), pad="conflict-free") ... acc = acc + contract(a, b, axis="k")'  //  140:594-596: '(b) **name-genericity** — axis names are domain-specific, so a function must never work only for one concrete name.'  /"
   - Proposal: Tile/contract axes are derived from the captured operands' Named types (contract(a, b) with the unique-shared-axis rule; zeros_like from the result type); name literals permitted only to break genuine ambiguity (two shared axes) or fix output order. Names-in-types makes name-generic kernels specialize per name through FnType with zero new mechanism.
20. **[HIGH / CONFIRMED] tile-dsl-reduce-erasure** — 130 §5's centerpiece GEMM writes the blocked-k reduction as a kind="seq" carried accumulation (`for kb in tiles("k"): acc = acc + contract(...)`, 130:186-192), erasing the inter-block associative-combine structure that 130's own working-representation principle (130:28-31) and its own diagnosis of seq loops (130:52-55) require to stay declared, and that §4.2's reduce-kind loops/comprehensions (130:123-141) exist to express — so any reassociating rewrite below the tile DSL (e.g. split-k under 140 §3.2's license-relative chains) must recover the reduce rather than read it; the fix is to state the k-loop as a reduce-kind comprehension before stage 4 implements the sketch as written.
   - Evidence: 130:186-192 (the GEMM sketch's k-loop with carried acc) vs 130:28-31 (named reduce is the working representation until target selection) and 130:123-141 (§4.2 defines reduce-kind loops and `sum(... for k in axis(...))` comprehensions for exactly this); LEVELS.md:79-83 states lowering must preserve reduce/scan structure; the probe's chain step R3 (split-k) is licensed by reassociativity, which is only statable against a declared reduce.
   - Verifier: Trigger: any post-tile rewrite that reassociates across k blocks (split-k, tree accumulation) must recover the combine from a carried seq accumulation — the exact pattern-recovery 130 §1 names as the defect of seq loops, and the §5 comment ("sequential BLOCKED loop") plus §4.2's default kind="seq" (130:127-128) make the erasure explicit on the doc's own terms; §4.2 provides the reduce-comprehension surface (130:138-141) the sketch declines to use, and §9's open questions never exempt the k-loop. One caveat for the report: the LEVELS.md:79-83 citation ("preserve reduce/scan structure on **machine-bound** dims") supports only weakly, since the blocked-k loop is sequential-time, not machine-bound — 130's own text is the load-bearing evidence. Best available defense (target selection already happened at `kind="gpu.tile"`, so the seq loop is the author's committed schedule) fails because §6/§8-stage-4 place a further descent (warp decomposition, mma selection) below the tile kernel, and split-k is a schedule choice at exactly that lower tier. — "130:187,191: `for kb in tiles("k"):  # sequential BLOCKED loop over axis k` ... `acc = acc + contract(a, b, axis="k")` — vs 130:28-31: "Named `contract` / `reduce` / `map` are the working representation until target selection. The scalar core is the FLOOR for scalar targets — never a mid-level that "
   - Proposal: Rewrite the centerpiece: blocked-k is a reduce-kind comprehension (`acc = sum(contract(...) for kb in tiles(k, TK))`) elaborating to reduce with combine=add; sequential staging of tiles remains the backend's schedule, not the surface's semantics. Fix the doc before stage 4 implements the sketch as written.
21. **[HIGH / CONFIRMED] lowering-registry-fingerprint** — The §3.4 registry of certified lowerings cannot be built on today's artifact identity: the sole content-addressed fingerprint is Node.key over the fully-lowered scalar region (registry.py:190-191, ir.py:13-14), computed after matmul's contraction structure is erased to core.for/mul/add (transforms.py:376-396) and at whole-region rather than chunk granularity; named-level-equivalent chunks with different scalar elaborations therefore key differently, matching scalar regions to tile lowerings would violate the no-unlowering constraint both streams hold (130:22-31, 140:477-482), and no existing doc (130 §8, cache.py:8-11, 140 §7.7 which covers only specialization-layout skeletons) places a fingerprint at the named-op level — so a named-op-level chunk keyspace is an unstated prerequisite of §3.4 belonging in the L4 design brief.
   - Evidence: matmul expands directly to scalar core.for mul/add at lowering with no contraction node ever existing (src/pdum/dsl/stdlib/transforms.py:381-396); the artifact tier is content-addressed on the scalar IR's Node.key; charter §3.4 (140:289-301) requires 'content-addressed fingerprint of the IR chunk' at a level where 'attention chunk normalizes to the same fingerprint'; 130's own principle puts the working representation at named contract/reduce/map (130:28-31) but neither 130 §8 nor any cache doc places the fingerprint there.
   - Verifier: Trigger confirmed: the only fingerprint in the system is region.key, computed on the fully-lowered scalar IR (registry.py:175→191) after _matmul has erased contraction structure (transforms.py:376-396), and it is whole-region only — no chunk-granular keyspace exists; grep finds zero occurrences of "fingerprint"/"chunk" in 130, so no doc relocates the key to the named-op level, and the no-unlowering constraint (140:477-482, 130:28-31) forbids recovering the structure from the scalar key. — "transforms.py:396: `return ctx.emit("core.for", lo, hi, zero, regions=(Region(params=(iv, acc), body=(y,)),), node=node)` — matmul expands to a scalar loop at lowering; ir.py:13-14: "``Node.key`` is a memoized sha256 **content key** — the artifact-tier cache key"; registry.py:190-191: `artifact = se"
   - Proposal: Define chunk_fp = hash of the named-op Program in tensorlib canonical() layout form (the no-unlowering line, which is also charter §7.7's structural-skeleton sketch); the artifact tier gains a second keyspace (chunk_fp -> chain + certified artifact). This is a prerequisite of §3.4, so it belongs in the L4 design brief, stage-ordered before the tile family lands.
22. **[HIGH / CONFIRMED] no-dispatch-sequencing-time-loop** — No construct in the joint system holds the PDE time loop or ping-pong swap: Pipeline fuses exactly one value through unary stages into a single artifact (combinators.py:262-272) and its planned execution is fuse-then-launch, not sequencing (:220-221); no dispatch-sequencing construct exists or appears in the 020 step-14 remaining list (020:324) or 130 §4.3 (130:143-160); tensorlib's fold has exactly the needed carry contract with multi-state support (ir.py:159-181; physics.py fdtd1d state=("E","H"), carry={"E":E1,"H":H1}) but is unreachable from pdum.dsl (no fold emitter in src/pdum/dsl); and since L2 bufferization is deferred (140:712-713), the "mutation is storage-level, syntax stays pure" stance has no operative mechanism at the joint surface — host Python owns the loop and the user hand-manages the swap. The gap is pre-registered as open in charter §7.3/Probe B (140:435-438, 575-582), so the finding's contribution is the two-concepts answer and the fold-mirroring dispatch-level sequencer, not the discovery of the gap.
   - Evidence: combinators.py:262-272 (one value, unary stages); 020_implementation-plan.md:324 remaining-list and 130 §4.3:143-160 contain no dispatch sequencing; tensorlib fold exists with carry contract ir.py:159-181 and two-state PDE use physics.py:53-64,116-127; charter §7.3 poses the question at 140:575-582
   - Verifier: CONFIRMED: writing Probe B's ping-pong time loop today triggers it — build_pipe raises VerifyError for anything but one unary-chained value (combinators.py:262-272), no construct in src/pdum/dsl or the 020/130 plans records a dispatch sequence (grep hits only charter §7.3 posing the question), and tensorlib fold — which carries exactly the needed contract, two-state included (physics.py fdtd1d state=("E","H")) — has no emitter in pdum.dsl; with L2 bufferization explicitly deferred (140:712-713), the pure-syntax stance has no joint-surface mechanism and the user hand-manages the swap. Caveat: the gap is charter-pre-registered as an open question (§7.3, Probe B 140:435-438), so this is a confirmed structural gap plus a proposed answer, not a silent-wrong-behavior finding. — "combinators.py:263 "raise VerifyError(f\"pipe threads exactly one value; got {len(arg_types)} arg types\")"; ir.py:177-178 "# Sequential by definition — the reference semantics of time-stepped state # (PDE leapfrog, linear-attention/SSM matrix states)."; 140:580-582 "a pipeline that fuses when it ca"
   - Proposal: A host-level fold/sequence combinator mirroring tensorlib fold's carry contract (state names, fixed-layout carries, steps=T) as the dispatch-level pipeline — the §7.3 answer is TWO concepts, not one; this construct is also the exact seam L2 bufferization later consumes to assign alternating buffers. Until it lands, the boundary statement: host Python owns time loops, the swap is visible, and the finding-1 alias refusal is mandatory to keep the raw form inside the subset.
23. **[HIGH / CONFIRMED] out-aliasing-silent-corruption** — A C-backend grid dispatch whose out= array aliases a captured read buffer silently returns corrupted results (verified: right-shift stencil over an aliased 8-element array yields all zeros instead of the shifted values) because make_grid_launcher (backends/c.py:315-338) validates dtype, contiguity, and rank but never checks memory overlap between out and the buffer leaves, which are passed as raw pointers (c.py:268-277) into an in-artifact domain loop — violating the refusal-first mandate (charter 140:47-48).
   - Evidence: Verified by execution (scratchpad/probe_b_pingpong.py check2); launcher validates dtype/contiguity/rank only at backends/c.py:326-335; buffers enter as raw pointers c.py:268-277; refusal-first mandate charter 140:47-48
   - Verifier: Trigger reproduced by re-execution this session: passing the captured read array as out= to a right-shift grid kernel returns all zeros with no error; grep confirms no shares_memory/alias guard anywhere in src/pdum/dsl, and docs/design/100_arrays-and-axes.md:148 lists aliasing rules as still-needed future work, so nothing guards or decides this elsewhere. — "c.py:326-335 validates only dtype/contiguity/rank then adopts the array ("out = spec  # adopt: the array's shape IS the domain"); reproduced this session: "check2: aliased out silently wrong: True got: [0. 0. 0. 0. 0. 0. 0. 0.] expected: [0. 0. 1. 2. 3. 4. 5. 6.]""
   - Proposal: Refuse at launch when out overlaps any buffer leaf (np.shares_memory over the leaves channel, once per dispatch, cheap vs the domain loop) with a designed message quoting the fix ('write into a second buffer — ping-pong'); long-term, in-place legality is an L2 bufferization decision informed by tensorlib's exact footprint/overlaps algebra (DESIGN.md white-box aliasing; REPRESENTATIONS.md:40-45), never a user gesture.
24. **[HIGH / CONFIRMED] fragment-color-results** — The demo fragment family (simple_shader.fragment — plain "fragment" is explicitly reserved for a not-yet-built real family, wgsl.py:453-456, 010:679-681) cannot express color results end to end: tuple returns refuse loudly on the WGSL renderer (VerifyError, no core.tuple rendering), the fragment artifact hardcodes scalar-broadcast-to-grayscale and a '-> f32' render signature, and in-kernel record construction refuses (base_lang.py:390-392, not :383-384 as cited, for a global class name; surfaces.py registers methods but no constructor) — while the docstring (wgsl.py:11), the taught ch10 chapter, and pack.py:94-95 still present colors/vec4 as having arrived with tuples at step 10, a promise step 10 did not honor. The refusals are loud (charter-compliant); the confirmed defect is the stale first-order contract in current docs plus unreachable Vec-result machinery, with severity tempered by demo scope.
   - Evidence: wgsl.py:11 and 010:695-696 ('colors wait for tuples, step 10') vs wgsl.py:342-344 (scalar broadcast to grayscale rgba, current source) and wgsl.py:98-100 (render signature is always '-> f32'); wgsl.py:142-148 (empty CODE_FOR_OP + no core.tuple case -> VerifyError; the tuple-retirement decomposition only folds extract-of-tuple, stdlib/__init__.py:40-43, so a returned tuple reaches the renderer and refuses); base_lang.py:358-392 (no record-constructor rule; a record class name lowers as a captured value -> 'value here, not callable' :383-384); surfaces.py:89-91 (record fields limited to float/int/bool); ResultPlan tuple rebuild already exists on the host side (pack.py:106)
   - Verifier: Trigger reproduced on live source with GPU available: tuple-returning simple_shader.fragment kernel refuses at render (empty CODE_FOR_OP wgsl.py:425, no core.tuple case wgsl.py:142-148; retirement decomposition stdlib/__init__.py:41-43 only folds extract-of-tuple); scalar-grayscale hardcode and '-> f32' signature (wgsl.py:98-100, 342-344) persist while the docstring, ch10 chapter (build_chapters.py:2614-2615), and pack.py:94-95 ("a fragment shader yields vec4") all present colors/vec4 as arrived at step 10 — steps are past 14 and the step-9 deviation (010:695-696) was never closed; Python backend meanwhile renders tuple results fine (python.py:99-100, verified (3.0, 6.0)), so the ResultPlan rebuild half already works. — "wgsl.py:11 "result broadcast to grayscale rgba (colors arrive with tuples, step 10)" vs wgsl.py:343-344 "let v = kernel_body(pos.x, pos.y);\n  return vec4f(v, v, v, 1.0);" — and live run: a fragment kernel returning (r,g,b) raises "VerifyError: wgsl backend has no rendering for 'core.tuple'"; record"
   - Proposal: Close the loop the plan already opened: composite fragment results as vec4/record through the existing ResultPlan rebuild, a vec4 output contract on the fragment family, and record construction as the missing half of surface C (constructor call rule emitting core.tuple with a Record type). Until then the family docstring and chapter should stop citing step 10 as the arrival point — the promise is three steps stale.
25. **[HIGH / CONFIRMED] derivative-ambient-param** — D's contract — differentiate wrt ALL root params by position, refuse (lazily, at cone-reach) any non-float param — makes the same kernel's fwidth compute on f64-param families (default python, wgsl compute per wgsl.py:72) but raise MissingRule on the C grid family (i64 coords, c.py:299), verified by execution; fwidth/ddx/ddy (graphics.py:45-55) are correct only when params 0,1 happen to be the two pixel coordinates; and the mechanism guarantees refusal for Probe A's planned fragment shaders whenever the differentiated value depends on a varyings record or other non-float param. No test exercises D on the grid or wgsl families. A wrt-less all-params D is family-dependent in meaning and availability, supporting the charter §7.5 ambient-parameter replacement.
   - Evidence: stdlib/transforms.py:312-335 (_d_operator loops over every root param) and :326-327 (seed refuses non-float params with MissingRule); demo/graphics.py:45-55 (ddx=D(v)[0], ddy=D(v)[1], fwidth=|d0|+|d1| — positional, only correct when params are the two pixel coords); tests/test_transforms.py:263 (comment: 'fwidth needs two coordinates; this kernel has one'); wgsl.py:72 ((f64,)*argcount) vs backends/c.py:299 ((i64,)*n) — D works on one compute family, refuses on the other; no test exercises D/fwidth on the wgsl fragment family (test_transforms.py runs all D tests on the default python backend)
   - Verifier: Trigger reproduced by execution: the identical kernel source computes fwidth on the f64-param family and raises MissingRule on the C grid family (i64 coords) — an equal-citizens divergence with zero test coverage of D on grid/wgsl families (grep confirms). One refinement: the non-float refusal fires lazily only when the differentiated value's cone reaches the non-float param (transforms.py:62-73 seed at core.param leaves), so "any fragment shader with varyings refuses outright" holds for values that use the varyings (the normal case), not unconditionally; the fragment family itself is [planned] (Probe A), so that half is mechanism-verified extrapolation rather than executed today. — "transforms.py:313,320: "argc = len(ctx.root.params) ... for j in range(argc)"; transforms.py:327: "raise MissingRule(f\"D differentiates w.r.t. float params; param {k} is {t!r} ...\")"; wgsl.py:72: "return (f64,) * pyfunc.__code__.co_argcount" vs c.py:299: "return (i64,) * (t.pyfunc.__code__.co_argc"
   - Proposal: Adopt the charter §7.5 ambient-parameter rule at the family level: a kernel family declares its distinguished ambient coordinates (with axis names), fwidth/ddx/ddy default wrt=ambient, and D(v, wrt=...) selects params, captures, or ambient axes explicitly — making the operator well-defined for fragment shaders with varyings and identical across compute families. This confirms and sharpens the charter's D-replacement candidate; a wrt-less all-params D should not survive into the vertex/fragment era.
26. **[HIGH / CONFIRMED] render-loop-ownership** — The design-committed 070 §4 frame contract (draw(target) = acquire fresh swapchain view, begin/end its own render pass, submit) and today's FragmentProgram artifact (owns encoder, offscreen target, pass, submit, and a blocking readback; wgsl.py:369-402) both place frame/pass/submit ownership on our side, directly contradicting charter Probe A's requirement (140:412-414) to contribute draws into a frame owned by foreign code — an overlay cannot encode into our closed pass, and our swapchain acquisition violates the stated ownership rule; the encode-vs-submit seam the fix needs already exists in ComputeProgram._encode_frame (wgsl.py:250-265, submit separate at :288) but is absent from the fragment path and from any design doc. (Caveat: strict impossibility of coexistence is overstated — a host could add a second LoadOp.load pass after our submit — but the ownership contradiction with 140:412-414 stands as written.)
   - Evidence: 070_backends-notes.md:143-150 ('draw(target) = fresh swapchain view -> render pass -> set pipeline/bind group -> draw(3) -> submit'); demo/simple_shader/wgsl.py:381-399 (FragmentProgram.__call__ creates encoder, begins/ends render pass, submits, reads back); charter 140:409-417 (host owns loop/swap chain/GPU state; imgui coexists)
   - Verifier: Trigger: charter 140:412-414 requires contributing draws into a foreign-owned frame ("we never own the frame, the swap chain, or GPU global state"), while design-committed 070 §4 has draw(target) acquire the swapchain view, own the render pass, and submit — and today's FragmentProgram.__call__ additionally owns the encoder, an offscreen target texture, and a blocking readback; no doc or source anywhere provides a foreign-pass/render-bundle seam (grep for bundle/draw_into/imgui hits only the charter). — "070_backends-notes.md:145-146: "`draw(target)` = fresh swapchain view → render pass → set pipeline/bind group → draw(3) → submit" | 140:412-414: "the render loop belongs to foreign code: an imgui overlay (or equivalent) runs in the same loop and must keep working. We contribute *dispatches into* a f"
   - Proposal: Redefine the per-frame deliverable as an encodable, not an executed frame: pso.record(...) returning a WebGPU render bundle (or a draw_into(pass_encoder) callable) recorded against a declared target-format contract, with pipeline/bind-group state staying artifact-tier in FastRecord exactly as today (registry.py:195-201); submit, pass boundaries, and the swap chain belong to the host. The compute analogue (ComputeProgram._encode_frame, wgsl.py:250-265) already separates encode from submit — promote that seam to the API instead of burying it.
27. **[HIGH / CONFIRMED] graphics-scheduling** — The founding domain has no scheduled installment: the draw(target) window surface — originally in step 9's scope (020:236) — shipped offscreen-only and is deferred as "next graphics step" with no step number (010:698-699, repeated at 010:857 and build_chapters.py:2658/3153); 020's Phase IV table (steps 11-17) contains no graphics step and its "Later, unscheduled" list's only graphics-adjacent entry is the anywidget canvas; vertex-side machinery (vertex family, varyings, MRT, instancing) appears in no *scheduled* plan text — only 070:27-28's "vertex I/O (WGSL/MSL only)" phrase, 070:14's target list, and R12/R15 research notes — while steps 14-16 are tensor/grad/backend+GEMM machinery.
   - Evidence: 010_proposed-architecture.md:697-699 ('fragment renders offscreen only; the draw(target) window surface is 070 §4's committed design, next graphics step'); 020_implementation-plan.md:317-327 (Phase IV table steps 11-17: arrays, transforms, seams, tensors, grad, CUDA/Metal, units — no graphics step); build_chapters.py:2658-2660 ('the interactive window demo returns with the graphics draw(target) surface'); the only vertex-side plan text anywhere is one phrase 'texture sampling, vertex I/O (WGSL/MSL only)' at 070_backends-notes.md:27-28; grep over docs/design + src for varying/instanc/MRT/vertex-array hits only the charter itself
   - Verifier: Trigger is concrete and double-deferred: the window/draw surface was in step 9's original scope (020:236 "offscreen + window runtime"), slipped ("fragment renders offscreen only"), and no plan artifact reschedules it — the Phase IV table and the "Later, unscheduled" list both lack a graphics step, and 010:857 defers it a second time with the readback-latency fix riding on it; vertex family/varyings/MRT/instancing have no scheduled home anywhere (only 070:28's one phrase and R12/R15 research matrices). — "020_implementation-plan.md:317 "## Phase IV — width (steps 11–17 ...)" table rows 11-17 = arrays/C, vmap+jvp, seams+over, tensors, grad, CUDA+Metal+TILE, units (no graphics row); 020:357 "**Later, unscheduled** ... notebook/anywidget live canvas ..."; 010_proposed-architecture.md:698-699 "fragment r"
   - Proposal: Schedule the graphics step explicitly (vertex family + draw-into-pass surface + composite fragment results) as a named installment with a chapter gate, before or parallel to step 16; if graphics is deliberately post-M4, record that decision and its reason in 020 — breadth-is-identity (charter §1) makes silent deferral of the founding domain a charter violation.
28. **[HIGH / CONFIRMED] kink-subgradient-divergence** — The two streams have contradictory and mutually non-transpose at-kink derivative conventions: pdum's JVP rules pick one side at abs/min/max kinks via core.select (transforms.py:210-227) but no test pins this (test_jvp_rules.py:94 deliberately evaluates away from kinks, and no other pdum test covers ties/abs(0)/branch boundaries), while tensorlib pins max-tie as full-cotangent-to-every-tied-element ([1,1,0], summing to 2 — not any convex-combination subgradient); the charter (140, §7.5 differentiation section) explicitly leaves pdum's convention unadjudicated, so the joint system currently has no coherent subgradient contract for the forward-mode convex-optimization consumer.
   - Evidence: tests/test_jvp_rules.py:94 (POINT chosen away from kinks) vs explorations/tensorlib/tests/test_autodiff.py:377-387 (test_reduce_max_tie_overcount_is_pinned); charter §7.5 'today's jvp rules already return *a* subgradient at kinks ... accident or a commitment' (140 charter :617-621)
   - Verifier: Concrete contradiction: pdum's max JVP (src/pdum/dsl/stdlib/transforms.py:210-227, _r_minmax("ge") emitting core.select) routes the full tangent to ONE tied argument ([1,0] at a tie), while tensorlib's reduce-max VJP pins full cotangent to EVERY tie ([1,1,0], sum 2, not a convex-combination subgradient); these linear maps are not transposes of each other, and no pdum test probes any at-kink point — the only POINT is deliberately away from kinks — so pdum's side rests solely on unpinned code behavior the charter itself calls "accident or a commitment". — "tests/test_jvp_rules.py:94 "POINT = (1.7, 2.3)  # away from kinks (abs/min/max/floor/branch switch elsewhere)"; explorations/tensorlib/tests/test_autodiff.py:377-387 "test_reduce_max_tie_overcount_is_pinned ... # documented caveat: every tied element receives the full cotangent ... np.testing.assert"
   - Proposal: Pin pdum's at-kink values (abs(0), min/max ties, branch boundary) in test_jvp_rules, then adjudicate one convention across streams per §2.4 rule-table-vs-derived; the convex-optimization probe (E) needs a real subgradient contract, so the tie-overcount cannot stay a per-stream caveat.
29. **[HIGH / CONFIRMED] matmul-silent-ub** — matmul over Named arrays with mismatched shared-axis extents silently returns the wrong value (A's inner extent drives the trip count), and test_refusal_contract.py:116-137 freezes that silent wrong answer as spec inside the refusal-contract battery itself — colliding with charter §1 refusal-first; the guard that would catch it (transforms.py:369-372) is dead code (matmul requires NamedArray, Shaped+Named doesn't exist), the extents are available at capture time (arrays.py:76-85) so a stage-time refusal is feasible without splitting the cache, and since the UB posture is a decided position (130 §4.1, attributed to 100), the collision must be escalated to the human per charter §8 rather than left frozen.
   - Evidence: tests/test_refusal_contract.py:116-137 (assert cell(0, 0) == 3.0); charter §1 corollary 'Refusal-first is load-bearing' (docs/design/140_critical-assessment-charter.md:46-48)
   - Verifier: Trigger is concrete and frozen: Named (2,3)x(5,4) over shared axis "inner" returns 3.0 with no refusal, and the refusal-contract battery asserts that number as spec; the existing extent guard (transforms.py:369-372) is unreachable because matmul requires NamedArray and Shaped+Named does not exist (test line 129), while Named exposes .shape at capture time (arrays.py:83-85) so a stage-time check is feasible. One correction to the candidate: the decided posture lives in 130 §4.1 (docs/design/130_tensors-tiles-and-over.md:108-110, which attributes it to 100), not verbatim in 100 — the §8 escalation still applies. — "tests/test_refusal_contract.py:137: "assert cell(0, 0) == 3.0  # sums A's inner extent (3), documented UB for mismatch"; charter :47-48: "Refusal-first is load-bearing. Every flaw claim of the form \"X silently does the wrong thing\" outranks \"X is missing.\"""
   - Proposal: Add a shared-axis extent check when both operands carry static extents (Named/Shaped) and refuse with a D17-style recipe naming both extents; if runtime extents make this genuinely uncheckable at trace time, state that boundary explicitly instead of pinning the wrong number. Escalate per §8: the test freezes a decided 100 posture.
30. **[HIGH / CONFIRMED] mutation-stance-triangle** — Three mutation stances coexist with no joint decision: pdum.dsl's grid family exposes exactly one mutable `out` with all arrays read-only captures (test_grid.py:2, arrays.py:23, 130:106-107,153); tensorlib defers all mutation/storage to L2 bufferization ("commit late", PHILOSOPHY.md:54-58); and KV-cache decode is excluded as "mutation" by tensorlib's recorded boundaries (LEVELS.md:140-141, zoo/__init__.py:19-20) while pdum.dsl has no decode story either — a gap the charter itself pre-registers (140:519-523, §7.1) and charges Probes B/D to close, so the finding is a confirmed open collision already on the run's docket rather than a newly found contradiction.
   - Evidence: docs/design/130_tensors-tiles-and-over.md:105,153 (out=(W,H) launch-domain contract); explorations/tensorlib/PHILOSOPHY.md:54-59 ('Transform where it is safe; commit late'); explorations/tensorlib/zoo/__init__.py:19-21 and LEVELS.md:140-141 (KV-cache decode: mutation — recorded boundary); charter §7.1, Probe D KV-cache clause (docs/design/140:519-523)
   - Verifier: Confirmed by direct quotes from all three positions plus the charter's own admission that "Two independent deferrals do not add up to a decision" (140:522); caveat: the proposal is not novel — charter §7.1 and Probe D already assign exactly this forcing obligation to the run, so the finding's value is as a verified gap statement, not a new discovery. Candidate's zoo citation path corrected to explorations/tensorlib/tensorlib/zoo/__init__.py:19-20. — "LEVELS.md:140-141: "Recorded boundaries (deliberate exclusions): MoE routing / top-k (data-dependent gather), KV-cache decode (mutation), dynamic shapes." | PHILOSOPHY.md:57-58: "Storage (buffers, in-place, offsets) is deliberately LAST, adopted only when rewriting stops." | tests/test_grid.py:2: "O"
   - Proposal: Probes B/D must force the verdict: either the L2 bufferization story absorbs `out` and ping-pong (surface stays pure, storage alternates underneath) and KV-cache gets an L2-level append/ring-buffer story, or the boundary is drawn explicitly: 'mutating decode is outside the subset; refuse with: KV-cache decode requires storage-level mutation (L2); express the step functionally or use the recorded exclusion.'
31. **[HIGH / CONFIRMED] rule-table-growth-polarity** — pdum.dsl and tensorlib both bottom out in hand-written scalar derivative rules, but their decided growth policies are opposite: design 110 §3 declares the JVP column open to per-op registration and the transforms.py:143 refusal actively directs users to register hand rules, while tensorlib's PHILOSOPHY.md:73-78 and mdsl.py:18-22 declare hand gradient tables frozen, routing new ops through defmarker composites whose partials are derived by tree rewriting (autodiff.py:274-283); left coexisting, the joint system's default door for a new differentiable op is the hand-table registration tensorlib's discipline forbids — a documented decided-position collision (charter §2.4/§8) requiring adjudication, not a design-taste difference.
   - Evidence: src/pdum/dsl/stdlib/transforms.py:141-143,217-228 (JVP_RULES + open-registration refusal) vs explorations/tensorlib/PHILOSOPHY.md:73-78 ('Hand-maintained rule tables are where semantic rot begins; we let them stop growing'), explorations/tensorlib/tensorlib/autodiff.py:225-289 (hand primitive chain + derived composites + loud refusal), mdsl.py:18-25
   - Verifier: Trigger: any user adding a differentiable custom op today is told by the pdum.dsl refusal (transforms.py:143) to register a hand JVP rule — the exact move tensorlib's PHILOSOPHY names as where semantic rot begins — and both polarities are DECIDED positions (110 §3 vs PHILOSOPHY/mdsl), making this a §8 collision requiring human adjudication, which charter §2.4 pre-registers as unresolved ("must be adjudicated, not left to coexist by silence"). Two refinements verified against source: (a) the literal JVP_RULES dict (transforms.py:217-228) holds only 8 math primitives — the "~20" figure includes the inline structural cases at transforms.py:60-140 (mul/div/pow/cast/select/tuple/if/for etc.); (b) tensorlib's own refusal (autodiff.py:285-288 "add one to pw_rule or declare it in _GRADIENT_FREE") also names hand-table growth for true primitives, so the asymmetry is in the default path for NEW ops (pdum: table registration is the only offered door; tensorlib: defmarker composites derive partials automatically, mdsl.py:17-22, so the refusal is reserved for genuine new leaves), not in refusal wording alone — the proposal's fix (repoint transforms.py:143 at a derived path) should acknowledge tensorlib still permits leaf additions. — "transforms.py:143: `raise VerifyError(f"no jvp rule for {op!r} — register one (the transform column is open)")`; design 110 §3: "registered in a table the way spellings are (custom ops bring their own jvp rule — surface A grows a column, not a mechanism)"; PHILOSOPHY.md:77-78: "Hand-maintained rule "
   - Proposal: Adjudicate (charter §2.4 pre-registered): freeze JVP_RULES at the ~20 scalar primitives as the declared leaf table (defensible per charter) and route all new differentiable ops through a defmarker-style derived path in the joint frontend; make the refusal message at transforms.py:143 point to the derived path, not to table registration.
32. **[HIGH / CONFIRMED] workhorse-status** — No canon doc (010/020/090/100/110/130) assigns per-element host dispatch the explicit status charter §1 demands (first-class, debug-only, or removed), even though it is the taught ch14 benchmark baseline (build_chapters.py:4108-4113), the cell-by-cell test oracle (charter 140:54), and canonically condemned as "the wrong execution shape" (130:46-48); one candidate citation is misattributed — "never claims the default route" is 010_proposed-architecture.md:933 (C-target ledger), not 100 §5 — and the proposed fix (a one-paragraph status ruling in 090 §4 or a 010 ledger entry, most plausibly debug/oracle-grade with grid/domain dispatch as the sole performance-legitimate path) is exactly the charter-mandated deliverable.
   - Evidence: scripts/book/build_chapters.py:4108-4113 (per-lane scalar dispatch as the ch14 benchmark baseline); 010_proposed-architecture.md:938-942 (ray-march verdict: 'per-pixel-per-call is the wrong granularity'); 100_arrays-and-axes.md §5 (scalar C 'never claims the default route'); no canon doc states a status.
   - Verifier: Trigger: canon simultaneously condemns per-lane dispatch ("wrong execution shape", 130:46-48; "wrong granularity", 010:938-942) and institutionalizes it (ch14 baseline at build_chapters.py:4108-4113; oracle role per charter 140:53-55) while a grep across 010/020/090/100/110/130 finds no first-class/debug-only/removed ruling — the nearest crumbs (130 §4.3 weaving "as the reference lowering"; 010:933 C "never claims the default route") are lowering/routing facts, not a status. The charter registers the gap and delegates the ruling to this run, so this finding is the intended discharge, not a duplicate decision. — "140_critical-assessment-charter.md:53-57: "Face the workhorse honestly. Per-element host dispatch of scalar kernels exists today, is the test suite's cell-by-cell oracle, and is the baseline in the step-14 gate. The assessment must assign it an explicit status — first-class, debug-only, or removed —"
   - Proposal: Add a one-paragraph status ruling to canon (likely 090 §4 or a 010 ledger entry): scalar per-element dispatch is debug/oracle-grade, named as such in the book, with the grid/domain path as the only performance-legitimate invocation — or explicitly declare it first-class and defend it against charter §1.
33. **[HIGH / CONFIRMED] kind-on-registry-not-kernel** — Execution family is a property of the dispatching registry plus the backend's structural inspection of the target, not of the Handle: the declared kind string is consulted only as a routing key into registry.routes with silent fallback to the default backend (registry.py:107,121-124,133), is never validated anywhere (capture.py:149-151), and carries no family semantics — the grid backend derives its domain contract from the Handle's argcount and Over-chain structure, ignoring kind entirely (c.py:290-299) — so the same kind="device" Handle is a per-cell scalar kernel through one registry and a whole-domain grid kernel through another (test_grid.py:21-24,158-167), a configuration the charter itself pre-registers as an unresolved obligation (§2.4, charter lines 196-199).
   - Evidence: registry.py:112-124 (routes, default claiming); c.py:385-389 (install_grid retargets); test_grid.py:21-24,159-167 (same Handle, two families); capture.py:150-152 (kind uninterpreted at capture); api.py:24 (kind='device' default); charter §2.4 first obligation
   - Verifier: Trigger demonstrated in-repo: test_grid.py:158-167 dispatches the identical Handle chain (kind="device" via api.py:24 default) as a whole-domain grid kernel through grid_registry() (test_grid.py:21-24, install_grid default=True at c.py:385-389) and as per-cell scalar through a scalar-C registry; c.py:290-299 derives the grid domain contract from Handle structure without ever reading target.kind, and capture.py:149-151 states kind is unvalidated at capture. — "registry.py:107 `self.routes: dict[str, str] = {}  # kind -> backend name; absent kinds use the default`; registry.py:114 `self.routes.update(dict.fromkeys(kinds, backend.name))  # roles ship WITH their backends`; capture.py:150-151 `The role, interpreted by dialects/backends ("device", "fragment", "
   - Proposal: Adjudicate per §2.4: adopt CuTe's refinement — roles by context is fine, but the launch boundary and the domain contract should be declared properties of the callable (on Handle/DerivedValue), with the registry choosing only the backend, not the execution semantics.
34. **[HIGH / CONFIRMED] default-family-polarity** — The scalar per-call execution family — loopable per element from the host, the shape of the founding example's forbidden form — is the DEFAULT registry's routed default (the demo Python reference backend installs itself with default=True at package import), carries no debug-grade marking or refusal, and is exercised as a live per-lane path in the step-14 gate; the efficient one-dispatch grid family is explicit opt-in via extend()+install_grid. The polarity of charter §1 is inverted in the shipped wiring, with the softening facts that the default target is a CPU reference backend (not a GPU launch) and c.py:396 marks the routing topology as interim pending step-14 tier-1 dispatch; charter §1's "face the workhorse" corollary confirms no status decision exists.
   - Evidence: python.py:182-185 (install(DEFAULT, default=True)); c.py:392-399 (C never claims default unless asked); c.py:385-389 (install_grid is explicit opt-in); test_grid.py:163-165 (per-lane triple-loop dispatch as live baseline); charter 140 §1:33-57
   - Verifier: Trigger: importing any pdum.dsl module wires the scalar per-call Python backend as DEFAULT's default (demo/simple_shader/python.py:182,185 via __init__.py:18-19); a user host-looping a scalar Handle over an array gets per-element dispatch with no refusal, while the one-dispatch grid family requires explicit extend()+install_grid (c.py:385-389, "explicit choice"); test_grid.py:163-165 runs the per-lane triple loop as a live path. Nuance: the routed default is the CPU reference target, not a GPU launch, and c.py:396 marks the topology interim ("until the device axis brings tier-1 dispatch at step 14") — so the accurate statement is polarity inversion of the founding FORM, not a literal shipped per-pixel GPU loop. Charter §1:53-57 confirms the status question is open, not decided. — "python.py:182 "registry.register_backend(replace(PYTHON, code_for_op=dict(PYTHON.code_for_op)), default=True)" + :185 "install(DEFAULT)"; c.py:385-386 "def install_grid(registry, *, default: bool = False, kinds: tuple = ()) -> None: \"\"\"The GRID family record: explicit choice, like the scalar C ta"
   - Proposal: Assign the workhorse an explicit status per charter §1: either mark the scalar per-cell family debug-grade (a named 'oracle' kind that refuses silent use as a production path) or make a domain-shaped family the default for kernels with array captures; the status decision is the deliverable, not a code band-aid.
35. **[HIGH / PLAUSIBLE] contract-primitive-vs-mac-recognition** — 130 mandates a stated tensor.contract op as the working representation until target selection, while tensorlib deliberately has no contract primitive (matmul = repeat+mul+reduce normal form, COMPUTE.md:159-165; MAC recognized as fusion only in opcount.py) — a genuine decided-position divergence that the charter (140:691-694) mislabels as an "independent convergence" and that must be adjudicated when L4 opens (does the backend select mma from a stated contract, or recognize mul→reduce-sum?); it is not, however, 130's scalar-loop un-lowering failure mode, since reduce structure is preserved as named IR and no L4 backend lowering exists yet to exhibit a concrete miss.
   - Evidence: docs/design/130_tensors-tiles-and-over.md:22-30,49 ('must not require lowering... and un-lowering to rediscover that a loop nest was a matmul'; 'Named contract/reduce/map are the working representation') vs explorations/tensorlib/COMPUTE.md §4 matmul walkthrough, explorations/tensorlib/CONCERNS.md #23 ('MACs are a recognized FUSION... not a primitive'), explorations/tensorlib/LEVELS.md:79-83 (preserve only reduce/scan structure)
   - Verifier: The divergence is real but the "exact failure mode 130 names" framing is overstated: 130 prohibits un-lowering from scalar loops, and tensorlib preserves named reduce structure with LEVELS explicitly licensing pattern-matching over it. Would confirm: an L4 lowering design where mma selection depends on a "solely-consumed" recognition rule and a real case (e.g. mul product saved for backward) where recognition misses a contraction that a stated contract op would have kept. — "130:28-30: "Named contract / reduce / map are the working representation until target selection." opcount.py:12-14: "MACs are not a primitive either — they are a FUSION ... (a pointwise mul consumed solely by a reduce-sum)." LEVELS.md:82-83: "Lowering must PRESERVE reduce/scan structure on machine-b"
   - Proposal: Adjudicate at the joint level: either promote contract to an IR-level op/annotation the fingerprint and L4 partitioner key on (130's stance), or amend 130 to accept reduce-structure + declarative fusion recognition as 'stated enough'; a collision between decided positions in both streams — escalate to human per charter §8.
36. **[MEDIUM / CONFIRMED] backend-enumerates-transforms** — The C grid family derives its launch-domain parameter contract by isinstance-unwrapping the stdlib Over class inside the backend (c.py:291-298, the backend's sole stdlib import), because the DerivedValue protocol (derived.py:24, registry.py:67) exposes only fntype/fp/captures/kind and no domain/param-arity aspect; consequently every other DerivedValue (Jvp — tested at test_grid.py:107-113 — and Pipeline) gets a blanket "no domain contract" refusal, and the second existing family (wgsl) already re-implements its own parallel pyfunc-sniff (wgsl.py:66-71) that refuses Over-chains entirely, so transform-awareness is per-backend hand enumeration rather than a protocol read; the intended refusal boundary itself is correct and preserved by the proposed fix.
   - Evidence: c.py:291-299 (from ..stdlib.transforms import Over; isinstance walk); wgsl.py:66-71 (parallel pyfunc-sniffing refusal for pipelines); test_grid.py:107-113 (jvp refused); derived.py:21-43 (the protocol that should carry this)
   - Verifier: Trigger is concrete and tested: backends/c.py:291 is the only stdlib import in the backend (all other imports are ..kernel.*, c.py:34-37), so the grid family's `_grid_param_types` reaches over the layering boundary to isinstance-enumerate Over; any other DerivedValue lacks `pyfunc` and hits the blanket refusal (test_grid.py:107-113 pins jvp: "no domain contract"; combinators.py:180 Pipeline has no pyfunc either), and the wgsl family independently re-implements target-shape sniffing (wgsl.py:66-71 getattr(target, "pyfunc")) that refuses Over-chains outright with a pipelines-only message — so over'd kernels compose on the C grid but not on wgsl compute, purely because each backend hand-rolls the derivation. No design doc decides this coupling (010:692/080:31/090:122 only name the param_types column; 130 never assigns the domain contract to a place), and derived.py:1-12 explicitly bills DerivedValue as "the ONE wrapper protocol", which the isinstance walk bypasses. One refinement: the refusals themselves are intended, tested behavior the proposal keeps; the finding is the mechanism (class enumeration in backends), not the boundary. — "c.py:291-298: "from ..stdlib.transforms import Over ... while not hasattr(t, \"pyfunc\"): if isinstance(t, Over): lanes, t = lanes + 1, t.captures[0]; continue; raise VerifyError(f\"the grid launches kernels and over-chains; {type(t).__name__} has no domain contract\")" — while derived.py:24 gives t"
   - Proposal: Move the domain/param-derivation contract onto the DerivedValue/Handle protocol (e.g. a 'domain_params' aspect or FnType-visible arity that transforms compose), so backends read a protocol instead of importing satellite classes; refusal for transforms without the aspect stays.
37. **[MEDIUM / CONFIRMED] D-vs-grid-coordinate-types** — The in-kernel D operator refuses non-float params (transforms.py:326-327, proven by test_refusal_contract.py test_D_refuses_integer_params) while the two compute families derive contradictory coordinate types — C grid: (i64,)*n at c.py:290-299; wgsl: (f64,)*argc at wgsl.py:63-72 — and these derived types become the kernel's param types (registry.py:167-175), so the screen-space-derivative idiom type-checks under wgsl compute but refuses under the grid family for the same source; D also offers no wrt selection (loops over all params, transforms.py:320, always returning a tuple, :335). This violates the maximal-common-syntax rule (charter 140:109-110); neither 130 §4.3 nor any other doc adjudicates the coordinate-type divergence, and charter §7.5 (140:609-621) already pre-registers D(foo, wrt(x)) as the candidate redesign.
   - Evidence: c.py:290-299 (_grid_param_types returns (i64,)*n); wgsl.py:63-72 (_param_types returns (f64,)*argc); transforms.py:315,325-327 (D refuses int params); test_refusal_contract.py:55; charter §2.1 equal citizens:110-115, §7.5 candidate D(foo, wrt(x)):609-621
   - Verifier: CONFIRMED trigger: the same kernel body containing D(expr-of-coordinates) type-checks under the wgsl compute family (params derived as f64, wgsl.py:63-72, via registry.py:167-175) but raises MissingRule "D differentiates w.r.t. float params" under the grid family (params derived as i64, c.py:290-299); the refusal path is proven by tests/test_refusal_contract.py test_D_refuses_integer_params, and no design doc (130 §4.3 included) adjudicates the coordinate-type divergence. Refinement: cite the maximal-common-syntax rule (140:109-110) rather than equal-citizens (140:106-108), since the C grid family is the CPU backend, not one of the three GPU citizens; also note D-through-wgsl is untested end-to-end (works at the type-contract level, no launcher test exists). — "c.py:299: `return (i64,) * (t.pyfunc.__code__.co_argcount + lanes)` — wgsl.py:72: `return (f64,) * pyfunc.__code__.co_argcount` — transforms.py:327: `raise MissingRule(f"D differentiates w.r.t. float params; param {k} is {t!r} [{fmt(ctx.loc(node))}]")` — transforms.py:335: `return ctx.emit("core.tup"
   - Proposal: Unify the coordinate-type contract across compute families (decide i64-with-explicit-cast or f64 once, for all backends), and adopt the §7.5 wrt-selection redesign of D rather than patching the seed to tolerate ints.

## 7. Breadth check

# Breadth check — charter §10 deliverable 7

## Headline

The verdicts survive breadth better than the schedule does. Re-examined against the full roster (graphics/procedural, audio/music, discrete simulation, convex optimization, geometric algebra, PDE physics, ML), **no hierarchy or architecture verdict is rejected**, and the three doors the synthesis prioritizes (scan, operators, record construction) are all breadth-driven, not ML-driven. Six flags stand: (1) the founding domain — graphics — has **zero scheduled installment** in 020 Phase IV while the tile/tensor track is CUDA/Metal-first, the exact drift "breadth is identity" exists to catch; (2) the certified-rewrite vocabulary is **contraction-shaped** — its fusion rule ("elision of identity materializations") cannot certify the scheduled stencil flagship, so the architecture verdict's evidence base is ML-only where its rule set is thinnest; (3) A1's precision-demotion license and audio's f32-adoption gap are **one concept stated twice** — generalize to a single carrier-relative precision story; (4) n-ary pipe widening is ML-evidenced only — keep it in the already-designated differentiable-programming satellite; (5) the canonized KV-decode spelling is the ML instance of a generic ring/window-adoption pattern whose breadth twin is the audio delay line; (6) the dispatch-sequencing ruling (`host.fold`) covers only the owned-loop half of a roster where three loop domains are foreign. **The someone-else's-loop unification was verified named exactly once** (Probe E §0, per charter 140:539-544), with §7.3/§7.7 referencing rather than renaming it. Concepts affirmed *at* breadth, against suspicion: named axes (convex ISTA's zero-transpose Gram steps), `over` (audio polyphony, test_grid.py:52-64), the grid family (all four sketches + PDE + ML), the scan door (audio workhorse first, attention second), records (GA + graphics jointly), the jvp table's kink commitment (convex-driven), collectives-read-off-algebra (audio linrec, mdsl.py:400-413), and — from canon's own hand — reverse-mode's flagship chapter is a *shader* design-optimization loop, not a training loop (020:325).

## 0. Someone-else's-loop: named once — VERIFIED

The charter requires the unification be named once (140:539-544: "render loops, audio callbacks, physics ticks, training steps... one concept, not four; the assessment should name it once"). Audit of the four deliverable documents: **defined once** at Probe E §0 ("The someone-else's-loop constraint (named once)"), with the contract's four verified components (fresh-closure-per-iteration, registry.py:134-137 re-read this session; mutated numpy captures legal; `out=` adoption, c.py:315-336; grid refuses the argument channel, registry.py:167-169 re-read — making the idiom load-bearing). The hierarchy verdict introduces no competing name (its matrix note 2 "a pass we do not own" is an instance, not a re-derivation); the architecture verdict is silent; the synthesis references the definition site explicitly (§7.7 "canonize the fresh-closure idiom once for all foreign-loop domains (Probe E §0)"). One caution recorded, not a finding: the *constraint* (someone-else's-loop) and its *answer* (fresh-closure idiom) are named in two places with one canonization obligation each (Probe E §0 close; synthesis §7.7) — the canon doc must carry both under one roof so they do not drift into two docs. See flag 6 for the fold/foreign split that same doc must state.

## 1. Verdict-by-verdict roster re-examination

Legend per verdict: **B+** breadth-affirmed (non-ML evidence exists), **B0** breadth-neutral (domain-generic by construction), **FLAG n** ML+PDE-only, demotion/generalization proposed.

| Verdict | Roster status | Evidence |
|---|---|---|
| §2.1 L0 device functions + struct amendment | **B+** — the struct prerequisite is jointly demanded by GA (multivector records, Probe E §3) and graphics (color/varyings, Probe A via hierarchy Amendment 1); zero ML demand needed to justify priority | Probe E §3; hierarchy §2.1 L0 A1 |
| §2.1 L1 equal-citizens breach | **FLAG 1** — sharpened: not just parity divergence but scheduling absence of the founding domain | 020:317-327, 347-354 (read: no vertex/fragment/PSO/render step anywhere); 020:326 (step 16 = "CUDA + Metal backends" carrying the tile family + GEMM chapter) |
| §2.1 L2 dissolution + two missing operators | reverse-mode operator **B0** (probes evidence it via ML, but canon's own grad flagship is a shader: "grad of a smooth SDF shader; a design-optimization mini-loop," 020:325; convex objectives and inverse-PDE are further native clients); n-ary composition **FLAG 4** | 020:325; Probe E §6 |
| §2.2 one-IR ladder + fork escalation | **B0** — ladder and machine tree are machine-generic (LEVELS.md:11-30 re-read); watched, not flagged: only ML+PDE probes reach L3+, so upper-rung designs should cite a graphics-compute or large-sim client when L4 reopens | LEVELS.md:11-30, 45-61 |
| §2.2 collectives-not-ops | **B+** — audio's IIR parallelization via the associative linrec combine is the non-ML confirmation | mdsl.py:400-413 (read: the docstring's linrec IS the one-pole filter); Probe E §7 |
| §2.3 synthesis: three doors, no new language | **B+** — scan: audio workhorse + attention + convex running reductions; operators: GA; records: GA + graphics. All three doors have their *primary* client outside ML | Probe E §7; hierarchy §2.3 |
| §2.4a/b @jit dual role; kind two-concept split | **B0** — the validated kind vocabulary must enumerate graphics kinds (vertex/fragment) on equal terms; covered by FLAG 1's scheduling proposal | registry.py:121-124 (read: unknown kind silently routes to default) |
| §2.4c `over` reabsorbed | **B+** — polyphony (`axis="voice"`) and batch are the same one-more-coordinate move; reabsorption verdict unaffected | test_grid.py:52-64 (read) |
| §2.4d `matmul` conviction (macro erasing contraction) | **B+** — conviction is domain-neutral; the *front* it keeps is breadth-vindicated by ISTA's two zero-transpose contractions | transforms.py:338-396 (read: `core.for` mul/add expansion); Probe E §2, §5 |
| §2.4d `Pipeline` demotion | **B+ for the demotion** — pipeline fit zero of four breadth sketches; ML-leaning per Probe E §7 | combinators.py:262-272 (read: unary constraints); Probe E §6 |
| §2.4e jvp table + kink commitment | **B+** — the subgradient-commitment obligation is convex-optimization-driven, the clearest case of a non-ML domain shaping kernel semantics | Probe E §2; synthesis §7.5 |
| §2.4f step-14 sample + workhorse status | **B+ with a note** — the pathology is cross-domain (attention loops, test_grid.py:120-123 read; audio IIR O(N²), Probe E §1); note: when the flagship is rewritten (V5), the gate battery should gain a non-ML twin (scan-form IIR or stencil) so the founding gate stops being attention-only | test_grid.py:116-128, 145 (read) |
| §3 architecture (A1-A8, both amendments) | sustained, but **FLAG 2** (rule vocabulary contraction-shaped; Probe C evidence GEMM+attention only) and **FLAG 3** (precision license stated in tensor-core terms; audio/graphics are the native precision clients) | LEVELS.md:109-114, 183-184, 207-210 (read); memory.py:21 (read); c.py:324-327 (read); tensor.py:48 (read) |
| §7.1 state/mutation incl. KV rejection | Amendments 1-2, 4 **B0**; Amendment 3 **FLAG 5** (canonize the pattern, not the ML instance) | test_array_args.py:30-33 (read: rank-generic prefix hit); c.py:325-330 (read: adopt path) |
| §7.2 residency classes | **B+** — audio f32 gap and Probe A adoption drove it; folds into FLAG 3 | c.py:324-327 |
| §7.3 pipelines: two concepts + fold mirror | decision **B+** (breadth unanimity: fuse ≠ sequence); the fold spelling **FLAG 6** (owned-loop-shaped) | Probe E §6; synthesis §7.3 |
| §7.4 named axes + launch conviction | **B+** — convex is the strongest names-first evidence in the assessment; GA counter-case keeps names honest; no dual-universe or genericity pressure in any sketch | Probe E §5; test_array_args.py:76-77 (read: same names, bigger operands, warm) |
| §7.5 derivative family map | **B+** — forward/ambient: graphics (fwidth) + convex (subgradients); jvp-over-arrays deferral **affirmed as boundary** (contributed nothing at breadth, Probe E §7); scalar-only checks re-read at transforms.py:294-296, 326-327 | transforms.py:290-335 (read) |
| §7.6 kernel stays tensor-free; scan door first | **B+** — the priority ordering is breadth's doing: audio (workhorse), then attention, then convex | Probe E §1, §7 |
| §7.7 caching restatements + skeleton | **B0** — fresh-closure canonization is the cross-domain item; skeleton amendments domain-neutral | registry.py:132-137 (read) |
| §7.8 refusal three-tier + one voice | **B0** — breadth sketches' refusals (alias, Named-out, scatter boundary, f32-adopt) already read in one voice when shaped to the contract | synthesis §7.8 |

**Boundary verdicts re-checked at breadth (all stand):** scatter-shaped discrete work outside the subset with the sketched refusal (Probe E §4 — a §1 success); vector-jvp deferral (correct posture, no breadth demand); f32-adopt refusal correct-but-gap (FLAG 3); no breadth domain was forced into tensor-shaped clothes — GA's escape into records rather than a fake length-4 axis is the system refusing the wrong clothes, as designed.

## 2. Flags

**FLAG 1 (high) — the founding domain has no scheduled installment; the tile track is CUDA/Metal-first.** 020 Phase IV's steps 11-17 (020:317-327, read this session) build arrays/C, vmap+jvp, seams+over, tensors-on-CPU, grad, CUDA+Metal(+tile+GEMM), units; the "Later, unscheduled" list (020:347-354) adds t-strings, disk cache, solver satellite, live canvas, tutorials, and the differentiable-programming satellite. No line schedules vertex kind, PSO pairing, MRT, or the encode-into-a-foreign-pass seam — while step 16 (020:326) lands the tile family on CUDA+Metal with the GEMM chapter. Under "breadth is identity" (140:59-68) the drift is happening by *scheduling*, not by concept. Proposal: the direction memo must slot a graphics installment (vertex/fragment kinds in the validated-kind vocabulary, PSO pairing as the third composition semantics, encodable-into-foreign-pass deliverable) before or alongside the tile step, and declare every parity divergence a capability-gated refusal.

**FLAG 2 (high) — the certified-rewrite vocabulary is contraction-shaped and cannot certify the scheduled stencil flagship.** The rule list is split∘merge=id, guard/slice commutation, reassociation, "fusion = elision of identity materializations" (LEVELS.md:109-114, read). A fused stencil chain — a scheduled L4 flagship (LEVELS.md:183-184; K-D at 207-210: "a fused stencil chain (heat/FDTD step)") and the shared workhorse of PDE physics and discrete simulation — consumes intermediates at *offsets*, so stage fusion requires overlapped-tiling/halo-recompute rewrites in no rule list; architecture A1-A8 never touch the class, and Probe C's ≤8-rewrite evidence covers GEMM and attention only. Proposal: generalization — add the overlapped-split/halo-recompute class to the certified-rule roadmap and make the stencil flagship the L4 brief's non-contraction acceptance test; three roster domains (PDE, discrete sim, graphics convolution) are its clients.

**FLAG 3 (medium) — precision-demotion is one framework concept stated twice: A1's descent license and the kernel-level dtype gap.** A1 derives the precision license from f16 tensor-core GEMM (ML clothing); the roster's native precision clients are audio (grid family hardcodes result dtype and refuses f32 out, c.py:324-327 read — forcing alloc+copy in a real-time callback, Probe E E6) and graphics f16 shading. Proposal: one carrier-relative precision concept (tensor.py:48, read) with two surfaces — a family element-dtype parameter at kernel level (fixes E6, no copy), the license class at descent level — so precision never reads as a tensor-core/ML feature.

**FLAG 4 (medium) — n-ary/pytree pipe widening is ML-evidenced only; keep it satellite.** The ruling "widen `|` to n-ary when landing Probe D's optimizer" rests solely on Adam/clipping; at breadth, pipeline fit was zero of four sketches (Probe E §6) and Probe E §7 lists Pipeline-as-fusion among concepts "consistent with keeping them satellite, not kernel." Canon already reserves the slot: the differentiable-programming satellite with "optax-shaped gradient chains" (020:351-354, read). Proposal: demotion — unary `|` stays kernel (combinators.py:262-272); n-ary/pytree composition lands in that satellite; promotion to kernel requires a second, non-ML demand (candidate to re-probe once scan lands: audio mix graphs, which are n-ary by nature).

**FLAG 5 (medium) — canonize the ring/window-adoption *pattern*, not the KV-decode instance.** Synthesis §7.1 Amendment 3 canonizes "the Probe D spelling" (preallocated buffer + `out=` row adoption + prefix-slice recapture; rank-generic warm hits pinned at test_array_args.py:30-33, read) as the supported boundary sample — named as KV cache, an ML sample. The identical mechanism is audio's delay line/ring buffer (the workhorse Probe E's IIR spelling could not reach) and discrete sim's history windows. Proposal: generalization — the canon sample presents the pattern once with two worked instances, KV decode (ML) and a delay line (audio), so the boundary sample does not enter canon as an ML feature.

**FLAG 6 (low) — the dispatch-sequencing ruling is owned-loop-shaped; the foreign half must be co-equal.** §7.3 lands sequencing as `host.fold(step, init, ...)` mirroring tensorlib fold — right for loops we own (convex's 200 iterations, sim ticks) but inexpressible for the roster's three foreign loops (render loop, audio callback, engine tick — Probe E §0; 140:539-544), where there is no loop to fold and the answer is encode-bundles + the fresh-closure idiom. Proposal: generalization — the canon doc naming the someone-else's-loop constraint states fold (owned) and encode-bundle+fresh-closure (foreign) as the two halves of one sequencing story, preventing fold from becoming a default that quietly assumes we own the loop — the training-loop assumption in disguise.

## 3. Keep-list (concepts cleared of ML suspicion this check)

Named axes and matmul-by-name (convex), `over`-as-one-more-coordinate (audio/ML twin), the grid family and capture/fresh-closure (all seven domains), scan/defreducer (audio-first), records+operators (GA+graphics), jvp-with-kink-commitment (convex), collectives-read-off-algebra (audio linrec), reverse-mode-at-assemblage (canon's own flagship is a shader, 020:325), refusal-first contract (domain-neutral). Watched, unflagged: ladder rungs L3+ (machine-generic by construction, but every probe that reaches them is ML+PDE — the reopened L4 brief should carry one non-ML client through).

## 8. Direction memo

# Direction memo — charter §10 deliverable 8

**Citation keys.** HV = hierarchy verdict (§2 sections as written there); AV = architecture verdict (attack lines A1–A8, amendment rulings); CS = cross-cutting synthesis (§7.1–§7.8); BC = breadth check (flags 1–6, §0–§3); F# = verified finding number from the ranked findings list (1–37; duplicates noted where merged: F6≈F23, F5≈F13≈F18, F4≈F25, F2≈F27). Plan of record under revision: 020 Phase IV (020:317-327, 347-354). L4 questions: LEVELS.md:189-217 (K-A..K-F).

---

## (a) Ordering and content of the next installments — what changed vs 020, and why

020's standing order is: step 14 remaining (tensor dialect ops, comprehensions, DPS, over-emits-map-IR) → 15 grad → 16 CUDA+Metal+tile+GEMM → 17 units, with graphics nowhere (F2/F27, BC flag 1). The assessment changes the order in five ways and the content of nearly every step. Revised sequence:

### Installment 0 — the pre-calcification batch (new; immediate; mostly small diffs + canon paragraphs)

Not in 020 at all. Everything here is either a silent-wrongness fix (charter §1: wrong outranks missing) or a status ruling that must enter canon before the next feature lands on top of it.

1. **Out-aliasing overlap refusal** at the grid launcher (F6/F23, CS §7.1 A2): `np.shares_memory` over buffer leaves, refuse with the ping-pong message. The probe suite's gravest verified flaw; verified twice by execution.
2. **Named-as-`out=` designed refusal** (CS §7.4) — currently a raw TypeError at the exact seam where transposition is silent corruption.
3. **Unknown-kind dispatch refuses** instead of silently routing to the default backend (HV §2.4b, F33; registry.py:122). Kind becomes a validated vocabulary — the two-concept split (Handle kind = authoring/ABI contract; registry routes = execution family) enters canon.
4. **Record-value crashes become designed refusals** (F14): `_arith` requires Scalar/Vec operands with a message naming the fixes; whole-record return gets the arrays.py:213-style refusal. Blocks the first path every record-curious user hits.
5. **Workhorse status ruling into canon** (F32, F34, HV §2.4f): per-lane host dispatch = debug/oracle-grade, the differential-testing baseline; per-lane dispatch of lane-bearing kernels outside test infra is the convicted pattern (warn now, refuse when the grid family generalizes). One paragraph in 090 §4 or the 010 ledger; the ch14 baseline is renamed accordingly.
6. **Doc repairs before they calcify**: 130 §5 GEMM sketch rewritten — derived axis names not literals (F19), blocked-k as a reduce-kind comprehension not a carried seq loop (F20), token discipline made real or declared implicit-with-checker (F9); stale placement.py gradients docstring (HV §2.2); fragment-family/ch10 "colors arrive with tuples, step 10" promise (F24); KV-cache exclusion text in zoo/LEVELS reworded to representation-level-only (F16).
7. **JVP kink pinning** (F28, CS §7.5): pin abs(0)/min-max-tie/branch-boundary values in test_jvp_rules as a commitment; cross-stream convention goes to arbitration (part f).
8. **matmul extent-mismatch**: stage-time refusal when both operands carry static extents, replacing the frozen 3.0 UB pin — pending arbitration item f.2 (F29).

Why first: every item is cheap, none blocks on the fork adjudication, and each closes a hole that the following installments would otherwise inherit and teach.

### Installment 1 — the emission seam (new; the single highest-priority artifact)

HV §2.3 Amendment 1: the synthesis presupposes surface→laddered-IR emission and zero code path exists. Contingent on arbitration f.1 resolving as recommended (tensor dialect = frontend emitting tensorlib's Program), this installment is the concrete plan in part (d). It precedes the tensor-dialect installment because every subsequent verdict (one AD, chunk fingerprints, scan, fold sequencing) lands on this seam.

### Installment 2 — tensor dialect, re-scoped (replaces "step 14 remaining")

Changed vs 020: the dialect is a **frontend, not a second IR** (F5/F13/F18; HV §2.2 ruling). Content:

- **Comprehension/reduce spelling** (130 §4.2) landing first — it is the enforcement prerequisite for no-extent-loops (HV §2.1 L0 A2, F11).
- **Scan/fold door promoted into this installment** (F15; CS §7.6: "the single highest-value primitive"). Changed vs 020, where scan appears nowhere. Justified by audio (the workhorse IIR is O(N²)-only today), then attention streaming softmax, then convex running reductions — breadth-driven, not ML-driven (BC §1).
- **matmul repaired**: keep the name-pairing front (breadth-vindicated by ISTA, BC), emit a preserved contraction whose `core.for` expansion becomes the *default decomposition*, never the birth form (HV §2.4d; F20/F21/F35 adjacent).
- **`over` emits map/bind structure** (F12): the already-planned step, now explicitly the reabsorption — `over` and tensorlib `bind`/`Dim.level` become one binding concept at different tree depths; the backend's isinstance-unwrap of `Over` dies, replaced by a DerivedValue domain aspect (F36).
- **Flagship + gate rewrite** (F11): attention in named reduce/comprehension form, gate re-scoped to it, ch14 prose fixed, plus a non-ML gate twin (scan-form IIR or stencil, BC §1 note). Then — and only then — stage the extent-loop refusal at L0 with a debug-grade escape (HV §2.1 L0 A2: refusing before the replacement exists strands the only working spelling).
- DPS array results as planned (unchanged).
- **Ring/window-adoption canon sample** (F16, BC flag 5): the pattern presented once with two instances — KV decode and an audio delay line — pinned by a compile-once-under-`no_compile` test and the aliasing refusal from installment 0.

### Installment 3 — grad + sequencing, re-scoped (replaces step 15)

020's resequencing (grad after tensors) is **affirmed** — AV Amendment 2 strengthens it: combine-introducing rewrites must precede `grad`, which requires named forms to exist first. Changed content:

- **No second AD.** The reverse-mode surface operator lowers to tensorlib `grad` + `dce` (keep-set = wrt/freezing; two-call idiom per Probe D), contingent on arbitration f.1. 020's "partial-eval + transpose ~450 lines" is replaced by the seam + a thin operator.
- **Flagship unchanged and load-bearing for breadth**: the SDF-shader design-optimization loop (020:325) — canon's own proof that reverse mode is not ML-owned (BC keep-list).
- **`host.fold` dispatch sequencing** lands here (F22, CS §7.3): the pdum.dsl mirror of tensorlib fold with an explicit carry contract — the training step, the PDE time loop, and convex iteration all need it, and it is the seam L2 bufferization later consumes (K-F). The (op, roles)→semantics-tag registry gains the third tag; `|` stays fuse-only. n-ary/pytree composition goes to the differentiable-programming satellite, already reserved in 020's backlog (BC flag 4) — promotion to kernel requires a second non-ML demand (candidate: audio mix graphs, re-probe after scan).
- Kink-convention adjudication (f.5) must be closed before grad ships, so forward and reverse agree.

### Installment 4 — graphics (new; before or alongside installment 5)

BC flag 1 / F2/F27: the founding domain has zero scheduled installment while the tile track is CUDA/Metal-first — drift by scheduling, which "breadth is identity" exists to catch. Content: vertex + fragment kinds in the validated kind vocabulary (equal terms with compute); composite fragment results (vec4/record through the existing ResultPlan rebuild + record construction as surface-C's missing half, F24); PSO pairing as the third composition semantics (never punned on `|`, HV matrix note 8); the **encodable deliverable** — `pso.record(...)` render bundle / `draw_into(pass_encoder)`, replacing 070 §4's convicted `draw(target)`-owns-the-pass design (F26); ambient-axis declaration per family enabling `D(x, wrt=axis)` and fixing fwidth (F4/F25/F37, part e). Every parity divergence declared as a capability-gated refusal, never discovered (HV §2.1 L1 A2).

### Installment 5 — CUDA + Metal + tile (step 16, with preconditions)

Survives, but gains the L4-brief preconditions (part b): chain-as-stored-artifact (F7), widened registry key (F8), license taxonomy with the precision class (F10), WF certificates (F9), dtype-exact memory sim (AV A3), the repaired 130 §5 sketch (installment 0.6). WebGPU is present in the tile plan via declared capability-gated refusals (subgroup limits), not absence. Tile surface rides the one frontend machine into the one representation (HV §2.1 L1 A3, §2.3).

### Step 17 units — unchanged.

---

## (b) The L4 design brief — K-A..K-F answered/framed from assessment evidence

This memo is the brief that reopens the paused tensorlib stream (charter §8 sequencing). Per question (LEVELS.md:189-217):

**K-A — What is a kernel in the IR?** ANSWERED: **annotation-in-IR + region-in-the-lowering-registry** (Probe C resolution, adopted by AV). The kernel boundary is an erasure-preserving grouping annotation (the lean in LEVELS is affirmed — kernels change cost, not meaning; the erasure discipline has now paid off a third time via placed-backward). The *authored descent* — the tile-level body — is not IR: it lives as the value of a certified-lowerings registry entry (chain + authored region + artifact + assurance tier, F8), keyed by the chunk fingerprint (F21). This keeps one IR while giving the descent a home.

**K-B — Does anything NEW appear at L4?** ANSWERED: split+bind is unchanged one tier down (HV §2.2 "same move" AFFIRMED; Probe C). Three genuinely new things, all predicates/decisions over existing machinery, none representation: (1) the **capacity WF predicate** (Σ staged bytes ≤ level capacity) — which MUST run dtype-exact, never on the 8-byte shadow convention (AV A3: an f16 plan over-counted 4× refuses valid tilings; dtype-exact sizes are promoted from "later" to an L4 precondition); (2) **ordering/tokens** — race-freedom as a WF check on the elaborated form (AV A2, F9), with the UX correction that tokens may be implicit at the surface provided the checker owns the discipline; (3) **materialization-boundary placement**.

**K-C — Objective and legality.** AMENDED: "legality ≈ convex instruction sets" covers only one third of legality. A certified descent = (equivalence chain) + (per-level WF certificate: race-freedom, capacity, convexity), with WF predicates checked on the *result*, not derived from the chain (AV A2, F9). The objective (minimize parent-memory traffic under child capacity, red-blue pebbling) is affirmed, with two amendments: the cost oracle is dtype-exact (AV A3), and the pipeline is a **descend-and-revisit loop with declared invalidation edges** — fusion invalidates checkpoint plans and traffic plans; placement invalidates partition candidates; one-pass is the special case when no edge fires (AV A6, generalizing Amendment 2). Manual-directives-first stands; the naive→flash move becomes a *registered named rewrite* whose license is the declared combine, so the fusion decision stops leaking into L0 authorship (AV §2 residual).

**K-D — Flagships.** Flash attention and tiled GEMM are affirmed with strong evidence: both decompose into ≤8 named rewrites with exactly one non-layout rewrite each (Probe C; AV Amendment 1 sustained). The **fused stencil chain is the critical flagship** and currently uncertifiable: overlapped-split/halo-recompute is a missing rule class — outside split∘merge=id (disjoint by construction) and outside fusion-as-elision (halo recompute increases ops) (F1, BC flag 2). Brief directive: add the overlapped-split/halo-recompute class to the certified-rule roadmap and make the stencil chain the **non-contraction acceptance test** of L4 — its clients are PDE, discrete sim, and graphics convolution, which also discharges the BC watch-item that every probe reaching L3+ was ML+PDE. Flagship gates must pin adversarial input families (−inf masks, cancellation, non-divisible tails), not only random draws (AV A8).

**K-E — Cost plumbing.** FRAMED, not answered: per-kernel footprint = peak_memory with local=level, per-kernel traffic = materialization-boundary bytes — composition into the L5 timeline stays deferred. What the assessment adds now: (1) **assurance tier and input-domain coverage are recorded fields of every registry entry**, and §3.4's trust claim attaches only at tier ≥2 (AV A8); (2) the license taxonomy the costs are measured under = {none, reassociation, **precision-demotion**}, with equivalence stated over the **carrier (ℝ) denotation** and tolerances part of the license declaration (F10, AV A1) — without this no real tensor-core descent is certifiable at all. BC flag 3: precision is ONE carrier-relative concept with two surfaces — a family element-dtype parameter at kernel level (fixes the audio f32-adopt gap, no copy in the callback) and the license class at descent level — so it never reads as an ML/tensor-core feature.

**K-F — Relationship to L2.** CONFIRMED and extended: kernel boundaries define what materializes; bufferization consumes them; L2 ordering = after fusion decisions. Three additions: (1) the dispatch-sequencing construct (`host.fold`, F22) is the exact seam L2 bufferization later consumes to assign alternating buffers — land it before L2; (2) the ring/window-adoption boundary sample carries an **erasure obligation**: when the seam lands, the same surface program must emit a pure Program whose bufferization reproduces the row-write (CS §7.1 A3); (3) in-place legality is only ever an L2-certified rewrite, never a user gesture — the interim contract is the installment-0 alias refusal (F6/F23).

**Brief items beyond K-A..K-F** (prerequisites the questions assumed): the chain is a mandatory content-addressed *output* of tile-DSL elaboration stored beside the artifact, checker-reconstruction demoted to a migration tool (F7, AV A7); the registry key = (normalized chunk skeleton, boundary contract incl. saved-set demand + layout classes, license set, capability set, rules-generation) with the payoff honestly re-scoped to license-and-demand equivalence classes (F8, AV A4); chunk_fp defined on the named-op Program in canonical() form — a keyspace that cannot exist until matmul stops erasing contraction (F21); a Program-normalization pass is either scheduled or §3.4 is advertised as a private cache (AV A5); the capture→leaf naming law (F17, part d) is part of the emission contract; L4 designs must carry one non-ML client end-to-end (BC watch).

---

## (c) The tensorlib promotion question

**Recommendation: staged promotion, gated on the seam — not now, not wholesale.**

- **Now (stays in explorations/):** nothing moves before arbitration f.1 and the installment-1 seam exist. Promotion without a consumer freezes API surface prematurely; the seam is the consumer.
- **Promote first, by reference not by move:** the *contracts* — the zero-import Node/Program schema (mdsl.py) as the emission target; `Layout` (affine map + box) as the adoption descriptor for foreign buffers (CS §7.2: Probe A independently reinvented exactly it — one concept, not two); `Tensor.carrier` as the semantics/cost split the license taxonomy is stated over (F10).
- **Promotion gate:** when (i) f.1 is decided in tensorlib's favor, (ii) the emission seam has round-trip tests in both directions (part d step 6), and (iii) the stale-doc items are fixed (installment 0.6) — then tensorlib's core (`ir`, `mdsl`, `layout`, `autodiff`, `transforms`, `placement`) leaves `explorations/` into a first-class package. The zoo and reference execution layer stay permanently as the **denotational oracle / differential-testing spec layer** — that is their chartered long-term role (charter §8), not a transitional status; they are never a quarry to strip.
- **If f.1 is decided the other way** (tensorlib demoted to oracle-only, machinery ported), promotion is moot and the port plan replaces this section — but the assessment's evidence weighs heavily against that branch: the ladder is affirmed by construction, and tensorlib's positions won every contested adjudication this run (collectives, growth polarity, carrier, traffic-pass; HV §2.2, CS §7.5, F10).

---

## (d) Frontend→Node-schema integration: concrete first steps

Correction inherited from F18: the charter's own deliverable title ("Node-schema") under-scopes the question — the declared adapter seam is marker-body granularity while Probe D assumes Program-level emission. The plan below is Program-level, contingent on f.1.

1. **Adjudicate the fork** (f.1) — one human session over F5/F13/F18/F35 before any 130 stage-2 code. Recommended resolution: 130's tensor dialect is re-scoped as a frontend whose lowering emits tensorlib Programs (named map/reduce/contract/scan/fold preserved); tensorlib grad+dce is the one AD; 130 stage 3's Grad is cancelled as a separate implementation.
2. **Define the emission contract**: the capture→leaf naming law (F17) — leaf name = the freevar-name chain through nested Handles/Derived wrappers, prefixed by stage/layer index exactly as build_pipe already prefixes env paths; pinned by a rebuild-stability test (a rebuilt closure maps the same capture to the same leaf name). This is the binder-seam move: captured tensors become `input` leaves.
3. **Minimal emitter**: the comprehension/reduce surface (130 §4.2 spelling) lowers a Handle body to a Program; `ir.run` is the tier-0 executor; the pdum.dsl cache fronts it (types-only keys, identity guards unchanged — CS §7.7 verified this machinery handles live knobs at zero recompiles).
4. **Re-front matmul**: keep the unique-shared-axis name-pairing front; emit a preserved contraction (contract instr or contract-marked reduce, per f.6); today's core.for expansion becomes the registered default decomposition.
5. **Identity**: add the chunk_fp keyspace (hash of the named-op Program in canonical() layout form, F21); apply the CS §7.7 skeleton amendments — the structural skeleton replaces NamedArray's order-carrying dims tuple as identity (landing together with stride-structure when adoption widens beyond C-contiguous), and the specialization/artifact tiers stay distinguished.
6. **Test battery, both directions**: erasure round-trips (named/positional twins → one artifact); grad-through-seam equals tensorlib grad on zoo samples; decode-loop compile-once under `no_compile` + the aliasing refusal (F16); fresh-closure idiom and mutated-capture re-extract pinned (CS §7.7 — currently unpinned load-bearing behavior); cross-Handle record-sharing guard re-key or weakref (CS §7.7).
7. **Executor**: `host.fold` mirroring tensorlib fold's carry contract — the first assemblage-level runtime, discharging HV §2.1 L2's "one-invocation-many-dispatches has no executor."

---

## (e) Remove-or-rename list (before it calcifies)

| Item | Fate | Authority |
|---|---|---|
| `over` as standalone transform | Reabsorbed: surface verb for split+bind emitting map/bind structure; one binding concept with tensorlib `bind`/`Dim.level`; 16× gate, weave-as-lowering, trailing-lane contract all survive | HV §2.4c, F12 |
| Backend isinstance-unwrap of `Over` | Removed: domain contract becomes a DerivedValue protocol aspect; backends read structure, never satellite classes | F36, F12 |
| `D` (all-params, no wrt) | Replaced: families declare named ambient axes; `D(x, wrt=axis_name)`; coordinate dtypes unified across families; fwidth's hardcoded d[0]/d[1] falls out; wrt-as-structure stays deferred | F4/F25/F37, CS §7.5 |
| `matmul` lowering-time macro | The contraction-erasing expansion removed (becomes default decomposition of preserved structure); name-pairing front kept; extent-mismatch UB → refusal (pending f.2) | HV §2.4d, F21, F29 |
| `Pipeline` as "the load-bearing idiom" | Demoted to value-fusion along one thread; `|` keeps fuse semantics only; dispatch sequencing = `host.fold`; PSO pairing = third semantics tag; n-ary → differentiable-programming satellite | F3, F22, CS §7.3, BC flag 4 |
| "the transform column is open" (transforms.py:143 refusal + 110 §3 framing) | Narrowed to the growth law: primitive rule set frozen; new differentiable ops are composites/markers with derived partials — never new rows; refusal points to the derived path (pending f.4) | F31, HV §2.4e, CS §7.5 |
| `kind` as free string + silent default routing | Validated vocabulary (incl. vertex/fragment on equal terms); unknown kind refuses; two-concept split (Handle=ABI contract, registry=execution family) | F33, HV §2.4b |
| Step-14 flagship loops + ch14 prose | Rewritten in named reduce/comprehension form when the spelling lands; gate re-scoped; non-ML gate twin added; then staged extent-loop refusal with debug escape | F11, BC §1 |
| `draw(target)` owns-the-pass (070 §4 committed design) | Replaced by encodable: render bundle / `draw_into(pass_encoder)`; submit/pass/swap-chain belong to the host; promote the existing encode-vs-submit seam to API | F26 |
| test_refusal_contract matmul==3.0 pin | Removed pending f.2 — a frozen silent wrong answer inside the refusal battery itself | F29, CS §7.8 |
| Per-lane host dispatch (unnamed status) | Named: debug/oracle-grade in canon and in the ch14 baseline prose | F32/F34, HV §2.4f |
| KV-cache exclusion wording (zoo/LEVELS) | Reworded representation-level-only; canon sample = the ring/window-adoption *pattern* with KV-decode + audio delay-line instances | F16, CS §7.1 A3, BC flag 5 |
| 130 §5 GEMM sketch | Rewritten: derived axis names (F19), reduce-kind blocked-k comprehension (F20), real or checker-owned token discipline (F9) — before stage 4 implements it as written | F19/F20/F9 |
| Stale docstrings | placement.py "gradients do not yet carry bindings"; wgsl.py/ch10 "colors arrive with tuples, step 10" | HV §2.2, F24 |

---

## (f) Human arbitration required (charter §8 governance — collisions with decided positions)

Ordered by urgency; none may be resolved silently.

1. **The IR fork** (F5/F13/F18; HV §2.2; CS §7.6). 130 stages 2–3 (decided-PROPOSED: pdum.dsl-owned tensor dialect + second AD) vs charter §5's front-tensorlib mandate. Blocks installments 1–3. Memo recommendation: frontend-emits-Program, one AD. Must be decided before any stage-2 code.
2. **matmul extent-mismatch UB** (F29). The 130 §4.1/100 UB posture is decided and frozen as spec (test asserts 3.0) — collides with charter §1 refusal-first. Recommendation: stage-time refusal where static extents exist; explicit boundary statement otherwise.
3. **Pipeline: one operator vs three semantics** (F3). 040 §2.3/§2.6 decided ONE `|` with role-driven fuse-vs-orchestrate on the semantics-tag registry; the assessment rules three visible semantics, never implicit choice. Recommendation: keep the one gating registry (040's mechanism wins) but require the semantics be visible at the composition site (040's single-spelling stance loses).
4. **Rule-table growth polarity** (F31). 110 §3's open jvp column vs tensorlib PHILOSOPHY's frozen tables. Recommendation: adopt the growth law (freeze primitives, derive composites), acknowledging tensorlib still permits genuine new leaves.
5. **At-kink subgradient convention** (F28). pdum's pick-a-side select vs tensorlib's full-cotangent-to-every-tie — not transposes of each other; both now pinned or to-be-pinned. One convention must be chosen for the joint system before grad ships.
6. **contract as primitive vs recognized fusion** (F35). 130's stated-contract-until-target-selection vs tensorlib's deliberate no-contract-primitive (MAC as recognized fusion) — mislabeled a convergence by the charter. Decides what the L4 backend selects mma from; rides f.1.
7. **Precision/carrier restatement of §3.2** (F10, AV A1). Amending the charter's own two-license taxonomy to carrier-denotation equivalence + precision-demotion license; tensorlib COMPUTE §2b already decided the carrier split, so likely a ratification — but it amends a decided charter position.
8. **Graphics installment insertion** (F2/F27, BC flag 1). Changes 020's decided Phase IV sequencing; if graphics is deliberately post-M4 instead, that decision and its reason must be recorded in 020 — silent deferral of the founding domain is a charter violation.
9. **KV exclusion rejection** (F16, CS §7.1 A3). The synthesis rejects a recorded tensorlib boundary (zoo exclusions) as a joint-system exclusion — a decided-position collision even though both streams' texts call it a deferral.
10. **Workhorse status + default-family polarity** (F32/F34). The canon addition itself (debug/oracle-grade; whether a domain-shaped family becomes the default for array-capturing kernels) is a decided-posture change to 010/090's routing story.

---

**One-line summary.** Fix the silent-wrongness seams and stale flagship docs now (installment 0); adjudicate the IR fork and build the emission seam before any tensor-dialect code (installments 1–2, arbitration f.1); grad stays after tensors but becomes a thin operator over tensorlib's AD plus `host.fold` sequencing (installment 3); insert the graphics installment before the CUDA/Metal/tile step to arrest the scheduling drift (installment 4); the tile step proceeds only with the L4 brief's chain/license/WF/dtype preconditions and the repaired 130 §5 (installment 5); tensorlib promotes in stages gated on the seam, with its reference layer permanently the oracle.

## 9. Cross-cutting synthesis (full)

# Cross-cutting synthesis — charter §7, all eight questions

## Headline

Rulings: §7.1 AMENDED — the storage-level mutation stance survives as representation doctrine but is convicted at the joint surface (silent out-aliasing corruption, verified) and the KV-cache exclusion is REJECTED (Probe D runs decode today). §7.3 DECIDED — value-fusion and dispatch-sequencing are two concepts (a third, pair-into-PSO, exists); one gating registry, never one operator with implicit choice. §7.4 AFFIRMED with a launch-boundary conviction. §7.5 the ambient/tape rule is ratified and sharpened — the tape is never a runtime object; subgradient-at-kinks becomes a COMMITMENT; the rule table resolves as convergence under a stated growth law. §7.6 kernel stays tensor-free; the 130-vs-tensorlib IR fork escalates. §7.7 fingerprint sketch ENDORSED with two amendments. §7.8 not one voice — three tiers.

---

## 7.1 State and mutation — ruling: AMENDED

**The incumbent stance (mutation is storage-level; syntax stays pure; L2 commits late) is SUSTAINED as representation doctrine.** Tensorlib's `fold` keeps both zoo PDEs pure including the two-state {E,H} carry (ir.py:159-181; Probe B citing physics.py:53-64,116-127), and the L2 deferral reasoning holds (LEVELS.md:180-182). Every probe expressed state as host-threaded values + fresh closures at exactly one compile (Probe B: spec misses = 1, guard misses = 0, verified live; Probe E: three loops warm under `no_compile`).

**Amendment 1 — the stance is VACUOUS at today's joint surface.** pdum.dsl has no fold, no dispatch sequencing, no path to the tensorlib representation; the user is the bufferizer — sees the swap, keys Handles by `id(buffer)`, does alias analysis by hand while the exact alias theory (footprints/overlaps) sits unused across the seam (Probe B; DESIGN.md §2). The stance survives only if the seam lands (fold mirror, §7.3) and the raw form is guarded (Amendment 2).

**Amendment 2 — refusal-first is violated at the write seam, the probe suite's gravest verified flaw.** `out=` aliasing a captured read buffer silently corrupts (Probe B verified: all-zeros vs expected shift, no error); the grid launcher checks dtype, contiguity, and rank only — no overlap check exists (c.py:315-338, re-verified this session). Boundary drawn: out overlapping any captured read buffer is OUTSIDE the subset; refuse with "out aliases captured array 'u' — write into a second buffer (ping-pong), or wait for certified in-place." Safe in-place (FDTD's reads-at-i-only) returns later as an L2-certified rewrite, never a user gesture — today it is the user's unchecked hand analysis (Probe B boundary 1).

**Amendment 3 — the KV-cache exclusion is REJECTED as a joint-system exclusion.** Probe D demonstrates decode inside the subset today with zero new language: preallocated buffer + `out=` row adoption + rank-generic prefix-slice recapture (cache hits pinned by test_array_args.py:32-33). The recorded exclusions (zoo/__init__.py:19-21; both streams) are representation-level deferrals only; two independent deferrals do not add to a decision — charter §6 Probe D's charge, now discharged. Ruling: canonize the Probe D spelling as the supported boundary sample now; the representation story stays deferred to L2 with an erasure obligation — when the seam lands, the same surface program must emit a pure Program whose bufferization reproduces the row-write.

**Amendment 4 — mutable host state is legal and must be pinned.** In-place-mutated numpy captures pass the identity guard and are re-extracted per dispatch (registry.py:83-92,137; Probe E verified). No test pins this or the fresh-closure idiom (§7.7).

## 7.2 Memory residency and ownership — recommendation + one stated open question

Three residency classes, distinguished in the type, with the runtime (never syntax) moving data:

1. **Captured host values — the copy-per-dispatch contract.** AFFIRMED as shipped and measured: uniforms as ordinary Python objects, per-draw `write_buffer`, zero recompiles across value changes (Probe A; test_backend_wgsl.py:53-60; registry.py:126-137). The per-draw copy IS the contract (charter Probe A brief) — the probe's clearest winner.
2. **Adopted foreign buffers — the sharpest case, and the joint seam.** Foreign buffers enter via `out=` with dtype/contiguity/rank checked against the artifact contract (c.py:326-335) — correct refusals. Recommendation: the adoption descriptor for anything richer (Probe A's `adopt_vertex_buffer`) should BE tensorlib's Layout (affine map + box) — Probe A independently reinvented exactly it (Probe A F6), so one concept, not two. The audio f32 gap (grid kernels refuse f32 out, forcing an alloc+copy inside a real-time callback — Probe E E6) is a correct refusal marking a missing family parameter (element dtype), not a new concept.
3. **Owned/backend-allocated** — 090 §5 OWNED/ADOPTED, [planned]; no probe contradicts it.

**Open question, stated:** closing over GPU-resident arrays (Probe A's vertex shader over GPU buffers) has no mechanism — no ValueKind, no guard story for device memory (see §7.7). It is the next 090 §5 installment, not resolvable from probe evidence.

## 7.3 Pipelines: value- vs dispatch-level — DECIDED: two concepts (and a third semantics), one gating machinery

The charter's candidate "one pipeline that fuses when it can and records a command sequence when it can't" is **REJECTED**. Evidence is unanimous: the shipped `|` fuses unary stages into one artifact (build_pipe, combinators.py:254-278; unary constraints verified at combinators.py:262-263,271-272) and served **zero** probe programs — Probes A, B, and E each report "pipeline fit: zero," and Probe D's training step needs n-ary composition the unary pipe cannot express. What every probe wants instead is **dispatch sequencing** — frames, sub-steps, residual→prox with different domains per stage, training steps — which has no syntax and no plan step (020:324 remaining list and 130 §4.3 cover neither; Probe B). Probe A adds a third semantics neither fuses nor sequences: vertex→fragment pairs into a PSO with the rasterizer between the stages.

An operator that silently chooses fuse-vs-record hides exactly the boundary this language exists to make loud: materialization between dispatches is the cost event, and §1's founding example is a dispatch-granularity mistake. Ruling:

- Keep `|` = value-level fusion (assemblage + device-function composition); widen to n-ary through the roles machinery when landing Probe D's optimizer (the demand exists).
- Dispatch-level sequencing enters as the pdum.dsl mirror of tensorlib `fold` — `host.fold(step, init, dim=..., steps=...)` with an explicit carry contract (Probe B B3; ir.py:159-181 is the op, one seam away) — plus the planned orchestrate encode plan the wgsl refusal already names (wgsl.py `_param_types` refusal text, verified).
- Pairing (PSO) becomes a third semantics tag. The existing registry already gates by `(op, left-role, right-role) → semantics tag` ("fuse", "terminal", later "orchestrate") (combinators.py:67-71, verified) — that is the right seam: ONE gating machinery, THREE semantics, and the semantics must be visible at the composition site, never inferred.
- Until the sequencing construct lands, bare host statements are the honest spelling (Probe E: "read fine") — do not force `|`.

## 7.4 Named axes — AFFIRMED, with the launch boundary convicted

**Placement:** names-in-types with erasure is AFFIRMED — named/positional twins share one artifact (arrays.py:276-281; test_arrays.py:187), the one ladder principle the code already implements. Names run end-to-end on data through indexing and contraction; the breadth star is Probe E's ISTA: `matmul(A, x, i)` contracts "col", `matmul(A, r, j)` contracts "row" — both Gram-step contractions with **zero transposes**, names-first earning its keep outside ML entirely.

**Where names stop is the misfit: the launch boundary, completely.** Domain coords are positional `c0..ck` (c.py:208-217); grid params bind anonymously by order; `Named` as `out=` dies with a raw `TypeError` (Probe B verified; c.py:332 `tuple(spec)` path re-verified); canon's own tests need `moveaxis` to interpret grid results (test_grid.py:105,145). The write side is precisely where transposition is silent corruption (raw pointer writes, c.py:326-327). Ruling: (i) a designed refusal for Named-out now; (ii) "the out array's named shape IS the launch domain" (130 §4.1, verified at 130:100-110) is the right seat — affirmed as the goal; machine-bound dims keep axis identity and no charts, per LEVELS surface discipline, which Probes B and E both respected without strain.

**Name-genericity:** AFFIRMED as-is — the factory takes the dims tuple, names ride the type, each name-set is one more specialization axis (Probe B verified mechanically). No probe surfaced name-genericity pressure in leaf programs (Probe E: all four domains used concrete names); no new construct is needed.

**No dual universes:** AFFIRMED by construction via erasure. One amendment from breadth: struct fields are NOT named dims — GA's multivector components belong in records, not a length-4 axis (Probe E) — while tensorlib holds "a struct of same-typed fields is a categorical dim in disguise" (COMPUTE.md, verified in the complex-carrier passage). Adjudication: one concept viewed twice — record type at kernel level (positional fields, zero cost), categorical dim at assemblage level (D18 labels) — connected by a declared isomorphism at the seam, never two universes. This lands with the record system, which is half-built regardless (Probe E E2/E3).

## 7.5 The derivative family — kind × lens × level, mapped

| kind | lens | level | status |
|---|---|---|---|
| forward | function-centric (`jvp(f)`) | L2 transform | [exists], scalar args only (transforms.py:294-296); array tangents refuse correctly (test_array_args.py:91-92) |
| forward | value-centric (`D(x)`, `fwidth`) | in-kernel L0 | [exists], all-params/no-wrt (transforms.py:312-335, verified) — convicted below |
| backward | function-centric (`grad`) | assemblage | [exists in tensorlib only] — program transformation, derived adjoints, keep-set DCE; **no pdum.dsl operator exists at all** (Probe D) |
| backward | value-centric (tape lens) | — | nowhere, and ruled to stay nowhere at kernel level |

**The ambient/tape rule — RATIFIED and sharpened.** Value-centric derivatives w.r.t. distinguished ambient parameters are forward duals: no tape, zero cost unused (test_transforms.py:142) — D's mechanism confirms the charter's rule. Value-centric w.r.t. arbitrary upstream inputs needs recorded provenance — but at the assemblage level the program already IS data (linear SSA), so tensorlib's `grad` needs no runtime tape either. Sharpened rule: **the tape is never a runtime object at any level.** Value-centric backward is admitted exactly where the program is data (assemblage) and refused inside kernels; value-centric forward is admitted in kernels w.r.t. ambient parameters only. That is the precise boundary §7.5 asked for.

**D's current design is convicted** (finding 3): all-params-always fails the founding domain — any fragment signature richer than bare float pixel coords refuses (`D differentiates w.r.t. float params`, transforms.py:326-327 verified; Probe A F3) — and coordinate dtypes diverge per family (grid i64, c.py:299; wgsl f64, `_param_types` verified), so `fwidth` works on one compute family and refuses on the other, violating equal citizens. Right-level fix: families declare **named ambient axes**; `D(x, wrt=axis_name)` selects them — which also fixes `fwidth`'s hardcoded `d[0]/d[1]` (graphics.py:45-55) and merges with §7.4's launch-domain naming. `wrt`-as-structure stays deferred: no probe demanded it. The orphaned `D(x, wrt=local)` deferral (110 §5, dropped by resequencing) is superseded by this ruling.

**Subgradient ruling: COMMITMENT, not accident.** Each kink pick (abs pred "ge", min "le", max "ge" — transforms.py:223-227 verified) is a valid subgradient element; Probe E's convex consumer is served by determinism, not change. Obligations: pin at-kink values in tests (test_jvp_rules.py:94's POINT is deliberately away from kinks — a coverage gap), and document "jvp returns a fixed subgradient selection at kinks" as frozen contract. Cross-stream: tensorlib pins its tie behavior (reduce-max gives the full cotangent to every tied element, test_autodiff.py:377-387); pdum.dsl must pin its analogue — kink semantics currently diverge unpinned between streams.

**Rule-table adjudication (§2.4): CONVERGENCE, once the growth law is stated.** Both streams have a hand table over scalar primitives (JVP_RULES, transforms.py:217-228, 8 entries verified; tensorlib's if/elif chain, autodiff.py:225-289) and both derive composites (through-the-table differentiation; defmarker tree rewriting, defreducer BPTT, fold self-application). The collision is growth polarity, not existence. The law: **the table grows only when a scalar primitive joins the core; everything else derives.** Restate pdum.dsl's "the transform column is open" claim (transforms.py:141-143) to this law — as written it is broader than the code (inventory F6).

## 7.6 Where tensors come in — the kernel stays tensor-free; the fork escalates

**AFFIRMED by breadth:** no probe needed tensor semantics in the kernel; the five surfaces carried GA vocabulary with zero kernel changes (Probe E); Pipeline-as-fusion and jvp-over-arrays — the ML-leaning concepts — contributed nothing at breadth (Probe E §7), consistent with satellite status. Tensors enter exactly twice: as erased type freight at the seam (Named/Shaped) and as the satellite tensor dialect whose extent iteration lives in map/reduce/scan/fold.

**The no-extent-loops principle is RIGHT and currently unenforceable.** Probe E's IIR proves it is an asymptotic trap, not style: the only expressible spelling is O(N²) per block, admitted rather than refused, while tensorlib holds the exact answer (defreducer linrec, mdsl.py:400-413 per Probe E). The flagship attention sample violates the principle **by mandate** (020:324 specifies softmax as loops; build_chapters.py:4078-4083). Ruling: the principle stands; the sample must be rewritten — but only after the primitives exist. **The scan door is the single highest-value primitive to land**: it serves audio (workhorse), attention (streaming softmax), and convex (running reductions), and its semantics sit one seam away.

**The gravest structural fact (Probe D), escalated:** 130 plans a second tensor IR (`tensor.map/reduce/contract` defaulting to `core.for` decompositions) and a second AD inside pdum.dsl, while the charter mandates fronting tensorlib's Program (§5) and mdsl's zero-import Node schema was built precisely for external frontends (mdsl.py:1-17). The joint seam has **zero tests** (behavior-spec inventory). This is an unadjudicated fork, not a gap — escalated per §8 governance (finding 2). Interim conviction both streams' own principle supplies: today's `matmul` is a lowering-time macro that erases contraction at birth (transforms.py:381-396), violating "named forms are the working representation until target selection" (130 pre-§1) and "preserve reduce/scan structure" (LEVELS.md:79-83) simultaneously — the standing counter-example Probe C independently convicts.

## 7.7 Caching identity at the edges — restated per situation; the sketch ENDORSED with two amendments

Restatements, each from probe evidence:

- **Per-step parameter updates / live knobs:** types-only key + identity guards + per-dispatch extract handles them at zero recompiles — verified live (Probe B: factory-per-iteration, 1 compile; Probe E: three loops warm). UNPINNED: no test covers the fresh-closure idiom or mutated-capture re-extract. Pin both; canonize the fresh-closure idiom once for all foreign-loop domains (Probe E §0 — it is load-bearing, not stylistic, because the grid family refuses the argument channel, registry.py:167-169).
- **Cross-Handle record sharing:** ping-pong's compile-once rests on two Handles hitting one record with guards referencing only the first Handle's cells forever (registry.py:83-92; cache.py:62-70) — correct today because extract re-reads current captures, but unpinned and it pins the first Handle's buffers alive via the guard tuple (Probe B finding 4). Fix at the right level: re-key guards per probing Handle or weakref the cells; add the pinning test (finding 7).
- **Adopted/GPU-resident buffers:** identity-guard-plus-re-extract works only because extract reads host memory. Device-resident mutation under adoption needs an epoch/ownership handshake — open question, stated; no probe reached it.
- **Recorded command sequences (Probe A bundles):** key = (PSO specialization key, buffer-set identity, target formats); re-record on key change, replay otherwise — consistent with the format-in-key precedent (registry.py:30-31; 070:148-150).
- **Out never enters the key** — affirmed and pinned (test_grid.py:38-39); domains stage.

**The layout-fingerprint sketch: ENDORSED with two amendments.** The skeleton — rank, dim names/axis tags, zero/negative-stride structure, guard form, chart presence, machine bindings — computed on `canonical()` (order-free identity verified, layout.py:19,384-387), numeric content staged, codegen-relevant numerics as opt-in Literal-style predicates: this matches every pinned cache behavior (rank-generic hits, test_array_args.py:32-33,76-77) and the Literal-is-the-one-opt-in doctrine (types.py:116-131). **Amendment 1:** the skeleton must REPLACE NamedArray's order-carrying dims tuple as identity, not supplement it — `_NamedKind.typeof` bakes ordered dims into the type (arrays.py:131-134, verified), so presentation order will split the cache exactly as collision F3 warns. Caveat recorded: under v1's C-contiguous restriction order IS semantic (strides follow dims order), so the amendment lands together with stride-structure in the skeleton, at the moment adoption widens beyond C-contiguous. **Amendment 2:** distinguish the two tiers — bindings/guards/charts enter the *specialization* key where they change emitted structure, while the *artifact* tier stays content-addressed on Node.key as today; the sketch as written conflates them. Honest status: none of this exists in code — the evaluation is of the sketch, and the sketch survives.

## 7.8 Refusal UX — NOT one voice: three tiers observed

1. **Designed refusals** — strong on both streams: pdum.dsl's frozen by contract test (test_refusal_contract.py); tensorlib's quote the fix verbatim ("slice(x=(1, 4))" — test_compute.py:12-17, D17). The grid argument-channel refusal is exemplary (registry.py:167-169).
2. **Internals-speak crashes where refusals belong:** whole-record expressions die with "no slot for env path (0,) ... run NORMALIZE_ENV first" (Probe E E2); `extend()`+record fails with "no ValueKind registered for 'MV2'" (Probe E E3); `Named` as out= is a raw `TypeError` (Probe B).
3. **Silence where refusal is mandatory:** out-aliasing corruption (Probe B, verified); and the matmul extent mismatch **pinned as documented UB returning 3.0** (test_refusal_contract.py:129-137, re-verified) — a silent wrong answer frozen as spec, in direct collision with §1's refusal-first while tensorlib's guard machinery proves the joint system can check it. This is a decided position (the 100 posture) colliding with the charter's own principle — escalated per §8 (part of finding 8).

**The one-voice contract:** adopt tensorlib's quote-the-fix as the law, pdum.dsl's test-freezing as the mechanism. Concretely: one shared refusal shape — what happened, the principle violated, the quoted fix, the source loc — carried as a formatting contract across both streams' exception types (MissingRule/NameFateError/VerifyError; ValueError/NotImplementedError/SignatureError), frozen by a joint contract-test battery. The probe-sketched refusals (alias, Named-out, scatter boundary, f32-adopt) all read in one voice when written to that shape.

### Appendix: unverified low-ranked candidates (dropped at the verification cap)

- [medium] ad-extension-surface: The AD extension surface is narrower than advertised: JVP_RULES entries receive only (emit, node, args, materialized-tangents) with no region support and no structural-zero participation, while all core-op and control-flow linearization is a closed if-chain inside _Tangents._rule — a dialect op with regions or zero-aware rules cannot register a linearization, and the hand table itself collides with tensorlib's derive-don't-enumerate philosophy (charter §2.4 requires adjudication). (transforms.py:59-147 (closed if-chain incl. core.if/core.for handling); transforms.py:141-147 (JVP_RULES fallback: rules get pre-materialized tangents, no regions); transforms.py:217-228 (8-entry hand table); charter §2.4:206-213)
- [medium] doc-status-staleness: 130 is marked PROPOSED while its stage 1 and half of stage 2 have landed and been canonized in the 010 ledger, and 110's header already treats 130 §7 as landed — canon readers cannot distinguish landed from planned. (130_tensors-tiles-and-over.md line 3 ('Status: PROPOSED'); 010_proposed-architecture.md:984-1035 (step 13 seams + step 14 first installment ledger entries, kernel 1280/1500); 110_transforms-and-derivatives.md lines 3-6 ('AMENDED … §5's composition deferral are resolved'); 020 step-14 row (landed vs 'remaining' split).)
- [medium] pipeline-house-style: 020's commitment that 'combinator style is the house style for examples' from the CPU-backend chapter on is contradicted by the taught model: no pipeline appears after ch04; ch12-ch14 exclusively teach closure-factory + direct dispatch, so the charter §5 load-bearing composition idiom is untested against arrays, transforms, and the grid. (020_implementation-plan.md:143-145 (house-style commitment); grep of scripts/book/build_chapters.py: pipeline references confined to lines 744-1102 (ch03/ch04); ch12-ch14 samples (e.g. build_chapters.py:4074-4104) use make_* factories only; charter §5 'Composition via capture + pipeline' bullet.)
- [medium] collectives-adjudication: Cross-axis collectives are held in three mutually inconsistent positions across the two streams: deferred ops over woven axes (110), named tile ops (130 §6), and tensorlib/charter's 'collectives are not ops — a cost pass reads them off the algebra'. (110_transforms-and-derivatives.md §1 ('cross-lane operations … have no home in a woven representation') and §5 (deferred 'collectives over woven axes'); 130_tensors-tiles-and-over.md §6.1 (reduce/scan/sort as named tile ops); charter §2.2 ('Collectives are not ops; they are read off the algebra by a cost pass').)
- [medium] backend-parity: 130's tile-family design treats CUDA as the neutral target — `contract` '≡ tensor cores' / 'maps 1:1 onto mma.sync', staged plan is 'CUDA + tiles … Metal follows', WebGPU absent — against the charter's equal-citizens rule and 010's own three-compute-target diagram. (130_tensors-tiles-and-over.md §5 (mma.sync mapping; ThunderKittens/Triton framing) and §8.4 ('CUDA + tiles … Metal follows', no WebGPU row); charter §2.1 ('no probe or design may treat one of them as the neutral default') and Probe C backend-parity bullet; 010 §0 diagram (WGSL/CUDA-C/Metal/C/Python).)
- [medium] order-carrying-type-identity: pdum.dsl's NamedArray type identity is an ordered dims tuple (repr and typeof preserve order), so same-named arrays in different presentation orders are different specialization keys — colliding with tensorlib's D5/'order never' where canonical() is the order-free identity, and with the charter §7.7 sketch that fingerprints must be computed on canonical() so presentation order never splits the cache. (src/pdum/dsl/stdlib/arrays.py:59,62,131-134 (dims: tuple ordered; repr 'array<dtype,d1,d2>'; typeof keeps order) vs explorations/tensorlib/tensorlib/layout.py:17-19 (D5), explorations/tensorlib/README.md:29-31, charter docs/design/140:659-670)
- [medium] ml-pde-scoping-vs-breadth: Tensorlib's sufficiency thesis and its entire validation corpus are explicitly ML+PDE-scoped ('suffice for most of modern deep learning and much of PDE physics'; zoo = transformers + heat/FDTD only), and its named exclusions (gather/scatter, data-dependent control flow, no branching) fall exactly on the charter's breadth roster (discrete simulation's branch-heavy ticks, convergence-tested convex solvers, audio callbacks) — so joint-system verdicts built on tensorlib evidence inherit an ML+PDE bias the charter names as the failure mode. (explorations/tensorlib/COMPUTE.md:9-11 (thesis), COMPUTE.md §5.4-5.5 (gather excluded, control flow deferred), explorations/tensorlib/tensorlib/zoo/__init__.py:9-18 (spanning set is transformers+physics), mdsl.py:30-33 (no control flow, where-is-the-branch); charter §1 'Breadth is identity' (docs/design/140:60-68))
- [medium] joint-seam-untested: The charter's declared target — pdum.dsl syntax emitting tensorlib Programs (capture→input-leaf naming, wrt→DCE keep-set) — has zero tests in either stream; every 'the frontend can target it without rewrite' claim rests on README prose, not pinned behavior. (charter §5 ('the joint system is the target', quoting tensorlib README's pluggable-frontend claim, 140 charter :330-337); full listing of tests/ and explorations/tensorlib/tests/ shows no test importing both pdum.dsl and tensorlib)
- [medium] refusal-voice-fragmentation: Refusal contracts are pinned per-stream with incompatible taxonomies — pdum freezes MissingRule/VerifyError match-strings, tensorlib uses ValueError/TypeError/SignatureError/NotImplementedError with D17 quote-the-fix recipes — and no test anywhere pins exception class or message style across the seam (charter §7.8's one-voice question). (tests/test_refusal_contract.py:33-174 (MissingRule/VerifyError matches) vs explorations/tensorlib/tests/test_compute.py:12-17 ('slice(x=(1, 4))' recipe), test_placement.py:37-45 (bind recipe), test_signatures.py:76-109 (SignatureError); charter §7.8 (140 charter :671-674))
- [medium] aliasing-and-mutation-unpinned: Neither stream pins any behavior when the grid out array aliases a captured/argument array, nor what per-step in-place mutation of a captured array does to capture guards and cache identity — exactly the semantics Probe B's ping-pong and Probe D's parameter updates rely on. (tests/test_grid.py:67-91 (launcher validates dtype/contiguity/rank only; no aliasing case); tests/test_array_args.py and tests/test_refusal_contract.py contain no mutation-after-capture test; tensorlib excludes mutation by design (charter §7.1, 140 charter :564-571))
- [medium] uniform-sharing: A vertex+fragment pair sharing uniforms has no representation: each Handle gets its own env, plan, staging, and uniform buffer, so a Camera captured by both stages is packed and uploaded twice per frame, and WebGPU compiles the pair into ONE pipeline whose bind-group layouts must agree — nothing computes a merged layout. (registry.py:195-201 (per-record plan/staging/launch); wgsl.py:354-366 (per-artifact uniform buffer and env-only bind group); capture.py:88-96 (env per Handle); no cross-handle plan machinery exists anywhere in kernel/pack.py (plan_from_types takes one env_types tuple, pack.py:164))
- [medium] vertex-array-adoption: Backend-allocated vertex arrays have a type-system slot but no entry mechanism: the Array device axis exists and 090 §5 fixes the OWNED/ADOPTED zero-copy contract, but every array ValueKind hardcodes device='cpu' and refuses non-numpy, and a vertex array needs what a raw GPU buffer does not carry — an attribute layout (strides, offsets, per-instance step), which is exactly a tensorlib Layout over a foreign buffer. (types.py:85-86 (device axis, 'dispatch tier 1 at step 14'); 090_core-and-extensions.md:143-171 (OWNED/ADOPTED, zero-copy both directions, 'first real implementation rides the graphics draw surface work'); arrays.py:94-101 (_summarize: numpy-only, C-contiguous-only, device 'cpu' literal); arrays.py:144-154 (the xarray kind — 'typeof IS the adapter' — is the adoption pattern to extend); tensorlib layout.py:1-19 (affine map + box + named dims + byte strides = a vertex-buffer-layout descriptor verbatim))
- [medium] raster-composition-semantics: vertex->fragment composition is a third pipeline semantics that the machinery anticipates but no concept covers: pipe fuses by inlining and orchestrate (planned) sequences dispatches, while a draw pair must pair two kernels into a PSO with a rasterizer between them and check the varyings interface — and that interface check is impossible today because FnType carries no result type, so the vertex stage's output record is unknown until lowering. (combinators.py:254-278 (build_pipe = sequential inlining, one threaded value); 070:125-134 (orchestrate encode plan, planned); combinators.py:67-78 (register_composition semantics tags; 'fragment' pre-named as a future role); types.py:176-189 (FnType = template + env_types only); wgsl.py:66-71 (pipelines refused on the wgsl family today))
- [medium] instancing-as-binding: Instancing maps onto the axis-binding move — bind the 'instance' axis to the draw command's instance dimension — which supports the charter's over-reabsorption hypothesis, but over's current contract blocks it twice: the lane must be a trailing i64 argument (instance count is launch-domain data, not a per-call argument) and 'no capture carries the axis' refuses when per-instance data is an adopted GPU buffer rather than a Named numpy capture. (transforms.py:271-272 (trailing-i64 lane contract), :283-286 (refuses unless a capture carries the axis); c.py:290-299 (the grid backend unwraps Over chains to turn lanes into domain coordinates — binding lives in the registry, not the representation); charter 140:200-204 (over as syntax-level shadow of split+bind, to be reabsorbed))
- [medium] nameless-launch-boundary: Axis names stop dead at the launch boundary: domain coordinates are positional row-major, over-lanes trail by an outermost-last convention, the out array is nameless, and passing a Named array as out= raises a raw TypeError ('Named' object is not iterable) rather than a designed refusal — reintroducing silent transposition exactly at the raw-pointer write side. (Verified TypeError by execution (check3, from tuple(spec) at backends/c.py:332); positional coords c.py:208-217; lane order transforms.py:253-257; canon tests need moveaxis/transpose to interpret results test_grid.py:105,145; names-kill-transposition posture arrays.py:19-22)
- [medium] unpinned-cross-handle-cache-sharing: Compile-exactly-once for ping-pong rests on cross-Handle specialization-record sharing that no test pins, and the shared record's guards reference only the first Handle's closure cells forever — correct today only because extract re-reads the current target's captures, and it pins the first Handle's buffers alive via the guard tuple. (Verified misses=1 across two Handles with different captured buffers (check1); guards built from the compiling target's cells registry.py:83-92; identity check against those same cells on every probe cache.py:62-70,206-222; no test in tests/ creates two factory Handles over different arrays under no_compile (test_grid.py:36-39 reuses identical captures))
- [medium] boundary-condition-two-languages: Boundary conditions are written twice in the joint system with no bridge: branchy per-cell index arithmetic in pdum.dsl (four conditional ghost reads per heat2d cell) versus declarative shift+slice+pad guard-fill in tensorlib where 'the boundary condition IS the guard fill' — and no plan brings guards/fill to the pdum.dsl surface. (B1 cell body (verified working) needs 4 conditional expressions; tensorlib form physics.py:36-42 and guarded.py; 130 §4.2:130-141 plans tensor.slice but no guards/fill)
- [medium] grid-reads-only-by-closure: The grid family conflates coordinate params with data operands: because any positional argument is refused when param_types derives, kernels can read buffers ONLY via closure, so a kernel cannot declare 'I read U and write U_next' and the dispatch site shows no dataflow (the verified host loop must key Handles by id(src)). (registry.py:167-169 (refusal of args when params derived); _grid_param_types makes every param a coordinate c.py:290-299; argument arrays work on the scalar family test_array_args.py:25-77; B1 using-side dict keyed by buffer id (verified program))
- [medium] token-discipline: 130's token story is internally inconsistent: §4.4 claims a misordered kernel is a type error because cooperative ops consume tokens, but §5's own GEMM discards the barrier() token and contract takes no token operand — the flagship is ill-typed by its own rules, or the tokens are implicit and the type-error claim is vacuous. (130:170-174 ('cooperative ops consume tokens... a misordered kernel is a type error (missing token), not a race') vs 130:187-191 (barrier() result unused; contract(a, b, axis="k") tokenless); 130:283-285 leaves only token granularity open, not threading visibility.)
- [medium] partition-surface-missing: §3.1 step 3 (partition: name the IR chunks that become kernels) has no surface, no API, and no representation in either stream — the architecture's pivotal step is currently unwritable. (tensorlib ir.py has no grouping construct (Program is a flat instr tuple, ir.py:92-119) and its transforms are DCE/checkpoint only; pdum.dsl has nothing between whole-Handle dispatch and nothing; charter §3.1 step 3 (140:235-240) presumes an analysis that 'identifies chunks'; LEVELS.md K-A (190-197) debates the kernel's IR form but no doc sketches how a human/agent DESIGNATES one.)
- [medium] webgpu-tile-parity: The tile plan encodes CUDA as neutral: contract is defined as '≡ tensor cores' / 'maps 1:1 onto mma.sync' and stage 4 is 'CUDA + tiles... Metal follows' with no WebGPU row, though WGSL has no base-spec matrix op — violating the equal-citizens rule the charter re-affirms for this probe. (130:191,203-204 (contract ≡ tensor cores, mma.sync), 130:267-270 (stage 4: CUDA then Metal; WebGPU absent); charter 140:103-108 ('the three compute targets are equal citizens... no probe or design may treat one of them as the neutral default') and Probe C bullet 140:470-475; the decomposition gate that would give WebGPU a valid lowering already exists in the design (130:130-137).)
- [medium] ad-partition-mechanism: The flash evidence for amendment two is currently mis-homed: the fused (flashsm) form is CHOSEN by the L0 author (naive=False), so the fusion decision leaks into §3.1 step 1 instead of being a lowering rewrite — and the tile vocabulary cannot express the streaming form anyway, because 130 §6.1's reduce takes only scalar monoid ops while flash needs structured (m,l,o) state. (zoo/attention.py:122-142 (author selects flashsm vs naive softmax at build time); tests/test_zoo.py:66-79 (backward derived from the combine, equal to naive — the license works); 130:212-215 (reduce/scan/sort with scalar op=max style only); tensorlib defreducer carries exactly the needed structured state (mdsl.py:24-30).)
- [medium] nary-composition-gap: The load-bearing capture+pipeline idiom cannot express the training step's essential DAGs — global-norm clipping (grads list → scalar → scaled list) and the Adam step (params+moments in, params+moments+loss out) — because Pipeline threads exactly one value and stages must be unary, so the probe's optimizer had to fall back to ordinary host Python everywhere. (src/pdum/dsl/combinators.py:262-263 ('pipe threads exactly one value'), 271-272 (stage must take exactly one argument); P5/P6 programs in the digest — no tagged construct could carry the multi-tensor DAG)
- [medium] wrt-cost-semantics: tensorlib grad's wrt parameter computes the full backward for ALL variables and only filters the returned mapping — freezing saves zero work unless the caller separately composes dce with the right keep-set — a cost-semantics trap in a library whose identity is cost transparency, and the surface our syntax must not replicate. (explorations/tensorlib/tensorlib/autodiff.py:867-876 (wrt filters the grads dict after the full backward is built); explorations/tensorlib/tensorlib/transforms.py:48-59 + explorations/tensorlib/tests/test_transforms.py:48-58 (pruning happens only via the separate dce call))
- [medium] state-residency: Every candidate training-loop style re-crosses the host boundary with the full parameter+moment set each step — functional threading re-feeds all state through run/dispatch inputs, and no device-resident persistent state, buffer donation, or L2 story exists in either stream — so training is host-bandwidth-bound by construction and the moment-buffer question is really the first L2 requirement. (explorations/tensorlib/tensorlib/ir.py:289-296 (run takes all inputs per invocation, pure); src/pdum/dsl/kernel/registry.py:126-137 (dispatch extracts and packs all capture leaves per call); L2 deferred per inventory (LEVELS L2 'deliberately deferred'); tests/test_runtime.py:47-54 proves identity stability of the rebuild loop but not residency)
- [medium] rng-dropout: Initialization poses no cache collision (host-side values, type-keyed identity), but device-side randomness — dropout, the one RNG a training transformer needs per-step per-element — has no primitive, no plan, and no recorded boundary in either stream. (explorations/tensorlib/tensorlib/zoo/gpt2.py:30-40 (init = host numpy rng); src/pdum/dsl/kernel/valuekind.py:157-167,181-189,211 (all captures fingerprint by type — seeds never enter keys); no rand op in tl/ir.py op tables (ir.py:126-145, 67-68) and none in 130's dialect plan (130:123-141))
- [medium] executor-seam: One-invocation-many-dispatches has no owner: nothing exists or is planned that takes an emitted tensor-tier program and issues multiple kernel dispatches (charter §3.1 steps 3-5 have zero code), so every tensor-tier program in this probe executes only on the reference interpreter — and no doc names the executor seam that the using-side syntax must stay agnostic to. (explorations/tensorlib/tensorlib/ir.py:289-296 (run = the only executor of Programs); src/pdum/dsl/kernel/registry.py:126-137 (dispatch executes exactly one artifact per call); 130:250-270 (staged plan contains no Program-partitioning or multi-dispatch step))
- [medium] surface-c-global-by-accident: Surface C (record) registration is two-homed and layering-hostile: Registry() defaults to the shared global BUILTINS kind table so record(fresh_registry, cls) mutates every registry's decoration-time typing (illusory extension locality), while record(DEFAULT.extend(), cls) — the documented layering route — fails at @jit decoration because make_handle types captures against global BUILTINS, not the session registry. (registry.py:99 (table: KindTable = BUILTINS default, shared object); surfaces.py:106 (registry.table.register); capture.py:141 (make_handle table=BUILTINS default) and capture.py:152-153 ('the session registry, later' — recorded TODO); empirically verified (scratchpad probe_e.py): record on DEFAULT.extend() then @jit -> TypeError 'no ValueKind registered for MV2' (valuekind.py:140); test_surfaces.py:245-263 passes only because Registry() shares BUILTINS.)
- [medium] operator-extension-no-surface: Per-type operator extension (geometric product as *, and equally complex numbers, quaternions, units) has no registration surface: the base pack maps ast operators to core ops unconditionally, and the working zero-kernel-diff route is wholesale BinOp rule-chaining plus a manual generation bump that the friendly surfaces normally do for you — undisciplined for composition and a stale-cache footgun. (base_lang.py:27-34,95-99 (_BINOPS unconditional); five surfaces enumerated surfaces.py:1-14 cover named calls/methods/ops/spellings only; feasibility verified empirically (scratchpad probe_e4.py: chained ext.lower_rules[ast.BinOp] + ext.specializations.bump_generation() ran (R*v)*rev(R) correctly, zero kernel imports); _invalidate exists only inside the friendly doors surfaces.py:30-37.)
- [medium] records-capture-only: The struct system charter 2.1 promises is half-built: records enter kernels only as captures — there is no in-DSL construction (MV2(...) refuses as an unknown call) and computed aggregates degrade to Tuples that lose the type identity, so method chains like a.gp(b).gp(a) refuse after one step; GA survives only by abandoning typed structs for bare 4-tuples plus overload vocabulary. (Empirically verified (scratchpad probe_e2.py): a.gp(b)[0] works (-8.75), a.gp(b).gp(a) -> MissingRule 'no method gp registered for (f64, f64, f64, f64)', MV2(x,...) -> MissingRule 'cannot call MV2'; methods must return tuples since nothing emits a Record-typed value (base_lang.py:363-368 method path requires Record base; surfaces.py:88-91 fields scalar-only); tuple spelling verified working (probe_e4.py rotor sandwich); charter 140:90-93 shared-struct promise.)
- [medium] foreign-loop-f32-seam: The foreign-loop seam forces a per-callback allocation+copy for audio: callbacks hand f32 buffers, the grid launcher correctly refuses to adopt them against f64 kernels, but the base vocabulary cannot produce f32 values at all (casts go to f64/i64/bool only; f32-load times f64-literal refuses strictly with no way to cast the literal down), so f32-end-to-end compute is unreachable without rule surgery — and separately, jvp at kinks returns a deterministic valid subgradient (abs'(0)=+1, relu'(0)=+1) that the convex consumer needs committed and pinned, not left as an accident. (c.py:326-327 dtype adopt refusal (pinned test_grid.py:86-87); base_lang.py:36 _CASTS={float:f64,int:i64,bool}; empirically verified (scratchpad probe_e.py): f32*0.5 -> 'core arithmetic is strict: f32 vs f64', pure f32 add works and grid fills float32 out (c.py:312 _NPKIND has f32); kink values verified jvp(abs)(0.0,1.0)=(0.0,1.0), jvp(max(x,0))(0.0,1.0)=(0.0,1.0) per transforms.py:222-227 pred choices; at-kink behavior deliberately untested at POINT test_jvp_rules.py:94.)
- [medium] kind-vocabulary-unvalidated: Kernel kind is an unvalidated free string living in two unadjudicated homes (Handle field vs registry route): the Handle's kind stays 'device' under grid dispatch, and an unknown kind silently routes to the default backend instead of refusing. (src/pdum/dsl/kernel/capture.py:149-152 (read: 'capture is agnostic and does not validate it'); registry.py:121-124 (read: absent kinds use default_backend); test_grid.py:20-24,159-167 (read: same Handle, family chosen by registry); canon divergence #6 (kind two-homed, 010 §2.4 vs install_grid).)
- [medium] workhorse-status-unassigned: Per-lane host dispatch is the default installed family and the gate's baseline yet no canon doc assigns it a status — the charter §1 explicit obligation (first-class / debug-only / removed) is unmet. (src/pdum/dsl/demo/simple_shader/python.py:182 (read: PYTHON registered default=True); tests/test_grid.py:159-166 (read: per-lane triple loop as gate baseline); 010:940-942 'the wrong granularity' with no status assigned (canon inventory divergence #3).)
- [medium] jvp-growth-policy-and-kinks: The 'open transform column' framing overstates (JVP_RULES is 8 entries plus a closed core-op if-chain), and at-kink subgradient behavior (abs/min/max pick a side) is an unlabeled accident validated only away from kinks — the rule-table collision is adjudicated: table stays, but only under a declared frozen-primitives growth law with pinned kink semantics. (src/pdum/dsl/stdlib/transforms.py:204,217-228 (read: 8-entry table, 'surface-A-shaped' comment); tests/test_jvp_rules.py:94 POINT 'away from kinks' (behavior-spec inventory); tensorlib's own hand chain over primitives with derived composites (tl/autodiff.py:225-289, tensorlib inventory); charter §2.4/§7.5.)
- [medium] composition-forms-conflated: Four distinct composition semantics — fuse-inline value pipe, n-ary/pytree composition, dispatch sequencing, and vertex-fragment PSO pairing — are unowned or punned onto the strictly unary Pipeline, which cannot express gradient clipping, Adam, any DAG, or a render pair. (src/pdum/dsl/combinators.py:262-263,271-272 (read: exactly one threaded value, one arg per stage); Probe D P5/P6 (clipping/Adam inexpressible in the load-bearing idiom); Probe A (pair-into-PSO is a third semantics; draw(target) design incompatible with foreign loops); Probe C §8 (rewrite chains must not pun on |); charter §7.3.)
- [medium] normalization-pass-unscheduled: §3.4's cross-model reuse ('every model whose attention chunk normalizes to the same fingerprint') presupposes a Program-level normalization pass that exists nowhere (ir.py has none; canonical() is layout-only) and is scheduled nowhere; today's matmul additionally erases the contraction at birth, so pdum.dsl never even produces the named chunk to fingerprint. (explorations/tensorlib/tensorlib/ir.py (no canonical/normalize, grep this session); explorations/tensorlib/LEVELS.md:160-188 (no normalization step); src/pdum/dsl/stdlib/transforms.py:338-396)
- [medium] cost-oracle-dtype-blind: The capacity well-formedness predicate K-B needs (staged bytes <= sm capacity) would run on a cost model with uniform 8-byte itemsize, over-counting f16 staging 4x and wrongly refusing valid tilings — the oracle cannot see the precision dimension the descent optimizes. (explorations/tensorlib/tensorlib/memory.py:21 ('uniform 8-byte itemsize ... dtype-exact sizes later'); explorations/tensorlib/LEVELS.md:198-201,211-214 (K-B/K-E capacity plumbing))
- [medium] waterfall-vs-revisit: §3.1's enrich-then-partition ordering is invalidated beyond AD: checkpointing's min-cut (banned-op capacities over the pre-fusion program) and placement traffic plans are both stale after L4 fusion changes the op set and saved-set — Amendment 2 is one instance of cost-bearing passes not commuting with structure-changing rewrites. (explorations/tensorlib/tensorlib/transforms.py:16-27 (ban policy over pre-partition ops); docs/design/140_critical-assessment-charter.md:233-247; explorations/tensorlib/tests/test_zoo.py:66-79 (fusion changes saved-set))
- [medium] named-axes-launch-boundary: Names stop dead at the launch boundary — Named as out= raises a raw TypeError and domain coordinates are anonymous positional — exactly where transposition is silent corruption via raw pointer writes. (Probe B verified TypeError; c.py:332 tuple(spec) path (re-read); positional coords c.py:208-217; canon tests need moveaxis, test_grid.py:105,145; planned seat verified at 130:100-110 (out's named shape IS the launch domain))
- [medium] caching-order-identity: NamedArray's type identity is order-carrying (ordered dims tuple baked into the type), colliding with tensorlib's canonical() order-free identity — presentation order will split the cache at the seam; the §7.7 skeleton must replace, not supplement, the dims tuple. (arrays.py:131-134 (re-read: dims = tuple(...) into _summarize); layout.py:19,384-387 canonical() order-free (re-read); charter §7.7 sketch; rank-generic hits pinned test_array_args.py:32-33,76-77)
- [medium] cross-handle-guard-sharing: Ping-pong's compile-exactly-once rests on cross-Handle record sharing that no test pins, and the shared record's guard tuple references only the first Handle's cells — keeping its buffers alive for the record's lifetime. (registry.py:83-92 guard construction; cache.py:62-70 identity guards; Probe B finding 4 (verified live: two Handles, one compile, guards on first Handle only))
- [medium] refusal-voice-fracture: Refusals split into three tiers — designed, internals-speak crash, and silence — with no shared voice across the two streams, and one silent wrong answer is frozen as spec (matmul extent mismatch returns 3.0 as documented UB). (test_refusal_contract.py:129-137 (re-read: UB pinned, no refusal); Probe E E2/E3 internals-speak ("no slot for env path (0,)", "no ValueKind registered for 'MV2'"); Probe B raw TypeError; tensorlib quote-the-fix voice test_compute.py:12-17)
- [medium] precision-one-concept: A1's precision-demotion license (stated in f16 tensor-core terms) and the kernel-level f32-adoption gap (grid family refuses f32 out, forcing alloc+copy in a real-time audio callback) are one framework-level precision concept stated twice. (c.py:324-327 (grid_kind dtype refusal), tensor.py:48 (carrier), memory.py:21 (dtype-blind shadow convention); Probe E finding E6; architecture A1)
- [medium] nary-pipe-ml-only: Widening `|` to n-ary/pytree composition is evidenced solely by Probe D's optimizer (Adam/clipping); pipeline fit was zero in all four breadth sketches, making the widening ML-dialect work, not kernel work. (combinators.py:262-272 (unary constraints); Probe E §6 (pipeline fit zero of four) and §7 (satellite-consistent); 020:351-354 (differentiable-programming satellite already designated in canon))
- [medium] kv-decode-generalize-pattern: Canonizing 'the Probe D KV-decode spelling' as the supported boundary sample enters an ML-named instance into canon when the mechanism (preallocated buffer + out= sub-array adoption + prefix-window recapture) is a generic ring/window pattern whose breadth twin is the audio delay line. (test_array_args.py:30-33 (rank-generic prefix-slice warm hits), c.py:325-330 (adopt path); synthesis §7.1 Amendment 3; Probe E §1 (delay-line-shaped workhorse unreachable today))
- [low] stale-placement-docstring: placement.py's module docstring still claims 'Forward programs only in v1 — gradients do not yet carry bindings', contradicted by the landed placed-backward (autodiff restamps bindings and re-binds backward repeats; PLACEMENT.md and README record it) — stale documentation inside the evidence base this assessment treats as frozen. (explorations/tensorlib/tensorlib/placement.py:21-22 vs explorations/tensorlib/tensorlib/autodiff.py:168-172,295-313 (bind emission) and explorations/tensorlib/PLACEMENT.md 'Placed backward (landed after v1)', README.md:110-114)
- [low] perf-gate-number-drift: Both perf gates' pinned thresholds contradict their own documentation: the charter calls over's gate '16×' while the only pinned gate asserts ≥10×, and test_bench's docstring says 'fail 10 µs' while the assert fails at 40 µs — so the de-facto contract is the weaker number in every case. (tests/test_grid.py:148-173 ('≥10×', assert speed >= 10) vs charter §2.4 'Its 16× gate' (140 charter :207-209); tests/test_bench.py:40-49 (docstring 'alarm 5 µs / fail 10 µs' vs assert t.minimum < 40e-6))
- [low] single-out-multi-field-boundary: Multi-field sub-step kernels are excluded — one out per dispatch, scalar-per-cell results — forcing coupled same-domain updates into multiple dispatches with duplicated reads; acceptable as operator splitting but currently an implicit boundary, not a stated one. (Scalar-only result refusal c.py:204-205 (designed VerifyError); exactly one Out appended per dispatch registry.py:137; FDTD B2 runs as 2 dispatches/step (verified pattern); fragment MRT is in scope for Probe A per charter 140:393-394)
- [low] tile-size-identity: Two mechanisms claim the same job for tile sizes: 130 assigns TM/TN/TK to the 040 §3c config bracket ('block value-specializes') while LiteralType already exists as 'the ONE value-in-type opt-in' — and the needed polarity split (tile extents specialize, problem extents stage) is exactly what the existing type system expresses. (130:176-178 (tile sizes are config-bracket values) vs src/pdum/dsl/kernel/types.py:116-131 (LiteralType, 'the ONE value-in-type opt-in'); the staging polarity for problem shape is pinned behavior (domain never enters identity — tests/test_grid.py comment at :38-39 per inventory, and rank-generic cache hits in test_array_args).)
- [low] assurance-tier-overdraft: §3.4's 'certification makes agent-authored compilers trustworthy' overdraws tier-1: the only operative witness today is random-small-input numeric comparison with non-compositional tolerances, the flagship's associativity license is declared-not-verified, and random draws miss adversarial domains (masked -inf softmax rows, cancellation, tail tiles). (explorations/tensorlib/LEVELS.md:100-114; explorations/tensorlib/tensorlib/mdsl.py:293 ('declared, not verified'); explorations/tensorlib/tests/test_zoo.py:77-79 (rtol 1e-9/1e-6))
- [low] fold-owned-loop-bias: The §7.3 dispatch-sequencing answer (host.fold mirroring tensorlib fold) covers only owned loops, while three of the roster's loop domains are foreign (render loop, audio callback, engine tick) where no loop exists to fold — leaving the foreign half implicit risks fold becoming a default that assumes loop ownership, the training-loop assumption in disguise; separately verified: the someone-else's-loop unification is named exactly once (Probe E §0), with §7.3/§7.7 referencing, not renaming. (synthesis §7.3; Probe E §0; charter 140:539-544; registry.py:134-137 and 167-169 (fresh-closure extract + grid argument-channel refusal, re-read))
