# 170 — The integration, clean break: one workspace, two packages

**Status: PROPOSED — for owner review before any code changes.** This document
supersedes 160 (and with it the 150 direction memo, 130 stages 2–3, and the
array/transform canon of 100/110). It is written in the present tense: it
describes the system we are building, not a diff against the system we had.
History, provenance, and the record of what was reversed live in Appendix A
and in git; nothing else in this document looks backward.

---

## 1. The system

### 1.1 One workspace, two published packages

The repo is a uv workspace. The root is an unpublished development shell
(docs, scripts, workspace glue). Two members are published to PyPI in
lockstep:

- **`packages/dsl` → `pdum.dsl`** (dist `habemus-papadum-dsl`, the existing
  PyPI project — the name continues, the contents are the new core). The
  compiler-infrastructure package: reflection capture and name-fate analysis,
  `typeof`/ValueKind/KindTable type dispatch, the two-tier type-keyed cache
  (specialization + content-addressed artifacts), the Node/Region IR with the
  rewrite driver and lowering machinery, marshaling (PackPlan/ResultPlan),
  the registry with its five extension surfaces, the events seam, and:
  - **`pdum.dsl.scalar`** — the scalar statement language: the device-function
    syntax (if/for, strict joins, single tail return), records, the scalar
    intrinsics, and the straight-line/control-flow analysis used as a
    well-formedness predicate by every branchless tier.
  - **`pdum.dsl.render`** — the shared dominator-walking emitter that any
    backend renderer builds on.
  - **The reference evaluator** — a minimal, clean pure-Python executor for
    scalar kernels and device functions. This is the scalar tier's oracle:
    the twin every future device backend is differential-tested against. It
    is deliberately small and deliberately slow.
  - **The fuse pipe** — `|` composition of device functions
    (`twill(4,3) | weave | zoom(...)`), re-authored clean: stages, roles, and
    fusion-by-inlining, with the role/rule registry living on the Registry
    (never module globals). `|` means fuse-inline and nothing else.
- **`packages/tensorlib` → `pdum.tl`** (dist name: Q9). The assemblage
  language and representation: the layout algebra (affine map + box + guards
  + charts + units + placement), carriers, the four compute primitives
  (pointwise/reduce/scan/fold), the Program/Instr IR, reverse-mode AD with
  derived adjoints, the transforms (requested-gradients DCE, min-cut
  checkpointing, revolve), the cost semantics (opcount, peak memory,
  traffic), placement, signatures — converted onto `pdum.dsl`'s caching,
  naming, and capture. **`pdum.tl.zoo`** ships inside it, and `ir.run` is the
  tensor tier's reference executor and oracle. Tensorlib's design docs
  (LEVELS, PHILOSOPHY, DESIGN, CONCERNS, COMPUTE, REPRESENTATIONS,
  PLACEMENT) move with the package and are revised where this document
  changes them (Build, mdsl, carrier/dtype).

There are **no device backends in the tree**. Device backends (WebGPU, CUDA,
Metal) are built fresh in the L4 era against the distilled backend notes
(§3.3) and the two reference executors. The reference executors are the
exception to the purge: oracles are not backends.

### 1.2 The syntax stack

Six tiers, all vocabularies + well-formedness predicates over one frontend
machine (capture → typeof → lower), never separate grammars:

1. **Assemblage** — tensorlib's language: straight-line tensor programs over
   the four primitives and layout ops; host Python provides all control flow
   and composition. One optional entry-point annotation (`@assemblage`)
   provides lazy build-on-first-call with the two-tier cache; without it,
   call sites hand-manage building.
2. **The shared expression syntax** — ONE Python-expression form for scalar
   functions (pointwise marker bodies, reduction combines) AND straight-line
   tensor fold steps. Type-directed lifting: scalar-typed values lower to
   the scalar core; tensor-typed values lift arithmetic pointwise, plus view
   methods (shift/slice/pad/rename/with_charts/repeat/bind), reduce, and the
   two-operand reduce form. The invariant is straight-line/no-branching
   (enforced by `pdum.dsl` control-flow detection), not pointwise-only.
3. **`@compute` kernels** — coordinates via the `thread_idx(...)` ambient
   intrinsic; **explicit stores into writable argument buffers**
   (`img[y, x] = f(y, x)`); launch configuration at invocation only;
   **function-valued arguments** that inline at specialization with the
   argument's FnType (including capture types) in the cache key.
4. **Tile DSL** (L4 authoring) — stage/barrier/accum vocabulary; capacity,
   race, and convexity certificates checked on the elaborated result.
5. **Warp DSL** — straight-line, uniform-control, lane-complete;
   shuffle/ballot/mma-fragment intrinsics.
6. **Vendor escapes + external oracles** — vendor-namespace ops at visible
   portability cost; raw CUDA C / Numba kernels as test fixtures only. There
   is no CUDA-clone language.

Standing invariants: branchless at the top (assemblage) and bottom (warp),
control flow confined to scalar kernels and the host; ambient coordinates
ARE the launch-domain iotas, and the iota→thread_idx descent never
materializes them; invocation concerns (blocks, shared memory, streams,
pipelining) never appear in user programs — they become visible exactly
where a transformation step introduces them (§5 S.6).

**There is no `out=` invocation channel.** Kernels write through explicit
stores into writable arguments; foreign buffers enter through boundary
descriptors (§4); allocation conveniences can return later as runtime sugar.
Two principles of the old channel survive as law: the launch domain never
enters cache identity, and a writable argument that overlaps a readable
capture/argument refuses at dispatch with the ping-pong message.

### 1.3 The calling-convention matrix

Rows = caller, columns = callee. **I** inline, **L** launch, **C** compose,
**R** refuse. Graphics rows are absent pending Q4 — a recorded deferral.
Host-calling any kernel per-element remains available as debug/oracle use
only.

| caller ↓ \ callee → | Host | Device fn | @compute | Tile | Warp | Assemblage |
|---|---|---|---|---|---|---|
| **Host Python** | — | L (oracle) | L: launch config at invocation; day-one writable/readable overlap refusal; fn-valued args FnType-keyed | L | R | L(many); optional `@assemblage` |
| **Device fn** | R | I | I(body), kind-checked | R | R | R |
| **@compute** | R | I (incl. passed-in fns) | I(body), kind-checked | R | I (lane intrinsics) | R |
| **Tile** | R | I (epilogues) | I(body) | I | I (mma fragments) | R |
| **Warp** | R | I (scalar only) | R | R | I | R |
| **Assemblage** | — | C (marker: declared, never a callback) | L(select) via certified lowerings | L(select) | R | C: fold takes a Program |

Invariants: launches from non-host callers refuse (no dynamic parallelism —
the launch boundary is host-only and explicit); `kind` is a validated
vocabulary, checked at dispatch and at cross-family inline (a body using the
`thread_idx` ambient inlined into a non-kernel context refuses); the
composition semantics never share syntax (`|` fuse-inline; sequencing = host
Python + fold; PSO pairing rides Q4; rewrite chains have their own form).

### 1.4 Caching: one mechanism, three keyspaces

- **Assemblage tier.** Building a Program is the compile step. Tier 1 keys
  `(fp_head, arg_fp, generation)`; tier 2 is content-addressed on the built
  Program and supports **derivation-under-cache**: partials, component
  markers, and adjoint scanners are cache entries computed on demand from
  cache entries. Names are part of the cached artifact's ABI: `grad` returns
  a name-keyed map, so the capture→name law (Q5) is pinned by a
  rebuild-stability test. Open: the structural-capture identity rule (Q1).
- **Kernel tier.** The thesis unchanged: types and identity in the key,
  never values; live-knob captures ride the uniform channel at zero
  recompiles; function-valued arguments put the argument Handle's FnType in
  `arg_fp` — a different pipeline shape is a new artifact, the same shape
  with different captured values is a warm hit plus a uniform rewrite.
- **Descent tier.** `chunk_fp` over the named-op Program in canonical form;
  registry key = (normalized chunk skeleton, boundary contract incl.
  saved-set demand and layout classes, license set, capability set,
  rules-generation); value = chain + authored region + artifact + assurance
  tier; the chain is a mandatory content-addressed output of elaboration;
  trust attaches at assurance tier ≥ 2. Until a Program-normalization pass
  exists (Q12), both content-addressed tiers are honest **private caches**.

### 1.5 Extensions, punning, events

Vendor op namespaces spelled by one backend; capability flags checked at
build; `code_for_op` key-presence as the capability bit; capability-gated
`debug.print`; `record.artifact` as the escape hatch; rule-of-three before
any runtime abstraction. The adopt descriptor is where dictated encodings
live (§4) — buffer interop and precision facts are one concept. Foreign
tensor libraries are interop partners; `pdum.tl` is in-house. The events
seam lives in `pdum.dsl` and gains emission points at `pdum.tl`'s
compile-ish seams (Program build, adjoint derivation, descent
certification), so `forbid`/`no_compile` can pin "this training loop builds
zero Programs" exactly as they pin kernel compiles today.

### 1.6 The zoo gate

The acceptance instrument for every migration step. The zoo's tests run in
CI from step P2 on and stay green through every conversion:

1. Forward denotations vs numpy (rtol 1e-9 / atol 1e-12).
2. Gradients vs finite differences, **indexed by input names** — name
   identity is load-bearing; grad-map key drift is a failure.
3. flash == naive: forward AND the derived backward (no hand rule).
4. FDTD gradients carry their staggered charts.
5. Placement erasure bit-exact; exactly two gpu all-reduces on the megatron
   block; the erased program communicates nothing.
6. Cost oracles stable, modulo the one deliberate re-derivation when
   descriptor-fed sizes land (§4).
7. **Naming-law pins** — literal expected names (`"L0.wq"`, fold param
   names, derived `name.d{i}`) hardcoded in tests; a rebuilt closure maps
   the same capture to the same name. No dual-running of old builders: the
   contract is pinned by expectations, not by comparison with dead code.
8. The shared-axis extent refusal, pinned in the joint refusal battery.

---

## 2. Versioning and release

All packages are versioned **in lockstep** and published to PyPI **in
lockstep** by the release workflow. Concretely:

- One version source of truth: `packages/dsl/src/pdum/dsl/__init__.py`.
  `scripts/_versioning.py` re-points its anchor there and enrolls every
  `packages/*/pyproject.toml` (plus the root) as lockstep version files —
  bumps are applied across all packages in unison, never individually.
- `release.yml` builds and publishes **all** workspace members' dists in one
  run, tagged once. `habemus-papadum-dsl` continues as the dist of
  `pdum.dsl`; the `pdum.tl` dist name is Q9.
- The root pyproject becomes an unpublished virtual root: workspace glue,
  dev dependency groups (members via `[tool.uv.sources] workspace = true`),
  docs tooling. Nothing installs the root from PyPI.
- Release runs remain deliberate acts through the existing workflow trigger;
  nothing in the migration publishes as a side effect.

## 3. The purge

Git history is the archive. The living tree carries only the go-forward
system. Three rules govern every deletion: **distill before deleting** (the
load-bearing knowledge lands in canon or tests first), **pin contracts with
literal expectations** (never by dual-running dead code; where a comparison
fixture is genuinely needed during a conversion it is temporary scaffolding,
deleted when its gate retires), and **new code never references old code**.

### 3.1 Deleted from the pdum.dsl side

`dsl_reference/` (the frozen M0 asset — history now); `stdlib/arrays.py` and
`stdlib/transforms.py` (Named, over, jvp-as-pdum-concept, matmul — the
tensor tier owns all of it; the tangent-engine rows merge into the one
derivative table first, §5 S.2); the array half of batteries; `backends/c.py`
and the whole demo tree (python renderer, wgsl runtime, graphics batteries —
the fwidth residue is re-created at the shader tier when Q4 lands);
`combinators.py` (the fuse pipe is re-authored clean in `pdum.dsl`);
`viz.py` and `bench.py` (rebuilt against the new surfaces when needed); the
book (`docs/book/`, `scripts/book/`) — the new packages grow their own
documentation; the tests of every deleted module.

### 3.2 Deleted from the tensorlib side

`build.py` (the Build name-manager — replaced by core naming + the shared
syntax); mdsl's Sym tracer, `defmarker`/`defreducer` entry points, process
registries and `node_digest` (replaced by the AST producer and
cache-backed registries). **Surviving mdsl parts relocate:** the
`Arg/Const/Prim` Node schema (the declared stability boundary — the AST
frontend is a new *producer* for it, its five consumers never change), the
symbolic `diff`/`_D` machinery, and the CompositeReducer BPTT engine. The
`defreducer`-shaped *declaration API* (state/element/lift/combine/init/
project + declared associativity) survives with producer-swapped bodies — a
reducer is a structured declaration, not one expression.

### 3.3 Distillations (written in the same commits as their deletions)

- **Backend notes** (one short doc): the numeric policy (truncating integer
  div/mod with exact twins; float mod = fmod; u64/inf/nan constant
  refusals), artifact-carries-its-contract (the header that made the grid
  launcher's dtype/contiguity/rank checks possible), the WebGPU runtime
  learnings (synchronous readback is a fixed-latency protocol act;
  timestamp-query timing; uniform-plan/bind-group layout; encode and submit
  are separate acts and the encodable is the API), and the
  bench/instrumentation methodology (warmup, tuned evals, minimum as
  estimator; phase decomposition by seam-wrapping).
- **The aliasing lesson**: writable/readable overlap is silent corruption;
  it becomes a day-one refusal test at the `@compute` store seam (P7), not
  a memory.
- **The refusal voice**: one shape — what happened, the principle violated,
  the quoted fix, the source location — seeded as the joint refusal battery
  (P3) that every later refusal extends.
- **The oracle status rule**: per-element host dispatch of kernels is
  debug/oracle-grade; the reference executors are the only per-element
  consumers. Stated here once; it is the rule.

### 3.4 Docs and notebooks

`docs/design/010–160` move to `docs/design/history/` in the purge commit;
mkdocs nav is rebuilt around this document and the two packages' docs. What
was still load-bearing in 140/150 is restated in §8 (the L4/L2 runway) so
nothing in history needs to be read to proceed. Tensorlib notebooks 00–06
(layout, units, views, guards, adjoints) teach surviving semantics and move
with the package; 07–13 are re-authored or dropped as their APIs change
during P4–P6 — they stay out of published nav until revised. Fixes that
survive regardless of the purge land at extraction, not before: the
unknown-kind dispatch refusal and the record-value designed refusals (both
live in surviving core modules).

---

## 4. Precision

**Facts at the boundary, choices in the interior, carrier semantics
throughout.**

**The contract.** Semantics are carrier-valued end to end
(bool/int/rat/real/complex). No compute dtype exists on the user surface.
Dtype is a property of buffers and encodings **at the boundary**, recorded
in load/adopt/out descriptors, never on tensors mid-computation. **Exact
decode:** every finite bit pattern is a specific rational (int4+scale
decodes to `scale[g]·q`), so the denotation stays exact over exactly-known
inputs — a bf16 checkpoint is a *fact* the descriptor records, not a
semantic property of the program. **Explicit rounding:** where rounding IS
the semantics (QAT, stochastic rounding), it is the explicit exact op
`round_to(encoding)`; its AD rule is a declared policy (Q8). **The
discipline:** every precision appearance is a boundary fact, a descent
choice, or an explicit `round_to`; mid-program astype-as-semantics does not
exist and the IR has no op for it. Two edges, decided: inf/nan bit patterns
refuse at decode by default (an extended-real carrier is a recorded future
door, Q8); writing real-carrier results into an encoded out-buffer rounds
as part of the boundary contract (`denotation = encode_out ∘ f ∘
decode_in`) — the one implicit rounding, declared, never silent.

**The boundary descriptor** = buffer handle + Layout + Encoding (+ carrier
+ units) — today's `Tensor`, re-roled to the boundary. `Encoding` outgrows
np.dtype as a small hierarchy (NumpyEncoding; QuantGroupEncoding for
int4-nibbles-plus-scales over two buffer regions; FormatEncoding for
bgra8unorm-srgb with the transfer curve in decode), each declaring its exact
decode/encode. Flows: checkpoint load (file metadata → per-weight
descriptors), foreign adoption (an audio callback's f32 buffer is a
descriptor fact, not a refusal), writable arguments (descriptor + writable
flag + the overlap refusal). Interior program values carry carrier + units +
layout shadows only; the IR cannot mint encoding-bearing values (enforced at
the IR/signature layer). The reference executors' float64 interior is a
declared oracle property, never semantics. **Strides are element-unit in
semantic layouts, byte-elaborated at the boundary/lowering where the
encoding is known** (Q6); interior shadow layouts are relative-nesting-only,
and bufferization re-derives byte layouts from assigned encodings.

**The interior.** Precision enters at exactly three lowering points: (1)
**descent licenses** — taxonomy {none, reassociation, precision-demotion},
equivalence stated over the carrier denotation, tolerances and input domain
in the declaration, the license set in the registry key; the numeric tier
monitors divergence and never certifies it; (2) **L2 storage assignment** —
materialized intermediates get encodings chosen at bufferization; (3)
**machine-tree byte predicates** — consumers, not choosers; capacity checks
read lowering annotations plus boundary facts (there are no user types to
read). The worked check: weights bf16 and activations f32 as facts, a
real-carrier contraction as the program, f16 tiles + f32 accumulators under
a license at descent, the out descriptor encoding the result — and the
4×-overcount failure mode of byte-blind capacity checking is impossible by
construction.

**Fallback, with criterion.** The two-surface model (a family element-dtype
parameter + the descent license) is the recorded fallback. Before L4, the
mixed-precision/QAT sample (master vs bf16 weights, loss scaling) is
written in boundary-facts terms; fall back iff a required program's
*meaning* — not cost — depends on an interior encoding that is neither a
boundary fact nor expressible as `round_to`. Falling back is a written
owner decision, never a silent switch.

---

## 5. The syntax stack, worked

Tags: **[now]** = surviving code runs this today; **[build]** = this plan
builds it.

### S.1 Assemblage

```python
def rmsnorm(x, g, *, feat="e", eps=1e-5):          # straight-line; host Python composes
    ms = (x * x).mean(feat)
    sd = (ms + eps).sqrt()
    xn = x / sd.repeat(feat, x.extent(feat))       # broadcast stays a DECLARATION
    return xn * g.repeat_like(x, but=feat)
```

Ops are tensorlib's own; methods are sugar over emission; the alignment
refusal is unchanged. SSA names come from Python binding names via core
name-fate analysis; input names are declared, and `grad`'s map stays keyed
on them; dotted prefixes (`L{i}.wq`) come from host-level declaration.
**Lift rule (normative):** Python *numbers* lift to consts aligned to the
tensor operand (dims and charts inherited) — the one implicit lift,
const-only; tensor–tensor misalignment always refuses; `repeat` stays
explicit. **Vocabulary completeness:** the committed method set includes
`.bind(level=...)`, `iota_of(t, dim)`, and the two-operand reduce form
(the zoo needs all three); `Program`/`Instr` remain public hand-
constructible data, and hand-emits get names through the same core naming
contract. **Entry point:** `@assemblage` [build] — exactly a pdum Handle
(phase-A capture, two-tier cache, build at first call → Program + input
layouts). Subject to the Q1 identity decision for structural captures.

### S.2 The shared expression syntax

Type-directed lifting; straight-line enforced at lowering; bounded `if`/
`for` exist only in the scalar statement language, never over tensor-typed
values.

One definition, two consumers [build — the gate is a differential]:

```python
def gelu(x):
    return 0.5 * x * (1 + tanh(GELU_C * (x + 0.044715 * x*x*x)))
```

— lowered as a pointwise marker body under `ir.run` AND inlined as a device
function into a `@compute` kernel; the two paths must agree numerically.
The Node schema is the lowering target; the AST producer replaces the
tracer; captured constants become Consts with FnType-in-key semantics.

A reduction combine with record state [build]:

```python
def flashsm_combine(L, R):                     # State = (m, den, o)
    m = maximum(L.m, R.m)
    sl, sr = exp(L.m - m), exp(R.m - m)
    return State(m, L.den*sl + R.den*sr, L.o*sl + R.o*sr)
```

— inside the surviving structured reducer declaration (state/element/lift/
combine/init/project + declared associativity). The acceptance criterion of
this whole tier: marker-body granularity is preserved — declared, traceable,
differentiable bodies and declared combines — so derived composites (flash's
derived backward) keep deriving. That is a gate, not an aspiration.

A tensor-typed fold step [build]:

```python
def fdtd_step(E, H):
    dE = (E.shift(x=-1).slice(x=(0, N-1)) - E.slice(x=(0, N-1))).with_charts(x=h_chart)
    H1 = H + c * dE                            # c lifts, inheriting dims AND charts
    dH = (H1.slice(x=(1, N-1)) - H1.shift(x=1).slice(x=(1, N-1))).with_charts(x=e_chart)
    E1 = E + c * dH.pad(x=(0, N), fill=0.0)
    return E1, H1                              # carry; layout preserved — checked
```

Step input names = param names per the fold contract; the chart-inheriting
lift removes rechart boilerplate; nesting is just a function. (The captured
`N` in slice/pad is the Q1 counterexample, live in this sample.)

**The one derivative table.** The scalar tangent rules and the marker
partial rules are the same object — op → linearization, None = gradient-
free — and merge into one table in the core transform column.
`CompositeMarker.partial(i)` is forward-tangent application over the
lowered body (basis seed, DCE, registered `name.d{i}`) — derivation-under-
cache; the reducer BPTT engine consumes partials through the same
interface. **At-kink law:** the table is one-sided and partitions — at a
tie, exactly one operand receives the cotangent (first-wins). Reduce
adjoints derive through the pairwise combine and therefore inherit the
partition law; the change from the old full-cotangent-to-every-tie pin is
a deliberate, pinned semantics change (Q7), decided before convex
consumers exist.

### S.3 @compute kernels [build]

```python
@compute
def my_shader(f, img):
    y, x = thread_idx("y", "x")                # ambient intrinsic, NOT positional params
    img[y, x] = f(y, x)                        # explicit store into a WRITABLE ARGUMENT

f = twill(4, 3) | weave | zoom(center=(20, 50), r=20, scale=5)
my_shader(f, img, launch=grid(blocks=ceil_div(img.shape, 16), threads=(16, 16)))
```

- **Function-valued arguments**: the caching half is the existing thesis
  (same FnType → warm hit, captured values ride the uniform channel;
  swapping a stage → new artifact). The lowering half is built here:
  inlining through an FnType-typed parameter, arg-rooted ABI slots, the
  Handle value passed to the build alongside arg types. Guard policy for
  argument Handles is a recorded design decision inside P7.
- **Stores**: the effect-ordering representation (ordered body statements vs
  token threading) is the design sub-deliverable that gates this step (Q2),
  owner-reviewed before any store lowering lands; L2 bufferization consumes
  the same mechanism later. Day-one contract: writable-argument overlap
  with any readable capture/argument refuses with the ping-pong message;
  in-place returns only ever as an L2-certified rewrite.
- **Launch config**: invocation-only, rides the launcher, never any key.
  Threads-per-block is a value-specialized bracket (re-render on change, no
  identity change); blocks/streams are pure launcher data.
- **Iota unification**: the same kernel is expressible as pointwise over
  coordinate iotas; the iota→thread_idx descent is a rewrite stage whose WF
  predicate is "no iota reaches a materialization boundary"; the fused and
  assemblage forms are differential-tested against `ir.run` on small
  domains.

### S.4 Vertex/fragment [reserved — Q4]

`@vertex`/`@fragment` share the ambient contract; `fwidth` is the
wrt-ambient derivative at this tier; vertex→fragment pairing is PSO
composition (its own semantics, never `|`); the per-frame deliverable is an
**encodable** (render bundle / draw-into-pass) — the host owns the pass,
the submit, and the swap chain. FnType reserves an optional result-type
slot now (cheap at extraction; needed for the varyings interface check, and
useful immediately for fold-step diagnostics).

### S.5 Tile and warp [reserved — the L4 brief governs, §8]

The authored descent is not IR: it is the value of a certified-lowerings
registry entry; the kernel boundary in the Program is an erasure-preserving
annotation. Tile vocabulary: stage/barrier/accum, tile loops as split+bind
one level down; WF certificates checked on the result — capacity
(dtype-exact, from descriptors and annotations), race-freedom
(checker-owned tokens), convexity. Whether `mma` pattern-matches
mul→reduce or requires a stated annotation is the open half of Q-f6,
answered inside the L4 design. Warp: straight-line post-unroll, uniform
control, lane-complete. Below both: vendor punning and external oracle
fixtures.

### S.6 Invocation concerns appear only in transformation steps

```python
prog = flash_attention().program                   # zero launch facts
d = descend(prog, kernel=annotate(prog, [...]))    # kernel boundary = annotation
d = d.split("s", 64).bind("s.outer", "sm")
d = d.stage("k", at="shared", double_buffer=True)  # smem + pipelining visible HERE
d = d.launch(threads=(128,))                       # step-level, value-tier
art = d.certify()                                  # chain + WF certs + registry key
assert bitexact(run(prog, env), art.run(env))      # the zoo denotation is the oracle
```

---

## 6. The migration plan

Sequential steps; each ends with the surviving suites green. Gate suites:
**D** `packages/dsl` tests, **T** `packages/tensorlib` tests, **Z** the zoo
gate (§1.6), **B** lint + budget + docs build. Tests of deleted modules are
deleted with them — "green" means the go-forward system, not the museum.

**P0 — Workspace + release machinery (M).** `packages/dsl` and
`packages/tensorlib` skeletons; root becomes the unpublished virtual root;
`_versioning.py` re-pointed (anchor = `packages/dsl/.../__init__.py`;
lockstep enrollment of all member pyprojects); `release.yml` reshaped to
build/publish all members in lockstep; CI reshaped around the surviving
suites; loc-budget buckets redrawn (the tripwire discipline continues,
pointed at the new packages). GATE: CI green; `_versioning.py current`
passes; a local `uv build` of both members succeeds; no publish.

**P1 — The purge + the core lands (L).** git-mv the kernel engine into
`packages/dsl` (with `pdum.dsl.scalar`, `pdum.dsl.render`); write the
reference evaluator (clean, small); land the extraction-time fixes
(unknown-kind dispatch refusal; record-value designed refusals; the
FnType result-type slot reservation); re-author the fuse pipe on the
Registry. Delete everything in §3.1 plus `docs/book`; move design docs
010–160 to `history/`; write the §3.3 distillation notes in the same
commits. GATE: D green (ported scalar-language battery with literal
expectations, run on the reference evaluator); B green; the tree contains
no import of anything deleted.

**P2 — Tensorlib promoted (M).** git-mv `explorations/tensorlib/tensorlib`
→ `packages/tensorlib/src/pdum/tl` (zoo inside), tests to the member,
conftest path-hack gone, notebooks per §3.4. The zoo gate enters CI —
live from here to forever. GATE: T + Z green at the new paths; `import
pdum.tl`.

**P3 — Onto the core (L).** Process registries die: cache-backed
registries with derivation-under-cache; core naming replaces Build's hint
dedup (Build itself still alive for the zoo builders until P5); the Node
schema relocates to its declared stability-boundary module; events
emission at Program-build/adjoint seams; the joint refusal battery seeded
(§3.3) with the extent refusal pinned (gate 8). GATE: T + Z green;
idempotence pin (re-registering an identical marker yields one entry).

**P4 — The shared expression syntax (L).** The AST producer replaces the
Sym tracer; straight-line detection replaces trace-time refusal; the
structured reducer declaration keeps its API with producer-swapped bodies;
record-typed reducer state; tensor-typed lifting lowers fold steps to step
Programs; the merged derivative table lands; `CompositeMarker.partial`
re-derived under cache; **the at-kink re-pin lands as its own commit with
Q7 sign-off**. The tracer survives this step only as in-repo test
scaffolding for the producer-equivalence check and is deleted at the end
of the step. GATE: Z green with markers/reducers/fold-steps re-authored —
denotation-identical including flash's derived backward and FDTD charts;
the two-consumers differential (S.2) once P7 lands is noted as pending.

**P5 — `@assemblage`; Build dies (M).** Zoo builders re-authored in
S.1/S.2 (bind/iota_of/two-operand-reduce vocabulary included);
`@assemblage` lands (after the Q1 decision); the naming-law literal pins
land (gate 7); Build and the remaining mdsl entry points are deleted.
GATE: Z green on the re-authored builders with hardcoded name
expectations; D + T green.

**P6 — Precision: boundary-facts (M).** The §4 design lands before any L4
work: dtypes split (carriers semantic; Encoding hierarchy); descriptors;
constructors re-roled as boundary acts; IR-cannot-mint-encodings enforced;
machinery dtype sites converted; `round_to` with its declared AD rule (Q8);
descriptor-fed dtype-exact sizes replace the 8-byte convention (cost
oracles re-derived as their own reviewable diff); the QAT sample written
and evaluated against the fallback criterion. GATE: Z green with zero
denotation changes; the re-derived cost diff reviewed.

**P7 — `@compute` (L).** In order: the effect-ordering design (Q2,
owner-reviewed) → store lowering with the day-one overlap refusal →
`thread_idx` + the iota→thread_idx descent stage → function-valued-argument
lowering (guard policy recorded) → launch config. Execution runs on the
reference evaluator; device backends remain L4-era work. GATE: the S.3
example runs on the reference evaluator; the iota-unification differential;
the two-consumers differential (S.2); key-discipline pins (shape miss /
value hit / launch never keys / fn-swap miss); overlap refusals; the
compile-once thesis test for function-valued arguments.

**P8 — Runway handoff (S).** The tiled-matmul zoo entry; the license-schema
stub + the worked GEMM license declaration; the L2 blocker list; the open
registry (Q-items below, streams, device-resident state, normalization,
warp vocabulary, external-oracle fixtures, the operators door, adversarial
input families for flagship gates — one −inf-mask attention case seeded
into Z now). GATE: Z green including tiled-matmul; this document's §8
handed to the L4/L2 work as its brief; L4 tile work may start once Q4 is
decided.

**Not doing:** async, in any form. A CUDA-clone language. A pdum tensor
dialect, translation frontend, or second AD. `out=`. Archives, shims,
frozen-fixture museums, or dual-running dead code beyond within-step
scaffolding. Device backends before the L4 era. Publishing as a side
effect of migration.

---

## 7. Owner decisions

- **Q1 (high, before P5) — assemblage-tier identity for structural
  captures.** A captured int used structurally (fold counts, slice/pad
  extents) fingerprints identically across values under type-keyed
  identity, which would silently reuse the wrong Program. Options:
  operand-derived extents mandatory (`x.extent(d)`, refuse structural
  captured ints); promote structural captures into the type
  (Literal/Shaped precedent); or value-key the assemblage tier (building a
  Program is cheap; a wrong one is not — the plan's lean).
- **Q2 (high, gates P7) — store effect-ordering**: ordered body statements
  vs token threading. L2 consumes the same mechanism; this outlives
  `@compute`.
- **Q3 (high, P4 design review) — the differentiability acceptance
  criteria**: marker-body granularity and declared combines preserved so
  derived composites keep deriving; the flash derived-backward gate is the
  instrument.
- **Q4 (high, precondition of L4 tile work) — graphics.** Schedule the
  vertex/fragment tier or defer it with recorded reasons. S.4 reserves the
  shape either way.
- **Q5 (high, before P5) — the naming law's exact scope**: which names are
  contract (inputs, fold params, derived suffixes) vs internal; the
  deterministic scheme for anonymous temporaries; pinned by the
  rebuild-stability test.
- **Q6 (high, P6 review) — stride units**: element-unit semantic layouts
  with byte elaboration at the boundary (the plan's recommendation) vs
  bytes-in-representation with descriptor-supplied itemsize everywhere.
- **Q7 (med, before the P4 merge) — the at-kink partition law**: first-wins
  ties as table law, replacing full-cotangent-to-every-tie; the alternative
  (a permanent hand adjoint contradicting the derivation machine) is
  recorded as rejected-by-default.
- **Q8 (med, declared in P6) — precision edges**: inf/nan decode stance and
  `round_to`'s AD policy (straight-through vs zero).
- **Q9 (med, before the first lockstep release) — the `pdum.tl` dist
  name** (`habemus-papadum-tl` proposed).
- **Q10 (med, P8 registry) — stream/overlap semantics**: named at
  transformation-step level, designed nowhere; no `stream=` in any sample
  until designed (L4/L5 ownership).
- **Q11 (med, P8 registry) — device-resident persistent state / buffer
  donation**: the first real L2 requirement; the epoch/ownership handshake
  for adopted device buffers rides with it.
- **Q12 (low, P8 registry) — Program normalization**: schedule it in the
  L4 runway or the content-addressed tiers stay honest private caches.

---

## 8. The runway (the L4/L2 brief, self-contained)

**L4 — the kernel language.** A kernel is an erasure-preserving grouping
annotation in the Program; the authored descent lives as the value of a
certified-lowerings registry entry, keyed per §1.4. Tiling is split+bind
one tree level down; the three genuinely new things at L4 are predicates
and decisions, not representation: the capacity WF predicate (dtype-exact,
from descriptors + lowering annotations — never a byte-blind convention),
ordering/race-freedom (tokens owned by the checker even if implicit at the
surface), and materialization-boundary placement. Legality = the
equivalence chain (a sequence of named certified rewrites — split, bind,
reorder-under-license, fuse-as-elision, pad-with-guards, plus the
overlapped-split/halo-recompute class the stencil flagship needs) AND the
per-level WF certificate checked on the result. The objective: minimize
parent-memory traffic under child capacity; the pipeline is a
descend-and-revisit loop with declared invalidation edges (fusion
invalidates checkpoint and traffic plans; placement invalidates partition
candidates); combine-introducing rewrites precede `grad`,
split/bind/place commute with it; the naive→flash move is a registered
named rewrite whose license is the declared combine. Flagships: tiled
GEMM, flash attention, and the fused stencil chain as the non-contraction
acceptance test; flagship gates pin adversarial input families (−inf
masks, cancellation, non-divisible tails), never random draws alone.
Assurance tier and input-domain coverage are recorded fields of every
registry entry; the cross-model reuse payoff attaches only once Program
normalization exists (Q12). Open inside the brief: the mma
selection question (pattern-match vs stated annotation), stream semantics
(Q10), warp vocabulary and external-oracle fixtures (recorded deferrals),
and Q4 as the start precondition.

**L2 — bufferization.** Inputs already in the tree: the exact alias theory
(footprint/overlaps/injectivity), writes-through-views gated on
injectivity, materialize-elision when nesting holds. Ordering: after
fusion decisions — bufferization consumes kernel boundaries. The blocker
list: value numbering for recompute duplicates (name ≠ value); chart-
denominator normalization for codegen; interior-encoding assignment from
§4 (interior shadow layouts are relative-nesting-only; L2 re-derives byte
layouts from assigned encodings); the ring/window boundary sample with
both instances (KV-decode and the audio delay line) and its erasure
obligation (the same surface program, bufferized, reproduces the
row-write); device-resident persistent state and the epoch/ownership
handshake (Q11); and the store effect-ordering mechanism from P7, which
bufferization consumes directly.

**What the whole system is, on completion.** One frontend machine with six
vocabularies over it; one AD with one derivative table and derivation-
under-cache; one cache mechanism with three keyspaces and an honest
statement of what is private until normalization lands; one refusal voice
with a joint battery; one precision doctrine — facts at the boundary,
choices at lowering, carriers throughout; two reference executors as
permanent oracles; and a zoo whose denotations gated every step here and
gate every step from here.

---

## Appendix A — history (the only backward-looking section)

This document supersedes: **160** (the prior integration plan — its
archive/shim/fixture posture is replaced by the purge; its owner-notes
drove this rework), the **150** direction memo (installments absorbed or
dissolved; the L4/L2 brief restated in §8 so 150 can retire), **130**
stages 2–3 (tensor dialect and second AD — cancelled), the **100/110**
array/transform canon (the stdlib it described is deleted), and the
**090 §3** coords-as-params profile (replaced by the ambient intrinsic)
plus its §5 async path (dropped). The at-kink re-pin (Q7) deliberately
changes a previously pinned tensorlib behavior. The precision doctrine
(§4) follows a mid-planning retraction: an interim carrier-only-surface
stance was withdrawn when the owner observed that loaded weights dictate
encodings — precision facts needed somewhere to live, and the boundary is
where they live. The `out=` channel, per-element dispatch as a production
path, `over`/`jvp`/`matmul`/`Named` as pdum concepts, `Build`, the mdsl
tracer, and the demo backends are all deliberately absent from this
document's system: they are described in `docs/design/history/` and in
git history, which is the archive. Design docs 010–160 move to
`docs/design/history/` at P1; the pre-integration tensorlib snapshot is
the `explorations/tensorlib` tree before P2, findable by tag.
