# 170 — The integration: one workspace, two packages

**Status: RATIFIED SPEC.** This document supersedes 160 (and with it the 150
direction memo, 130 stages 2–3, and the array/transform canon of 100/110).
It is written in the present tense: it describes the system being built.
History lives in Appendix A and in git; nothing else in this document looks
backward.

---

## 1. The system

### 1.1 One workspace, two published packages

The repo is a uv workspace. The root is an unpublished development shell
(docs, scripts, workspace glue). Two members are published to PyPI in
lockstep:

- **`packages/dsl` → `pdum.dsl`** (dist `habemus-papadum-dsl`). The
  compiler-infrastructure package: reflection capture and name-fate analysis,
  `typeof`/ValueKind/KindTable type dispatch, the two-tier type-keyed cache
  (specialization + content-addressed artifacts), the Node/Region IR with the
  rewrite driver and lowering machinery, marshaling (PackPlan/ResultPlan),
  the registry with its five extension surfaces, the events seam, and:
  - **`pdum.dsl.value`** — the value language: the statement syntax for
    device functions (if/for, strict joins, single tail return), **value
    types** (the `is_bits` class: scalars, records, nested records, and
    their methods — anything stack-allocatable), the scalar intrinsics, and
    the straight-line/control-flow analysis used as a well-formedness
    predicate by every branchless tier.
  - **`pdum.dsl.render`** — the shared dominator-walking emitter that any
    backend renderer builds on.
  - **The reference evaluator** — a minimal, clean pure-Python executor for
    kernels and device functions. This is the value tier's oracle: the twin
    every future device backend is differential-tested against. It is
    deliberately small and deliberately slow.
  - **The fuse pipe** — `|` composition of device functions
    (`twill(4,3) | weave | zoom(...)`): stages, roles, and
    fusion-by-inlining, with the role/rule registry living on the Registry
    (never module globals). `|` means fuse-inline and nothing else.
- **`packages/tensorlib` → `pdum.tl`** (dist `habemus-papadum-tl`). The
  assemblage language and representation: the layout algebra (affine map +
  box + guards + charts + units + placement), carriers, the four compute
  primitives (pointwise/reduce/scan/fold), the Program/Instr IR,
  reverse-mode AD with derived adjoints, the transforms
  (requested-gradients DCE, min-cut checkpointing, revolve), the cost
  semantics (opcount, peak memory, traffic), placement, signatures — built
  on `pdum.dsl`'s caching, naming, and capture. **`pdum.tl.zoo`** ships
  inside it, and `ir.run` is the tensor tier's reference executor and
  oracle. Tensorlib's design docs (LEVELS, PHILOSOPHY, DESIGN, CONCERNS,
  COMPUTE, REPRESENTATIONS, PLACEMENT) live with the package, revised where
  this document changes them.

There are **no device backends in the tree**. Device backends (WebGPU, CUDA,
Metal) are built fresh in the L4 era against the distilled backend notes
(§3.3) and the two reference executors. The reference executors are the
exception to the purge: oracles are not backends.

**Oracle execution is always spelled.** Reference execution is invoked by
name — `reference(f)(...)` (and `ir.run` at the tensor tier) — never by a
plain call silently interpreting. A plain call on a kind with no routed
backend refuses; it does not degrade to interpretation. This keeps the
oracle path from ever masquerading as a production path.

### 1.2 The syntax stack

Six tiers, all vocabularies + well-formedness predicates over one frontend
machine (capture → typeof → lower), never separate grammars:

1. **Assemblage** — tensorlib's language: straight-line tensor programs over
   the four primitives and layout ops; host Python provides all control flow
   and composition. One optional entry-point annotation (`@assemblage`)
   provides lazy build-on-first-call with the two-tier cache; without it,
   call sites hand-manage building.
2. **The shared expression syntax** — ONE Python-expression form for value
   functions (pointwise marker bodies, reduction combines) AND straight-line
   tensor fold steps. Type-directed lifting: value-typed operands lower to
   the value core; tensor-typed operands lift arithmetic pointwise, plus
   view methods (shift/slice/pad/rename/with_charts/repeat/bind), reduce,
   and the two-operand reduce form. Expressions range over the full value-
   type class — records, nested records, and their methods — not only
   scalars. The invariant is straight-line/no-branching (enforced by
   `pdum.dsl` control-flow detection), not pointwise-only.
3. **`@compute` kernels** — coordinates via the `thread_idx(...)` ambient
   intrinsic; **explicit stores into writable argument buffers**
   (`img[y, x] = f(y, x)`); launch configuration at invocation only;
   **function-valued arguments** that inline at specialization with the
   argument's FnType (including capture types) in the cache key. Buffers may
   be **tensors of value types**: a tensor-of-structs element loads and
   stores as a record value; nesting and field padding are encoding facts
   (§4), invisible to the kernel body; record methods participate in the
   expression syntax.
4. **Tile DSL** (L4 authoring) — stage/barrier/accum vocabulary; capacity,
   race, and convexity certificates checked on the elaborated result.
5. **Warp DSL** — straight-line, uniform-control, lane-complete;
   shuffle/ballot/mma-fragment intrinsics.
6. **Vendor escapes + external oracles** — vendor-namespace ops at visible
   portability cost; raw CUDA C / Numba kernels as test fixtures only. There
   is no CUDA-clone language.

Standing invariants: branchless at the top (assemblage) and bottom (warp),
control flow confined to value-language kernels and the host; ambient
coordinates ARE the launch-domain iotas, and the iota→thread_idx descent
never materializes them; invocation concerns (blocks, shared memory,
streams, pipelining) never appear in user programs — they become visible
exactly where a transformation step introduces them (§5 S.6).

**There is no `out=` invocation channel.** Kernels write through explicit
stores into writable arguments; foreign buffers enter through boundary
descriptors (§4); allocation conveniences can return later as runtime sugar.
Two principles survive as law: the launch domain never enters cache
identity, and a writable argument that overlaps a readable capture/argument
refuses at dispatch with the ping-pong message.

### 1.3 The calling-convention matrix

Rows = caller, columns = callee. **I** inline, **L** launch, **C** compose,
**R** refuse. Reference execution appears only through its spelled form.

| caller ↓ \ callee → | Host | Device fn | @compute | Vertex/Fragment | Tile | Warp | Assemblage |
|---|---|---|---|---|---|---|---|
| **Host Python** | — | L, spelled `reference(f)` only | L: launch config at invocation; overlap refusal; fn-valued args FnType-keyed | C(PSO)→encode into a foreign pass | L | R | L(many); optional `@assemblage` |
| **Device fn** | R | I | I(body), kind-checked | R | R | R | R |
| **@compute** | R | I (incl. passed-in fns) | I(body), kind-checked | R | R | I (lane intrinsics) | R |
| **Vertex/Fragment** | R | I | R | C (varyings record: vertex→fragment via PSO, not a call) | R | R | R |
| **Tile** | R | I (epilogues) | I(body) | R | I | I (mma fragments) | R |
| **Warp** | R | I (value-typed only) | R | R | R | I | R |
| **Assemblage** | — | C (marker: declared, never a callback) | L(select) via certified lowerings | L(encode) | L(select) | R | C: fold takes a Program |

Invariants: launches from non-host callers refuse (no dynamic parallelism —
the launch boundary is host-only and explicit); `kind` is a validated
vocabulary, checked at dispatch and at cross-family inline (a body using the
`thread_idx` ambient inlined into a non-kernel context refuses); the
composition semantics never share syntax (`|` fuse-inline; sequencing = host
Python + fold; vertex→fragment pairing is PSO composition; rewrite chains
have their own form).

### 1.4 Caching, identity, and names

One mechanism (`pdum.dsl` cache), three keyspaces:

- **Assemblage tier.** Building a Program is the compile step. Tier 1 keys
  `(fp_head, arg_fp, generation)`; tier 2 is content-addressed on the built
  Program and supports **derivation-under-cache**: partials, component
  markers, and adjoint scanners are cache entries computed on demand from
  cache entries.
- **Kernel tier.** Types and identity in the key, never values; live-knob
  captures ride the uniform channel at zero recompiles; function-valued
  arguments put the argument Handle's FnType in `arg_fp` — a different
  pipeline shape is a new artifact, the same shape with different captured
  values is a warm hit plus a uniform rewrite.
- **Descent tier.** `chunk_fp` over the named-op Program in canonical form;
  registry key = (normalized chunk skeleton, boundary contract incl.
  saved-set demand and layout classes, license set, capability set,
  rules-generation); value = chain + authored region + artifact + assurance
  tier; the chain is a mandatory content-addressed output of elaboration;
  trust attaches at assurance tier ≥ 2. Until the Program-normalization
  pass lands (scheduled in the L4 runway, §7), the content-addressed tiers
  are declared **private caches**.

**Structural values are `Literal`-typed — three doors, one mechanism.**
Specialization is influenced by (i) the value's own type, (ii) call-site
wrappers (`Literal(5)`), and (iii) **definition-site annotations**
(`def make_step(n: Literal[int], ...)`). An annotation is a local coercion
applied at binding time: the binder promotes the value's type before
fingerprinting. No unification, no solver — the table-of-rules methodology
is untouched. Identity stays type-keyed at every tier. The enforcement:
a **structural slot** (slice/pad extents, fold counts, dim counts) accepts
only `Literal`-typed values; a plain int reaching one refuses with a
designed message naming the annotation fix. Wrong-Program reuse from
structural captures is thereby impossible, with no value-keyed special
cases anywhere.

**The naming law.** Programs identify tensors by string names; `grad`
returns name-keyed maps; tests and user code index by name. Names divide
into **contract** and **internal**. Contract names come only from declared
things: function parameter names, explicit composition-site prefixes
(`"L0."`), declared fold-state fields, and derived suffixes (`name.d{i}`,
`.rc`). Internal names (intermediates) derive from Python binding names as
debugging niceties only — they are excluded from content-addressing, so
renaming a local variable never changes program identity or the public
ABI. Anonymous temporaries get a deterministic scheme; a rebuilt closure
maps the same capture to the same name, pinned by a rebuild-stability
test.

### 1.5 Extensions, punning, events

Vendor op namespaces spelled by one backend; capability flags checked at
build; `code_for_op` key-presence as the capability bit; capability-gated
`debug.print`; `record.artifact` as the escape hatch; rule-of-three before
any runtime abstraction. The adopt descriptor is where dictated encodings
live (§4) — buffer interop and precision facts are one concept. Foreign
tensor libraries are interop partners; `pdum.tl` is in-house. The events
seam lives in `pdum.dsl` and has emission points at `pdum.tl`'s
compile-ish seams (Program build, adjoint derivation, descent
certification), so `forbid`/`no_compile` can pin "this training loop builds
zero Programs" exactly as they pin kernel compiles.

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
lockstep** by the release workflow:

- One version source of truth: `packages/dsl/src/pdum/dsl/__init__.py`.
  `scripts/_versioning.py` anchors there and enrolls every
  `packages/*/pyproject.toml` (plus the root) as lockstep version files —
  bumps apply across all packages in unison, never individually.
- `release.yml` builds and publishes **all** workspace members' dists in one
  run, tagged once.
- Dist names follow the namespace: `pdum.<name>` publishes as
  `habemus-papadum-<name>` — `habemus-papadum-dsl`, `habemus-papadum-tl`,
  and every future member likewise.
- The root pyproject is an unpublished virtual root: workspace glue, dev
  dependency groups (members via `[tool.uv.sources] workspace = true`),
  docs tooling. Nothing installs the root from PyPI.
- Release runs remain deliberate acts through the workflow trigger; nothing
  in the migration publishes as a side effect.

## 3. The purge

Git history is the archive. The living tree carries only the go-forward
system. Three rules govern every deletion: **distill before deleting** (the
load-bearing knowledge lands in canon or tests first), **pin contracts with
literal expectations** (never by dual-running dead code; where a comparison
fixture is genuinely needed during a conversion it is temporary
scaffolding, deleted when its gate retires), and **new code never
references old code**.

### 3.1 Deleted from the pdum.dsl side

`dsl_reference/`; `stdlib/arrays.py` and `stdlib/transforms.py` (Named,
over, jvp-as-pdum-concept, matmul — the tensor tier owns all of it; the
tangent-engine rows merge into the one derivative table first, §5 S.2); the
array half of batteries; `backends/c.py` and the whole demo tree (python
renderer, wgsl runtime, graphics batteries — the fwidth residue is
re-created at the shader tier in P8); `combinators.py` (the fuse pipe is
re-authored clean in `pdum.dsl`); `viz.py` and `bench.py` (rebuilt against
the new surfaces when needed); the book (`docs/book/`, `scripts/book/`) —
the new packages grow their own documentation; the tests of every deleted
module.

### 3.2 Deleted from the tensorlib side

`build.py` (the Build name-manager — replaced by core naming + the shared
syntax); mdsl's Sym tracer, `defmarker`/`defreducer` entry points, process
registries and `node_digest` (replaced by the AST producer and cache-backed
registries). **Surviving mdsl parts relocate:** the `Arg/Const/Prim` Node
schema (the declared stability boundary — the AST frontend is a new
*producer* for it; its consumers never change), the symbolic `diff`/`_D`
machinery, and the CompositeReducer BPTT engine. The `defreducer`-shaped
*declaration API* (state/element/lift/combine/init/project + declared
associativity) survives with producer-swapped bodies — a reducer is a
structured declaration, not one expression.

### 3.3 Distillations (written in the same commits as their deletions)

- **Backend notes** (one short doc): the numeric policy (truncating integer
  div/mod with exact twins; float mod = fmod; u64 constant refusals),
  artifact-carries-its-contract, the WebGPU runtime learnings (synchronous
  readback is a fixed-latency protocol act; timestamp-query timing;
  uniform-plan/bind-group layout; encode and submit are separate acts and
  the encodable is the API), and the bench/instrumentation methodology
  (warmup, tuned evals, minimum as estimator; phase decomposition by
  seam-wrapping).
- **The aliasing lesson**: writable/readable overlap is silent corruption;
  it is a day-one refusal test at the `@compute` store seam (P7), not a
  memory.
- **The refusal voice**: one shape — what happened, the principle violated,
  the quoted fix, the source location — seeded as the joint refusal battery
  (P3) that every later refusal extends.
- **The oracle status rule**: per-element host dispatch of kernels is
  debug/oracle-grade; the reference executors are the only per-element
  consumers, and reference execution is always spelled (§1.1).

### 3.4 Docs and notebooks

`docs/design/010–160` move to `docs/design/history/` in the purge commit;
mkdocs nav is rebuilt around this document and the two packages' docs.
Everything still load-bearing from prior canon is restated in §7 (the
runway), so nothing in history needs to be read to proceed. Tensorlib
notebooks 00–06 teach surviving semantics and move with the package; 07–13
are re-authored or dropped as their APIs change during P4–P6 — they stay
out of published nav until revised. Fixes that survive regardless of the
purge land at extraction: the unknown-kind dispatch refusal and the
record-value designed refusals (both in surviving core modules).

---

## 4. Precision and boundaries

**Facts at the boundary, choices in the interior, carrier semantics
throughout.**

**The contract.** Semantics are carrier-valued end to end
(bool/int/rat/real/complex). No compute dtype exists on the user surface.
Dtype is a property of buffers and encodings **at the boundary**, recorded
in load/adopt/writable-argument descriptors, never on tensors
mid-computation. **Exact decode:** every finite bit pattern is a specific
rational (int4+scale decodes to `scale[g]·q`), so the denotation stays
exact over exactly-known inputs — a bf16 checkpoint is a *fact* the
descriptor records, not a semantic property of the program. **Explicit
rounding:** where rounding IS the semantics (QAT, stochastic rounding), it
is the explicit exact op `round_to(encoding)`; its AD rule is
**straight-through by default** (the QAT convention — a zero derivative
would make every quantized parameter untrainable), with zero available by
declaration. **The discipline:** every precision appearance is a boundary
fact, a descent choice, or an explicit `round_to`; mid-program
astype-as-semantics does not exist and the IR has no op for it. Edge
rules: inf/nan bit patterns **refuse at decode** (an extended-real carrier
is a recorded future opt-in); writing real-carrier results into an encoded
writable argument rounds as part of the boundary contract
(`denotation = encode_out ∘ f ∘ decode_in`) — the one implicit rounding,
declared, never silent.

**Buffers.** A Buffer is a thin shim over a rank-1 DLPack-style handle:
data pointer, **explicit device**, length. Our Layout addresses it; an
Encoding interprets it. Host buffers are the degenerate case; device
buffers carry the same shape (zero-copy interop in both directions; reads
are valid only where the host can reach). Device-resident persistent state
and the epoch/ownership handshake for adopted device buffers are L2's
first requirement and build on this same handle.

**The boundary descriptor** = Buffer + Layout + Encoding (+ carrier +
units). `Encoding` is a small hierarchy — NumpyEncoding (including
**structured encodings**: field names, offsets, padding — the memory shape
of tensors-of-structs), QuantGroupEncoding (int4 nibbles + per-group
scales over two buffer regions), FormatEncoding (e.g. bgra8unorm-srgb with
the transfer curve in decode) — each declaring its exact decode/encode.
Flows: checkpoint load (file metadata → per-weight descriptors), foreign
adoption (an audio callback's f32 buffer is a descriptor fact), writable
arguments (descriptor + writable flag + the overlap refusal). The
*logical* record type (field names and types) is the interior value type;
offsets, padding, and alignment are encoding facts the interior never
sees. Interior program values carry carrier + units + layout shadows only;
the IR cannot mint encoding-bearing values (enforced at the IR/signature
layer). The reference executors' float64 interior is a declared oracle
property, never semantics.

**Strides are bytes.** Layouts address bytes everywhere, unchanged — this
is what makes `field()` on structured dtypes a free view (offset bump +
dtype change, padding skipped) and keeps the whole affine/guard algebra
integer. The reconciliation with the precision doctrine is a rule about
meaning, not units: **interior shadow layouts are structural only** —
nesting and aliasing information, never byte-authoritative. Byte truth
enters exactly twice: at boundary descriptors, and at L2's encoding
assignment, from which bufferization re-derives interior byte layouts.
Sub-byte encodings (nibbles) are the Encoding's decode concern over byte
regions, never the affine map's.

**The interior.** Precision enters at exactly three lowering points: (1)
**descent licenses** — taxonomy {none, reassociation, precision-demotion},
equivalence stated over the carrier denotation, tolerances and input
domain in the declaration, the license set in the registry key; the
numeric tier monitors divergence and never certifies it; (2) **L2 storage
assignment** — materialized intermediates get encodings chosen at
bufferization; (3) **machine-tree byte predicates** — consumers, not
choosers; capacity checks read lowering annotations plus boundary facts.
The worked check: weights bf16 and activations f32 as facts, a
real-carrier contraction as the program, f16 tiles + f32 accumulators
under a license at descent, the writable argument's descriptor encoding
the result — and byte-blind capacity mistakes are impossible by
construction.

**Fallback, with criterion.** The two-surface model (a family
element-dtype parameter + the descent license) is the recorded fallback.
Before L4, the mixed-precision/QAT sample (master vs bf16 weights, loss
scaling) is written in boundary-facts terms; fall back iff a required
program's *meaning* — not cost — depends on an interior encoding that is
neither a boundary fact nor expressible as `round_to`. Falling back is a
written owner decision, never a silent switch.

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
explicit. **Structural values are Literal-typed** (§1.4): extents entering
slice/pad/fold slots come from `x.extent(d)`, from `Literal(...)` at the
call site, or from a `Literal[int]`-annotated parameter; a plain int in a
structural slot refuses, naming the annotation fix. **Vocabulary
completeness:** the committed method set includes `.bind(level=...)`,
`iota_of(t, dim)`, and the two-operand reduce form; `Program`/`Instr`
remain public hand-constructible data, and hand-emits get names through
the same core naming contract. **Entry point:** `@assemblage` [build] —
exactly a pdum Handle (phase-A capture, two-tier cache, build at first
call → Program + input layouts).

### S.2 The shared expression syntax

Type-directed lifting; straight-line enforced at lowering; bounded `if`/
`for` exist only in the value language, never over tensor-typed values.
Expressions range over value types: records construct, destructure, nest,
and carry methods; a method is a device function with `self` first, usable
over value-typed and tensor-typed receivers alike.

One definition, two consumers [build — the gate is a differential]:

```python
def gelu(x):
    return 0.5 * x * (1 + tanh(GELU_C * (x + 0.044715 * x*x*x)))
```

— lowered as a pointwise marker body under `ir.run` AND inlined as a
device function into a `@compute` kernel; the two paths must agree
numerically.

**Marker-body granularity is a hard gate.** A marker (gelu's formula) and
a reducer's combine are small named bodies the AD machinery differentiates
*by inspection* — derived partials by tree rewriting; flash attention's
backward exists because the combine is such a body. The AST producer must
therefore lower one marker to one named, inspectable body tree over
primitives (captured constants become Consts), and one combine to the same
inside its structured declaration — never inlined away into the calling
program, never an opaque call. The flash derived-backward test enforces
this; if it fails, the frontend is wrong, not the gate.

A reduction combine with record state [build]:

```python
def flashsm_combine(L, R):                     # State = (m, den, o) — a record
    m = maximum(L.m, R.m)
    sl, sr = exp(L.m - m), exp(R.m - m)
    return State(m, L.den*sl + R.den*sr, L.o*sl + R.o*sr)
```

A tensor-typed fold step [build]:

```python
def fdtd_step(E, H, n: Literal[int]):
    dE = (E.shift(x=-1).slice(x=(0, n-1)) - E.slice(x=(0, n-1))).with_charts(x=h_chart)
    H1 = H + c * dE                            # c lifts, inheriting dims AND charts
    dH = (H1.slice(x=(1, n-1)) - H1.shift(x=1).slice(x=(1, n-1))).with_charts(x=e_chart)
    E1 = E + c * dH.pad(x=(0, n), fill=0.0)
    return E1, H1                              # carry; layout preserved — checked
```

Step input names = param names per the fold contract; the chart-inheriting
lift removes rechart boilerplate; nesting is just a function; the
`Literal[int]` annotation puts the structural extent in the type (§1.4).

**The one derivative table.** The value-tier tangent rules and the marker
partial rules are the same object — op → linearization, None =
gradient-free — one table in the core transform column. The table grows
only when a primitive joins the core; everything else derives:
`CompositeMarker.partial(i)` is forward-tangent application over the
lowered body (basis seed, DCE, registered `name.d{i}`) — derivation-under-
cache; the reducer BPTT engine consumes partials through the same
interface. **At-kink law:** the table is one-sided and partitions — at a
tie, exactly one operand receives the cotangent (first-wins), and reduce
adjoints derive through the pairwise combine, inheriting the partition
law. `jvp` returns a fixed subgradient selection at kinks; this is frozen
contract, pinned at the kink points.

### S.3 @compute kernels [build]

```python
@compute
def my_shader(f, img):
    y, x = thread_idx("y", "x")                # ambient intrinsic, NOT positional params
    img[y, x] = f(y, x)                        # explicit store into a WRITABLE ARGUMENT

f = twill(4, 3) | weave | zoom(center=(20, 50), r=20, scale=5)
my_shader(f, img, launch=grid(blocks=ceil_div(img.shape, 16), threads=(16, 16)))
```

- **Function-valued arguments**: same FnType → warm hit, captured values
  ride the uniform channel; swapping a stage → new artifact. The lowering:
  inlining through an FnType-typed parameter, arg-rooted ABI slots, the
  Handle value passed to the build alongside arg types; the guard policy
  for argument Handles is recorded in P7's design (captures freeze at
  construction, fingerprints recompute per call).
- **Stores and ordering**: the IR represents ordering as **token
  threading** — a store consumes and produces an ordering token, so
  ordering is ordinary dataflow and tile barriers and L2 bufferization
  consume the same mechanism later. The **frontend policy** is program
  order: one implicit token threads through all stores in statement order,
  giving exactly the semantics the Python reads; tokens never appear in
  user syntax. Finer policies (per-buffer tokens, barrier tokens) refine
  the threading without changing the IR. Day-one contract: a writable
  argument overlapping any readable capture/argument refuses with the
  ping-pong message; in-place returns only ever as an L2-certified
  rewrite.
- **Tensors of value types**: `img[y, x]` on a struct-element buffer loads
  a record; stores accept a record; the element's memory shape is the
  descriptor's structured encoding (§4).
- **Launch config**: invocation-only, rides the launcher, never any key.
  Threads-per-block is a value-specialized bracket (re-render on change,
  no identity change); blocks/streams are pure launcher data.
- **Iota unification**: the same kernel is expressible as pointwise over
  coordinate iotas; the iota→thread_idx descent is a rewrite stage whose
  WF predicate is "no iota reaches a materialization boundary"; the fused
  and assemblage forms are differential-tested against `ir.run`.

### S.4 Vertex/fragment [build — P8]

`@vertex`/`@fragment` share the ambient contract. **Varyings are a
record**: the vertex kernel returns it, the fragment kernel receives it —
the value-type system is the interface machinery, and the
interpolation contract is declared per field. `fwidth` is the wrt-ambient
derivative at this tier. Vertex→fragment pairing is PSO composition (its
own semantics, never `|`); the per-frame deliverable is an **encodable**
(render bundle / draw-into-pass) — the host owns the pass, the submit, and
the swap chain. FnType carries an optional result-type slot (reserved at
extraction; the varyings interface check needs it, and fold-step
diagnostics use it immediately). Semantics land against golden artifacts
plus a minimal reference interpolator; the GPU rasterization path arrives
with the L4-era backends.

### S.5 Tile and warp [reserved — the L4 brief governs, §7]

The authored descent is not IR: it is the value of a certified-lowerings
registry entry; the kernel boundary in the Program is an erasure-
preserving annotation. Tile vocabulary: stage/barrier/accum, tile loops as
split+bind one level down; WF certificates checked on the result —
capacity (byte-exact, from descriptors and annotations), race-freedom
(checker-owned tokens — the same tokens as §S.3), convexity. Whether `mma`
pattern-matches mul→reduce or requires a stated annotation is answered
inside the L4 design. Warp: straight-line post-unroll, uniform control,
lane-complete. Below both: vendor punning and external oracle fixtures.

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
deleted with them — green means the go-forward system.

**P0 — Workspace + release machinery (M).** `packages/dsl` and
`packages/tensorlib` skeletons; root becomes the unpublished virtual root;
`_versioning.py` re-anchored with lockstep enrollment; `release.yml`
reshaped to build/publish all members in lockstep; CI reshaped around the
surviving suites; loc-budget buckets redrawn (the tripwire discipline
continues). GATE: CI green; `_versioning.py current` passes; a local
`uv build` of both members succeeds; no publish.

**P1 — The purge + the core lands (L).** git-mv the kernel engine into
`packages/dsl` (with `pdum.dsl.value`, `pdum.dsl.render`); write the
reference evaluator; land the extraction-time fixes (unknown-kind dispatch
refusal; record-value designed refusals; the FnType result-type slot);
re-author the fuse pipe on the Registry. Delete everything in §3.1 plus
`docs/book`; move design docs 010–160 to `history/`; write the §3.3
distillation notes in the same commits. GATE: D green (the value-language
battery with literal expectations, on the reference evaluator); B green;
the tree contains no import of anything deleted.

**P2 — Tensorlib promoted (M).** git-mv `explorations/tensorlib/tensorlib`
→ `packages/tensorlib/src/pdum/tl` (zoo inside), tests to the member,
conftest path-hack gone, notebooks per §3.4. The zoo gate enters CI — live
from here on. GATE: T + Z green at the new paths; `import pdum.tl`.

**P3 — Onto the core (L).** Process registries die: cache-backed
registries with derivation-under-cache; core naming replaces Build's hint
dedup (Build itself alive for the zoo builders until P5); the Node schema
relocates to its stability-boundary module; events emission at
Program-build/adjoint seams; the joint refusal battery seeded with the
extent refusal pinned. GATE: T + Z green; idempotence pin (re-registering
an identical marker yields one entry).

**P4 — The shared expression syntax (L).** The AST producer replaces the
Sym tracer; straight-line detection replaces trace-time refusal; the
structured reducer declaration keeps its API with producer-swapped bodies;
record-typed reducer state; **value-type expansion** (nested records and
methods in the expression syntax); tensor-typed lifting lowers fold steps
to step Programs; the `Literal` annotation door and the structural-slot
refusal; the merged derivative table with the first-wins at-kink re-pin
(its own commit, kink points pinned). The tracer survives this step only
as scaffolding for the producer-equivalence check and is deleted at the
end of it. GATE: Z green with markers/reducers/fold-steps re-authored —
denotation-identical including flash's derived backward and FDTD charts;
the marker-granularity gate (S.2).

**P5 — `@assemblage`; Build dies (M).** Zoo builders re-authored in
S.1/S.2 (bind/iota_of/two-operand-reduce vocabulary included);
`@assemblage` lands; the naming-law literal pins land (gate 7); Build and
the remaining mdsl entry points are deleted. GATE: Z green on the
re-authored builders with hardcoded name expectations; D + T green.

**P6 — Precision and boundaries (M).** The §4 design lands before any L4
work: dtypes split (carriers semantic; the Encoding hierarchy including
structured encodings); the DLPack-shim Buffer with explicit device;
descriptors; constructors re-roled as boundary acts;
IR-cannot-mint-encodings enforced; machinery dtype sites converted;
`round_to` with straight-through AD; inf/nan decode refusals;
descriptor-fed byte-exact sizes replace the 8-byte convention (cost
oracles re-derived as their own reviewable diff); the QAT sample written
and evaluated against the fallback criterion. GATE: Z green with zero
denotation changes; the re-derived cost diff reviewed.

**P7 — `@compute` (L).** In order: token-threading store representation
with the program-order frontend policy → store lowering with the day-one
overlap refusal → `thread_idx` + the iota→thread_idx descent stage →
tensors-of-value-types (structured-encoding loads/stores) →
function-valued-argument lowering (guard policy recorded) → launch config.
Execution on the reference evaluator; device backends remain L4-era work.
GATE: the S.3 example runs on the reference evaluator; the
iota-unification differential; the two-consumers differential (S.2);
key-discipline pins (shape miss / value hit / launch never keys / fn-swap
miss); overlap refusals; the compile-once thesis test for function-valued
arguments; a struct-element kernel round-trips through a structured
encoding.

**P8 — Graphics (M).** The `@vertex`/`@fragment` kinds in the validated
vocabulary; varyings-as-records with per-field interpolation declarations;
PSO pairing as its own composition semantics; the encodable deliverable
(render bundle / draw-into-pass); `fwidth` as the wrt-ambient derivative.
Semantics against golden artifacts + the minimal reference interpolator.
GATE: a vertex+fragment pair lowers, pairs, and encodes; the varyings
record round-trips; goldens pinned; D + Z green.

**P9 — Runway handoff (S).** The tiled-matmul zoo entry; the
license-schema stub + the worked GEMM license declaration; the L2 blocker
list; the open registry (streams, device-resident state, normalization,
warp vocabulary, external-oracle fixtures, the operators door, adversarial
input families — one −inf-mask attention case seeded into Z now). GATE: Z
green including tiled-matmul; §7 handed to the L4/L2 work as its brief.

**Not doing:** async, in any form. A CUDA-clone language. A pdum tensor
dialect, translation frontend, or second AD. `out=`. Archives, shims,
frozen-fixture museums, or dual-running dead code beyond within-step
scaffolding. Device backends before the L4 era. Publishing as a side
effect of migration.

---

## 7. The runway (the L4/L2 brief, self-contained)

**L4 — the kernel language.** A kernel is an erasure-preserving grouping
annotation in the Program; the authored descent lives as the value of a
certified-lowerings registry entry, keyed per §1.4. Tiling is split+bind
one tree level down; the three genuinely new things at L4 are predicates
and decisions, not representation: the capacity WF predicate (byte-exact,
from descriptors + lowering annotations), ordering/race-freedom (the
token mechanism of §S.3, owned by the checker), and materialization-
boundary placement. Legality = the equivalence chain (a sequence of named
certified rewrites — split, bind, reorder-under-license, fuse-as-elision,
pad-with-guards, plus the overlapped-split/halo-recompute class the
stencil flagship needs) AND the per-level WF certificate checked on the
result. The objective: minimize parent-memory traffic under child
capacity; the pipeline is a descend-and-revisit loop with declared
invalidation edges (fusion invalidates checkpoint and traffic plans;
placement invalidates partition candidates); combine-introducing rewrites
precede `grad`, split/bind/place commute with it; the naive→flash move is
a registered named rewrite whose license is the declared combine.
Flagships: tiled GEMM, flash attention, and the fused stencil chain as
the non-contraction acceptance test; flagship gates pin adversarial input
families (−inf masks, cancellation, non-divisible tails), never random
draws alone. Assurance tier and input-domain coverage are recorded fields
of every registry entry. Scheduled inside the L4 runway: the
**Program-normalization pass** (the precondition of the registry's
cross-model payoff; until it lands the content-addressed tiers stay
declared-private), **stream/overlap semantics** (no `stream=` appears in
any sample until this design exists), the warp vocabulary, the
external-oracle fixtures, the per-type operator-extension door (geometric
algebra's registration surface), and the `mma` selection question
(pattern-match vs stated annotation). Device backends are built here,
fresh, against the distilled notes and the reference executors.

**L2 — bufferization.** Inputs already in the tree: the exact alias
theory (footprint/overlaps/injectivity), writes-through-views gated on
injectivity, materialize-elision when nesting holds. Ordering: after
fusion decisions — bufferization consumes kernel boundaries. The blocker
list: value numbering for recompute duplicates (name ≠ value);
chart-denominator normalization for codegen; interior-encoding assignment
(§4 — interior shadow layouts are structural-only; L2 re-derives byte
layouts from assigned encodings); the ring/window boundary sample with
both instances (KV-decode and the audio delay line) and its erasure
obligation (the same surface program, bufferized, reproduces the
row-write); **device-resident persistent state** and the epoch/ownership
handshake for adopted device buffers (on the DLPack-device Buffer, §4);
and the token mechanism from P7, which bufferization consumes directly.

**What the whole system is, on completion.** One frontend machine with
six vocabularies over it; one AD with one derivative table and
derivation-under-cache; one cache mechanism with three keyspaces and an
honest statement of what is private until normalization lands; one
refusal voice with a joint battery; one precision doctrine — facts at the
boundary, choices at lowering, carriers throughout; value types — records
included — flowing from device functions through buffers to varyings; two
reference executors as permanent, always-spelled oracles; and a zoo whose
denotations gate every step.

---

## Appendix A — history

This document supersedes 160 (whose archive/shim posture the purge
replaced), the 150 direction memo (its L4/L2 brief is restated in §7), 130
stages 2–3 (cancelled), the 100/110 array/transform canon (its stdlib is
deleted), 090 §3's coords-as-params profile and §5's async path. Notable
reversals decided along the way: tensorlib's Build/mdsl frontend and
process registries are replaced by `pdum.dsl` infrastructure; the
carrier+dtype-on-Tensor contract retreats to boundary descriptors; the
at-kink adjoint convention changed from full-cotangent-to-every-tie to the
first-wins partition law; an interim carrier-only precision stance was
retracted in favor of boundary-facts when it was observed that loaded
weights dictate encodings. `out=`, `over`, `jvp`-as-pdum-concept,
`matmul`, `Named`, per-element dispatch as a production path, and the
demo backends are deliberately absent from this system. Design docs
010–160 live in `docs/design/history/`; git history is the archive.
