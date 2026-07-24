# 190 — The integration: the final spec

**Status: RATIFIED SPEC, pending final owner review.** This document merges
and supersedes 170 (the clean-break plan) and 180 (the GPT-2 red-team
exercise): the principles 180 produced are stated here as core spec, and
its GPT-2 program survives as the worked example (§6). It is written in
the present tense: it describes the system being built. History lives in
Appendix A and in git; nothing else in this document looks backward.

---

## 1. The system

### 1.1 One workspace, two published packages

The repo is a uv workspace. The root is an unpublished development shell
(docs, scripts, workspace glue). Two members are published to PyPI in
lockstep:

- **`packages/dsl` → `pdum.dsl`** (dist `habemus-papadum-dsl`). The
  compiler-infrastructure package: reflection capture and name-fate
  analysis, `typeof`/ValueKind/KindTable type dispatch, the two-tier
  type-keyed cache (specialization + content-addressed artifacts), the
  Node/Region IR with the rewrite driver and lowering machinery,
  marshaling (PackPlan/ResultPlan), the registry with its five extension
  surfaces, the events seam, and:
  - **`pdum.dsl.value`** — the value language: the statement syntax for
    device functions (if/for, strict joins, single tail return), **value
    types** (the `is_bits` class: scalars, records, nested records, and
    their methods — anything stack-allocatable), the scalar intrinsics,
    and the straight-line/control-flow analysis used as a
    well-formedness predicate by every branchless tier.
  - **`pdum.dsl.render`** — the shared dominator-walking emitter that any
    backend renderer builds on.
  - **The reference evaluator** — a minimal, clean pure-Python executor
    for kernels and device functions: the value tier's oracle, the twin
    every future device backend is differential-tested against.
    Deliberately small, deliberately slow.
  - **The fuse pipe** — `|` composition of device functions
    (`twill(4,3) | weave | zoom(...)`): stages, roles, and
    fusion-by-inlining, with the role/rule registry living on the
    Registry (never module globals). `|` means fuse-inline and nothing
    else.
- **`packages/tensorlib` → `pdum.tl`** (dist `habemus-papadum-tl`). The
  assemblage language and representation: the layout algebra (affine map
  + box + guards + charts + units + placement), carriers, the four
  compute primitives (pointwise/reduce/scan/fold), the Program/Instr IR,
  reverse-mode AD with derived adjoints, the transforms
  (requested-gradients DCE, min-cut checkpointing, revolve), the cost
  semantics (opcount, peak memory, traffic), placement, signatures —
  built on `pdum.dsl`'s caching, naming, and capture. **`pdum.tl.zoo`**
  ships inside it, and `ir.run` is the tensor tier's reference executor
  and oracle. Tensorlib's design docs live with the package, revised
  where this document changes them.

There are **no device backends in the tree**. Device backends (WebGPU,
CUDA, Metal) are built fresh in the L4 era against the distilled backend
notes (§3.3) and the two reference executors. The reference executors are
the exception to the purge: oracles are not backends.

**Oracle execution is always spelled.** Reference execution is invoked by
name — `reference(f)(...)` (and `ir.run` at the tensor tier) — never by a
plain call silently interpreting. A plain call on a kind with no routed
backend refuses; it does not degrade to interpretation.

### 1.2 The syntax stack

Six tiers, all vocabularies + well-formedness predicates over one
frontend machine (capture → typeof → lower), never separate grammars:

1. **Assemblage** — tensorlib's language: straight-line tensor programs
   over the four primitives and layout ops; host Python provides all
   control flow and composition. One optional entry-point annotation
   (`@assemblage`) provides lazy build-on-first-call with the two-tier
   cache; without it, call sites hand-manage building.
2. **The shared expression syntax** — ONE Python-expression form for
   value functions (pointwise marker bodies, reduction combines) AND
   straight-line tensor fold steps. Type-directed lifting: value-typed
   operands lower to the value core; tensor-typed operands lift
   arithmetic pointwise, plus view methods
   (shift/slice/pad/rename/with_charts/repeat/bind), reduce, and the
   two-operand reduce form. Expressions range over the full value-type
   class — records, nested records, and their methods — not only
   scalars. The invariant is straight-line/no-branching, not
   pointwise-only.
3. **`@compute` kernels** — coordinates via the `thread_idx(...)` ambient
   intrinsic; **explicit stores into writable argument buffers**
   (`img[y, x] = f(y, x)`); launch configuration at invocation only;
   **function-valued arguments** that inline at specialization with the
   argument's FnType (including capture types) in the cache key. Buffers
   may be **tensors of value types**: a struct element loads and stores
   as a record value; nesting and field padding are encoding facts (§4),
   invisible to the kernel body.
4. **Tile DSL** (L4 authoring) — stage/barrier/accum vocabulary;
   capacity, race, and convexity certificates checked on the elaborated
   result.
5. **Warp DSL** — straight-line, uniform-control, lane-complete;
   shuffle/ballot/mma-fragment intrinsics.
6. **Vendor escapes + external oracles** — vendor-namespace ops at
   visible portability cost; raw CUDA C / Numba kernels as test fixtures
   only. There is no CUDA-clone language.

Standing invariants: branchless at the top (assemblage) and bottom
(warp), control flow confined to value-language kernels and the host;
ambient coordinates ARE the launch-domain iotas, and the iota→thread_idx
descent never materializes them; invocation concerns (blocks, shared
memory, streams, pipelining) never appear in user programs — they become
visible exactly where a transformation step introduces them (§5 S.6).

**There is no `out=` invocation channel.** Kernels write through explicit
stores into writable arguments; foreign buffers enter through boundary
descriptors (§4); allocation conveniences can return later as runtime
sugar. Two principles survive as law: the launch domain never enters
cache identity, and a writable argument that overlaps a readable
capture/argument refuses at dispatch with the ping-pong message.

### 1.3 The calling-convention matrix

Rows = caller, columns = callee. **I** inline, **L** launch, **C**
compose, **R** refuse. Reference execution appears only through its
spelled form.

| caller ↓ \ callee → | Host | Device fn | @compute | Vertex/Fragment | Tile | Warp | Assemblage |
|---|---|---|---|---|---|---|---|
| **Host Python** | — | L, spelled `reference(f)` only | L: launch config at invocation; overlap refusal; fn-valued args FnType-keyed | C(PSO)→encode into a foreign pass | L | R | L(many); optional `@assemblage` |
| **Device fn** | R | I | I(body), kind-checked | R | R | R | R |
| **@compute** | R | I (incl. passed-in fns) | I(body), kind-checked | R | R | I (lane intrinsics) | R |
| **Vertex/Fragment** | R | I | R | C (varyings record: vertex→fragment via PSO, not a call) | R | R | R |
| **Tile** | R | I (epilogues) | I(body) | R | I | I (mma fragments) | R |
| **Warp** | R | I (value-typed only) | R | R | R | I | R |
| **Assemblage** | — | C (marker: declared, never a callback) | L(select) via certified lowerings | L(encode) | L(select) | R | C: fold takes a Program |

Invariants: launches from non-host callers refuse (no dynamic
parallelism — the launch boundary is host-only and explicit); `kind` is a
validated vocabulary, checked at dispatch and at cross-family inline (a
body using the `thread_idx` ambient inlined into a non-kernel context
refuses); the composition semantics never share syntax (`|` fuse-inline;
sequencing = host Python + fold; vertex→fragment pairing is PSO
composition; rewrite chains have their own form).

### 1.4 Caching and identity

One mechanism (`pdum.dsl` cache), three keyspaces:

- **Assemblage tier.** Building a Program is the compile step. Tier 1
  keys `(fp_head, arg_fp, generation)`; tier 2 is content-addressed on
  the built Program and supports **derivation-under-cache**: partials,
  component markers, and adjoint scanners are cache entries computed on
  demand from cache entries.
- **Kernel tier.** Types and identity in the key, never values;
  live-knob captures ride the uniform channel at zero recompiles;
  function-valued arguments put the argument Handle's FnType in
  `arg_fp` — a different pipeline shape is a new artifact, the same
  shape with different captured values is a warm hit plus a uniform
  rewrite.
- **Descent tier.** `chunk_fp` over the named-op Program in canonical
  form; registry key = (normalized chunk skeleton, boundary contract
  incl. saved-set demand and layout classes, license set, capability
  set, rules-generation); value = chain + authored region + artifact +
  assurance tier; the chain is a mandatory content-addressed output of
  elaboration; trust attaches at assurance tier ≥ 2. Until the
  Program-normalization pass lands (scheduled in the L4 runway, §8), the
  content-addressed tiers are declared **private caches**.

**Structural values are `Literal`-typed — three doors, one mechanism.**
Specialization is influenced by (i) the value's own type, (ii) call-site
wrappers (`Literal(5)`), and (iii) **definition-site annotations**
(`d: Literal[int]` on a config dataclass field, `n: Literal[int]` on a
parameter). An annotation is a local coercion applied at binding time:
the binder promotes the value's type before fingerprinting. No
unification, no solver. Identity stays type-keyed at every tier. The
enforcement: a **structural slot** (slice/pad extents, fold counts, dim
extents in declarations) accepts only `Literal`-typed values; a plain
int reaching one refuses with a designed message naming the annotation
fix.

### 1.5 The scope, the naming law, and the substrate contracts

**What threads through a model: the scope, and nothing else.** A model
factory takes exactly two things — **the scope `s`** and **your data
`cfg`** — by a crisp criterion: *the scope carries what must agree with
the naming law (path-addressed facets); `cfg` carries what doesn't (your
values, closed over like any values)*. The scope is an explicit,
immutable value — your position in the model's one address space —
carrying that path's facets:

- **Path.** `s / "attn"` derives a child scope; the path IS the
  structural address the naming law reifies. There is no second
  path-shaped thing: parameters, randomness, and policies are all
  addressed by it, so they can never drift apart.
- **Parameters.** `s.param(name, **dims)` — dim names as keyword keys,
  extents as values — declares a leaf at `path.name` and returns its
  tensor (virtual, loaded, or initialized; the code cannot tell).
  Declaration is idempotent; a conflict refuses.
- **Randomness.** The scope carries the randomness root (a program
  input); streams derive from **site paths** — `dropout(x, p, s /
  "attn_drop")` gets `fold_in(root, path)`. No key is threaded anywhere.
- **Policies.** An open, string-keyed set of aspects scoped to the
  subtree: `s.with_(mode="eval")`, `s.with_(trainable=False)`, later
  `init=`, precision regions, modes not yet invented. The scope
  *interprets none of them*; library idioms read them by convention
  (dropout reads `mode`; grad reads `trainable`). `training` is not a
  blessed boolean in any signature — it is one policy among many.

**Policies are identity-bearing.** A mode selects which Program gets
built, but a plain captured bool has the same *type* either way, so
type-keyed identity would let a train build and an eval build collide in
the cache. The scope folds its policy map into build identity the way
`Literal` folds values into types — the mode-as-loose-bool mistake is
unwritable.

**Context managers are sugar, never a global.** `with
s.with_(mode="eval") as se:` is lexical sugar over deriving an explicit
value, equally writable without `with`. There is no module-level
"current scope" stack, ever. The only mutation anywhere is the leaf
registry's build-time collection.

**The two-layer discipline.** Only one layer of model code touches the
scope. **The standard library is parameter-blind**: `layernorm`,
attention cores, dense blocks — functions from tensors to tensors,
parameters passed as ordinary arguments; no scope, no names, no
knowledge that provisioning exists. **The binding layer owns names**:
makers hold the scope, declare leaves, and hand tensors to library
functions. Name assignment happens in exactly one visible place — never
inside library code.

**The maker convention.** A maker is a plain function `(s, cfg) → unit`.
Composition points name their sub-scopes explicitly (`make_attn(s /
"attn", cfg)`), giving the **level-first hierarchy** checkpoints use
(`h.3.attn.wq`). Two blocks declared under one path collide loudly — a
designed refusal naming the fix — never an auto-suffix. `s.seq(name,
maker, cfg, n=…)` is the one sequencing combinator, and it is thin
enough to print (§6.3). `|` composes **units only** — never makers: a
maker-level pipe would be a third composition semantics punned onto the
operator, which this spec forbids.

**The naming law.** Programs identify tensors by string names; `grad`
returns name-keyed maps; loading, initializing, freezing, and RNG
streams all join on names. Names divide into **contract** (parameter
declarations, composition-site path segments, fold-state fields, derived
suffixes `name.d{i}`/`.rc`) and **internal** (intermediates — derived
from Python binding names as debugging niceties, excluded from
content-addressing, so renaming a local never changes program identity
or the public ABI). Anonymous temporaries get a deterministic scheme; a
rebuilt closure maps the same capture to the same name, pinned by a
rebuild-stability test.

**The substrate contracts, and the fashion test.** Everything user-facing
above the substrate is host-level *convention* over four pinned
contracts: (i) leaves are declared at paths — the naming law; (ii) units
are tensor→tensor functions; (iii) unit composition is build-time
function composition; (iv) identity follows the scope rules — policies
identity-bearing, structural values Literal-typed. The de-risk criterion
for any future idiom is the **fashion test**: Flax-style modules,
PyTorch-style modules, a curried point-free style, and the plain-maker
style must all be expressible as satellites over these four contracts —
and they are. Only the contracts are spec; idioms above them are
convention.

### 1.6 Provisioning

**The resting state of a model is virtual, and the builder is the single
source of truth.** A fresh scope holds nothing; running the builder
against it *collects* the spec — every `s.param(...)` registers a leaf
(name, dims, extents, carrier; **no buffer**), and the makers capture
exactly the tensors they declared. They cannot tell virtual from real,
because `typeof` is identical (layout + carrier; no buffer in the type).
There is no separately-maintained parameter table; a schema-first door
remains (`scope(schema=…)` validates declarations; provisioning
validates against a checkpoint's manifest either way), but the code is
authoritative.

Materialization is a separate, pluggable act joining on contract names,
with a no-waste law — **no allocate-then-overwrite, no gratuitous
copies, anywhere**:

- **Virtual** (the resting state): the full Program builds; cost and
  placement analyses read layouts and never values; nothing allocates;
  only execution refuses, quoting the fix. **Cache dividend, pinned:**
  virtual and provisioned builds have identical types, therefore
  identical fingerprints — analyze first, provision later, hit warm.
- **Load**: checkpoint entries become boundary descriptors over mmap'd
  regions directly — Buffer (DLPack shim) + Layout + Encoding (the
  file's dtype as a *fact*, exact-decoded per §4). Zero host copies; one
  explicit device transfer per buffer at provisioning, when backends
  exist. Foreign naming schemes are handled by translation tables —
  data, not code. Tied weights stored once arrive as one buffer → one
  leaf.
- **Init**: strategies keyed by name pattern (or scoped as an `init=`
  policy at declaration regions); each leaf's values are the closed-form
  random field `normal(fold_in(init_key, leaf_name), leaf_layout)` (§1.7)
  — materialized directly into the leaf's one allocation, or generated
  on-device by the same field lowered to Philox threads. Same key → same
  init, forever, on any device.

**What the scope is (and is not).** A string-keyed address space whose
**flat name space is primary** — `s / "h" / "3"` is a prefix *view* — and
which exists **only at build time**: after the build, the Program has
named inputs, and runtime state (weights, grads, moments) is plain
name-keyed dicts. It is deliberately not a pytree subsystem: pytree
machinery exists to turn arbitrary containers into positional argument
lists, and nothing here consumes positions — Programs, `grad` maps,
provisioning, and optimizers all **join on names**, so zipping state is
a dict join. The scope stays thin by law: path, registry, randomness
root, small policy map — interpreting nothing; on the order of a hundred
lines.

**Name stability, stated honestly.** Leaf-level and block-internal edits
never churn names (declare-at-use puts the edit and the name on the same
line; RNG streams are name-derived and ride along). The one instability
is **index-derived layer names**: inserting a layer shifts `h.{i}` after
it and a loaded checkpoint stops joining — universal across frameworks,
mitigated by the same translation tables foreign checkpoints already
need.

**Beyond ML.** None of this is an ML concept. In scientific computing
the three scenarios are: load = observational data entering as boundary
facts; init = synthesized initial conditions from closed-form fields;
virtual = costing a solver before buying the cluster. Scope declarations
carry units and charts as naturally as extents (`s.param("dt",
unit=u.s)`), and policies cover precision regions or boundary-condition
variants exactly as they cover train/eval.

### 1.7 Randomness

Forced by principles this spec already holds (purity, content-addressed
caching, recompute-based checkpointing):

1. **Randomness is a counter-based, coordinate-indexed, closed-form
   field.** `uniform(key, layout)` is a pure function of (key, lattice
   coordinates) — Philox-class bits, element *i* computed directly, no
   sequential state. It is a `FunctionalBuffer`-class citizen exactly
   like `iota`: zero memory, exact under view ops, free in the cost
   models, materialized only at a boundary that demands it. Bits are
   exact (`u32/2³²` is a rational) — carrier-consistent.
2. **Keys are ordinary values; the scope carries the root; streams
   derive from site paths** (`fold_in(root, path)`) and step indices
   (`fold_in(root, t)`) — insertion-stable and refactor-stable where
   positional splitting is not. No key is threaded through model code;
   `split` exists underneath and is rarely touched.
3. **Dropout is an idiom, not an op**:
   `where(uniform(stream, x.layout) < p, 0, x / (1-p))` — and it is
   **mode-aware**: it reads the scope's `mode` policy and is the
   identity under eval, so mode branches live in the idiom, not in user
   code. Train/eval are build-time variants — two cached Programs,
   distinguished by identity-bearing policy. AD falls out of existing
   rules (comparisons gradient-free; the mask acts as a constant field).
4. **The recompute theorem, pinned:** checkpointing and revolve
   recompute forward segments; the mask field regenerates bit-identically
   (same key, same coordinates), so gradients under recompute are exact
   *by construction*. One test pins it: revolve-checkpointed training
   step ≡ store-all, with dropout on.
5. **Device lowering is the same story**: Philox is pure integer
   arithmetic — a value-language device function (or vendor intrinsic),
   lowered like iota→thread_idx. Oracle and device produce bit-identical
   masks, so differential testing survives dropout.

Recorded stance: all randomness is named and keyed — reproducible by
default, always. Deferred: generator ceremony beyond Philox4x32,
distributions beyond uniform/normal (composites derive),
stochastic-rounding interplay, rejection sampling (variable consumption
is not straight-line — outside the subset, refused with the boundary
stated).

### 1.8 Extensions, punning, events

Vendor op namespaces spelled by one backend; capability flags checked at
build; `code_for_op` key-presence as the capability bit; capability-gated
`debug.print`; `record.artifact` as the escape hatch; rule-of-three
before any runtime abstraction. The adopt descriptor is where dictated
encodings live (§4) — buffer interop and precision facts are one
concept. Foreign tensor libraries are interop partners; `pdum.tl` is
in-house. The events seam lives in `pdum.dsl` and has emission points at
`pdum.tl`'s compile-ish seams (Program build, adjoint derivation,
descent certification), so `forbid`/`no_compile` can pin "this training
loop builds zero Programs" exactly as they pin kernel compiles.

### 1.9 The zoo gate

The acceptance instrument for every migration step. The zoo's tests run
in CI from step P2 on and stay green through every conversion:

1. Forward denotations vs numpy (rtol 1e-9 / atol 1e-12).
2. Gradients vs finite differences, **indexed by input names** — name
   identity is load-bearing; grad-map key drift is a failure.
3. flash == naive: forward AND the derived backward (no hand rule).
4. FDTD gradients carry their staggered charts.
5. Placement erasure bit-exact; exactly two gpu all-reduces on the
   megatron block; the erased program communicates nothing.
6. Cost oracles stable, modulo the one deliberate re-derivation when
   descriptor-fed sizes land (§4).
7. **Naming-law pins** — literal expected names (`"h.0.attn.wq"`, fold
   param names, derived `name.d{i}`) hardcoded in tests; a rebuilt
   closure maps the same capture to the same name. No dual-running of
   old builders: the contract is pinned by expectations, not by
   comparison with dead code.
8. The shared-axis extent refusal, pinned in the joint refusal battery.
9. **The tied-gradient pin**: one leaf declared once and captured twice
   receives one gradient — the summed contributions (§6.3).
10. **The virtual↔provisioned pin**: identical fingerprints, warm hit
    across provisioning (§1.6).

---

## 2. Versioning and release

All packages are versioned **in lockstep** and published to PyPI **in
lockstep** by the release workflow:

- One version source of truth: `packages/dsl/src/pdum/dsl/__init__.py`.
  `scripts/_versioning.py` anchors there and enrolls every
  `packages/*/pyproject.toml` (plus the root) as lockstep version files —
  bumps apply across all packages in unison, never individually.
- `release.yml` builds and publishes **all** workspace members' dists in
  one run, tagged once.
- Dist names follow the namespace: `pdum.<name>` publishes as
  `habemus-papadum-<name>`.
- The root pyproject is an unpublished virtual root: workspace glue, dev
  dependency groups (members via `[tool.uv.sources] workspace = true`),
  docs tooling.
- Release runs remain deliberate acts through the workflow trigger;
  nothing in the migration publishes as a side effect.

## 3. The purge

Git history is the archive. The living tree carries only the go-forward
system. Three rules govern every deletion: **distill before deleting**
(the load-bearing knowledge lands in canon or tests first), **pin
contracts with literal expectations** (never by dual-running dead code;
a comparison fixture genuinely needed during a conversion is temporary
scaffolding, deleted when its gate retires), and **new code never
references old code**.

### 3.1 Deleted from the pdum.dsl side

`dsl_reference/`; `stdlib/arrays.py` and `stdlib/transforms.py` (Named,
over, jvp-as-pdum-concept, matmul — the tensor tier owns all of it; the
tangent-engine rows merge into the one derivative table first, §5 S.2);
the array half of batteries; `backends/c.py` and the whole demo tree
(python renderer, wgsl runtime, graphics batteries — the fwidth residue
is re-created at the shader tier in P8); `combinators.py` (the fuse pipe
is re-authored clean in `pdum.dsl`); `viz.py` and `bench.py` (rebuilt
against the new surfaces when needed); the book (`docs/book/`,
`scripts/book/`); the tests of every deleted module.

### 3.2 Deleted from the tensorlib side

`build.py` (the Build name-manager — replaced by core naming + the
shared syntax); mdsl's Sym tracer, `defmarker`/`defreducer` entry
points, process registries and `node_digest` (replaced by the AST
producer and cache-backed registries). **Surviving mdsl parts
relocate:** the `Arg/Const/Prim` Node schema (the declared stability
boundary — the AST frontend is a new *producer* for it; its consumers
never change), the symbolic `diff`/`_D` machinery, and the
CompositeReducer BPTT engine. The `defreducer`-shaped *declaration API*
(state/element/lift/combine/init/project + declared associativity)
survives with producer-swapped bodies — a reducer is a structured
declaration, not one expression.

### 3.3 Distillations (written in the same commits as their deletions)

- **Backend notes** (one short doc): the numeric policy (truncating
  integer div/mod with exact twins; float mod = fmod; u64 constant
  refusals), artifact-carries-its-contract, the WebGPU runtime learnings
  (synchronous readback is a fixed-latency protocol act;
  timestamp-query timing; uniform-plan/bind-group layout; encode and
  submit are separate acts and the encodable is the API), and the
  bench/instrumentation methodology (warmup, tuned evals, minimum as
  estimator; phase decomposition by seam-wrapping).
- **The aliasing lesson**: writable/readable overlap is silent
  corruption; it is a day-one refusal test at the `@compute` store seam
  (P7), not a memory.
- **The refusal voice**: one shape — what happened, the principle
  violated, the quoted fix, the source location — seeded as the joint
  refusal battery (P3) that every later refusal extends.
- **The oracle status rule**: per-element host dispatch of kernels is
  debug/oracle-grade; the reference executors are the only per-element
  consumers, and reference execution is always spelled (§1.1).

### 3.4 Docs and notebooks

`docs/design/010–180` move to `docs/design/history/` in the purge
commit; mkdocs nav is rebuilt around this document and the two packages'
docs. Everything still load-bearing from prior canon is restated in §8
(the runway), so nothing in history needs to be read to proceed.
Tensorlib notebooks 00–06 teach surviving semantics and move with the
package; 07–13 are re-authored or dropped as their APIs change during
P4–P6. Fixes that survive regardless of the purge land at extraction:
the unknown-kind dispatch refusal and the record-value designed refusals
(both in surviving core modules).

---

## 4. Precision and boundaries

**Facts at the boundary, choices in the interior, carrier semantics
throughout.**

**The contract.** Semantics are carrier-valued end to end
(bool/int/rat/real/complex). No compute dtype exists on the user
surface. Dtype is a property of buffers and encodings **at the
boundary**, recorded in load/adopt/writable-argument descriptors, never
on tensors mid-computation. **Exact decode:** every finite bit pattern
is a specific rational (int4+scale decodes to `scale[g]·q`), so the
denotation stays exact over exactly-known inputs — a bf16 checkpoint is
a *fact* the descriptor records, not a semantic property of the program.
**Explicit rounding:** where rounding IS the semantics (QAT, stochastic
rounding), it is the explicit exact op `round_to(encoding)`; its AD rule
is **straight-through by default** (a zero derivative would make every
quantized parameter untrainable), with zero available by declaration.
**The discipline:** every precision appearance is a boundary fact, a
descent choice, or an explicit `round_to`; mid-program
astype-as-semantics does not exist and the IR has no op for it. Edge
rules: inf/nan bit patterns **refuse at decode** (an extended-real
carrier is a recorded future opt-in); writing real-carrier results into
an encoded writable argument rounds as part of the boundary contract
(`denotation = encode_out ∘ f ∘ decode_in`) — the one implicit rounding,
declared, never silent.

**Buffers.** A Buffer is a thin shim over a rank-1 DLPack-style handle:
data pointer, **explicit device**, length. Our Layout addresses it; an
Encoding interprets it. Host buffers are the degenerate case; device
buffers carry the same shape (zero-copy interop in both directions).
Device-resident persistent state and the epoch/ownership handshake for
adopted device buffers are L2's first requirement and build on this same
handle.

**The boundary descriptor** = Buffer + Layout + Encoding (+ carrier +
units). `Encoding` is a small hierarchy — NumpyEncoding (including
**structured encodings**: field names, offsets, padding — the memory
shape of tensors-of-structs), QuantGroupEncoding (int4 nibbles +
per-group scales over two buffer regions), FormatEncoding (e.g.
bgra8unorm-srgb with the transfer curve in decode) — each declaring its
exact decode/encode. The *logical* record type (field names and types)
is the interior value type; offsets, padding, and alignment are encoding
facts the interior never sees. Interior program values carry carrier +
units + layout shadows only; the IR cannot mint encoding-bearing values
(enforced at the IR/signature layer). The reference executors' float64
interior is a declared oracle property, never semantics.

**Strides are bytes.** Layouts address bytes everywhere, unchanged —
this is what makes `field()` on structured dtypes a free view (offset
bump + dtype change, padding skipped) and keeps the whole affine/guard
algebra integer. The reconciliation with the precision doctrine is a
rule about meaning, not units: **interior shadow layouts are structural
only** — nesting and aliasing information, never byte-authoritative.
Byte truth enters exactly twice: at boundary descriptors, and at L2's
encoding assignment, from which bufferization re-derives interior byte
layouts. Sub-byte encodings (nibbles) are the Encoding's decode concern
over byte regions, never the affine map's.

**The interior.** Precision enters at exactly three lowering points:
(1) **descent licenses** — taxonomy {none, reassociation,
precision-demotion}, equivalence stated over the carrier denotation,
tolerances and input domain in the declaration, the license set in the
registry key; the numeric tier monitors divergence and never certifies
it; (2) **L2 storage assignment** — materialized intermediates get
encodings chosen at bufferization; (3) **machine-tree byte predicates**
— consumers, not choosers; capacity checks read lowering annotations
plus boundary facts. The worked check: weights bf16 and activations f32
as facts, a real-carrier contraction as the program, f16 tiles + f32
accumulators under a license at descent, the writable argument's
descriptor encoding the result — byte-blind capacity mistakes are
impossible by construction.

**Fallback, with criterion.** The two-surface model (a family
element-dtype parameter + the descent license) is the recorded fallback.
Before L4, the mixed-precision/QAT sample (master vs bf16 weights, loss
scaling) is written in boundary-facts terms; fall back iff a required
program's *meaning* — not cost — depends on an interior encoding that is
neither a boundary fact nor expressible as `round_to`. Falling back is a
written owner decision, never a silent switch.

---

## 5. The syntax stack, worked per tier

Tags: **[now]** = surviving code runs this today; **[build]** = this
plan builds it.

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
name-fate analysis; input names are declared, and `grad`'s map stays
keyed on them. **Lift rule (normative):** Python *numbers* lift to
consts aligned to the tensor operand (dims and charts inherited) — the
one implicit lift, const-only; tensor–tensor misalignment always
refuses; `repeat` stays explicit. **Structural values are
Literal-typed** (§1.4): extents entering structural slots come from
`x.extent(d)`, `Literal(...)`, or an annotated parameter; a plain int
refuses, naming the fix. **Vocabulary completeness:** the committed
method set includes `.bind(level=...)`, `iota_of(t, dim)`, and the
two-operand reduce form; `Program`/`Instr` remain public
hand-constructible data, and hand-emits get names through the same core
naming contract.

### S.2 The shared expression syntax

Type-directed lifting; straight-line enforced at lowering; bounded
`if`/`for` exist only in the value language, never over tensor-typed
values. Expressions range over value types: records construct,
destructure, nest, and carry methods; a method is a device function with
`self` first, usable over value-typed and tensor-typed receivers alike.

One definition, two consumers [build — the gate is a differential]:

```python
def gelu(x):
    return 0.5 * x * (1 + tanh(GELU_C * (x + 0.044715 * x*x*x)))
```

— lowered as a pointwise marker body under `ir.run` AND inlined as a
device function into a `@compute` kernel; the two paths must agree
numerically.

**Marker-body granularity is a hard gate.** A marker (gelu's formula)
and a reducer's combine are small named bodies the AD machinery
differentiates *by inspection* — derived partials by tree rewriting;
flash attention's backward exists because the combine is such a body.
The AST producer must lower one marker to one named, inspectable body
tree over primitives (captured constants become Consts), and one combine
to the same inside its structured declaration — never inlined away,
never an opaque call. The flash derived-backward test enforces this.

A reduction combine with record state, and a tensor-typed fold step
[build]:

```python
def flashsm_combine(L, R):                     # State = (m, den, o) — a record
    m = maximum(L.m, R.m)
    sl, sr = exp(L.m - m), exp(R.m - m)
    return State(m, L.den*sl + R.den*sr, L.o*sl + R.o*sr)

def fdtd_step(E, H, n: Literal[int]):
    dE = (E.shift(x=-1).slice(x=(0, n-1)) - E.slice(x=(0, n-1))).with_charts(x=h_chart)
    H1 = H + c * dE                            # c lifts, inheriting dims AND charts
    dH = (H1.slice(x=(1, n-1)) - H1.shift(x=1).slice(x=(1, n-1))).with_charts(x=e_chart)
    E1 = E + c * dH.pad(x=(0, n), fill=0.0)
    return E1, H1                              # carry; layout preserved — checked
```

**The one derivative table.** The value-tier tangent rules and the
marker partial rules are the same object — op → linearization, None =
gradient-free — one table in the core transform column. The table grows
only when a primitive joins the core; everything else derives:
`CompositeMarker.partial(i)` is forward-tangent application over the
lowered body (basis seed, DCE, registered `name.d{i}`) —
derivation-under-cache; the reducer BPTT engine consumes partials
through the same interface. **At-kink law:** the table is one-sided and
partitions — at a tie, exactly one operand receives the cotangent
(first-wins), and reduce adjoints derive through the pairwise combine,
inheriting the partition law. `jvp` returns a fixed subgradient
selection at kinks; this is frozen contract, pinned at the kink points.

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
  ride the uniform channel; swapping a stage → new artifact. The
  lowering: inlining through an FnType-typed parameter, arg-rooted ABI
  slots, the Handle value passed to the build alongside arg types; the
  guard policy for argument Handles is recorded in P7's design.
- **Stores and ordering**: the IR represents ordering as **token
  threading** — a store consumes and produces an ordering token, so
  ordering is ordinary dataflow, and tile barriers and L2 bufferization
  consume the same mechanism later. The **frontend policy** is program
  order: one implicit token threads through all stores in statement
  order; tokens never appear in user syntax. Day-one contract: a
  writable argument overlapping any readable capture/argument refuses
  with the ping-pong message; in-place returns only ever as an
  L2-certified rewrite.
- **Tensors of value types**: `img[y, x]` on a struct-element buffer
  loads a record; stores accept a record; the element's memory shape is
  the descriptor's structured encoding (§4).
- **Launch config**: invocation-only, rides the launcher, never any key.
  Threads-per-block is a value-specialized bracket (re-render on change,
  no identity change); blocks/streams are pure launcher data.
- **Iota unification**: the same kernel is expressible as pointwise over
  coordinate iotas; the iota→thread_idx descent is a rewrite stage whose
  WF predicate is "no iota reaches a materialization boundary"; the
  fused and assemblage forms are differential-tested against `ir.run`.

### S.4 Vertex/fragment [build — P8]

`@vertex`/`@fragment` share the ambient contract. **Varyings are a
record**: the vertex kernel returns it, the fragment kernel receives it
— the value-type system is the interface machinery, with the
interpolation contract declared per field. `fwidth` is the wrt-ambient
derivative at this tier. Vertex→fragment pairing is PSO composition (its
own semantics, never `|`); the per-frame deliverable is an **encodable**
(render bundle / draw-into-pass) — the host owns the pass, the submit,
and the swap chain. FnType carries an optional result-type slot
(reserved at extraction). Semantics land against golden artifacts plus a
minimal reference interpolator; the GPU rasterization path arrives with
the L4-era backends.

### S.5 Tile and warp [reserved — the L4 brief governs, §8]

The authored descent is not IR: it is the value of a certified-lowerings
registry entry; the kernel boundary in the Program is an
erasure-preserving annotation. Tile vocabulary: stage/barrier/accum,
tile loops as split+bind one level down; WF certificates checked on the
result — capacity (byte-exact, from descriptors and annotations),
race-freedom (checker-owned tokens — the same tokens as §S.3),
convexity. Whether `mma` pattern-matches mul→reduce or requires a stated
annotation is answered inside the L4 design. Warp: straight-line
post-unroll, uniform control, lane-complete. Below both: vendor punning
and external oracle fixtures.

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

## 6. Worked example: GPT-2 end to end

The standing red-team exercise, kept in the spec as its proof of use:
one definition serving load, init, virtual analysis, and training with
freezing — with the compiler's view checked at the end. Heads are born
as dims (`wq: (d, nh, hk)` — never split into existence); the causal
mask is an iota comparison; token embedding uses `take` (§6.6f).

### 6.1 Config and library (parameter-blind)

```python
@dataclass(frozen=True)
class GPT2Config:
    # structural fields: Literal-annotated (the definition-site door, §1.4)
    d: Literal[int]; nh: Literal[int]; hk: Literal[int]; m: Literal[int]
    v: Literal[int]; t_max: Literal[int]; layers: Literal[int]
    # live knobs: plain floats — captured values, never in identity
    eps: float = 1e-5; scale: float = 0.0
    p_attn: float = 0.1; p_resid: float = 0.1; p_embd: float = 0.1

GELU_C = 0.7978845608028654

def gelu(x):                                   # value-language body; partials derive
    return 0.5 * x * (1 + tanh(GELU_C * (x + 0.044715 * x*x*x)))

def layernorm(x, g, b, *, feat, eps):          # assemblage helper: ordinary Python
    mu = x.mean(feat)
    xc = x - mu.repeat(feat, x.extent(feat))
    sd = ((xc * xc).mean(feat) + eps).sqrt()
    return xc / sd.repeat(feat, x.extent(feat)) * g.repeat_like(x, but=feat) \
           + b.repeat_like(x, but=feat)

def causal_softmax(sc, *, q="t", k="s"):
    mask = iota_of(sc, k) <= iota_of(sc, q)    # closed form; costs nothing
    sm   = where(mask, sc, const_like(sc, -1e9))
    e    = exp(sm - sm.max(k).repeat_like(sm, but=None, dim=k))
    return e / e.sum(k).repeat_like(e, but=None, dim=k)
```

### 6.2 The makers (binding layer): declare-at-use, mode-free dropout

```python
def make_attn(s, cfg):
    D, H, K = cfg.d, cfg.nh, cfg.hk
    ln1g, ln1b = s.param("ln1g", d=D), s.param("ln1b", d=D)
    wq = s.param("wq", d=D, nh=H, hk=K)
    wk = s.param("wk", d=D, nh=H, hk=K)
    wv = s.param("wv", d=D, nh=H, hk=K)
    wo = s.param("wo", nh=H, hk=K, d=D)

    def attn(h):
        a  = layernorm(h, ln1g, ln1b, feat="d", eps=cfg.eps)
        q  = contract(a, wq)                           # unique shared axis: "d"
        k  = contract(a.rename(t="s"), wk)
        v  = contract(a.rename(t="s"), wv)
        sc = contract(q * cfg.scale, k, axis="hk")     # "nh" rides; axis named to
                                                       # break the genuine ambiguity
        pr = dropout(causal_softmax(sc), cfg.p_attn, s / "attn_drop")
        cx = contract(pr, v, axis="s")
        o  = contract(cx, wo, axis=("nh", "hk"))
        return h + dropout(o, cfg.p_resid, s / "resid_drop")
    return attn

def make_mlp(s, cfg):
    D, M = cfg.d, cfg.m
    ln2g, ln2b = s.param("ln2g", d=D), s.param("ln2b", d=D)
    w1, b1 = s.param("w1", d=D, m=M), s.param("b1", m=M)
    w2     = s.param("w2", m=M, d=D)

    def mlp(h):
        a = layernorm(h, ln2g, ln2b, feat="d", eps=cfg.eps)
        m = gelu(contract(a, w1) + b1.repeat_like(a, but="m"))
        return h + dropout(contract(m, w2), cfg.p_resid, s / "resid_drop")
    return mlp
```

`dropout` reads the scope's `mode` policy — no `if training:` appears
anywhere in user code. An architecture edit touches the `s.param` lines
and the body that uses them, in one file, and nothing else.

### 6.3 Assembly: level-first names, `seq`, the tie

```python
def make_block(s, cfg):
    return make_attn(s / "attn", cfg) | make_mlp(s / "mlp", cfg)

def make_gpt2(s, cfg):
    wte = s.param("wte", v=cfg.v, d=cfg.d)             # declared ONCE — tied below
    wpe = s.param("wpe", t=cfg.t_max, d=cfg.d)
    lnfg, lnfb = s.param("lnfg", d=cfg.d), s.param("lnfb", d=cfg.d)

    def embed(ids):
        tok = wte.take(ids, dim="v")                   # gather — §6.6f
        e   = tok + wpe.slice(t=(0, ids.extent("t")))
        return dropout(e, cfg.p_embd, s / "embd_drop")

    trunk = s.seq("h", make_block, cfg, n=cfg.layers)  # h.0.attn.wq, h.1.mlp.w1, ...

    def head(h):
        hf = layernorm(h, lnfg, lnfb, feat="d", eps=cfg.eps)
        return contract(hf, wte, axis="d")             # TIED: the same object

    return assemblage(embed | trunk | head)

train_model = make_gpt2(root.with_(mode="train"), cfg)   # two Programs, both cached —
eval_model  = make_gpt2(root.with_(mode="eval"),  cfg)   # policies are identity-bearing
```

`seq` is deliberately **thin enough to print** — it is the explicit host
loop, named; the loop form remains legal and identical in meaning:

```python
def seq(s, name, maker, cfg, n):           # cfg may be a value or a fn of i
    units = [maker(s / name / str(i), cfg(i) if callable(cfg) else cfg)
             for i in range(n)]
    return pipe(units)                     # n-fold unit composition
```

`|` here is build-time function composition threading one value (`h`) —
the same fuse semantics, realized as program-fragment composition; the
scope and `cfg` ride as closed-over symbols, never threaded values.
**The tie**: `wte` is declared once and the same object is captured by
both `embed` and `head` → one input leaf (capture identity decides); its
gradient is the summed contribution automatically (zoo gate pin 9).

### 6.4 The three provisionings

```python
root  = scope()                                  # rng root = a program input
model = make_gpt2(root.with_(mode="train"), cfg) # builds AND collects
root.spec()      # derived: "h.0.attn.wq": (d:768, nh:12, hk:64), ...

# Virtual (the resting state): analyze with zero allocation
ops_count(model.program); peak_memory(model.program, schedule)

# Load: mmap'd safetensors → boundary descriptors, zero host copies
weights = provision(root, source=safetensors("gpt2.st"))

# Init: strategies by name pattern over closed-form random fields
weights = provision(root, source=init(
    root_key / "init",
    default   = normal(std=0.02),
    overrides = {
        "*.ln?g": ones,   "*.ln?b": zeros,   "*.b?": zeros,
        "*.wo":   normal(std=0.02 / sqrt(2 * cfg.layers)),   # scaled resid init
    },
))
```

Each init leaf is `normal(fold_in(init_key, leaf_name), leaf_layout)` —
materialized directly into the leaf's one allocation (or generated
on-device by the same field lowered to Philox threads). No scenario
contains an allocate-then-overwrite or a gratuitous copy.

### 6.5 Training: trainable by default, frozen by name or by region

```python
trainable, frozen = root.partition(freeze=["h.0.*", "wpe"])   # post-hoc door
# (the in-model door: declaring under s.with_(trainable=False, mode="eval"))

step  = grad(loss_fn, wrt=trainable)      # keep-set → requested-gradients DCE:
                                          # frozen weights' backward work is PRUNED
opt   = adam(trainable)                   # moment dicts keyed by the same names

for t, batch in enumerate(data):
    lr    = sched(t)                                      # live knob: value, never keys
    grads = step(weights, batch, root_key / ("step", t))  # per-step streams, warm cache
    weights, opt = opt.update(weights, grads, lr)
```

Declaration is the default keep-set; invocation overrides it. A
different freeze-set is a different derived Program — cached like any
other. For gradient control on *activations*, `stop_gradient(x)` is a
plain IR op (identity forward, zero backward) — dataflow, not scope.

### 6.6 The compiler's view: granularity checked per optimization

**(a) Distribution — GREEN.** Named dims + `bind`; Megatron's block
already proved collectives are *read off the algebra* (two all-reduces,
discovered not written) and the placed backward carries bindings. GPT-2
distributes by binding `nh` (head parallel) or `d`/`m` (Megatron-style);
the batch dim, when it arrives, is one more named dim.

**(b) Tiling and mma selection — YELLOW, known and bounded.** `contract`
is the `repeat·mul·reduce` normal form; L4 selects tensor cores by
*recognizing* mul-solely-consumed-by-reduce. The known miss: if an
optimizer saved the raw product `a*w` for backward, the pattern breaks.
Worked against GPT-2: the standard saved set (min-cut with contractions
recompute-banned) does **not** save the product, so recognition holds on
this model. The residual risk is the open mma half of the L4 brief
(pattern-match vs stated annotation); the naive→flash registered rewrite
shows the escape hatch (a recognized pattern promoted to an annotation).

**(c) Fusion, including flash + dropout — GREEN, better than baseline.**
Flash derives from the declared online-softmax combine (backward
derived, no hand rule), and because the attention-dropout mask is a
closed-form field over coordinates, **fusing dropout into the flash
kernel materializes nothing** — the kernel computes mask bits from
(key, coords) in-register, exactly what hand-written flash
implementations do with in-kernel Philox, arrived at here by
construction.

**(d) Checkpointing/revolve — GREEN.** The §1.7 recompute theorem: masks
regenerate exactly. Dropout adds `where` + comparison nodes — pointwise,
recompute-cheap, precisely what the cheap-chain heuristic wants to
recompute rather than save.

**(e) Module boundaries — GREEN.** After composition, "this was the
attention block" survives in the **level-first name prefixes**
(`h.3.attn.*`) — machine-readable and stable under the naming law.
Partitioning operates on dataflow plus names; the kernel boundary is an
annotation anyway; a block-scope annotation is available as an
erasure-preserving addition if L4 partition search wants it.

**(f) Gather — the one primitive GPT-2 forces.** Token embedding
(`wte.take(ids, dim="v")`) is data-dependent indexing; its adjoint
(scatter-add into the embedding gradient) is exactly how the tied `wte`
trains. One-hot-contract is semantically correct but violates the
no-waste law (T·V work for T lookups). `take` + scatter-add adjoint +
cost entries land at P9 (§7). Deferred beyond it: top-k/MoE routing;
sampling stays **host-side** in the inference loop (logits out, host
samples, next token in).

**(g) Sequence length — stated honestly.** Extents are structural
(Literal-typed): a new `t` is a new Program. Building is cheap and
cached per length; the practical idiom is length bucketing (pad to the
bucket, mask via closed-form comparisons). Decode-time KV caching
composes as the ring/window boundary sample from the L2 runway.

**(h) What the representation cannot express — unchanged and intended.**
Data-dependent control flow inside programs, dynamic shapes *within* one
Program, mutation outside the store seam: stated subset boundaries with
refusals. GPT-2 needed none of them.

---

## 7. The migration plan

Sequential steps; each ends with the surviving suites green. Gate
suites: **D** `packages/dsl` tests, **T** `packages/tensorlib` tests,
**Z** the zoo gate (§1.9), **B** lint + budget + docs build. Tests of
deleted modules are deleted with them — green means the go-forward
system.

**P0 — Workspace + release machinery (M).** `packages/dsl` and
`packages/tensorlib` skeletons; root becomes the unpublished virtual
root; `_versioning.py` re-anchored with lockstep enrollment;
`release.yml` reshaped to build/publish all members in lockstep; CI
reshaped around the surviving suites; loc-budget buckets redrawn (the
tripwire discipline continues). GATE: CI green; `_versioning.py
current` passes; a local `uv build` of both members succeeds; no
publish.

**P1 — The purge + the core lands (L).** git-mv the kernel engine into
`packages/dsl` (with `pdum.dsl.value`, `pdum.dsl.render`); write the
reference evaluator; land the extraction-time fixes (unknown-kind
dispatch refusal; record-value designed refusals; the FnType
result-type slot); re-author the fuse pipe on the Registry. Delete
everything in §3.1 plus `docs/book`; move design docs 010–180 to
`history/`; write the §3.3 distillation notes in the same commits.
GATE: D green (the value-language battery with literal expectations, on
the reference evaluator); B green; the tree contains no import of
anything deleted.

**P2 — Tensorlib promoted (M).** git-mv `explorations/tensorlib/tensorlib`
→ `packages/tensorlib/src/pdum/tl` (zoo inside), tests to the member,
conftest path-hack gone, notebooks per §3.4. The zoo gate enters CI —
live from here on. GATE: T + Z green at the new paths; `import pdum.tl`.

**P3 — Onto the core (L).** Process registries die: cache-backed
registries with derivation-under-cache; core naming replaces Build's
hint dedup (Build itself alive for the zoo builders until P5); the Node
schema relocates to its stability-boundary module; events emission at
Program-build/adjoint seams; the joint refusal battery seeded with the
extent refusal pinned. GATE: T + Z green; idempotence pin
(re-registering an identical marker yields one entry).

**P4 — The shared expression syntax (L).** The AST producer replaces
the Sym tracer; straight-line detection replaces trace-time refusal;
the structured reducer declaration keeps its API with producer-swapped
bodies; record-typed reducer state; **value-type expansion** (nested
records and methods); tensor-typed lifting lowers fold steps to step
Programs; the `Literal` annotation door and the structural-slot
refusal; **the random-field primitives** (uniform/normal as closed-form
FunctionalBuffer citizens) with their derivative-table entries; the
merged derivative table with the first-wins at-kink re-pin (its own
commit, kink points pinned). The tracer survives this step only as
scaffolding for the producer-equivalence check and is deleted at the
end of it. GATE: Z green with markers/reducers/fold-steps re-authored —
denotation-identical including flash's derived backward and FDTD
charts; the marker-granularity gate (S.2).

**P5 — The scope; `@assemblage`; Build dies (M).** The scope lands:
`param` declaration door, path derivation, policy facets
(**identity-bearing** — part of this step's gate), `seq`, `partition`;
provisioning (collection, `provision(source=safetensors|init)`, the
virtual resting state); `@assemblage`; zoo builders re-authored as
makers (level-first names; bind/iota_of/two-operand-reduce vocabulary
included); the naming-law literal pins land (gate 7), plus the
tied-gradient pin (gate 9) and the virtual↔provisioned pin (gate 10);
Build and the remaining mdsl entry points are deleted. GATE: Z green on
the re-authored builders with hardcoded name expectations; a
policy-collision test (train vs eval builds never share identity);
D + T green.

**P6 — Precision and boundaries (M).** The §4 design lands before any
L4 work: dtypes split (carriers semantic; the Encoding hierarchy
including structured encodings); the DLPack-shim Buffer with explicit
device; descriptors; constructors re-roled as boundary acts;
IR-cannot-mint-encodings enforced; machinery dtype sites converted;
`round_to` with straight-through AD; inf/nan decode refusals;
descriptor-fed byte-exact sizes replace the 8-byte convention (cost
oracles re-derived as their own reviewable diff); the QAT sample
written and evaluated against the fallback criterion. GATE: Z green
with zero denotation changes; the re-derived cost diff reviewed.

**P7 — `@compute` (L).** In order: token-threading store representation
with the program-order frontend policy → store lowering with the
day-one overlap refusal → `thread_idx` + the iota→thread_idx descent
stage → tensors-of-value-types (structured-encoding loads/stores) →
function-valued-argument lowering (guard policy recorded) → launch
config → the Philox device function and **the recompute-theorem test**
(revolve-checkpointed training step ≡ store-all, dropout on). Execution
on the reference evaluator; device backends remain L4-era work. GATE:
the S.3 example runs on the reference evaluator; the iota-unification
differential; the two-consumers differential (S.2); key-discipline pins
(shape miss / value hit / launch never keys / fn-swap miss); overlap
refusals; the compile-once thesis test for function-valued arguments; a
struct-element kernel round-trips through a structured encoding.

**P8 — Graphics (M).** The `@vertex`/`@fragment` kinds in the validated
vocabulary; varyings-as-records with per-field interpolation
declarations; PSO pairing as its own composition semantics; the
encodable deliverable (render bundle / draw-into-pass); `fwidth` as the
wrt-ambient derivative. Semantics against golden artifacts + the
minimal reference interpolator. GATE: a vertex+fragment pair lowers,
pairs, and encodes; the varyings record round-trips; goldens pinned;
D + Z green.

**P9 — Gather; runway handoff (S).** `take`/gather lands: op + layout
semantics + scatter-add adjoint + cost entries (the training-loop story
is honest before handoff), with the GPT-2 embedding as its zoo sample.
Then the handoff: the tiled-matmul zoo entry; the license-schema stub +
the worked GEMM license declaration; the L2 blocker list; the open
registry (streams, device-resident state, normalization, warp
vocabulary, external-oracle fixtures, the operators door, adversarial
input families — one −inf-mask attention case seeded into Z now). GATE:
Z green including tiled-matmul and the embedding sample; §8 handed to
the L4/L2 work as its brief.

**Not doing:** async, in any form. A CUDA-clone language. A pdum tensor
dialect, translation frontend, or second AD. `out=`. A global
"current scope" stack. Archives, shims, frozen-fixture museums, or
dual-running dead code beyond within-step scaffolding. Device backends
before the L4 era. Publishing as a side effect of migration.

---

## 8. The runway (the L4/L2 brief, self-contained)

**L4 — the kernel language.** A kernel is an erasure-preserving grouping
annotation in the Program; the authored descent lives as the value of a
certified-lowerings registry entry, keyed per §1.4. Tiling is
split+bind one tree level down; the three genuinely new things at L4 are
predicates and decisions, not representation: the capacity WF predicate
(byte-exact, from descriptors + lowering annotations),
ordering/race-freedom (the token mechanism of §S.3, owned by the
checker), and materialization-boundary placement. Legality = the
equivalence chain (a sequence of named certified rewrites — split, bind,
reorder-under-license, fuse-as-elision, pad-with-guards, plus the
overlapped-split/halo-recompute class the stencil flagship needs) AND
the per-level WF certificate checked on the result. The objective:
minimize parent-memory traffic under child capacity; the pipeline is a
descend-and-revisit loop with declared invalidation edges (fusion
invalidates checkpoint and traffic plans; placement invalidates
partition candidates); combine-introducing rewrites precede `grad`,
split/bind/place commute with it; the naive→flash move is a registered
named rewrite whose license is the declared combine. Flagships: tiled
GEMM, flash attention, and the fused stencil chain as the
non-contraction acceptance test; flagship gates pin adversarial input
families (−inf masks, cancellation, non-divisible tails), never random
draws alone. Assurance tier and input-domain coverage are recorded
fields of every registry entry. The mma selection question
(pattern-match vs stated annotation) carries the §6.6b GPT-2 worked
case. Scheduled inside the L4 runway: the **Program-normalization
pass** (the precondition of the registry's cross-model payoff; until it
lands the content-addressed tiers stay declared-private),
**stream/overlap semantics** (no `stream=` appears in any sample until
this design exists), the warp vocabulary, the external-oracle fixtures,
and the per-type operator-extension door (geometric algebra's
registration surface). Device backends are built here, fresh, against
the distilled notes and the reference executors.

**L2 — bufferization.** Inputs already in the tree: the exact alias
theory (footprint/overlaps/injectivity), writes-through-views gated on
injectivity, materialize-elision when nesting holds. Ordering: after
fusion decisions — bufferization consumes kernel boundaries. The
blocker list: value numbering for recompute duplicates (name ≠ value);
chart-denominator normalization for codegen; interior-encoding
assignment (§4 — interior shadow layouts are structural-only; L2
re-derives byte layouts from assigned encodings); the ring/window
boundary sample with both instances (KV-decode and the audio delay
line) and its erasure obligation (the same surface program, bufferized,
reproduces the row-write); **device-resident persistent state** and the
epoch/ownership handshake for adopted device buffers (on the
DLPack-device Buffer, §4); and the token mechanism from P7, which
bufferization consumes directly.

**What the whole system is, on completion.** One frontend machine with
six vocabularies over it; one AD with one derivative table and
derivation-under-cache; one cache mechanism with three keyspaces and an
honest statement of what is private until normalization lands; one
refusal voice with a joint battery; one precision doctrine — facts at
the boundary, choices at lowering, carriers throughout; value types —
records included — flowing from device functions through buffers to
varyings; one scope carrying names, randomness, and policies for every
domain's models; two reference executors as permanent, always-spelled
oracles; and a zoo whose denotations gate every step.

---

## Appendix A — history

This document supersedes 170 and 180 by merging them, and through them:
160 (whose archive/shim posture the purge replaced), the 150 direction
memo (its L4/L2 brief is restated in §8), 130 stages 2–3 (cancelled),
the 100/110 array/transform canon (its stdlib is deleted), 090 §3's
coords-as-params profile and §5's async path. Notable reversals decided
along the way: tensorlib's Build/mdsl frontend and process registries
are replaced by `pdum.dsl` infrastructure; the carrier+dtype-on-Tensor
contract retreats to boundary descriptors; the at-kink adjoint
convention changed to the first-wins partition law; an interim
carrier-only precision stance was retracted in favor of boundary-facts
when it was observed that loaded weights dictate encodings; a
double-curried maker convention with maker-level pipes was considered
and rejected (third pipe semantics; ceremony; end-applied cfg); an
ambient current-scope veneer was considered and declined in favor of
explicit scope threading (identity soundness at the assemblage
boundary; deferred-call semantics; the field's precedent). `out=`,
`over`, `jvp`-as-pdum-concept, `matmul`, `Named`, per-element dispatch
as a production path, and the demo backends are deliberately absent
from this system. Design docs 010–180 live in `docs/design/history/`;
git history is the archive.
