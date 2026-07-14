# Proposed architecture: the pdum.dsl kernel

**Status:** synthesized proposal, ready for review. Not yet code.

**Provenance.** Produced by a structured multi-agent study on 2026-07-11, fully
auditable under [`docs/design/research/`](research/):

- **R1–R9** — deep-dives into numba, DaCe, xDSL/MLIR, JAX, kernel DSLs
  (cupy.jit/Triton/MLX/Taichi), torch.compile, minimal compilers
  (tinygrad/nanopass/QBE/egglog), user types & units, and PEP 750 t-strings —
  all verified against live July-2026 sources.
- **V1–V5** — consolidated verdicts on the five open questions (frontend, IR,
  hooks, marshaling, transformations).
- **P1–P3** — three independently-authored architecture proposals (JAX school,
  MLIR school, nanopass/tinygrad school) implementing the verdicts.
- **J1–J3** — three adversarial judgments (maintainer, performance, pragmatist
  lenses). **All three independently selected P3 as the winning structure**,
  with a consistent list of grafts from P1 and P2.

This document is P3's structure with every judge-mandated graft applied, and
the accounting corrections the judges demanded. §10 is the ledger of what came
from where. Where this document and a research doc disagree, this document wins.

**Companion documents** (the numbered canon, `docs/design/README.md`): this file is
the master (010) and is kept current; detailed notes elaborate without
overriding — `020_implementation-plan.md` (steps, gates, the book);
`022_closure_specialization.md` + `024_dsl_caching_layer.md` (the pre-M0
evidence analyses: why numba can't, and the hazard checklist);
`030_deep-learning-notes.md` (the differentiable-programming satellite);
`040_combinators-notes.md` (pipelines, roles, the bracket config contract,
DPS/outputs — informing §2.11's bidirectional marshaling and the step-7
`ResultPlan`); `050_provenance_tracking.md` (source locations: the MLIR-lite
algebra, the rewrite inherit-default, the starting-region contract);
`060_rendering-notes.md` (static notebook widgets: fragment/style
composability, CSS-only interactivity, the jsdom dev loop);
`070_backends-notes.md` (the backend roster and taxonomy, bridge
ride-vs-own verdicts, the compute invocation surface, the graphics draw
surface, packaging/CI — informing §2.10's remaining columns and revising
§5's step-14 module-map line); `080_backend-organization.md` (families vs
targets vs cells, three-tier backend resolution, the `dsl.demo` special
case, the backends/ contribution contract); `090_core-and-extensions.md`
(the punning charter: dialect and runtime core+extensions conventions,
stdlib minimalism, the buffer/tensor-interop contract, the multi-device
testing ladder); `100_arrays-and-axes.md` (the array type algebra and its
two satellite refinements, the pedantic indexing decision incl. named
axes, array marshaling, statement policy, the C target, scope cuts);
`110_transforms-and-derivatives.md` (the SIMT-vmap spike finding, named
vmap, the tangent engine behind jvp and the in-kernel `D`, named
contraction); `120_events-and-instrumentation.md` (**proposal**: the event
seam that generalizes the ad-hoc cache counters and `no_compile()`,
structured sampled tracebacks, the `Memo` primitive, the §6 budget increase
it asks for, and the removal of `bench.py`'s monkeypatch).

---

## 0. The architecture in one paragraph

A **~1000-line kernel** of twelve frozen data structures and two engines (one
rewrite driver, one generic packer), in which the caching thesis is
**unrepresentable to violate**: the IR has no field that can hold a captured
value, so artifacts *can only* be keyed on types. Phase A (decoration) is pure
reflection — code object + closure cells + a memoized source snapshot — and
cannot fail or compile. Phase B lowers the function's AST once per type
signature through a fused typing+lowering pass into a typed region-based
micro-IR, rewrites it through declaratively-registered rule sets down a
legality-checked dialect ladder (`core.*` → `abi.slot` → backend source text),
and installs a **FastRecord** so that every subsequent call executes exactly:
**key build + guard compares + one dict probe + precompiled extract + pack +
launch** — no registry, no AST, no IR object on the hot path. Everything else —
syntax, batteries, types, backends, transformations, mini-languages — attaches
through **exactly five registration surfaces** on one explicit `Registry`, and
"a new capability lands with zero kernel diffs" is an enforced CI test, not an
aspiration.

```
                    frontends (satellites)
   Python AST lowerer      t-string sub-parsers      raw_kernel escape hatch
          \                       |                          |
           v                      v                          |
   ┌─────────────────────────────────────────────┐           |
   │ SURFACE: core.* (+ domain dialects,         │           |
   │   e.g. units.*, ein.* — fold to core)       │           |
   └─────────────────────────────────────────────┘           |
          | transform passes (grad/vmap: rule-matrix columns)|
          v                                                  |
   ┌─────────────────────────────────────────────┐           |
   │ MID: core.* only          (legal = {core})  │           |
   │   simplify + shared decompositions gated    │           |
   │   on the target backend's op set            │           |
   └─────────────────────────────────────────────┘           |
          | backend legalize_params RuleSet                  |
          v                                                  |
   ┌─────────────────────────────────────────────┐           |
   │ ABI: core.* ∖ {core.env} ∪ abi.slot         │           |
   │   (legal = {core, abi}); slot table ⇒       │           |
   │   PackPlan; Node.key ⇒ artifact-cache key   │           |
   └─────────────────────────────────────────────┘           |
          | backend render (code_for_op tables)              v
          v                                            source string
   WGSL / CUDA-C / Metal / C / Python source  ──►  backend runtime compiles + launches
```

Every arrow is a `RuleSet` run by the same rewrite driver; every box boundary
is a machine-checked legality declaration; the type-keyed cache sits **above**
this whole picture — the ladder runs only between a cache miss and a
`FastRecord`.

---

## 1. The five decisions (from the verdicts)

| # | Question | Decision | Why (one line) |
|---|---|---|---|
| D1 | Frontend | **Reflection capture (phase A) + AST lowering (phase B).** No bytecode, no tracing for acquisition. | Bytecode = permanent CPython treadmill (numba: 268 version forks; Dynamo: 71 branches in one file); tracing can't see data-dependent control flow and invites const-baking (JAX's documented ~240× retrace pathology). Every kernel DSL studied (cupy.jit, Triton, Taichi) is AST-based. Source-unavailable fails loudly at phase B with a per-backend `raw_kernel` escape hatch. → `V1-frontend.md` |
| D2 | IR | **Purpose-built micro-IR (~650 lines): one immutable `Node(op, type, args, attrs, regions)`, structured control flow as exactly three region ops (`if`/`for`/`call`), memoized content hash, MLIR-flavored printing.** Not xDSL. | xDSL fits conceptually but is 146 kLOC of 0.x churn contributing nothing to our novelty; the MLIR *concepts* (regions, dialect namespaces, value/attr split, legality) are stolen wholesale. MLIR-flavored text keeps later migration a refactor. Pinned xDSL survives as an optional dev-time differential oracle. → `V2-ir.md`, `R3-xdsl.md` |
| D3 | Hooks | **One explicit `Registry` + an op×aspect rule matrix, exposed through exactly five registration surfaces** (ops/rules, `@overload` batteries, type extensions, backends, transformations). No module-level registries. | numba's `@overload` (batteries written in the DSL subset, compiled per target — 491 portable vs 272 hand-lowered) is the proven batteries economics; JAX's rule matrix is the proven transformation economics; both fit one registry. → `V3-hooks.md` |
| D4 | Marshaling | **ValueKind (typeof/leaf_types/flatten/fingerprint per Python type) → backend-owned PackPlan of Slots → per-entry FastRecord.** Two-tier keying: code-changing concerns key the artifact; bytes-changing concerns (concrete units) key a cheap pack-plan memo. | numba's datamodel/ArgPacker and JAX's pytrees agree on the shape: logical leaves declared once, physical spelling backend-owned. Units: dimension-in-Type, unit-in-converter — unit tweaks must never recompile (unxt's retrace hazard avoided). → `V4-marshaling.md`, `R8-user-types.md` |
| D5 | Transforms | **IR-to-IR passes whose per-op content is rules (`jvp`/`transpose`/`batch` columns) in the same registry; `grad(f)`/`vmap(f)` mint `Derived` template identities flowing into the unchanged specialization cache.** Backend-native AD only as a `custom_vjp`-shaped escape hatch and test oracle. | JAX's rule matrix as content model, tinygrad's rewrite passes as execution model (tinygrad's whole AD is 132 lines of rules). Derived identities make `grad(f)` rebuilt per frame an ordinary cache hit. → `V5-transforms.md` |

---

## 2. Core primitives

Twelve data structures. Everything else in the system is a function over these.
(Field lists are normative in shape, not in name.)

### 2.1 `Type` — the structural lattice (`kernel/types.py`, ~65 LOC)

Frozen, slotted, structurally hashed, serializable from day one. Honest widths
(`int` buckets to i64/u64/bigint-error); narrowing is a backend `type_map`
decision, never a `Type` property.

```python
class Type: ...                                     # frozen subclasses only
class Scalar(Type):   kind: str                     # "f64" "i64" "u64" "bool" "f32" "i32" "u32"
class Vec(Type):      elem: Scalar; n: int          # IR-level only: typeof NEVER produces Vec
class Tuple(Type):    elems: tuple[Type, ...]       # the honest tuple summary (see §10 ledger, 2026-07-11)
class Array(Type):    dtype: Type; ndim: int; layout: str; byteorder: str; writeable: bool
class Record(Type):   name: str; fields: tuple[tuple[str, Type], ...]
class FnType(Type):   template: TemplateId; env_types: tuple[Type, ...]    # the thesis
class LiteralType(Type): base: Type; value: Hashable   # the ONE value-in-type opt-in
# reserved (no schema change later): Quantity(Type): rep: Type; dim: Dim
```

### 2.2 `TemplateId` — code identity as a sum type (~15 LOC)

```python
class Base(TemplateId):    code: types.CodeType     # VALUE-compared (CPython), never id()
class Derived(TemplateId): tag: str; base: TemplateId; static_params: tuple[...]
```

`grad(f)` and `f` can never collide; `grad(f)` rebuilt per iteration hits.

### 2.3 `SourceSnapshot` (~10 LOC) — `(text, filename, firstlineno, qualname)`
taken at decoration while `linecache` is coherent; memoized per code object. On
a phase-B miss, `compile(text)` must be value-equal to the template code or we
raise — a stale-source compile is impossible.

### 2.4 `Handle` — a first-class DSL closure (~25 LOC)

```python
class Handle:
    fntype:   FnType                    # (TemplateId, env_types) — structural identity
    env:      tuple[object, ...]        # capture VALUES, co_freevars order; never keyed
    env_fp:   tuple[Hashable, ...]      # precomputed structural fingerprints (hot key part)
    snapshot: SourceSnapshot
    kind:     str
    registry: Registry
    def __call__(self, *args): ...      # the hot path (§4.3)
```

### 2.5 `Node` / `Region` — the entire IR (`kernel/ir.py`, ~115 LOC)

```python
class Node:                              # frozen; a node IS its SSA value
    op:      str                         # dialect-namespaced: "core.add", "abi.slot", "wgsl.frag_coord"
    type:    Type                        # exactly one result type
    args:    tuple[Node, ...]
    regions: tuple[Region, ...]          # nonempty only for core.if / core.for / core.call
    attrs:   tuple[tuple[str, Hashable], ...]   # compile-time constants — INSIDE structural identity
    loc:     Loc | None                  # EXCLUDED from identity
    key:     bytes                       # memoized content hash over (op,type,attrs,args,regions)

class Region:
    params: tuple[Node, ...]             # typed "core.param" binders
    body:   tuple[Node, ...]             # ordered; last is core.yield
```

**The load-bearing negative invariant:** no field reachable from `Node` can
hold a captured value. A runtime capture is `core.env(attrs=(("slot",k),))`; a
`Literal`-lifted capture is `core.const` with the value in `attrs`, entering
`Node.key` and visibly auditable in printed IR. numba's
`ir.FreeVar(idx, name, value)` — the anti-pattern — is excluded *by
construction*, and CI greps that no kernel type reachable from `Node` has an
`object`-typed field except `attrs`.

Exactly **three region ops** (`core.if`, `core.for`, `core.call`),
constitutionally: any proposed fourth is priced at ~180 lines × live transform
columns (autodidax-measured) before acceptance.

### 2.6 `OpDef` — ops as data, never Node subclasses (`kernel/ops.py`, ~45 LOC)

```python
class OpDef:
    name: str; type_rule: Callable; traits: frozenset[str]; nregions: int = 0
```

Core dialect ≈ 30 ops: arith, compare, select, `vec`/`extract`/`field`, `env`,
`const`, `param`, `yield`, `if`, `for`, `call`, boundary-only `load`/`store`,
`cast`. A dialect is a dict of OpDefs; installing one is a dict merge.

### 2.7 `Pat` / `RuleSet` / `Stage` — rewrites + legality (`kernel/rewrite.py`, ~150 LOC)

The **only pass mechanism in the system** (tinygrad's PatternMatcher shape:
915 declarative rules over one ~150-line driver is the existence proof).

```python
Pat      = (op | ops, arg sub-patterns, attr guard)
RuleSet  = list[tuple[Pat, Callable[[Builder, Match], Node | None]]]
Stage    = (name: str, rules: RuleSet, legal: frozenset[str])   # legal OP NAMESPACES at output

def run_stage(region, stage, ctx):
    out = rewrite(region, stage.rules)      # post-order, rebuild-on-change, fixpoint
    check_legal(out, stage.legal)           # ~30-line namespace check; error names op + loc
    return out
```

Stage legality is **always on** (P2 graft): progressive lowering is only
auditable if "which dialects may exist after stage N" is machine-checked; the
check is O(ops) string-prefix tests — free at our scale.

### 2.8 `Registry` — the one extension object (`kernel/registry.py`, ~60 LOC)

```python
class Registry:
    ops:        dict[str, OpDef]
    rules:      dict[tuple[str, str], Callable]   # (op, aspect); aspects: "lower_ast",
                                                  # "eval", "jvp", "transpose", "batch", "unit"
    overloads:  dict[Hashable, list[tuple[type, Callable]]]     # target-token MRO selected
    methods:    dict[tuple[type, str], list[tuple[type, Callable]]]
    valuekinds: dict[type, ValueKind]
    backends:   dict[type, Backend]
    transforms: dict[str, Transform]
    generation: int
    def extend(self) -> Registry: ...             # ChainMap layering: stdlib → user → session
    def install(self, dialect: Dialect): ...      # pure sugar: a bundle of surface entries
```

No module-level registries anywhere; the stdlib populates `DEFAULT` through the
same five public surfaces users get. Target tokens are a plain class lattice
(`Generic → CPU/GPU → WGSL/CUDA/Metal/PyEval/C`) resolved by MRO. `Dialect` is
a bundling value with a test asserting it adds **no** resolution semantics
(P2 graft).

### 2.9 `ValueKind` — one registration, three views (`kernel/valuekind.py`, ~70 LOC)

```python
class ValueKind(Protocol):
    def typeof(self, v) -> Type            # full structural lattice (hazard-doc corners)
    def leaf_types(self, t: Type) -> tuple[Leaf, ...]   # STATIC: derivable from Type alone
    def flatten(self, v) -> tuple[object, ...]          # DYNAMIC: hot path; CI allocation budget
    def fingerprint(self, v) -> Hashable                # cheap structural tag
```

Closed, core-owned leaf vocabulary backends are total over: `ScalarLeaf(kind)`,
`BufferLeaf`, `ShapeLeaf(axis)`, `StrideLeaf(axis)`, `EnvLeaf` (recursive).
Kernel ships scalar/tuple/Handle kinds; ndarray lives in `stdlib/` (kernel has
zero NumPy dependency). **Fingerprint soundness is a CI property fuzz**
(P1 graft): `fingerprint(a) == fingerprint(b) ⟹ typeof(a) == typeof(b)` —
fingerprint collisions are silent wrong *hits*, the worst failure class.

`typeof` is a **summary function, not a class lookup** — how rich a summary
(rank-only vs shape-in-type vs value buckets) is the per-kind dial; see §13.

### 2.10 `Backend` — a capability record (instances live outside the kernel)

```python
class Backend:
    token:       type                        # TargetToken subclass
    type_map:    dict[Type, str]             # honest type → backend spelling (narrowing HERE)
    code_for_op: dict[str, Callable | str]   # intrinsic table; keys() = capability declaration
    extra_rules: RuleSet                     # backend legalizations (incl. legalize_params)
    render:      Callable[[Region, Backend], str]   # typed IR → source text
    runtime:     Runtime                     # plan(types)→PackPlan; compile(src)→Artifact;
                                             # make_launcher(artifact, plan)→Callable
    params_key:  Callable[..., Hashable]     # backend params that enter the specialization key
```

Shared decompositions (`sqrt → pow(x,.5)`, `mean → sum/len`) are gated on
`code_for_op.keys()`: a backend naming the op natively skips them; one that
doesn't gets them free. Budget per source-emitting backend: 50–300 renderer +
130–220 runtime lines (tinygrad-calibrated: WGSL 115, Metal 52, CUDA 78).

### 2.11 `LeafPath` / `SlotSpec` / `PackPlan` (`kernel/pack.py`, ~80 LOC)

```python
class LeafPath: root: str; index: int; sub: tuple[int, ...]
class SlotSpec: source: LeafPath; convert: Affine | None; dest: PhysicalDest
class PackPlan: slots: tuple[SlotSpec, ...]; staging_size: int
                # interpreted by one ~30-line generic struct.pack_into loop
```

Built once per cache entry **from types alone**; `PhysicalDest` vocabulary
(`UniformSlot(offset,fmt)`, `KernelArg(index,ctype)`, `CField(offset,ctype)`,
`PyArg(index)`) is backend-owned and never appears in kernel types. Physical
slots are also represented as `abi.slot` IR ops at the ABI stage (P2 graft), so
the marshaling decision is printable and golden-testable.

### 2.12 `FastRecord` — the compiled cache entry (`kernel/cache.py`)

```python
class FastRecord:
    artifact: object          # pipeline / RawKernel / cfunc / exec'd Python fn
    guards:   tuple           # precomputed (cell_or_dict, name, expected) identity pairs (P1 graft)
    extract:  Callable        # closure-compiled: (env, args, staging) → buffer_leaves
                              #   byte-packable leaves written into staging directly (fused);
                              #   buffer-class leaves returned as a tuple (P1+P2 grafts merged)
    plan:     PackPlan
    staging:  bytearray       # preallocated, reused per call
    launch:   Callable        # (staging, buffer_leaves) → run   (P2 graft: the leaves channel
                              #   is how fresh buffer pointers/shapes reach CUDA/Metal per call)
```

Installed atomically under a per-key future (reentrant; recursive entries
published forward-declared — hazard doc).

---

## 3. Core hooks — the five surfaces

All registration happens at import/compile time; **the hot path is
registry-free**. The five decorator signatures are the only stable contract;
IR internals, renderer internals, and registry storage are explicitly unstable.

**A — ops and rules (the matrix).**
`defop(name, type_rule, traits, nregions)` + `@rule(op, aspect)`. Aspects are
columns: `"lower_ast"` (how the accepted Python subset widens — a registration,
not a lowerer edit), `"eval"`, `"jvp"`, `"transpose"`, `"batch"`, `"unit"`.
Columns are declared-empty day 1; `MissingRule(op, aspect, loc)` names exactly
what to register.

**B — batteries via `@overload` (numba, stolen wholesale).**
`@overload(math.sqrt, target=Generic)` returns a DSL-subset `impl` selected on
argument *types*; compiled through the same `(FnType, arg_types)` cache as user
code; target-token MRO picks the most derived registration. One call-typing
resolution order, implemented once: `__pdum_dsl__`-convertible → overloads
(MRO-filtered) → methods/attrs (type-MRO) → error naming callee, arg types, and
every declined entry.

**C — type extensions (three registrations).**
`@valuekind(Color)` (value→Type+leaves+fingerprint), `@overload_method` /
`@overload_attribute` (methods in the DSL subset, erased to free functions —
which WGSL requires anyway), and rarely a `lower_ast` rule for novel syntax.
Logical leaves only — physical spelling is the backend planner's job (the rule
whose violation killed numba's datamodel extensions twice).

**D — backends.** `register_backend(Backend(...))`. A backend needing more
than its line budget indicates a missing shared decomposition rule, not a
bigger backend.

**E — transformations.** `Transform(tag, passes)` + the `custom_jvp` /
`custom_vjp` escape hatch (same shape for user functions and backend-native
gradients, e.g. MLX). `grad(f)`/`vmap(f)` return ordinary `Handle`s with
`Derived` identities.

**Worked examples** (full walkthroughs in `research/P3-nanopass-school.md` §2):
adding `sqrt` ≈ 15 lines, zero kernel files; a `Color` record with `to_oklab()`
≈ 40 lines via surfaces B+C, captured Colors become three uniform slots on WGSL
and three struct fields on C automatically; the C backend ≈ 350 lines via
surface D with every Generic battery working immediately; `grad` = a
`transforms/ad.py` package via surface E plus per-op rule one-liners.

---

## 4. The dataflow

### 4.1 Phase A — capture (every closure construction; inside the user's loop)

```python
def make_handle(fn, kind, registry):
    code   = fn.__code__
    snap   = _SNAPSHOTS.get(code) or _take_snapshot(fn)          # memo per code object
    vals   = tuple(safe_cell(c) for c in fn.__closure__ or ())   # co_freevars order
    env_fp = tuple(registry.fingerprint(v) for v in vals)        # structural tags, memoized
    fntype = _FNTYPES.get((code, env_fp)) or _build_fntype(...)  # full typeof on memo miss
    return Handle(fntype, vals, env_fp, snap, kind, registry)
```

No parse, no IR, no compile — phase A **cannot fail** (missing source is
phase B's loud `NoSourceError`, with remediation text and the per-backend
`raw_kernel` escape hatch). Cost ~1–2 µs.

### 4.2 Phase B — the HIT path (every hot-loop iteration; ~15 lines total)

```python
def __call__(self, *args):
    key = (self._key_head,                    # memoized (template_fp, env_fp) — P2 graft
           _fp_tuple(args), _BACKEND_FP, self.registry.generation)
    rec = _RECORDS.get(key)                                   # ONE dict probe
    if rec is None or not _guards_ok(rec.guards):             # N pointer compares — P1 graft
        rec = _miss(self, args, key)                          # refuse-or-recompile, never stale
    buffer_leaves = rec.extract(self.env, args, rec.staging)  # fused reads + struct.pack_into
    return rec.launch(rec.staging, buffer_leaves)             # write_buffer+draw / kernel launch
```

**That is the entire per-call cost:** key build + guard compares + one probe +
extract/pack + launch. No registry access, no typeof lattice walk, no AST, no
IR object exists on this path. Budget: single-digit µs pure Python
(CI: alarm 5 µs, fail 10 µs — P2 thresholds), with two pre-shaped,
contract-preserving escalations (exec-generated per-template binder à la
Triton, then a narrow native fastpath à la JAX).

On the native (Rust/C++) escalation, for calibration (2026-07-11): generic
fingerprint/traversal work is bound by the CPython object protocol, so a
native port of the *same walk* buys ~2–5×, not orders of magnitude — the
exec-generated binder (which eliminates the generic walk) comes first. The
place native genuinely wins big is array-leaf fingerprints via the NumPy C
API (struct reads vs. attribute protocol). All of it is decided by the step-9
microbench gate with profiles; the hit-path contract stays frozen so
escalations are transparent.

### 4.3 Phase B — the MISS path (once per type signature)

1. Full `typeof` of env+args (verify fingerprint, intern types).
2. **Snapshot coherence:** `compile(snap.text)` value-equal to the template
   code (modulo filename/lineno) or raise.
3. `ast.parse` with real filename; select `FunctionDef`; drop decorators.
4. `classify_names` — closed fate taxonomy: param | `core.env` slot | intrinsic
   | Handle callee (FnType-typed) | allowed folded const | error-or-Literal-lift.
   Dependency-closure tags recorded → become `FastRecord.guards`.
5. **Fused typing+lowering** forward pass dispatching on `lower_ast` rules and
   the overload resolution order → flat, fully-typed core-dialect program;
   every node carries `loc`.
6. If `Derived`: run the transform's passes (rule-matrix driven).
7. `run_stage` ladder: simplify + gated decompositions (legal = {core}) →
   backend `legalize_params` splitting each logical `core.env`/arg into
   physical `abi.slot` ops (legal = {core, abi}). Per-frame flatten is now
   structurally impossible (M0's fault cured).
8. **Artifact-cache probe** on `(Node.key, backend.token, codegen_flags)` — a
   hit skips render+compile entirely.
9. `backend.render` → source; `runtime.compile` → artifact.
10. `runtime.plan(types)` → PackPlan; `build_extractor(leaf_paths)` →
    closure of pure attribute/index reads; guards precomputed.
11. `FastRecord` installed under a per-key future.

### 4.4 Caches and keys

```
specialization cache    (template_fp, env_fp, arg_fp, backend_fp+params, generation) → FastRecord
   | miss only
artifact cache  (Node.key content hash, backend.token, codegen_flags) → Artifact
   | miss only
render + backend compile        (+ tier-2 pack-plan memo for units/byte options)
```

- Template identity = code object **by value** (unchanged notebook re-run hits;
  edit misses) wrapped in `TemplateId`.
- The fast key must contain **every** component of the full key — enforced by a
  perturbation test: mutate each declared key dimension one at a time and
  assert the *named tier* of miss (P2 graft).
- Dependency drift (frozen globals, rebound helpers) is checked per call via
  `FastRecord.guards` — refuse or recompile, **never silently stale** (P1
  graft; fixes the value-equal-code/different-globals hazard).
- Two-tier law: every future feature must declare *which tier misses when this
  changes* as its first design-review question.
- `cache.py` includes LRU eviction + retirement of superseded templates from
  day 1 (hazard-doc L-cache leak; judge-mandated, unbudgeted in all three
  proposals — budgeted here).
- Disk cache (later): structural keys only — `TemplateId` lowers to
  `(filename, qualname, source_hash)`; PackPlans rebuilt, never persisted.

---

## 5. Module map (honest accounting)

### The kernel — CI-capped: target ≈1000, hard cap 1150

| Module | LOC | Responsibility |
|---|---|---|
| `kernel/types.py` | 65 | Type lattice + TemplateId, frozen/serializable |
| `kernel/capture.py` | 85 | phase A: make_handle, safe_cell, snapshot memo, Handle |
| `kernel/valuekind.py` | 70 | ValueKind protocol + scalar/tuple/Handle kinds + fingerprints |
| `kernel/ir.py` | 115 | Node, Region, content hash, Builder, structural verifier |
| `kernel/ops.py` | 45 | OpDef, traits, ~30-op core dialect table |
| `kernel/rewrite.py` | 150 | Pat, RuleSet, the one driver, **Stage legality (always-on)**, match log |
| `kernel/lower.py` | 135 | phase B driver: coherence check, classify_names, fused typing+lowering dispatching on `lower_ast` rules |
| `kernel/registry.py` | 60 | Registry, DEFAULT, token lattice + MRO, Dialect install |
| `kernel/cache.py` | 105 | two-tier cache, FastRecord, **guards**, per-key futures, generation, **LRU/retirement**, per-tier miss counters, no_compile mode |
| `kernel/pack.py` | 80 | LeafPath/SlotSpec/PackPlan, generic packer, build_extractor |
| `kernel/printer.py` | 60 | MLIR-flavored textual form (golden tests; migration insurance) |
| `kernel/api.py` | 50 | @jit, Handle.__call__ hot path, NoSourceError, MissingRule |
| **kernel subtotal** | **1020** | |
| `backends/python.py` | 110 | day-1 backend: render to Python source + exec/PyArg runtime — reference semantics, zero deps |
| **in-budget total** | **1130** | CI line check + PR delta bot from day one |

**The honesty clause** (P2 graft, all three judges): the kernel's `lower.py` is
a *driver*; the `lower_ast` **rule content** that gives the language its width
is a separately CI-counted satellite bucket (`stdlib/lower_rules/`) budgeted at
**800–1500 lines at maturity** (V1's calibration). The 1020-line kernel figure
is credible *because* that bucket is on the books, not hidden.

### Satellites (attach via the five surfaces; zero kernel diffs — CI-enforced)

| Component | LOC est. | Surface |
|---|---|---|
| `stdlib/` (lower_ast rule packs; ndarray ValueKind; math/vec/swizzle overloads; ~10 batteries) | 800–1500 | A/B/C |
| `backends/wgsl/` (renderer ~150 + uniform planner & wgpu runtime ~200 — M0's `layout.py` generalized) | ~350 | D |
| `backends/c/`, `backends/cuda/` (CuPy RawKernel), `backends/mlx/` | ~300–400 each | D |
| `transforms/` (vmap ~100; jvp ~100; transpose ~450; rule packs) | ~800 | E + columns |
| `units/` (Dim, Quantity, unit column, Affine converters) | ~250 | A/C + PackPlan |
| `tstring/` (einops-like mini-languages; PEP 750) | per language | frontend sugar → core dialect |
| `tools/` (xDSL differential oracle, golden harness, stage grammars, attr lint) | ~350 | consumes printer output |

---

## 6. The CI constitution (budgets are architecture)

The prime directive made mechanical — all gates land with the vertical slice,
not later:

1. **Kernel line cap** (1500 as of 2026-07-13; see the ledger) + per-file
   caps + per-backend caps (≤300 render, ≤220 runtime) + PR line-delta bot.
   **The policy, restated after the cap did its job** (the kernel froze at
   1147 across two satellite-only steps): caps are **tripwires for a
   conversation, not walls**. Crossing any cap requires a ledger entry
   stating what the lines bought; the test stays exactly as strict as ever
   (a soft-warn gate is a dead gate) — only the numbers move, deliberately.
   And one virtue is INVERTED now that it has bitten twice: a satellite
   that needs visibility or a hook must **ask for a seam** (ledger entry,
   cap negotiation) — monkeypatching live kernel state to avoid a
   negotiation is what is forbidden. "Zero kernel edits" remains the
   default posture for FUNCTIONALITY; it was never meant to price
   OBSERVABILITY out of the kernel (design 120 §1.3 is the case study).
2. **Extension-locality test:** adding `sinh`, a record method, and a new
   statement form must produce **zero kernel diffs**. Run in M1, not M5.
3. **Thesis test:** 300 frames of the disk demo with moving captures ⇒
   `compiles == 1`, under `no_compile` assertion mode after frame 1.
4. **Hit-path microbenchmark:** alarm 5 µs / fail 10 µs; allocation budget on
   `flatten`.
5. **Perturbation key test:** every declared key dimension mutated ⇒ the named
   tier misses.
6. **Fingerprint-soundness fuzz** (collisions are silent wrong hits) +
   randomized Python-vs-WGSL differential runs.
7. **Backend-seam differential:** WGSL readback matches the Python backend
   image within tolerance.
8. **Attr lint:** only `LiteralType`-originated constants may appear as
   `core.const` attrs in printed IR.
9. **Anti-pattern grep:** no kernel type reachable from `Node` has an
   `object`-typed field except `attrs`.
10. **Golden printed IR** at each stage boundary (post-lower, post-legalize).

Review heuristics with teeth: any proposed kernel component that is neither a
primitive nor one of the two engines is a registration in disguise (P1); any
proposed fourth region op costs ~180 lines × live transform columns, priced
before acceptance (P1); every feature declares its cache tier (V4).

---

## 7. Day-1 vertical slice

The orbiting-disk demo (`reference/demos/disk.py`) reproduced on the new kernel with
**both** the WGSL backend and the Python backend — proving the backend seam,
the specialization cache, and the hot path in one milestone (≈1130 in-budget + ~350
WGSL + ~120 stdlib slice):

1. `@jit(kind="fragment")` → Handle; the demo loop rebuilds it every frame.
2. Language subset as `lower_ast` rules: float arith, compare, `IfExp`,
   tuple-return → `core.vec`, assignment, swizzle via `@overload_attribute`,
   `FragCoord` as a WGSL-dialect op registered *from the backend package*.
3. Python backend renders the fragment to a function; 64×48 CPU reference image.
4. WGSL backend: renderer + uniform planner + wgpu runtime; `cx, cy, radius` →
   three ScalarLeafs → one packed uniform buffer.
5. All ten CI gates from §6 enforced from this milestone.
6. Explicitly deferred: `core.for`, arrays, transforms, other backends — each
   lands later through a surface, never a kernel edit. **That is the test of
   the architecture.**

---

## 8. Milestones as risk retirement

| M | Ships | Retires (detector from §9) |
|---|---|---|
| M1 | vertical slice + extension-locality test + ~10-battery WGSL port | lower.py-monolith risk; batteries-economics risk (count forked lines; numba's portable ratio ≈2:1 is the floor) |
| M2 | arrays + `core.for` + C backend (cheapest second real backend) + ray-march spike + **vmap-over-if/for spike on the Python backend** (>350 lines ⇒ re-hear tinygrad direct-VJP — P1 graft) | three-region-ops-too-weak risk; planner-vocabulary risk; transform-taxation risk |
| M3 | vmap + jvp columns | rule-matrix economics |
| M4 | transpose/grad (~450 lines, its own milestone) | AD architecture |
| M5+ | CUDA/MLX backends, units (after the first Quantity user exists), t-string einops, disk cache | — |

---

## 9. Risks and detectors

1. **Hot-path cost creep** (fingerprints grow structural detail). Detector:
   microbench + allocation gates; pre-shaped escalations preserve the contract.
2. **`lower.py` becomes the monolith** (numba's `typeinfer.py` failure mode).
   Detector: per-file cap + extension-locality test in M1.
3. **Three region ops prove too weak** (ray-marching wants early exit).
   Detector: M2 spike; decide *then* between an early-exit `core.for` variant
   (priced) vs frontend rejection.
4. **WGSL shrinks the portable battery layer** (no i64, no recursion,
   uniformity rules) below numba's ~2:1 portable ratio. Detector: M1 gate.
5. **Cache-key incompleteness** — the invisible failure class. Detector:
   perturbation test + per-tier miss counters + guards (refuse-or-recompile).

---

## 10. Grafts & deviations ledger

Adopted from **P1 (JAX school)**: `FastRecord.guards` on the hit path;
fingerprint-soundness fuzz; fused extract-writes-staging; the fourth-region-op
price list; the pre-M2 vmap spike with quantified fallback; the
"registration in disguise" review question.

Adopted from **P2 (MLIR school)**: always-on Stage legality; the `abi.slot`
ABI stage (marshaling printable/golden-testable); the `launch(staging, leaves)`
buffer channel; honest lowerer accounting (the satellite rule-pack bucket);
the perturbation key test; the attr lint; `Dialect` as pure bundling sugar;
memoized key head; microbench thresholds.

Synthesizer's additions beyond the judges: LRU eviction + template retirement
budgeted into `cache.py` day 1 (the judges flagged it cross-cutting and
unbudgeted); `extract` returns buffer leaves while packing bytes in place
(merging P1's fusion with P2's channel so neither scalars nor buffers pay for
the other).

**2026-07-12 (step 7): marshaling landed bidirectional; `aspect` unifies the
Type-keyed registries.** Three deviations from §2.9/§2.11, all review-driven:
(1) the static view is keyed by **Type**, not by value class (`FnType` is
produced by both `Handle` and `Pipeline` — a genuine many-to-one), so
`leaf_types` became `KindTable.register_aspect("leaves", …)`; `child` (descend
one step) and `rebuild` (reassemble a result) joined it as aspects, replacing
what began as a module-global rebuild dict — one MRO lookup, one layering
story, `extend()` copies them all. (2) `flatten` is now a **required** third
`ValueKind` view, checked loudly at `register()` rather than deep inside a
packer loop. (3) `build_extractor` compiles the plan's leaf paths into one
getter per slot (§4.3.10 as written) — the first draft re-dispatched through
`flatten` every frame, which is M0's per-frame walk wearing a new hat; the
alignment law (`flatten` ≡ compiled getters) is now a test, and the fuzz packs
each leaf with its *declared* format so a bool↔i64 drift can't hide behind
`bool`'s int subclassing. `pack.py` cap raised 150 → 175 consciously (the
output half + the compiled extractor were not in the §5 estimate).

**2026-07-12 (step 10): the five surfaces live — THE KERNEL IS FINISHED at
exactly 1150/1150.** Surfaces: A `registry.ops` + `defop`; B
`registry.overloads` (intrinsic name→op; `@overload` = capture-free
DSL bodies inlined per call site; record methods keyed `(RecName, meth)`);
C `@record` dataclasses (Record type + kind + methods — the ch07a jitclass
story); D `code_for_op` spelling tables, whose KEY-presence gates shared
decompositions (one mechanism also retires tuples on targets that cannot
spell them); E `extend()` layering + `load_entry_points` (080's contract).
Base pack widened 10→15 forms (aug-assign, short-circuit and/or via core.if
with lazy branches, chained compares sharing operands, tuples+unpack+
subscript, attributes). Deliberate deviations: overload target-token MRO
DEFERRED until a battery body must differ per target (decompositions cover
today's cases); `@overload_attribute` and `to_oklab` deferred (want vec
math); the lowering rules dict doubles as the CONTEXT DOOR
(`"__registry__"` planted by `_build` per build — extend()-safe). Found
live: `f16` as an Env member name is reserved WGSL (offset-16 member) —
prefix is now `m`. Batteries: 8 intrinsic ops × 2 target spellings vs 10
DSL-written portables; every new target pays ~8 spellings and inherits the
portables free — the numba-2:1 economics measured in ch11.

**2026-07-12 (ch10 walkthrough): backend organization settled — 080.**
User-caught naming debt: the vertical-slice implementations were claiming
the real backends' names ("python", "wgsl-*"). Settled: kinds are declared
by FAMILY packages, targets by backend packages, routing binds thin CELLS
(family × target, sparse, holes loud); backend resolution is three-tier
(data-driven via the device axis in Array types → explicit override →
routed default; no activation API). The ch09/ch10 pair moved to
`pdum.dsl.demo.simple_shader` with dotted cell names
(`demo.simple_shader.python`, `demo.simple_shader.wgsl.{compute,fragment}`)
— fused family+target on purpose, which is exactly why they left
`backends/`: that package is now a PEP 420 namespace (no __init__) — the
contribution point, entry-point group `pdum.dsl.backends` specified in 080,
implemented at step 10. Rename was free TODAY (fps live only in-process);
it would not have been after disk persistence. `families/` is deliberately
NOT created until the second compute target forces it (step 14). Same
hygiene for KIND strings: the demo registers `simple_shader.compute`/
`simple_shader.fragment`, reserving plain `compute`/`fragment` for the real
families; `device` keeps its name (stdlib's, semantics already final).

**2026-07-12 (step 9): M1 COMPLETE — the WGSL backend, compute-led.** Two
backends, one IR, the thesis measured on both: 119 fresh closures + a
mid-loop RESOLUTION change on the GPU = zero recompiles (the domain rides
the leaves channel via `out=`; `@workgroup_size` bakes into the artifact
text exactly as §3b/spec require); differential gate GPU-vs-Python
branch-exact on the disk, <5e-6 on smooth kernels (f64 vs f32 seam);
~1.8 ms/frame dominated by the deliberate synchronous readback (render
loops pay only write_buffer + encode). Landed: per-role routing
(`Registry.routes`, kinds ship WITH their backends), the §2.10 Backend
columns `plan`/`param_types`/`make_launcher` (python takes every default),
`out=` as launcher data through dispatch. Deviations, all deliberate:
compute-family contract v1 = params ARE thread coordinates and the call
passes the domain (explicit out-BUFFERS wait for ndarrays); fragment
broadcasts its scalar to grayscale rgba (colors wait for tuples, step 10);
workgroup size fixed (64 / 8×8) until the bracket-schema surface;
fragment renders offscreen only (the `draw(target)` window surface is
070 §4's committed design, next graphics step). GPU cells in the book are
`gpu`-tagged: the harness probes and skips without an adapter, committed
outputs baked on the M3 survive (R17's nbclient pattern; the probe is
three-state — present / absent / BROKEN-fails-loudly). Step-9 review
(medium, 7 angles, 15 findings, 11 fixed): Env struct members now follow
the slot FORMAT (the f32-only first draft reinterpreted int/bool capture
bits as float garbage) and are generated from the PLAN, hole-free (WGSL
lays members sequentially — a folded capture would have shifted every
later member's bytes); non-finite floats and over-i32 int constants refuse
loudly; `out=` rides the leaves channel in an `Out` TAG (peeled by type,
never tail position — no ch12 buffer-leaf collision); a kind-scoped
backend registration no longer claims the default slot; positional args on
derived-param kernels refuse; bind groups/dims/views cached at
domain-change tier per R12's ranking; and the dominator-placed emission
walker is now ONE shared module (`backends/_emit.py`) under both
renderers — the house rule's third-copy trigger, honored.

**2026-07-12 (backend detour, pre-step-9): bridges owned, invocation
surface committed.** Six-agent research fan-out (research/R12–R17; synthesis
`070_backends-notes.md`). Decisions: (1) **own the CUDA stack via
cuda.core** (1.0-stable; opt-in caching keeps our two tiers the only
authority; `cuLaunchKernel`'s `void**` params admit a once-per-FastRecord
pointer table INTO staging — the purest `launch(staging, leaves)` of any
target) — §5's "backends/cuda/ (CuPy RawKernel)" line is superseded, cupy
demoted to optional fallback/allocator; (2) **own the Metal stack**
(ctypes-objc/PyObjC; MLX's name-keyed never-recheck kernel cache is
incompatible with a content-addressed artifact tier, `mx.eval`-per-call
blows the hit budget, and `setBytes`/`setBuffer` ARE staging/leaves; MLX
reserved as an optional thin satellite); both rides failed for the same
three reasons — caching, marshaling, scheduler — which are the three things
this framework exists to own. (3) Step 9 leads with COMPUTE, fragment as a
thin same-step variant (workgroup_size is pipeline-creation-time, confirming
block-in-artifact-key; compute exercises every marshaling contract with the
fewest parts; ch08's staging ABI verified live on the M3's Metal backend).
(4) Invocation: explicit-DPS `out=` returning destinations, out-shape RULES
(registration, never program analysis), config schema `[grid, block, smem,
stream]` through the §3c pipeline (block value-specializes — WGSL forces
it; grid/smem/stream strip to the leaves channel; smem refused on WGSL),
ping-pong chaining as the `orchestrate` tag's encode plan. (5) Shader-family
dialect layering confirmed (R15): shader-core / compute-family (+capability
flags) / fragment-family / per-target packs; numeric policy legislated in
core (trunc div-mod, twin-raises on div-zero, no NaN-as-data).

**2026-07-12 (step 8): first execution — Registry v1, Python backend, the
hot path.** The thesis is now a *measurement*: 299 fresh closures under
`no_compile()`, zero compiles, ~5–7 µs/frame INCLUDING full phase-A rebuild;
fused pipelines dispatch through the same path (199 fresh pipelines, zero
compiles); the content-addressed tier proven end-to-end (identical bodies
from different def-sites: two specializations, ONE artifact; generation bump
leaves tier 2 untouched). Deviations from the written design, all deliberate:
(1) `Backend` v1 carries only {name, render, compile, fp} — `type_map`,
`code_for_op`-gated decompositions, and `params_key` wait for their first
consumer (WGSL); (2) the Python backend passes scalar *arguments* through
the staging buffer alongside captures — deliberately uniform-block-shaped so
the ABI is proven on CPU, not the fastest local call; composite args stay
refused until the arg-side normalize (arrays step); (3) `core.if` renders as
eager-both-branches + conditional expression (pure ops; the fragment-shader
execution shape anyway); (4) guards are per-captured-cell identity triples
`(cell, "cell_contents", value)`, and they **recurse into captured kernels**
(review-caught: an inlined callee's drift must guard the entry) — drift is
counted and rebuilt against the frozen env (decoration-time semantics), never
silently served; refuse-vs-recompile remains a dial; (5) batteries: each
satellite ships an explicit `install(registry)` and calls `install(DEFAULT)`
at import — `import pdum.dsl` is batteries-included, a hand-built `Registry`
receives the identical dialect through the same seam (surface E is not a
singleton in disguise), and the batteries dispatcher never clobbers one a
user installed first. Step-8 review corrections worth naming: the Python
renderer emits **lazy `if`/`else` statements** with dominator-based node
placement (the eager first draft broke guard-then-divide — crashing on
exactly the guarded input); the tier-2 key is `(region.key, backend.fp)` (a
version bump is a new artifact world — `name` alone served stale codegen);
`backend_for` trips loudly when a second backend registers before per-role
routing exists; the hit path is a single-lock `probe()` (guards inline, LRU
touch, retirement moved to the miss path). Known deferred: pipeline entries
escape template retirement (step-10 registry work); `FastRecord.staging` is
single-threaded by design (the render-loop contract); guard drift on a
long-lived handle recompiles per call — loud but wasteful, policy dial open.

**2026-07-12 (step 7 review): `Stage.forbid` — conversion targets cannot
express op-level elimination.** The claim "the `{core, abi}` legality set
proves no logical capture survived legalization" was **false**: `core.env` is
in the `core` namespace, so `check_legal` passed regions still holding it
(verified). The guarantee held only because the rewrite rule was total — one
partial rule (e.g. skipping buffer captures for the ndarray kind's own stage)
would have silently resurrected M0's per-frame flatten with no test failing.
`Stage` therefore gains `forbid: frozenset` of op *names*, checked after
fixpoint alongside `legal`; `legalize_params` declares
`forbid={"core.env"}`. MLIR's conversion target is a dialect-level concept;
this is the op-level complement it lacks. Any future stage that eliminates
specific ops of a legal dialect must declare it the same way.

**2026-07-12 (ch07a walkthrough): single tail return.** The base language
takes the strict end of the field's split (numba unifies return paths;
Taichi refuses all but one): **one `return`, at the tail of the body —
`core.yield` IS the return.** No return-path unification, no return-join
phis, ever; a mid-body `return` is a loud lowering error whose message
names the tail. Consistent with strict joins (same-type-or-loud) and the
reason numba's `typeinfer` fixpoint has no counterpart here. Recorded in
the plan's step-10/11 language notes; the ch07a matrix row updated.

**2026-07-12 (step 5): provenance schema committed** — MLIR-lite location
algebra (`Loc`/`CallLoc`/`FusedLoc`, outside identity, anti-pattern-gated),
rewrite-driver inherit-default via `Builder.default_loc` (fresh nodes inherit
the replaced node's loc; survivors keep their own — preserving DAG sharing),
loc-bearing type errors. Contract: starting region for an AI consumer, not
DWARF. Details: `050_provenance_tracking.md`.

**2026-07-12 (step 5): equality saturation evaluated as THE core — rejected
with measurements, retained as the §12 satellite.** egglog probe (ch06): the
phase-ordering win is real (`x*2+x*3 → x*5`, ~1 ms) but the costs are
disqualifying for a core: ~20 ms saturate+extract at kernel scale (the whole
miss budget) vs microseconds greedy; ~1.5 s import vs the zero-dep kernel;
bounded-iteration heuristics vs golden tests and deterministic content keys;
no home for non-equational passes (slot numbering, AD, rendering); binders
(`core.for`/`core.call`) are the classic e-graph hard case. Confirms V2/R7.
The sanctioned home stays: an opt-in `Region → Region` optimizer pass.

**2026-07-12 (step 4, ch05 walkthrough): strict core arithmetic.** Core
arith/cmp require same-type operands; every conversion is an explicit
`core.cast` in the IR (the kernel's `promote()` was deleted — promotion
policy had leaked into the kernel, the same class of mistake as tuple→Vec).
Promotion, where a language wants it, is a dialect's lowering policy —
Julia's architecture (stdlib methods), MLIR/LLVM/WGSL's strict operands.
Payoffs: emitters never invent conversions; AD sees matching types by
construction; the content hash reflects the exact computation.

**2026-07-11 (step 1, ch01 walkthrough):** the lattice gains `Tuple(elems)`.
An early step-1 draft summarized homogeneous scalar tuples as `Vec` at capture
time — a shader-dialect interpretation leaking into the identity layer, caught
at the walkthrough. The rule now: **`typeof` produces `Tuple`, never `Vec`**
(element-wise, arity in the identity, heterogeneous fine). `Vec` remains IR-
level only, produced by dialect lowering rules (`core.vec`); whether a captured
`Tuple((f64, f64))` packs as one `vec2<f32>` uniform or two scalar slots is the
backend's PackPlan decision. This restores M0's documented "two type levels"
split.

Deviations from verdicts carried over from P3 (all flagged there): first
backend is a source renderer, not an eval-rule interpreter (the `"eval"` column
stays reserved; ~150-line cost if wrong); printer trimmed to 60 kernel lines
with the harness in `tools/`; V3's hook kernel compressed to ~380 in-kernel
lines by sharing the rule matrix (overflow capped at `registry.py ≤ 150` before
revisiting); ndarray ValueKind in `stdlib/` (packaging, not architecture).

**2026-07-12 (step 10b): measurement is a satellite — zero kernel edits,
by construction.** `bench.py` (125 lines) wraps the `FastRecord` seams the
architecture already exposed: `extract` and `launch` are plain dataclass
fields, so `instrument()` swaps in timestamping shims and restores them —
the phase decomposition (key+probe / extract / pack / launch) needed no
hook API, which is the §2.12 design paying out. `benchmark()` is
BenchmarkTools-shaped (warmup, tune evals above a resolution floor, sample
to budget; **minimum** as headline). GPU depth rides the demo runtime's
`timed_call` (WebGPU `timestamp-query`, begin/end-of-pass ns). The step-9
microbench thresholds are now real pytest gates (alarm 5 µs, fail 40 µs
with CI margin; measured hit path 2.4 µs). **The finding that rewrites
ch10's story:** the ~2 ms/frame is 431 µs encode+submit + **4.7 µs GPU** +
**1629 µs readback**, and readback is ~constant from 64² (16 KB) to 1024²
(4 MB) — it is *fixed sync latency* (submit→wait→map), not bandwidth. The
per-frame cost ceiling is the synchronous-readback protocol, not the
dispatch machinery — the async/persistent-surface story (graphics `draw`)
is where that latency dies, deferred with the draw surface itself. Found
live: an Env member named `f16` is a reserved WGSL identifier → members are
`m{offset}` + regression golden. Review hardening: negative-tick clamp in
`timed_call` (drivers may report non-monotonic pass timestamps); stale-record
guards in both instruments (a guard drift mid-loop now fails loud instead of
averaging one stale frame into the rest); and `gpu_timeline` drives its frames
through `registry.dispatch` itself with `launch` shimmed to `timed_call` — the
measurement tool executes the real hit path, not a hand-replayed copy that
could drift from it.

**2026-07-12 (post-10b pause): the punning charter — 090.** User-driven
pause before step 11, two threads. (1) **Stdlib minimalism as policy**: the
squatting test (would a third-party library plausibly own this name with
richer semantics? → not stdlib) extends 080's kind-hygiene rule from names
to packages. `Color` + `dot2/length2/lerp2` moved from `stdlib/batteries`
to `pdum.dsl.demo.graphics`, which is NOT auto-imported — one explicit
import wires it in, demonstrating the ecosystem-package workflow the five
surfaces exist for. Stdlib = base pack + scalar intrinsic core + scalar
lingua-franca helpers, full stop. (2) **Core+extensions ("punning") at the
dialect and runtime layers** — 090 charters both: vendor op namespaces
(spelled by one backend, never decomposed = visible portability opt-out),
capability flags checked at build (step 14), the runtime's do/refuse list
(cuda.core IS the CUDA runtime; we own only the contract + the thin Metal
shim), artifact capability protocols (`timed_call` the shipped precedent;
`record.artifact` public = the runtime escape hatch), the buffer/interop
contract (device axis in Array types; OWNED/ADOPTED leaves; zero-copy both
directions via DLPack/CAI/buffer-protocol; readback may degenerate to a
sync on unified memory), and the multi-device testing ladder (fake-runtime
conformance suite → probe-gated device layer → cross-device differential;
CUDA box enters at step 14 primarily via handoff-doc + parallel agent).
DECISION: no abstract runtime class now — rule of three; extract at step
14 from wgpu + cuda.core + Metal. The vertical seed continues: step 11
consumes 090 §5 immediately.

**2026-07-12 (step 11): DATA AND LOOPS — statements, arrays, the C target,
and named axes (design 100).** One kernel line spent in the whole step:
`Array.device` (090's dispatch axis; wiring waits for a second device).
Everything else satellites. (1) **Statements**: `if`/`for` joined the base
pack — strict joins (same type on both paths, no unification), loop
carries as ONE yielded value (multi-name joins ride a literal
`core.tuple`, preserving the walker's single-yield contract and keeping C
scalarizable), single tail return ENFORCED (return inside a branch/loop is
a policy refusal), `while`/`break`/`continue` refused BY POLICY (bounded
loops, R11's line). SOUNDNESS CATCH made at design time: `core.param`
identity is structural, so a loop binder reusing index 0 would collide
with function param 0 in content keys — two different programs, one
artifact. Loop binders carry TUPLE indices `("loop", *inline-prefix,
seq)` — unique across inlining AND deterministic from source order alone
(the first draft's shared counter made content keys depend on process
history; review-caught, now pinned by a determinism test). Also
review-driven: names born in only one `if` suite are branch-local
temporaries that DIE with the suite (the first draft refused them,
outlawing innocent scratch variables); an empty-carry loop lowers its
body before refusing, so nested unsupported constructs surface their OWN
refusal instead of a misleading carry complaint. (2) **Arrays are captures**
(v1: read-only, C-contiguous, indexed to scalars; args/views/writes are
recorded cuts, 100 §6): rank-generic `Array` summary — shape and strides
are STAGING VALUES (i64 slots legalized through the ordinary `core.env`
sub-path → `abi.slot` route, zero new legalization), the payload rides the
leaves channel bound by the one new dialect op `array.buffer`;
`ShapedArray` turns the §13 dial (shape in type, strides const-fold,
staging shrinks). The thesis extends to data: new shape = cache hit.
(3) **Named axes — the xarray exercise** (user-directed: "the most
pedantic possible, as long as the machine code is efficient"):
`NamedArray.dims` in the type; `isel(y=…, x=…)` keywords MANDATORY on
named arrays, positional REFUSED (no back door for the transposition
bug); label-based `sel` deferred (labels are host-side values). The
refinement is ERASED at emission, so a named kernel and its positional
twin produce IDENTICAL content keys — tier 2 compiles ONE artifact for
both (pinned by test; first draft leaked `NamedArray` into the buffer
node's type and the artifact cache caught the lie: 2 artifacts). xarray's
`DataArray` adopts via its numpy payload, dims read from the value —
renamed dims are a different type. (4) **The C target** (`backends/c.py`,
the contribution point's first citizen; bucket raised 150→500
consciously): C99 via the shared dominator walker (which grew `core.for`
+ a statement-skip so C can scalarize tuples into lane variables — no
structs), `cc -O2 -shared` + ctypes, scalar returns v1, never claims the
default route. Numeric policy enforced BOTH sides now: the python twin
spells i64 div/mod as trunc (`int(a/b)`, `int(math.fmod)`) — C's native
behavior; differential-tested on all four sign combinations.
(5) **Ray-march spike verdict** (M3, ch12): expressiveness GO — a 32-step
sphere tracer is `for`+`if`+carries+batteries; python 22.4 µs/ray vs C
3.2 µs/ray (~7× body speedup), but ~2.4 µs of the C number is DISPATCH —
per-pixel-per-call is the wrong granularity for CPU frames; frames want
domain calls (ch10's `out=`) or DPS out-arrays (chaining/step 14). Tests
199; notebooks 19/19 (ch12-data-and-loops + ch12a delta interlude).

**2026-07-12 (step 12): TRANSFORMS — vmap, jvp, in-kernel `D`, and named
contraction (design 110).** THE SPIKE FINDING, which reshaped the step:
**our vmap is SIMT-shaped, not SIMD-shaped.** It never widens values with
a batch dimension (JAX's way — where batched predicates force
execute-both-and-select, breaking lazy branches, and batched trip counts
need masking); it adds ONE trailing i64 lane parameter and WEAVES it into
accesses whose NamedArray captures carry the mapped axis — the same move
that made compute-family params thread coordinates. Consequences:
intermediates stay scalar, `if`/`for` need ZERO new transform machinery,
the lazy-branch guarantee survives vmap (no `where` wart), and the
priced 180-lines-per-region-op tax never materializes: the transforms
satellite landed at 338 counted lines after review hardening (<350 — NO
re-hear). Recorded loss:
cross-lane collectives have no home in a woven representation (GPU-shaped,
deferred). Surfaces: `vmap(f, axis="name")` — named-first (user-directed):
weaves named captures, broadcasts the rest, REFUSES when nothing carries
the axis; the woven name is scoped away inside the body (`isel` of it
refuses); axis name rides the `Derived("vmap", …)` identity — new batch
size = 0 recompiles. `jvp(f)(*args, *tangents)` → (primal, tangent): ONE
tangent engine (per-op linearization rules, surface-A-shaped column;
None-is-zero algebra so untouched slices cost nothing; branches get a
parallel lazy `core.if`; loops WIDEN — carry becomes (primal, tangent),
primal consumers re-pointed at lane 0) — matches finite differences
through branches AND loops (|Δ|≈9e-9); fresh closures under transforms =
0 compiles (the thesis survives its first transform). **`D(x)`**
(user-directed, GLSL's dFdx analytically): partials of any intermediate
w.r.t. the enclosing kernel's params via basis seeding of the SAME engine;
structured values differentiate structurally; D-free kernels mint
unchanged artifacts (pinned); compute shaders have no quads, so analytic
`D` is the only derivative there — ch13 demos fwidth anti-aliasing with a
one-pixel edge at two zooms on CPU. GL vocabulary (ddx/ddy/fwidth) =
demo.graphics batteries, not stdlib. **Named contraction** (stretch):
`matmul(A, B, i, j)` pairs the UNIQUE shared axis name — woven axes
excluded FIRST, which is exactly why `vmap(cell, axis="batch")` gives
batch matmul for free (matches np.matmul; new batch AND inner extents =
cache hits, trip count reads the shape slot). The "rules engine" fear
dissolved into a dozen lines shaped like a type rule; einsum generality
deliberately not built.

**2026-07-13 (step 13, stage 1): THE SEAMS — and vmap becomes `over`
(design 130 §7, under 120's tripwire policy).** The step-12 review traced
four of its nine bug classes to ONE cluster: build-scoped state smuggled
through the rules dict (five string-keyed doors), a duplicated lower tail,
and three hand-rolled wrapper protocols. Bought from the kernel, each cap
move ledgered: (1) `Lowerer.context` — one dict per build threaded through
`inline`; the registry plants itself in it, transforms merge woven axes,
tangent memos live in it; ALL string doors deleted. (2) `Lowerer.root` +
`params` — the in-kernel derivative reads the kernel's own params from the
seam; the root-argc planting dance is gone. (3) `Lowerer.binder` — the
deterministic tuple-index invariant (("b", *prefix, seq)) is kernel law
now, not a satellite convention. (4) `lower_handle(context=, prefix=)`
with Derived-base RE-ENTRY: a build rule that lowers its base re-enters
lower_handle, so a Derived base dispatches to its own build rule with the
merged context — **transform composition is this re-entry, not a
mechanism**. `over(over(g, axis="b2"), axis="b1")` works (lanes trail
outermost-last; duplicate axes refuse); jvp∘over refuses on the lane type.
(5) `kernel/derived.py` (NEW, cap 45, 26 used): `DerivedValue` — the ONE
wrapper protocol + ONE MRO-covered ValueKind; Pipeline/Over/Jvp are
subclasses; `_guards` recurses into wrapper captures (closing the step-11
noted gap). RENAME: vmap → **`over`** (no users; the JAX prior contradicts
ours on every axis — argument-position vs capture-name, call-once-batched
vs per-lane coordinate). Kernel 1280/1500 (measured by scripts/loc_budget.py at commit time — the audit caught a stale pre-commit number here once; totals are pasted from fresh runs now). ALSO: the book builder moved
INTO the repo (`scripts/book/build_chapters.py`) after two parallel
sessions edited the same chapter from separate scratchpads — the single
source of truth for docs/book is now PR-visible (the ch11b lesson).

**2026-07-13 (the budget conversation): caps become tripwires; the kernel
buys its event seam.** The 1150 cap did exactly what it was for — the kernel
froze at 1147 through two satellite-only steps and five surfaces — and then
began producing the wrong artifact: with 3 lines of headroom, step 10b's
instrumentation shipped as a MONKEYPATCH on live `FastRecord` fields rather
than as a seam, and the entire miss path (lower/rewrite/render/`cc`) stayed
dark because its phases are locals no satellite can reach (120 §1.3). Policy
change, user-directed: `KERNEL_TOTAL_CAP` 1150 → **1500** (a tripwire, not a
wall; every cap crossing still demands a ledger entry; the budget TEST stays
exactly as strict); per-file caps `cache.py` 165→175, `registry.py` 110→140 (120 §9 guessed
125; the real twin with batched phase emission measured 132),
new `events.py` 55 (kernel) and 300 (recorder satellite). One virtue
inverted, now that it has bitten twice: satellites needing visibility must
ASK FOR A SEAM — monkeypatching live kernel state is what is forbidden.
Implementation of 120 (event seam, recorder, `Memo`, monkeypatch deletion)
proceeds with three review amendments: `capture.py`'s weak/plain memo
containers keep their lifetime semantics (events only, no `Memo` migration);
the recorder keys its span stacks per-thread (a depth column alone
interleaves wrongly under the cross-thread compiles §8 champions); and the
forbidden-miss path preserves `_explain`'s "nearest entry differs in" text.

---

## 11. What remains open (deliberately)

- **Aliasing between captured buffers** (same array captured twice, mutated
  through one name): decided per backend the first time correctness depends on
  it — irrelevant | normalized at marshal | relational key component (V4).
- **Early-exit control flow**: deferred to the M2 ray-march spike with a
  pre-priced decision procedure.
- **`while` / unbounded loops**: not in the language until a domain use case
  forces the conversation; the decision procedure is the same as region-op #4.
- **Native fastpath timing**: the two escalation stages are pre-shaped but not
  scheduled; the microbench gate decides when.
- **Disk-cache format**: structural key design is settled (V4); serialization
  format and dependency-closure hashing are M5+ work.
- **Third-party analysis stages**: an always-on domain analysis attaches today
  as a `Dialect`-bundled stage guarded on "program contains my ops/types"
  (§12). If stage insertion ever needs real semantics (ordering constraints
  between third-party stages), that becomes a deliberate sixth-surface
  conversation, priced with the same constitutional discipline as region op #4.

---

## 12. Heavyweight analyses as satellites (the hackability stress test)

Can a domain package drop an equality-saturation engine, an SMT solver, or a
convex optimizer into the middle of compilation? Yes — and the reason is
structural. Three properties of the kernel make heavyweight engines cheap to
integrate, and one discipline forces them into the *right place*:

1. **Everything expensive runs at miss time.** The hot path is sealed
   (§4.2), but the miss path between a cache miss and a `FastRecord` is
   ordinary Python with no time budget. A 200 ms solve once per type signature
   is invisible to a loop that hits the cache ten thousand times — and because
   analyses are deterministic functions of inputs already in the key (types,
   attrs, op graph), their results are cached by the same machinery.
2. **The IR is a frozen value with a tiny vocabulary.** ~30 ops, immutable
   nodes, memoized content hash, pure regions with typed yields, no CFG.
   Exporting to another formalism is a fold over a tree; importing the answer
   back is `Builder` emission. E-graph and SMT encoders choke on mutable
   SSA+CFG+effects (heroic in LLVM); we are on the trivial side by
   construction — `Node.key` is literally what hash-consing e-graphs want.
3. **Analyses take per-op semantics from the rule matrix, not hardcoding.** A
   solver pass asks the registry for the `(op, "unit")` or `(op, "range")`
   rule of each node it meets; ops added later by other packages participate
   by registering that aspect's rule, and a missing rule is a named
   `MissingRule(op, aspect, loc)`, never silent wrongness.

Mechanically, a domain package ships (a) a new **aspect column** of per-op
rules (Surface A) and (b) a **pass** — and a pass is just `Region → Region` or
`Region → diagnostics`. Nothing requires a pass to use the Pat/RuleSet driver
internally; inside it, anything importable is fair game. The solver dependency
lives in the satellite; the kernel never imports it (same pattern as the xDSL
oracle in `tools/`).

### Sketch A — units via equality saturation (egglog)

Units checking that also *places conversions optimally* (fuse factors, hoist
them off the per-pixel path):

```python
@rule("core.mul", "unit")
def _(d1, d2): return d1 * d2              # dimensions multiply
@rule("core.add", "unit")
def _(d1, d2): require(d1 == d2); return d1

def unit_pass(region, ctx):                # satellite: units/saturate.py
    eg = egglog.EGraph()
    emit_terms(eg, region, ctx.rules_for("unit"))  # fold over frozen Nodes
    eg.register(CONV_RULES)   # convert(a→b)∘convert(b→c)=convert(a→c);
                              # convert distributes over add; convert(const) folds
    eg.saturate()
    best = eg.extract(cost=count_runtime_multiplies)
    return rebuild_region(best)
```

The two-tier law decides where results land: a conversion attached to a
*captured* value becomes a `SlotSpec.convert` affine — the **pack tier** — so
switching a knob from millimeters to inches re-runs a cheap memo, rewrites
pack bytes, and never recompiles the shader. A conversion forced
mid-expression becomes an explicit `core.mul` by a `core.const`, visible in
printed IR and golden-testable. Inconsistent dimensions → the e-graph never
merges the classes → an error naming the two `loc`s.

### Sketch B — Z3 for bounds proofs

Per-op `"range"` rules emit interval/affine facts (`core.for` induction
variable ∈ [0, n); `i*4+c` propagates). A pass collects constraints for every
`core.load` and asks Z3 to prove `index < length`. Three outcomes, all clean:

- **Proved** → backend legalization *elides the runtime clamp* it would
  otherwise insert (optimization gated on proof).
- **Refuted** → compile error carrying the model as a counterexample ("when
  k=3, index = 66 ≥ len = 64"), mapped to `loc` via the constraint→node
  provenance table kept while encoding.
- **Unknown/timeout** → keep the clamp. Soundness never depends on the solver
  succeeding.

The same shape serves WGSL uniformity analysis (prove a loop bound uniform
across the workgroup before permitting a barrier).

### Sketch C — CVXPY at compile time

A battery like `approx(sin, tol=1e-4, interval=(0, tau))` whose `@overload`
runs a minimax polynomial fit (CVXPY) *during lowering* and emits the
coefficients as `core.const`s. `tol`/`interval` are attrs — compile-time
constants — so the solve is a deterministic function of the artifact key and
caches like everything else.

If the solve depends on a **value** (a target response curve tweaked per
frame), types-not-values makes "solver inside the compiler, per frame"
illegal — correctly — and forces an explicit choice: `Literal`-lift the spec
(value enters the key; recompile per spec — right when it changes rarely), or
run the solver *outside* the kernel in plain Python each frame and feed its
outputs in as ordinary captures (right when it changes every frame; the
solver's outputs are just uniforms). The "which tier misses when this
changes?" review question resolves it in one sentence.

### Caveats (the price of admission)

- **Determinism is a contract.** Solver outputs must be deterministic
  functions of key inputs or artifact/disk caching breaks — pin seeds; version
  the solver in the disk-cache toolchain tag.
- **Error attribution takes deliberate work.** Solvers answer globally; keep
  the constraint→node map while encoding so unsat cores come back as `loc`s.
- **Stage insertion is bundling sugar today** (a `Dialect`-contributed stage
  guarded on op/type presence, pay-for-what-you-use); if it ever needs
  ordering semantics between third parties, that is a priced sixth-surface
  decision (§11).

---

## 13. Value-dependent typing: the summary-function dial

"Type" in this architecture never meant "the Python class." It means **the
structural summary that a ValueKind's `typeof` chooses to extract from the
value** — the class is only the dispatch key for *which* summarizer runs. The
precedent has been in the design since M0: int range-bucketing gives
`typeof(5) = i64` but `typeof(2**63) = u64` — two values of one Python class
with different types, because the summary looked at the value. Shape-dependent
compilation is the same move, one notch richer.

### The spectrum, as three `typeof` bodies

**Rank-generic (stdlib default).** Shape stays *out* of the type and flows as
runtime data:

```python
class NdArrayKind:
    def typeof(self, a):     return Array(dtype, a.ndim, layout, byteorder, writeable)
    def leaf_types(self, t): return (BufferLeaf(), *ShapeLeafs, *StrideLeafs)
    def flatten(self, a):    return (a, *a.shape, *strides)     # shape words per call
    def fingerprint(self, a): return ("nd", dtype.num, a.ndim, layout_bit)
```

Same dtype/rank, different shape → same type → **cache hit**; new shape words
just ride the pack path. Right default for CUDA/C-style backends.

**Shape-in-type (opt-in kind).** For static loop bounds, unrolling, WGSL
fixed-size arrays:

```python
class ShapedArray(Type): dtype: Type; shape: tuple[int, ...]

class ShapedNdArrayKind:
    def typeof(self, a):     return ShapedArray(dtype, a.shape)
    def leaf_types(self, t): return (BufferLeaf(),)        # NO ShapeLeafs — shapes bake in
    def fingerprint(self, a): return ("snd", dtype.num, a.shape)   # soundness law forces this
```

Same shape → hit; resized → miss → recompile with the dims as constants. This
is JAX's model (jit keys on dtype+shape avals). Triton shows the midpoint:
*coarsened* value properties in the key ("divisible by 16"), which is also
just a `typeof` body returning a bucket.

Selection is explicit at the capture site (a wrapper like `shaped(arr)`, or a
one-off `Literal(x.shape)` lift) — never a global mode.

### The instance-level protocol

For objects that know their own type, a `__dsl_type__(self) -> Type` dunder is
checked before the class registry (companion to the `__pdum_dsl__`-convertible
slot in the call-resolution order). Prior art: numba's `_numba_type_`, DaCe's
`__descriptor__`, JAX's aval protocol. An *unregistered* type with no dunder is
a loud `typeof` error — never "fall back to the Python class," which would put
an unsound key in the cache.

### Guardrails (why this doesn't become a type-theory project)

1. **Types are hashable summaries, not predicates.** Frozen, structural,
   serializable; no computation inside a type. Relational properties ("these
   two arrays have equal length") are miss-time analysis aspects (§12), not
   types.
2. **The fingerprint-soundness fuzz polices enrichment mechanically.** Put
   shape in the type but forget it in the fingerprint and CI catches the
   silent wrong-hit.
3. **The cost is visible and named.** Shape-in-type = one artifact per shape;
   LRU eviction bounds it and per-tier miss counters name the cause
   ("arg 0: shape (64,48) → (128,96)").
4. **Backends demand richness where their ABI needs it**, rather than the
   frontend imposing it globally: WGSL's `legalize_params` rejects a
   rank-generic array in a uniform slot with an error telling you to use the
   shaped kind or lift the shape.

---

## 14. Build vs fork: why not tinygrad

Recorded as a considered alternative (full measurements:
`research/R7-minimal-compilers.md`). tinygrad validates our *mechanisms* — one
node type, rules as data, ~100-line renderers, content-hash compile cache,
CI line budget — and we stole all of them. It does not contain our *product*:

1. **Programming model.** tinygrad's language is the Tensor API (lazy tensor
   combinators → scheduler → fused kernels). It has no frontend that compiles
   a user's Python function body; pdum's unit of programming is a scalar
   kernel body with control flow. The part tinygrad lacks is exactly the part
   that is our thesis (capture, typeof, AST lowering, marshaling).
2. **Caching polarity — the decisive inversion.** In tinygrad a captured
   Python float in a tensor expression becomes a `CONST` in the graph: new
   value → new graph hash → recompile. Value variation without recompile
   requires explicit integer symbolic `Variable`s or manual tensor inputs.
   **tinygrad makes *dynamic* the explicit opt-in; pdum makes *static* the
   explicit opt-in (`Literal`).** For live-knob domains, the default polarity
   is the product.
3. **TinyJit is capture-and-replay, not a specializing JIT.** It records a
   launch sequence and replays it; a shape mismatch *raises* rather than
   respecializes; Python-side control flow is frozen at capture; there is no
   notion of user-code identity, so live-coding invalidation
   (edit-and-rerun → natural miss) has no home.
4. **No extension surfaces.** Fixed ~90-op enum, fixed scalar dtype set (no
   records/structs), no registry/dialect layering, no user types, no units,
   no method attachment, no mini-language seam. Our five surfaces are the
   product; adding them to a fork is surgery on a fast-moving upstream whose
   own 25k-line cap leaves no room for merge-back (and whose style — dense
   one-liners to stay under budget — R7 flags as a readability cost we should
   not copy).
5. **No graphics runtimes.** tinygrad's WebGPU/Metal backends emit *compute*
   kernels for tensor ops; fragment pipelines, canvases, uniform-buffer
   render loops (and audio callback loops) are runtimes we build either way.

Fork economics: the inheritable part (rewrite engine ~150 lines, renderers
~50–115 each) is the cheapest ~10% and is already budgeted in our kernel; the
missing parts (frontend, specialization cache, marshaling, hooks, domain runtimes) are
the expensive 90% and the point of the project — while a fork drags ~10 kloc
of scheduler/movement/runtime machinery our domains don't use.

The forward-looking relationship instead: **mine it** (done), **oracle it**
(differential-test our numerics against it where ops overlap), and optionally
**target it** — once vmap exists, a Surface-D backend could lower vmapped
pdum kernels to tinygrad tensor ops and inherit its fused CUDA/Metal execution
for array workloads, with tinygrad's own artifact cache below ours.
