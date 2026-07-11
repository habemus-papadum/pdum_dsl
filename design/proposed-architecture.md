# Proposed architecture: the pdum.dsl kernel

**Status:** synthesized proposal, ready for review. Not yet code.

**Provenance.** Produced by a structured multi-agent study on 2026-07-11, fully
auditable under [`design/research/`](research/):

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
| D5 | Transforms | **IR-to-IR passes whose per-op content is rules (`jvp`/`transpose`/`batch` columns) in the same registry; `grad(f)`/`vmap(f)` mint `Derived` template identities flowing into the unchanged thesis cache.** Backend-native AD only as a `custom_vjp`-shaped escape hatch and test oracle. | JAX's rule matrix as content model, tinygrad's rewrite passes as execution model (tinygrad's whole AD is 132 lines of rules). Derived identities make `grad(f)` rebuilt per frame an ordinary cache hit. → `V5-transforms.md` |

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
class Vec(Type):      elem: Scalar; n: int
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
    params_key:  Callable[..., Hashable]     # backend params that enter the thesis key
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
thesis cache    (template_fp, env_fp, arg_fp, backend_fp+params, generation) → FastRecord
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

1. **Kernel line cap** ≤1150 + per-file caps + per-backend caps (≤300 render,
   ≤220 runtime) + PR line-delta bot.
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

The orbiting-disk demo (`docs/demos/disk.py`) reproduced on the new kernel with
**both** the WGSL backend and the Python backend — proving the backend seam,
the thesis cache, and the hot path in one milestone (≈1130 in-budget + ~350
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

Deviations from verdicts carried over from P3 (all flagged there): first
backend is a source renderer, not an eval-rule interpreter (the `"eval"` column
stays reserved; ~150-line cost if wrong); printer trimmed to 60 kernel lines
with the harness in `tools/`; V3's hook kernel compressed to ~380 in-kernel
lines by sharing the rule matrix (overflow capped at `registry.py ≤ 150` before
revisiting); ndarray ValueKind in `stdlib/` (packaging, not architecture).

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
missing parts (frontend, thesis cache, marshaling, hooks, domain runtimes) are
the expensive 90% and the point of the project — while a fork drags ~10 kloc
of scheduler/movement/runtime machinery our domains don't use.

The forward-looking relationship instead: **mine it** (done), **oracle it**
(differential-test our numerics against it where ops overlap), and optionally
**target it** — once vmap exists, a Surface-D backend could lower vmapped
pdum kernels to tinygrad tensor ops and inherit its fused CUDA/Metal execution
for array workloads, with tinygrad's own artifact cache below ours.
