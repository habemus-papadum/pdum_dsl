# V4 — Verdict: the marshaling/data-model layer

*Consolidation verdict for the pdum.dsl redesign. Decides desiderata open questions
§7.6 (marshaling/ABI) and §7.7 (units placement), and the cached-extraction fix for
the M0 per-frame `flatten` fault line. Inputs: R1 (numba datamodel/ArgPacker), R4
(JAX pytrees/dispatch), R5 (Triton/cupy/MLX/Taichi marshaling), R8 (records, GPU
ABIs, units), R6 (torch guards, as the counter-model), plus
`design/dsl_caching_layer.md` (the hazard analysis this layer must satisfy) and the
frozen reference asset (`src/pdum/dsl_reference/`). July 2026.*

---

## 1. Recommendation

**Build the marshaling layer as three small, separately-cached pieces:**

1. **A `ValueKind` registry** (the pytree move, fused with `typeof`): one
   registration per Python type provides `typeof(v) -> Type` (frozen, structural,
   hashable — the *only* thing that enters the artifact cache key),
   `flatten(v) -> ordered logical leaves` (the runtime data), and
   `fingerprint(v)` (the cheap structural tag for the hot path). One registration,
   three derived views — `typeof` and `flatten` cannot drift apart.

2. **A backend-owned `PackPlan`**: at compile time (phase-B miss), the backend maps
   the *types* — never the values — to a flat list of `Slot` descriptors, each
   `(leaf source path, converter, physical destination)`. The physical-destination
   vocabulary (uniform offset, kernel-arg index, ctypes field, Python positional)
   is defined by the backend; the structural recursion (type tree → flat leaf
   list) is core. This is numba's datamodel/ArgPacker *shape* with the llvmlite
   vocabulary surgically removed, and it is what Triton's
   `build_signature_metadata` + generic native launcher already productionized.

3. **A `FastRecord` per cache entry**: `(artifact, extractor, pack_plan, launcher)`
   built once at miss time. The per-call hit path is exactly:
   **fingerprint → one dict probe → run precompiled extractor → generic pack →
   launch.** No AST touch, no re-inlining, no type-lattice walk, no layout
   re-derivation — this kills the M0 per-frame `flatten` flaw structurally.

**Units:** dimension lives in the `Type` (artifact key), concrete unit lives in the
`PackPlan` converter (a second-tier memo keyed additionally on concrete units).
Unit change = µs pack-plan rebuild; dimension change = honest type change;
never a recompile. **Value-dependent specialization:** a single explicit
`Literal[v]` lift that (a) enters the type/key and (b) surfaces to the backend as
a constant/IR attribute and (c) *removes* the corresponding slot from the pack
plan. No implicit value leakage into keys, ever.

---

## 2. Rationale

Four independent production systems converge on the same marshaling skeleton, and
the one that deviates (JAX) is the documented anti-model for our thesis:

- **numba** (R1 §4): per-type `DataModel` with tree-shaped `get_argument_type()`,
  flattened to primitive leaves by a ~200-line `ArgPacker`, per-target override via
  `DataModelManager.copy()/chain()`. An `ArrayModel` flattens to 5 + 2·ndim scalars
  — literally "one logical value → N physical parameters" solved by structural
  recursion over per-type models. Its fatal, twice-punished flaw: the leaves speak
  llvmlite `ir.Type`, so every extension died in the numba-cuda fork (~155k-line
  vendor) and again in the MLIR rewrite. Copy the recursion, reject the vocabulary.
- **Triton** (R5 §2): per-kernel signature metadata built *once*
  (`expand_signature` → flatten tuples, drop constexprs) and interpreted by *one
  generic native packer* per call. Constexpr params vanish from the physical
  signature; specialization hints enter both the cache key **and** the IR as
  attributes, keeping key and codegen in lockstep. This is our `Literal` lift,
  mechanically proven.
- **JAX pytrees** (R4 §4): the decisive conceptual split — every logical value
  divides into a **static half** (treedef/aux_data → hashable, into the cache key)
  and a **dynamic half** (ordered leaves → runtime data), via one user-extensible
  registry that serves every transformation boundary. For pdum.dsl, `Type` *is*
  the treedef and leaves are what the pack plan consumes. JAX simultaneously
  proves the negative: consts baked into `ClosedJaxpr` + id()-keyed cache means a
  rebuilt closure retraces perpetually (~240×). Marshal captures as parameters;
  never bake.
- **GPU ABIs** (R8 §1.3): the same 3-float struct is align16/size12 (WGSL uniform,
  with member-alignment rounding to 16), align4/size12 (CUDA float3), align16/
  size16 (Metal float3). Layout is *provably* not a property of the logical type;
  it must be a backend-computed plan or the frontend key drags WGSL-vs-CUDA facts
  into cache identity.
- **Units** (R8 §2–3): every compiled unit system (F#, mp-units, uom, Unitful,
  unxt) keeps units on the static side and erases them from runtime data; the two
  that put the *concrete unit* in the JIT key (Unitful, unxt's
  `eqx.field(static=True)`) demonstrably recompile on unit change and multiply
  specializations. Dimension-in-type + unit-in-pack-plan is the only placement
  consistent with both "types, not values" and "the loop stays hot".
- **torch.compile** (R6 §2) is the control experiment for the dispatch story:
  cache keys as discovered value/identity guards cost O(entries × guards) per call,
  forced a full C++ rewrite plus unsafe-skip escape hatches. A type-keyed
  dict probe is the O(1) degenerate case of guards — the layer's job is to keep it
  degenerate by making sure nothing on the hit path re-inspects object graphs.

The hazard analysis (`dsl_caching_layer.md`) is satisfied point-by-point in §3.7.

---

## 3. The design, concretely

### 3.1 The typeof path: `ValueKind` registry

One registration per Python type (dispatched via `functools.singledispatch`, like
numba's `typeof_impl`, plus a `_pdum_type_`-style duck hook for user classes):

```python
# core/marshal.py  (~200-250 lines with the builtin kinds)

class Leaf:                      # the CLOSED logical leaf vocabulary (core-owned)
    pass

@dataclass(frozen=True)
class ScalarLeaf(Leaf):   kind: str          # 'f64'|'i64'|'u64'|'bool' — honest widths
@dataclass(frozen=True)
class BufferLeaf(Leaf):   pass               # pointer-like: ndarray / GPU buffer handle
@dataclass(frozen=True)
class ShapeLeaf(Leaf):    axis: int          # one extent word
@dataclass(frozen=True)
class StrideLeaf(Leaf):   axis: int
@dataclass(frozen=True)
class EnvLeaf(Leaf):      pass               # a nested DSL closure's Env (recursive)

class ValueKind(Protocol):
    def typeof(self, v) -> Type: ...                   # -> frozen structural Type
    def leaf_types(self, t: Type) -> tuple[Leaf, ...]: # STATIC: derivable from Type alone
    def flatten(self, v) -> tuple[object, ...]:        # DYNAMIC: values, same order/arity
    def fingerprint(self, v) -> Hashable: ...          # cheap structural tag (§3.6)

register_kind(np.ndarray, NdarrayKind())    # etc.
```

Non-negotiable properties, each backed by a hazard in `dsl_caching_layer.md`:

- `typeof` implements the full lattice from the hazard doc: **int range-bucketing**
  (int64/uint64/bigint-error), ndarray = `(dtype, ndim, layout, byteorder,
  writeable)` with 1-D-contiguous canonicalization, tuples element-wise **with
  arity**, `None` as a singleton type, nested DSL closures as their `FnType`.
  R5 corroborates the array shape independently: cupy `CArray(dtype, ndim,
  contig, …)` and Taichi `(element_type, ndim, …)` both chose dtype+rank+flags,
  never shape values.
- `leaf_types(t)` is a function of the **Type**, not the value — so the backend
  can build the whole pack plan at compile time from `env_types`/`arg_types`
  alone, and `flatten(v)` at call time is guaranteed to produce matching arity.
  (JAX's discipline: treedef static, leaves dynamic, flatten runs per call and is
  therefore required to be cheap.)
- The leaf vocabulary is **small and closed** (scalars, buffer, shape/stride
  words, nested env). Backends are total over it; adding a leaf kind (e.g. a
  future `TextureLeaf`, `SamplerLeaf`) is a deliberate core version event, not a
  per-backend negotiation. This is the anti-llvmlite decision: leaves are
  *logical* kinds; physical spelling is the backend's.

Example: a captured `np.float32` array of ndim 2 →
`Type = ArrayType(f32, 2, 'C', '<', True)`;
`leaf_types` → `(BufferLeaf(), ShapeLeaf(0), ShapeLeaf(1))` (strides omitted for
layout `'C'` — demand-driven, MLX-style: the plan only binds what the emitted code
references); `flatten(arr)` → `(arr, arr.shape[0], arr.shape[1])`.

### 3.2 The physical layout: backend-owned `PackPlan`

```python
# core/plan.py (~150 lines) + per-backend planner (~100-200 lines each)

@dataclass(frozen=True)
class LeafPath:
    root: Literal['env', 'arg']
    index: int                    # which capture / which argument
    sub: tuple[str | int, ...]    # path through nested envs / struct fields
    leaf: int                     # index into that value's flatten() output

@dataclass(frozen=True)
class SlotSpec:
    source: LeafPath
    convert: Converter | None     # unit scale/offset, width cast (f64->f32) — §3.5
    dest: PhysicalDest            # BACKEND-DEFINED leaf vocabulary

# Backend-defined dests, e.g.:
#   wgsl:  UniformSlot(offset=16, fmt='<f'), BindingSlot(group=0, binding=2)
#   cupy:  KernelArg(i=3, ctype='float'), StructField(arg=0, np_dtype_offset=8)
#   c:     CField(offset, ctype)
#   python: PyArg(i)

@dataclass(frozen=True)
class PackPlan:
    slots: tuple[SlotSpec, ...]
    staging_size: int             # bytes of uniform staging, 0 if N/A
    def pack_into(self, staging, leaves): ...   # generic interpreter, ~40 lines

class Backend(Protocol):
    def plan(self, env_types, arg_types, units, opts) -> PackPlan: ...
    def compile(self, flat_ir, env_types, arg_types, opts) -> Artifact: ...
    def make_launcher(self, artifact, plan) -> Callable: ...
```

Design commitments:

- **The plan is data interpreted by one generic packer** (Triton's evolution away
  from per-kernel generated C launchers; numba's ArgPacker rationale — flattening
  to primitive leaves side-steps per-architecture struct ABI rules entirely).
- **Offsets/alignment are computed by the backend's layout planner** and never
  appear in frontend `Type` equality (R8 lesson 2: `StructType(tag, fields)`
  hashes on fields; `(backend, StructType) → layout` is a backend memo). numba's
  `Record.make_c_struct` (~40 lines) is the copyable offset-computation model for
  the C/CUDA planners; the M0 WGSL `layout.py` already is the WGSL planner —
  it generalizes from scalars to structs/vecs without changing role.
- **cupy struct trick where cheap**: a backend may choose to spell N leaves as one
  physical aggregate (CUDA `CArray<T,ndim>` struct arg = ptr+shape+strides as a
  single kernel param with a matching `np.dtype`); that is a `dest` decision
  inside the plan, invisible to core.
- **Lifted (`Literal`) captures produce no slot** — their value is already in the
  artifact (Triton constexpr behavior). The plan and the key are derived from the
  same type tuple, so they cannot disagree about which values are physical.

### 3.3 Cached extraction: fixing the per-frame `flatten` flaw

M0's `webgpu/runtime.py::update` calls `passes/inline.py::flatten(program)` every
frame — a full re-lowering (AST → IR, `Lowerer(...)` per handle) and re-inline just
to *collect current uniform values*. The redesign separates the two products that
`flatten` currently conflates:

- **Structure** (merged uniform names/types/order, the inlined IR) is a pure
  function of `(FnType, arg_types)` — computed once at compile time.
- **Values** are reachable from the live `Handle` by stable paths: at compile
  time, the inliner records, for every merged uniform, its `LeafPath` — e.g.
  `('env', 0, (), 0)` for entry capture `x`, or `('arg', 0, ('scale',), 0)` for
  capture `scale` of the device closure passed as argument 0. Paths are stable
  because `co_freevars` order is fixed per code object and `FnType` equality
  guarantees the same code objects and env shapes.

The extractor is then built **once** per cache entry:

```python
def build_extractor(paths: tuple[LeafPath, ...], kinds) -> Callable:
    # closure-compile the access chain per path; no IR, no AST at call time
    steps = [compile_path(p, kinds) for p in paths]     # p -> (lambda env, args: leaf)
    def extract(env, args):
        return [s(env, args) for s in steps]
    return extract
```

Per frame, the rebuilt closure presents a *new* `Handle` with the *same* `FnType`;
`extract(handle.env, args)` walks dict lookups and `flatten` calls only. For the
M0 demo this turns per-frame cost from "re-lower + re-inline two functions" into
"~6 dict reads + one `struct.pack_into` + `queue.write_buffer`". A later
optimization (Triton's `create_function_from_signature`) can `exec` a fused
extractor+packer per template; the contract already permits it (§3.6).

Semantics note (hazard doc, rebind-vs-mutate): the extractor reads the *current*
env each call — reference-snapshot semantics. Mutating a captured array's contents
is visible (numba-like reflection); rebinding a capture to a new-typed value
changes `env_types` at phase A and misses correctly.

### 3.4 Opt-in value-dependent specialization: `Literal`

One mechanism, three obligations, Triton-proven:

```python
@dataclass(frozen=True)
class LiteralType(Type):
    base: Type
    value: Hashable            # in __eq__/__hash__ — this IS the opt-in

n = lift(512)                  # or dsl.Literal(512); per-capture/per-arg granularity
```

1. **Key**: `typeof(lift(512)) = LiteralType(i64, 512)` — enters
   `env_types`/`arg_types`, so each value compiles separately. Loudly documented
   inverse default: unmarked captures are runtime data (the hazard doc's
   "deliberate exception", numba `prefer_literal`, Julia `Val{}`).
2. **Codegen**: the backend sees the value as a constant / IR attribute (Triton's
   hint→`tt.divisibility` pattern: key and codegen derived from the same lift, so
   they cannot drift).
3. **Plan**: no slot emitted (constexpr dropped from the physical signature).

Explicitly rejected in the same breath: implicit hints (Triton's default-on
divisibility/==1, cupy's undeclared `index_32_bits`) — R5 documents both as
recompile-surprise generators; Dynamo's default value specialization needed an
entire counter-machinery (automatic dynamic, recompile limits). If
divisibility-style hints ever prove necessary for performance, they arrive as
*declared* lifts (`lift_align(arr, 16)`), never as silent key components. And
never key on `id()` (Taichi `ti.template()`): where "this specific buffer" is
wanted, lift a stable token.

### 3.5 Units: dimension in the key, unit in the converter

Adopt R8 §3.2's placement, wired into this layer:

- `typeof(3.0 * mm) = QuantityType(f64, Dim(L=1))` — `Dim` is a frozen vector of
  `Fraction` exponents. The artifact cache key contains **dimensions only**;
  kernels are emitted in a canonical basis (SI base; unxt-style declarable unit
  systems later). Dimension errors are compile-time type errors.
- The **pack-plan memo is the second tier**: keyed
  `(FnType, arg_types, concrete_units, backend_byte_opts)`. A unit change (mm →
  inch) misses only here — rebuild is a µs-scale Python list build; the artifact
  is untouched. `SlotSpec.convert = Affine(scale=1e-3, offset=0.0)` is applied
  during `pack_into` as one fused multiply-write.
- **Value-preserving rules at the pack boundary** (mp-units' split): float reps
  convert implicitly; integer reps only when the factor is exact; otherwise a
  compile-time error naming the explicit cast. Affine units (°C) are a separate
  point-type when units ship — a converter is not made "smarter" to hide affine
  semantics.
- **Escape hatch is the same `Literal` lift**: a user who wants the factor
  constant-folded (or the concrete unit in the key) lifts it — no units-specific
  mechanism.

The generalized rule (R8 lesson 5), which becomes this layer's law: **anything
that changes what bytes get written where — but not what code runs — belongs in
the pack-plan key; anything that changes generated code belongs in the artifact
key.** (Backend params like target texture format that alter emitted WGSL go in
the artifact key — the M0 cache-hygiene gap; endianness/unit/width-cast details go
in the plan key.)

### 3.6 The hot loop: exactly what runs per call on a hit

```python
# The entire hit path — key build + pack + launch, pure Python:
def __call__(handle, *args):
    fp  = (handle.fntype_fp, fingerprint_tuple(args))      # (1) key build
    rec = _fast_cache.get(fp)                               # (2) one dict probe
    if rec is None:
        return _slow_path(handle, args)                     # full typeof -> compile
                                                            #   -> plan -> FastRecord
    leaves = rec.extract(handle.env, args)                  # (3) precompiled paths
    rec.plan.pack_into(rec.staging, leaves)                 # (4) generic pack
    return rec.launch(rec.staging, leaves)                  # (5) write_buffer + draw /
                                                            #     kern(grid, block, args)
```

- `fingerprint_tuple` is the hazard doc's **structural** fingerprint (tuple arity,
  int range bucket, array `(dtype, ndim, layout, byteorder, writeable)` — reading
  `arr.flags`, so cached per ndarray id+version where possible), memoized per code
  object for the env half: phase A on a rebuilt closure is `co_freevars` zip +
  per-value fingerprint, no full lattice walk. Fingerprint miss falls back to full
  `typeof` (sound fast key, per the hazard doc).
- `FastRecord = (artifact, extract, plan, staging, launch)` is JAX's
  `MeshExecutableFastpathData` shape in pure Python. JAX needed C++ because its
  per-call key is a full pytree flatten over nested args; ours is a code-object
  fingerprint plus a handful of capture tags — R4's analysis says pure Python
  reaches the "parameter write" floor *iff nothing on the hit path re-derives
  structure*, which this design enforces by construction.
- **Budget and escalation path**: target single-digit µs per call; measure from
  day one (`explain_cache_misses`-style counters naming which key component
  missed — code edit vs env-type change vs backend param vs unit tier). If the
  budget ever bites, the two escalations are already-shaped: (a) `exec` a fused
  per-template binder (Triton), (b) a narrow native fastpath behind the same
  cache contract (JAX). Neither changes any interface.
- Thread safety per the hazard doc: `_slow_path` populates under a per-key future
  (compiling/ready states); `generation` is read once at compile start; the
  `FastRecord` is installed atomically after both artifact and plan exist.

### 3.7 Hazard-analysis compliance checklist (`dsl_caching_layer.md`)

| Hazard | Where this design answers it |
|---|---|
| typeof lattice (int buckets, array flags, tuple arity, None, nested FnType) | `ValueKind.typeof` rules, §3.1 |
| Structural fingerprint, not `type(v)` tags; array fp not free | `ValueKind.fingerprint`, memoized; §3.6 |
| Rebind vs mutate vs drift | snapshot-at-capture env_types; extractor reads current refs; §3.3 |
| Value-dependent lifts explicit | `LiteralType`, §3.4 — one key/codegen/plan triple |
| Generation / world age | artifact-key component; FastRecord invalidated with entry |
| Env-layout memo as side table, not FnType field | pack-plan memo keyed by `(FnType, arg_types, units, byte_opts)` |
| L-cache leak (dead templates) | FastRecord map holds code objects via the same bounded LRU as L2; weakref discipline per R4 lesson 3 |
| Thread safety, recursive compiles | per-key future; plan built inside compile job; §3.6 |
| Disk key | `Type`/`Dim`/`LiteralType` are frozen dataclasses that serialize structurally (R8 lesson 8: numba structrefs defeat the disk cache because their types don't serialize — ours must from day one); plans are cheap to rebuild, never persisted |

### 3.8 Size estimate

| Piece | Est. lines | Prior-art calibration |
|---|---|---|
| `ValueKind` registry + builtin kinds + fingerprints | ~250 | numba `typeof.py` core is small; JAX pytree mini-registry = 193 lines in autodidax |
| `LeafPath`/`SlotSpec`/`PackPlan` + generic packer | ~200 | numba `packer.py` = 213 lines |
| Extractor builder | ~80 | closure-compiling dict paths |
| Fast dispatch + counters | ~120 | |
| WGSL planner (generalize M0 `layout.py`) | ~200 | M0 layout.py exists; struct rules from R8 §1.3 |
| CUDA/CuPy planner | ~150 | `make_c_struct` model ~40 lines + np.dtype builder |
| Units converters (when they ship) | ~150 | mp-units rules table |

Core marshaling ≈ **650 lines** + ~150-200 per backend planner — comfortably
inside the ~1000-line-kernel ambition, consistent with the prime directive
(a new backend adds a planner + dest vocabulary + launcher; it touches no core
file).

---

## 4. Considered and rejected

| Alternative | Why it lost |
|---|---|
| **numba datamodel verbatim** (four physical representations returning backend-IR types from core models) | The interface is target-generic but the vocabulary is LLVM; that single decision broke every numba extension twice (numba-cuda's ~155k-line vendor of core; the MLIR rewrite invalidating all `@intrinsic`/datamodel code). We keep the tree-flatten recursion and per-target override idea; leaves become logical kinds, physical spelling moves to the backend planner. Also: four representations is more than a source-DSL needs — we have no SSA-register/data split; value+argument views suffice, folded into `Type` + `PackPlan`. |
| **JAX-style const baking** (captures hoisted into the artifact) | The anti-thesis, with numbers: rebuilt closure → perpetual retrace, ~240× documented slowdown, diagnostics built solely to catch it. Captures must be parameters with a marshaling annotation. |
| **Guard-based caching** (Dynamo: discovered value/identity predicates checked per call) | O(entries × guards) linear scan vs one dict probe; needed a 301KB C++ guard-tree rewrite, fail-count reordering, recompile limits, and unsafe-skip stances. Right for arbitrary Python; wrong for a closed DSL whose key is a total function of inputs. We do import its *inventory* as a key checklist (backend identity + params, ambient state, generation) and note the relational-aliasing hole (§5). |
| **Full unit-in-type** (Unitful.jl, unxt `static=True`) | Provably recompiles on unit change (a slider retype mid-interaction = shader rebuild) and multiplies specializations (Unitful's documented type-instability). Violates "the loop stays hot" for a value-adjacent change. |
| **Runtime unit wrappers** (Pint/unyt/astropy model) | Cannot pass a JIT boundary at all (numba #5827); the community workaround (strip `.magnitude` at the boundary) *is* our design done manually. |
| **Implicit value specialization** (Triton default-on hints; cupy `index_32_bits`) | Documented recompile surprises; keys gain undeclared value components. Mechanism adopted, polarity inverted: hints exist only as explicit lifts. |
| **Identity-keyed anything** (Taichi `ti.template()` → `id()`) | GC id reuse aliases keys; leaks via strong refs; no cross-run stability. The founding anti-pattern. |
| **Per-kernel generated/compiled launchers as the primary mechanism** (old-Triton C launchers) | Triton itself retreated to one generic metadata-driven packer. Generated binders are kept as a *later, contract-preserving* optimization of the hit path, not the architecture. |
| **Single-tier keying** (units, byte-level options, layouts all in the artifact key) | Turns byte-plumbing changes into compiles; the two-tier split (expensive cache coarse on types/dims, cheap memo fine on units/byte options) is what keeps interactive unit/format tweaks at µs. |
| **Layout in the frontend type** (offsets in `StructType.__eq__`) | The same struct has three layouts across WGSL/CUDA/Metal; frontend keying on any of them either fragments the cache per backend or lies. Layout is a `(backend, Type)` memo. |

---

## 5. Confidence and what would change my mind

**High confidence (would bet the architecture on it):** the overall shape —
static-type/dynamic-leaf split, backend-owned flat slot plans built at compile
time, generic packer, opt-in `Literal`, O(1) dict-probe dispatch. Evidence is
convergent across numba (ArgPacker), Triton (signature metadata), JAX (pytrees),
Taichi (`set_arg_*` slots), and MLX (declarative binding), and the counter-models
(JAX consts, Dynamo guards, Unitful units) each failed in the specific way the
thesis predicts.

**Medium confidence, with named tests:**

- *Pure-Python hit-path speed.* Claimed: fingerprint + probe + extract + pack +
  launch fits single-digit µs. numba judged Python too slow for this and wrote
  2,850 lines of C — but its key is heavier and its call granularity finer than a
  per-frame draw. **Changes my mind:** a day-one microbenchmark of the M0 demo
  hit path exceeding ~10 µs after the extractor fix → pull the exec-generated
  binder forward (it's designed in, §3.6); if even that fails at 60–120 Hz with
  hundreds of programs, add the narrow native fastpath behind the same contract.
- *Dimension-only types suffice.* No production system runs exactly
  dimension-in-key + unit-in-plan (unxt is nearest and chose unit-static).
  **Changes my mind:** prototype showing pack-time converters mishandle a real
  domain (integer-rep audio buffers, affine temperature chains, exactness
  arguments in optimization) → default the concrete unit into the type behind an
  auto-lift while keeping the dimension tier for sharing.
- *Leaf vocabulary closure.* Scalars/buffer/shape/stride/env may undercount GPU
  realities (textures, samplers, atomics, bind-group semantics). **Changes my
  mind:** the WebGPU backend needing per-backend leaf kinds that don't generalize
  → weaken "closed core set" to "core set + namespaced backend leaves that only
  that backend's planner may emit", at the cost of some cross-backend totality.
- *Aliasing.* A pure type key cannot express "captures A and B share storage"
  (Dynamo's relational guards name this hole). For WGSL uniform copies it is
  moot; for CUDA in-place kernels it may not be. **Changes my mind:** a backend
  whose correctness depends on aliasing → add a normalize-at-marshal step or an
  explicit relational key component, decided per backend, never discovered per
  call.

---

## Design lessons for pdum.dsl

1. **One registration, three views.** `typeof`, `flatten`, and `fingerprint` come
   from a single `ValueKind` registration per Python type; `leaf_types` is
   derivable from the `Type` alone. This makes key/plan/extraction drift
   structurally impossible — the property whose absence cost cupy
   (`index_32_bits` leak) and numba (Python/C `typeof` synced "by comment").
2. **Compile-time plans, call-time interpretation.** Everything expensive —
   layout, offsets, converters, extraction paths — is computed at phase-B miss
   and stored in one `FastRecord`. The hit path is: key build + dict probe +
   extract + pack + launch, and *nothing else*. Re-deriving any structure per
   call (M0's `flatten`) is the layer's one forbidden act.
3. **Backends own physical vocabulary; core owns structural recursion.** Core
   flattens `Type` trees to logical leaves; the backend planner maps leaves to
   `PhysicalDest`s and computes ABI layout (`plan_layout` per R8: WGSL 16-byte
   uniform rounding, CUDA np.dtype mirror, Metal packed types). No backend type
   ever appears in core, and no core type ever encodes bytes.
4. **Two-tier keying is the law of the layer**: changes-what-code-runs → artifact
   key (types, dimensions, lifts, backend codegen params, generation);
   changes-what-bytes-go-where → pack-plan key (concrete units, width casts,
   byte-level backend options). Interactive tweaks must only ever hit the second
   tier.
5. **`Literal` is the only door for values**, and it opens three things at once:
   the key, the backend constant, and slot elision. Per-capture granularity,
   loud documentation of the default ("unmarked captures are runtime values"),
   and a compile-count lint per template (Dynamo's recompile-limit repurposed as
   a warning).
6. **Units are converters, not types** — until a user lifts them. Dimensions
   type-check kernels; the canonical basis keeps artifacts shared; the pack-plan
   converter does the scaling with mp-units' value-preservation rules. This
   answers desiderata §4.4 literally: conversion happens "at the level where it
   arranges the arguments passed to compiled functions."
7. **Ship the observability with the cache**: per-tier miss counters that name
   the differing key component, and a `no_compile` assertion mode for the render
   loop. JAX and torch both had to retrofit exactly this; a type-keyed design can
   make miss explanations precise (which capture's type changed) instead of
   forensic.
8. **Design for the escalation you hope to avoid**: the dispatch contract
   (fingerprint → record → extract/pack/launch) must be implementable by an
   exec-generated per-template binder and by a native fastpath without interface
   change. Triton and JAX both landed there; pdum.dsl should be able to follow
   without a redesign — but only after measurement says so.
