# P3 — Architecture proposal: the nanopass kernel

**School:** nanopass / tinygrad. One tiny uniform IR node type + an ops table; all
compiler logic as data-driven pattern-rewrite rules run by one generic driver;
per-backend renderers of 100–300 lines emitting source text; numba-style explicit
registries for typing/lowering/marshaling; many small isolated passes; a
CI-enforced line-budget culture.

**Inputs honored:** the five verdicts (V1 frontend, V2 IR, V3 hooks, V4
marshaling, V5 transforms), `docs/desiderata.md`, `design/dsl_caching_layer.md`,
and the M0 reference asset at `src/pdum/dsl_reference/`. Every deviation from a
verdict is flagged in §8.

**Identity constraint restated:** compiled artifacts are cached on the TYPES of
captures and arguments, never their values. Phase A (closure capture at
decoration) is compile-free. A hot loop that rebuilds closures with fresh values
every iteration executes exactly: key build + value pack + launch.

---

## 1. Core primitives

Twelve data structures. Everything else in the system is a function over these.

### 1.1 `Type` — the structural type lattice (~65 LOC, `kernel/types.py`)

Frozen, slotted dataclasses; structural `__eq__`/`__hash__`; serializable from day
one (disk-cache requirement, V4). Honest widths: Python `int` buckets to
i64/u64/bigint-or-error; narrowing to f32/i32 is a backend decision recorded in
the backend's `type_map`, never in these objects.

```python
class Type: ...                                        # abstract, frozen subclasses only
@dataclass(frozen=True, slots=True)
class Scalar(Type):   kind: str                        # "f64" "i64" "u64" "bool" "f32" "i32" "u32"
@dataclass(frozen=True, slots=True)
class Vec(Type):      elem: Scalar; n: int             # vec2/3/4
@dataclass(frozen=True, slots=True)
class Array(Type):    dtype: Type; ndim: int; layout: str; byteorder: str; writeable: bool
@dataclass(frozen=True, slots=True)
class Record(Type):   name: str; fields: tuple[tuple[str, Type], ...]
@dataclass(frozen=True, slots=True)
class FnType(Type):   template: TemplateId; env_types: tuple[Type, ...]   # the caching thesis
@dataclass(frozen=True, slots=True)
class LiteralType(Type):  base: Type; value: Hashable  # the ONE value-in-type exception (explicit opt-in)
# reserved, no schema change needed later (V4): Quantity(Type): rep: Type; dim: Dim
```

### 1.2 `TemplateId` — code identity as a sum type (~15 LOC, in `types.py`)

V5's requirement, day 1: `grad(f)` and `f` must never collide, and `grad(f)`
rebuilt per loop iteration must hit.

```python
@dataclass(frozen=True, slots=True)
class Base(TemplateId):     code: types.CodeType       # VALUE-compared (CPython semantics), never id()
@dataclass(frozen=True, slots=True)
class Derived(TemplateId):  tag: str; base: TemplateId; static_params: tuple[tuple[str, Hashable], ...]
```

### 1.3 `SourceSnapshot` — decoration-time source (~10 LOC, `kernel/capture.py`)

`(text, filename, firstlineno, qualname)`, taken at decoration when code object
and `linecache` are coherent (V1). Memoized per code object. On a phase-B miss,
`compile(text)` must be value-equal to the template code (ignoring
filename/firstlineno) or we raise — stale-source compiles are impossible.

### 1.4 `Handle` — a first-class DSL closure (~25 LOC, `kernel/capture.py`)

The phase-A product: `(FnType, env values, snapshot)`. The env is runtime data
only; **no cache key ever touches it**.

```python
class Handle:
    fntype:   FnType                      # (TemplateId, env_types) — the structural type
    env:      tuple[object, ...]          # capture VALUES, co_freevars order; never keyed
    env_fp:   tuple[Hashable, ...]        # precomputed structural fingerprints (hot-path key part)
    snapshot: SourceSnapshot
    kind:     str                         # "device" | "fragment" | "compute" | ...
    registry: Registry                    # which world this handle compiles in
    def __call__(self, *args): ...        # the hot path, §4.3
```

### 1.5 `Node` / `Region` — the entire IR (~115 LOC, `kernel/ir.py`)

V2 verbatim: one immutable node type, structured control flow as regions, no CFG.
The single most important negative invariant: **there is no field that can hold a
capture value.** `core.env` carries only `slot=k` in attrs; numba's
`ir.FreeVar(idx, name, value)` is excluded by construction.

```python
Attr = tuple[str, Hashable]                            # ints/str/Type/tuples thereof

@dataclass(frozen=True, slots=True, weakref_slot=True)
class Node:
    op:      str                          # dialect-namespaced: "core.add", "wgsl.frag_coord"
    type:    Type                         # node IS its SSA value; exactly one result type
    args:    tuple[Node, ...] = ()        # operands (producer references)
    regions: tuple[Region, ...] = ()      # nonempty only for core.if / core.for / core.call
    attrs:   tuple[Attr, ...] = ()        # compile-time constants — inside structural identity
    loc:     Loc | None = None            # (file, line, col, end_line, end_col); EXCLUDED from key
    @cached_property
    def key(self) -> bytes: ...           # memoized sha256 over (op, type, attrs, args.key, regions.key)

@dataclass(frozen=True, slots=True)
class Region:
    params: tuple[Node, ...]              # "core.param" typed binders, attrs=(("index", i),)
    body:   tuple[Node, ...]              # ORDERED; last element is core.yield
```

Exactly three region-carrying ops forever-until-forced: `core.if` (two pure
regions, identical yield types), `core.for` (counted loop, explicit fixed-type
carries), `core.call` (sub-program). Every additional one taxes every
transformation (~180 lines each, autodidax-measured, V5).

Capture classification is one lowering decision made syntactic:

- runtime capture → `core.env` with `attrs=(("slot", k),)` — re-marshaled per call
- `Literal`-lifted capture → `core.const` with the value in attrs — enters
  `Node.key`, recompiles per value, visibly auditable in the printed IR

### 1.6 `OpDef` — op metadata, never a Node subclass (~45 LOC, `kernel/ops.py`)

```python
@dataclass(frozen=True)
class OpDef:
    name:      str                        # "core.add"
    type_rule: Callable                   # (arg_types, attrs) -> Type, or raises TypeErrorAt(loc)
    traits:    frozenset[str] = frozenset()   # {"Pure", "Commutative", "LinearIn:0", ...}
    nregions:  int = 0
```

Core dialect table (~30 ops): arith (`add sub mul div pow neg`), compare, select,
`vec`/`extract`/`field`, `env`, `const`, `param`, `yield`, `if`, `for`, `call`,
`load`/`store` (boundary-only effects, V5), `cast`.

### 1.7 `Pat` / `RuleSet` — declarative rewrites + one driver (~120 LOC, `kernel/rewrite.py`)

The only pass mechanism in the system. tinygrad's `PatternMatcher` shape:

```python
@dataclass(frozen=True)
class Pat:                                # matches on op name, arg sub-patterns, attr guards
    op: str | tuple[str, ...]; args: tuple[Pat | Var, ...] | None = None
    guard: Callable[[Match], bool] | None = None

RuleSet = list[tuple[Pat, Callable[[Builder, Match], Node | None]]]

def rewrite(prog: Region, rules: RuleSet, *, fixpoint=True, name="") -> Region:
    ...  # post-order walk over ordered bodies; rebuild-on-change (immutability);
         # per-pass match logging under a debug flag; ~120 lines total
```

`Builder` is the ~30-line emit helper (`b.emit(op, *args, **attrs)`,
`b.const(v)`) shared by lowering, rewrites, and transformation rules.

### 1.8 `Registry` — the one explicit extension object (~60 LOC, `kernel/registry.py`)

No module-level registries anywhere (V3). The stdlib populates `DEFAULT` through
the same five public surfaces users get. `jit(fn, registry=...)` defaults to it.

```python
@dataclass
class Registry:
    ops:        dict[str, OpDef]
    rules:      dict[tuple[str, str], Callable]      # (op_name, aspect) -> rule fn; aspects:
                                                     # "lower_ast" | "eval" | "jvp" | "transpose" | "batch" | "unit" ...
    overloads:  dict[Hashable, list[tuple[type, Callable]]]   # callee key -> [(TargetToken, ovl_fn)]
    methods:    dict[tuple[type, str], list[tuple[type, Callable]]]  # (TypeClass, name) -> ...
    valuekinds: dict[type, ValueKind]                # Python type -> typeof/flatten/fingerprint/leaf_types
    backends:   dict[type, Backend]                  # TargetToken -> Backend record
    transforms: dict[str, Transform]
    def extend(self) -> "Registry": ...              # ChainMap layering: stdlib -> user -> session
```

Target tokens are a plain class lattice resolved by MRO — `Generic → CPU/GPU →
WGSL/CUDA/Metal/PyEval/C` — with no registries attached to the classes.

### 1.9 `ValueKind` — one registration, three views (~70 LOC + builtins, `kernel/valuekind.py`)

V4's contract. `leaf_types` is derivable from the `Type` alone (plans build
without values); `flatten` is the **only** extension callable permitted on the hot
path and carries a CI allocation budget.

```python
class ValueKind(Protocol):
    def typeof(self, v) -> Type: ...                     # full structural lattice (range-bucketed ints,
                                                         #   array flags/byteorder/writeable — hazard doc)
    def leaf_types(self, t: Type) -> tuple[Leaf, ...]:   # STATIC: from Type alone
    def flatten(self, v) -> tuple[object, ...]: ...      # DYNAMIC: values, same arity/order
    def fingerprint(self, v) -> Hashable: ...            # cheap structural tag; memoized where legal
```

Closed core-owned leaf vocabulary backends are total over: `ScalarLeaf(kind)`,
`BufferLeaf`, `ShapeLeaf(axis)`, `StrideLeaf(axis)`, `EnvLeaf` (nested closure,
recursive). Kernel ships scalar/tuple/Handle kinds; the ndarray kind lives in
`stdlib/` so the kernel has zero NumPy dependency.

### 1.10 `Backend` — a capability record (~sketch here; instances live outside the kernel)

```python
@dataclass(frozen=True)
class Backend:
    token:       type                                 # TargetToken subclass
    type_map:    dict[Type, str]                      # honest type -> backend spelling (narrowing lives HERE)
    code_for_op: dict[str, Callable[..., str]]        # the renderer's intrinsic table
    extra_rules: RuleSet                              # backend-specific legalizations
    render:      Callable[[Region, Backend], str]     # typed IR -> source text (~80-150 lines each)
    runtime:     Runtime                              # the physical half:

class Runtime(Protocol):
    def plan(self, env_types, arg_types, units, opts) -> PackPlan   # types only, once per cache entry
    def compile(self, source: str, opts) -> Artifact
    def make_launcher(self, artifact, plan) -> Callable             # the hot-path callable
```

Shared decomposition rules (e.g. `sqrt → pow(x, .5)`, `mean → sum/len`) are
parameterized by `backend.code_for_op.keys()`: a backend that names the op
natively skips the decomposition; one that doesn't gets it for free (V2/V3).

### 1.11 `PackPlan` / `SlotSpec` / `LeafPath` — logical→physical marshaling (~80 LOC, `kernel/pack.py`)

V4 verbatim. Built once per cache entry from types alone; interpreted by one
generic packer; per-backend `PhysicalDest` vocabulary (`UniformSlot(offset,fmt)`,
`KernelArg(index,ctype)`, `CField(offset,ctype)`, `PyArg(index)`) never appears in
kernel types.

```python
@dataclass(frozen=True)
class LeafPath:  root: str; index: int; sub: tuple[int, ...]     # env/arg -> nested env -> leaf
@dataclass(frozen=True)
class SlotSpec:  source: LeafPath; convert: Affine | None; dest: PhysicalDest
@dataclass(frozen=True)
class PackPlan:
    slots: tuple[SlotSpec, ...]; staging_size: int
    def pack_into(self, staging, leaves): ...        # ~30-line generic interpreter: struct.pack_into loop
```

Two-tier keying is law (V4): anything changing generated code (types, Literal
lifts, backend codegen params, generation) keys the artifact; anything changing
only bytes written (concrete units, width casts) keys a cheap PackPlan memo.

### 1.12 `FastRecord` — the compiled cache entry (~inside `kernel/cache.py`)

```python
@dataclass
class FastRecord:
    artifact: object            # pipeline / RawKernel / cfunc / exec'd Python fn
    extract:  Callable          # closure-compiled leaf extractor: attribute/index reads only
    plan:     PackPlan
    staging:  bytearray | list  # preallocated
    launch:   Callable          # artifact+plan pre-bound
```

Installed atomically under a per-key future (hazard doc: reentrant, forward-declared
for recursion). The thesis cache maps
`(template_fp, env_fp, arg_fp, backend_fp, generation) -> FastRecord`; the
artifact cache below maps `(Node.key, backend.token, codegen_flags) -> Artifact`,
curing M0's backend-params-missing-from-key fault.

---

## 2. Core hooks

Exactly five registration surfaces (V3). All operate at compile/typing time; the
hot path is registry-free. The five decorator signatures are the only stable
contract; IR internals, renderer internals, registry storage are unstable.

### Surface A — ops and rules (the rule matrix)

```python
def defop(name, *, type_rule, traits=(), nregions=0, registry=DEFAULT) -> str: ...
def rule(op: str, aspect: str, *, registry=DEFAULT):        # the single rule-registration door
    def deco(fn): registry.rules[(op, aspect)] = fn; return fn
    return deco
```

Aspects are columns: `"lower_ast"` (AST node → IR, how the accepted Python subset
widens without touching the lowerer driver), `"eval"` (reference semantics /
finite-difference harness), `"jvp"`, `"transpose"`, `"batch"`, `"unit"`. Columns
are declared empty day 1 (~10 lines); `MissingRule(op, aspect, loc)` names
exactly what to register. Widening syntax = adding a `(ast.NodeType, "lower_ast")`
entry — a column in the same matrix, not a sixth surface.

### Surface B — batteries via `@overload` (numba, stolen wholesale)

```python
@overload(math.sqrt, target=Generic, registry=DEFAULT)   # also @overload("clip") for DSL-only names
def sqrt_ovl(x):                                         # x is a pdum TYPE, not a value
    if isinstance(x, Scalar) and x.kind in ("f32", "f64"):
        def impl(x): return _intrinsic("math.sqrt", x)   # DSL-subset Python
        return impl
@overload_method(Record, "mean") ...
@overload_attribute(Vec, "xy") ...
```

Impls compile through the same `(FnType, arg_types)` cache as user code; per-op
artifact cache below deduplicates across users. Target-token MRO picks the most
derived registration (`@overload(np.mean, target=CUDA)` beats Generic on CUDA).
One call-typing resolution order, implemented once in the lowerer:
`__pdum_dsl__`-convertible → overloads (MRO-filtered) → methods/attrs (type-MRO)
→ error naming callee, arg types, and every declined entry.

### Surface C — type extensions (three registrations)

```python
@valuekind(Color)                     # 1. value -> Type + leaves + fingerprint
class ColorKind: ...
@overload_method(ColorType, "to_oklab")   # 2. methods, in the DSL subset
def color_to_oklab(c): ...
# 3. (only if the type has novel syntax) a ("lower_ast"/type) rule — rare
```

`leaf_types` declares logical leaves only; physical spelling (uniform offsets, C
struct fields) is the backend planner's job — the rule whose violation killed
numba's datamodel extensions twice (V3/V4).

### Surface D — backends

```python
def register_backend(b: Backend, *, registry=DEFAULT): registry.backends[b.token] = b
```

Budget per source-emitting backend: 50–300 renderer lines + 130–220 runtime lines
(tinygrad-measured). A backend needing more indicates a missing shared
decomposition rule, not a bigger backend.

### Surface E — transformations

```python
@dataclass
class Transform:
    tag: str                              # "grad", "vmap" — becomes Derived(tag, ...)
    passes: tuple[Callable, ...]          # Region -> Region drivers over the rule matrix

def custom_jvp(handle): ...               # the single escape hatch; identical shape for
def custom_vjp(handle): ...               #   user functions and backend-native gradients (MLX)
```

`grad(f)` / `vmap(f)` return ordinary `Handle`s with `Derived` template identity;
they flow into the same thesis cache unchanged.

### End-to-end walkthroughs

**Adding `sqrt`.** (1) `defop("math.sqrt", type_rule=same_float, traits={"Pure"})`;
(2) `@overload(math.sqrt)` above so the *name* resolves inside kernels;
(3) one `code_for_op` entry per backend that has it natively
(`"math.sqrt": lambda a: f"sqrt({a})"` for WGSL/C, `f"math.sqrt({a})"` for Python);
(4) one shared decomposition rule `Pat("math.sqrt", x) → b.emit("core.pow", x, b.const(0.5))`
auto-applied to backends lacking the table entry. Later, `@rule("math.sqrt","jvp")`
one-liner. Zero kernel files touched. ~15 lines total.

**Adding a `Color` record with a method.** `ColorType = Record("Color", (("r",f32),...))`;
`@valuekind(Color)` (typeof→ColorType, flatten→(r,g,b), leaf_types→3×ScalarLeaf,
fingerprint→one interned tag); `@overload_method(ColorType, "to_oklab")` with a
DSL-subset impl. A captured `Color` becomes three uniform slots on WGSL / three
struct fields on C, automatically, because the planner consumes leaves.
`c.to_oklab()` in a kernel types and inlines at phase B; `c.r` is
`core.field(attrs=(("name","r"),))`. ~40 lines, no kernel edits.

**Adding the C backend.** `class C(CPU)` token; `Backend(token=C,
type_map={f64:"double",...}, code_for_op≈60 entries, extra_rules=[vec→scalar-loop
rules], render=render_c ≈120 lines walking regions into C text,
runtime=CRuntime ≈150 lines: `plan` lays leaves into a `CField` struct
(numba `Record.make_c_struct` shape), `compile` = cc + ctypes, launcher = cfunc
call)`; `register_backend`. ~350 lines in `backends/c/`, zero kernel edits, and
every Generic-target overload battery works immediately.

**Adding `grad`.** Ship `transforms/ad.py`: a jvp pass driver (~100 lines,
builder-passing rules `JvpRule(b, primals, tangents, params) -> (outs, tangent_outs)`),
partial-eval + transpose (~400–500 lines, its own milestone), plus one
`@rule(op, "jvp")` / `@rule(op, "transpose")` per differentiable op.
`grad(f)` = `Handle` with `Derived("grad", base, (("wrt",0),))`. Pipeline per
derived key: base IR → transform passes → backend lowering, cached with the
artifact; the hot path never re-runs a transform. Captures are already typed env
slots, so `grad` w.r.t. a capture is expressible for free (V5).

---

## 3. Module map

### The kernel (`src/pdum/dsl/kernel/`) — CI-capped, target ≈1000, hard cap 1150

| Module | LOC | Responsibility |
|---|---|---|
| `types.py` | 65 | Type lattice + TemplateId sum type, frozen/serializable |
| `capture.py` | 85 | phase A: `make_handle`, `safe_cell`, SourceSnapshot memo, `Handle` |
| `valuekind.py` | 70 | ValueKind protocol + registry + scalar/tuple/Handle kinds + fingerprints |
| `ir.py` | 115 | `Node`, `Region`, content hash, `Builder`, structural verifier |
| `ops.py` | 45 | `OpDef`, traits, the ~30-op core dialect table |
| `rewrite.py` | 120 | `Pat`, `RuleSet`, the one rewrite driver, match logging |
| `lower.py` | 135 | phase B: snapshot-coherence check, `classify_names`, fused typing+lowering driver dispatching on `("lower_ast")` rules |
| `registry.py` | 60 | `Registry`, `DEFAULT`, target-token lattice + MRO resolution |
| `cache.py` | 85 | thesis cache + artifact cache, per-key futures, generation, per-tier miss counters, `no_compile` assertion mode |
| `pack.py` | 80 | `LeafPath`/`SlotSpec`/`PackPlan`, generic packer, `build_extractor` |
| `printer.py` | 60 | MLIR-flavored textual form (golden tests; migration insurance) |
| `api.py` | 45 | `@jit`, the `Handle.__call__` hot path, `NoSourceError`, `MissingRule` |
| **kernel subtotal** | **965** | |
| `backends/python.py` | 110 | first backend: renderer emitting Python source (~70) + exec/PyArg runtime (~40) — reference semantics, zero deps |
| **total in budget** | **1075** | CI line-budget check + PR delta bot from day one |

### Outside the kernel budget (attach via the five surfaces only)

| Component | LOC est. | Attaches via |
|---|---|---|
| `stdlib/` (ndarray ValueKind, math/vec/swizzle overloads, ~10 batteries: mean, clip, smoothstep, color) | 300–600 | Surfaces A/B/C on `DEFAULT` |
| `backends/wgsl/` (renderer 150 + uniform-layout planner & wgpu runtime 200 — M0's `layout.py` generalized) | ~350 | Surface D |
| `backends/c/`, `backends/cuda/` (CuPy RawKernel), `backends/mlx/` | ~300–400 each | Surface D |
| `transforms/` (vmap ~100 + rules; jvp ~100; transpose ~450; batch/jvp/transpose rule packs) | ~800 total, shipped in V5's cost order | Surface E + rule matrix columns |
| `tstring/` (einops-like mini-languages; PEP 750, 3.14) | per language | frontend sugar emitting core-dialect nodes; no new hook kind |
| `units/` (Dim, Quantity, unit rule column, Affine converters) | ~250 | Surfaces A/C + PackPlan converters |
| `tools/` (xDSL differential oracle ~200 pinned dev-extra; golden harness; per-stage grammar RuleSets) | ~350 | consumes printer output; never imported by kernel |

---

## 4. The dataflow

### 4.1 Phase A — capture (per closure construction, inside the user's hot loop)

```python
def make_handle(fn, kind, registry):
    code = fn.__code__
    snap = _SNAPSHOTS.get(code) or _take_snapshot(fn)          # memo per code object
    vals = tuple(safe_cell(c) for c in fn.__closure__ or ())   # co_freevars order (sorted)
    env_fp = tuple(registry.fingerprint(v) for v in vals)      # structural tags, memoized
    fntype = _FNTYPES.get((code, env_fp)) or _build_fntype(code, vals, registry)  # full typeof on memo miss
    return Handle(fntype, vals, env_fp, snap, kind, registry)
```

No parse, no IR, no compile, ever — phase A cannot fail on missing source
(`NoSourceError` is phase B's, with remediation text and the per-backend
`raw_kernel` escape hatch). Cost: cell reads + fingerprint probes, ~1–2 µs.

### 4.2 Phase B — call, miss path (once per type signature)

1. Full `typeof` on env+args (fingerprint memo missed or first sight).
2. **Snapshot coherence**: `compile(snap.text)` value-equal to template code
   (ignoring filename/firstlineno) or raise — stale source cannot compile.
3. `ast.parse(snap.text)` with real filename, re-based lines; select the
   `FunctionDef`, drop `decorator_list`.
4. `classify_names` — closed fate taxonomy: parameter / `core.env` slot /
   dialect intrinsic / other `Handle` as `FnType`-typed callee / explicitly-allowed
   folded constant / error-or-Literal-lift. Dependency-closure tags
   (`(name, id(__globals__))` + transitive Handle keys) recorded for drift checks.
5. `TypedLowerer` — ONE fused typing+lowering forward pass dispatching on
   `("lower_ast")` rules and the overload resolution order → flat, fully-typed,
   pure core-dialect program; every node carries `loc`. Assignments functionalized;
   effects confined to the boundary store sink.
6. If `TemplateId` is `Derived`: run the transform's passes (rule-matrix driven)
   → new typed program.
7. `rewrite(prog, backend.extra_rules + decompositions(backend.code_for_op.keys())
   + legalize_params)` — the last splits each logical `core.env` into N physical
   leaves (array → BufferLeaf + ShapeLeafs; scalar → uniform slot) and numbers
   slots. Per-frame flatten is structurally impossible after this (M0 fault cured).
8. Artifact-cache probe on `(prog.key, backend.token, codegen_flags)` — a hit
   skips render+compile entirely (e.g. same body, different capture *names*).
9. `backend.render(prog)` → source text; `runtime.compile` → artifact.
10. `runtime.plan(env_types, arg_types, units, opts)` → `PackPlan`;
    `build_extractor(leaf_paths)` → closure of pure attribute/index reads.
11. `FastRecord` assembled and installed under a per-key future (reentrant;
    recursive entries published forward-declared).

### 4.3 Phase B — call, HIT path (every hot-loop iteration)

```python
def __call__(self, *args):                                      # Handle.__call__, ~15 lines
    key = (self._tid_fp, self.env_fp, _fp_tuple(args),          # (1) key build
           _ACTIVE_BACKEND_FP, _GENERATION)
    rec = _RECORDS.get(key)                                     # (2) ONE dict probe
    if rec is None: rec = _miss(self, args, key)
    leaves = rec.extract(self.env, args)                        # (3) precompiled reads
    rec.plan.pack_into(rec.staging, leaves)                     # (4) struct.pack_into loop
    return rec.launch(rec.staging)                              # (5) write_buffer+draw / cfunc / fn
```

That is the entire hit path: **key build + value pack + launch, nothing else.**
No registry access, no typeof lattice walk, no AST, no IR object exists on this
path. Budget: single-digit µs pure Python, with two pre-shaped contract-preserving
escalations (exec-generated per-template binder à la Triton, then a narrow native
fastpath à la JAX).

### 4.4 Key anatomy and cache placement

```
thesis cache (kernel/cache.py)     (TemplateId_fp, env_fp, arg_fp, backend_fp, generation) -> FastRecord
   |  miss only
   v
artifact cache (kernel/cache.py)   (Node.key content hash, backend.token, codegen_flags) -> Artifact
   |  miss only
   v
render + backend compile
```

Template identity = code object by VALUE (unchanged notebook re-run hits; edits
miss via field inequality) wrapped in the `TemplateId` sum type; env/arg types via
structural fingerprints with full-`typeof` fallback; generation bumped by
redefinition and dependency-closure drift (refuse-or-recompile, never silently
stale). The Dynamo guard inventory is the key-completeness checklist; per-tier
miss counters name the differing component; `no_compile` assertion mode wraps the
render loop in tests.

---

## 5. Desiderata mapping

- **WebGPU backend** — first real backend: `Backend` record + uniform-layout
  planner generalizing M0's `layout.py`; renderer ~150 LOC (Surface D).
- **CUDA backend** — CuPy `RawKernel` runtime + CUDA-C renderer; leaves →
  `KernelArg` dests (Surface D, ~350 LOC).
- **Metal backend** — MLX custom-kernel runtime; its `raw_kernel` API is also the
  no-source escape-hatch floor (Surface D).
- **Python backend** — day-1 bundled renderer to Python source; zero-dependency
  reference semantics and test oracle.
- **C backend** — renderer + cc/ctypes runtime, `CField` planner (Surface D).
- **Records + methods** — `Record` type + `@valuekind` + `@overload_method`;
  field access is `core.field`; captured records flatten to leaves (Surface C).
- **Units** — `Quantity(rep, Dim)` with rational-exponent `Dim` in the artifact
  key; concrete units live in `SlotSpec.convert=Affine` at the pack tier — unit
  tweaks never recompile (V4 two-tier law).
- **Autodiff/vmap** — Surface E transforms over the same IR via rule-matrix
  columns; `Derived` identities flow into the unchanged thesis cache.
- **t-string mini-languages** — frontend sugar / overload callees emitting
  core-dialect nodes; template string is the source, so `NoSourceError`-immune.
- **Batteries** — `@overload` impls in the DSL subset, compiled per target,
  shared decompositions gated on backend op sets.
- **Value-dependent specialization** — explicit `Literal(v)` lift only: enters
  the type, surfaces as `core.const`, elides its pack slot. No implicit hints.
- **Live-coding invalidation** — code-object value equality (unchanged re-run
  hits) + generation counter + dependency-closure drift check in the key.

---

## 6. Day-1 vertical slice

**Goal:** the orbiting-disk demo (`docs/demos/disk.py`) reproduced on the new
kernel with **both** the WebGPU backend and the Python backend — proving the
backend seam, the thesis cache, and the hot path in one milestone.

Contents (≈1075 kernel + ~350 WGSL backend + ~120 stdlib slice):

1. `capture.py` + `api.py`: `@jit(kind="fragment")` returns a `Handle`; the demo
   loop rebuilds it every frame.
2. Language subset in `lower.py` + core `lower_ast` rules: float arithmetic,
   compare, `IfExp`, tuple-return → `core.vec`, local assignment, attribute
   swizzle via `@overload_attribute`, `FragCoord` intrinsic as a WGSL-dialect op
   registered from the backend package.
3. `backends/python.py`: renders the fragment program to a Python function;
   a 64×48 CPU render produces a reference image.
4. `backends/wgsl/`: renderer + uniform planner + wgpu runtime; `cx, cy, radius`
   captures → three `ScalarLeaf`s → one packed uniform buffer.
5. Acceptance gates, all CI-enforced from this milestone:
   - `compiles == 1` over 300 frames with moving `cx/cy` (thesis test), under
     `no_compile` assertion mode after frame 1;
   - WGSL 64×48 readback matches the Python backend image within tolerance
     (backend-seam differential test);
   - hit-path microbenchmark: `Handle.__call__` overhead budget asserted
     (< ~10 µs interpreter floor), `flatten` allocation budget;
   - golden printed IR at each stage (post-lower, post-legalize) via `printer.py`;
   - kernel line-budget check (≤1150) + per-file caps.
6. Explicitly deferred out of the slice: `core.for`, arrays, overloads beyond the
   swizzle, transforms, all other backends — each lands later through a surface,
   never through a kernel edit (that is the test of the architecture).

---

## 7. Risks and early detectors

1. **Hot-path cost creep.** Pure-Python key-build+pack drifts past the budget as
   fingerprints grow structural detail (array flags reads aren't free).
   *Detector:* the day-1 microbenchmark gate + allocation budget on `flatten`;
   escalation path (exec'd binder, native fastpath) pre-shaped so the contract
   never changes. *Early test:* 10k-frame disk demo; assert overhead p50/p99.
2. **`lower.py` becomes the monolith.** Fused typing+lowering is the natural sink
   for every new syntax/type rule; 135 lines becomes 1300 (numba's `typeinfer.py`
   failure mode). *Detector:* CI per-file cap + an extension-locality test: adding
   `sinh`, a record method, and a new statement form must produce zero kernel
   diffs. Run that test in M1, not M5.
3. **Three region ops prove too weak.** `break`/early-return/while (ray-marching)
   pressure the frozen control-flow set; ad-hoc op #4 taxes every transform
   column. *Detector:* M2 spike — port a bounded ray-march loop; if it needs
   unstructured exits, decide *then* between an early-exit-carrying `core.for`
   variant (one op, all columns budgeted) vs. frontend rejection, per V5's
   re-open trigger.
4. **WGSL shrinks the portable battery layer.** If WGSL restrictions (no i64,
   no recursion, uniformity rules) force per-backend forks of most overloads, the
   batteries economics collapse below numba's ~2:1 portable ratio. *Detector:*
   V3's M1 gate — port ~10 representative batteries to WGSL + Python and count
   forked lines.
5. **Cache-key incompleteness.** A forgotten key component (backend flag, unit
   mode, drifted global) silently reuses wrong artifacts — the worst failure class
   because it's invisible. *Detector:* property test generated from the Dynamo
   guard checklist: for every declared key-relevant input, mutate it and assert a
   tier-appropriate miss; per-tier miss counters name the differing component;
   dependency-drift check refuses or recompiles, never serves stale.

---

## 8. Deviations from the verdicts (flagged)

1. **First backend is a source-emitting renderer, not an eval-rules interpreter**
   (V2 budgeted a ~220-line reference interpreter; V5 wants day-1 `eval_rules`).
   The school says renderers emit source; a 110-line Python renderer gives
   reference semantics through `exec`, halves the mechanism count, and still
   serves as the finite-difference oracle by executing single-op programs. The
   `"eval"` rule column stays reserved; if per-op interpretation is later needed
   (stepping debugger, symbolic checks), it lands in `tools/` without kernel
   change. Cost of being wrong: ~150 lines in `tools/`.
2. **Printer trimmed to 60 lines and golden/grammar harness moved to `tools/`**
   (V2 counted 120 in-kernel). The textual form and day-1 golden tests survive;
   only the line-budget accounting moves. MLIR-flavored syntax kept, so the
   xDSL/MLIR migration insurance is intact.
3. **V3's 750–900-line "hook kernel" is compressed to ~380 in-kernel lines**
   (registry 60 + rewrite 120 + ops 45 + valuekind 70 + parts of lower/pack). The
   five surfaces and resolution order are implemented exactly as specified; the
   compression comes from sharing the rule matrix across all five and pushing
   overload *content* to `stdlib/`. If resolution corner cases (MRO ambiguity,
   ChainMap layering) overflow, the overflow is capped at `registry.py ≤ 150`
   before we revisit.
4. **ndarray ValueKind lives in `stdlib/`, not the kernel** — packaging, not
   architecture: the kernel imports neither NumPy nor any backend. Everything
   else follows the verdicts as written.

---

## Design lessons for pdum.dsl

- **Make the anti-pattern unrepresentable, don't police it.** `core.env` has no
  value field; `Node.attrs` is the only value-shaped slot and it *is* the
  `Literal` opt-in. The caching thesis then holds by type-checking the IR, not by
  review vigilance. Adopt this test day 1: grep-level assertion that no kernel
  type has a field typed `object` reachable from `Node` except `attrs`.
- **One rewrite driver, everything is a rule.** Lowering handlers, backend
  legalizations, decompositions, transform columns, even debug grammar checks are
  `(pattern, fn)` data against one 120-line driver. Every new mechanism proposal
  must first prove it cannot be a rule.
- **The hit path is a compiled artifact too.** `FastRecord.extract` and `launch`
  are built once per cache entry exactly like the GPU artifact; treat any code on
  the hit path that consults a registry or walks a lattice as a build-system bug.
  Enforce with the `no_compile` mode plus the microbenchmark gate, both in CI
  from the vertical slice onward.
- **Two caches, two laws.** Types/codegen params key the artifact tier; byte-level
  concerns (units, widths) key the pack tier. Every future feature must declare
  its tier in its design note — "which tier misses when this changes?" is the
  first review question.
- **Budgets are architecture.** The kernel cap (≤1150), per-file caps, per-backend
  caps (≤300 render + ≤220 runtime), and the extension-locality test (new
  feature ⇒ zero kernel diff) are the prime directive made mechanical. When a cap
  breaks, the design conversation happens *then*, with the overflow as evidence.
- **Sequence risk retirement, not features.** M1 = vertical slice + extension-
  locality test + 10-battery WGSL port; M2 = arrays/`core.for` + ray-march spike +
  C backend (cheapest second real backend, proves the planner vocabulary);
  M3 = vmap + jvp; M4 = transpose/grad; units after the first `Quantity` user
  exists. Each milestone's exit criterion is a detector from §7 going green.
