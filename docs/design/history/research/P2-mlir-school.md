# P2 — Architecture proposal: the MLIR school

*A complete architecture candidate for the pdum.dsl redesign, designed from the
MLIR school of thought: a tiny dialect-layered IR with regions for control flow,
progressive lowering from a surface dialect to backend dialects, and declarative
pattern rewrites — where extension means new dialects, new ops, and new patterns.
Inputs: V1–V5 verdicts, R1–R9 reports, `docs/desiderata.md`,
`design/dsl_caching_layer.md`, and the frozen M0 asset. July 2026.*

---

## 0. Positioning and verdict conformance

The MLIR school's condition ("use xDSL if the verdict supports it, else a
purpose-built mini-dialect infrastructure") is resolved by V2's evidence: **the
kernel is a purpose-built micro-IR** that adopts MLIR's *concepts* wholesale —
dialect-namespaced ops, the value/attribute split, regions with typed block
arguments and value yields, op traits, declarative rewrite patterns, an
MLIR-flavored textual form — while owning every line the cache key rests on.
xDSL survives as a pinned dev-only differential-testing oracle (V2 §5.8).

**No verdict is deviated from.** The school adds two sharpenings, flagged here
because they go slightly beyond the verdicts' letter:

1. **Stage legality (ConversionTarget-lite), always on.** V2 puts per-stage
   grammar specs under a debug flag. This proposal promotes a *cheap subset* of
   that to an always-on invariant: every lowering stage declares the set of op
   namespaces legal at its output, and the rewrite driver errors (naming the op
   and its source `loc`) if an illegal op survives fixpoint. This is MLIR's
   conversion-target legality reduced to a ~30-line namespace check; the full
   per-stage grammar RuleSets remain debug-only exactly as V2 specifies.
   Justification: progressive lowering is only auditable if "which dialects may
   exist at stage N" is machine-checked, and the check is O(ops) string-prefix
   tests — free at pdum scale.
2. **`Dialect` as a bundling value, not a sixth hook.** V3's law is "five
   registration surfaces and no sixth." The MLIR school's unit of shipping is
   the dialect, so this proposal adds a `Dialect` dataclass that *aggregates*
   registrations across the five surfaces (ops + rules + overloads + kinds) and
   installs them in one `registry.install(dialect)` call. It introduces no new
   registration semantics — it is a tuple of surface-(a)–(e) entries with a
   name. The five decorator signatures remain the only stable contract.

Everything else is the verdicts assembled into one machine: V1's
reflection-capture + AST-lowering frontend, V2's `Node/Region` micro-IR and
two-layer cache stack, V3's Registry + op×aspect rule matrix, V4's
ValueKind/PackPlan/FastRecord marshaling triple, V5's transformations as
rule-matrix passes with derived template identities.

### The dialect ladder (the school's one-picture summary)

```
                    frontends (satellites)
   Python AST lowerer      t-string sub-parsers      raw_kernel escape hatch
          \                       |                          |
           v                      v                          |
   ┌─────────────────────────────────────────────┐           |
   │ SURFACE: core.* (+ domain dialects:         │           |
   │   rec-sugar folds to core; units.*; ein.*)  │           |
   └─────────────────────────────────────────────┘           |
          | transform passes (grad/vmap: aspect columns)     |
          v                                                  |
   ┌─────────────────────────────────────────────┐           |
   │ MID: core.* only  (legal = {core})          │           |
   │   simplify + decompositions gated on the    │           |
   │   target backend's code_for_op keys         │           |
   └─────────────────────────────────────────────┘           |
          | backend legalize_params RuleSet                  |
          v                                                  |
   ┌─────────────────────────────────────────────┐           |
   │ ABI: core.* ∖ {core.env}  ∪  abi.slot       │           |
   │   (legal = {core, abi}); slot table ⇒       │           |
   │   PackPlan; Node.key ⇒ artifact-cache key   │           |
   └─────────────────────────────────────────────┘           |
          | backend render RuleSet (code_for_op tables)      v
          v                                            source string
   WGSL / CUDA-C / Metal / C / Python source  ──►  backend runtime compiles+launches
```

Every arrow is a `RuleSet` run by the same ~170-line rewrite driver. Every box
boundary is a legality declaration. The type-keyed cache sits *above* this whole
picture (the ladder runs only between a tier-1 miss and a `FastRecord`).

---

## 1. Core primitives

The kernel's essential data structures. All are frozen dataclasses; all
key-participating ones serialize structurally (V4's disk-cache requirement).

| # | Primitive | Responsibility | LOC |
|---|---|---|---:|
| 1 | `Type` family | structural types; the sole vocabulary of cache identity | 70 |
| 2 | `TemplateId` | code identity as a sum type: base code object or derived | 15 |
| 3 | `FnType` | the function type `(TemplateId, env_types)` | 10 |
| 4 | `Handle` / `SourceSnapshot` | phase-A product: fntype + env values + source | 40 |
| 5 | `Node` / `Region` | the entire IR data model, with memoized content hash | 140 |
| 6 | `OpDef` + traits | ops as data; generic verifier; dialects as dict merges | 90 |
| 7 | `Pat` / `RuleSet` / `Stage` | declarative rewrites + stage legality | 170 |
| 8 | `Registry` / `Dialect` | the one explicit extension store; five surfaces | 120 |
| 9 | `Leaf` / `ValueKind` | typeof/flatten/fingerprint per Python type | 55 |
| 10 | `LeafPath` / `SlotSpec` / `PackPlan` | logical leaves → physical destinations; generic packer | 65 |
| 11 | `Backend` / `Runtime` | capability record; codegen/runtime seam | 30 |
| 12 | `FastRecord` + engine caches | tier-1 thesis cache, tier-2 artifact cache, hit path | 120 |

Concrete field lists:

```python
# ── kernel/ir/types.py (~70) ────────────────────────────────────────────────
class Type: ...                                  # open set; all frozen dataclasses
@dataclass(frozen=True) class Scalar(Type):   kind: str            # "f32" "f64" "i32" "i64" "u32" "bool"
@dataclass(frozen=True) class Vec(Type):      elem: Scalar; n: int
@dataclass(frozen=True) class Array(Type):    dtype: Type; ndim: int; layout: str
                                              # byteorder/writeable per hazard doc; shapes are RUNTIME data
@dataclass(frozen=True) class Record(Type):   name: str; fields: tuple[tuple[str, Type], ...]
@dataclass(frozen=True) class FnRef(Type):    fntype: "FnType"     # a Handle passed as a value
@dataclass(frozen=True) class LiteralType(Type): base: Type; value: Hashable   # THE value opt-in (V4 §3.4)
# future, zero schema change: Quantity(Type): rep: Scalar; dim: Dim   (rational exponents)

# ── kernel/capture.py (~110 incl. make_handle/snapshot) ────────────────────
TemplateId = CodeId | DerivedId
@dataclass(frozen=True) class CodeId:     code: CodeType            # VALUE-compared (co_code/co_consts/...)
@dataclass(frozen=True) class DerivedId:  tag: str; base: TemplateId; static: tuple[Hashable, ...]
                                          # ("grad", CodeId(...), (0,)) — V5's derived identity
@dataclass(frozen=True) class FnType:     template: TemplateId; env_types: tuple[Type, ...]

@dataclass(frozen=True) class SourceSnapshot: text: str; filename: str; firstlineno: int; qualname: str

class Handle:                       # NOT frozen: env is per-instance runtime data
    fntype:  FnType                 # cache identity — types only
    env:     dict[str, object]      # capture VALUES, co_freevars order; never in any key
    src:     SourceSnapshot | None  # taken at decoration; None ⇒ NoSourceError at phase B
    kind:    str                    # "device" | "fragment" | "compute" | ...
    fp_head: Hashable               # memoized (template fp, env fingerprint) fast-key half

# ── kernel/ir/node.py (~140) ────────────────────────────────────────────────
Attr = tuple[str, Hashable]
@dataclass(frozen=True, slots=True, weakref_slot=True)
class Node:
    op:      str                        # dialect-namespaced: "core.add", "abi.slot", "wgsl.sample"
    type:    Type                       # the node IS its SSA value; single result (projections for multi)
    args:    tuple["Node", ...] = ()    # operands = RUNTIME data — never in the key's value dimension
    attrs:   tuple[Attr, ...]   = ()    # compile-time constants — inside structural identity
    regions: tuple["Region", ...] = ()  # nonempty only for core.if / core.for / core.func / core.call
    @cached_property
    def key(self) -> bytes: ...         # recursive sha256 over (op, type, attrs, args.key, regions.key)

@dataclass(frozen=True, slots=True)
class Region:
    params: tuple[Node, ...]            # "core.param" typed binders, attrs=(("index", i),)
    body:   tuple[Node, ...]            # ORDERED (effect order is free); ends in core.yield
# side channel: LOCS: WeakKeyDictionary[Node, Loc]  — loc spans NEVER enter Node.key (V1 §6, V2)

# ── kernel/ir/ops.py (~90) ──────────────────────────────────────────────────
@dataclass(frozen=True)
class OpDef:
    arity: int | str                    # 2 | "variadic" | "3+carries"
    n_regions: int = 0
    attr_schema: tuple[tuple[str, type], ...] = ()
    traits: frozenset[str] = frozenset()      # "pure","commutative","terminator","linear_in:0",...
    verify: Callable | None = None
CORE_OPS: dict[str, OpDef] = {...}      # a dialect IS a dict[str, OpDef]; Context = dict merge

# ── kernel/ir/rewrite.py (~170) ─────────────────────────────────────────────
@dataclass(frozen=True) class Pat:  ops: frozenset[str] | None; type: Type | None
                                    args: tuple["Pat|str", ...] | None; name: str | None
class RuleSet:                       # list[(Pat, fn)] indexed by root op; composable with +
class Stage(NamedTuple):
    name:  str                       # "simplify", "legalize_params", "render"
    rules: RuleSet
    legal: frozenset[str]            # OP NAMESPACES legal at output: {"core"}, {"core","abi"}
def run_stage(region, stage, ctx) -> Region:
    out = rewrite(region, stage.rules)              # bottom-up fixpoint, recurses into regions
    check_legal(out, stage.legal)                   # ConversionTarget-lite: error names op + loc
    return out

# ── kernel/marshal.py (~120) ────────────────────────────────────────────────
class Leaf: ...                      # CLOSED core-owned logical vocabulary (V4 §3.1)
@dataclass(frozen=True) class ScalarLeaf(Leaf): kind: str
@dataclass(frozen=True) class BufferLeaf(Leaf): ...
@dataclass(frozen=True) class ShapeLeaf(Leaf):  axis: int
@dataclass(frozen=True) class StrideLeaf(Leaf): axis: int
@dataclass(frozen=True) class EnvLeaf(Leaf):    ...          # nested closure env (recursive)

class ValueKind(Protocol):           # ONE registration per Python type → three views
    def typeof(self, v) -> Type: ...                          # the only artifact-key input
    def leaf_types(self, t: Type) -> tuple[Leaf, ...]: ...    # STATIC: from Type alone
    def flatten(self, v) -> tuple[object, ...]: ...           # DYNAMIC: hot path; no allocation beyond a tuple
    def fingerprint(self, v) -> Hashable: ...                 # structural fast tag (arity/range/flags)

@dataclass(frozen=True) class LeafPath: root: Literal["env","arg"]; index: int
                                        sub: tuple[str | int, ...]; leaf: int
@dataclass(frozen=True) class SlotSpec: source: LeafPath; convert: "Converter|None"; dest: object
                                        # dest is BACKEND-defined: UniformSlot/KernelArg/CField/PyArg
@dataclass(frozen=True) class PackPlan:
    slots: tuple[SlotSpec, ...]; staging_size: int
    def pack_into(self, staging, leaves): ...                 # ONE generic interpreter, ~40 lines

# ── kernel/registry.py (~120) ───────────────────────────────────────────────
@dataclass
class Registry:
    ops:        dict[str, OpDef]
    rules:      dict[tuple[str, str], Callable]        # (op, aspect): "type","eval","jvp","transpose","batch","unit"
    overloads:  dict[object, list[OverloadEntry]]      # np.mean / "mean" -> entries
    methods:    dict[tuple[type, str], list[OverloadEntry]]
    attrs:      dict[tuple[type, str], list[OverloadEntry]]
    typeof:     TypeofDispatcher                       # singledispatch + __pdum_type__ duck hook
    kinds:      dict[type, ValueKind]
    backends:   dict[type[Target], "Backend"]
    transforms: dict[str, "Transform"]
    def extend(self) -> "Registry": ...                # ChainMap layering; stdlib -> user -> session
    def install(self, d: "Dialect") -> None: ...       # bundle install (school sharpening #2)

@dataclass(frozen=True)
class Dialect:                                         # a NAMED TUPLE OF surface entries — no new semantics
    name: str
    ops: dict[str, OpDef] = ...
    rules: dict[tuple[str, str], Callable] = ...
    overloads: tuple[...] = (); kinds: tuple[...] = ()

# ── kernel/engine.py (~120) ─────────────────────────────────────────────────
@dataclass
class Backend:                                          # surface (d) capability record
    target:      type[Target]                           # WGSL/CUDA/Metal/PyEval/C token (class lattice, MRO)
    type_map:    dict[Type, str]
    code_for_op: dict[str, str | Callable]              # surface (a); ALSO the capability declaration
    legalize:    RuleSet                                 # backend-local pre-render rewrites
    plan:        Callable[..., PackPlan]                 # (env_types, arg_types, units, opts) -> PackPlan
    render:      Callable[[Node], str]                   # a RuleSet-driven emitter; types -> source
    runtime:     "Runtime"                               # compile(source)->Artifact; launch; SEPARATE seam
    raw_kernel:  Callable                                # MLX-shaped escape hatch / NoSourceError remedy

@dataclass
class FastRecord:                                        # one per tier-1 entry; the whole hit path
    artifact: object; extract: Callable; plan: PackPlan; staging: bytearray | None; launch: Callable

# caches (all in engine.py):
#   tier-1 (thesis):   (FnType, arg_types, target, codegen_opts, generation) -> FastRecord   [LRU + per-key future]
#   tier-1.5 (plan):   (FnType, arg_types, concrete_units, byte_opts)        -> PackPlan     [cheap memo]
#   tier-2 (artifact): (Node.key, renderer.name, renderer_flags)             -> Artifact     [+ optional disk]
```

Constitutional invariants carried by these primitives:

- **The IR cannot hold a capture value.** There is no node constructor taking a
  runtime value: captures are `core.env slot=k : T` ops (operand-free), and only
  `LiteralType`-originated constants become `core.const value=v` attrs. numba's
  `ir.FreeVar(idx, name, value)` is excluded by type (V1 lesson 4, V2 lesson 2).
- **attrs ∈ Node.key, args ∉ Node.key's value dimension.** "What recompiles"
  has a syntactic answer, auditable in printed IR.
- **`loc` spans and analysis results live in weak side tables**, never in
  structural identity.
- **Exactly three region-carrying ops** — `core.if` (two pure sub-regions with
  identical yield types), `core.for` (counted, explicit typed carries),
  `core.call`/`core.func` — each addition taxes every transformation ~150–200
  lines and is treated as a constitutional amendment (V2/V5).

---

## 2. Core hooks

All extension flows through the explicit `Registry` (passed as a parameter:
`jit(fn, registry=DEFAULT)`; the stdlib is a client of the same five public
surfaces — no module-level registries anywhere, per V3).

### The five surfaces (registration API)

```python
# (a) ops + per-backend intrinsic spellings
sqrt = defop("core.sqrt", arity=1, traits={"pure"},
             type_rule=float_unary, registry=DEFAULT)
backend.code_for_op["core.sqrt"] = "sqrt({0})"          # one line per backend; absence ⇒
                                                        # shared decomposition rules fire
                                                        # (gated on code_for_op.keys())
@rule("core.sqrt", "eval")                              # aspect column entries
def _(x): return math.sqrt(x)

# (b) batteries — @overload family; impls in the DSL subset, compiled per target
@overload(np.mean)                                      # also @overload("mean")
def mean_ovl(a):                                        # a is a TYPE; runs at typing time
    if isinstance(a, Array): ...
        def impl(a): ...                                # DSL-subset Python
        return impl
    return None                                         # decline
@overload_method(Array, "mean")(mean_ovl)
@overload(np.mean, target=CUDA)                         # per-backend fast path wins by token MRO
def mean_cuda(a): ...

# (c) type extensions — three registrations
@typeof_impl.register(MyClass)
def _(v): return MyType(...)                            # or duck: MyClass.__pdum_type__
register_kind(MyClass, MyKind())                        # typeof + leaf_types + flatten + fingerprint
@overload_method(MyType, "frob")
def frob_ovl(x): ...

# (d) backends — capability record under a target-token lattice
registry.register_backend(Backend(target=C, type_map=..., code_for_op=...,
                                  legalize=..., plan=..., render=..., runtime=...))

# (e) transformations — aspect columns + pass drivers + escape hatches
registry.transforms["batch"] = VmapPass()               # ~100-line generic driver
@rule("core.sin", "jvp")
def sin_jvp(b, primals, tangents, params): ...          # builder-passing; emits eqns
@custom_jvp                                             # user/backend-native derivative,
def turbulence(x): ...                                  # same registration shape as built-ins
@turbulence.defjvp
def turbulence_jvp(primals, tangents): ...
```

Resolution orders, implemented once (V3 §implication 2): call typing walks
`__pdum_dsl__`-convertible → `registry.overloads[callee]` filtered by
target-token MRO (most-derived first) → `methods`/`attrs` with type-MRO
fallback → error naming callee, arg types, and declined entries. Missing
transformation rules raise `MissingRule("op 'x' has no 'jvp' rule — grad(f)
touched it at loss.py:12")`.

### Walkthrough 1 — adding `sqrt` end to end

1. `defop("core.sqrt", ...)` with a type rule (surface a): ~3 lines.
2. `@overload(math.sqrt)` maps the Python name into the DSL (surface b): ~5 lines.
3. One `code_for_op` entry per backend that has it natively: `"sqrt({0})"`
   (WGSL), `"sqrtf({0})"` (C), `"sqrt({0})"` (CUDA): 1 line each. A backend
   without it triggers the shared `pow`-based decomposition rule automatically.
4. `@rule("core.sqrt","eval")` for the Python backend/reference semantics: 1 line.
5. Later, `@rule("core.sqrt","jvp")` when AD ships: ~3 lines. Nothing else moves.

### Walkthrough 2 — a `Color` record with a method

```python
Color = record_type("Color", r=f32, g=f32, b=f32)   # stdlib sugar over surface (c):
# generates: Record("Color", (("r",f32),("g",f32),("b",f32)))
# + ValueKind: leaf_types -> (ScalarLeaf('f32'),)*3 ; flatten -> (c.r,c.g,c.b)
# + constructor overload: Color(r,g,b) -> core.construct : Record; field access -> core.field name="r"

@overload_method(Color.type, "to_oklab")
def to_oklab(c):
    def impl(c):
        l = 0.4122 * c.r + 0.5364 * c.g + 0.0514 * c.b
        ...
        return Color(l_, m_, s_)
    return impl
```

`col.to_oklab()` in user code types via the methods table, inlines the impl
through the same `(FnType, arg_types)` cache, and lowers to pure `core.*` ops.
A captured `Color` marshals as three `ScalarLeaf`s; the WGSL planner packs them
into the uniform buffer with 16-byte struct rounding — offsets never touch the
frontend `Type` (V4 §3.2). Zero kernel changes.

### Walkthrough 3 — adding the C backend

```python
c_backend = Backend(
    target      = C,                                     # token in the CPU branch of the lattice
    type_map    = {Scalar("f32"): "float", Scalar("i64"): "int64_t", ...},
    code_for_op = {"core.add": "({0}+{1})", "core.sqrt": "sqrtf({0})", ...},  # ~30 entries
    legalize    = RuleSet([bool_to_int, select_to_ternary]),
    plan        = c_planner,        # leaves -> CField(offset,ctype)/PyArg via
                                    # numba Record.make_c_struct-style layout (~150 lines)
    render      = c_render,         # RuleSet emitter: core+abi nodes -> one C function (~150 lines)
    runtime     = CcRuntime(),      # cc -shared / ctypes.CDLL; launch = ctypes call (~180 lines)
)
registry.register_backend(c_backend)
```

Total ~350–500 lines, no kernel edits, tinygrad-calibrated (V3 §d). Portable
batteries written via `@overload` compile for it immediately; decomposition
rules fill any `code_for_op` gaps.

### Walkthrough 4 — adding `grad`

```python
def grad(f: Handle, argnums=0) -> Handle:
    return derive(f, tag="grad", static=(argnums,))
    # -> Handle(FnType(DerivedId("grad", f.fntype.template, (argnums,)), f.env_types),
    #           env=f.env, src=f.src)   — an ORDINARY template; phase A stays compile-free
```

`grad(loss)` rebuilt every loop iteration is a **cache hit**: `DerivedId` is a
value, compared structurally (V5 §4). On the tier-1 miss, the pipeline inserts
the transform stage: base IR → type inference → `jvp_pass ∘ partial_eval ∘
transpose_pass` (each a ~100-line generic driver consulting its aspect column)
→ backend lowering. Transformed IR is cached with the artifact; the hot path
never re-runs a pass. Backend-native gradients (MLX) enter only as
`custom_jvp/custom_vjp` registrations whose lowering calls a backend kernel —
semantic ownership of `grad` stays in the rule table (V5).

### Walkthrough 5 — an einops t-string (no new hook kind)

`t"b h w c -> b (h w) c"` is a frontend: the template string *is* the source
(no snapshot hazard), its sub-parser runs at phase B and emits `core.*` nodes
(or `ein.*` ops from an `ein` Dialect that lowers to core in the surface→mid
stage). It registers as an overload callee or frontend sugar — V3 §implication 8.

---

## 3. Module map

**Kernel — the budgeted ~1000 lines.** The kernel is the substrate that owns
every correctness invariant: IR + rewrites + registry + capture + marshal
vocabulary + the caches. It is frontend-agnostic (consumes `core.*` Nodes from
any producer) and backend-agnostic (emits via capability records).

| Module | LOC | Responsibility |
|---|---:|---|
| `kernel/ir/types.py` | 70 | frozen structural `Type` set (open, serializable) |
| `kernel/ir/node.py` | 140 | `Node`/`Region`, memoized `key`, `Builder`, loc side table |
| `kernel/ir/ops.py` | 90 | `OpDef` + traits, `CORE_OPS`, generic verifier |
| `kernel/ir/rewrite.py` | 170 | `Pat`/`RuleSet`, fixpoint driver, `Stage` legality, match log |
| `kernel/ir/printer.py` | 110 | MLIR-flavored text; golden-file substrate; xDSL escape hatch |
| `kernel/registry.py` | 120 | `Registry` + `extend()` + `Dialect` bundle + `defop`/`rule` |
| `kernel/capture.py` | 110 | `TemplateId`/`FnType`/`Handle`, `make_handle`, source snapshot, `derive()` |
| `kernel/marshal.py` | 120 | `Leaf` vocabulary, `ValueKind` registry, `LeafPath`/`SlotSpec`/`PackPlan`, generic packer, extractor builder |
| `kernel/engine.py` | 120 | tier-1/1.5/2 caches, fingerprints, `FastRecord`, per-key futures, generation, miss counters, `no_compile` mode |
| **kernel total** | **~1050** | |

**Satellites — outside the kernel budget; each attaches through a named seam
and is independently iterable.**

| Component | LOC | Attaches via |
|---|---:|---|
| `front/lower.py` — AST parse, fate classification, fused typing+lowering | ~900 | consumes `Registry`, emits `core.*`; snapshot-coherence check on miss |
| `front/overload.py` — `@overload` family + typing-time resolution | ~250 | surface (b); consulted by the lowerer's call-typing step |
| `std/kinds.py` — builtin ValueKinds (scalars, ndarray, tuple, Handle) | ~180 | surface (c) into `DEFAULT` |
| `std/records.py` — `record_type` sugar | ~120 | surfaces (b)+(c) |
| `std/batteries/*` — mean, clip, smoothstep, color, … | grows | surface (b), DSL subset, portable-by-default |
| `backends/python/` — eval-rule interpreter + PyArg planner + runtime | ~300 | surface (d); reference semantics + FD test oracle |
| `backends/wgsl/` — planner (M0 `layout.py` generalized), renderer, wgpu runtime | ~550 | surface (d) |
| `backends/c/`, `backends/cupy/`, `backends/mlx/` | ~350–500 ea | surface (d), later |
| `transforms/` — vmap (~100), jvp (~100), reverse = partial-eval + transpose (~450) | staged | surface (e) aspect columns; `derive()` identities |
| `dialects/units.py` — `Quantity`/`Dim`, unit aspect, pack converters | ~250 | Dialect bundle; tier-1.5 plan memo |
| `dialects/ein.py` — t-string einops frontend | ~300 | overload callee emitting core ops |
| `dev/xdsl_oracle.py` — pdum-IR→xDSL translator, differential tests | ~200 | dev extra only; never imported by kernel |

Day-one culture (V2 §5.7): CI line budget with per-PR delta on the kernel,
golden printed IR at every stage, rewrite-match logging under a debug flag,
per-stage grammar spec RuleSets in debug mode, per-tier cache-miss counters and
an allocation-budget test on `flatten`.

---

## 4. The dataflow

### Phase A — capture (at decoration, inside the user's hot loop; compile-free)

```python
@jit(kind="fragment")          # runs EVERY frame in the driving pattern
def shader(): ...
```

1. `code = fn.__code__` — identity in hand; no parse, no bytecode.
2. `vals = tuple(safe_cell(c) for c in fn.__closure__ or ())` (empty-cell guard).
3. `env_types = typeof_tuple(vals)` — via ValueKind fingerprints memoized per
   code object: for a rebuilt closure this is a `co_freevars` zip + per-value
   structural tags (int range bucket, tuple arity, array flags), not a lattice walk.
4. `fntype = FnType(CodeId(code), env_types)`; `env = dict(zip(co_freevars, vals))`.
5. `src = snapshot_source(fn)` — eager, at the only moment `linecache` and the
   code object are guaranteed coherent; may be empty (phase A still succeeds).
6. Return `Handle`. Cost: a few dict/tuple builds + fingerprints. **No failure
   modes except typeof errors; never compiles.**

### Phase B — call: the hot-loop cache hit (the entire per-call path)

```python
def __call__(handle, *args):
    fp  = (handle.fp_head, fingerprint_tuple(args), ACTIVE.target_fp)   # 1. key build
    rec = _tier1.get(fp)                                                # 2. one dict probe
    if rec is None: return _miss(handle, args)                          #    (slow path below)
    leaves = rec.extract(handle.env, args)                              # 3. precompiled LeafPath closures
    rec.plan.pack_into(rec.staging, leaves)                             # 4. generic struct.pack_into
    return rec.launch(rec.staging, leaves)                              # 5. write_buffer+draw / kernel launch
```

That is the whole hit path: **key build + value pack + launch — no AST, no IR,
no registry, no layout derivation, no allocation beyond tuples.** Budget:
single-digit µs pure Python; pre-shaped contract-preserving escalations are the
exec-generated per-template binder (Triton) then a narrow native fastpath (JAX)
(V4 §3.6). Fingerprint miss falls back to full `typeof` (sound fast key).

### Phase B — the miss (once per type signature, per backend, per generation)

1. **Full key**: `(FnType, arg_types, target_token, codegen_opts, generation)`
   — the Dynamo-checklist-complete tier-1 key; code identity is inside `FnType`
   via value-compared `CodeId` (unchanged notebook re-run = hit; edit = miss).
   Dependency-closure hash (folded globals `(name, id(__globals__))`, callee
   Handle keys) checked here; drift ⇒ refuse/recompile, never silently stale.
2. **Per-key future** installed (`compiling`/`ready`); `generation` read once.
3. **Coherence check**: `compile()` the snapshot, require value-equality with
   the template code object (ignoring filename/firstlineno); empty snapshot ⇒
   `NoSourceError` naming the function and remedies (file/IPython cell, or
   `backend.raw_kernel`).
4. **Lower** (`front/lower.py`): `ast.parse` → fate classification (parameter /
   `EnvVar` slot / registry intrinsic / Handle callee / allowed folded constant
   / error-or-Literal-lift) → one fused typing+lowering forward pass emitting
   `core.*` with `loc` side-channel; overload/method resolution consults the
   Registry. Captures become `core.env slot=k : T`; `Literal` lifts become
   `core.const value=v` (in `Node.key` by construction, slot elided).
5. **Transform stage** (only for `DerivedId` templates): aspect-column passes
   over the typed IR; transformed IR cached with the artifact.
6. **Mid stage** (`legal={"core"}`): simplify + decompositions gated on the
   target backend's `code_for_op.keys()`.
7. **ABI stage** (`legal={"core","abi"}`): backend `legalize_params` RuleSet
   splits each logical `core.env`/arg into N physical `abi.slot` ops (array →
   buffer + shape words; scalar → uniform slot); slot numbering; the slot table
   plus the backend planner yields the `PackPlan` (tier-1.5 memo, additionally
   keyed on concrete units/byte opts). The inliner records a `LeafPath` per slot.
8. **Artifact**: probe tier-2 on `(Node.key, renderer.name, renderer_flags)`;
   miss ⇒ render stage (code_for_op tables + type_map, a RuleSet emitter) ⇒
   `runtime.compile(source)` (optional disk layer keyed on source text).
9. **FastRecord** assembled — artifact + closure-compiled extractor (from the
   LeafPaths; built once) + plan + staging + launcher — and installed
   atomically under the future. Subsequent frames take the 5-step hit path.

The type-keyed cache therefore sits entirely in `kernel/engine.py`, above the
IR; the dialect ladder runs only between step 4 and step 8, exactly once per
tier-1 key.

---

## 5. Desiderata mapping

| Desideratum | Where it lands |
|---|---|
| WebGPU/WGSL backend | first `Backend` record: M0 `layout.py` → planner, renderer RuleSet (~150), wgpu runtime; uniform staging + `write_buffer` launch |
| CUDA via CuPy | `Backend(target=CUDA)`: `KernelArg`/`StructField` dests (cupy CArray struct trick as a dest decision), `RawKernel` runtime; ~400 lines, zero kernel edits |
| Metal via MLX | `Backend(target=Metal)`: `mx.fast.metal_kernel` as both runtime floor and `raw_kernel` hatch; leaves map to MLX arrays |
| Python backend | eval-rule interpreter over `core.*` (aspect column "eval"), `PyArg` planner; zero-dependency floor, reference semantics, FD oracle for AD rules |
| C backend | walkthrough 3: render to C source + cc/ctypes runtime + `make_c_struct` planner |
| records + methods | `Record` core type + `core.construct`/`core.field` ops + `record_type` sugar + `@overload_method`; leaves = per-field scalars |
| units | `Quantity(rep, Dim)` with rational-exponent `Dim` in the artifact key; canonical-basis emission; `Affine(scale, offset)` converters applied in `pack_into` at the tier-1.5 plan memo — unit tweaks miss only the µs tier; escape hatch = the same `Literal` lift |
| autodiff / vmap | surface (e): empty aspect columns + `linear_in` traits reserved day 1; `derive()` identities; ship order type → vmap (~100) → jvp (~100) → reverse (~450) |
| t-string mini-languages | additional frontends emitting `core.*` (template string is the source); optionally a `Dialect` (`ein.*`) lowered by patterns in the surface→mid stage |
| batteries | `@overload` impls in the DSL subset, target-token MRO, compiled through the same `(FnType, arg_types)` cache; decompositions gated on `code_for_op.keys()`; CI portable:bound ratio metric (floor 2:1) |
| value-dependent specialization opt-in | `LiteralType(base, value)`: enters the key, surfaces as `core.const` attr, elides its `PackPlan` slot — one lift, three coupled effects; no implicit hints, never `id()`-keyed |
| live-coding invalidation | code-object value-equality (unchanged re-run hits, edit misses), global `generation` in the tier-1 key (sledgehammer accepted for M1; dependency graph later), dependency-closure hash checked at call time, per-tier miss counters naming the differing component |

---

## 6. Day-1 vertical slice

**Reproduce the orbiting-disk demo (`docs/demos/disk.py`) on the new kernel,
running unmodified user code on BOTH the WebGPU backend and the pure-Python
backend — proving the backend seam, the dialect ladder, and the thesis cache in
one milestone.**

Language subset needed: float arithmetic and comparisons, tuple construction,
attribute access (`FragCoord.xy` swizzle), conditional expression (lowered to
`core.if` with two pure regions and value yields), scalar captures.

Build list (~2.5 kloc total):

1. The full kernel (~1050): all nine modules — none is optional, all are exercised.
2. `front/lower.py` subset (~500 of the eventual ~900): expressions,
   conditional expression, fate classification, snapshot discipline.
3. `std/kinds.py` for floats/ints/tuples/Handles (~120).
4. `backends/python/` (~300): eval rules for the ~15 day-1 ops; renders the
   frame to a numpy array (headless-testable).
5. `backends/wgsl/` (~550): planner generalized from M0 `layout.py`, renderer
   RuleSet, wgpu runtime with glfw window.

Exit criteria (each is a named test in CI from this day forward):

- `disk.py --frames 120` on WGSL: `compiles=1`, `frames=120`; the same script
  with `backend=python` produces pixel-identical (within float tolerance)
  frames — the backend-seam proof.
- Hit-path microbenchmark: rebuilt-closure call < 10 µs (alarm at 5 µs) —
  retires the pure-Python-hot-path risk or triggers the exec-binder escalation.
- `no_compile` assertion mode active during the frame loop (a second compile
  raises).
- Golden printed IR at four stages (surface core / mid / abi / rendered WGSL
  and Python source) with stage-legality checks passing.
- Miss-counter test: change a capture's *type* (int→float) ⇒ tier-1 miss named
  "env_types[0]"; change the target texture format ⇒ tier-1 miss named
  "codegen_opts"; edit the function body ⇒ miss named "template".

---

## 7. Risks and early tests

1. **The pure-Python hit path misses the µs budget** (numba judged Python too
   slow and wrote 2,850 lines of C). *Early test:* the day-1 microbenchmark
   above, in CI from the vertical slice onward. *Mitigation, pre-shaped and
   contract-preserving:* exec-generated per-template fused extractor+packer
   (Triton), then a narrow native fastpath (JAX) — the `FastRecord` contract
   admits both without interface change.
2. **Fixpoint rewriting over ordered region bodies proves clumsy** — body
   rebuild churn, ordering bugs when moving pure ops across effectful ones.
   *Early test:* the vertical slice already runs three stages of rewrites over
   region bodies for two backends; count body rebuilds per compile in the match
   log and golden-test stage outputs. *Trigger:* persistent churn/ordering bugs
   ⇒ reopen tinygrad's pure-graph encoding (V2's named fallback).
3. **Kernel-budget creep / the lowerer bloats into the kernel** — the fused
   typing+lowering pass has the widest surface (~900 lines) and the strongest
   gravitational pull toward the kernel. *Early test:* CI line budget with
   per-PR delta on `kernel/`; architectural check that `kernel/` never imports
   `front/`. *Trigger:* kernel drifting past ~2–3 kloc or the OpDef verifier
   reinventing IRDL's constraint solver ⇒ reopen xDSL (the MLIR-flavored
   printer keeps that a refactor).
4. **WGSL's restrictions shrink the portable battery layer** (no recursion,
   limited loop forms) below numba's ~2:1 portable:bound ratio, collapsing the
   batteries economics. *Early test (milestone 1):* port 10 representative
   batteries (mean, clip, smoothstep, color conversions) to the WGSL backend;
   track the CI ratio metric. *Mitigation:* grow surface-(a) intrinsics and
   per-target overload entries — the surfaces already allow it.
5. **Silent key incompleteness or value leakage** — a value smuggled into
   `attrs` "to make it work", or an ambient dimension (backend flag, unit,
   generation) missing from its tier. *Early test:* an attr lint (only
   `LiteralType`-originated constants may appear as `core.const` attrs, checked
   on printed IR), plus a differential key test that perturbs every ambient
   dimension one at a time and asserts the expected tier of miss by name. The
   two-tier law ("changes-what-code-runs ⇒ artifact key;
   changes-what-bytes-go-where ⇒ plan key") is enforced as a test, not a comment.

---

## Design lessons for pdum.dsl

1. **The MLIR ideas, not the MLIR dependency.** Dialect-namespaced ops on one
   frozen `Node`, regions with typed binders and yields, traits, declarative
   patterns, progressive lowering, MLIR-flavored text — all fit in ~680 owned
   IR lines. Never let a 0.x dependency own the `__eq__`/`__hash__` the cache
   key rests on; keep xDSL as a pinned dev oracle and a pre-paid migration path.
2. **Make lowering stages declare their legality.** `Stage(name, rules, legal)`
   costs ~30 lines and turns "progressive lowering" from a convention into a
   machine-checked invariant with `loc`-bearing errors. Debug-mode grammar
   RuleSets sit on top for full per-stage specs.
3. **One rewrite driver, everything is rules**: simplification, decomposition
   (gated on `code_for_op.keys()`), backend legalization, param legalization
   (marshaling!), AD/vmap columns, and rendering all run through the same
   ~170-line engine. New feature = new rules + maybe a new dialect; the `Node`
   class and the driver never change.
4. **The caching thesis has three syntactic enforcement points**: `EnvVar`/
   `core.env` cannot hold a value (frontend), `attrs`-vs-`args` decides key
   membership (IR), and `LiteralType` is the single door for values (marshal) —
   opening the key, the constant, and slot elision together so they cannot drift.
5. **The kernel is frontend- and backend-agnostic by construction**: it
   consumes `core.*` Nodes from any producer (AST lowerer, t-string parsers,
   `raw_kernel`) and emits through capability records. The AST lowerer — the
   largest single component — is a satellite, which is what keeps the kernel
   at ~1000 lines honestly rather than by accounting tricks.
6. **The hit path is the product; defend it structurally.** `FastRecord` means
   the per-frame cost is key build + dict probe + extract + pack + launch.
   Everything expensive is computed at miss time and cached; `flatten` is the
   only extension code on the hot path and carries an allocation-budget test.
7. **Reserve the transformation seams on day 1 even though AD ships last**:
   empty aspect columns, `linear_in` traits, `DerivedId` in the template
   identity, pure eqn regions, exactly three region ops, captures as typed
   params. Cheap now, near-impossible to retrofit (JAX's consts and FX's
   missing control flow are the standing warnings).
8. **Ship the observability with the cache**: per-tier miss counters that name
   the differing key component, `no_compile` mode for render loops, printed-IR
   golden files per stage, rewrite-match logs, and a CI line budget. These are
   the cultural mechanisms that kept every exemplar kernel small and every
   cache honest.
