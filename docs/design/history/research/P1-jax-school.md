# P1 — Architecture Proposal: the JAX School

**A ~1000-line kernel of primitives with rule slots, transformations as
interpreters over a micro-IR, and a pytree-style marshaling registry —
Autodidax restraint applied to the type-keyed-caching thesis.**

Author role: architect (JAX school). Inputs: `docs/desiderata.md`,
`design/dsl_caching_layer.md`, verdicts V1–V5
(`design/research/V1-frontend.md` … `V5-transforms.md`), the frozen M0
reference at `src/pdum/dsl_reference/`.

---

## 0. Stance and deviations

The JAX school says: define a **small closed set of primitives**; attach all
per-primitive knowledge (typing, evaluation, lowering, differentiation,
batching) as **rules in per-aspect slots**; implement every capability —
including compilation itself — as an **interpreter over the same program
representation**; marshal values through a **registry of flatten/unflatten
pairs** so the kernel never knows a concrete Python type. Autodidax proves the
whole discipline fits in a few hundred lines when you refuse cleverness.

This proposal follows the V1–V5 verdicts essentially as written — they are
already JAX-school verdicts. Three reconciliations/deviations, flagged now:

1. **Single-result nodes + projections (V2) over multi-result eqns (V5).**
   V2 specifies `Node` with one result plus projection ops; V5 asks for
   "multiple results per eqn." I side with V2: one node shape keeps every
   rule signature and the rewrite driver uniform (fewer node shapes is the
   JAX-school prime rule). Multi-result ops (loops with carries, `if` with
   multiple yields) return a **tuple-typed node**; `core.proj` extracts
   elements; transform rules receive/return tuples of builders' values, so V5's
   rule signatures are unchanged in practice. Cost: one extra trivial op.
2. **`FastRecord.guards` on the hit path.** The hard constraint says the hot
   hit is "key build + value pack + launch, nothing else." V1 independently
   requires call-time dependency-drift checking (Triton-style). I fold the
   drift check into key build as a precomputed tuple of `is`-comparisons
   (single-digit pointer compares) stored on the `FastRecord`. It is
   constant-time, allocation-free, and I count it as part of key build.
3. **The printer's *test harness* lives outside the kernel budget.** The
   MLIR-flavored printer itself (~60 lines) is in the kernel (V2: day-one
   infrastructure); the golden-file harness, per-stage grammar `RuleSet`s, and
   rewrite-match logging (~120 lines) are `devtools/`, imported only by tests
   and debug flags. This is bookkeeping, not a semantic seam change.

Everything else — two-layer frontend, micro-IR over xDSL, five hook surfaces,
three-piece marshaling, transforms as rule-matrix passes — is adopted verbatim.

---

## 1. Core primitives

The kernel's whole vocabulary. Everything else in the system is expressed as
registrations against these. LOC estimates are for the primitive plus its
immediate methods, and are counted inside the module budgets of §3.

| # | Primitive | One-line responsibility | LOC |
|---|-----------|------------------------|-----|
| P1 | `Type` | structural, frozen, serializable description of a DSL value; sole input to artifact keys | ~70 |
| P2 | `TemplateId` | value-compared identity of a program template: base code object or derived transform | ~25 |
| P3 | `SourceSnapshot` | decoration-time source capture with coherence check against the code object | ~20 |
| P4 | `FnType` / `Env` / `Handle` | the closure split: (identity, typed environment) as a pure value + runtime payload | ~60 |
| P5 | `Node` / `Region` | the entire IR: one immutable node shape with content hash; three region ops | ~110 |
| P6 | `OpDef` + traits | declaration of a primitive op: name, arity, traits; rules live in the Registry matrix | ~30 |
| P7 | `Registry` | the one explicit extension object: op×aspect rule matrix + five registration surfaces | ~75 |
| P8 | `RuleSet` / rewrite driver | declarative (pattern, fn) rewriting over regions; runs all compiler logic | ~135 |
| P9 | `ValueKind` + leaf vocabulary | per-Python-type typeof/flatten/fingerprint; closed core leaf set | ~60 |
| P10 | `PackPlan` / `Slot` / `PhysicalDest` | backend-owned types-only plan mapping logical leaves to physical destinations | ~40 |
| P11 | `FastRecord` | per-cache-entry precomputed hit path: artifact, extractor, plan, staging, launcher, guards | ~30 |
| P12 | `Backend` | capability record: token, type_map, code_for_op, legalize rules, render, runtime | ~30 |
| P13 | Two-tier cache | thesis cache (types→FastRecord) over kernel compile cache (IR hash→artifact) | ~85 |

### P1. `Type`

```python
@dataclass(frozen=True)
class Type:              # abstract; all subclasses frozen, structural eq/hash
    ...
class Scalar(Type):      kind: str; bits: int          # i64, u64, f32, f64, bool, bigint→error
class Vec(Type):         elem: Scalar; n: int
class Array(Type):       dtype: Scalar; ndim: int; layout: str; byteorder: str; writeable: bool
class TupleT(Type):      elems: tuple[Type, ...]       # arity is part of the type
class RecordT(Type):     name: str; fields: tuple[tuple[str, Type], ...]
class FnT(Type):         template: TemplateId; env_types: tuple[Type, ...]
class LiteralT(Type):    base: Type; value: Hashable   # the ONE value-in-type exception
class NoneT(Type):       pass
# later, no redesign (V4): class QuantityT(Type): rep: Scalar; dim: Dim  # Dim = rational exponents
```

Honest types (desiderata §9): `typeof(int)` range-buckets to i64/u64/error;
narrowing to f32 is a backend `type_map` decision. Every `Type` serializes
structurally from day one (`.key` property → nested tuples of strs/ints) —
the disk-cache requirement V4 imposes.

### P2. `TemplateId` — a sum type from day 1 (V5)

```python
class TemplateId: ...
class BaseId(TemplateId):     code: CodeType             # value-compared (co_code/co_consts/...)
class DerivedId(TemplateId):  tag: str; base: TemplateId; static: tuple[Hashable, ...]
# grad(f) → DerivedId("grad", BaseId(f.__code__), (argnums,)) — value-compared,
# so grad(f) rebuilt every loop iteration is a cache HIT and never collides with f.
```

### P3. `SourceSnapshot`

`(text, co_filename, co_firstlineno, co_qualname)` captured at decoration time
(code object and `linecache` guaranteed coherent — V1). On a phase-B miss,
`compile(text)` must be value-equal to the template code object modulo
filename/firstlineno, or we raise: stale-source compiles are impossible by
construction.

### P4. `FnType` / `Env` / `Handle`

```python
FnType = FnT(template=TemplateId, env_types=tuple[Type,...])   # pure value, hashable
Env    = tuple[object, ...]     # capture VALUES in co_freevars order; never hashed, never keyed

@dataclass
class Handle:                   # what @jit returns; phase-A product; compile-free
    fn_type: FnType
    env: Env
    env_fp: tuple               # structural fingerprint of env, computed once at capture
    snapshot: SourceSnapshot | None    # None ⇒ phase B raises NoSourceError with remedies
    registry: Registry
    def __call__(self, *args): return dispatch(self, args)     # phase B
```

### P5. `Node` / `Region` — the entire IR (V2 verbatim)

```python
@dataclass(frozen=True)
class Node:
    op: str                                  # dialect-namespaced: "core.add", "math.sqrt"
    type: Type                               # intrinsic field; single result (tuple-typed if multi)
    args: tuple[Node, ...]                   # runtime operands (SSA-as-nodes)
    attrs: tuple[tuple[str, Hashable], ...]  # compile-time constants — IN the content hash
    regions: tuple[Region, ...] = ()
    @cached_property
    def key(self) -> bytes: ...              # memoized structural blake2b content hash

@dataclass(frozen=True)
class Region:
    params: tuple[Node, ...]                 # typed binders (core.param nodes)
    body: tuple[Node, ...]                   # ordered; last is core.yield
```

**Exactly three region-carrying ops, frozen**: `core.if` (two pure
sub-regions, identical yield types — purity so vmap can lower to
both-branches+select), `core.for` (counted loop, explicit fixed-type carries —
the only form reverse mode handles), `core.call` (sub-program). Every
additional higher-order op costs ~1 rule per existing transformation aspect
(~180 lines each in autodidax) — the cap is a budget decision, enforced.

**The anti-numba invariant, structural**: captures lower to
`core.env(slot=k)` — an op with a slot attr and a type, **no value field
exists in the IR**. A `Literal`-lifted capture lowers instead to
`core.const(value=v)` whose value is an attr and therefore in `Node.key`
(recompiles per value, visibly, auditable in printed IR). These are the only
two fates of a capture, decided once at lowering (V2).

### P6. `OpDef` + traits

```python
@dataclass(frozen=True)
class OpDef:
    name: str; nargs: int; traits: frozenset  # Pure, Commutative, LinearIn(i), ...
    nregions: int = 0
```

Rules do NOT live on the OpDef — they live in the Registry's op×aspect matrix,
so adding an aspect (jvp) later touches zero op declarations (V3/V5).

### P7. `Registry` — the one explicit extension object (V3)

```python
class Registry:
    ops:       dict[str, OpDef]
    rules:     dict[tuple[str, str], Callable]   # (op_name, aspect) → rule; aspects:
               # "type", "eval" day 1; "jvp", "transpose", "batch", "unit" reserved (~10 lines to declare)
    overloads: dict[object, list[tuple[TargetToken, Callable]]]   # np.clip → [(Generic, impl_builder)]
    methods:   dict[tuple[Type, str], list[tuple[TargetToken, Callable]]]
    typeof:    singledispatch registry            # Python type → ValueKind
    backends:  dict[TargetToken, Backend]
    bindings:  dict[int, str]                     # id(global obj) → op name (name classification)
    generation: int                               # world age; bumped on any redefinition
    def derive(self) -> Registry: ...             # copy-on-write child (scoped extension)
```

**No module-level registries anywhere.** `jit(fn, registry=DEFAULT)`; the
stdlib populates `DEFAULT` through the same five public surfaces users get.
Missing rule ⇒ `MissingRule(op, aspect, source_loc)` — never a fallback.

### P8. `RuleSet` / rewrite driver

`(pattern, fn)` pairs; pattern = op name + optional predicate; one generic
~135-line driver does post-order fixpoint rewriting over region bodies with
per-pass match logging under a debug flag. All compiler logic — decomposition,
legalize_params, canonicalization — is RuleSets run by this one driver.

### P9. `ValueKind` + leaf vocabulary (V4)

```python
@dataclass(frozen=True)
class ValueKind:
    typeof: Callable[[object], Type]             # sole artifact-key input
    flatten: Callable[[object], tuple]           # ordered logical leaves (values)
    fingerprint: Callable[[object], Hashable]    # hot-path structural tag
    leaf_types: Callable[[Type], tuple[Leaf, ...]]  # derivable from Type ALONE (plans without values)
# closed, core-owned leaf set — backends must be total over it:
ScalarLeaf(dtype) | BufferLeaf(dtype, ndim) | ShapeLeaf(i) | StrideLeaf(i) | EnvLeaf(fn_type)
```

One registration per Python type yields all four. Static/dynamic split: the
static part feeds `typeof`, the dynamic part feeds `flatten`.

### P10. `PackPlan` / `Slot` / `PhysicalDest` (V4)

```python
Slot = (path: LeafPath,            # (root env|arg, index, sub-path, leaf index) — recorded by the inliner
        convert: Affine | None,    # units tier-2: scale/offset applied during pack
        dest: PhysicalDest)        # backend vocabulary, OPAQUE to the kernel:
# UniformSlot(offset, fmt) | KernelArg(index, ctype) | CField(offset, ctype) | PyArg(index)
PackPlan = tuple[Slot, ...]        # built ONCE per cache entry from types alone
```

One generic packer interprets any PackPlan (`struct.pack_into` per Slot).
Backend leaf vocabulary (WGSL `@group/@binding`, ctypes, CUDA arg structs)
appears only inside `Backend.runtime.plan_layout` — the rule whose violation
killed numba's datamodel extensions twice (V3).

### P11. `FastRecord`

```python
@dataclass
class FastRecord:
    artifact: object          # pipeline / RawKernel / cdll fn / python callable
    extract:  Callable        # closure-compiled once per entry: (env, args, buf) → fills buf via LeafPaths
    plan:     PackPlan
    staging:  bytearray       # reused per call
    launch:   Callable        # backend runtime launcher
    guards:   tuple           # precomputed (cell, expected) identity pairs; drift ⇒ treat as miss
```

Installed atomically under a per-key future (thread-safety per
dsl_caching_layer.md).

### P12. `Backend` (V3 §d)

```python
@dataclass(frozen=True)
class Backend:
    token: TargetToken               # Generic → CPU/GPU → WGSL/CUDA/Metal/C/Py lattice, MRO-resolved
    type_map: dict[Type, str]        # honest type → backend width (explicit narrowing)
    code_for_op: dict[str, str | Callable]     # intrinsic emitters; keys() = capability declaration
    legalize: RuleSet                # splits core.env into N physical envs (ptr+shape), slot numbering
    render: Callable[[Node], str]    # codegen seam: types+IR → source text
    runtime: Runtime                 # runtime seam: plan_layout, compile(source)→artifact, launch
    params_key: Callable[..., Hashable]  # backend params that must enter the thesis key (M0's cured fault)
```

### P13. The two-tier cache (V2/V4)

```python
THESIS:  dict[(TemplateId, env_fp, arg_fp, token, backend_params, generation)] -> FastRecord
KERNEL:  dict[(Node.key, renderer_name, renderer_flags)] -> artifact
```

Tier law (V4): anything changing generated code keys THESIS+KERNEL; anything
changing only bytes-written (concrete units, width casts) keys only a cheap
pack-plan memo. Per-tier miss counters name the differing key component;
`no_compile` assertion mode for render loops; per-template compile-count lint.

---

## 2. Core hooks

Exactly **five** registration surfaces (V3). All ship day 1; AD columns are
declared-empty until transforms land. The five signatures are the *only*
stable contract; IR internals, renderer internals, registry storage are
explicitly unstable.

### Surface (a) — ops and per-backend intrinsics

```python
registry.defop("math.sqrt", nargs=1, traits={Pure})

@registry.rule("math.sqrt", "type")
def _(ctx, x):                        # ctx carries loc for diagnostics
    return require_float(ctx, x)

@registry.rule("math.sqrt", "eval")   # Python backend AND finite-difference oracle
def _(x): return math.sqrt(x)

wgsl.code_for_op["math.sqrt"] = "sqrt({0})"        # per-backend table entry
c.code_for_op["math.sqrt"]    = "sqrtf({0})"

registry.bind_global(math.sqrt, "math.sqrt")       # name classification: captured
                                                    # `math.sqrt` global → intrinsic

# shared decompositions, gated on capability declarations:
@registry.decomposition("math.sinh", requires={"math.exp"})
def _(b, x): return b.mul(b.f(0.5), b.sub(b.op("math.exp", x), b.op("math.exp", b.neg(x))))
```

**End-to-end**: a backend that declares `math.sqrt` in `code_for_op` emits it
directly; one that declares only `math.exp` gets `sinh` via the decomposition
RuleSet; one that declares neither raises `MissingRule("math.sinh", "lower@c",
loc)` pointing at the user's source line. Adding `sqrt` = 4 registrations +
one table line per backend; zero kernel edits.

### Surface (b) — batteries (`@overload` / `@overload_method`)

```python
@overload(np.clip, target=Generic)         # written in the DSL subset itself
def clip_generic(x, lo, hi):
    if isinstance(x, Scalar):
        def impl(x, lo, hi): return min(max(x, lo), hi)
        return impl                        # None ⇒ decline; next candidate by target-token MRO
```

Overload impls compile through the **same** `(FnType, arg_types)` thesis
cache as user code — no third cache (V3). Resolution order, implemented once:
`__pdum_dsl__`-convertible → `registry.overloads[callee]` filtered by
target-token MRO (most-derived first) → methods/attrs with type-MRO fallback →
error naming callee, arg types, and declined entries.

### Surface (c) — type extensions (three registrations, or one sugar)

```python
@register_record(registry)                 # sugar generating all three registrations + ctor overload
@dataclass(frozen=True)
class Color:
    r: float; g: float; b: float
# generated: typeof → RecordT("Color", (("r",f64),...)); ValueKind.flatten → (v.r, v.g, v.b);
#            leaf_types → (ScalarLeaf(f64),)*3; overload of Color(...) as core.record_pack

@overload_method(RecordT("Color", ...), "to_hsv", target=Generic)
def _(c):
    def impl(c):
        mx = max(c.r, max(c.g, c.b)); mn = min(c.r, min(c.g, c.b))
        ...
        return Color(h, s, v)
    return impl
```

**End-to-end**: `col = Color(0.9, 0.4, 0.1)` captured by a shader; phase A
types it `RecordT("Color",...)` (in the key), flattens to 3 scalar leaves;
WGSL's planner packs them into the uniform buffer; `col.to_hsv()` in the body
resolves through the methods table, inlines the compiled impl. New frame with
a new `Color` value: same type, zero compiles, 12 bytes repacked.

### Surface (d) — backends

```python
register_backend(registry, Backend(
    token=CToken(parent=CPUToken),
    type_map={f64: "double", f32: "float", i64: "int64_t", ...},
    code_for_op={"core.add": infix("+"), "math.sqrt": "sqrt({0})", ...},
    legalize=c_legalize,        # Array env → BufferLeaf ptr CField + ShapeLeaf CFields
    render=render_c,            # IR → one .c translation unit
    runtime=CRuntime(),         # plan_layout → packed argument struct via CField;
))                              # compile: cc -O2 -shared + ctypes; launch: fn(byref(struct))
```

**End-to-end (adding the C backend)**: one new directory `backends/c/`
(~150-line renderer + ~180-line runtime + ~60-line legalize/planner — inside
the tinygrad-measured 50-300 + 130-220 envelope), one `register_backend`
call. If C needs more, the missing abstraction belongs in shared
decomposition rules, not in the backend (V3). The kernel is untouched —
enforced by an import linter (kernel may not import `backends/`).

### Surface (e) — transformations (aspect columns + escape hatch)

```python
# Day 1 (~10 lines): declare empty columns so ops can register aspects lazily
registry.declare_aspects("jvp", "transpose", "batch", "unit")

# Per-op rules, builder-passing signatures (V5):
@registry.rule("math.sqrt", "jvp")
def _(b, (x,), (tx,), params):
    y = b.op("math.sqrt", x)
    return (y,), (b.mul(tx, b.div(b.f(0.5), y)),)

@registry.rule("core.mul", "batch")
def _(b, (x, y), (bx, by), params): ...

# Transform = pipeline of ~100-line generic pass drivers that recurse into regions:
grad = Transform("grad", passes=(jvp_pass, partial_eval_pass, transpose_pass))
vmap = Transform("vmap", passes=(batch_pass,))

g = grad(f)     # phase A only: Handle with DerivedId("grad", f.template_id, (0,)) — NO compile

# The single escape hatch (also the only sanctioned home for backend-native AD):
@custom_vjp
def fancy(x): ...
@fancy.defvjp
def _(res, ct): ...      # an MLX-backed intrinsic may register rules whose LOWERING calls
                          # an MLX kernel; semantic ownership of grad stays in this table
```

**End-to-end (adding grad)**: ship order type→vmap(~100-line pass + 1
rule/op)→jvp(~100)→reverse(jvp+partial-eval+transpose, ~400-500, own
milestone; tinygrad's 132-line direct-VJP table is the fallback). Pipeline
runs once at phase B per derived key; transformed IR cached with the
artifact; **the hot path never re-runs a transform**. `grad(f)` rebuilt every
iteration hits because `DerivedId` is value-compared.

### t-strings: no sixth hook

`t"b (c h) w -> b c h w"` (PEP 750, 3.14): the `Template`'s static strings
are source (immune to `NoSourceError`), get a `ValueKind` whose `typeof`
hashes the static parts (into the key) and whose `flatten` yields the
interpolations as **typed env leaves** (marshaled per call, never recompiled).
The einops mini-compiler is an overload on the Template type that emits
core-dialect nodes. Frontend sugar + surfaces (b)/(c); zero kernel changes.

---

## 3. Module map

### Kernel (budget ≈1000; CI line-budget bot caps at 1100, PR delta report)

| Module | LOC | Responsibility |
|---|---|---|
| `kernel/types.py` | 70 | P1: structural Type dataclasses, `.key` serialization, interning |
| `kernel/capture.py` | 100 | P2–P4: TemplateId, SourceSnapshot, safe_cell, `make_handle`, `@jit` — phase A, compile-free, never fails on missing source |
| `kernel/ir.py` | 130 | P5: Node/Region, content hash, Builder |
| `kernel/ops.py` | 85 | P6: OpDef/traits, core-dialect table (`env const arg tuple proj yield if for call` + arith/cmp), verifier |
| `kernel/registry.py` | 75 | P7: Registry object, five surfaces' storage, resolution order, `derive()`, MissingRule |
| `kernel/lower.py` | 165 | V1 phase B: parse snapshot, coherence check, name classification (closed fate taxonomy), **one fused typing+lowering forward pass** (dialect tables as parameters), loc side channel |
| `kernel/rewrite.py` | 135 | P8: RuleSet, pattern match, fixpoint driver, named passes |
| `kernel/marshal.py` | 105 | P9: ValueKind registry, structural fingerprints, LeafPath extractor compiler (closure-compiled once per entry), generic `pack_into` packer |
| `kernel/cache.py` | 85 | P11/P13: two-tier cache, FastRecord, per-key futures, generation, guards, per-tier miss counters, `no_compile` mode |
| `kernel/printer.py` | 60 | MLIR-flavored textual form (round-trip target for golden tests) |
| **total** | **1010** | |

The kernel imports **nothing** from `backends/`, `transforms/`, `batteries/`,
`ext/` — enforced by import-linter in CI. The stdlib populates `DEFAULT`
registry via the same five public surfaces.

### Outside the kernel (attachment point in parentheses)

| Component | LOC | Attaches via |
|---|---|---|
| `devtools/` (golden-file harness, grammar RuleSets, xDSL oracle translator — pinned dev extra, never imported by kernel) | 120 + 200 | printer / P8 |
| `backends/python/` — eval_rules interpreter + PyArg runtime; reference semantics + FD oracle | 220 | surfaces (a),(d) |
| `backends/wgsl/` — renderer 230, uniform layout planner 200 (M0's `layout.py` generalized), wgpu runtime 220 | 650 | (a),(d) |
| `backends/cuda/` — CUDA renderer + `cupy.RawKernel` runtime, KernelArg dests (numba `Record.make_c_struct` model) | 380 | (a),(d) |
| `backends/mlx/` — Metal via `mx.fast.metal_kernel`; also hosts custom_vjp-backed native intrinsics | 380 | (a),(d),(e) |
| `backends/c/` — .c renderer + cc/ctypes runtime, CField dests | 390 | (a),(d) |
| `batteries/` — mean, clip, smoothstep, color conversions, … (~30-50 each, DSL subset) | grows | (b) |
| `transforms/` — vmap 100+rules, jvp 100+rules, reverse 450 | 650+ | (e) |
| `ext/units/` — QuantityT/Dim types, unit rules ("unit" aspect column), Affine slot converters | 260 | (c),(e), P10 |
| `ext/einops/` — t-string mini-language | 180 | (b),(c) |
| `raw_kernel` escape hatch per backend (MLX-shaped: name, inputs, outputs, source_body) — also hosts the compiler's own output | in backend budgets | (d) |

---

## 4. The dataflow

### Phase A — capture (at decoration; every loop iteration; compile-free)

```
@jit / disk(cx, cy, r) executes the def:
 1. code = fn.__code__                       # C-speed attribute reads
 2. vals = tuple(safe_cell(c) for c in fn.__closure__ or ())
 3. env_fp = fingerprint_tuple(vals)         # structural: int range bucket, tuple arity,
                                             # array (dtype,ndim,layout,flags); scalars ~free
 4. fn_type = FnT(BaseId(code), types-from-fp-memo)   # full typeof only on fp-memo miss
 5. snapshot = source text captured ONCE per code object (memoized side table)
 → Handle(fn_type, vals, env_fp, snapshot, registry)  # NO parse, NO IR, NO compile
```

Phase A **always succeeds** — missing source, weird captures, everything is
deferred to phase B's loud errors (V1).

### Phase B — call: the hit path (what a hot loop actually executes)

```python
def dispatch(handle, args):
    key = (handle.fn_type.template, handle.env_fp, fingerprint_tuple(args),
           BE.token, BE.params, handle.registry.generation)      # ← key build
    rec = THESIS.get(key)
    if rec is not None and _guards_ok(rec.guards):               # N pointer compares
        rec.extract(handle.env, args, rec.staging)               # ← value pack:
        #   closure-compiled leaf reads (LeafPath) + struct.pack_into + unit Affine converts
        return rec.launch(rec.staging)                           # ← launch:
        #   WGSL: queue.write_buffer(uniform, staging); draw   |  CUDA: kernel(grid, args)
    return _miss(handle, args, key)
```

That is the entire per-call cost on a hit: **one tuple build, one dict probe,
guard compares, one extractor run, one pack, one launch.** No AST, no IR, no
flatten-walk, no registry lookups (all hooks ran at compile time; V3's
"registry-free hot path"). Budget: single-digit µs pure Python, with two
pre-shaped contract-preserving escalations (exec-generated per-template
binder à la Triton, then a narrow native fastpath à la JAX) (V4).

### Phase B — the miss path (once per type signature)

```
 1. full typeof of env+args (fingerprint was the fast proxy; verify + intern)
 2. parse SourceSnapshot with real filename/lineno; coherence-check compile() vs code object
 3. classify names → closed fates: param | EnvVar slot | intrinsic (bindings) |
    Handle callee (FnT-typed) | allowed folded const | NoSourceError/Literal-lift/error
 4. fused typing+lowering forward pass → core-dialect Node graph
    (captures → core.env(slot=k) — no value field EXISTS; Literal lifts → core.const)
 5. if DerivedId: run transform pipeline (jvp/partial-eval/transpose/batch) — rule-matrix
    interpreters over the same nodes; cache transformed IR with the entry
 6. rewrite passes: decompositions gated on BE.code_for_op.keys(); BE.legalize splits
    logical core.env → N physical envs (ptr+shape / uniform slots); slot numbering
 7. KERNEL cache probe on (Node.key, renderer, flags) → render + backend compile on miss
 8. BE.runtime.plan_layout → PackPlan; inliner's LeafPaths → closure-compile extractor
 9. assemble FastRecord(artifact, extract, plan, staging, launch, guards);
    install atomically under per-key future; bump miss counter naming the differing component
```

### Where the caches sit

- **Thesis cache** (`kernel/cache.py`, in-process dict): above everything;
  keyed on types/identity only; the IR is touched only between miss and
  artifact, never on a hit.
- **Kernel compile cache**: below transforms and rewrites; content-addressed
  on `Node.key` — two different templates that lower to identical IR share an
  artifact; backend params in the key (M0's fault cured structurally).
- **Pack-plan memo** (tier 2): `(entry, concrete-units/byte-options)` →
  PackPlan; interactive unit tweaks miss only here.
- **Disk cache** (later): structural keys only — `TemplateId` lowers to
  `(filename, qualname, source_hash)`, Types via `.key`; PackPlans rebuilt,
  never persisted (V4).

### Key anatomy (Dynamo guard checklist as completeness spec — V1)

`(template_id, env_types, arg_types, backend_token, backend_params,
generation)` in the key; dependency-closure drift (globals a template read at
lower time) as `FastRecord.guards` checked per call — refuse/recompile on
drift, never silently stale; aliasing between captured buffers decided
explicitly per backend (irrelevant | normalized at marshal | relational key
component), triggered the first time a backend's correctness depends on it (V4).

---

## 5. Desiderata mapping

| Desideratum | One-line answer |
|---|---|
| WebGPU | first real backend: WGSL renderer + uniform layout planner (M0's `layout.py` generalized) + wgpu runtime; UniformSlot dests |
| CUDA/CuPy | Backend record rendering CUDA source into `cupy.RawKernel`; KernelArg dests; ~380 LOC, zero kernel edits |
| Metal/MLX | Backend record over `mx.fast.metal_kernel`; doubles as custom_vjp host for native-gradient intrinsics |
| Python | day-1 backend = the `eval` aspect column interpreted directly; reference semantics + finite-difference oracle + zero-dependency floor |
| C | Backend record: render .c, `cc -shared` + ctypes runtime, CField struct dests |
| Records + methods | `@register_record` (typeof + ValueKind + ctor overload) + `@overload_method`; RecordT structural, so records key the cache honestly |
| Units | QuantityT(rep, Dim) with rational-exponent Dim in the artifact key (tier 1); concrete unit → Affine slot converter applied during pack (tier 2); mp-units value-preservation rules; ships later with zero redesign |
| Autodiff / vmap | aspect columns in the rule matrix + ~100-line pass drivers; DerivedId identities flow through the unchanged thesis cache; grad-of-capture expressible for free because captures are typed env slots, never consts |
| t-string mini-languages | Template statics are source and key the cache; interpolations are typed env leaves marshaled per call; einops = overload emitting core nodes |
| Batteries | `@overload` impls in the DSL subset, compiled per target through the same cache; shared decompositions gated on `code_for_op.keys()` |
| Value-dependent specialization | one explicit `Literal[v]` lift with three coupled effects: enters the key, becomes core.const attr, elides its Slot; no implicit hints, never id()-keyed |
| Live-coding invalidation | code-object value equality (unchanged re-run hits, edit misses) + generation in the key + per-call dependency guards; per-tier miss counters say *why* you recompiled |

---

## 6. Day-1 vertical slice

**Goal: the orbiting-disk demo on the new kernel, on TWO backends, proving the
backend seam and the thesis on day 1.**

Scope (nothing more):

1. Kernel modules of §3, restricted where possible: `lower.py` handles
   expressions, tuple returns, ternary (`core.if`), captures, intrinsic calls
   — no loops, no assignments yet.
2. Ops: `core.{env,const,arg,tuple,proj,yield,if,call}`, `add sub mul div neg
   lt`, `math.{cos,sin,sqrt}` (surface-a registrations with type+eval rules).
3. `backends/python/`: eval-rule interpreter + PyArg planner (~180 lines) —
   registered via `register_backend`, proving surface (d).
4. `backends/wgsl/`: fragment-shader subset renderer, uniform planner
   (scalars + vec captures — already past M0's scalars-only limit), wgpu
   runtime with `write_buffer` launch.
5. `demos/disk.py` ported: closure rebuilt every frame with fresh `cx, cy`.

**Acceptance criteria (each is a test):**

- `compiles == 1` after 240 frames on WGSL; `no_compile` assertion mode armed
  inside the frame loop (any phase-B miss after warmup raises).
- The same Handle runs on the Python backend with zero source changes
  (`jit(..., backend="python")`); rendered 64×64 frame matches WGSL output
  within f32 tolerance — the **differential oracle** exists from day 1.
- Hot-hit microbench: 100k rebuild+dispatch iterations, p50 per-iteration
  ≤ 10 µs (phase A + hit path), profile shows only key build + extract +
  pack + launch frames.
- `grep`-proof seam: kernel imports no backend module (import-linter green);
  the Python backend landed as a directory + one `register_backend` call.
- Printed IR golden files for the disk shader at: post-lower, post-legalize
  (WGSL) — locking the textual form early.
- Kernel LOC report ≤ 1100.

Explicitly deferred from the slice: `core.for`, arrays, records, overload
batteries beyond the three math ops, transforms (columns declared, empty),
CUDA/MLX/C, units, t-strings, disk cache.

---

## 7. Risks and early tests

**R1 — The pure-Python hot path misses the budget.** Phase A + hit path in
interpreted Python may exceed single-digit µs once records/arrays appear
(array fingerprints read `.flags` — not free).
*Early test:* day-1 microbench in CI with a regression gate; a second bench
with an array capture the week arrays land. *Mitigation, pre-shaped:*
exec-generated per-template binder (Triton), then narrow native fastpath
(JAX); both preserve the FastRecord contract.

**R2 — The region-op × aspect tax bites when control flow meets transforms.**
Each of if/for/call costs a rule per aspect (~180 lines/op/transform in
autodidax); if `core.for` carries prove awkward (early-exit, `break`), the
matrix strains, and pressure mounts to add a fourth region op — the design's
death by a thousand ops.
*Early test:* a 2-day spike **before** M2: vmap over `core.if` + `core.for`
on the Python backend only, including batched-predicate → both-branch+select.
If the spike exceeds ~350 lines total, re-hear tinygrad's direct-VJP/flat
approach per V5's fallback.

**R3 — WGSL restrictions shrink the portable battery layer.** If most
batteries need per-backend `code_for_op` entries instead of Generic
overloads, surface (b) collapses to surface (a) and battery economics die.
*Early test (retire in M1):* port ~10 representative batteries (mean, clip,
smoothstep, color conversions) to WGSL; require ≥ 2:1 portable-to-bound
ratio (numba's measured ratio, V3).

**R4 — Silent staleness / wrong-reuse from key or fingerprint gaps.** The
thesis cache is the product; a fingerprint collision (tuple arity, int range,
array layout) or a missing key component (backend param, globals drift,
aliasing) produces wrong pixels, not crashes.
*Early tests:* property-based fuzz that `fingerprint(a) == fingerprint(b) ⟹
typeof(a) == typeof(b)`; continuous python-vs-wgsl differential runs on
randomized expression programs; the Dynamo guard checklist as a review gate
for every new backend/feature; per-tier miss counters watched in demos
(a counter that *doesn't* move when it should is the alarm).

**R5 — Kernel creep and seam erosion.** M0's core imported WGSL tables once;
it will try again — via a "temporary" import, a battery in the kernel, a
1300-line lowerer.
*Early tests:* CI import-linter (kernel ↛ backends/transforms/batteries/ext)
from day 1; line-budget bot with PR deltas; the real acceptance test is
**M2 = CUDA backend lands touching only `backends/cuda/` + registrations**,
inside the 50-300 renderer + 130-220 runtime envelope — if it can't, the
missing abstraction is named and moved into shared rules, not patched around.

---

## Design lessons for pdum.dsl

1. **Make the thesis structural, not disciplinary.** `core.env` has no value
   field; `Node.attrs` is the only place values can enter the IR, and only
   via the explicit `Literal` lift. Value-independence of compilation is then
   a property of the data structures, not of reviewer vigilance. Any future
   node/field proposal that could hold a capture value should be rejected on
   this ground alone.
2. **One rule matrix, declared before it's needed.** Declaring empty
   `jvp/transpose/batch/unit` columns on day 1 costs ~10 lines and buys the
   guarantee that transforms never require touching op definitions or the
   kernel. Conversely, every new higher-order op taxes every column — treat
   the if/for/call cap as a budget line item with an owner, and price any
   fourth region op at ~180 lines × live transforms before accepting it.
3. **The hit path is a compiled artifact too.** Everything per-call must be
   *precomputed into the FastRecord at miss time* — extractor closures,
   pack plans, staging buffers, guards. The design rule: no registry, no
   isinstance-walk, no AST, and at most one extension callable
   (`ValueKind.flatten`, allocation-budget-tested) may execute on a hit.
4. **Two-tier keying is the law that makes units and live-tweaking cheap.**
   Code-shaping facts (types, dims, Literals, backend codegen params,
   generation) → artifact key; bytes-shaping facts (concrete units, widths)
   → pack-plan memo. Every new feature must declare which tier each of its
   parameters keys — make that a required line in feature design docs.
5. **Backends are data; capability is `code_for_op.keys()`.** Shared
   decompositions gated on declared op sets are what keep the Nth backend at
   ~300-500 lines. The moment a backend "needs" kernel changes, the change
   belongs in a shared RuleSet. Enforce with the import linter and the M2
   CUDA acceptance test.
6. **The Python backend is not a nicety — it is the oracle.** eval rules give
   reference semantics, the finite-difference harness for every AD rule, and
   the differential tester for every GPU backend, all for ~220 lines. It
   ships in the day-1 slice specifically so the WGSL backend is never the
   definition of correctness.
7. **Identity is a sum type; derivations are identities.** `DerivedId("grad",
   base, static)` makes `grad(f)` an ordinary template with an ordinary cache
   row — transforms compose with the thesis instead of bypassing it. Any
   future feature that creates programs (t-string DSLs, partial evaluation,
   raw_kernel) should mint a TemplateId the same way rather than inventing a
   parallel cache.
8. **Observability of the cache is a product feature.** Per-tier miss
   counters that *name the differing key component*, `no_compile` assertion
   mode, and compile-count lints are what make "the loop stays hot" testable
   and user-debuggable. Ship them in the vertical slice, not later.
9. **Spend the LOC budget where rules can't reach.** The fused
   typing+lowering pass (165 lines) and the rewrite driver (135) are the only
   real "engines" in the kernel; everything else is dataclasses and dicts.
   If a proposed kernel component is neither a primitive nor one of these two
   engines, it is probably a registration in disguise — push it out through
   one of the five surfaces.

---

*File: `/Users/nehal/src/pdum_dsl/design/research/P1-jax-school.md`. Companion
verdicts: V1–V5 in this directory. The frozen M0 reference remains the
behavioral spec for the day-1 slice's demo.*
