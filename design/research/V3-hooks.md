# V3 — Verdict: the extension/hook architecture

*Consolidation verdict for the pdum.dsl redesign, July 2026. Inputs: R1 (numba),
R2 (DaCe), R3 (xDSL), R4 (JAX), R7 (minimal compilers), with R8 (user types)
consulted for the type-extension surface. Question: how do users and the stdlib
register new functions, methods, types, backends, and transformations without
touching the kernel?*

---

## Recommendation

Build **one registry object and one rule matrix**, and expose exactly **five
registration surfaces** over them:

| Surface | Mechanism | Day 1? |
|---|---|---|
| (a) per-backend intrinsics | `code_for_op` table in the backend's capability record + shared decomposition rules parameterized by the backend's declared op set (tinygrad) | yes |
| (b) library batteries | `@overload` / `@overload_method` / `@overload_attribute` — pure-Python impls written **in the DSL subset**, selected by type at typing time, compiled by the active backend's own jit, filtered by target-token MRO (numba) | yes |
| (c) type extensions | three registrations per type: `typeof` (value→type), `marshal` (flatten/unflatten with a static/dynamic split, pytree-shaped), and methods via (b) | yes |
| (d) backend registration | a `Backend` dataclass (target token + renderer capability record + runtime) registered into the Registry; target tokens form a `Generic→CPU/GPU→WGSL/CUDA/...` class lattice resolved by MRO (numba's 169-line `target_extension.py`, minus the global registries) | yes |
| (e) transformations | aspect **columns** in the rule matrix `rules[(op, aspect)]` — `"type"`, `"lower"` populated day 1; `"vmap"`, `"jvp"`, `"transpose"`, `"unit"` are later columns requiring **zero kernel change**; plus `custom_jvp`/`custom_vjp`-shaped per-function escape hatches (JAX) | matrix day 1, AD columns later |

All registration goes through **explicit `Registry` objects** (layered via
`ChainMap`-style `extend()`), never module-level dicts with lazy loaders. The
stdlib is itself a client of these five surfaces: nothing in `pdum/std/` may
touch the kernel. Estimated cost of the entire hook kernel: **~750–900 lines**.

---

## Rationale

The evidence from the five reports converges on a single economic fact:
**extension code survives exactly as long as it avoids the backend's codegen
vocabulary, and it stays cheap exactly when one definition serves all
consumers.** Numba provides the controlled experiment (R1): its 491 `@overload`
batteries — pure Python written against the type system — moved to NVIDIA's GPU
target as a 98%-verbatim copy and survived both the numba-cuda fork and the MLIR
rewrite, while its 272 hand-lowered builtins and 166 `@intrinsic`s (LLVM-bound)
died twice. JAX (R4) shows the complementary result for transformations: when an
op is just a name and every meaning is a rule in a per-aspect registry, adding a
transformation costs one interpreter plus one rule column (~300 lines for the
whole matrix in autodidax), and adding an op never touches existing
transformations. tinygrad (R7) shows the same matrix works for *backends*: with
shared decomposition rules gated on each renderer's declared primitive set, a
full WGSL backend is 115 lines. DaCe (R2) confirms the method/operator/attribute
lookup tables with MRO fallback cost ~165 lines and are fully separable from any
IR style. The failures are equally consistent: numba's two parallel registration
namespaces (typing templates vs. `@lower_builtin`) require hand-synchronization;
its lazy global registries make import order semantics and generate documented
circular-import workarounds; and both numba's datamodel and xDSL's dependency
story (R3) show that letting a foreign vocabulary (llvmlite types, a 0.x
package's `__eq__` semantics) into the extension contract is the thing that
eventually breaks every extension at once. So: one definition per battery,
rules-in-a-matrix for everything op-shaped, backend vocabulary confined to the
backend record, and explicit registries.

---

## The recommended hook set, concretely

### 0. The two shared structures

**Target tokens** — a plain class lattice, no registries attached:

```python
class Target: ...
class Generic(Target): ...
class CPU(Generic): ...
class GPU(Generic): ...
class WGSL(GPU): ...
class CUDA(GPU): ...
class Metal(GPU): ...
class PyEval(CPU): ...          # the zero-dependency interpreter backend
class C(CPU): ...
```

A registration declares the *most general* token it serves; resolution picks the
most-derived applicable entry for the active target (`issubclass` walk — numba's
MRO rule, R1 §5.1). `target=Generic` is the portable default and should be ~95%
of the stdlib, mirroring numba's measured 491-vs-272 pyramid.

**The Registry** — one explicit object, passed to the frontend/compiler at
construction (this is the M0 "dialect must be an input to the frontend, not a
dependency of it" fix, and the anti-global-registry lesson from R1 §6):

```python
@dataclass
class Registry:
    ops:       dict[str, Op]                                  # "core.sqrt" -> Op
    rules:     dict[tuple[Op, str], Callable]                 # (op, aspect) -> rule
    overloads: dict[object, list[OverloadEntry]]              # np.mean / qualname -> entries
    methods:   dict[tuple[type, str], list[OverloadEntry]]    # (ColorType, "to_oklab")
    attrs:     dict[tuple[type, str], list[OverloadEntry]]
    typeof:    TypeofDispatcher                               # singledispatch + __pdum_type__
    marshal:   dict[type, MarshalSpec]                        # DSLType class -> flatten/unflatten
    backends:  dict[type[Target], Backend]
    transforms: dict[str, Transform]                          # aspect name -> driver

    def extend(self) -> "Registry": ...   # ChainMap layering: stdlib -> user -> session
                                          # (numba DataModelManager.copy()/chain(), R1 §4.3)

DEFAULT = Registry(...)   # decorators default to this; every decorator takes registry=
```

`OverloadEntry = (target_token, prefer_literal, impl_generator)`. Layering via
`extend()` gives per-project and per-session registries without global mutation;
the decorators' `registry=DEFAULT` default keeps numba-level ergonomics.

An `Op` is a name plus metadata — **never** a subclass of the IR node type
(tinygrad discipline: one node shape, open op set, closed node class):

```python
@dataclass(frozen=True)
class Op:
    name: str                      # dialect-namespaced: "core.sqrt", "units.convert"
    arity: int | None
    traits: frozenset[Trait]       # Pure, Commutative, Linear-in(argpos), ...

def defop(name, *, type_rule, traits=(), registry=DEFAULT) -> Op:
    op = Op(name, ...); registry.ops[name] = op
    registry.rules[(op, "type")] = type_rule
    return op

def rule(op, aspect, *, registry=DEFAULT):    # the single rule-registration door
    def deco(fn): registry.rules[(op, aspect)] = fn; return fn
    return deco
```

The `traits` field carries the transposability metadata R4 lesson 6 demands
(which args an op is linear in) so the reverse-mode column can be added later
without re-touching op definitions.

### (a) Per-backend intrinsics — the ~30-op floor

An intrinsic is an op the backend renders natively. It lives in the backend's
`code_for_op` table, and **the shared decomposition rules are gated on that
table's keys** (tinygrad's `get_late_rewrite_patterns(supported_ops)`, R7 §1.5):

```python
# in the WGSL backend record:
code_for_op = {
    ops.sqrt:  "sqrt({0})",
    ops.fma:   "fma({0},{1},{2})",
    ops.sin:   "sin({0})",
    ...                                   # ~30 entries
}
# in the C backend record:
code_for_op = {ops.sqrt: "sqrtf({0})", ...}

# shared, written once in core rules:
def decompositions(supported: set[Op]) -> list[RewriteRule]:
    out = []
    if ops.pow not in supported:  out.append(pow_to_exp_log)
    if ops.sinh not in supported: out.append(sinh_to_exp)
    ...
    return out
```

`sqrt → WGSL sqrt vs C sqrtf` is therefore a **one-line table entry per
backend**, and an op absent from a backend's table falls through to portable
decomposition automatically. Backend-specific quirks that aren't a format string
(WGSL packed 8/16-bit emulation, `a != a → isNan`) go in the backend's
`legalize` rewrite list, applied before rendering. The primitive set should be
kept deliberately small (~25–35 ops: arithmetic, compares, select, a few
transcendentals, load/store/index, range/if); everything above it is surface (b).

### (b) Library batteries — `@overload`, stolen wholesale

The single most evidence-backed decision in this verdict. One pure-Python
function provides typing + implementation + staging; it is called with **types**
at typing time and returns an impl written **in the DSL subset**, which the
active backend's own jit compiles (and the inliner may inline):

```python
@overload(np.mean)                          # also: @overload("mean") for DSL-only names
def mean_ovl(a):                            # `a` is a pdum TYPE, not a value
    if isinstance(a, ArrayType):
        acc0 = zero_of(a.dtype)             # type-time constant, closed over
        def impl(a):                        # DSL-subset Python; compiled per target
            s = acc0
            for i in range(a.size):
                s = s + a[i]
            return s / a.size
        return impl
    return None                             # decline -> next entry / error

@overload_method(ArrayType, "mean")(mean_ovl)     # method syntax, same impl
@overload_attribute(ArrayType, "T")
def array_T(a): ...

@overload(np.mean, target=CUDA)             # optional per-backend fast path,
def mean_cuda(a): ...                       # wins over Generic by MRO
```

Resolution: the frontend's call-typing step looks up
`registry.overloads[callee]`, filters entries by `issubclass(active_target,
entry.target)`, most-derived first, and calls each generator with the argument
types until one returns an impl. The impl's compilation request re-enters the
normal `(FnType, arg_types)` cache — batteries are cached and specialized
exactly like user code, per backend, for free.

Note the deliberate irony flagged in R1 §3.2: inside an overload generator,
closing over type-time constants (`acc0`) is *staging*, the one place where
"freeze the capture" is what you want — because the generator runs at typing
time, per type signature, its closures are by construction type-keyed. No
special mechanism needed.

The `prefer_literal=True` flag on an entry requests `Literal[...]`-lifted
argument types (the explicit value-dependent opt-in from
`dsl_caching_layer.md`), matching numba's `prefer_literal` and keeping
value-specialization out of the default path.

### (c) Type extensions — three registrations per type

A new type (record, `Color`, future `Quantity`) plugs in with:

```python
# 1. value -> type (feeds capture typing and the cache key; must be cheap)
@typeof_impl.register(Color)
def _(v): return ColorType()               # or duck-typed: Color.__pdum_type__

# 2. marshaling: static/dynamic split (JAX pytree shape, R4 §4)
register_marshal(ColorType,
    leaf_types = (f32, f32, f32),                     # static: enters FnType via typeof
    flatten    = lambda c: (c.r, c.g, c.b),           # dynamic: per-call, must be cheap
    unflatten  = lambda r, g, b: Color(r, g, b))      # for Python backend / returns

# 3. methods & attributes: surface (b)
@overload_method(ColorType, "to_oklab")
def color_to_oklab(c):
    def impl(c): ...                                   # DSL subset
    return impl
```

Rules with teeth (from R4's pytree contract): the static half must be hashable
and participates in `FnType`/cache keys; `flatten` runs per call and must not
allocate beyond a tuple. The per-backend **layout planner** (leaves → uniform
buffer offsets / kernel args / ctypes fields) is *not* part of this surface — it
belongs to the backend's runtime (surface d), which receives only `leaf_types`.
This is the numba-datamodel fix (R1 §4.5): structural recursion in core, leaf
vocabulary per backend, so no type extension ever names a backend type.

Records are a stdlib client of this surface: `RecordType` is a frozen dataclass
whose identity **is** its field layout (name, dtype, offset — R8 §1.1), with
field getattr as a core op and methods via `@overload_method`. Units, when they
arrive, put the dimension/unit in the **static half** (so it is in the type and
the cache key) and register a pack-time conversion in the marshaling plan (R8
§3.2) plus, later, a `"unit"` aspect column for checking — no new hook kind
needed.

### (d) Backend registration — the capability record

A backend is data plus two functions, registered under a target token:

```python
@dataclass
class Backend:
    target:      type[Target]                    # WGSL
    type_map:    dict[DSLType, str]              # f32 -> "f32" / "float" / ...
    code_for_op: dict[Op, str | Callable]        # surface (a)
    legalize:    list[RewriteRule]               # backend-local pre-render rewrites
    render:      Callable[[IRModule], str]       # types -> source text
    runtime:     Runtime                         # buffers -> dispatch (separate seam!)

class Runtime(Protocol):
    def plan_layout(self, leaf_types) -> Layout       # once per FnType, cached w/ artifact
    def pack(self, layout, leaves) -> None            # the hot-path parameter write
    def compile(self, source) -> Artifact
    def launch(self, artifact, packed) -> Any

registry.register_backend(wgsl_backend)
```

Renderer and runtime never import each other (tinygrad's measured seam, R7
§1.5); the compile cache key is `(IR content hash, backend.target, backend
config flags)` below the type-keyed cache (R7 lesson 5, curing M0's cache
hygiene faults). Evidence-based budget: **a new source-emitting backend is
50–300 renderer lines + 130–220 runtime lines** (WGSL 115 / Metal ~52 / CUDA ~78
in tinygrad); if a backend needs more, the missing piece belongs in shared
decomposition rules, not in the backend.

### (e) Transformation registration — aspect columns, present from day 1

The rule matrix is the transformation hook. Day 1 populates two columns
(`"type"` from `defop`, `"lower"` implicitly via `code_for_op` + decomposition);
a transformation later is:

```python
# one driver: an IR -> IR pass that walks eqns and consults its column
class Jvp(Transform):
    aspect = "jvp"
    def run(self, ir, registry): ...       # ~80-150 lines (autodidax-measured)

registry.transforms["jvp"] = Jvp()

# one rule per op, added next to the op or in a separate battery module:
@rule(ops.sin, "jvp")
def sin_jvp(primals, tangents): ...
@rule(ops.mul, "transpose")
def mul_transpose(ct, x, y): ...
@rule(ops.sin, "vmap")
def sin_vmap(args, dims): ...
```

Ops missing a rule fail only when that transformation reaches them — graceful
partiality (R4 §3). Higher-order ops (`if`, `for`) are the cost concentration
(~180 lines each per JAX's `cond`); keep the set to call/if/loop. tinygrad's
132-line `pm_gradient` (R7) is evidence the columns can even be plain rewrite
rule sets if the IR is a rewritable graph.

User escape hatch, same shape as a built-in (R4 §3, "a user device function
with a hand-written derivative registers exactly like an op"):

```python
@custom_jvp
def turbulence(x): ...          # DSL-subset function
@turbulence.defjvp
def turbulence_jvp(primals, tangents): ...
```

Crucially, **the kernel ships the matrix, not the transformations**: the only
day-1 obligations AD imposes are IR properties (typed binders, multiple results,
no hidden state) and the `Linear-in` trait slot on `Op` — both nearly free now,
both expensive to retrofit.

### Day-1 scope and line budget

| Component | Lines (est.) | Basis |
|---|---:|---|
| Registry + `extend()` layering | ~120 | numba `DataModelManager` 68 + manager glue |
| Target tokens + MRO resolution | ~80 | numba `target_extension.py` 169 incl. parts we drop |
| `defop` / `rule` / `Op` | ~60 | |
| `@overload` family + typing-time resolution | ~250 | numba `extending.py` 597 incl. numba-specific ceremony |
| `typeof` dispatch + `__pdum_type__` | ~60 | |
| marshal registry + static/dynamic contract | ~150 | JAX pytree core ~190 in autodidax |
| Backend record + registration + cache-key glue | ~90 | |
| `Transform` base + `custom_jvp` shells | ~100 | mostly deferred |
| **Total hook kernel** | **~910** | sits beside the ~300–600-line IR (V-IR question) |

---

## Considered and rejected

1. **Numba's dual namespace (typing templates + `@lower_builtin`) as the
   primary battery path.** Rejected: two parallel registries kept in sync by
   convention, and everything written this way is codegen-vocabulary-bound —
   the category of numba extension code that broke in both the numba-cuda fork
   and the MLIR rewrite (R1 §3.3, §5.2). We keep only its descendant idea:
   backend intrinsic *tables* (surface a), which are data, not paired code.

2. **xDSL dialects (the dependency) as the extension unit.** Rejected for the
   kernel: 0.x with ~4 breaking releases per 10, 146 kLOC of which ~12% is
   needed, and its extension machinery contributes nothing to the novel layer
   (caching, typeof, marshaling) while putting a third party's
   `__eq__`/`__hash__` semantics under the cache key (R3 §6). We keep the
   *concepts* — dialect-namespaced op names, value/attribute split, traits —
   inside our own registry, and keep xDSL as an optional pinned dev-time
   differential-testing oracle per R3's own verdict.

3. **JAX-style module-level dict registries** (`ad.primitive_jvps[prim] = fn`
   at import time). The matrix is adopted; the *storage* is rejected. Global
   mutable registries made import order semantics in numba (lazy
   `RegistryLoader`, five documented circular-import workarounds, R1 §6) and
   are why numba cannot say which targets exist at decoration time. Explicit
   `Registry` objects with a `DEFAULT` convenience preserve ergonomics without
   the failure mode.

4. **DaCe's replacements-registry (build-a-subgraph callables) as the primary
   battery surface.** Rejected as primary: it makes every battery author write
   IR-construction code against visitor/state internals — DaCe's numpy surface
   costs ~7.4k lines partly for this reason, and such code is coupled to the IR
   generation (R2 §3). Its genuinely good parts are absorbed: the
   method/attribute/operator tables with MRO fallback *are* our
   `methods`/`attrs` tables, but the registered value is an `@overload`-style
   impl generator, not a graph-builder.

5. **DaCe LibraryNode + per-target expansions as the day-1 shape for every
   battery.** Rejected for day 1: meta-node + expansion classes + environment
   declarations is heavier ceremony than `@overload`, and its main advantage
   (per-instance implementation selection, declarative build environments)
   matters for BLAS-scale native libraries pdum doesn't call. The idea survives
   in miniature as `@overload(..., target=CUDA)` fast-path entries overriding
   `Generic` ones.

6. **Per-feature IR node subclasses as the extension mechanism** (MLIR-style
   open class hierarchy). Rejected on tinygrad's evidence (R7 §5): one closed
   node shape with an open op set keeps every pass, rewriter, and cache-key
   hasher working for all future ops; extensibility lives in rules over the IR,
   not subclasses of it. xDSL/MLIR need open node classes for generality pdum
   doesn't have.

7. **Trace-based extension acquisition** (batteries as traced Python à la
   jnp.*). Rejected with the frontend decision (R4 §1): statement-heavy
   shader-style impls need real `if`/`for`; `@overload` impls go through the
   same AST frontend as user code, so batteries and user code share one
   language subset — one thing to document, one thing to test.

---

## Implications for the architecture

1. **The frontend takes a `Registry` parameter; there are no import-time
   registrations in the kernel.** `jit(fn, registry=DEFAULT)`; the stdlib
   populates `DEFAULT` via the same five public surfaces users get. This
   mechanically enforces the prime directive: if the stdlib can't add a feature
   through the hooks, the hooks are wrong.

2. **Call typing has one resolution order**, implemented once:
   `__pdum_dsl__`-convertible → `registry.overloads[callee]` (target-MRO
   filtered, most-derived token first, then registration order) →
   `registry.methods[(typeclass, name)]` with type-MRO fallback → error naming
   the callee, the arg types, and the nearest declined entries. (DaCe's
   resolution-order clarity + numba's MRO, minus interpreter-callback
   fallbacks.)

3. **The `(FnType, arg_types)` cache is the only compilation entry point, and
   overload impls go through it.** A battery impl compiled for `(ArrayType(f32,
   1, C),)` on WGSL is cached exactly like a user kernel; the per-op compile
   cache below it keys on `(IR hash, target, flags)`. No third cache for
   extensions.

4. **`Op.traits` is the forward-compatibility contract for transformations.**
   `Pure`, `Commutative`, `Linear-in(i)` must exist on day 1 even though
   nothing consumes `Linear-in` until reverse mode ships; the IR must have
   typed binders and multiple results now (R4 lesson 6).

5. **Backend leaf vocabulary never appears in core or in any extension
   signature.** Type extensions declare `leaf_types` in DSL types; the
   backend's `Runtime.plan_layout` is the only place `@group/@binding`,
   `ctypes`, or CUDA arg structs are spelled. This is the single rule whose
   violation killed numba's datamodel extensions twice (R1 §4.5).

6. **The dispatch hot path is registry-free.** All five surfaces operate at
   typing/compile time (phase B miss). The phase-A/phase-B hit path touches
   only: fingerprint → cache dict → precomputed `(artifact, layout, pack,
   launch)` fastpath record (R4 §6). `marshal.flatten` is the one extension
   callable on the hot path — document and enforce its cheapness (no
   allocation-heavy flattens; measure in ns in CI).

7. **Deprecate-nothing surface discipline:** the five decorator signatures
   (`defop`/`rule`, `overload*`, `typeof_impl`/`register_marshal`,
   `register_backend`, `Transform`/`custom_*`) are the *only* stable contract;
   IR internals, renderer internals, and the registry's storage layout are
   explicitly unstable. This is the boundary that history shows survives
   substrate rewrites (R1 §5.2).

8. **t-string mini-languages need no new hook kind**: an einops-like sub-DSL is
   a function that parses its template and returns/constructs core-dialect ops
   — it registers via `@overload` (as a callee) or as frontend sugar that emits
   ops, exactly like tinygrad's mixin layer (R7 lesson 10).

---

## Confidence and what would change my mind

**Confidence: high** on the two load-bearing choices — `@overload`-in-the-subset
as the battery surface (the only mechanism in the corpus with *measured*
cross-target portability: 491 portable vs 272 bound entries, a 98%-verbatim GPU
port, and survival through two substrate rewrites) and the op×aspect rule
matrix (independently converged on by JAX's registries and tinygrad's rule
sets, at measured costs of ~300 and ~150 lines respectively). **Medium** on the
specifics: the exact Registry layering API, the decomposition-gating mechanism
(`code_for_op.keys()` as the capability declaration), and the ~910-line budget
(±40%).

Evidence that would change the verdict:

- **If WGSL's language restrictions (no recursion, no function pointers,
  limited loop forms) make a meaningful fraction of subset-written batteries
  uncompilable on the WebGPU target**, the pyramid base grows: more surface-(a)
  intrinsics and per-target `@overload` entries than numba's ratios predict.
  Test early by porting 10 representative batteries (mean, clip, smoothstep,
  color conversions) to the WGSL backend in the first milestone.
- **If typing-time overload resolution shows up in interactive compile latency**
  (many entries × isinstance chains), the `overloads` table needs an index by
  leading arg typeclass — a data-structure change, not a surface change.
- **If reverse-mode AD proves not to decompose into per-op rules** for the
  structured-control-flow IR (i.e., it needs whole-program analyses beyond
  jvp+partial-eval+transpose), the `"jvp"`/`"transpose"` columns would be
  replaced by a pass-level hook — the matrix survives, the aspect granularity
  changes. Autodidax's 439-line existence proof makes this unlikely.
- **If the explicit-registry ergonomics fail in notebooks** (users forgetting
  `registry=` and losing registrations across cells), fall back to a
  process-global default registry with a snapshot/restore API — a concession,
  not a redesign.

---

## Design lessons for pdum.dsl

1. **Ship five hook surfaces and no sixth**: intrinsic tables (a), `@overload`
   batteries (b), typeof+marshal+methods for types (c), backend capability
   records (d), rule-matrix aspect columns + `custom_*` (e). Every desiderata
   capability — records, units, einops t-strings, autodiff, new backends — maps
   onto these five; a feature that doesn't fit is a design smell to resolve
   before coding.
2. **Write the stdlib exclusively through the public hooks**, in the DSL subset
   wherever possible; keep a counted ratio (portable `@overload` entries vs
   per-backend entries) in CI as the batteries-economics health metric, with
   numba's ~2:1 pyramid as the floor and better expected since our primitive
   set is deliberately small.
3. **Make the backend's declared op table the single source of truth for
   capability**: decomposition gating, legalization, and "does this program
   compile on this target" errors all read `code_for_op.keys()` — never a
   second capability list to drift.
4. **Registries are values, passed in** — `Registry.extend()` layering, no
   module-level lazy loaders, no import-order semantics. The compiler is a pure
   function of `(fn, FnType, arg_types, registry, backend)`.
5. **Reserve the aspect columns and `Linear-in` traits on day 1** even with AD
   unshipped; typed multi-result IR + op traits are the cheap-now,
   impossible-later part of transformation support.
6. **Keep every hook off the hot path except `marshal.flatten`**, and hold that
   one to an allocation-budget test. Registration surfaces exist at compile
   time; the render loop sees only the fastpath record.
