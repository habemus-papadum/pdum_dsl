# Caching layer for a numba-like DSL with reusable closure specializations

Design notes for a numba-like JIT whose caching layer reuses a compiled inner
across capture *values*, as long as the capture and argument *types* are stable.
Companion to `closure_specialization.md` (which explains why numba 0.65.1 does
**not** do this). Python mechanics below were checked on CPython 3.14. This is a
*reviewed* draft — a compiler-engineer critique pass corrected several
load-bearing claims; see "Review pass" at the end for the delta.

## Goal

One compile per *type* signature, reused across all *values*:

```
specialize(template, env_types, arg_types, generation) -> native code
```

is built once and reused for every `(env_values, arg_values)` whose types match.
The number of native compiles is bounded by the number of distinct
`(template, env_types, arg_types)` triples (times redefinition generations) — not
by the number of capture/argument *value* combinations. For `closure(5)` and
`closure(6)` feeding `m`, that means **one** compile of `inner` (for
`int64 -> int64`) and **one** compile of `m`, with the captured `5`/`6` flowing
through as runtime data.

This is the Julia model, not the numba model. The doc states the type system, the
cache structure, the closure-conversion ABI, the exact Python you use to extract
the keys, and — given how many of them there are — the correctness hazards.

## The reframing: a closure is (identity, typed env, env values)

Numba conflates the captured value into the type — it freezes `x` as a
compile-time constant, so `closure(5)` and `closure(6)` produce different code.
Julia splits the three concerns, and so does this design:

| Concern | Julia | This DSL |
|---------|-------|----------|
| Identity of the function | singleton type `typeof(f)` (one per definition) | the **code object** of the `def` site (compared by value) |
| Types of the captured variables | type parameters of the closure struct `#inner{T...}` | `env_types` tuple |
| Values of the captured variables | fields of the struct instance | `Env` instance (runtime data) |

The first two together form the **function type**; the third is a runtime value.
Specialization keys on the function type plus the argument types — never on the
captured values (with one deliberate exception: *value-dependent* types, below).
That single move is what makes the inner reusable.

## Two arrival points

The information needed to emit native code arrives at two distinct times:

| Phase | When | What you learn | What it produces |
|-------|------|----------------|------------------|
| **A — capture** | `closure(6)` returns | the template (code object) + the captured values, hence `env_types` | a `FnType` value + an `Env` instance. *No compilation.* |
| **B — call** | `m(a, f)` or `f(y)` runs | the argument values, hence `arg_types` | the compiled native function. *Compilation happens here.* |

There is exactly **one** cache that matters — the native-code cache, keyed on the
full `(FnType, arg_types, generation)` and populated at phase B. Phase A is
compile-free regardless. A *secondary* memo from `(template, env_types)` to the
**env layout** is a cheap optimization (it avoids recomputing the struct layout on
every `closure()` call), not a second architectural tier — earlier drafts oversold
it as one. Keep the "two arrival points" mental model; drop the "two necessary
caches" framing.

## Type system

### Template identity = the code object, compared by value

Use `func.__code__` as the identity of a `def` site. CPython compiles the inner's
code object **once** (it lives in the enclosing function's `co_consts`) and every
execution of that `def` reuses it, so all closure instances from one `def` share
it *within a process*:

```python
f5, f6 = closure(5), closure(6)
f5.__code__ is f6.__code__       # True  -- one code object per def site
```

But the cache must key on the code object via a dict/tuple, and **code objects
hash and compare by value, not identity** (verified):

```python
c1 = compile("x+1", "<f>", "eval"); c2 = compile("x+1", "<f>", "eval")
c1 is c2            # False
c1 == c2            # True   -- value equality over co_code/co_consts/co_firstlineno/...
{c1:'A'}.get(c2)    # 'A'    -- value-equal distinct objects HIT the same slot
```

This semantics — not object identity — drives the real behavior:

| Situation | Effect | Why |
|-----------|--------|-----|
| Repeated `closure()` in one process | hit | same code object |
| Re-run **unchanged** source (notebook cell, re-`exec`) | **hit** | re-compiled code object is value-**equal** to the old one |
| **Edit** the body and re-run | miss → recompile | `co_code`/`co_consts`/`co_firstlineno` differ → value-**unequal** |
| Two same-line lambdas with different bodies | distinct keys | distinct `co_consts` → unequal |

So invalidation-on-edit works, but because the *fields* differ, not because the
object is new. (An earlier draft claimed "new object → invalidation"; that was
backwards for an unchanged re-run.)

**Two correctness traps that fall out of value-equality:**

1. **`id(code)` reuse** — never key on `id(code)`; a collected code object's id is
   reused and would alias a new template. Keying on the object gives value-equality
   and is correct, but means you hold a strong ref (feeding the L-cache leak below).
2. **Globals collision under freeze-globals.** Two value-equal code objects can
   close over *different global namespaces* and, if you freeze globals as
   constants, must **not** share a specialization (verified: `return G + y`
   exec'd with `G=1` vs `G=2` yields value-equal code objects that behave
   differently). Therefore, when freezing globals, the cache key must also include
   a stable tag for the global environment (module name / `id(fn.__globals__)`)
   **and** the generation counter.

For an on-disk key, code-object value-equality does not survive a process; use
`(co_filename, co_firstlineno, co_qualname, source_hash, ...)` instead — see the
disk-cache section.

### Function type

`FnType` is a pure value:

```
FnType = (template_code, env_types)
```

Hashable; equal iff the code object is value-equal and `env_types` is equal. Keep
`FnType` a pure value — store the env *layout* in a side table keyed by `FnType`,
not as a field on `FnType` (a `layout` field would have to be excluded from
`__eq__`/`__hash__`, a footgun). This is the type a closure presents when passed
to another compiled function:

| Construction | `FnType` |
|--------------|----------|
| `closure(5)` | `(inner_code, (int64,))` |
| `closure(6)` | `(inner_code, (int64,))` — **same** → shared specialization |
| `closure(3.0)` | `(inner_code, (float64,))` — **different** → its own specialization |

This realizes requirement (1): a type unique per function (the code object) plus
the types of what it closes over, plus a way to instantiate it with fresh
closed-over values (the env layout below).

### `typeof`: value → type

`typeof` runs over every captured value (phase A) and every argument (phase B), so
it is on the hot path *and* it defines correctness — a too-coarse `typeof`
silently reuses the wrong specialization. The full lattice, with the corner cases
that an earlier draft skipped:

| Python value | DSL type | Note |
|--------------|----------|------|
| `int` | `int64` / `uint64` / **bigint-or-error** | **must branch on range.** `np.int64(2**70)` overflows; a captured `5` and `2**70` must not share a type or the value silently corrupts the `Env` field |
| `float`, `bool`, `complex` | `float64` / `bool` / `complex128` | |
| numpy scalar | by `dtype` | |
| `ndarray` | `(dtype, ndim, layout, byteorder, writeable)` | **canonicalize:** 1-D (and 0-stride) contiguous → `C`; track byteorder (`<i8`≠`>i8`), and writeable/read-only (capturing read-only then writing → corruption). A strided view is layout `A` |
| `None` | `none` (a singleton type) | distinct from any `Optional[T]` |
| optional/union values | `Optional[T]` / tagged union | needed once any path yields `None` |
| `str`, `bytes` | `unicode` / `bytes` | |
| `tuple` | recursive, **element-wise** with arity | `(1,2)` and `(1,2,3)` are different types; length is part of the type |
| `list`/`dict`/`set` | homogeneous reflected type, or unsupported | empty containers have no element type → either defer typing or reject |
| DSL closure | its `FnType` | nested closure; see env layout |
| plain Python callable / builtin / C func | an opaque-callable type or unsupported | **not** a `FnType` |

### Fingerprints: a fast key that is still structural

To avoid a full `typeof` per call, memoize, but the fingerprint must be
**structural**, not a tuple of `type(v)` tags — `type(v)` collides on tuple arity,
int range, and array layout (verified). A correct fingerprint includes:

- tuple length + each element's fingerprint (recursively),
- the int range bucket (int64 / uint64 / big),
- for arrays, `(dtype, ndim, layout, byteorder, writeable)` — which means reading
  `arr.flags`, so the array fingerprint is **not** free.

Then `(template, env_fingerprint)` and `(FnType, arg_fingerprint)` are sound fast
keys, falling back to full `typeof` on a fingerprint miss.

### Value-dependent specialization (the deliberate exception)

"Key on types, not values" is only universally safe in a type system with **no
value-dependent types**. Real kernels want some captured values as compile-time
constants: an array dimension that enables static shapes, a loop bound that enables
unrolling, a flag that selects a branch. Offer an explicit opt-in — a `Literal[v]`
/ `Val{v}` marker (numba's `Literal`, Julia's `Val`) that lifts a selected capture
into the *type*, so its value participates in the key and gets constant-folded.
Everything not marked stays runtime `Env` data. This recovers numba's per-value
optimization where you ask for it, while keeping the default reusable. State the
default loudly: unmarked captures are runtime values; marked captures recompile per
value.

### Env layout and constructor

For a `FnType`, derive an **environment layout**: named fields from `co_freevars`,
typed by `env_types`. `co_freevars` and `__closure__` align positionally, and
`co_freevars` is the compiler's **sorted** order, not source order (verified:
`def outer(z, scale, x)` → `co_freevars == ('scale','x','z')`):

```python
names  = inner.__code__.co_freevars                      # sorted, cell order
values = tuple(c.cell_contents for c in inner.__closure__)
env    = dict(zip(names, values))                        # the runtime payload
```

Recursion in the layout: when a free var is itself a DSL closure, its `env_types`
entry is a `FnType` and the runtime `Env` field holds **that closure's `Env`** (a
nested `(code_ptr, env)` pair). `pack` recurses.

A useful consequence of keying on env *types*: the captured values can be huge
arrays or otherwise unhashable — the cache key never touches them, only their
types. Keys stay small and cheap regardless of payload.

## Dispatch flow

```python
# Phase A — at closure(6)
def make_handle(fn):
    code  = fn.__code__
    cells = fn.__closure__ or ()                 # None when there are no free vars
    vals  = tuple(safe_cell(c) for c in cells)   # guard empty cells (recursion)
    etypes = typeof_tuple(vals)                  # env_types, structural fingerprint fast path
    ftype  = FnType(code, etypes)                # pure value; layout looked up in a side table
    env    = layout_for(ftype).pack(vals)        # runtime Env instance (new values)
    return Handle(ftype, env)

# Phase B — at handle(y) or inside a compiled m calling f(...)
def call(handle, args):
    atypes = typeof_tuple(args)
    key    = (handle.ftype, atypes, GENERATION)  # generation is part of the key
    native = L2.get(key)
    if native is None:
        native = compile_once(key)               # synchronized; see thread-safety
    return native(handle.env, *args)             # env passed as hidden first arg
```

`safe_cell` guards `cell_contents` raising `ValueError: Cell is empty` — a
self-referential recursive closure has a not-yet-bound self-cell at construction.

### How the higher-order kernel `m` gets shared too

`m(a, f)` is itself compiled. Its argument types are `(typeof(a), f.ftype)`.
Because `closure(5)` and `closure(6)` produce the **same** `ftype`, `m`
specializes once on `(array(int64,1,C), FnType(inner_code,(int64,)))` and is
reused. While compiling `m`, the call `f(a[i])` is resolved through `f.ftype`:
the template is known, so it triggers exactly one compile of the inner for
`(ftype, (int64,))`. The capture (`5` vs `6`) is never seen by the compiler — it
rides through `m` into the inner as the runtime `Env`. Net for the example: one
compile of `m`, one compile of `inner`, both reused on the second call.

## ABI / closure conversion

Compiled inner signature: `inner_native(env_ptr, *args)`; the caller passes
`env_ptr` from the runtime closure value, and the inner loads captures from it.
**Decide up front whether the DSL is fully monomorphizing.** If every signature
pins a concrete `FnType` (the model the rest of this doc assumes), then every call
site inside a given specialization sees exactly one `FnType` for `f` — always
monomorphic, the callee is statically known, the call is direct and can be inlined,
and no inline cache is needed. A polymorphic site only arises if the type system
admits **abstract/union** function types (an unspecialized `Callable`, or closures
of different `FnType` in one heterogeneous container). That is a legitimate choice,
but it is a *different* design: you then specify how union `FnType`s are formed and
keyed, and the site dispatches on `FnType` through a small inline cache (V8-style
PIC) with an indirect call. Pick one; do not assume both.

## Practical extraction in Python

Everything the keys need is reachable from the function object — no source parsing
required:

| Handle | Gives |
|--------|-------|
| `fn.__code__` | template identity (value-compared); `co_filename`, `co_firstlineno`, `co_qualname` |
| `fn.__code__.co_freevars` | captured-variable names, **sorted**, in cell order |
| `fn.__closure__` | tuple of `cell`s (`None` if no free vars); `cell.cell_contents` (may raise `ValueError` if empty) |
| `fn.__globals__` + freeze policy | referenced globals — but compute the freeze set carefully (below) |
| `fn.__defaults__` / `__kwdefaults__` | default arg values, if bound at definition |

**Freeze set, precisely.** `co_names ∩ __globals__` is both over- and
under-approximate (verified): `co_names` includes attribute names and builtins
(`len`, `some_attr`) that are not globals, and misses dynamic `globals()['x']`
access. Define the policy explicitly: which names are frozen constants, how
builtins are treated, and that dynamic global access is unsupported (or forces a
deopt).

### When you actually need to walk the stack

You do **not** need the stack for closure identity — you hold the function object,
so `__code__` is in hand. Walk the stack only when the DSL entry point is a
*call-site* construct and you want the invocation location (a per-call-site inline
cache or profile bucket):

```python
import sys
frame = sys._getframe(1)
site  = (frame.f_code, frame.f_lasti)   # code object + bytecode offset = exact, hashable call site
```

`(f_code, f_lasti)` is a precise per-site key (verified). Resolve once per site and
cache by that pair; keep it off the hottest path.

## Correctness and invalidation

| Hazard | Policy |
|--------|--------|
| **Capture rebinding vs mutation — three cases, not two.** | (a) *Rebind* the cell to a new-typed value → different `env_types` → different `FnType` → correct miss; re-derive `env_types` every phase A. (b) *Mutate the contents* of a captured array/list → the `Env` holds the **reference**, so mutations stay visible; "value-snapshot" is really *reference*-snapshot (for arrays this is numba-like reflection — fine, but say so). (c) *Type drift* under a lazy-cell mode requires re-deriving `env_types` and re-keying **every call**, which defeats the point — so default to snapshot-at-capture, offer lazy explicitly. |
| **Frozen globals.** | Folding globals into native code means a changed global is stale unless invalidated. Fold a global-env tag + generation into the key (above); decide whether builtins are frozen. |
| **Redefinition / world age.** | A single global `generation` counter bumped on any `@jit` redefinition, folded into the L2 key, is correct but a **sledgehammer** — it invalidates *all* native code, not selectively. Julia's world age is precise because it tracks a *dependency graph* (which specializations called the redefined method). If you inline inner into `m` or call frozen helpers directly, only such a graph invalidates the right backedges; the global counter recompiles everything. Also: `generation` is in the *key*, not in the held closure value — so a closure created before a bump adopts the new world on its next call. Surface that semantics. |
| **Specialization explosion.** | Bounded by distinct type triples (× generations), not values — but unbounded across types. Cap L2 with an LRU and `log()` evictions. |
| **L-cache growth / leak.** | The env-layout memo holds code objects by strong reference; every edit-and-rerun adds a value-unequal entry that never collides with the old, so it accumulates dead templates and layouts forever. Bound/evict it too, or retire superseded templates. |
| **Thread safety.** | The `get → compile → put` in the pseudocode races: two threads miss and both compile (wasted work, or JIT-symbol corruption). Use a per-key **future** with `compiling`/`ready` states (second arrival awaits the first). The nested compile of `inner` while compiling `m` means the compile lock must be **reentrant** (or per-key with a deadlock-free order). A half-built **recursive** entry must be published as a forward-declared symbol in a `compiling` state — callers bind the symbol, never call into incomplete code. Read `generation` once at the start of a compile and store the result under that value to avoid a mid-compile bump race. |

### Recursion: two distinct mechanisms

The earlier draft's "register the FnType before compiling the body" conflated two
things:

1. **Compile-time recursion (L2).** A recursive call needs the in-progress native
   specialization registered as a forward-declared symbol *before* lowering the
   body, so the call binds to it and is patched on completion. This is about the
   native cache, not the type.
2. **Self-referential type (phase A).** A recursive nested function captures
   *itself* as a free var (`co_freevars == ('fact',)`, the cell holds the function),
   so its `env_types` must contain its **own** `FnType`. Intern a placeholder
   `FnType` before resolving env types, or `typeof(self) → FnType → env_types →
   typeof(self)` recurses forever. The self-cell may also be empty at construction
   (`safe_cell` above).

## Disk cache (cross-process)

Code-object value-equality does not survive a process, so the on-disk key cannot
use the code object. It must:

- use `(co_filename, co_firstlineno, co_qualname, source_hash)` for the template;
- lower nested `FnType`s in `env_types`/`arg_types` **structurally** to
  `(source_pos, source_hash, …)` — they contain in-process code objects and are
  otherwise unserializable;
- hash the **dependency closure**, not just the `def`: frozen global *values* and
  the sources of transitively-called helpers (a changed frozen global or callee
  leaves `source_hash` unchanged → stale native code);
- include toolchain tags — DSL version, LLVM version, and CPU target (`march`) —
  or you load ABI-incompatible objects across upgrades.

## Worked trace (the `closure.ipynb` example)

In one process `closure(5)` and `closure(6)` share the **same** code object, so
the layout memo hits trivially; the interesting reuse is the native cache:

| Step | Native key `(FnType, arg_types, gen)` | Native compiles |
|------|----------------------------------------|-----------------|
| `closure(5)` | — (phase A only) | 0 |
| `m(a, closure(5))` | miss `m`; miss `inner@(i64,)` → compile both | **2** |
| `closure(6)` | — (phase A only) | 0 |
| `m(a, closure(6))` | **hit** `m`; **hit** `inner@(i64,)` | **0** |

Contrast with numba (`closure_specialization.md`): there, step 4 recompiles both
because numba keys on dispatcher object identity and freezes `x`. Here the capture
`6` is a new `Env` threaded through reused code.

## Where this sits relative to numba and Julia

| Aspect | numba 0.65.1 | Julia / GPUCompiler | This DSL |
|--------|--------------|----------------------|----------|
| Function identity | `Dispatcher` instance | singleton type per definition | code object per `def` site (value-compared) |
| Captured var | frozen constant | typed struct field (runtime) | typed `Env` field (runtime), opt-in `Literal` lift |
| Function type when passed | `Dispatcher(instance)` (identity) | structural | `FnType(code, env_types)` (structural value) |
| Specialization key | `(instance, arg_types)` | `MethodInstance(sig)` | `(FnType, arg_types, generation)` |
| Capture value varies, types fixed | recompiles | reuses | reuses |
| Invalidation | new instance | world age (dependency graph) | generation counter (global; graph = future work) |
| Compile job | `CompileResult` | `CompilerJob(fn, world, params)` | `(FnType, arg_types, generation)` |

GPUCompiler is the closest prior art: its codegen cache keys on a `CompilerJob`
bundling the function type, the signature, and compilation params / world age —
the structural key advocated here.

## Open decisions

1. **Snapshot vs lazy cells** — default snapshot-at-capture (predictable; reference
   semantics for mutable contents); lazy as opt-in (must re-key every call).
2. **Global handling** — freeze + global-env tag + generation. Define builtin
   treatment and reject (or deopt) dynamic global access.
3. **Monomorphize fully, or admit abstract function types?** — decides whether you
   need PIC call sites at all.
4. **Invalidation granularity** — accept the global generation sledgehammer, or
   build the dependency graph that real world-age precision requires.
5. **Value-dependent lifts** — which captures get `Literal`/`Val` treatment, and
   the syntax for requesting it.
6. **L2 + layout-memo eviction** — LRU bound, and whether to persist native code to
   disk under the structural + dependency-closure + toolchain key above.

## Review pass: what the critique changed

A compiler-engineer review pass corrected these load-bearing claims from the first
draft (all re-verified):

| Was claimed | Correction |
|-------------|-----------|
| Code object keys by **identity**; "new object → cache invalidation" | Keys by **value**; unchanged re-run *hits*, edits miss via field inequality; value-equal code objects with different globals collide (freeze-globals bug) |
| Two necessary caches (L1 type / L2 native) | One native cache; the env-layout memo is a minor optimization, not a tier |
| `typeof(int) → int64` | Must range-bucket (int64/uint64/bigint-or-error) or large ints silently corrupt |
| Fingerprint = tuple of `type(v)` tags | Must be structural (tuple arity, int range, array layout/byteorder) — and array fingerprints aren't free |
| "Value-snapshot at phase A" | Reference-snapshot for mutable captures; rebind/mutate/drift are three cases |
| "Register FnType before compiling body" (recursion) | Two mechanisms: L2 forward-declaration *and* self-referential `FnType` interning; empty self-cell guard |
| Generation counter ≈ world age | Global counter is a sledgehammer; precise invalidation needs a dependency graph |
| Mono/poly call sites coexist | Full monomorphization ⇒ no PIC; PIC only with abstract function types — pick one |
| `co_names ∩ __globals__` = freeze set | Over/under-approximate (attrs, builtins, dynamic access) — define precisely |
| Disk key `(file, line, source_hash, types)` | Add dependency-closure hash + toolchain/target; lower nested `FnType`s structurally |

Still open (flagged, not resolved): the dependency-graph invalidator, the full
`typeof` lattice for containers/None/callables, and a concrete `Literal` lift API.
