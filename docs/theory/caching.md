# Caching & Generations

The cache is the whole point of the framework: **one compiled artifact per type
signature, reused across all capture values.** This page is the model behind
[`cache.py`](../reference/core.md) and [`jit.py`](../reference/core.md). The original,
longer design note is
[`design/dsl_caching_layer.md`](https://github.com/habemus-papadum/pdum_dsl/blob/main/design/dsl_caching_layer.md);
this page describes what is actually implemented and where it diverges.

## The reframing

Numba conflates a captured value *into* the type — it freezes `x` as a compile-time
constant, so `closure(5)` and `closure(6)` produce different code. We split the three
concerns apart:

| Concern | Where it lives |
|---|---|
| **identity** of the function | the **code object** of the `def` site (compared by value) |
| **types** of the captured variables | `env_types` — part of the `FnType` |
| **values** of the captured variables | the `Handle.env` — runtime data, never in a key |

The first two together are the **function type** (`FnType`); the third is runtime data.
Specialization keys on the function type + argument types, **never** on captured values.
For WebGPU, that third concern is exactly the uniform-buffer contents.

## `FnType`: identity by value

```python
FnType = (template_code_object, env_types)   # a frozen, hashable value
```

The template is the function's **code object**, and CPython code objects compare *by
value* (over `co_code`, `co_consts`, `co_firstlineno`, …), not by identity. That single
fact gives the right invalidation behavior for free:

| Situation | Result | Why |
|---|---|---|
| Repeated closure construction in one process | **hit** | same code object |
| Re-run the **unchanged** source (loop, re-run a notebook cell) | **hit** | the recompiled code object is value-*equal* |
| **Edit** the body and re-run | **miss** → recompile | `co_code`/`co_consts` differ → value-*unequal* |
| A capture's **type** changes | **miss** | different `env_types` → different `FnType` |
| A capture's **value** changes (same type) | **hit** | values aren't in the key |

A capture that is itself a `Handle` contributes its own `FnType` to `env_types` — nested
closures compose structurally (this is what makes `img` capturing `weave` produce a
stable type across frames).

!!! note "Globals are not frozen yet"
    The design note warns that if you *freeze globals* as constants, two value-equal code
    objects with different global namespaces must not share a specialization (the key must
    then include a global-env tag). We do **not** freeze globals today, so this hazard is
    latent; revisit it before adding global constant-folding.

## The key and the dispatch

```python
key = (fntype, arg_types, current_generation())
artifact = cache.get_or_compile(fntype, arg_types, compile_fn)
```

- `arg_types` are the types of a `Program`'s arguments (for `shader(img)`, that is
  `(img.fntype,)`). For a no-argument fragment shader it is `()`.
- `current_generation()` is a global counter (below).
- On a **miss**, `compile_fn()` runs once and its result is stored. On a **hit**, the
  stored artifact is returned. `compile_count` and `hit_count` are exposed so a render
  loop can assert "compiled once" (see the [WebGPU guide](../guide/webgpu-shaders.md)).

### `typeof` is on the hot path *and* defines correctness

Every capture (phase A) and every argument (phase B) goes through `typeof`. A too-coarse
`typeof` silently reuses the wrong specialization, so it is careful where it counts:

- **`bool` before `int`** (`bool` is a subclass of `int`).
- **int range-bucketing**: `i64` if it fits signed 64-bit, else `u64`, else an error —
  a captured `5` and `2**70` must not share a type or the uniform value would be
  corrupted. (See [Type System](type-system.md).)

!!! note "Planned: structural fingerprints"
    The design note proposes a cheap *structural fingerprint* fast-path (tuple arity, int
    bucket, array flags) to avoid a full `typeof` per call on the hottest paths. Not
    implemented; today the `Type` value itself is the key.

### Thread safety

`get_or_compile` uses a per-key future: the first caller to miss a key creates an entry
and compiles; concurrent callers that find an in-progress entry **wait** on its event
rather than compiling a second time. A failed compile drops the slot so a later call can
retry. (Reentrant/recursive compilation — forward-declaring an in-progress specialization
so a recursive call can bind to it — is described in the design note but not needed yet,
since shaders are not self-recursive.)

## Generations: live coding

`generation` is a single global counter folded into every key:

```python
from pdum.dsl import bump_generation, current_generation
bump_generation()    # invalidate ALL specializations
```

- It lives **in the key, not in the `Handle`** — so a closure built before a bump adopts
  the new world on its next draw.
- It is a **sledgehammer**: a bump invalidates everything, not just what changed.
- Most live-coding invalidation does not need it: editing a body changes the code object
  → a new `FnType` → a natural miss. `bump_generation()` is for invalidating when
  something *outside* the tracked types changes (e.g. a frozen global, once that exists).

!!! note "Planned: precise invalidation (world age)"
    Julia's world age is precise because it tracks a *dependency graph* — which
    specializations called a redefined method — and invalidates only those. We have the
    global counter; the dependency graph is future work. Until then, prefer relying on
    natural code-object invalidation and reserve `bump_generation()` for coarse resets.

## Value extraction each frame

The cached artifact captures *structure* (WGSL + uniform layout). The *values* must be
re-collected from the current (rebuilt-every-frame) closures:

- **Today:** `Drawer.update` calls `flatten(program)` every frame. `flatten` re-lowers the
  small function ASTs, inlines, and collects the merged uniform `values`. Because the
  merged names/types/order are deterministic from the types, they line up with the cached
  layout, and only `layout.pack(values)` + `write_buffer` run after the cache hit. The
  re-lowering is cheap but not free.
- **Planned:** compute and cache the *extraction paths* (how to walk the live `Handle`
  tree to each merged uniform) at compile time, so per-frame value collection needs no
  re-lowering at all.

## Limits and planned work

| Concern | Status |
|---|---|
| One artifact per `(FnType, arg_types, generation)` | **implemented** |
| Per-key future (no double compile) | **implemented** |
| Code-object value-equality invalidation | **implemented** (in-process) |
| Global generation counter | **implemented** |
| Structural fingerprint fast-path | planned |
| Cached extraction paths (skip per-frame re-lowering) | planned |
| LRU eviction (specialization explosion is bounded by *types*, not values, but unbounded across types) | planned |
| On-disk, cross-process cache (structural key + dependency-closure hash + toolchain tag) | planned |
| Dependency-graph world age | planned |
| Backend params in the key (e.g. WGSL target format) | not yet (one `Drawer` per format) |

See [`design/dsl_caching_layer.md`](https://github.com/habemus-papadum/pdum_dsl/blob/main/design/dsl_caching_layer.md)
for the full hazard analysis (capture rebinding vs mutation, recursion, disk-key
construction) that informs the planned items.
