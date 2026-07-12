# Overview & Timing

This page explains how the pieces fit together and — the part that matters most — **what
information is known at which moment**, and therefore *when* each thing happens. The whole
design follows from separating two arrival points of information.

## The two arrival points

The data needed to emit and run code arrives at two distinct times:

| Phase | When | What you learn | What it produces |
|---|---|---|---|
| **A — capture** | at `@jit` decoration (and every closure rebuild) | the template (code object) + captured *values* → their *types* | a `FnType` + a `Handle` (env values). **No compilation.** |
| **B — compile/run** | at `drawer.update(...)` | the argument *types*, then everything | the WGSL + pipeline (on a miss); a uniform write (always) |

Everything downstream is a consequence: because *types* are all known by phase B and the
cache keys on types, changing a *value* between frames never re-enters compilation.

## A frame, step by step

Consider `drawer.update(shader(make_img(k)))` inside a loop.

```
decoration (once per closure construction)         ── PHASE A ──────────────
  @jit(kind="device") def weave(...)  -> make_handle: read code object,
  @jit(kind="device") def img(...)        co_freevars, closure cells; typeof
  @jit(kind="fragment") def shader(...)   each capture; build FnType + Handle.
                                          No compilation.

program construction
  shader(make_img(k))  -> Handle.__call__ -> Program(entry=shader, args=(img,))
                          arg_types = (img.fntype,)   # a structural FnType

draw                                                ── PHASE B ──────────────
  drawer.update(program):
    1. flatten(program)         # inline img+weave into shader; merge uniforms;
                                #   collect current values  (cheap, pure Python)
    2. key = (shader.fntype, arg_types, current_generation())
    3. cache.get_or_compile(key, build):
         miss -> build: emit WGSL + layout; create_shader_module;
                        create_render_pipeline + uniform buffer + bind group   (~1-10 ms)
         hit  -> reuse the cached pipeline                                       (~0 ms)
    4. layout.pack(flat.values); queue.write_buffer(...)                        (~0.1 ms)
  drawer.show():
    encode render pass over a full-screen triangle; submit
```

The first iteration pays the compile; every later iteration with the same capture *types*
takes the hit path + a uniform write. That is the entire performance story.

!!! note "Implementation detail: per-frame `flatten`"
    Today `flatten()` runs **every** frame to collect the *current* captured values
    (it re-lowers the small function ASTs). Only the expensive GPU build is cached. The
    merged uniform *structure* (names/types/order) is deterministic from the types, so it
    matches the cached layout across frames. A planned optimization caches "extraction
    paths" so values can be re-collected without re-lowering. See
    [Caching](caching.md).

## The layered architecture

```
  ┌─────────────────────────── core (backend-independent) ───────────────────────────┐
  │  jit.py        @jit, make_handle, Handle, Program           (phase A capture)       │
  │  types.py      Type lattice, typeof, typeof_tuple           (value -> type)         │
  │  cache.py      SpecCache, generation                        (type -> artifact)      │
  │  ir.py         typed IR nodes                                                        │
  │  frontend/ast_lower.py   source AST -> IR                   (the "syntax" seam)      │
  │  passes/infer.py         type every IR node                                          │
  │  passes/inline.py        flatten: inline device fns, merge uniforms                  │
  └─────────────────────────────────────────────────────────────────────────────────┘
  ┌──────────────────── WGSL backend ────────────────────┐ ┌──────── runtime ─────────┐
  │  backends/wgsl/intrinsics.py  dialect tables           │ │  webgpu/runtime.py        │
  │  backends/wgsl/layout.py      env types -> uniform     │ │   Context (device)        │
  │  backends/wgsl/emit.py        IR -> WGSL text          │ │   Drawer (pipeline +      │
  │  backends/wgsl/compile.py     flatten -> WgslModule    │ │     uniform buffer)       │
  └───────────────────────────────────────────────────────┘ └───────────────────────────┘
```

The boundary is deliberate: the core knows nothing about WGSL or `wgpu`. The
[Designing a DSL](../guide/designing-a-dsl.md) guide swaps the bottom two boxes for a
different target while reusing the top.

!!! warning "Known coupling"
    Two core modules (`ast_lower`, `infer`) currently import the WGSL *dialect tables*
    (`INTRINSIC_NAMES`, `BUILTIN_CALLS`, `INTRINSIC_WGSL`) from `backends/wgsl/`. That is
    a temporary upward dependency; the plan is to pass a backend-supplied dialect object
    into the frontend instead. Worth flagging in review.

## Two data flows

Compilation and execution are two different flows that meet at the uniform buffer.

**Structure (cached, recomputed only on a type/edit/generation change):**

```
source --ast_lower--> IR --inline--> flat IR --infer--> typed IR --emit--> WGSL --+--> pipeline
                                          \--build_layout--> uniform layout --------+
```

**Values (every frame):**

```
current closures --flatten--> merged uniform values --layout.pack--> bytes --write_buffer--> GPU
```

The structure flow determines *what* the shader is; the value flow feeds it *parameters*.
The cache is exactly the line between them.

## Why this shape

- **Type-keyed, value-agnostic** caching is what numba does *not* do (it freezes captures
  as constants and keys on dispatcher identity, so every new closure recompiles — see
  [`docs/design/022_closure_specialization.md`](https://github.com/habemus-papadum/pdum_dsl/blob/main/docs/design/022_closure_specialization.md)).
  Keying on types is the Julia model and the reason a render loop is free.
- **Full monomorphization** (every call site has one statically-known target) means device
  calls inline with no dynamic dispatch — the functional composition collapses to a single
  kernel, matching the hand-written version. The cost is that genuinely polymorphic call
  sites are not supported; that is an accepted trade (see
  [WGSL Backend](wgsl-backend.md)).
- **Capture = (identity, typed env, env values)** is the decomposition that makes the env
  layout double as the uniform-buffer layout. See [Caching](caching.md).

Continue to [Caching & Generations](caching.md) for the cache model, or
[Type System](type-system.md) for how values become types.
