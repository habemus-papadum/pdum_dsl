# pdum.dsl

!!! warning "These docs describe the frozen Milestone-0 reference asset"
    The implementation documented here is preserved intact at
    `pdum.dsl_reference` (code in `src/pdum/dsl_reference/`, tests in
    `reference/tests/`) while `pdum.dsl` itself is redesigned — see
    [`docs/desiderata.md`](../desiderata.md) for the redesign brief and
    `reference/README.md` for how to run the frozen asset.

A Python **DSL compiler framework**: a [numba](https://numba.pydata.org/)-like
`@jit` decorator workflow with **Julia-like type-keyed caching**. Its first concrete
use case compiles Python functions into **WebGPU/WGSL fragment shaders** where
captured Python closure values become **uniforms** — so you can drive a shader from
a tight Python render loop without recompiling on every frame.

The framework is two cleanly separated halves:

- a **use-case-independent core** — closure capture, a structural type system, a
  type-keyed specialization cache, a typed IR, and inlining; and
- **pluggable backends/runtimes** — the first being WGSL emission + a `wgpu` runtime.

---

## The one idea

A closure is **(code identity, typed environment, environment values)**. Compilation
is keyed on the *types*, never on the *values*. When you map that onto WebGPU, every
piece lands on something real:

| Closure concept | WebGPU realization |
|---|---|
| typed env layout (from `co_freevars` × env types) | the **uniform-buffer struct layout** |
| env *values* (runtime data, never in the cache key) | the **uniform-buffer contents** |
| compiled artifact keyed on `(FnType, arg_types, generation)` | the **WGSL module + render pipeline** |
| compile once per type, reuse across values | build the pipeline once, `write_buffer` per frame |
| full monomorphization → inline | device functions collapse into one kernel |

The payoff: recreating a pipeline costs ~1–10 ms; rewriting a uniform buffer costs
~0.1 ms. Because the cache keys on types, moving a parameter around a render loop is a
buffer write, not a recompile. See [Theory → Overview](theory/overview.md).

## Quickstart

```python
import math
from pdum.dsl_reference import builtins, jit
from pdum.dsl_reference.webgpu import Context

def disk(cx, cy, radius):
    @jit(kind="fragment")
    def shader():
        x, y = builtins.FragCoord.xy           # pixel coordinates
        dx = x - cx                            # cx, cy, radius are captured -> uniforms
        dy = y - cy
        d2 = dx * dx + dy * dy
        return (1.0, 0.5, 0.0) if d2 < radius * radius else (0.05, 0.05, 0.12)
    return shader

ctx = Context()
canvas, drawer = ctx.window_drawer(size=(640, 480), title="orbiting disk")
t = [0.0]

def frame():
    t[0] += 0.03
    w, h = drawer.target.size
    drawer.update(disk(w/2 + 180*math.cos(t[0]), h/2 + 120*math.sin(t[0]), 70.0))
    drawer.show()
    canvas.request_draw(frame)

ctx.run(canvas, frame)
# The shader compiles exactly once; each frame only rewrites the uniform buffer.
```

A runnable version is in [`reference/demos/disk.py`](https://github.com/habemus-papadum/pdum_dsl/blob/main/reference/demos/disk.py):

```bash
uv run python reference/demos/disk.py            # interactive window
uv run python reference/demos/disk.py --frames 120   # render N frames, print compile count, exit
```

## Where to go next

- **[WebGPU Shaders guide](guide/webgpu-shaders.md)** — the recommended starting point:
  how to think about and write shaders with this framework.
- **[Designing a DSL guide](guide/designing-a-dsl.md)** — use the reusable core to build
  a *different* DSL or backend.
- **[Theory & Internals](theory/overview.md)** — how it works, what is known when, and why.
- **[API Reference](reference/core.md)** — every public symbol.

## Status

This is an early, evolving project. **Milestone 0 (the end-to-end vertical slice) is
implemented and tested**; much of the expressive surface is still planned. Each guide
marks features as implemented or planned inline, but here is the headline:

!!! success "Implemented (Milestone 0)"
    - `@jit(kind="fragment")` and `@jit(kind="device")`; phase-A capture → `Handle`
    - Structural type system + `typeof` (scalars with int range-bucketing, tuples, `FnType`)
    - Type-keyed `SpecCache` + a global `generation` counter
    - AST → typed IR → WGSL for an explicit Python subset (arithmetic, comparisons,
      ternary, swizzles, `builtins.FragCoord`, builtin math calls, tuple→vector returns)
    - Monomorphic inlining of device functions, including higher-order `shader(img)`
    - `wgpu` runtime: `Context`, `Drawer` (offscreen + glfw window), uniform layout with
      correct WGSL alignment
    - **Scalar** captured uniforms (`int → i32`, `float → f32`, `bool → u32`)

!!! note "Planned (not yet implemented)"
    - Vector/tuple/struct captured uniforms; user-defined structs (`@gpu_struct`)
    - Lists/indexing + `len()` (the "color table"); the `🌺` pipe/compose operator
    - `Literal`/`Val` value-dependent specialization (bake a capture as a `const`)
    - `vertex`/`compute` stages, varyings/mesh data, textures, storage buffers
    - Reserved `resolution`/`time`/`mouse` uniforms (sentinels exist; not wired)
    - Dependency-graph invalidation ("world age"), cache eviction, on-disk cache
    - Additional backends (LLVM/CPU/CUDA) + numba type interop
    - `if`/`for` statements and multi-statement device functions in shader bodies

See [`design/`](https://github.com/habemus-papadum/pdum_dsl/tree/main/design) for the
original design notes that this implementation is based on.
