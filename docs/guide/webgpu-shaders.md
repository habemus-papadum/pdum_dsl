# WebGPU Shaders

This guide is for writing **GPU fragment shaders in Python** with `pdum.dsl`. You
write ordinary-looking Python functions that compute a color per pixel; the framework
compiles them to WGSL, turns the Python values they close over into GPU uniforms, and
runs them — recompiling only when the *shape* (types) of your shader changes, not when
its parameters change.

!!! info "How to read this guide"
    Features that work today are shown in normal prose and code. Anything not yet
    implemented is called out in a **Planned** admonition so you can tell the present
    shape of the language from where it is going.

## Mental model

Think of a fragment shader as a pure function **`(pixel) → color`** that runs once per
pixel, massively in parallel, on the GPU. In `pdum.dsl`:

- The **function body** is the per-pixel computation. It is *lowered from its source*
  to WGSL — it never actually runs as Python.
- The **values it closes over** (free variables) become **uniforms** — a small block of
  read-only data uploaded to the GPU once per frame. These are your "knobs".
- Rebuilding the closure with new knob values each frame is cheap: the compiled shader
  is reused and only the uniform block is rewritten.

This is the same model as a ShaderToy shader plus uniforms, but the uniforms are just
the Python variables your function captured — no manual buffer wiring.

## The two decorators

```python
from pdum.dsl_reference import builtins, jit

@jit(kind="fragment")     # an entry point: (pixel) -> color
def shader():
    ...

@jit(kind="device")       # a helper callable from shaders/other helpers
def helper(x, y):
    ...
```

`@jit` does **not** compile anything. At decoration time it performs *phase-A capture*:
it reads the function's code object and the values in its closure cells, computes the
**types** of those captures, and returns a [`Handle`](../reference/core.md). Compilation
happens later, the first time you draw (see [Theory → Overview](../theory/overview.md)).

!!! note "Planned: other stages"
    `kind="vertex"` and `kind="compute"` are accepted by the decorator but not yet
    emitted by the backend. Vertex shaders + varyings (for mesh data) are on the roadmap.

## Coordinates: `builtins.FragCoord`

A fragment shader gets the pixel's framebuffer coordinate from `builtins.FragCoord`:

```python
from pdum.dsl_reference import builtins

@jit(kind="fragment")
def shader():
    x, y = builtins.FragCoord.xy   # x, y in physical pixels; origin at top-left
    ...
```

`builtins.FragCoord` is `vec4` (`@builtin(position)` in WGSL); `.xy`, `.x`, `.y` are
swizzles. Coordinates are in **physical pixels** (so on a HiDPI display a 640×480 window
has a larger framebuffer — read `drawer.target.size` for the actual size).

!!! note "Planned: more intrinsics"
    `builtins.resolution`, `builtins.time`, and `builtins.mouse` exist as sentinels but
    are not yet wired into emission. For now, pass resolution/time in as captured values
    (they become uniforms automatically).

## Captures become uniforms

Any free variable your shader (or a device function it inlines) closes over becomes a
uniform. The cache keys on the *types* of those captures, so:

```python
def disk(cx, cy, radius):          # cx, cy, radius are captured
    @jit(kind="fragment")
    def shader():
        x, y = builtins.FragCoord.xy
        d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
        return (1.0, 0.5, 0.0) if d2 < radius * radius else (0.0, 0.0, 0.0)
    return shader
```

- `disk(100.0, 120.0, 50.0)` and `disk(300.0, 80.0, 50.0)` produce the **same shader**
  (same capture types) → compiled once, drawn with different uniform contents.
- `disk(100.0, 120.0, 50)` (an `int` radius) is a **different** shader (a different type)
  → its own compilation.

!!! warning "Today: scalar uniforms only"
    Captured uniforms must currently be **scalars**: `int → i32`, `float → f32`,
    `bool → u32`. Capture vector components as separate scalars (as `cx, cy` above)
    rather than `center=(cx, cy)`.

    A captured Python *tuple* is typed as a `TupleType`, which the uniform layout does
    not yet accept — so vector/tuple/struct uniforms are **planned**, not implemented.
    (The WGSL layout and emitter already understand `vec`/`struct` types; the missing
    piece is mapping a captured value to them — see
    [Type System](../theory/type-system.md).)

### What triggers a recompile?

| Change | Effect |
|---|---|
| A capture **value** changes (same type) | **No recompile** — just a uniform write |
| A capture **type** changes (`float`→`int`, etc.) | New `FnType` → its own compilation |
| You **edit** the function body and re-run | New code object → cache miss → recompile |
| You re-run the **unchanged** body (loop iteration, re-run a notebook cell) | Cache **hit** |
| `bump_generation()` is called | All specializations invalidated |

This is the whole point — see [Caching & Generations](../theory/caching.md).

## The Python you can write (today)

Shader and device bodies accept a small, explicit subset of Python. This is deliberate:
the subset *is* the language definition (see [WGSL Backend](../theory/wgsl-backend.md)).

**Supported:**

- assignment and tuple-unpacking: `a = expr`, `x, y = vec_expr`
- arithmetic: `+`, `-`, `*`, `/`, `**` (→ `pow`), `%`, unary `-`
- comparisons: `<`, `>`, `<=`, `>=`, `==`, `!=` (a single comparison, not chained)
- the conditional expression: `a if cond else b` (→ WGSL `select`)
- tuple literals as vectors: `return (r, g, b)` → `vec3`, `(r, g, b, a)` → `vec4`
- attribute swizzles: `v.x`, `v.xy`, `v.xyz`, ...
- the `builtins.FragCoord` intrinsic
- builtin math calls: `sqrt`, `abs`, `floor`, `fract`, `sin`, `cos`, `min`, `max`,
  `length`, `mix`, `clamp` (see `BUILTIN_CALLS` in
  [`backends/wgsl/intrinsics.py`](../reference/wgsl.md))
- calls to **device functions** (resolved and inlined)

**Return value:** a fragment returns a color. A `(r, g, b)` tuple is padded to
`vec4(rgb, 1.0)`; a 4-tuple is used as-is; a scalar becomes grayscale.

!!! note "Planned"
    `if`/`for` statements, `while`, local mutation, lists/dicts and indexing, `len()`,
    string handling, multi-statement device functions, and user-defined struct field
    access are **not** supported yet. Today, branch with the conditional expression and
    keep device functions to a single `return`.

### Integer/float mixing

The emitter inserts `f32(...)` conversions where an integer meets a float (WGSL has no
implicit numeric conversion), so `x - k` with `x: f32` and a captured `int k` works.
This coercion is intentionally minimal today (integer → `f32` in a float context only);
richer numeric typing is planned.

## Composition: device functions and higher-order shaders

Device functions are helpers; a shader can take other functions as arguments. Both are
**inlined** — the final WGSL is a single function with no call overhead.

```python
@jit(kind="fragment")
def shader(f):                      # higher-order: takes a device function
    i, j = builtins.FragCoord.xy
    v = f(i, j)
    return (v / 800.0, 0.3, 0.6)

def make_img(k):
    @jit(kind="device")
    def weave(x, y):
        return x + y + k            # k captured -> uniform
    @jit(kind="device")
    def img(x, y):
        return weave(x, y) + 10     # device fn calling a device fn
    return img

drawer.update(shader(make_img(7)))  # shader applied to img
```

`shader(make_img(7))` does not run anything — calling a `Handle` builds a deferred
[`Program`](../reference/core.md) (the partial application). At draw time, `img` and
`weave` are inlined into `shader`, and `k` flows in as the single uniform. The emitted
fragment is literally:

```wgsl
@fragment
fn fs_main(@builtin(position) fragcoord: vec4<f32>) -> @location(0) vec4<f32> {
  let _t1 = fragcoord.xy;
  let i = _t1.x;
  let j = _t1.y;
  let v = (((i + j) + f32(u.weave2_k)) + f32(10));
  return vec4<f32>(vec3<f32>((v / 800.0), 0.3, 0.6), 1.0);
}
```

!!! note "Constraint (today)"
    A device function must be a **single `return` expression** so inlining is pure
    substitution. Multi-statement device functions (with `let`-hoisting) are planned.

!!! note "Planned: the functional pipeline style"
    The motivating example (see [`design/Generative Julia.ipynb`](https://github.com/habemus-papadum/pdum_dsl/blob/main/design))
    builds shaders by *composing* higher-order functions —
    ```python
    shader = twill | widen(2) | weave(palette) | zoom(center=(0.75, 0.4))
    ```
    — via a "flower" pipe operator (`🌺`). The higher-order machinery that makes this
    inline correctly is implemented; the `pipe`/compose *sugar* (a `func.py`) and the
    helpers (`weave`, `zoom`, color tables) are **not yet written**. See
    [Designing a DSL](designing-a-dsl.md) for how composition lowers.

## The runtime

A [`Context`](../reference/runtime.md) owns the GPU device. A [`Drawer`](../reference/runtime.md)
owns one shader's compiled pipeline + uniform buffer and updates uniforms each frame.

```python
from pdum.dsl_reference.webgpu import Context

ctx = Context()                                   # picks an adapter + device

# Option A — a live window (glfw):
canvas, drawer = ctx.window_drawer(size=(640, 480), title="my shader")

# Option B — offscreen (headless; for tests / image output):
drawer = ctx.offscreen_drawer(size=(256, 256), format="rgba8unorm")
```

Drawing is two steps:

- `drawer.update(program)` — *phase B*. Resolves the specialization (compiles once on a
  miss; on a hit reuses the cached pipeline) and writes the current uniform values.
- `drawer.show()` — encodes a render pass over a full-screen triangle and submits it.

For a window, drive it with a per-frame callback:

```python
def frame():
    drawer.update(disk(cx, cy, r))   # cx, cy, r recomputed each frame
    drawer.show()
    canvas.request_draw(frame)       # re-arm

ctx.run(canvas, frame)               # enters the event loop (blocking)
```

For offscreen rendering, call `update`/`show` in a plain loop and read pixels back:

```python
drawer.update(disk(32.0, 32.0, 20.0))
drawer.show()
rgba = drawer.target.read_pixels()   # tightly-packed RGBA bytes, row 0 at top
```

!!! note "Planned: image/notebook output"
    `read_pixels()` returns raw bytes today. Saving to PNG and a live notebook
    (anywidget) canvas are planned.

### Verifying the no-recompile property

Every `Drawer` exposes counters so you can assert the thesis:

```python
for k in range(120):
    drawer.update(disk(10.0 + k, 32.0, 20.0))   # value changes, type stays float
    drawer.show()

assert drawer.compile_count == 1                # built the pipeline once
assert drawer.uniform_writes == 120             # rewrote uniforms every frame
```

The cost model behind this: recreating a pipeline is ~1–10 ms, while a uniform
`write_buffer` is ~0.1 ms. Keeping `compile_count` at 1 is what makes parameter sweeps
and animation run at full speed.

## Gotchas

- **Source must be available.** Bodies are lowered from `inspect.getsource`. Functions
  defined where source can't be recovered (some REPLs) won't lower; a bytecode fallback
  is planned. Notebook cells and script files are fine.
- **`FragCoord` is in physical pixels.** Use `drawer.target.size` for the framebuffer
  size rather than the logical window size.
- **One target format per `Drawer`.** The cache key does not include the target texture
  format today, so use one `Drawer` per output format.
- **Name the intrinsic namespace `builtins`.** The lowerer recognizes the attribute
  chain `builtins.FragCoord` structurally; importing it under a different alias won't be
  recognized yet.

## Full example

See [`docs/demos/disk.py`](https://github.com/habemus-papadum/pdum_dsl/blob/main/docs/demos/disk.py)
for a complete, runnable window demo, and the tests in
[`reference/tests/test_m03_thesis.py`](https://github.com/habemus-papadum/pdum_dsl/blob/main/reference/tests/test_m03_thesis.py)
and [`reference/tests/test_m04_inline.py`](https://github.com/habemus-papadum/pdum_dsl/blob/main/reference/tests/test_m04_inline.py)
for offscreen, asserted versions.
