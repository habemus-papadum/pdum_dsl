# WGSL Backend & Runtime

This page is the deep dive into the reference backend: how a Python function becomes WGSL
text and a `wgpu` pipeline, and how the runtime drives it. It covers
[`frontend/ast_lower.py`](../reference/frontend.md),
[`passes/inline.py`](../reference/frontend.md),
[`backends/wgsl/`](../reference/wgsl.md), and
[`webgpu/runtime.py`](../reference/runtime.md).

## 1. Lowering: source AST → IR

`Lowerer` ([`ast_lower.py`](../reference/frontend.md)) parses the function's source
(`inspect.getsource`, dedented) and walks the `FunctionDef`. It needs the **code object**
to classify names:

- a name in `co_freevars` → `Name(scope="uniform")` (a captured value)
- a name in the first `co_argcount` of `co_varnames` → `Name(scope="arg")` (a parameter)
- anything else assigned → `Name(scope="local")`

The accepted subset is deliberately small (it *is* the language definition):

| Python | IR |
|---|---|
| `a = expr` | `Let(a, expr)` |
| `x, y = vec` | a temp `Let` + one `Let` per component via `Swizzle` (`x = t.x`, `y = t.y`) |
| `return expr` | `Return(expr)` |
| `a + b`, `-a`, `a ** b`, `a % b` | `BinOp` / `Unary` |
| `a < b`, … (single, unchained) | `Compare` |
| `a if c else b` | `Select` |
| `(r, g, b)` | `MakeVec` |
| `v.xy` | `Swizzle` |
| `builtins.FragCoord` | `Intrinsic("frag_coord")` (recognized structurally) |
| `f(x, y)` | `Call("f", [...])` — builtin **or** device fn |

A `Call` to any name lowers to `ir.Call`; whether it is a builtin (`sqrt`) or a device
function is decided later, by the inliner. Unsupported constructs (statements `if`/`for`,
chained comparisons, keyword args, indexing) raise `LoweringError`.

## 2. Inlining: `flatten`

`flatten(program)` ([`inline.py`](../reference/frontend.md)) produces one flat
`Function` plus the merged uniform set. Because the design is **fully monomorphizing**,
every call site resolves to exactly one target, so inlining is pure substitution:

- The entry's captured **device handles** (free vars that are `Handle`s) and its
  **higher-order arguments** (`shader(img)` → param `f` bound to `img`) populate a
  *resolver* (name → `Handle`).
- A `Call` whose name is in the resolver is **inlined**: the device function's body
  (required to be a single `return`) is substituted with its parameters bound to the
  caller's argument nodes; nested device calls recurse.
- Each function's scalar captures are registered as merged uniforms with a **deterministic
  prefix** (`""` for the entry, `"{name}{counter}_"` for an inlined device fn), so names
  like `weave2_k` are stable across frames.

The result is `Flattened(fn, names, types, values)`: `fn` is the inlined IR, `names`/
`types` are the merged uniform *structure*, and `values` are the *current* captured values.

!!! note "Single-return constraint"
    Device functions must be a single `return` expression today. Multi-statement device
    functions require hoisting their `let`s into the caller before the call site — planned.

## 3. Inference

`infer_function(flat.fn, layout.uniform_types())` types every node (see
[Type System](type-system.md)). It runs against the **narrowed** uniform types (so a
uniform `Name` is `i32`/`f32`, matching what the buffer stores), and literals infer as
`i32`/`f32`. There are no `arg`-scope names left after inlining (device params were
substituted), so `arg_types` is empty for the WGSL path.

## 4. Emission: IR → WGSL text

`emit_module(fn, layout)` ([`emit.py`](../reference/wgsl.md)) is a pure `IR -> str`
function. It assembles four parts:

1. the **uniform struct** (`layout.struct_wgsl()`),
2. the binding `@group(0) @binding(0) var<uniform> u: Uniforms;`,
3. a fixed **full-screen-triangle vertex shader** (`vs_main`) — three clip-space verts
   cover the framebuffer, the ShaderToy trick,
4. the **fragment entry** `fs_main(@builtin(position) fragcoord: vec4<f32>) ->
   @location(0) vec4<f32>`.

Key emission rules:

- **Names**: uniform → `u.{name}`, local/arg → `{name}`; the `frag_coord` intrinsic →
  `fragcoord`.
- **Numeric coercion**: WGSL has no implicit int→float conversion, so an integer operand in
  a float context is wrapped in `f32(...)` (`_emit_f` / `_is_floatish`). `pow` always
  coerces its args to `f32`.
- **`Select`**: emitted as WGSL `select(false_value, true_value, condition)` — note the
  argument order.
- **Color coercion** (`_emit_color`): a returned `vec3` → `vec4(rgb, 1.0)`; `vec4` →
  as-is; `vec2` → `vec4(xy, 0, 1)`; a scalar → grayscale `vec4(vec3(s), 1.0)`.

The result is a single inlined `fs_main` — for `shader(img)` with `img` calling `weave`:

```wgsl
struct Uniforms { weave2_k: i32, };
@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> { ... }

@fragment
fn fs_main(@builtin(position) fragcoord: vec4<f32>) -> @location(0) vec4<f32> {
  let _t1 = fragcoord.xy;
  let i = _t1.x;
  let j = _t1.y;
  let v = (((i + j) + f32(u.weave2_k)) + f32(10));
  return vec4<f32>(vec3<f32>((v / 800.0), 0.3, 0.6), 1.0);
}
```

## 5. Uniform layout: WGSL alignment

`build_layout(names, types)` ([`layout.py`](../reference/wgsl.md)) assigns byte offsets
following the WGSL uniform address-space rules — the classic footgun is that `vec3` has
**size 12 but alignment 16**:

| Type | Size | Align |
|---|---|---|
| `f32` / `i32` / `u32` | 4 | 4 |
| `vec2<T>` | 8 | 8 |
| `vec3<T>` | 12 | **16** |
| `vec4<T>` | 16 | 16 |
| `array<T, N>` | N · stride | align(T) |
| struct | — | max(member align); total rounded up to 16 for `var<uniform>` |

Each field's offset is rounded up to its alignment; the struct's total size is rounded up
to a multiple of 16. `Layout.pack(values)` writes each value at its offset (`f32`→`<f`,
`i32`→`<i`, `u32`→`<I`); an empty struct emits a `_pad: vec4<f32>` member because WGSL
forbids empty structs. This is unit-tested against the spec example in
[`tests/test_m02_wgsl.py`](https://github.com/habemus-papadum/pdum_dsl/blob/main/tests/test_m02_wgsl.py).

## 6. The runtime

[`webgpu/runtime.py`](../reference/runtime.md) wires the cache to `wgpu`.

- **`Context`** owns the adapter + device (`gpu.request_adapter_sync().request_device_sync()`).
- **Targets** abstract where pixels go: `OffscreenTarget` (an owned texture +
  `read_pixels()` with the 256-byte row-stride dance) and `WindowTarget` (a `rendercanvas`
  present surface). Both expose `get_view()` and `size`.
- **`GpuProgram`** is the cache's **artifact type** for WebGPU: a compiled `pipeline`,
  `bind_group`, `uniform_buffer`, and the `layout`.
- **`Drawer.update(program)`** is phase B: `flatten` → `cache.get_or_compile(entry.fntype,
  arg_types, _build)` → `layout.pack(flat.values)` → `queue.write_buffer`. `_build` is the
  expensive, cached step (`create_shader_module`, `create_render_pipeline`, allocate the
  uniform buffer + bind group).
- **`Drawer.show()`** encodes a render pass over the full-screen triangle (`draw(3, 1)`)
  and submits.
- **`Context.run(canvas, frame_fn)`** wires a per-frame callback into the `rendercanvas`
  glfw event loop.

### Cost model

Recreating a pipeline is ~1–10 ms; a uniform `write_buffer` is ~0.1 ms. The cache keeps
`_build` to once per type signature, so a render loop is dominated by the buffer write —
this is what makes parameter animation free. `wgpu` does not persist shader/pipeline
compilation between processes, which is why an on-disk cache is on the roadmap.

## Limitations & expansion points

| Area | Today | Planned |
|---|---|---|
| Stages | fragment only (+ inlined device fns) | vertex + varyings (mesh data), compute |
| Inputs | scalar uniforms via captures; `FragCoord` | vec/struct uniforms, `resolution`/`time`/`mouse`, vertex attributes |
| Resources | one uniform bind group | textures/samplers, storage buffers, multiple bind groups |
| Control flow | conditional *expression* only | `if`/`for` statements, multi-statement device fns |
| Data | scalars | lists/arrays + indexing, `len()`, user structs |
| Cache key | `(FnType, arg_types, generation)` | + target format, + toolchain (disk cache) |
| Dialect tables | imported by core from this backend | injected as a backend-supplied dialect |

The emitter and layout already understand `vec`/`struct`/`array` types, so several of these
(notably vector uniforms) are gated only on the `typeof`/capture path, not on the backend.
