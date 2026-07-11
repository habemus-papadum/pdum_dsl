# R5 — The kernel-embedding DSL family: how small Python-subset kernel compilers actually work

Research report for the pdum.dsl redesign. Investigates four systems on a spectrum
from "full Python-subset frontend" to "no frontend at all": **cupy.jit**,
**Triton**, **MLX custom kernels**, and **Taichi**. All claims verified against
live sources in July 2026.

**Versions verified current as of 2026-07-11:**

| System | Current version | Status |
|---|---|---|
| CuPy (`cupyx.jit`) | 14.1.1 stable (Jan 2026); 15.0.0a1 in dev | active; `cupyx.jit` shipped, still marked experimental-adjacent |
| Triton | 3.7.0 (May 2026); 3.6.0 (Jan 2026) | very active (PyTorch ecosystem) |
| MLX | 0.32.0 | active; **now has `mx.fast.cuda_kernel` alongside `metal_kernel`** |
| Taichi | 1.7.4 (Jul 2025 — last release) | maintenance slowed; community forks (`taichi-forge`, `gstaichi`) emerging |

Sources: [CuPy releases](https://github.com/cupy/cupy/releases),
[Triton 3.7.0 release](https://github.com/triton-lang/triton/releases/tag/v3.7.0),
[MLX fast API docs](https://ml-explore.github.io/mlx/build/html/python/fast.html),
[Taichi PyPI](https://pypi.org/project/taichi/).

---

## 1. cupy.jit (`cupyx.jit.rawkernel`)

The closest existing thing to pdum.dsl's wanted CuPy backend: a Python-subset →
CUDA C transpiler in **~1,800 lines of Python**, with per-signature compilation
driven by call-time argument types.

Primary sources:
[`cupyx/jit/_interface.py`](https://github.com/cupy/cupy/blob/main/cupyx/jit/_interface.py) (264 lines),
[`cupyx/jit/_compile.py`](https://github.com/cupy/cupy/blob/main/cupyx/jit/_compile.py) (1,033 lines),
[`cupyx/jit/_cuda_types.py`](https://github.com/cupy/cupy/blob/main/cupyx/jit/_cuda_types.py) (348 lines),
[`cupyx/jit/_cuda_typerules.py`](https://github.com/cupy/cupy/blob/main/cupyx/jit/_cuda_typerules.py) (167 lines).

### Frontend technique

- `@rawkernel(mode='cuda', device=False)` wraps the function in a `_JitRawKernel`
  via `functools.update_wrapper`. **No work at decoration time** (same as pdum's
  phase A being compile-free, though cupy.jit doesn't even snapshot captures).
- Source is recovered with `inspect.getsourcefile()` + `inspect.getsourcelines()`,
  then `ast.parse()`. Lambdas get special handling: the whole source file is
  parsed and the AST walked to find the lambda between known line numbers
  (`_parse_function_object()`), specifically to survive notebook environments
  where `getsourcelines` returns un-dedentable snippets.

### Type inference and per-signature compilation

At `__call__` time each argument is mapped to a type object:

```python
if isinstance(x, cupy.ndarray):
    t = _cuda_types.CArray.from_ndarray(x)   # (dtype, ndim, is_c_contiguous, index_32_bits)
elif numpy.isscalar(x):
    t = _cuda_typerules.get_ctype_from_scalar(self._mode, x)
```

`CArray` is the structural array type: fields `(dtype, ndim, is_c_contiguous,
index_32_bits)`. Its `__hash__` is `hash(str(self))` and `__eq__` compares the
stringified form `CArray<{ctype}, {ndim}, {c_contiguous}, {index_32_bits}>` —
i.e. the C++ template instantiation string *is* the cache key component. Note
`index_32_bits` — an optimization property of the concrete array (fits in 32-bit
indexing) leaks into the type key. This is value-dependent specialization,
undeclared: two arrays differing only in size can compile twice.

**Cache structure** (in `_JitRawKernel`):

```python
self._cache        # {(in_types, device_id): cupy.cuda.Function}
self._cached_codes # {in_types: transpile Result}  — generated CUDA source, for inspection
```

Cache miss → `_compile.transpile(func, attributes, mode, in_types, ret_type)` →
`Result(func_name, code, return_type, backend, options, ...)` → compiled through
CuPy's normal `compile_with_cache` (which adds its own on-disk nvcc/nvrtc cache
keyed on source hash + compiler options).

### Codegen: single-pass typed transpile

`_compile.py` is one abstract-interpretation pass that **infers types and emits
CUDA C text simultaneously** — there is no separate IR and no fixpoint:

- An `Environment` holds `consts` (compile-time constants), `params` (typed
  runtime `Data` objects from `in_types`), `locals`, `ret_type`, and a `generated`
  accumulator. Name resolution is locals → params → consts.
- `_transpile_stmt` handles `Return` (return type unified across paths),
  `Assign`/`AugAssign`, `For` (**`range()` only**, with `#pragma unroll` support),
  `While`, `If` (constant-folded when the test is a compile-time `Constant`),
  `Pass`/`Break`/`Continue`. Rejected: nested functions, classes, imports,
  exceptions, context managers, `*args`/`**kwargs`, defaults.
- `_transpile_expr` handles `BinOp`/`UnaryOp` via `_eval_operand()` (CuPy's
  ufunc type rules), `Call` (builtins, ufuncs, casts, other `@rawkernel
  (device=True)` functions — recursion detected by storing a `None` placeholder),
  `Subscript`, `Tuple` (lowered to `STD::make_tuple`), `IfExp`, `Name`,
  `Constant`. If all operands of an op are compile-time constants, the op runs in
  Python at transpile time (`_cuda_typerules.get_pyfunc()`).

### Captures/globals: silently frozen (the anti-pattern)

Constants come from `inspect.getclosurevars()` — globals, nonlocals, builtins are
snapshotted into `env.consts` **at transpile time, per `in_types`**. Consequence:
change a global's *value* without changing any argument *type* → the cached kernel
is silently stale. This is exactly the numba capture-freezing behavior
`design/closure_specialization.md` documents; cupy.jit has no invalidation story
at all (no generation counter, no dependency hash).

### Marshaling

- Arrays go to the device as a `CArray<T, ndim, contig, idx32>` struct — pointer
  **plus** shape and strides packed into one aggregate kernel parameter by CuPy's
  `cupy.cuda.Function` launch machinery. This is the "one logical value → multiple
  physical fields" pattern, solved by making the physical group a single C++
  struct so the CUDA ABI sees one parameter.
- Scalars are normalized before launch to the numpy scalar type matching the
  inferred CUDA type (special-casing float16 `dtype.char == 'e'`).
- Launch: `kern((grid,), (block,), (args...), shared_mem, stream)`.

---

## 2. Triton (`@triton.jit`)

The industrial-strength member of the family, and the richest prior art for cache
keys and value-based specialization.

Primary sources:
[`python/triton/runtime/jit.py`](https://github.com/triton-lang/triton/blob/main/python/triton/runtime/jit.py) (1,178 lines),
[`python/triton/compiler/compiler.py`](https://github.com/triton-lang/triton/blob/main/python/triton/compiler/compiler.py) (513 lines),
[`python/triton/compiler/code_generator.py`](https://github.com/triton-lang/triton/blob/main/python/triton/compiler/code_generator.py) (1,707 lines),
[`third_party/nvidia/backend/driver.py`](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/driver.py) (403 lines).

### Frontend

`JITFunction.__init__` takes `inspect.getsourcelines()`, strips the decorator
prefix with `src[re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE).start():]`, and
parses lazily: `parse()` → `ast.parse(self._src)`. `code_generator.py` is an
`ast.NodeVisitor` that walks the tree and **builds MLIR operations directly**
through a pybind'd builder — no text emission; the output is a `ttir` module
(Triton's MLIR dialect). Type propagation is again single-pass and forward, from
the signature; `tl.constexpr` values are Python values during the walk, so
`if CONST_FLAG:` folds naturally by executing the Python `if` over a constexpr.

### Function-level cache key: source + dependency closure

`JITFunction.cache_key` is a hash of the function's own source **plus a recursive
hash of everything it references**: `DependenciesFinder` (an AST visitor) resolves
every global name used by the kernel, keyed `(var_name, id(fn.__globals__))`, and
folds in the `cache_key` of any transitively-called `@triton.jit` function. At
launch, `run()` re-checks captured globals and **raises** if one changed:

```python
if (newVal := globals_dict.get(name, not_present)) != val:
    raise RuntimeError(...)  # global changed under a compiled kernel
```

So Triton neither silently freezes (cupy) nor recompiles (Julia): it *detects and
refuses*. This dependency-closure hash is precisely what
`design/dsl_caching_layer.md` prescribes for the disk cache.

### Per-call specialization: types + value properties

Each `KernelParam` carries flags parsed from annotations: `is_constexpr`
(annotation contains `"constexpr"`), `is_const` (`"const"` annotation or `*k`
pointer-to-const), `do_not_specialize` (by index or name),
`do_not_specialize_on_alignment`.

For every non-constexpr argument, `specialize_impl` (now a **native C++ function**
in current Triton — it was Python; moved down for launch latency) returns a tuple
`(type_string, specialization_hint)`:

- **divisibility by 16** of pointers and integers → recorded as a hint that
  becomes the `tt.divisibility = 16` attribute in the IR (enables vectorized
  loads / coalescing);
- **integer value == 1** → specialized as constant 1 (folds strides, enables
  simplifications);
- floats (`fp*`, `bf*`) and `u1` never specialize; `do_not_specialize` opts out
  per-argument.

The full specialization list — `("constexpr", value)` entries for constexpr
params, `(type, hint)` for the rest — is the per-call key:

```python
key = (tuple(specialization), str(options))
kernel = self.device_caches[device][0].get(key, None)
```

**This is default-ON value-dependent specialization with per-arg opt-out** — the
inverse of pdum's stance. The known cost: pass an oddly-offset tensor once and
you get a second compile of the "same" kernel (a perennial Triton user surprise).
For pdum this is the strongest prior art that (a) value properties *should* be
expressible in the key, (b) making them implicit trades predictability for peak
performance, and (c) the mechanism is clean either way: **hints enter the cache
key AND are handed to the compiler as IR attributes**, keeping key and codegen in
lockstep.

### Launch fast path: the generated binder

`create_function_from_signature()` `exec`s a purpose-built Python function per
`JITFunction` whose parameter list mirrors the kernel's signature (with defaults
baked in). Its body runs `specialize_impl` per arg and returns
`(bound_args, specialization, options)` with zero generic introspection per call.
This "compile the dispatcher itself" trick is the standard answer to hot-launch
overhead and directly applicable to pdum's `typeof`-fingerprint fast path.

### Full compile key and pipeline

`ASTSource` bundles `(fn, signature: dict, constexprs, attrs)`; its hash:

```python
key = f"{self.fn.cache_key}-{str(self.attrs)}-{sorted_sig}-{constants_key}"  # → sha256
```

The on-disk compilation key additionally folds in the **backend name, backend
options (`num_warps`, `num_stages`, ...), and relevant environment variables**
(`get_cache_key()`). Compilation walks a stage dict the backend populates via
`add_stages()`, mapping file extensions to transforms:

```
ttir → ttgir → llir → ptx → cubin        (NVIDIA)
ttir → ttgir → llir → amdgcn → hsaco     (AMD)
```

Each stage's artifact is dumped and inspectable (`CompiledKernel.asm["ttgir"]`
etc.). The backend-as-stage-dict is a very clean backend seam: a backend = a
hash + an options dataclass + an ordered dict of IR-to-IR functions.

### Launch ABI: metadata-driven generic launcher

Notable evolution: older Triton **generated and compiled a C launcher per
kernel**; current Triton has replaced this with **one generic native `launch`
entry point driven by per-kernel signature metadata**. In
`third_party/nvidia/backend/driver.py`, `CudaLauncher.__init__` does, once per
compiled kernel:

```python
expanded_signature = expand_signature(signature.values(), tensordesc_meta, "nvTmaDesc")
self.arg_annotations   = annotate_arguments(expanded_signature)   # C objects to flatten tuples / drop constexpr
self.kernel_signature  = make_kernel_signature(expanded_signature)  # via utils.build_signature_metadata
```

`make_kernel_signature` **flattens nested tuple arguments** into a flat physical
parameter list and drops `"constexpr"` entries; the native `launch` then walks
the metadata to pack the actual `cuLaunchKernel` argument buffer. This is
exactly pdum's "one logical value → N physical parameters" marshaling table,
productionized: *a per-specialization descriptor, built once at compile time,
interpreted by a generic packer at call time.*

---

## 3. MLX custom kernels — the no-frontend floor

Primary sources:
[Custom Metal Kernels doc](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html),
[`mlx/fast.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/fast.cpp) (958 lines, custom-kernel portion a fraction of that),
[`mlx/backend/metal/custom_kernel.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/custom_kernel.cpp) (**127 lines**).

### API

```python
kernel = mx.fast.metal_kernel(
    name="myexp",
    input_names=["inp"], output_names=["out"],
    source="""
        uint elem = thread_position_in_grid.x;
        out[elem] = metal::exp(inp[elem]);
    """,                        # BODY ONLY — signature is generated
    header="",                  # optional definitions above the kernel
    ensure_row_contiguous=True, atomic_outputs=False,
)
out = kernel(inputs=[a], template=[("T", mx.float32)],
             grid=(n,1,1), threadgroup=(256,1,1),
             output_shapes=[a.shape], output_dtypes=[a.dtype],
             init_value=0, verbose=False)[0]
```

MLX generates the full `[[kernel]]` signature: one `device` buffer per declared
input/output (dtype taken from the arrays passed at call time — the generated
kernel name is mangled with the dtypes, e.g.
`custom_kernel_myexp_float_float16_t_float16_t`), template parameters from
`template`, and thread-position attributes **detected by scanning the source
string** for names like `thread_position_in_grid`.

### Binding rules — the interesting part

- `ensure_row_contiguous=True` (default): inputs are copied to row-contiguous if
  needed; **only the data pointer is bound** — no shape/strides.
- `ensure_row_contiguous=False`: `inp_shape`, `inp_strides`, `inp_ndim` buffers
  are **appended automatically, but only if the source string references them**;
  user indexes via the provided `elem_to_loc(elem, inp_shape, inp_strides,
  inp_ndim)` helper.

So the logical-array → physical-parameters expansion (pointer [+ shape + strides
+ ndim]) is *demand-driven by the kernel body*. Caching is per
`metal_kernel`-object + template-instantiation (each `mx.fast.metal_kernel()`
call makes a new Metal library — the docs explicitly say build once, call many).

### The floor now spans two backends

As of MLX 0.32.0 the `mx.fast` namespace has **`cuda_kernel`** and
**`precompiled_cuda_kernel`** with the same declared-inputs shape
([fast API index](https://ml-explore.github.io/mlx/build/html/python/fast.html)).
The same "body + declared names + template args + grid" contract now drives both
Metal and CUDA — strong evidence this is the right *backend floor*: the minimal
API every pdum backend should expose beneath any frontend, and the natural
escape hatch for users who want to hand-write one kernel without leaving the
framework. Total cost per backend is tiny (127 lines for the Metal binding).

---

## 4. Taichi (brief)

Primary source:
[`python/taichi/lang/kernel_impl.py`](https://github.com/taichi-dev/taichi/blob/master/python/taichi/lang/kernel_impl.py) (1,245 lines).
Status caveat: last release 1.7.4 (July 2025); development has slowed and forks
(`taichi-forge`, `gstaichi`) have appeared. Study it as design prior art, not as
a dependency.

- **Frontend**: `getsourcelines()` → `ast.parse(textwrap.dedent(...))` →
  `transform_tree(tree, ctx)` where an `ASTTransformerContext` carries globals,
  argument metadata, and source locations. The transformer visits Python AST and
  calls into the C++ `ASTBuilder` to construct Taichi's CHI IR (so like Triton:
  AST walk emits IR ops directly, no text).
- **Kernel templates & cache key**: `ti.template()` parameters plus structural
  features of other args drive instantiation. `TaichiCallableTemplateMapper.
  extract_arg()` produces the key per argument:
  - `ti.template()` → **the object's `id()`** (pointer address / weakref) —
    identity-keyed specialization, i.e. the numba anti-pattern, chosen
    deliberately for fields (a field *is* a global buffer);
  - `NdarrayType` annotation → `(element_type, len(shape), needs_grad, boundary)`
    — a clean structural tuple: dtype + rank, not shape values;
  - primitives → the value only when template'd; annotated scalars are runtime
    args and not in the key.
  `lookup()` interns each new key to a dense instance id; `materialize(key, ...)`
  compiles once per key into `self.compiled_kernels[key]`.
- **Argument binding**: `launch_kernel()` fills a `launch_ctx` with typed setters
  — `set_arg_ndarray` (Taichi ndarrays, optional grad buffer),
  `set_arg_ext_array` (zero-copy NumPy/PyTorch/Paddle: base pointer + shape, with
  element dims stripped), `set_arg_matrix`, etc.; `recursive_set_args` flattens
  nested `ArgPackType` structures into slots (64-arg limit). Again: per-type
  logical→physical setters over a flat slot array.

---

## 5. Synthesis: the common shape

Every member of the family that has a frontend has **the same skeleton**:

| Stage | cupy.jit | Triton | Taichi | MLX fast |
|---|---|---|---|---|
| Source recovery | `inspect.getsource*` + `ast.parse` | same (+ regex decorator strip) | same (+ `textwrap.dedent`) | — (user writes body) |
| Decoration-time work | none | hash source, find deps | extract/validate signature | build kernel object |
| Typing | call-time, from arg values | call-time + annotations | call-time template features | dtypes of passed arrays |
| Analysis style | single forward pass, types + emission interleaved | single forward pass emitting MLIR ops | single forward pass emitting CHI IR ops | — |
| Cache key | `(in_types, device_id)`; types stringify to C++ template names | `(specialization, options)` per call; sha(source+deps+sig+consts+backend+env) on disk | interned structural feature tuple | template-instantiation mangled name |
| Value specialization | implicit (`index_32_bits`), frozen consts | **default-on** divisibility/==1 hints, opt-out; `tl.constexpr` opt-in | `ti.template()` opt-in (by identity!) | explicit `template=[...]` |
| Globals/captures | silently frozen at transpile | dep-closure hashed; changed global **raises** | captured at materialize | n/a |
| Marshaling | array = one C++ struct (ptr+shape+strides) | flatten-tuples signature metadata + generic native packer | declared buffers; shape/strides appended iff referenced | typed `set_arg_*` into slots |
| Frontend size | ~1.8k lines | ~3.4k lines Python (jit+compiler+codegen) | ~1.2k lines Python + C++ transformer | ~130 lines/backend |

**Where they differ, and what's cleanest to steal:**

1. **Nobody in this family uses bytecode.** All four AST-subset frontends use
   `inspect.getsource` + `ast.parse` and accept the known failure modes (REPL
   source availability, decorator stripping) with small workarounds (cupy's
   lambda-by-line-number hunt, Triton's regex strip). Numba's bytecode frontend
   buys generality these systems don't need. For a *kernel subset* language, the
   AST route is the family consensus — relevant to desiderata open question 1.
2. **Nobody builds a standalone type-inference pass.** Typing is abstract
   interpretation fused with lowering: argument types flow forward through one
   AST walk that emits code/IR as it goes. Monomorphization-per-signature makes
   fixpoint inference unnecessary (loops: cupy just requires the loop var's type
   to be consistent). pdum's reference asset already discovered this shape.
3. **Cache keys are structural type descriptors, but everyone leaks *some*
   values in.** cupy leaks `index_32_bits`; Triton leaks divisibility/==1 by
   default; Taichi keys templates on `id()`. pdum's "types only, values by
   explicit opt-in" is stricter than any of them — Triton proves the mechanics
   (hint → key component + IR attribute), pdum should keep the opt-in polarity.
4. **The dependency-closure hash is Triton's** (`DependenciesFinder`,
   `(name, id(__globals__))` map, recursive `cache_key` of called jit functions,
   runtime raise on drift). It is the production version of the disk-cache
   prescription already in `design/dsl_caching_layer.md`.
5. **Marshaling converges on "flat physical slot list + per-kernel descriptor
   built at compile time."** Triton's `build_signature_metadata` (flatten tuples,
   drop constexprs, generic native packer) is the most explicit; cupy's trick of
   making the multi-field array parameter a single C++ struct is the cheapest;
   MLX's reference-driven conditional binding (pass strides only if the body
   uses them) is the most elegant.
6. **The backend floor is real and tiny.** MLX ships the same declarative
   raw-kernel contract on Metal *and now CUDA* in ~100-1000 lines per backend.
   A pdum backend that exposes exactly (source body, declared inputs/outputs,
   dtype/template substitution, launch geometry, arg binding) can host both the
   compiler's output and hand-written escape-hatch kernels through one door.
7. **Launch overhead is fought with generated dispatchers**, not faster generic
   code: Triton `exec`s a per-kernel Python binder; the specialize step got
   pushed to C++. pdum's hot-loop requirement (per-iteration ≈ a value write)
   will need the same move eventually: generate the phase-A/phase-B fast path
   per template.

---

## Design lessons for pdum.dsl

1. **Adopt the family-consensus frontend: `inspect.getsource` + `ast.parse`,
   single forward typed-lowering pass.** Budget evidence: a complete, useful
   Python→CUDA-C kernel language fits in ~1,800 lines (cupy.jit). Steal cupy's
   `Environment` shape (consts/params/locals/ret_type + resolution order) as the
   frontend's core data structure, but emit into pdum's IR instead of text, and
   take the dialect tables as an *input* (cupy hard-wires CUDA the same way M0
   hard-wires WGSL — don't copy that part).

2. **Make the marshaling layer a per-specialization descriptor interpreted by a
   generic packer** — Triton's `expand_signature` / `build_signature_metadata`
   pattern. At compile time, produce a flat list of physical slots, each tagged
   with (logical source: capture i / arg j / field path, physical form: raw
   scalar, pointer, shape word, uniform-buffer offset). At call time, a generic
   routine walks the descriptor and writes values. This one abstraction covers
   CUDA arg buffers, WebGPU uniform-buffer rewrites, C ABIs, and Python calls —
   and is the natural place for future unit auto-conversion (a conversion factor
   is just one more descriptor annotation applied during packing).

3. **Value-dependent specialization: keep opt-in polarity, copy Triton's
   mechanism.** Represent each lift as a *hint* that (a) becomes part of the
   cache key and (b) is attached to the IR as an attribute the backend may use
   (`divisibility=16`, `known_value=1`, `Literal[v]`). Triton shows key and
   codegen must be derived from the same hint list or they drift. Also copy
   `do_not_specialize`'s granularity lesson in reverse: pdum's opt-in should be
   per-capture/per-argument, not per-function.

4. **Steal `DependenciesFinder` for global/callee integrity.** Hash the
   dependency closure (referenced globals keyed `(name, id(__globals__))`,
   transitively-called jitted functions' keys) into the template identity, and
   *check* captured global values at call time — raising or recompiling on
   drift, never silently serving stale code (cupy.jit's silent freeze is the
   documented failure mode). This slots directly into the generation/world-age
   design in `design/dsl_caching_layer.md`.

5. **Define the backend floor as an MLX-shaped raw-kernel API** every backend
   must implement: `raw_kernel(name, input_decls, output_decls, source_body,
   header, launch_geometry, template_args)` with automatic signature generation
   and demand-driven metadata binding (bind shape/strides only when used). The
   compiled-from-Python path *targets this same API*, and users get a
   hand-written escape hatch per backend for free. MLX proves the cost is
   ~100-1,000 lines per backend and that the identical contract spans Metal and
   CUDA (and WGSL bind groups fit it naturally).

6. **Structure array types like `CArray` / Taichi's ndarray key: dtype + rank +
   layout flags, never shape values.** Both systems independently chose
   `(dtype, ndim, contiguity[, grad/boundary])` as the specialization unit —
   matching the `typeof` lattice already specified in
   `design/dsl_caching_layer.md`. Audit every flag admitted into the key
   (cupy's `index_32_bits` shows how a value property sneaks in unlabeled);
   pdum should require such flags to be declared as explicit lifts (lesson 3).

7. **Do not key anything on `id()`** — Taichi's `ti.template()` → object-address
   keys are the identity-caching anti-pattern pdum exists to avoid (aliasing
   after GC reuse, leaks via the mapper's strong refs, no cross-run stability).
   Where "specialize on this specific buffer" is genuinely wanted, express it as
   an explicit `Literal`-style lift of a *stable* token, not an address.

8. **Plan the launch fast path as generated code.** When the tight-loop
   overhead budget bites, generate (via `exec`, like Triton's
   `create_function_from_signature`) a per-template binder that inlines the
   fingerprint check and descriptor-driven packing for that specific signature,
   instead of walking generic metadata per call. Design the phase-B entry so
   this can be dropped in later without API change.
