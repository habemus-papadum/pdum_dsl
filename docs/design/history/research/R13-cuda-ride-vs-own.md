# R13 — CUDA bridge: ride CuPy vs own via cuda.core

*Research agent survey, 2026-07-12 backend detour. Verdict: OWN.
Consumed by `070_backends-notes.md`.*

## 1. CuPy RawKernel/RawModule mechanics (the RIDE option)

- **Compile**: lazy, at first `.kernel` access/call. `RawKernel`/`RawModule` →
  `_get_raw_module` decorated `@cupy._util.memoize(for_each_device=True)`,
  keyed on `(code, path, options, backend, translate_cucomplex,
  enable_cooperative_groups, name_expressions)` per device; plus a per-instance
  `self._kernel_cache = [None] * getDeviceCount()` of `Function` objects.
- **Disk cache**: `~/.cupy/kernel_cache` (`CUPY_CACHE_DIR`), key = hash of
  `((arch, options, nvrtc_version, backend) + arch_flags) + source +
  extra_source + cupy_cache_key [+ sorted name_expressions]`. Content-addressed
  — philosophically identical to pdum's tier-2 artifact cache, so
  *complementary, not competing*; the in-process memo duplicates
  `FastRecord.artifact` but is content-keyed and benign.
- **NVRTC**: default backend `nvrtc` (or `nvcc`), C++17 default since v14,
  `-arch=sm_*` → cubin when possible else PTX; `name_expressions` →
  `nvrtcAddNameExpression`/`getLoweredName`.
- **Arg packing** (`cupy/cuda/function.pyx` `_pointer()`): cupy.ndarray →
  **raw device pointer only** ("RawKernel ignores views" — shapes/strides never
  passed); numpy scalars → `CScalar`; Python `int/float/bool/complex` →
  `long long/double/bool/cuDoubleComplex` by value (silent-mismatch footgun);
  size-1 numpy arrays by value (the struct escape hatch). Launch builds a fresh
  `vector[void*] kargs` of per-arg `CPointer` objects **every call**, then
  `driver.launchKernel(..., kargs.data(), extra=0)` — kernelParams path, never
  the packed-buffer path.
- **Streams**: `Function.__call__(..., stream=None)` → thread-local current
  stream; `cupy.cuda.Stream.from_external()` wraps anything with
  `__cuda_stream__` (torch streams).
- **Foreign memory**: yes — `cupy.from_dlpack(torch_tensor)` zero-copy view →
  RawKernel arg; documented, stream-safe.
- **Weight**: `cupy-cuda12x` 14.1.1 wheel = **134 MB** linux x86_64;
  CUDA-major-locked package naming (`cupy-cuda11x/12x/13x`); still needs NVRTC
  from CTK or `[ctk]` extra.

## 2. Owning it (cuda-python)

- **Status change (decisive)**: `cuda.core` left experimental and is **1.0
  stable with SemVer as of CUDA 13.3 (May 2026); v1.1.0 July 9, 2026**.
  `cuda-bindings` 13.3.x; wheel ≈ **6 MB**; NVRTC via `nvidia-cuda-nvrtc-cu1x`
  wheel (~90 MB — a cost *shared* with CuPy) or system CTK.
- **cuda.core path**: `Device().set_current()` (primary context — same one
  torch/cupy use); `Program(code, "c++", ProgramOptions(std="c++17",
  arch=f"sm_{dev.arch}")).compile("cubin", name_expressions=(...,))` →
  `ObjectCode.get_kernel(name)` → `launch(stream, LaunchConfig(grid, block),
  ker, *args)`. Getting-started kernel runtime ≈ **30 lines**; production
  runtime with error/log capture ≈ 60–100 lines — inside §2.10's 130–220 budget.
- **Caching**: `Program.compile(cache=...)` is **opt-in and explicit**; pass
  nothing and cuda.core performs no hidden caching — pdum's two-tier cache
  remains the sole authority. (CuPy's is always-on.)
- **Arg packing**: `cuda.core` `ParamHolder` accepts
  Buffer/bool/int/float/complex/numpy scalars/ctypes; Python `int` →
  `intptr_t` pointer "without any warning" (harmless — PackPlan knows every
  slot's type); it `PyMem_Malloc`s per scalar per call and does **not** accept
  a pre-packed buffer.
- **The staging-buffer ABI — YES, it maps directly**:
  1. **Confirmed in Python**: `cuda.bindings.driver.cuLaunchKernel` accepts
     `kernelParams` as a raw `void**` address / array of pointer addresses
     (`test_kernelParams.py`). Since `FastRecord.staging` is preallocated and
     never resized, a **persistent pointer table into staging at each SlotSpec
     offset can be built once per FastRecord**; per call = extract-packs
     staging + write buffer-leaf pointers + one FFI call. Zero per-arg Python
     conversion — the literal `launch(staging, buffer_leaves)` contract.
  2. The C-level `extra` path (`CU_LAUNCH_PARAM_BUFFER_POINTER`/`_SIZE`/`_END`)
     takes exactly one packed buffer = staging itself; the Python binding's
     list form is unconfirmed (tests pass `extra=0`), so route 1 is the safe
     design.
- **Outputs/allocation**: owning launch does NOT force owning allocation —
  allocate via the user's library (torch/cupy through a pluggable allocator
  hook) or `cuda.core` `DeviceMemoryResource`/`Buffer`;
  `cuda.core.utils.StridedMemoryView` consumes any DLPack/CAI object for leaf
  extraction without CuPy.
- **Streams (minimal own story, ~20 lines)**: one stream per backend instance
  (or accept any `__cuda_stream__` object); when materializing DLPack views
  call `__dlpack__(stream=s)` — CUDA semantics per array-API spec (`None`/`1`
  legacy default, `2` per-thread, `>2` real stream, `0` disallowed, `-1` = no
  sync; **producer inserts the wait-event on the consumer's stream**). Launch
  with `hStream = s`.

## 3. Prior art

- **numba-cuda**: a decade maintaining its own ctypes `cudadrv/driver.py`; the
  NVIDIA-maintained `numba-cuda` **always uses cuda-python bindings — legacy
  ctypes removed**. Lesson: don't own the *FFI layer*; owning everything above
  it is cheap now that NVIDIA owns the bindings.
- **NVIDIA Warp** (closest shape to pdum: codegen → C++/CUDA source): owns
  compile via NVRTC (bundled headers, no CTK), owns its disk cache
  (`~/.cache/warp/<ver>`, ModuleHasher over source+options+config), owns
  launch, ships pre-compiled kernels in wheels. Owns the stack.

## 4. Comparison

| Criterion | RIDE cupy.RawModule | OWN cuda.core (+bindings escalation) | Hybrid |
|---|---|---|---|
| Lines to build | ~40–60 | ~60–100 v1; +50–80 for pointer-table launch | ~80–120 + cupy glue |
| Caching control | disk cache complementary; in-process memo redundant, benign; not cleanly off | **fully ours**; cuda.core cache strictly opt-in | ours |
| PackPlan→ABI fit | **poor**: per-call Python arg tuple; cupy re-marshals each arg — double-marshaling against §2.11/§2.12 | **direct**: persistent void** into staging; one FFI call/launch | direct |
| Interop (DLPack-first) | fine, but forces cupy import even for torch users | **native**: StridedMemoryView, `__dlpack__(stream=)`, no tensor-lib dep | forces cupy |
| Dependency weight | **134 MB**, CUDA-major-named packages | **~6 MB** bindings (+ NVRTC wheel ~90 MB common to both) | 134 MB |
| Stream story | current-stream thread-local + `Stream.from_external` | explicit stream + DLPack negotiation, ~20 lines | muddier |
| Risk | very mature; scalar-coercion/view footguns; hot-path µs churn | cuda.core 1.x young (pin `>=1.1,<2`) but SemVer + NVIDIA-backed | two half-owned layers |

## 5. Recommendation: OWN, via cuda.core — cupy demoted to optional allocator

1. Build `backends/cuda/` on cuda.core 1.x with NO `cache=` — pdum's caches
   stay the only caching authority. (1.0-stable removes the old
   "experimental" objection.)
2. Launch v1 with `cuda.core.launch()` (correct, ~0 lines), with a pre-shaped
   escalation to `cuda.bindings.cuLaunchKernel` + a once-per-FastRecord
   pointer table into `staging` — the *purest* realization of
   `launch(staging, buffer_leaves)` of any backend. Riding cupy would un-pack
   staging into Python objects per call and let cupy re-marshal them, taxing
   the 5 µs hit budget and re-doing the marshaling this framework exists to own.
3. DLPack-first for free: inputs via `__dlpack__(stream=our_stream)`; outputs
   via pluggable allocator (user's torch/cupy, else DeviceMemoryResource). A
   torch user never installs 134 MB of CuPy.
4. Prior art unanimous for this shape (numba-cuda migration; Warp). CuPy's one
   genuinely valuable layer — its content-addressed disk cubin cache — is ~40
   lines to replicate under pdum's planned §4.4 disk tier, keyed the same way.
5. Keep `RawModule` as a documented fallback, not a dependency: the Backend
   seam makes a 40-line cupy runtime a drop-in if cuda.core misbehaves.
   §5 module-map line "`backends/cuda/` (CuPy RawKernel)" to be revised.

Sources: cuda.core getting-started/API/Program docs; cuda-python releases;
CUDA 13.3 blog; cuda.bindings overview (SAXPY); test_kernelParams.py;
cuLaunchKernel driver docs; CuPy RawKernel/kernel-guide/performance docs;
cupy compiler.py/function.pyx/raw.pyx sources; CuPy interoperability;
cupy-cuda12x PyPI; numba CUDA bindings docs; numba-cuda install; Warp repo +
basics; array-API `__dlpack__` stream spec.
