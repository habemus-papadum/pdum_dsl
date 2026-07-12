# R14 — Metal bridge: ride mx.fast.metal_kernel vs own the stack

*Research agent survey, 2026-07-12 backend detour. Verdict: OWN.
Consumed by `070_backends-notes.md`.*

## 1. Riding: `mx.fast.metal_kernel` mechanics (MLX 0.32, current)

- **Contract**: `metal_kernel(name, input_names, output_names, source,
  header='', ensure_row_contiguous=True, atomic_outputs=False,
  compile_options={'math_mode': 'safe|relaxed|fast'})`. `source` is
  **body-only**; MLX auto-generates the `[[kernel]]` signature (buffer
  indices, thread builtins, and — when `ensure_row_contiguous=False` —
  auto-passes `a_shape`/`a_strides`/`a_ndim` per referenced input). The
  returned callable is keyword-only: `(inputs, output_shapes, output_dtypes,
  grid, threadgroup, template, init_value, verbose, stream)` — grid/
  threadgroup, template params, and explicit output shapes/dtypes at CALL
  time (matches pdum's explicit-outputs lean). `atomic_outputs` + `init_value`
  cover reductions. Dispatch is `dispatch_threads` (grid = threads,
  non-uniform threadgroups — fine on Apple Silicon).
- **Caching**: in-process only. `CustomKernel::eval_gpu` calls
  `Device::get_library(name_, compile_options_, builder)`; `device.cpp` shows
  `library_map_` is keyed by **name string only — a cache hit never re-checks
  source**. No content hashing, no disk persistence, no way to serialize
  compiled libraries. Riding would force pdum to embed its artifact
  content-hash in the kernel `name` to avoid stale-library collisions, and
  the disk artifact tier could only persist *source*, never binaries.
- **Streams/laziness**: outputs are lazy `mx.array`s; nothing runs until
  `mx.eval(out)` (blocking) / `mx.async_eval`; `mx.synchronize(stream)` for
  fences; kernel takes `stream=`. Minimal-correct sync: `mx.eval` per call —
  which forces graph-node build + schedule + encode + GPU wait per call, an
  overhead floor well above pdum's single-digit-µs hit budget unless calls
  stay inside MLX's lazy world.
- **Marshaling mismatch**: inputs must be `mx.array`s (bindings convert
  scalars via `to_array`). pdum's packed staging buffer would become a
  per-call `mx.array` (alloc + copy) instead of a `setBytes`; every buffer
  leaf round-trips through `mx.asarray`.
- **Interop**: MLX speaks DLPack both ways (`mx.from_dlpack`, `mx.asarray`
  zero-copy "when possible"; `np.array(a, copy=False)` zero-copy out). Torch
  MPS DLPack landed (pytorch#153789 closed; torch 2.12+ shared-storage MPS
  tensors import zero-copy). Cross-scheduler hazard: DLPack carries no fence
  on Metal — `torch.mps.synchronize()` before MLX reads, `mx.eval/synchronize`
  before torch reads.
- **Weight**: `mlx` + `mlx-metal` wheels ≈ **41–57 MB**, macOS ≥14 arm64 tags;
  a heavyweight, Apple-paced dependency for a ~200-line job.

## 2. Owning: Python-drives-Metal

- **Call sequence** (all documented API): `MTLCreateSystemDefaultDevice` →
  `newLibraryWithSource_options_error_` → `newFunctionWithName_` →
  `newComputePipelineStateWithFunction_error_` → per launch:
  `commandQueue.commandBuffer` → `computeCommandEncoder` →
  `setComputePipelineState_` → **`setBytes_length_atIndex_` for the packed
  staging buffer** (≤4 KB — a literal 1:1 realization of
  `PackPlan.UniformSlot`) → `setBuffer_offset_atIndex_` per leaf (`KernelArg`)
  → `setThreadgroupMemoryLength_atIndex_` if needed →
  `dispatchThreadgroups_threadsPerThreadgroup_` → `endEncoding`/`commit` →
  `waitUntilCompleted` only at readback.
- **Measured size**: alvinwan's standalone PyObjC compute script is **~85
  lines end-to-end**; tinygrad's whole production Metal runtime is **~240
  lines** (`ops_metal.py`) + **79 lines** of zero-dep ctypes `objc_msgSend`
  wrapper (`support/objc.py`) — including its fancy
  `MTLCodeGenServiceBuildRequest` compiler path and graph batching, neither
  needed here. A pdum runtime (device/queue singleton, compile+error-surface,
  pipeline cache, launcher closure, readback views) is realistically
  **150–250 lines**, inside the doc's 130–220 budget.
- **Buffer interop**: allocate leaves as
  `newBufferWithLength_options_(MTLResourceStorageModeShared)`; read back
  zero-copy via `buffer.contents().as_buffer(n)` → numpy view (unified
  memory). Torch MPS import: the storage data pointer of an MPS tensor IS the
  `id<MTLBuffer>` (PyTorch's own custom-kernel docs), reachable via
  `t.untyped_storage().data_ptr()` + an objc cast — zero-copy but leans on an
  internal; DLPack-kDLMetal capsule parsing is the documented alternative.
  `newBufferWithBytesNoCopy` is a trap for staging (page-aligned pointer AND
  length) — irrelevant since `setBytes` is the right tool for uniforms.
- **Pain**: PyObjC underscore-selector ergonomics and error-out-params; PyObjC
  release lag on new macOS SDKs (pyobjc#580). Both mitigated by the tinygrad
  option: 79 lines of ctypes, zero dependencies. Own-path exclusive upside:
  `MTLBinaryArchive` lets the artifact tier persist *compiled pipelines* to
  disk — impossible when riding.

## 3. Comparison

| Dimension | RIDE mx.fast.metal_kernel | OWN (PyObjC or ctypes-objc) |
|---|---|---|
| Lines to build | ~50 (wrapper) | ~150–250 (85-line existence proof; tinygrad 240+79) |
| Caching control | Name-keyed in-process memo, NO source check, no disk; must smuggle content hash into `name`; binaries unserializable | Both tiers fully owned; `MTLBinaryArchive` disk persistence possible |
| Staging-buffer fit | Per-call `mx.array` alloc+copy of staging blob | `setBytes` straight from the reused `bytearray` — exact PackPlan match |
| Interop/FFI | DLPack in/out incl. torch-MPS (2.12+); but all leaves forced through `mx.array` | Unified-memory views free; torch-MPS via storage-ptr or DLPack; capsule parsing ~60 lines |
| Dependency weight | 41–57 MB mlx+mlx-metal | pyobjc-framework-Metal (few MB) or **0 bytes** (ctypes) |
| Sync story | Lazy graphs; `mx.eval` per call (blocking, heavy) or batch-then-eval; cross-framework fences manual anyway | In-order queue; commit now, wait only at readback; `MTLSharedEvent` for cross-queue |
| Hot-path overhead | Graph build + scheduler + eval per call — blows the 5 µs budget | Encoder + dispatch, all under pdum's control |
| Risk | Apple-paced drift; two schedulers to debug through; name-collision footgun | We own sync correctness; objc bridge churn (tinygrad proves manageable) |

## 4. Recommendation: OWN it

The hunch "both are easy" is correct — and that is precisely why riding
loses. MLX's genuinely valuable properties (lazy fusion scheduler, AD, its
allocator) are all things pdum deliberately does NOT want in the launch path,
while the three things pdum's kernel is built around — the two-tier cache,
the packed staging buffer, the per-call leaves channel — are respectively
**undermined** (name-keyed memo, no binary persistence), **mismatched**
(staging → per-call `mx.array` copy instead of `setBytes`), and **taxed**
(every leaf laundered through `mx.asarray` + lazy eval) by the MLX bridge.
Build the owned runtime with tinygrad-style ctypes-objc (or PyObjC initially
— swappable behind `Backend.runtime`), `setBytes` for staging, `setBuffer`
for leaves, commit-now/wait-at-readback sync. Keep `mx.fast.metal_kernel` in
reserve as a *second, thin satellite backend* for users whose data already
lives in MLX's lazy world (the §14 "target it" posture, applied to MLX). One
flag: MLX autograd through custom kernels is the one capability the owned
path cannot inherit; already covered by the surface-E `custom_vjp` note.

Sources: MLX custom-Metal-kernels docs + metal_kernel API; mlx
custom_kernel.cpp + device.cpp get_library + python fast.cpp bindings; MLX
conversion/DLPack docs; mlx#1159; pytorch#153789; tinygrad ops_metal.py +
support/objc.py; alvinwan 85-line PyObjC gist + writeup; PyTorch custom MPS
kernel docs (MTLBuffer-from-storage); mlx-metal PyPI wheel sizes; pyobjc#580.
