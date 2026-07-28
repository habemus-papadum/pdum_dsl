# spike_metal — findings

(Written by the session lead from the spike agent's report; harness
blocks subagents writing report files. Entry:
`/Users/nehal/src/pdum_dsl/.venv/bin/python -B run.py
[diff|rows|math|mem|seam]`, whole suite ~22 s. Read `seam_ledger.py`
first — the a/b/c classification is runnable DATA. Dependency added to
the shared venv: `pyobjc-framework-Metal` 12.2.1 (+core, +cocoa),
additive only. `xcrun metal` is absent on this machine and does not
matter — `newLibraryWithSource:` needs no Xcode, so a JIT MSL backend
is viable with Command Line Tools only.)

## Verdict

The runtime/backend definitions **carve well but are incomplete**.
`msl_backend.py` imports no Metal; `metal_runtime.py` imports no pdum;
the separation was enforced physically, not asserted. The strongest
positive result: **the executor column really is the seam** — the
entire test-side diff between "run on WebGPU" and "run on Metal" is
ONE IDENTIFIER (`wgpu_artifact` → `metal_artifact`); staging, uniform
capture, fn-arg splicing, the overlap refusal, and writeback are
reused unmodified from `kernel.py:1082-1113`, for a target whose
memory model, binding model, dispatch model and source language all
differ. But the boundary is not a line — it is a **negotiated
contract with at least four clauses** (binding table, launch geometry,
guard/exactness convention, device representation of the staging
plan), plus a fifth category neither box owns (the target numeric
contract). A `Backend` returning `str` and a `Runtime` taking `str`
can express none of them; what a backend must return is
(source, launch-contract).

## 1. Differential validation — 11 subjects, four device columns

8 subjects verbatim from `test_conformance_kernels.py`; `banded` added
because **the battery has no `where`/select subject at all** (a
coverage hole — select-normal-form is the law the control-flow
position rests on); `copy_big`/`markers_big` deliberately ragged mod 8
to exercise the bounds guard.

- Device vs f64 reference: approximate as expected (worst 3.9e−06,
  inside the battery's rtol=1e-5).
- **WGSL vs Metal: bitwise exact (0.0) on all 10 translatable
  subjects.** Guard vs exact-grid dispatch: bitwise equal everywhere.
- `tanh_wide` REFUSES on both device paths — see §2.

**Caveat that must travel with the exactness result:** wgpu on this
machine lowers WGSL through Naga to MSL — the two device columns share
a math library and a GPU by construction. Exactness proves two
independently written translators emit arithmetically equivalent
programs; it proves nothing cross-vendor. A CUDA or Vulkan/Windows
column is the first honest test, and `tanh_wide` is standing evidence
it would find something.

## 2. The target numeric contract (unlooked-for finding)

**Metal's `tanh` returns NaN for |x| ≥ 44.3614** — exactly
`log(FLT_MAX)/2`, i.e. computed via `exp(2x)` (f32 overflow, inf/inf).
`MTLMathModeSafe` does NOT fix it (tested) — math library, not a
fast-math tradeoff. Both device paths fail identically, and fail as a
REFUSAL (`Tensor.from_numpy` rejects NaN at decode). Invisible in the
battery today only because `spiky` runs at 5×7. One-row fix, provably
free: `tanh(clamp(x,−20,20))` is bitwise-identical to unclamped tanh
on |x|≤20 and finite over [−200,200]; `tanh(20.0f) == 1.0f` exactly.

**This item has no code home under either definition** — the
translation row is faithful, the runtime never touches arithmetic, and
it is target-specific so not shared. It is a fourth category: the
TARGET NUMERIC CONTRACT. 210 has the right section and no mechanism.
`exp`, `sinh/cosh`, `pow` are the usual company (not surveyed).
Exhibit: `mathrows.py`.

## 3. WGSL vs MSL row diff — measured, not eyeballed

**A (emitted code):** 123/123 emitted statements (100.0%) convert
WGSL→MSL under just three lexical rules (cast spelling, `f` literal
suffix, C declaration form). No operator row differs: `select` has the
same argument order and semantics; all ten math builtins spell the
same; infix/comparison/bool-widening identical. The five real
differences, only two structural: buffers are entry-point parameters
`[[buffer(i)]]` with `const device` for read-only (vs module-scope
`@group/@binding` + `var<storage, read>`), and the entry point cannot
be `main` and carries no workgroup size (see FAIL-1).

**B (translator source):** difflib over the two `_translate`s: 145/171
code lines byte-identical (ratio 0.871). A is a fact about the
languages; B is a fact about OUR factoring — only B is ours to fix.

## 4. Unified memory

**Zero-copy adoption proved three ways** (`unified.py`,
`hasUnifiedMemory: True`): kernel stores visible in the numpy array
with no readback call (only `waitUntilCompleted`, a sync act, not a
transfer); host writes after buffer creation seen by the next dispatch
with no upload; `MTLBuffer.contents()` IS the numpy data pointer.
Mechanism: `newBufferWithBytesNoCopy:` over an mmap-backed
page-aligned numpy array.

**Protocol benchmark** (hand-written shaders both sides; wall ms,
min-of-reps, GPU pre-warmed): wgpu-full / metal-adopt = 5.5×, 5.9×,
7.8×, 10.8× at 4K/64K/1M/4M elements. 210's fixed-latency readback
reproduces exactly (1.37→2.23 ms for 1024× the data). Metal GPU timing
via `GPUStartTime/GPUEndTime` needs no query set and no feature
request — pointed contrast to spike_runner's H5. WebGPU's
`max_compute_workgroups_per_dimension`=65535 bit again; Metal has no
such limit.

**The deflation — the more important number.** Through the REAL
`_Artifact.launch`, the win almost vanishes: at 65 536 elements the
entire Metal-adopt device protocol is **0.22 ms** inside a **~400 ms**
launch — the device's share is under one part in a thousand. The cost
is `repack` (66–100%: `to_numpy(order=) + ascontiguousarray(f32)`,
the verbatim-shared line) plus `writeback` (the same path in
reverse). **`Tensor.to_numpy` is `itertools.product` over every
lattice point calling `.item()`** (tensor.py:184-200, self-described
"Materialize (naively) for testing") — a reference-tier materializer
on the hot path of EVERY device launch, both backends. On unified
memory the transfer was never the cost; removing it exposes our own
host-side repack as ALL of the cost. This is spike_runner's H4
arriving from the opposite direction.

## 5. Seam ledger (condensed; full data in `seam_ledger.py`)

- **(c) Shared/target-neutral:** region walking (genuinely shared
  already); marker tables (byte-identical dicts, now a 3rd copy);
  expression walker structure; buffer index arithmetic; writable
  determination; uniform slot collection; staging unpack; launch
  geometry (extents→threads); host repack; writeback discipline; the
  launch protocol (`kernel.py:1082-1113`, reused UNMODIFIED — the
  win); artifact/cache key discipline (decorative on both sides —
  FAIL-5).
- **(a) Backend (per-language):** literal rendering, cast spelling,
  declaration form, buffer/read-only declarations, entry-point name,
  thread-id builtin, preamble.
- **(b) Runtime (per-API):** device acquisition (singleton-no-options
  vs constructible), source→pipeline, buffer alloc/upload, zero-copy
  adoption (no WebGPU equivalent), resource binding (bind-group object
  vs `setBuffer:atIndex:` — Metal has no bind-group object),
  encode/dispatch, submit/sync, readback (round-trip vs view vs
  none-under-adoption), GPU timing.

## 6. Where the carve failed (the finding 280 needs)

- **FAIL-1 — workgroup size is backend in WGSL, runtime in Metal.**
  `@workgroup_size` is shader text; Metal's threadgroup size is a
  `dispatchThreadgroups:` argument appearing nowhere in source. Routed
  through `meta['threadgroup']` — a backend returning only `str`
  cannot express Metal. 210's sentence is WebGPU-shaped.
- **FAIL-2 — the bounds guard is emitted source whose necessity is a
  runtime choice.** WebGPU dispatches whole workgroups (guard
  needed); Metal's `dispatchThreads:` launches the exact grid (guard
  dead). Proven: guard-vs-exact bitwise equal on all subjects. The
  WGSL copy also FUSES extents→threads (shared) with
  threads→workgroups (runtime); splitting them exposed this.
- **FAIL-3 — the binding table is co-owned.** `layout="auto"` prunes
  unused bindings — backend code encoding a WebGPU pipeline-layout
  inference rule; Metal sets buffers by index, unused args harmless.
  Any Backend interface must return the binding table as DATA, never
  implicit in text.
- **FAIL-4 — the device representation of the staging plan belongs to
  neither, so both invent it** — and both silently narrow i32/i64
  slots to f32 (forbidden by 210's "narrowing is declared, never
  silent"). With a third backend the invention has happened three
  times. A hole in the SHARED tier; where spike_runner's H3 bites.
- **FAIL-5 — the executor column has no cache key; a second device
  backend makes the collision concrete.** `wgpu_artifact`/
  `metal_artifact` never consult the `(region.key, _EXECUTOR_FP)`
  cache; `WGPU_FP`/`METAL_FP` are decorative. Verified: repeated
  `wgpu_artifact(art)` recompiles (8–9 ms each), and the WGSL and MSL
  artifacts share an identical `region.key` — if they DID key today,
  they would collide. Backend identity must enter the content key
  now.
- **FAIL-0 — the target numeric contract** (§2): neither backend nor
  runtime nor shared.

## 7. The program diff and the proposed spelling

Today: one identifier (`wgpu_artifact` → `metal_artifact`). That
spelling must not become user-facing: it needs an `_Artifact` no
public API hands out (spike_runner H2), and reads as a conversion
function — the "magic compile" 270 refuses. Proposed, per no-magic and
the existing bracket precedent:

```python
from pdum.tl.runtime import metal, webgpu   # explicit, importable
kernel[on(metal)](src, dst)                 # per-launch selection
art = kernel.artifact(src, dst)             # the missing public door (H2)
art.on(metal).launch((src, dst))            # 270's incremental chain
```

Properties that fall out of the build: (1) `on(...)` must KEY the
artifact cache — `(region.key, backend_fp, runtime_fp)` (FAIL-5);
(2) `on(...)` names a PAIR — MSL-the-backend and Metal-the-runtime are
separable (the same MSL would serve a metal-cpp harness; the same
runtime would launch a precompiled `.metallib`); "usually 1:1" held
here only because we wrote both halves; (3) nothing in `on(...)`
implies a transfer — under adoption a tensor's memory is
simultaneously host memory and Metal device memory, and
`Buffer.device` being a bare string with no registry stops being
cosmetic the moment two real runtimes exist.

## Files

`msl_backend.py` (IR→MSL, imports no Metal) · `metal_runtime.py`
(PyObjC device/buffers/dispatch/timing, imports no pdum) ·
`metal_executor.py` (the glue; ~40 lines of substance, every block
tagged BACKEND/RUNTIME/SHARED) · `subjects.py` · `differential.py` ·
`rowdiff.py` · `mathrows.py` · `unified.py` · `seam_ledger.py` ·
`run.py` · `_paths.py`.
