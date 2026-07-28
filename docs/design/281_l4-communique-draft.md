# 281 — L4 communiqué (DRAFT): Metal-era learnings to integrate before backend work

**Status: DRAFT — proposed as a separate small PR to `main`** (an
amendment to `210_backend-notes.md`, which is the L4 builders' first
read), so the CUDA-era work starts with these in hand. Every claim
below is backed by rerunnable code in
`explorations/graphics/spike_metal/` (with `spike_runner/` and
`spike_controlflow/` as siblings); FINDINGS.md in each dir carries the
evidence.

## The ruling to integrate (owner, 2026-07-28)

**Thread sizing specializes; block sizing is launcher data.** 210's
sentence "workgroup size is pipeline-creation-time; dispatch
dimensions are launcher data" is WebGPU-shaped: in WGSL the thread
size is shader TEXT, in Metal it is a `dispatchThreadgroups:`
ARGUMENT appearing nowhere in source, in CUDA it is blockDim with
launch bounds making it compile-relevant. The general rule: **threads
per group is a SPECIALIZATION parameter — it keys the artifact and a
change recompiles — whether or not the target embeds it in source.
Number of groups is launch data everywhere and never specializes.**
Consequence: a backend cannot return only source text; it returns
(source, launch-contract), and the launch contract carries the thread
size as data.

## Runtime facts (measured on the M3 Ultra unless flagged)

- **PyObjC is entirely adequate for a Metal runtime.** No Swift/ObjC
  shim needed; selector mangling is mechanical; reading buffer
  contents and adopting host memory work directly on Python buffer
  objects. And **JIT MSL needs no Xcode**: `xcrun metal` is absent on
  the dev machine and it does not matter — `newLibraryWithSource:`
  compiles at runtime with Command Line Tools only.
- **Unified-memory adoption is real and bidirectional.**
  `newBufferWithBytesNoCopy:` over a page-aligned host allocation:
  kernel stores appear in the numpy array with NO readback call
  (`waitUntilCompleted` is a sync act, not a transfer); host writes
  after buffer creation are seen by the next dispatch with no upload;
  `MTLBuffer.contents()` IS the host pointer. The runtime interface
  must treat "no transfer exists" as a first-class case, and
  `Buffer.device`-as-string cannot answer "which device is this on"
  when memory is simultaneously host and device.
- **The bounds guard is a launch-contract clause, not universal
  source.** WebGPU dispatches whole workgroups (the overhang guard is
  required); Metal's `dispatchThreads:` launches exact grids and the
  guard is dead code (proven: guard-vs-exact bitwise equal on all
  subjects). CUDA dispatches whole blocks — WebGPU-shaped. Emit the
  guard per-runtime, from the contract.
- **OOB models differ structurally** (NOT tested — flagged as a model
  difference): WebGPU implementations bounds-clamp storage accesses
  per the spec's safety model, while Metal raw `device` pointers are
  genuine UB. The keying-ladder ruling (reference-certifies
  in-bounds before device runs) covers both — but never lean on a
  device-side safety net, and the descent-license language for
  device-tier OOB (BOUNDARIES) should record the per-target model.
- **GPU timing:** Metal's `GPUStartTime/GPUEndTime` is free — no
  query set, no feature request. WebGPU timestamp queries must be
  requested at DEVICE CREATION (the current `graphics._device()`
  singleton takes no feature requests — reach it before building
  runtimes). Rig gotchas worth inheriting: the last timestamp pair in
  a batch can resolve to zero on this driver (encode a discard rep);
  the GPU idles low — the first program timed in a process reads
  ~1.7× high (sustained warm-up, then min-of-samples); WebGPU caps
  `max_compute_workgroups_per_dimension` at 65535 where Metal has no
  such limit.

## Backend facts

- **Op-row portability is near-total between C-family shading
  languages:** 100% of emitted WGSL statements convert to MSL under
  three lexical rules (cast spelling, float-literal suffix,
  declaration form); no operator/builtin row differs — `select`
  argument order included. Expect similar for CUDA C. The shared
  expression-emitter design is justified; the marker tables now exist
  in three byte-identical copies and should exist once.
- **The binding table must be returned as DATA.** WebGPU's
  `layout="auto"` prunes unused bindings (a runtime rule the backend
  must anticipate in text); Metal has no bind-group object (buffers
  set by index, unused args harmless); CUDA has pointer args. Implicit
  binding-in-text does not survive three targets.
- **The staging plan needs a device-representation clause.** The plan
  describes HOST bytes only; three backends have now independently
  invented "uniforms arrive as flat f32" — and all silently narrow
  i64 slots to f32, which 210 forbids ("narrowing is declared, never
  silent"). Fix once, in the shared tier; a real WGSL uniform buffer
  additionally wants stride-16 emission from the plan.
- **Backend identity must enter the content key BEFORE a second
  device backend ships.** The WGSL and MSL artifacts of the same
  kernel share an identical `region.key`; today's `WGPU_FP` is
  declared but never keys anything, and executor-swapped artifacts
  recompile every call (8–9 ms per distinct pipeline).

## The target numeric contract (a category with no owner)

**Metal's f32 `tanh` returns NaN for |x| ≥ 44.3614** (exactly
`log(FLT_MAX)/2` — computed via `exp(2x)`); `MTLMathModeSafe` does
not fix it; both device paths refuse identically at result decode.
The one-row fix (`tanh(clamp(x, -20, 20))`) is bitwise-identical to
unclamped tanh wherever both are finite. `exp`, `sinh/cosh`, `pow`
are the usual company (unsurveyed). The category — per-target
math-library deviations, their fix rows, and the proofs the fixes are
free — has no home in a backend/runtime split; it needs a declared
mechanism (210 states the tolerance policy; nothing owns the rows).
Related battery gaps: the conformance kernel battery has NO
`where`/select subject, and its inputs are too small to reach any
math-library edge (tanh's argument never exceeds ~2 at 5×7) — the
adversarial-input-families doctrine applies to the battery itself.

## A measurement warning for the measure-as-you-compile thesis (270)

Through the real `_Artifact.launch`, host repack is **88–100% of
launch cost** on BOTH backends — `Tensor.to_numpy` is a per-element
reference-tier materializer (`itertools.product` + `.item()`,
self-described "for testing") sitting on every device launch. At
65 536 elements the entire Metal device protocol is 0.22 ms inside a
~400 ms launch. Until that path is fixed (or routed around), any
transform-run-measure loop over device executors measures the repack,
not the program. On unified memory the transfer was never the cost;
on discrete GPUs it will be — but today's numbers are noise either
way.

## Pointers

`explorations/graphics/{spike_metal,spike_runner,spike_controlflow}/FINDINGS.md`
(each rerunnable), `explorations/graphics/probe_kernel_bleed.py`, and
the campaign draft `docs/design/280_graphics-runtime.md`.
