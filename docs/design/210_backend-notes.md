# 210 — Backend notes: distilled knowledge for the L4-era builders

**Status: distillation (200 §3.3).** The device backends (C, WebGPU/WGSL)
were deleted at migration P1 — deliberately, with their learnings recorded
here. Whoever builds the fresh L4-era backends (CUDA, Metal, WebGPU) starts
from this page and the two reference executors, not from old code.

## Numeric policy (enforced on BOTH sides of every differential test)

- **Floats are IEEE NON-TRAPPING** (amended at P8, the numpy-authority
  ruling): the reference computes floats on numpy scalars, so 0/0 flows as
  nan and sqrt(−1) is nan — like a device, never a Python exception. The
  pre-amendment oracle trapped (ZeroDivisionError at a pole a GPU would
  sail through); tests that dodged poles are un-dodged.
- **Integer division/modulo are TRUNCATING** (C semantics), never Python's
  floored `//`/`%`. The reference twin computes them with exact integer
  helpers — routing through float division loses exactness past 2^53 (a
  review-caught silent-wrongness bug):
  `_tdiv(a,b) = q+1 if (q:=a//b) < 0 and q*b != a else q`;
  `_tmod(a,b) = a - _tdiv(a,b)*b`. These live in the reference evaluator's
  preamble today; every device backend must match them bit-for-bit.
- **Float modulo is `fmod`** (sign of the dividend) on both sides — C's `%`
  does not compile for doubles, Python's `%` is floored; both were caught by
  tests, neither is the policy. The reference spells it `np.fmod`.
- **u64 constants refuse** on targets whose literal range cannot carry them;
  inf/nan constants refuse at rendering (`repr(inf)` is not a literal — the
  reference spells `float('inf')` explicitly where a type allows it at all).
- f32 computes as f64 on the reference (Python has one float); narrowing
  becomes real per-device via a declared type map — never silently.

## The artifact carries its own contract

The grid-family bug class (150 F-series: dtype mismatch writing 8N bytes
into 4N; non-contiguous adoption clobbering neighbors; rank mismatch reading
past `dom[]`) had ONE root fix: **the compiled artifact carries its
input/output contract** (element kind, rank — as a header/metadata the
launcher parses), and **the launcher enforces it** (dtype, contiguity, rank
refusals) before any pointer crosses the ABI. Rebuild this pattern in every
device backend: contracts on artifacts, enforcement at launch, refusals that
quote the fix. The 200-era version generalizes it: boundary descriptors
(Buffer + Layout + Encoding) ARE the contract.

*Where a kernel executor plugs in (P8.5).* The kernel tier is two-tier
cached like the dsl tier: the specialization tier holds the launch
plan, and the content tier keys `(region.key, executor fp)` — today's
one executor is the numpy interpreter (`_executor` in `pdum.tl.kernel`
closes over the region). The P8 WebGPU conformance executor replaces
that body with render + compile behind the SAME content key and brings
`_EXECUTOR_FP` its second value; the per-launch scalar channel is
already physical (`abi.slot` offsets/fmts packed into staging bytes by
the launcher), so a device backend consumes it as a uniform buffer
without a representation change.

## WebGPU runtime learnings (measured on M3, step 9–10 era)

- **Synchronous readback is a fixed-latency protocol act, not bandwidth**:
  ~1.6 ms from 64² to 1024² — the submit→wait→map round-trip dominates and
  does not scale with size. Async/persistent-surface paths are where that
  cost dies; never benchmark a compute path through a sync readback and
  attribute the time to the kernel.
- **Timestamp queries** (begin/end-of-pass) are the honest GPU timer;
  request the feature at device creation when available; clamp tick deltas
  at ≥0 (drivers may report non-monotonic pass timestamps); cache the query
  set/buffer on the program object.
- **Encode and submit are separate acts** — one `_encode_frame` shared
  between the timed and untimed paths so they cannot drift, and the
  *encodable* is the API surface (the host owns passes and submits — the
  200 graphics tier's deliverable is a render bundle / draw-into-pass).
- **Uniform-buffer plan**: staging members are slot-format-typed (f32/i32/
  u32; bool reads `!= 0u` — bool is not host-shareable), members FROM the
  plan (hole-free), reserved words prefixed. The plan IS the ABI; both
  renderer and launcher read it, neither invents layout.
- **Thread sizing (threads per group) is a SPECIALIZATION parameter**
  (owner-ruled 2026-07-28, generalizing the earlier WebGPU-shaped rule):
  it keys the artifact and a change recompiles, whether or not the
  target embeds it in source (WGSL: shader text; Metal: a
  `dispatchThreadgroups:` argument appearing nowhere in source; CUDA:
  blockDim, with launch bounds making it compile-relevant). **Block
  sizing (group count) is launcher data everywhere** and never
  specializes. Consequence: a backend cannot return only source text —
  it returns (source, launch-contract), the contract carrying the
  thread size as data.

## Metal runtime learnings (measured on M3 Ultra, the graphics-campaign spikes, 2026-07)

Evidence: rerunnable spikes on the `worktree-graphics-design` branch
(`explorations/graphics/spike_metal/` and siblings, a FINDINGS.md in
each). Differentials there are three-way: f64 reference, WGSL/wgpu,
MSL/Metal — the two device columns bitwise-equal on every translatable
subject (caveat recorded there: wgpu reaches Metal via Naga on that
machine, so cross-vendor exactness is untested until CUDA/Vulkan).

- **PyObjC is adequate for a Metal runtime** — no Swift/ObjC shim;
  selector mangling is mechanical; reading buffer contents and adopting
  host memory work directly on Python buffer objects. **JIT MSL needs
  no Xcode**: `newLibraryWithSource:` compiles at runtime with Command
  Line Tools only.
- **Unified-memory adoption is real and bidirectional**:
  `newBufferWithBytesNoCopy:` over a page-aligned host allocation —
  kernel stores appear in the host array with NO readback call
  (`waitUntilCompleted` is synchronization, not transfer); host writes
  after buffer creation are seen by the next dispatch with no upload;
  `MTLBuffer.contents()` IS the host pointer. The runtime interface
  must carry "no transfer exists" as a first-class case, and
  `Buffer.device`-as-string cannot answer "which device is this on"
  when memory is simultaneously host and device.
- **The bounds guard is a launch-contract clause, not universal
  source.** WebGPU dispatches whole workgroups (the overhang guard is
  required); Metal's `dispatchThreads:` launches exact grids and the
  guard is dead code (guard-vs-exact proven bitwise equal); CUDA
  dispatches whole blocks — WebGPU-shaped. Emit the guard per-runtime,
  from the contract.
- **OOB models differ structurally (recorded, deliberately untested):**
  WebGPU implementations bounds-clamp storage accesses per the spec's
  safety model; Metal raw `device` pointers are genuine UB; CUDA sits
  on the Metal side. The keying-ladder ruling (reference certifies
  in-bounds before device runs) covers all of them — never lean on a
  device-side safety net, and record the per-target model in the
  descent-license language BOUNDARIES reserves for device-tier OOB.
- **GPU timing is free on Metal** (`GPUStartTime`/`GPUEndTime` on the
  command buffer — no query set, no feature request), unlike WebGPU's
  timestamp-query feature that must be requested at device creation —
  so device acquisition must accept feature requests. Rig gotchas,
  inherited: the last timestamp pair in a batch can resolve to zero
  (encode a discard rep); the first program timed in a process reads
  ~1.7× high (sustained warm-up, then minimum-of-samples); WebGPU caps
  `max_compute_workgroups_per_dimension` at 65535 where Metal has no
  such limit.
- **Op rows are near-universal across C-family shading languages**:
  100% of emitted WGSL statements convert to MSL under three lexical
  rules (cast spelling, float-literal suffix, declaration form); no
  operator/builtin row differs, `select` argument order included.
  Expect similar for CUDA C — the expression emitter and marker tables
  should exist once, not per backend (they exist three times today).
- **The binding table is contract DATA, never implicit in text.**
  WebGPU's `layout="auto"` prunes unused bindings (a runtime rule the
  backend must anticipate in source); Metal has no bind-group object
  (buffers set by index, unused arguments harmless); CUDA passes
  pointers. Implicit binding-in-text does not survive three targets.
- **The staging plan needs a device-representation clause.** The plan
  describes HOST staging bytes only; every backend so far has
  independently invented "uniforms arrive as a flat f32 array" — and
  each silently narrows i64 slots to f32, which the numeric policy
  above forbids. Fix once, at the shared tier; a true WGSL uniform
  buffer additionally wants stride-16 emission from the plan.
- **Backend identity must enter the content key BEFORE a second device
  backend ships.** The WGSL and MSL artifacts of one kernel share an
  identical region key: keyed today they would collide, and un-keyed
  they recompile on every executor swap (8–9 ms per distinct pipeline
  once the driver's shader cache stops flattering byte-identical
  source).
- **The target numeric contract is a category with no owner yet.**
  Metal's f32 `tanh` returns NaN for |x| ≥ 44.3614 (exactly
  `log(FLT_MAX)/2` — an `exp(2x)` implementation); `MTLMathModeSafe`
  does not fix it; `tanh(clamp(x, -20, 20))` is bitwise-identical
  wherever both are finite, so the fix row is free and provably so.
  `exp`, `sinh`/`cosh`, `pow` are the usual company — survey them on
  every new target. Per-target math rows and their freeness proofs
  need a declared home; note the conformance battery cannot see this
  class today (inputs too small to reach any math edge, and no
  `where`/select subject exists) — the adversarial-input-families
  doctrine applies to the battery itself.
- **Measurement warning for the 270 thesis**: through today's
  `_Artifact.launch`, host repack (`Tensor.to_numpy`, a per-element
  reference-tier materializer) is 88–100% of launch cost on every
  backend — the device's share is under one part in a thousand. Until
  that path is fixed or routed around, a transform-run-measure loop
  over device executors measures the repack, not the program.

## Instrumentation methodology (bench, deleted with its demo consumers)

BenchmarkTools-style adaptive micro-benchmarking: warmup, tune
evals-per-sample above a minimum-resolution floor, sample to a time budget,
**minimum as the headline estimator** (noise is one-sided). Phase
decomposition by SEAM-WRAPPING (FastRecord.extract/.launch are plain fields
— instruments are temporary shims restored in `finally`), never by editing
the dispatch path. Wall-clock CI gates are retry-once shaped: a real
regression fails twice; a scheduler blip does not.

## The aliasing lesson (carried as a day-one test at P7)

A writable output overlapping a readable capture/argument is **silent
corruption** (verified by execution, twice, in the 150 review). The store
seam refuses overlap at dispatch (`shares_memory` over the leaves) with the
ping-pong message; in-place returns only ever as an L2-certified rewrite.
This is a test to write the day the store path exists — not a memory.

## The refusal voice (seeded as the joint battery at P3)

One shape: **what happened, the principle violated, the quoted fix, the
source location.** Refusal messages are frozen behavior (the refusal
contract battery pins them by wording); a drifted message is an API break.
The oracle rule rides with it: per-element host dispatch is
debug/oracle-grade; reference execution is always spelled
(`reference(f)(...)`); a plain call on an unrouted kind refuses — it never
silently interprets.
