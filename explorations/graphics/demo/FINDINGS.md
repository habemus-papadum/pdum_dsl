# demo — findings

(Written by the session lead from the build agent's report; verified by
rerunning `./run.sh check` — all numbers below reproduced. Commands:
`./run.sh` (WebGPU window) · `./run.sh metal` (Metal window) ·
`./run.sh check [metal|webgpu]` (headless, all verifications, ~25 s) ·
`./run.sh dump` (adds the generated WGSL and MSL). Dependency added to
the shared venv: `pyobjc-framework-Quartz` 12.2.1, additive only.)

## What it is

One program — the rippled cylinder with five mouse-driven scalars
(rotation, ripple phase+depth, a clip-space spotlight) — running
windowed or headless on BOTH backends behind the 282 §7 ruled
selection spelling (empty `Generator`/`Runtime` dataclasses, a `Pair`,
prefab `webgpu`/`metal` instances, `--on` selects). All five scalars
ride the captured-scalar slot channel: a frame is
refresh-28-bytes → encode → present. `--offscreen` ASSERTS
compile-once (1 lowering for the whole run, 2 pipelines per backend,
zero rebuilds across an animated run INCLUDING a 9× resolution
change).

## Verification results

- **vs the numpy reference** (8 frames, atol 2e-3): both backends
  worst max|err| 3.89e-02 concentrated in **2 px / 49 152** — pixels
  whose local reference slope is 0.034, i.e. the disagreement equals
  the image's own step to its neighbour (AA-ramp pixels, not drift).
  `render_wgpu` misses the same class identically. Coverage sets agree
  exactly on every frame.
- **Cross-backend: bitwise identical, 8/8 frames** — now including
  RASTERIZATION, not just compute (caveat as before: wgpu reaches
  Metal via Naga on this machine; this proves our two translators are
  arithmetically equivalent, not cross-vendor portability).
- **This demo vs `render_wgpu`: bit-exact.** Near-miss worth
  recording: an early comparison ran the ripple on the HOST and
  differed by 1.8e-07 — f64 `sin` narrowed to f32 is not f32 `sin` —
  an easy way to build a "device golden" that is quietly a host
  golden. The comparison now runs its compute on device.

## Row-portability extends to the render stage

One expression generator serves both languages (`program.py::_Gen`
parameterized by a 4-field `Dialect`, no per-dialect branches);
`rowdiff.py` independently re-derives MSL from WGSL by lexical rule:
**91/91 emitted statements (100%)** — compute 13/13 reproducing
spike_metal, vertex+fragment 78/78 — under FOUR rules now: the
original three plus **R4, vector type spelling** (`vec4<f32>` →
`float4`), firing exactly once (clip-space position; compute is
scalar-only, which is why the spike never saw it).

Everything else is SHELL, structural not lexical (S1–S4 in
msl_glue.py):

- **S1 — the binding table must be PER-STAGE.** WGSL: one bind group,
  one index space shared by both stages. Metal: no bind-group object;
  `setVertexBuffer:atIndex:` and `setFragmentBuffer:atIndex:` index
  two separate tables — the same logical resource has two different
  indices depending on the reading stage (the fragment slot buffer is
  `@binding(2)` in WGSL, `[[buffer(0)]]` in MSL). spike_metal FAIL-3,
  strictly worse in the render stage: a backend interface returning
  the binding table as data must return it per stage.
- **S2 — the varying struct carries its interface differently**:
  `@location(i)` + `@interpolate(flat)` vs declaration-order +
  `[[flat]]`. (Struct members look like statements and are shell —
  the residue that made the first row count messy.)
- **S3 — stage is a return-type qualifier in MSL** (`vertex VOut
  vs_main`), and a single color target is implicit in the return type
  where WGSL writes `@location(0)`.
- **S4 — vertex pulling needs no vertex descriptor on either target**:
  `vertexDescriptor` stays nil exactly as `vertex={"buffers": []}`
  stays empty; the strided residency read is spelled identically
  modulo R1. No repack, no struct declaration needed (flat-f32 form).
- Third structural axis (confirmed): the scalar color0 → surface
  expansion means `r32float` and `bgra8unorm-srgb` are genuinely two
  compiled artifacts of one program on BOTH targets (`_SURFACE`
  tables); resolution keys nothing (verified at three sizes from one
  pipeline).

## Windowing gotchas (all commented in metal_window.py / wgsl_glue.py)

- **`backingScaleFactor()` returns 1.0 before the window is on
  screen** — read once at setup, you get a silent half-resolution
  drawable on Retina. Read per frame (also handles display moves).
- The CAMetalLayer must be assigned BEFORE `setWantsLayer_(True)`
  (layer-hosting vs layer-backed); the NSTimer must ride
  `NSRunLoopCommonModes` or frames freeze during title-bar drags and
  resizes; `drawableSize` is pixels, `bounds` is points;
  `nextDrawable()` may return nil and is re-requested per frame.
- The Metal loop currently calls `waitUntilCompleted` per frame
  because slot buffers are written IN PLACE on unified memory (no
  staging — the uniform IS host memory) and an in-flight frame could
  see torn uniforms. A real presenter wants an in-flight semaphore
  and rotating slot buffers — **a scheduling concern the encodable
  sketch has no seat for**; it's the one thing the windowed path
  needs that offscreen does not.
- WebGPU windowed path smoke-tested via rendercanvas' offscreen
  backend (same canvas API, synthetic pointer event moved the
  cursor); Metal path verified through a real CAMetalLayer drawable
  with a real encoded frame (status Completed). Only `app.run()` /
  actual glfw windows need human eyes.

## Encodable-shape consequences (feeds 280 §C)

- Confirms: compile-once/bind/encode survives a second, structurally
  different API unchanged; artifact key = (program, input layouts,
  target format), never resolution; residency is a COMPILE-time input
  and is target-neutral (the only device call in it is buffer
  creation, injected as a callback); bind ≠ encode pays off
  differently per target (queue write vs direct host-memory write) —
  same seam, invisible to the caller.
- **Contradicts: `encode(pass)` cannot be the portable primitive.**
  WebGPU begins a pass and hands you a pass object; Metal needs the
  full descriptor (attachment, load/store, clear) BEFORE an encoder
  exists. The portable primitive is
  `encode(command_buffer_or_encoder, target, clear=...)` — target and
  load-op are inseparable from pass creation on Metal.
- The `Pair` needs to carry its glue (today a dict keyed by pair —
  exactly the FAIL-5 hole restated at the selection tier).

## The unlooked-for finding: capture fingerprints discriminate `float` from `np.float64`

A cursor derived with `np.cos` makes the captured angle a
`numpy.float64`; the environment fingerprint changes
(`('angle','data','float')` → `(...,'float64')`) and the frame loop
SILENTLY recompiles the pipeline every frame — nothing in the library
warns; the only symptom is that the loop stops being warm. Caught
only by the spike-invented closure-swap guard (`P.check_swap`,
ported from spike_runner). For mouse-driven apps this is a
near-certain trap (toolkits hand you coordinates; numpy is the
natural way to derive from them). **Candidate ruling for cleanup: the
closure-swap/warmth guard is library-tier, not spike-tier.** The demo
coerces in `Mouse.set` with the reason in its docstring.
