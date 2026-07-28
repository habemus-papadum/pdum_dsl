# spike_runner — findings

(Written by the session lead from the spike agent's report; the durable
per-module rationale also lives in each module's docstring. Code:
`run.sh offscreen | timing_only 512x768 | window`.)

## Headline timings (Apple GPU/Metal, wgpu 0.31.1, 200 samples, min-as-headline per 210)

| path | ms |
|---|---|
| warm frame, no readback (window path) | 0.135 |
| warm frame incl. readback (offscreen) | 1.38 |
| cold `render_wgpu` alone | 3.46 |
| cold numpy ripple + cold `render_wgpu` (today's whole frame) | 5.66 |
| numpy reference `graphics.render` | 372 |

Per-phase (64×96 / 512×768, min): update 0.050/0.052, encode
0.069/0.078, submit 0.010/0.011, readback 1.163/1.413. GPU passes:
0.007 ms compute; render below timestamp resolution at 64×96.

Three facts: (1) 210's readback claim reproduces exactly — 64× the
pixels costs 1.23× the readback, and readback is 84% of the offscreen
frame; (2) host cost is size-independent (~0.14 ms at both
resolutions); (3) `render_wgpu`'s 3.46 ms is FLATTERED by the driver
shader cache (byte-identical WGSL → cached pipeline); with distinct
source each iteration, module+pipeline costs 8.3–9.3 ms (first-ever
121 ms). The one-shot design is affordable only while nothing
structural about the shader changes.

## Copies per frame (measured by instrumenting wgpu at class level)

- **warm:** 0 uploads | 2 `write_buffer` = 12 B | 1 readback 32768 B
  (the framebuffer — that's what "offscreen" means; windowed = 0) |
  0 shader modules | 0 pipelines.
- **today:** 8 uploads = 2320 B | 3 readbacks = 33536 B | 2 shader
  modules | 2 pipelines. Including the unavoidable-today
  device→host→device round-trip: 768 B down + 768 B up for geometry
  that never needed to leave the device (rippled mesh: readback after
  compute, re-interleave, re-upload for render). theta_in uploads
  though only _out is written; four field views = four `to_numpy`
  copies of two host buffers.

## Hurt-list (each item names what the runner had to work around)

- **H1 — No PSO artifact, no lowering cache.** The kernel tier has the
  two-tier cache (kernel.py:102-107) and a real `_Artifact`
  (kernel.py:1065-1113); the graphics tier has neither. `pair()`
  memoizes only the fragment's required-field set (graphics.py:305,
  331-332). `_lower_vertex`/`_lower_fragment` re-run AST lowering on
  every render. Measured: lowering + WGSL text = 0.937 ms — 7× the
  entire warm frame. The spike invented `wgsl_gen.RenderProgram` to
  have an object to hold the result. Everything below is downstream.
- **H2 — A compiled artifact cannot be obtained without launching it.**
  `ComputeKernel._invoke` (kernel.py:169-198) keys, compiles AND
  launches; the artifact never escapes. `runner.artifact_of` rebuilds
  the key from five private names (`KERNELS`, `_code_fp`, `_env_fp`,
  `_arg_fp`, `_compile`) — if `_invoke`'s key construction drifts, the
  spike silently compiles a second artifact into the same Memo. A
  public `kernel.artifact(*args)` costs one method.
- **H3 — Staging extraction fused into execution; the pack loop exists
  three times** (graphics.py:608-621, wgsl_executor.py:475-489,
  kernel.py:1100-1111; the spike adds a 4th and 5th copy). 210 says
  "the plan IS the ABI, neither invents layout" — today every consumer
  re-derives it AND each backend separately invents the device
  representation (flat `array<f32>`) the plan does not describe. The
  missing thing is one staging-plan object both host packer and
  backend read.
- **H4 — Every device path collapses layout at the buffer boundary —
  residency is actively destroyed, not just missing.** The Tensor
  already carries what's needed (mesh field views: offset 0/8, stride
  16, ONE 384 B Buffer). Both executors discard it
  (wgsl_executor.py:260 `ascontiguousarray(to_numpy)`; :581-583
  re-interleaves records via `np.stack`), manufacturing the contiguous
  layout the shader assumes and splitting two views of one buffer into
  two unrelated device buffers. `residency.py` proves it avoidable
  with zero semantic change: index each leaf through the view's own
  layout and ONE resident buffer serves the compute field views AND
  the render record (`b0[i*2]`, `b0[i*2+1]`) — verified bit-identical.
  Caveat: host interior is f64, device is f32 — element *indices*
  survive narrowing, which is why the strided read works; no encoding
  descriptor exists for it (`HOST_ITEM=8` hardcoded). This is 210's
  "Buffer + Layout + Encoding ARE the contract", unimplemented.
- **H5 — The device singleton admits no feature requests**, so 210's
  honest timer (timestamp queries) is unreachable through
  `graphics._device()` (graphics.py:718-733 — no `required_features`,
  and features cannot be added post-creation though this adapter
  offers them). The spike creates its own device and monkeypatches
  `graphics._GPU` so textures/samplers land on the same device.

Also real: **MRT is broken on device, not just missing** —
`render_wgpu(pso[config(taps=...)], ...)` hits an AttributeError
(wgsl_executor.py:409 reaches for `pso.vs` on a `_BoundPSO`).
`_env_staging` packs whatever closure it's handed — nothing validates
a swapped closure against the pipeline it reuses (the spike's
`_check_swap` guard). The slot channel binds as `var<storage, read>`
not a uniform buffer (wgsl_executor.py:216, 523, 526), contradicting
210 — and not casually fixable: WGSL uniform address space wants
stride-16 arrays, which the plan should emit. `_ARG_BINDINGS`
(kernel.py:108) is a process-global rebind channel two in-flight PSOs
would clobber. `Buffer.device` is a string label with no registry.
And `_lower_fragment`'s lattice-shaped `varying_types` is a FALSE
DEPENDENCY on resolution — a dummy (2,2) tensor yields bit-identical
output at every resolution; it is the only reason `render_wgpu` takes
`shape` at all.

## Verification

All 8 frames (angle AND ripple phase varying) **bit-identical** to
`render_wgpu` (max diff exactly 0.0) — one pipeline served all 8
angles and a 4×-larger attachment with no recompile; the
residency-strided read returns exactly what repack-and-upload returns.
Against the numpy reference at the conformance atol=2e-3: 6/8 frames
have zero pixels over tolerance; frames 1 and 7 put 2 of 6144 pixels
at 5.1e-3 — and `render_wgpu` misses the SAME pixels by the SAME
amount (steepest point of the analytic-AA ramp; coverage sets agree
exactly). So the conformance golden's atol is calibrated to its one
angle (1.2), not to the scene: a frame-loop conformance test sweeping
angles needs a scene-independent tolerance or a ramp-pixel exclusion.

## Encodable sketch (from what the build actually needed)

```python
res  = Residency(device)
prog = compile_pso(pso, inputs=(rippled,), args=(g,),
                   target=Target("bgra8unorm-srgb"), residency=res)  # once
prog.bind(angle=0.9)      # per frame: slot channel only; refuses a layout mismatch
prog.encode(render_pass)  # per frame: draw into a pass the HOST began
```

- Artifact key = (pso, input layouts, **target format**) — color0 is a
  scalar and the 1→4 channel surface expansion is baked into the WGSL,
  so offscreen r32float and swap-chain bgra8unorm-srgb are genuinely
  different artifacts. **Resolution must NOT key** (proven above).
- `encode()` takes a PASS, not an encoder — the spike's
  `encode(enc, view, clear=...)` variant is convenient and wrong: the
  host then can't put two PSOs in one pass, and the load/clear op (a
  frame-level decision) lands inside the per-draw object. 210's
  "render bundle / draw-into-pass" means `encode(pass)` is the
  primitive; `encode_frame(encoder, view, clear=...)` is a wrapper.
- `bind()` separate from `encode()` — slot refresh touches the queue,
  encode touches only the encoder; that separation is what let
  timestamp writes go in with no second code path (210's shared
  `_encode_frame`).
- Residency is passed to COMPILATION, not encode — binding indices and
  strided index expressions are baked into the WGSL, so two stages
  sharing a buffer must compile against the same registry. That IS the
  zero-copy mechanism.

## Not answered (scope, not blockers)

Multiple draws per pass / real render bundles; depth (the reference
has none, the far side overdraws — visible in `hero_256x384.png` — and
the PSO has no place to declare one); textures across a frame loop
(`upload()` creates a texture per call, no residency — expect H4
again); async readback / persistent surfaces. The windowed path builds
(bgra8unorm-srgb pipeline verified, shares `FrameRunner.encode` with
offscreen) but needs a human to run `./run.sh window` to confirm
visually.
