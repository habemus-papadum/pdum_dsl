# 070 — Backends: bridges, invocation, and the shader family

*Synthesis of the 2026-07-12 backend research detour (six-agent fan-out,
pre-step-9). Raw surveys: research/R12 (wgpu-py), R13 (CUDA ride-vs-own),
R14 (Metal ride-vs-own), R15 (shader-family matrix), R16 (invocation &
runtime), R17 (packaging/demos/CI). Local evidence: the ch08 staging ABI
verified driving a real WGSL uniform block, headless, on the M3 Ultra's
Metal backend (session probe, 2026-07-12). Companion to 010 §2.10–§2.12,
§4; extends 040 §3b/§3c. Decisions below are DESIGN-COMMITTED pending the
owner walkthrough; deviations from 010 are ledgered there.*

## 1. The roster and the taxonomy

Targets: WebGPU compute · WebGPU vertex+fragment · CUDA · Metal compute ·
a pure-Python reference twin per family (the ch09 backend is the first).

Machinery layers (R15 verdict — the two-tier hypothesis holds, with
fragment forced out as its own family by CUDA's missing raster stage):

```
shader-core        types (vecN, matN, scalars), ctors, READ-swizzles,
                   numeric policy (div/mod, overflow, NaN stance), f16 flag
  └─ compute       launch model, thread coords, workgroup memory, barriers,
     family        atomics (i32/u32 relaxed set = the 3-way intersection);
                   capability flags INSIDE the family: subgroups, i64,
                   float/64-bit atomics
  └─ fragment      position/front_facing/frag_depth, discard, derivatives,
     family        texture sampling, vertex I/O (WGSL/MSL only)
per-target packs   spelling tables + genuinely target-only features
                   (CUDA masks/scopes, MSL write-swizzles/quad ops,
                   WGSL override binding)
```

tinygrad's renderers (mostly per-target *string tables* over one IR), Warp's
single `wp.*` namespace over CUDA+CPU, and naga/tint's polyfill catalogs are
the prior-art proof this factoring works. Slang's capability atoms are the
mature endpoint for the flags.

**Numeric policy is legislated in shader-core, once** (R15 traps 1–3, 5):

- Integer `/` and `%` are TRUNCATING with sign-of-dividend `%` (all three
  GPU targets); the Python twin must never use native `//`/`%` (floor!).
- Div-by-zero / INT_MIN÷−1: WGSL defines (`x/0==x`, `x%0==0`), C++ targets
  are UB. POLICY: the twin RAISES (strict-core spirit: loud beats defined-
  but-surprising); GPU targets are documented framework-UB; a naga-style
  guard-polyfill is a per-target opt-in pack, not core.
- NaN: MSL fast-math is on by default, WGSL may assume NaN absent, CUDA is
  IEEE. POLICY: NaN-as-data is unsupported in the core language; targets
  keep their fast-math defaults; a `precise` capability flag can arrive with
  a real consumer.
- Signed overflow: WGSL wraps, C++ UB → twin masks to 32 bits; core says
  "wraps".
- Subgroup ops (when the capability lands): maskless, width-agnostic,
  `subgroup_size` exposed, documented non-uniform-control-flow-unsafe.

## 2. Bridge decisions: own both GPU stacks

**WebGPU — wgpu-py, settled** (R12). Compute and render share module/
buffer/bind-group machinery; `@workgroup_size` is pipeline-creation-time,
independently confirming the 040 §3b block=codegen-key split; per-frame fast
path is `queue.write_buffer` (M0 already did this right); lavapipe gives
full GPU-less CI (wgpu-py's own CI pattern).

**CUDA — OWN, via cuda.core** (R13). Decisive facts: `cuda.core` went 1.0
stable (SemVer, NVIDIA-backed; pin `>=1.1,<2`); its compile cache is
strictly OPT-IN so our two tiers stay the only caching authority; and
`cuLaunchKernel`'s `kernelParams` accepts a `void**` — since
`FastRecord.staging` never moves, a **pointer table into staging can be
built once per record**, making CUDA the purest realization of
`launch(staging, leaves)` anywhere. Riding CuPy would re-marshal every arg
per call (its `_pointer()` path), tax the 5 µs hit budget, and add a 134 MB
CUDA-major-named dependency; its one great layer (content-addressed disk
cubin cache) is ~40 lines under our planned disk tier. Prior art is
unanimous (numba-cuda migrated onto these bindings; Warp owns the same
NVRTC→cache→launch stack). CuPy remains a *documented fallback* runtime and
an optional allocator — never a required dependency. Allocation is
pluggable: user's torch/cupy arrays via DLPack, else
`cuda.core` `DeviceMemoryResource`.

**Metal — OWN, via ctypes-objc (or PyObjC to start)** (R14). Decisive
facts: MLX's kernel cache is **name-keyed and never re-checks source** —
incompatible with a content-addressed artifact tier; its lazy `mx.eval`
per call blows the hit budget; and `setBytes` is a *literal* realization of
our staging buffer, `setBuffer` of our leaves channel. tinygrad's 240+79
line Metal runtime is the existence proof inside our 130–220 budget;
`MTLBinaryArchive` even lets the disk tier persist compiled pipelines —
impossible when riding. MLX stays in reserve as a thin optional satellite
backend for users already in its lazy world (§14's "target it" posture).

The symmetry is the finding: both rides fail for the SAME three reasons —
caching we don't control, marshaling that undoes PackPlan, and a scheduler
between us and the queue. The framework's whole thesis is owning exactly
those three things.

## 3. Invocation surface (R16; extends 040 §3b/§3c)

```python
out = kernel[{"grid": g, "block": 64}](x, out=buf)   # explicit DPS; returns the destination(s)
a, b = kernel[cfg](x, out=(a, b))                    # N destinations; returned in out order
a, b = kernel[cfg](x)                                # ONLY with a registered out-shape rule; else loud
(b1, b2) > step1[c1] | step2[c2] | step3[c3]         # orchestrated ping-pong
```

- **Outputs are explicit, ufunc-discipline** (return the destinations —
  Warp's return-None kills chaining; CuPy ElementwiseKernel/NumPy `out=`
  is the precedent). NO output-shape inference from kernel bodies, ever
  (owner decision; no surveyed system does it). The middle path is an
  **out-shape rule column**: `out_shape(arg_types, arg_shapes, config) →
  shaped Types`, registered per kernel or role — registration, not
  analysis — feeding the existing ResultPlan seam; destinations reused
  while shapes hold. `out=` always wins over the rule. **ResultPlan already
  supports tuple results (ch08); multi-BUFFER destinations are the
  remaining wiring, confirmed as pure plan-vocabulary work.**
- **One config schema**, owned by the compute role, positional order
  `[grid, block, smem, stream]` (named via dict per the bracket contract),
  each component through the §3c strip→value→type pipeline:
  `grid` strips (runtime, rides the leaves channel); `block`
  value-specializes (artifact key — WGSL *forces* this; CUDA/Metal choose it
  for consistency, a CUDA-only kernel may demote it per §3c override);
  `smem` strips on CUDA/Metal and is REFUSED on WGSL (no dynamic form —
  model as a specialization constant when needed); `stream` strips and
  never keys anything (absent ⇒ bridge-ambient; wgpu has exactly one
  queue). A registered grid rule (`ceil_div(domain, block)`, Triton
  precedent) makes `grid` omittable.
- **Chaining** compiles the reserved `orchestrate` composition tag into an
  **encode plan**: all stages under one ordering token (one command
  encoder / stream / command buffer — every target guarantees as-if-serial
  ordering within it; barriers are never the user's concern). Buffer-pair
  application + uniform edges ⇒ swap schedule with parity owned by the
  plan; `>` returns `(result, scratch)` so re-application continues the
  ping-pong. Non-uniform edges: pair refused loudly; single-input form
  allocates per-edge intermediates via out-shape rules, owned by the
  pipeline's cache entry. Intermediates never touch the host; `>` yields
  DeviceValues, host bytes only at a materializer (`| collect`).
- **Streams** are launcher data (leaves channel), not identity. DLPack's
  `__dlpack__(stream=...)` producer-syncs contract is implemented at the
  DeviceValue/materializer boundary only.

## 4. The graphics surface (R12 + M0)

A compiled fragment kernel's artifact owns {render pipeline, bind group
layout, bind group, uniform buffer, plan} — M0's `GpuProgram`, now living
in `FastRecord.artifact`. Per frame: `shader(args)` = extract → pack →
`queue.write_buffer` (the ONLY per-frame uniform path), then
`draw(target)` = fresh swapchain view → render pass → set pipeline/bind
group → `draw(3)` → submit. Pipeline and shader-module creation are
ms-scale and miss-only; bind groups are cached in the record; target
FORMAT enters `Backend.params_key` (M0 baked it into the pipeline —
correct, now key-visible). Offscreen tests use M0's canvas-free
`OffscreenTarget`; the window path is rendercanvas+glfw as in M0.

## 5. Step ordering and plan deltas

- **Step 9 (WGSL, M1) leads with COMPUTE**, fragment follows as a thin
  variant in the same step sharing one runtime module (R12: render adds
  +40–60% runtime surface; compute exercises every marshaling contract
  with the fewest moving parts and is the path M0 did NOT prove; the local
  probe already ran the ABI end-to-end). M1's demo payoff (the disk in a
  window) still ships in step 9 via the fragment variant — the vertical
  slice is unchanged, just built compute-first.
- **Step 14 is revised**: "CuPy RawKernel + MLX custom-kernel backends" →
  **own runtimes: `backends/cuda/` on cuda.core, `backends/metal/` on
  ctypes-objc/PyObjC**, with cupy/MLX as optional thin satellite
  runtimes/allocators. 010 §5's module-map line to be read accordingly.
- **CUDA development is design-for-skip + cloud burst** (R17): codegen
  tested hardware-free via golden-source snapshots; execution tests
  hardware-marked; Modal label-triggered bursts (~$0.02/run) validate the
  last mile. No emulator in the loop.

## 6. Packaging, demos, CI (R17)

- Extras: `[webgpu]`, `[metal]`, `[cuda]` — core `dependencies` shrink to
  pure Python; wgpu moves into `[webgpu]`. The own-CUDA verdict shrinks
  `[cuda]` to `cuda-core`+`cuda-bindings`+NVRTC wheel (largely dissolving
  the cupy per-CUDA-major extras mess). mlx (if/when the satellite lands)
  marker-gated to darwin/arm64. Dev default: `uv sync --all-extras`
  (markers make it safe everywhere); one universal lockfile.
- Examples: flat `examples/<backend>/*.py` — plain, self-contained,
  probe-gated scripts (wgpu-py's shape); `tests/test_examples.py` runs each
  as a subprocess with per-directory markers. No uv workspace. PEP 723
  headers added post-release for scripts that leave the repo; inert in-repo.
- Gating: `pdum.dsl.testing` capability probes (cached), pytest markers +
  `PDUM_REQUIRE_<BACKEND>` fail-instead-of-skip in CI (wgpu-py's
  EXPECT_LAVAPIPE idiom — anti-rot), notebook `gpu` cell tag via nbclient's
  `skip_cells_with_tag` (GPU outputs baked locally on the M3, CI leaves
  them untouched).
- CI matrix: ubuntu lint+unit; ubuntu webgpu-on-lavapipe (REQUIRE'd);
  macos-15 arm64 Metal probe-first (tinygrad precedent; promote to
  REQUIRE'd when observed green); CUDA skipped on hosted runners, Modal
  on-demand via PR label.

## 7. Open questions (deliberately parked)

- Dynamic shared memory as a WGSL specialization constant (pipeline-keyed)
  — implement when the first shared-memory kernel lands, not before.
- Subgroups capability flag timing (wgpu native support present locally;
  naga's `enable` path still rocky).
- `repeat(step, n)` time-iteration combinator over the ping-pong plan.
- MLX satellite backend (and its custom_vjp hook) — after M2 transforms.
