> [!WARNING]
> **ARCHIVAL MATERIAL — NOT THE SOURCE OF TRUTH.** Frozen history from
> before the ratified spec; kept for archaeology only, not maintained —
> statements below may be wrong about the current system. The living
> canon is `docs/design/200_the-spec.md` (with 210/220/230/250/260).
> Index: `docs/design/history/README.md`.

# R12 — wgpu-py bridge deep-dive (WebGPU compute + render)

*Research agent survey, 2026-07-12 backend detour (pre-step-9). Local
grounding: M0 reference (`src/pdum/dsl_reference/`), wgpu-py 0.31.1 installed.
Consumed by `070_backends-notes.md`.*

Local versions: project pins `wgpu>=0.31.1` (installed 0.31.1 = latest stable,
released 2026-06-23; v0.32 unreleased), `rendercanvas 2.6.3`, `glfw`. M0
reference at `src/pdum/dsl_reference/webgpu/runtime.py` + `backends/wgsl/layout.py`.

## Q1 — Current API: compute vs render paths, shared machinery

| Machinery | Shared | Compute-only | Render-only |
|---|---|---|---|
| Adapter/device | `wgpu.gpu.request_adapter_sync()`, `adapter.request_device_sync()` (async variants return promise-likes w/ `.then()`/`.sync_wait()`) | | |
| Shader module | `create_shader_module(code=wgsl)` — one module can hold `@compute`, `@vertex`, `@fragment` entry points | | |
| Buffers | `create_buffer(size, usage, mapped_at_creation)`, `create_buffer_with_data(data, usage)` (py convenience), `queue.write_buffer`, `queue.read_buffer` (py convenience) | | |
| Bind groups | `create_bind_group_layout(entries)`, `create_bind_group(layout, entries)`, `create_pipeline_layout(bind_group_layouts)`, or `layout="auto"` (`wgpu.AutoLayoutMode.auto`) + `pipeline.get_bind_group_layout(i)` (both pipeline classes) | | |
| Pipeline | | `create_compute_pipeline(layout, compute=ProgrammableStage)` — ONE stage dict `{module, entry_point, constants}` | `create_render_pipeline(layout, vertex, fragment, primitive, depth_stencil, multisample)` — targets/formats, topology |
| Pass | `create_command_encoder()`, `queue.submit([enc.finish()])` | `begin_compute_pass()` → `set_pipeline`, `set_bind_group`, `dispatch_workgroups(x,y,z)` / `dispatch_workgroups_indirect(buf, offset)` | `begin_render_pass(color_attachments=[...])` → `set_pipeline`, `set_bind_group`, `draw(n)` |
| Output | | storage buffer + `copy_buffer_to_buffer` + `map_sync`/`read_mapped` (or `queue.read_buffer`) | texture: canvas `context.get_current_texture().create_view()` or owned texture (M0 `OffscreenTarget`) + `copy_texture_to_buffer` w/ 256-byte row alignment |
| Canvas | | | `rendercanvas` `RenderCanvas`, `canvas.get_wgpu_context()` (new name; M0 uses older `get_context("wgpu")`), `context.configure(device, format)`, `get_preferred_format` |

0.31.x additions: typed descriptor helpers (`wgpu.RenderPipelineDescriptor`,
`wgpu.VertexState`, … — plain dicts still accepted, M0-style). v0.32
(unreleased): push-constants renamed "immediates" (`var<immediate>`,
`encoder.set_immediates()`); `get_current_texture()` raises `wgpu.DrawCancelled`
on occlusion.

**Buffer discipline**: `uniform` = small, read-only, std140-like strict layout,
bound `{"buffer": {"type": "uniform"}}` — right for closure captures. `storage`
= large, `read` or `read_write`, relaxed layout (array stride = element size),
for tensor data. **Alignment** (WGSL memory layout): scalar align 4; vec2 8;
vec3 12-size/16-align; vec4 16. Uniform address space extras (the std140-like
part): struct members' align rounded up to 16, array element stride must be a
multiple of 16 — M0's `layout.py` handles scalar/vec correctly (vec3 footgun
documented there) but does NOT do arrays/nested structs, where uniform-vs-storage
rules diverge. **Fast path for per-frame uniforms**: `queue.write_buffer` —
Toji's buffer-uploads doc is unambiguous: writeBuffer is the recommended,
simplest AND high-performance route for small per-frame updates;
`mapped_at_creation` is for initialize-once data; staging-ring mapping is a
profiled-bottleneck-only escalation. M0 already does the right thing
(`Drawer.update` → `write_buffer`).

## Q2 — WGSL constraints

- **`@workgroup_size` is NOT dispatch-time — confirmed.** Parameters must be
  const-expressions OR override-expressions; override-expressions are evaluated
  exactly at `create_compute_pipeline` time via `constants: dict` in the stage
  dict (verified in local wgpu-py 0.31.1 `structs.ProgrammableStage`). Block
  size is fixed at pipeline creation at the latest — **validates the 040 §3b
  split**: block=codegen/artifact key, grid=runtime data to
  `dispatch_workgroups`. Nuance: `override wg: u32` moves block from
  shader-source key to pipeline key — one WGSL string, N pipelines. Given
  artifact = pipeline, an optional refinement (skips shader re-render+naga
  parse on block change), not a design change. Keep block in the artifact key
  either way.
- **`dispatch_workgroups_indirect(indirect_buffer, offset)`**: present in
  wgpu-py (verified locally) — grid from a GPU buffer (4×u32); future hook,
  irrelevant for step 9.
- **`var<storage, read_write>`**: core in compute + fragment stages;
  vertex-stage writable storage needs native feature `vertex-writable-storage`
  (present locally). Default `maxStorageBuffersPerShaderStage` = 8; compat-mode
  targets may give fragment 0 storage buffers — one more reason compute is the
  safer DPS output path.
- **Subgroups**: local Metal adapter exposes native features `subgroup` +
  `subgroup-barrier` (wgpu-native names; WebGPU standard name `subgroups` in
  `wgpu.FeatureName`). naga's `enable subgroups;` support has been rocky
  (gfx-rs/wgpu#7471). Usable-native-with-care, naming in flux. Do not build
  step 9 on it; a future capability declared via `Backend.code_for_op`.

## Q3 — Graphics invocation surface (per-frame flow + cost ranking)

Per-frame flow (wgpu-py triangle ≈12 lines; M0 `Drawer.update` + `show` same):
`queue.write_buffer(ubuf, 0, packed)` → `create_command_encoder` →
`target.get_view()` (window: `get_current_texture().create_view()` — fetched
fresh per frame, swapchain rotates) → `begin_render_pass(color_attachments=
[{view, load_op:"clear", store_op:"store"}])` → `set_pipeline` →
`set_bind_group(0, bg)` → `draw(3)` → `end` → `submit`.

**Cost ranking** (expensive → cheap):
1. `create_render_pipeline`/`create_compute_pipeline` — ms-scale (naga →
   MSL/SPIR-V/HLSL + driver compile). MISS-only; cache in `FastRecord.artifact`.
2. `create_shader_module` — naga parse+validate. MISS-only.
3. `create_buffer`/`create_texture` — allocation; create once, reuse while size
   holds (ResultPlan discipline).
4. `create_bind_group`(+layout) — cheapish but not free; once per (pipeline,
   buffer-set); recreate only when a bound buffer object is *replaced*.
5. `queue.write_buffer` — cheap, per-frame fast path.
6. Encoder + pass + draw + submit — cheap, inherently per-frame.

**Design implication**: the compiled fragment-kernel artifact should own
{pipeline, bind group layout, bind group, uniform buffer, PackPlan/Layout} —
exactly M0's `GpuProgram` — and per-frame do only {pack → write_buffer →
encode pass → draw}. Maps onto §2.12 `FastRecord`: `artifact` = `GpuProgram`,
`extract` packs into `staging`, `launch(staging, leaves)` = write_buffer +
encode + submit. M0 fault confirmed: M0 re-runs `flatten` every frame
(runtime.py:126) — the plan's `extract` closure kills that. Caveat: M0 bakes
`target.format` into the pipeline (runtime.py:153), so target format belongs in
the artifact/params key (`Backend.params_key`). Bind group is per-(artifact ×
uniform buffer), safe to cache in the record since the ubuf lives there too.
Offscreen tests: M0's canvas-free `OffscreenTarget` (own texture +
`copy_texture_to_buffer` + `map_sync`, 256-byte row stride) is the right CI
pattern; `rendercanvas` has an `offscreen` backend too, but no canvas at all is
simpler. Window path: `rendercanvas.glfw` + `canvas.request_draw(frame_fn)` +
`loop.run()` (M0's `Context.run`), `get_wgpu_context()` as current spelling.

## Q4 — Compute vs fragment for step 9: lead with COMPUTE

- Line costs (wgpu-py examples, excl. WGSL): compute_noop full path ≈45 LOC
  incl. readback; triangle ≈47 LOC (≈35 setup + 12/frame) *plus*
  offscreen-readback (~25 LOC, M0's `read_pixels`) *plus* fullscreen-triangle
  vertex shader + target/format plumbing when not using a live canvas.
  Runtime-side delta ≈ +40–60% for render.
- Compute is the natural DPS citizen (040 §3b: `abi.slot(out)` = storage
  buffer; grid auto-derived from output shape) — no texture, no 256-byte row
  padding, no format key, no vertex state; readback is `queue.read_buffer`. It
  exercises every §2.10–2.12 contract (PackPlan, extract, launch, artifact key
  incl. block) with the fewest moving parts, and it's the path M0 did NOT
  already prove.
- Fragment second, as a thin variant of the same Backend record: same
  shader-module/bind-group/uniform machinery, swap `compute=` stage for
  `vertex+fragment` states, `dispatch_workgroups` for
  `begin_render_pass+draw(3)`, storage-out for texture target — and port M0's
  `Drawer`/`OffscreenTarget` nearly verbatim (runtime.py is already the
  design's `launch` in embryo). Fragment is the demo-facing thesis showcase;
  compute is the contract-retiring first move. Fits the §2.10 budget (WGSL
  ≈115 renderer + 130–220 runtime lines) only if the two share one runtime
  module.

## Q5 — CI without a GPU: yes

wgpu-py's own CI runs the full suite GPU-included on `ubuntu-latest` only
(Python 3.11–3.14 + PyPy) using Mesa's software Vulkan driver
lavapipe + llvmpipe via `sudo apt install -y libegl1-mesa-dev libgl1-mesa-dri
libxcb-xfixes0-dev mesa-vulkan-drivers`, with `EXPECT_LAVAPIPE: true` asserting
the adapter. No SwiftShader. Compute and offscreen render (M0-style texture
readback) both work under lavapipe; no display server needed. For pdum.dsl CI:
same apt line, `WGPU_BACKEND_TYPE=Vulkan` if needed, gate tests with a
`request_adapter_sync` try/skip. Local macOS dev runs Metal natively.

## Sources

- https://github.com/pygfx/wgpu-py/blob/main/CHANGELOG.md
- https://github.com/pygfx/wgpu-py/blob/main/examples/compute_noop.py ; …/examples/triangle.py
- https://github.com/pygfx/wgpu-py/blob/main/.github/workflows/ci.yml
- https://wgpu-py.readthedocs.io/en/stable/generated/wgpu.GPUDevice.html
- https://toji.dev/webgpu-best-practices/buffer-uploads.html ; …/dynamic-shader-construction.html
- https://github.com/gpuweb/gpuweb/issues/1442 ; https://developer.mozilla.org/en-US/docs/Web/API/GPUDevice/createComputePipeline
- https://www.dneto.dev/posts/2025/wgsl-evaluation-phase/
- https://github.com/gfx-rs/wgpu/issues/7471 ; https://docs.rs/wgpu/latest/wgpu/struct.Features.html
- Local verification: wgpu-py 0.31.1 introspection; adapter features on Metal M3 Ultra;
  session probe: staging-as-uniform headless compute + per-frame write_buffer verified working.
