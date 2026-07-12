# R17 — Packaging, demos, CI for optional GPU backends

*Research agent survey, 2026-07-12 backend detour. Consumed by
`070_backends-notes.md`.*

## 1. pyproject sketch (extras + markers)

Move `wgpu/glfw/rendercanvas` out of core `dependencies` (core = pure-Python
kernel; backends optional). Extras ship in package metadata so users get
`pdum-dsl[webgpu]`; keep `[dependency-groups]` strictly for dev tooling (uv:
extras are published, groups are local-only).

```toml
[project]
dependencies = []  # kernel stays pure

[project.optional-dependencies]
webgpu = ["wgpu>=0.31.1", "rendercanvas>=2.6.3", "glfw>=2.10.0"]
metal  = ["mlx>=0.32; sys_platform == 'darwin' and platform_machine == 'arm64'"]
# CuPy: distribution name encodes CUDA major; an extra cannot autodetect it.
# spaCy precedent: one extra per CUDA line. No macOS wheels -> marker-gate.
cuda12 = ["cupy-cuda12x>=13; sys_platform != 'darwin'"]
cuda13 = ["cupy-cuda13x>=13; sys_platform != 'darwin'"]
cuda   = ["habemus-papadum-dsl[cuda12]"]           # documented default
all    = ["habemus-papadum-dsl[webgpu,metal,cuda]"]
```

- The cupy problem is real and unsolved upstream: `cupy-cuda12x`/`-cuda13x`
  are separate distributions that conflict if co-installed. spaCy's answer:
  per-version extras. Torch's: per-variant indexes + `tool.uv.conflicts`.
  Declaring the cuda12/cuda13 pair conflicting is cheap insurance.
  Auto-variant selection ("wheel variants"/WheelNext) is coming, not usable
  yet (uv#16522). NOTE: R13's own-verdict shrinks the cuda extra to
  `cuda-core`+`cuda-bindings`+`nvidia-cuda-nvrtc` wheels — pip-installable,
  driver-only, and NOT CUDA-major-named the same way; the per-major extras
  problem then mostly evaporates.
- `uv sync --all-extras` on the M3 Ultra: markers make it safe — mlx
  resolves, cupy lines resolve to nothing on darwin. `uv lock` is universal:
  one lockfile covers linux-CUDA too. Dev default: `uv sync --all-extras`, so
  a single `.venv` powers VS Code everywhere.

## 2. examples/ layout + PEP 723 verdict

**Verdict: PEP 723 works but should not be the primary in-repo mechanism.**
uv supports it fully — `uv run examples/foo.py` builds an isolated env from
the `# /// script` block, ignoring project deps; `[tool.uv.sources]` inside
script metadata can point at the local checkout editable. Costs: one
ephemeral env per script; VS Code/Pylance resolves against the workspace
interpreter (intellisense breaks unless the project venv also has the deps);
in-repo it duplicates what `--all-extras` gives. It IS the right tool for
snippets meant to LEAVE the repo — add 723 headers post-PyPI-release pinning
`pdum-dsl[webgpu]` from the index; headers are inert when the harness runs
`python examples/foo.py`.

**Recommended layout** (wgpu-py-shaped, no uv workspace):
```
examples/
  README.md
  webgpu/disk.py          # plain scripts, self-contained, probe-gated
  cuda/saxpy.py
  metal/mandelbrot.py
tests/test_examples.py    # discovers examples/**, runs each as subprocess,
                          # pytest.mark.<backend> per directory
```
Each script opens with the capability probe and exits 0 with a message when
hardware is absent — testable-but-platform-gated with zero infrastructure.
Precedents: wgpu-py = flat examples/ of standalone scripts + examples/tests/
running them; warp ships examples inside the package (`python -m
warp.examples...`) — only worth it for importable-from-wheel examples; mlx
keeps examples in a separate repo (don't). Single VS Code workspace + one
venv with all extras does the right thing.

## 3. Gating idioms

```python
# src/pdum/dsl/testing.py  (shared by pytest, notebooks, examples)
from functools import cache

@cache
def has_webgpu():
    try:
        import wgpu
        return wgpu.gpu.request_adapter_sync() is not None
    except Exception:
        return False

@cache
def has_cuda():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False

@cache
def has_metal():
    try:
        import mlx.core as mx
        return mx.metal.is_available()
    except Exception:
        return False
```
```python
# conftest.py — one suite, green anywhere, exercises whatever exists
PROBES = {"webgpu": has_webgpu, "cuda": has_cuda, "metal": has_metal}

def pytest_runtest_setup(item):
    for name, probe in PROBES.items():
        if name in item.keywords and not probe():
            if os.environ.get(f"PDUM_REQUIRE_{name.upper()}"):
                pytest.fail(f"CI expects {name} but probe failed")  # anti-rot
            pytest.skip(f"no {name} device")
```
Register markers in `[tool.pytest.ini_options]`. Philosophy: **skip** for
absent hardware; **xfail(strict)** only for known bugs on PRESENT hardware.
The `PDUM_REQUIRE_*` fail-instead-of-skip is wgpu-py's `EXPECT_LAVAPIPE`
idiom — without it a broken CI driver silently skips the GPU suite forever.
(Own-Metal note per R14: the metal probe becomes our own device query, not
mlx.)

**Notebooks**: nbclient natively skips cells by tag —
`skip_cells_with_tag` trait (default `"skip-execution"`). Gating is one flag:
probe in the harness, and when no GPU append
`--ExecutePreprocessor.skip_cells_with_tag=gpu`. Skipped cells keep their
committed outputs untouched — composes with the in-place execution model:
bake GPU outputs locally on the M3, CI executes CPU cells and leaves GPU
outputs as-is. The trait takes ONE tag: use a single `gpu` tag, per-backend
granularity via in-cell probe guards; papermill `--skip-tag` (repeatable)
exists if multi-tag is ever needed.

## 4. CI matrix

| Job | Runner | Backend reality | Notes |
|---|---|---|---|
| lint + kernel unit | ubuntu-latest | none | fast gate, pure-Python thesis tests |
| webgpu tests + notebooks | ubuntu-latest | lavapipe/llvmpipe (`apt install mesa-vulkan-drivers libegl1-mesa-dev libgl1-mesa-dri`) | wgpu-py's own CI is linux-only on exactly this; set `PDUM_REQUIRE_WEBGPU=1` |
| metal (+webgpu-via-Metal) | macos-15 (arm64) | **paravirtualized Metal — usable** | tinygrad runs `DEV=METAL` suites on hosted mac runners, with quirks (MetalGraph disabled when device name contains "virtual"; tensor cores emulated). MLX itself uses self-hosted macs and can hard-crash in headless VMs — land probe-first, promote to `PDUM_REQUIRE_METAL=1` once observed green |
| cuda | none (hosted) | skip via markers | no free OSS GPU runners; GH `gpu-t4-4-core` is paid $0.07/min; RAPIDS-style = self-hosted |
| cuda (on-demand) | ubuntu-latest → Modal | real T4/L4, label-triggered | `gpu-tests` PR label spins Modal GPU, runs pytest, comments on PR (Borda/affordable-GPU-CI pattern) |

## 5. No-local-CUDA workflow (M3 Ultra dev machine)

**Design-for-skip + cloud burst; do not build around an emulator.** The
backend-seam differential test (WGSL image ≈ Python image) makes CUDA a third
implementation behind the same seam — develop against the Python reference
semantics locally where `has_cuda()` is always False. Verification cadence:
Modal free tier ($30/mo credits, per-second billing; a pytest run on an L4 ≈
$0.01–0.02) via the label-triggered pattern, or ad-hoc `modal run gpu_ci.py`
before a release. Emulators: numba's `NUMBA_ENABLE_CUDASIM` only interprets
numba.cuda Python kernels — useless for emitted CUDA C; LeetGPU
(Accel-Sim-backed browser emulation) is handy for hand-poking a generated
kernel but isn't harness-able.

**Structural consequence**: since CUDA can never run in the default dev loop,
its unit tests must be pure on the codegen side — golden-source/IR snapshot
tests hardware-free; hardware-marked tests for execution only. That keeps the
CUDA backend ~90% developable on the M3 with Modal bursts validating the last
mile.

Sources: uv scripts guide + PyTorch guide + deps docs; uv#16522; CuPy install
docs; spaCy usage/PyPI; wgpu-py examples + ci.yml + testutils; NVIDIA/warp;
nbclient client.py; papermill changelog; tinygrad test.yml + #6949; mlx
workflow + #3148; runs-on.com/runners/gpu; GH GPU-runners changelog; RAPIDS
CI docs; Borda/affordable-GPU-CI; modal.com/pricing; numba-cuda simulator
docs; leetgpu.com.
