"""The seam, differentially: the WGSL backend must agree with the Python
reference on the same kernel bodies — that is M1's whole claim. GPU-gated:
skips without an adapter, and PDUM_REQUIRE_WEBGPU=1 turns skip into fail
(CI anti-rot, the EXPECT_LAVAPIPE idiom)."""

import os

import pytest

import pdum.dsl  # noqa: F401  — batteries
from pdum.dsl.demo.simple_shader import wgsl
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.cache import no_compile
from pdum.dsl.kernel.registry import DEFAULT

if not wgsl.is_available():
    if os.environ.get("PDUM_REQUIRE_WEBGPU"):
        pytest.fail("PDUM_REQUIRE_WEBGPU is set but no wgpu adapter answered", pytrace=False)
    pytest.skip("no wgpu adapter", allow_module_level=True)


@pytest.fixture(autouse=True)
def _cold_tier1():
    DEFAULT.specializations.bump_generation()


def make_pair(cx, cy, r, gain):
    """The SAME body, twice: once per-pixel on CPU (device), once per-thread
    on GPU (compute). The differential test is the backend seam's contract."""

    def body(i, j):
        x = i / 32.0 - 1.0
        y = j / 32.0 - 1.0
        dx = x - cx
        dy = y - cy
        return gain * (0.25 + 0.75 * (1.0 if dx * dx + dy * dy < r * r else 0.0)) * (0.9 + 0.05 * (x + y))

    cpu = jit(kind="device")(body)
    gpu = jit(kind="simple_shader.compute")(body)
    return cpu, gpu


def test_differential_compute_vs_python():
    cpu, gpu = make_pair(0.15, -0.2, 0.55, 1.0)
    img = gpu(out=(64, 64))
    worst = 0.0
    for j in range(64):
        for i in range(64):
            worst = max(worst, abs(img[j * 64 + i] - cpu(float(i), float(j))))
    assert worst < 5e-6  # f64 (CPU) vs f32 (GPU): agreement within f32 epsilon territory


def test_thesis_on_gpu_values_and_resolution_move_freely():
    make_pair(0.0, 0.0, 0.5, 1.0)[1](out=(32, 32))  # the one compile
    c0 = DEFAULT.specializations.compiles
    with no_compile():
        for f in range(1, 40):
            make_pair(0.01 * f, -0.2, 0.4, 1.0 + f)[1](out=(32, 32))
        make_pair(0.0, 0.0, 0.5, 1.0)[1](out=(128, 128))  # resolution is runtime data
    assert DEFAULT.specializations.compiles == c0


def test_fragment_renders_and_agrees_with_compute():
    def body(x, y):
        u = x / 32.0 - 1.0
        v = y / 32.0 - 1.0
        return 1.0 if u * u + v * v < 0.36 else 0.0

    frag = jit(kind="simple_shader.fragment")(body)
    comp = jit(kind="simple_shader.compute")(body)
    rows = frag(out=(64, 64))
    flat = comp(out=(64, 64))
    diff = sum(
        1
        for j in range(64)
        for i in range(64)
        if abs(rows[j][i] - flat[j * 64 + i]) > 1.5 / 255  # u8 quantization on the fragment path
    )
    # fragment samples at pixel CENTERS (pos.x = i + 0.5): boundary pixels may
    # legitimately disagree with the integer-coordinate compute grid
    assert diff <= 64 * 64 * 0.02


def test_uniform_layout_is_f32_narrowed():
    _, gpu = make_pair(0.5, 0.25, 0.5, 2.0)
    gpu(out=(8, 8))
    key = next(k for k in DEFAULT.specializations._ready if k[2][:2] == ("demo.simple_shader.wgsl", "compute"))
    rec = DEFAULT.specializations._ready[key]
    assert all(s.dest.fmt == "<f" for s in rec.plan.slots)  # f64 captures -> 4-byte f32 slots
    assert rec.plan.staging_size == 16  # four captures, densely packed
    assert "struct Env" in rec.artifact.__pdum_source__


def test_routing_device_kind_still_python():
    cpu, _ = make_pair(0.0, 0.0, 0.5, 1.0)
    assert abs(cpu(32.0, 32.0) - 0.9) < 1e-12  # exact f64 path: the python backend served it


def test_positional_args_on_derived_param_kernels_are_refused():
    """Params are thread coordinates: a positional arg would be silently dead
    (it fingerprints into the key but never reaches the GPU) — refuse loudly."""
    from pdum.dsl.kernel.ir import VerifyError

    @jit(kind="simple_shader.compute")
    def k(i):
        return i * 2.0

    with pytest.raises(VerifyError, match="positional arguments are not accepted"):
        k(999.0, out=4)


def test_1d_domain_including_non_multiple_of_workgroup():
    def make(scale):
        @jit(kind="simple_shader.compute")
        def k(i):
            return i * scale

        return k

    vals = make(0.5)(out=100)  # 100 is not a multiple of 64: the dims guard earns its keep
    assert len(vals) == 100
    assert vals[99] == 49.5 and vals[0] == 0.0


def test_int_and_bool_captures_reach_the_gpu_correctly():
    """The Env struct members follow the SLOT FORMAT (i32/u32), and bool reads
    compare against 0u — the review caught f32-only members reinterpreting
    integer bits as float garbage."""

    def make(n, flag, base):
        @jit(kind="simple_shader.compute")
        def k(i):
            m = n * 2
            b = base if flag else 0.0
            return i + float(m) + b

        return k

    assert make(3, True, 10.0)(out=4) == (16.0, 17.0, 18.0, 19.0)
    assert make(3, False, 10.0)(out=2) == (6.0, 7.0)


def test_domain_error_paths_are_loud():
    from pdum.dsl.kernel.ir import VerifyError

    @jit(kind="simple_shader.compute")
    def k(i, j):
        return i + j

    with pytest.raises(VerifyError, match="launch domain"):
        k()
    with pytest.raises(VerifyError, match="rank"):
        k(out=(64,))
