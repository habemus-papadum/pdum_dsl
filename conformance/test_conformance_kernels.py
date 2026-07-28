"""The conformance battery — compute kernels, reference vs WebGPU.

Executor-parameterized differentials (200 §7, the P8 gate): every case
runs on the f64 reference interpreter and on the translation-only WGSL
executor, and the assertion is agreement under the stated tolerance
(f32 device arithmetic; 210's policy on both sides). Skips are honest:
no adapter, or a named untranslatable op.
"""

import numpy as np
import pytest
from pdum.tl import Tensor, compute, f32, global_idx, i32, thread_idx  # noqa: F401 — bodies' globals
from pdum.tl.markers import exp, log, maximum, sin, sqrt, tanh  # noqa: F401 — bare in kernel bodies
from wgsl_executor import Untranslatable, wgpu_artifact

from pdum.dsl import jit, op


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def _require_device():
    pytest.importorskip("wgpu")
    try:
        from pdum.tl.graphics import _device

        _device()
    except Exception as exc:  # no adapter: the honest skip
        pytest.skip(f"no WebGPU device available: {exc}")


def differential(kernel, make_args, rtol=1e-5, atol=1e-6):
    """Run reference and device on identical fresh inputs; compare every
    writable tensor under the stated tolerance."""
    _require_device()
    from pdum.tl.kernel import KERNELS, _arg_fp, _code_fp, _env_fp

    ref_args = make_args()
    kernel(*ref_args)  # the reference launch (and the artifact warm-up)
    dev_args = make_args()
    key = (_code_fp(kernel.fn), _env_fp(kernel.fn), tuple(_arg_fp(a) for a in dev_args), ())
    art = KERNELS.peek(key)
    try:
        wgpu_artifact(art).launch(dev_args, {})
    except Untranslatable as exc:
        pytest.skip(f"no WGSL translation yet: {exc}")
    checked = 0
    for name in art.writable:
        i = art.params.index(name)
        want = ref_args[i].to_numpy()
        got = dev_args[i].to_numpy()
        np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)
        checked += 1
    assert checked, "a kernel with nothing writable proves nothing"


@compute
def ambient_field(img):
    i, j = global_idx("y", "x")
    img[i, j] = f32(i) * 0.25 + f32(j) * 0.5 + 1.5


def test_ambient_pointwise_store():
    differential(ambient_field, lambda: (T(np.zeros((5, 7)), ("y", "x")),))


@compute
def scaled_copy(src, dst):
    i, j = global_idx("y", "x")
    dst[i, j] = src[i, j] * 2.0 + 1.0


def test_identity_load():
    def mk():  # a FRESH seeded rng per call: both executors see identical inputs
        rng = np.random.default_rng(7)
        return (T(rng.standard_normal((4, 6)), ("y", "x")), T(np.zeros((4, 6)), ("y", "x")))

    differential(scaled_copy, mk)


_GAIN = 3.5


@compute
def gained(img):
    i, j = global_idx("y", "x")
    img[i, j] = (f32(i) + f32(j)) * _GAIN


def test_captured_scalar_rides_the_uniform_buffer():
    differential(gained, lambda: (T(np.zeros((3, 4)), ("y", "x")),))


@op
def twill(a, b):
    @jit()
    def go(x):
        return x * a + b

    return go


@op
def zoom(s):
    @jit()
    def go(x):
        return x * s

    return go


@compute
def shader(f, img):
    i, j = global_idx("y", "x")
    img[i, j] = f(f32(i) + f32(j))


def test_spliced_fn_arg_slots():
    f = twill(4.0, 3.0) | zoom(0.5)
    differential(shader, lambda: (f, T(np.zeros((3, 4)), ("y", "x"))))


@compute
def gather_diag(tex, img):
    (i,) = global_idx("y")
    img[i] = tex[i32(i), i32(i)]


def test_computed_index_read():
    differential(gather_diag, lambda: (T(np.arange(9.0).reshape(3, 3), ("y", "x")), T(np.zeros(3), ("y",))))


@compute
def box3(tex, img):
    (i,) = global_idx("y")
    acc = 0.0
    for di in range(3):
        acc = acc + tex[i32(i) + di]
    img[i] = acc / 3.0


def test_unrolled_neighborhood():
    differential(box3, lambda: (T(np.arange(5.0), ("y",)), T(np.zeros(3), ("y",))))


@compute
def spiky(img):
    i, j = global_idx("y", "x")
    v = maximum(f32(i), f32(j)) * 0.25 + (f32(i) - f32(j)) * 0.5
    img[i, j] = tanh(v) + sqrt(v * v + 1.0)


def test_markers_and_predicates():
    differential(spiky, lambda: (T(np.zeros((5, 7)), ("y", "x")),))


def _wrt_probe():
    @jit()
    def go(y, x):
        d = sqrt(y * y + x * x + 1.0)
        return with_respect_to(d, y)  # noqa: F821 — = y / sqrt(y² + x² + 1)

    return go


@compute
def ambient_derivative(f, img):
    i, j = global_idx("y", "x")
    img[i, j] = f(f32(i), f32(j))


def test_ambient_derivative_differential():
    """The wrt-ambient derivative on the DEVICE: the tangent engine
    splices at lower time, so the derivative reaches WGSL as ordinary
    scalar rows — differentially exact against the f64 reference within
    the stated f32 tolerance."""
    differential(ambient_derivative, lambda: (_wrt_probe(), T(np.zeros((5, 7)), ("y", "x"))))


# --- 290/211 battery subjects: the where/select rows, adversarial ranges, ----
# --- and the whole-chain-on-device rule (PR #8) ------------------------------


@compute
def masked_field(img):
    i, j = global_idx("y", "x")
    v = f32(i) * 0.5 - f32(j) * 0.25
    lo = v * 0.5
    if v > 1.0:  # store-free statement-if (290 §4.2) -> the where row
        lo = v * 2.0
    m = 1.0 if (v > -1.0 and v < 2.5) else 0.0  # and -> the min row; ifexp -> where
    img[i, j] = lo * m + maximum(v, 0.0)


def test_where_select_rows_have_a_subject():
    """The rows were live-but-untested (211 §2e): a swapped select argument
    order would have stayed green. This subject exercises where, min/max,
    and comparison rows differentially."""
    differential(masked_field, lambda: (T(np.zeros((6, 8)), ("y", "x")),))


# the adversarial input families (the doctrine applied to the battery itself):
# saturation, tiny normals, wide magnitudes — everything finite in f32, so the
# differential states honest tolerances instead of dodging edges.
_EDGES = np.array(
    [
        [1e-30, 1e-6, 0.5, 1.0, 4.0],
        [20.0, 44.0, 50.0, 80.0, 88.0],
        [1e6, 1e15, 1e30, 2.0, 9.0],
    ]
)


@compute
def math_edges(src, sat, wide):
    i, j = global_idx("y", "x")
    v = src[i, j]
    sat[i, j] = tanh(v) - tanh(v * 2.0) + exp(0.0 - v) * 0.5
    wide[i, j] = log(v) * 0.05 + sqrt(v) * 1e-16


def test_math_edges_survive_wide_ranges():
    """tanh saturation (|x| to 88 — the class Metal's exp(2x) tanh fails,
    210), log down to 1e-30, sqrt across 60 decades. Tolerances state the
    f32-input rounding honestly (log amplifies input eps to ~1e-7 abs)."""

    def mk():
        return (
            T(_EDGES.copy(), ("y", "x")),
            T(np.zeros_like(_EDGES), ("y", "x")),
            T(np.zeros_like(_EDGES), ("y", "x")),
        )

    differential(math_edges, mk, rtol=5e-5, atol=1e-6)


@compute
def chain_stage_one(src, mid):
    i, j = global_idx("y", "x")
    mid[i, j] = sin(src[i, j] * 1.7) * 40.0 + src[i, j] * 0.5


@compute
def chain_stage_two(mid, out):
    i, j = global_idx("y", "x")
    out[i, j] = tanh(mid[i, j]) * 2.0 + mid[i, j] * 0.125


def test_device_goldens_run_the_whole_chain_on_the_device_column():
    """PR #8's rule: a device golden whose upstream ran on the host is
    quietly a host golden (f64 sin narrowed to f32 is not f32 sin —
    measured 1.8e-07 masking real drift). Both stages run on BOTH columns;
    the comparison is end to end, so the device column's own f32 upstream
    is what the downstream stage consumes."""
    _require_device()
    from pdum.tl.kernel import _compile

    def mk():
        rng = np.random.default_rng(11)
        return T(rng.standard_normal((4, 5)) * 3.0, ("y", "x"))

    src_r, mid_r, out_r = mk(), T(np.zeros((4, 5)), ("y", "x")), T(np.zeros((4, 5)), ("y", "x"))
    chain_stage_one(src_r, mid_r)
    chain_stage_two(mid_r, out_r)
    src_d, mid_d, out_d = mk(), T(np.zeros((4, 5)), ("y", "x")), T(np.zeros((4, 5)), ("y", "x"))
    try:
        for k, args in ((chain_stage_one, (src_d, mid_d)), (chain_stage_two, (mid_d, out_d))):
            wgpu_artifact(_compile(k.fn, args)).launch(args, {})
    except Untranslatable as exc:
        pytest.skip(f"no WGSL translation yet: {exc}")
    np.testing.assert_allclose(out_d.to_numpy(), out_r.to_numpy(), rtol=5e-5, atol=2e-5)
