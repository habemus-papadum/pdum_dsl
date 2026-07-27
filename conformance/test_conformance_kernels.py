"""The conformance battery — compute kernels, reference vs WebGPU.

Executor-parameterized differentials (200 §7, the P8 gate): every case
runs on the f64 reference interpreter and on the translation-only WGSL
executor, and the assertion is agreement under the stated tolerance
(f32 device arithmetic; 210's policy on both sides). Skips are honest:
no adapter, or a named untranslatable op.
"""

import numpy as np
import pytest
from pdum.dsl import jit, op
from pdum.tl import Tensor, compute, f32, global_idx, i32, thread_idx  # noqa: F401 — bodies' globals
from pdum.tl.markers import maximum, sqrt, tanh  # noqa: F401 — bare in kernel bodies
from wgsl_executor import Untranslatable, wgpu_artifact


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
