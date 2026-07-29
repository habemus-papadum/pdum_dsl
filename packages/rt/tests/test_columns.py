"""The device differentials — reference vs WebGPU vs Metal.

Three-way, the discipline the campaign spikes established: the f64
reference states the answer, each column agrees with it under a stated
f32 tolerance, and the two columns agree with each other BITWISE. The
last one is the sharp assertion — it is what says the shared walker
emitted the same program twice, and it is why guard-vs-exact and
whole-groups-vs-exact-grid are contract clauses rather than semantics.
(Caveat of record: wgpu reaches Metal through Naga on this hardware, so
cross-VENDOR exactness is untested until the CUDA column lands.)

Skips are honest: no wgpu, no PyObjC, or no device.
"""

import dataclasses

import numpy as np
import pytest

from pdum.rt import acquire, executor_for, metal, webgpu
from pdum.tl import Tensor, compute, f32, global_idx, i32  # noqa: F401 — bare in kernel bodies
from pdum.tl.markers import maximum, tanh  # noqa: F401 — bare in kernel bodies

PAIRS = [pytest.param(webgpu, id="webgpu"), pytest.param(metal, id="metal")]
_MODULE = {webgpu: "wgpu", metal: "Metal"}


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def _require(pair):
    pytest.importorskip(_MODULE[pair])
    try:
        acquire(pair.runtime)
    except Exception as exc:  # no adapter / no device: the honest skip
        pytest.skip(f"no {type(pair.runtime).__name__} available: {exc}")


def run_on(pair, kernel, args):
    """Launch on a device column, returning (artifact, executor). The
    executor comes THROUGH the content door and only the executor column
    is swapped, so the launcher runs staging, rebind and the overlap
    refusal identically — the kernel-tier ``kernel[on(pair)]`` bracket is
    the next increment and folds these two lines away."""
    art = kernel.artifact(*args)  # the PUBLIC door (H2, landed)
    ex = executor_for(art, pair)
    assert executor_for(art, pair) is ex, "the content door must return the SAME executor"
    dataclasses.replace(art, executor=ex).launch(args)
    return art, ex


# --- the subjects ------------------------------------------------------------

_GAIN = 3.5


@compute
def saturating(src, dst):
    """Markers at WIDE range — the class Metal's exp(2x) tanh returns NaN
    for (|x| >= 44.36), which the numeric contract's row fixes."""
    i, j = global_idx("y", "x")
    v = src[i, j] * _GAIN
    dst[i, j] = tanh(v) + tanh(0.0 - v) * 0.5 + maximum(v, 0.0) * 1e-3


@compute
def masked(img):
    """The where/select rows: a store-free statement-if lowers to where,
    the value-tier conditional expression to select, and ``and`` to min."""
    i, j = global_idx("y", "x")
    v = f32(i) * 0.5 - f32(j) * 0.25
    lo = v * 0.5
    if v > 1.0:
        lo = v * 2.0
    m = 1.0 if (v > -1.0 and v < 2.5) else 0.0
    img[i, j] = lo * m + maximum(v, 0.0)


@compute
def gather_diag(tex, img):
    """A computed-index read (tl.read), on the 1-D thread policy."""
    (i,) = global_idx("y")
    img[i] = tex[i32(i), i32(i)] * 2.0


@compute
def ignores_one(unused, dst):
    """A param no row names: it must reach neither the binding table nor
    the upload, or WebGPU's pruned pipeline layout refuses the bind
    group (210). Nothing to compute — the point is that it launches."""
    i, j = global_idx("y", "x")
    dst[i, j] = f32(i) + f32(j)


def _wide():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((6, 8)) * 8.0
    a[0, :] = [1e-6, 0.5, 4.0, 12.0, 20.0, 25.0, -25.0, -1e-6]  # up to |v*GAIN| = 87.5
    return (T(a, ("y", "x")), T(np.zeros((6, 8)), ("y", "x")))


SUBJECTS = {
    "markers_wide": (saturating, _wide),
    "where_select": (masked, lambda: (T(np.zeros((6, 8)), ("y", "x")),)),
    "computed_index": (
        gather_diag,
        lambda: (T(np.arange(25.0).reshape(5, 5), ("y", "x")), T(np.zeros(5), ("y",))),
    ),
    "unused_param": (
        ignores_one,
        lambda: (T(np.ones((3, 4)), ("y", "x")), T(np.zeros((3, 4)), ("y", "x"))),
    ),
}


def _reference(kernel, mk):
    args = mk()
    kernel(*args)
    return args


@pytest.mark.parametrize("name", list(SUBJECTS))
@pytest.mark.parametrize("pair", PAIRS)
def test_column_agrees_with_the_reference(pair, name):
    _require(pair)
    kernel, mk = SUBJECTS[name]
    ref = _reference(kernel, mk)
    dev = mk()
    art, _ex = run_on(pair, kernel, dev)
    checked = 0
    for w in art.writable:
        i = art.params.index(w)
        np.testing.assert_allclose(dev[i].to_numpy(), ref[i].to_numpy(), rtol=1e-5, atol=1e-6)
        checked += 1
    assert checked, "a kernel with nothing writable proves nothing"


@pytest.mark.parametrize("name", list(SUBJECTS))
def test_the_two_columns_agree_bitwise(name):
    for pair in (webgpu, metal):
        _require(pair)
    kernel, mk = SUBJECTS[name]
    out = {}
    for pair in (webgpu, metal):
        args = mk()
        art, _ex = run_on(pair, kernel, args)
        out[pair] = [args[art.params.index(w)].to_numpy() for w in art.writable]
    for a, b in zip(out[webgpu], out[metal]):
        # f32 in, f32 out: the f64 host arrays hold exactly what each device
        # wrote, so equality here is BITWISE equality of the device results.
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("pair", PAIRS)
def test_the_contract_reaches_the_compiled_executor(pair):
    _require(pair)
    args = SUBJECTS["markers_wide"][1]()
    _art, ex = run_on(pair, saturating, args)
    assert ex.contract.math == ("tanh",)  # what it substituted, on the artifact
    assert ex.contract.thread_size == (8, 8, 1)
    assert ex.contract.guard == ("emitted" if pair is webgpu else "exact")
    assert len(ex.contract.bindings) == 1 and len(ex.contract.bindings[0]) == 3


@pytest.mark.parametrize("pair", PAIRS)
def test_thread_sizing_specializes_through_the_door(pair):
    _require(pair)
    args = SUBJECTS["where_select"][1]()
    art = masked.artifact(*args)
    a = executor_for(art, pair)
    b = executor_for(art, pair, thread_size=(16, 4, 1))
    assert a is not b and b.contract.thread_size == (16, 4, 1)
    dataclasses.replace(art, executor=b).launch(args)
    ref = _reference(masked, SUBJECTS["where_select"][1])
    np.testing.assert_array_equal(args[0].to_numpy(), ref[0].to_numpy())


@pytest.mark.parametrize("pair", PAIRS)
def test_an_unknown_feature_refuses_naming_the_known_ones(pair):
    pytest.importorskip(_MODULE[pair])
    with pytest.raises(ValueError, match="unknown feature 'wavefronts' — this column knows timestamps"):
        acquire(pair.runtime, features=("wavefronts",))


def test_timestamps_are_requested_at_creation_on_webgpu_and_free_on_metal():
    for pair in (webgpu, metal):
        _require(pair)
        dev = acquire(pair.runtime, features=("timestamps",))
        assert dev is not acquire(pair.runtime)  # features KEY the device
