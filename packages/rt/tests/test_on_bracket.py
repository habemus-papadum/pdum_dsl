"""The selection bracket and the public artifact door (increment 2 of 283).

``kernel[on(metal)](args)`` — one bracket, two selector kinds, any
order; ``kernel.artifact(*args)`` compiles without launching. The
two-line replace-dance the columns' tests used to do is folded away.
"""

import numpy as np
import pytest

from pdum.rt import metal, on, webgpu
from pdum.tl import Tensor, compute, config, global_idx
from pdum.tl.markers import tanh  # noqa: F401 — bare in kernel bodies


def T(arr, names=("y", "x")):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


@compute
def scaled(src, dst):
    i, j = global_idx("y", "x")
    dst[i, j] = tanh(src[i, j]) * 2.0 + 1.0


def _args():
    rng = np.linspace(-50.0, 50.0, 12).reshape(3, 4)  # crosses tanh's Metal edge
    return (T(rng), T(np.zeros((3, 4))))


def _require(pair):
    if pair is webgpu:
        pytest.importorskip("wgpu")
    else:
        pytest.importorskip("Metal")
    try:
        from pdum.rt import acquire

        acquire(pair.runtime)
    except Exception as exc:  # noqa: BLE001 — a skip, not a policy
        pytest.skip(f"no device for {pair}: {exc}")


def test_artifact_compiles_without_launching():
    src, dst = _args()
    art = scaled.artifact(src, dst)
    assert art.region is not None and "dst" in art.writable
    assert dst.to_numpy().max() == 0.0  # nothing launched, nothing stored
    assert scaled.artifact(src, dst) is art  # tier-1 cached, same entry


def test_bracket_refuses_a_non_selector():
    with pytest.raises(TypeError, match="config\\(\\.\\.\\.\\) object or a device pair"):
        scaled["metal"]


def test_bracket_refuses_doubled_selectors():
    with pytest.raises(TypeError, match="one bracket binds one pair"):
        scaled[metal][webgpu]
    with pytest.raises(TypeError, match="one bracket binds one config"):
        scaled[config()][config()]


@pytest.mark.parametrize("pair", (webgpu, metal), ids=("webgpu", "metal"))
def test_the_bracket_launches_on_device(pair):
    _require(pair)
    src, dst = _args()
    ref_src, ref_dst = _args()
    scaled(ref_src, ref_dst)  # the reference launch
    scaled[on(pair)](src, dst)
    np.testing.assert_allclose(dst.to_numpy(), ref_dst.to_numpy(), rtol=1e-5, atol=1e-6)


def test_both_orders_compose_and_agree_bitwise():
    _require(webgpu)
    _require(metal)
    a_src, a_dst = _args()
    b_src, b_dst = _args()
    scaled[on(webgpu)][config()](a_src, a_dst)  # pair then config
    scaled[config()][on(metal)](b_src, b_dst)  # config then pair
    assert np.array_equal(a_dst.to_numpy(), b_dst.to_numpy())  # the columns agree bitwise
