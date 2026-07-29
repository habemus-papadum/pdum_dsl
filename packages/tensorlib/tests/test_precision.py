"""Precision and boundaries (200 §4): facts at the boundary, choices in
the interior, carrier semantics throughout. Exact decode; inf/nan refuse at
decode; the IR cannot mint encodings; round_to is the one sanctioned door
(straight-through AD by default, zero by declaration); the QAT sample in
boundary-facts terms."""

import numpy as np
import pytest

from pdum.tl import Tensor, red, reduce
from pdum.tl.autodiff import grad
from pdum.tl.dialect import run_named
from pdum.tl.encoding import Descriptor, FormatEncoding, NumpyEncoding, QuantGroupEncoding, adopt
from pdum.tl.lifting import lift_step


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


# --- exact decode -----------------------------------------------------------


def test_numpy_encoding_decodes_exactly():
    enc = NumpyEncoding(np.float32)
    raw = np.array([0.5, -2.0, 3.25], dtype=np.float32)
    dec = enc.decode(raw)
    assert dec.dtype == np.float64
    np.testing.assert_array_equal(dec, raw.astype(np.float64))  # f32 ⊂ f64: exact


def test_structured_encoding_is_the_memory_shape_of_records():
    dt = np.dtype([("x", "<f4"), ("y", "<f8")], align=True)
    enc = NumpyEncoding(dt)
    raw = np.zeros(3, dtype=dt)
    raw["x"], raw["y"] = [1.5, 2.5, 3.5], [10.0, 20.0, 30.0]
    dec = enc.decode(raw)
    np.testing.assert_array_equal(dec["x"], [1.5, 2.5, 3.5])  # offsets/padding: encoding facts
    assert enc.nbytes(3) == 3 * dt.itemsize  # byte truth, descriptor-fed


def test_quant_group_decode_is_scale_times_q_exactly():
    enc = QuantGroupEncoding(group=4)
    q = np.array([1, -8, 7, 0, 2, 2, -1, 3])
    scales = np.array([0.5, 0.25])
    dec = enc.decode((q, scales))
    np.testing.assert_array_equal(dec, [0.5, -4.0, 3.5, 0.0, 0.5, 0.5, -0.25, 0.75])
    with pytest.raises(ValueError, match=r"int4 range \[-8, 7\]"):
        enc.decode((np.array([9]), np.array([1.0])))
    assert enc.nbytes(8) == 4 + 8  # nibbles + f32 scales: byte-exact


def test_format_encoding_srgb_transfer_curve_in_decode():
    enc = FormatEncoding("unorm8-srgb")
    raw = np.array([0, 255, 128], dtype=np.uint8)
    dec = enc.decode(raw)
    assert dec[0] == 0.0 and dec[1] == 1.0 and 0.21 < dec[2] < 0.22  # linear light
    assert enc.nbytes(3) == 3


# --- inf/nan refuse at decode ----------------------------------------------


def test_inf_nan_refuse_at_decode_and_at_the_degenerate_boundary():
    enc = NumpyEncoding(np.float32)
    with pytest.raises(ValueError, match="refuse at decode.*finite sentinels"):
        enc.decode(np.array([1.0, np.inf], dtype=np.float32))
    with pytest.raises(ValueError, match="refuse at decode.*degenerate boundary act"):
        Tensor.from_numpy(np.array([np.nan]), ("i",))


# --- the IR cannot mint encodings ------------------------------------------


def test_astype_as_semantics_does_not_exist():
    # the dialect op registry has no astype door: emitting one refuses
    from pdum.dsl.ir import Builder, VerifyError
    from pdum.dsl.ops import CORE_OPS
    from pdum.tl.dialect import TL_OPS, tensor_type_of_layout

    b = Builder({**CORE_OPS, **TL_OPS})
    x = b.param(0, tensor_type_of_layout(T(np.zeros(3), ("d",)).layout))
    with pytest.raises(VerifyError, match="unknown op 'tl.astype'"):
        b.emit("tl.astype", x, dtype="float16")


# --- round_to: the one sanctioned door --------------------------------------


def test_round_to_is_encode_decode_exactly():
    enc = QuantGroupEncoding(group=4)

    def step(w):
        wq = w.round_to(enc)
        return wq

    w = T([0.5, -0.3, 0.7, 0.1], ("d",))
    ls = lift_step(step, w=w.layout)
    got = run_named(ls.region, {"w": w}, ls.names)["wq"].to_numpy()
    q, scales = enc.encode(w.to_numpy())
    np.testing.assert_array_equal(got, enc.decode((q, scales)))  # exact roundtrip


def test_round_to_grad_is_straight_through_by_default():
    enc = NumpyEncoding(np.float16)

    def step(w):
        wq = w.round_to(enc)
        y = wq * wq
        s = reduce(red.sum, y, "d")
        return wq, s

    w = T([0.5, -0.25, 1.5], ("d",))
    ls = lift_step(step, w=w.layout)
    rg = grad(ls.region, "s", {"w": w}, names=ls.names)
    env = run_named(rg.region, {"w": w}, rg.names)
    got = env[rg.grads["w"]].to_numpy()
    wq = run_named(ls.region, {"w": w}, ls.names)["wq"].to_numpy()
    np.testing.assert_allclose(got, 2 * wq)  # d(sum wq²)/dw straight-through = 2·wq


def test_round_to_zero_grad_by_declaration():
    enc = NumpyEncoding(np.float16)

    def step(w):
        wq = w.round_to(enc, grad="zero")
        s = reduce(red.sum, wq, "d")
        return s

    w = T([1.0, 2.0], ("d",))
    ls = lift_step(step, w=w.layout)
    rg = grad(ls.region, "s", {"w": w}, names=ls.names)
    assert rg.grads["w"] is None  # declared zero: no gradient flows


# --- descriptors + adopt: interop and precision facts are one concept -------


def test_adopt_decodes_a_dictated_encoding_into_the_interior():
    raw = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32)
    t = adopt(raw, NumpyEncoding(np.float32), ("i", "j"))
    assert t.dtype == np.float64  # the declared oracle interior
    np.testing.assert_array_equal(t.to_numpy(), raw.astype(np.float64))


def test_descriptor_carries_byte_truth():
    lay = T(np.zeros((4, 6)), ("i", "j")).layout
    d16 = Descriptor(buffer=None, layout=lay, encoding=NumpyEncoding(np.float16))
    d64 = Descriptor(buffer=None, layout=lay, encoding=NumpyEncoding(np.float64))
    assert d16.nbytes == 24 * 2 and d64.nbytes == 24 * 8


# --- the QAT sample, in boundary-facts terms (the fallback criterion) -------


def test_qat_master_weights_train_through_quantized_forward():
    """The mixed-precision/QAT sample (200 §4): master weights live in the
    interior; the forward computes through round_to(bf16-class encoding);
    the checkpoint's dtype is a FACT a descriptor records. The gradient
    reaches the MASTER weights (straight-through), so they train. No
    required meaning here depends on an interior encoding that is neither
    a boundary fact nor round_to — the fallback criterion is NOT met, and
    the two-surface fallback stays unexercised."""
    enc = NumpyEncoding(np.float16)  # the bf16-class stand-in numpy ships

    def step(w, x):
        # w: master weights, interior, exact; wq: the quantized view
        wq = w.round_to(enc)
        xy = x * wq
        loss = reduce(red.sum, xy, "d")
        return loss

    rng = np.random.default_rng(0)
    inputs = {"w": T(rng.standard_normal(8), ("d",)), "x": T(rng.standard_normal(8), ("d",))}
    ls = lift_step(step, w=inputs["w"].layout, x=inputs["x"].layout)
    rg = grad(ls.region, "loss", dict(inputs), names=ls.names)
    env = run_named(rg.region, inputs, rg.names)
    g = env[rg.grads["w"]].to_numpy(order=("d",))
    np.testing.assert_allclose(g, inputs["x"].to_numpy())  # straight-through: x, not 0
    # one SGD step on the MASTER weights moves the quantized forward
    w2 = inputs["w"].to_numpy() - 0.5 * g
    loss1 = run_named(ls.region, inputs, ls.names)["loss"].item()
    loss2 = run_named(ls.region, {"w": T(w2, ("d",)), "x": inputs["x"]}, ls.names)["loss"].item()
    assert float(loss2) < float(loss1)


# --- cost oracles read boundary byte truth (the re-derivation) --------------


def test_cost_oracles_read_boundary_byte_truth():
    """Descriptor-fed byte-exact sizes (200 §4): an f16 boundary fact halves
    the INPUT bytes; the interior stays the DECLARED oracle f64 — byte truth
    enters at descriptors, never mid-program."""
    from pdum.tl.memory import peak_memory

    def step(w):
        y = w * w
        return y

    lay = T(np.zeros(16), ("d",)).layout
    ls = lift_step(step, w=lay)
    full = peak_memory(ls.region, {"w": T(np.zeros(16), ("d",))}, names=ls.names)
    fact16 = peak_memory(
        ls.region, {"w": Descriptor(buffer=None, layout=lay, encoding=NumpyEncoding(np.float16))}, names=ls.names
    )
    assert full.input_bytes == 16 * 8
    assert fact16.input_bytes == 16 * 2  # the checkpoint's dtype is a FACT
    assert full.alloc_bytes["y"] == fact16.alloc_bytes["y"] == 16 * 8  # declared interior


def test_quant_descriptor_bytes_are_sub_byte_exact():
    from pdum.tl.memory import peak_memory

    def step(w):
        y = w * w
        return y

    lay = T(np.zeros(64), ("d",)).layout
    ls = lift_step(step, w=lay)
    q = peak_memory(
        ls.region, {"w": Descriptor(buffer=None, layout=lay, encoding=QuantGroupEncoding(group=16))}, names=ls.names
    )
    assert q.input_bytes == 32 + 4 * 4  # nibbles + f32 scales, byte-exact
