"""The fusion registry's first two templates (330 §7.3): recognition,
generation, certification, and the refusal law."""

import numpy as np

from pdum.dsl.ir import Builder, Region
from pdum.dsl.ops import CORE_OPS
from pdum.dsl.types import f64
from pdum.tl.dialect import TL_OPS, run_region, tensor_type_of_layout
from pdum.tl.fusion import Plan, plan_region
from pdum.tl.tensor import Tensor
from pdum.tl.zoo.tiles import stencil_tile

OPS = {**CORE_OPS, **TL_OPS}


def _t(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def _contraction_epilogue_subject(M=6, K=8, N=5, seed=3, epilogue=True):
    """y = relu(a @ b + bias): the owner's motivating case — elementwise
    fused onto the back of a dense contraction."""
    rng = np.random.default_rng(seed)
    A = _t(rng.standard_normal((M, K)), ("m", "k"))
    B = _t(rng.standard_normal((K, N)), ("k", "n"))
    bias = _t(rng.standard_normal(N), ("n",))
    b = Builder(OPS)
    pa = b.param(0, tensor_type_of_layout(A.layout))
    pb = b.param(1, tensor_type_of_layout(B.layout))
    prod = b.emit("tl.pointwise", b.emit("tl.repeat_like", pa, pb), b.emit("tl.repeat_like", pb, pa), f="mul")
    z = b.emit("tl.reduce", prod, dims=("k",), f="sum")
    if epilogue:
        pc = b.param(2, tensor_type_of_layout(bias.layout))
        zb = b.emit("tl.pointwise", z, b.emit("tl.repeat_like", pc, z), f="add")
        y = b.emit("tl.pointwise", zb, b.emit("core.const", type=f64, value=0.0), f="maximum")
        region = Region(params=(pa, pb, pc), body=(b.emit("core.yield", y),))
        inputs = [A, B, bias]
    else:
        region = Region(params=(pa, pb), body=(b.emit("core.yield", z),))
        inputs = [A, B]
    return region, inputs


def test_contraction_with_epilogue_is_recognized_and_proved():
    """Template 2 end to end: matched, generated, and the certificate is
    PROVED-LICENSED — normalization walks the fused kernel back to the
    unfused subgraph's content key, epilogue included."""
    region, inputs = _contraction_epilogue_subject()
    plan = plan_region(region)
    (g,) = plan.groups
    assert g.template == "contraction-epilogue"
    assert g.certificate.verdict == "proved-licensed"
    assert g.certificate.licenses == ("gemm.k-reassoc",)
    assert g.confidence == "yellow"  # certified, unmeasured
    assert dict(g.params)["ki"] == 4  # K=8 -> largest divisor with >= 2 tiles
    # belt: the fused kernel computes the same values (license tolerance)
    got = run_region(g.kernel, inputs).to_numpy(order=("m", "n"))
    want = run_region(region, inputs).to_numpy(order=("m", "n"))
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)
    # and the epilogue really fused: relu(a@b + bias) in one kernel
    ref = np.maximum(inputs[0].to_numpy() @ inputs[1].to_numpy() + inputs[2].to_numpy(), 0.0)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-7)


def test_bare_contraction_is_recognized_too():
    region, inputs = _contraction_epilogue_subject(epilogue=False)
    (g,) = plan_region(region).groups
    assert g.template == "contraction-epilogue"
    assert g.certificate.verdict == "proved-licensed"


def test_map_chains_are_recognized_and_proved_exact():
    """Template 1: the stencil twin is a pure map chain — no staging pays,
    the group IS a tile kernel, and the certificate is exact by key."""
    region = stencil_tile().naive
    (g,) = plan_region(region).groups
    assert g.template == "map-chain"
    assert g.certificate.verdict == "proved-exact"
    assert g.confidence == "yellow"


def test_unrecognized_work_refuses_loudly():
    """330's law: no recognized schedule -> a RED uncompiled group naming
    the offender — never silent mediocrity, and never an exception (the
    pass reports; callers choose)."""
    b = Builder(OPS)
    x = b.param(0, tensor_type_of_layout(_t(np.zeros((4, 3)), ("t", "e")).layout))
    top = b.emit("tl.argtopk", x, dim="e", k=2, k_name="c")
    region = Region(params=(x,), body=(b.emit("core.yield", top),))
    plan = plan_region(region)
    (g,) = plan.groups
    assert isinstance(plan, Plan)
    assert (g.template, g.confidence, g.kernel) == ("uncompiled", "red", None)
    assert "tl.argtopk" in g.reason and "no recognized schedule" in g.reason
