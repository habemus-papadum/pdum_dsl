"""The fusion registry's first two templates (330 §7.3): recognition,
generation, certification, and the refusal law."""

import numpy as np

from pdum.dsl.ir import Builder, Region
from pdum.dsl.ops import CORE_OPS
from pdum.dsl.types import f64
from pdum.tl.dialect import TL_OPS, run_region, tensor_type_of_layout
from pdum.tl.fusion import Plan, plan_region
from pdum.tl.tensor import Tensor
from pdum.tl.zoo.tiles import flash_tile, stencil_tile

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


def test_stencils_are_recognized_and_staged():
    """Template 4: the stencil twin carries neighborhood reuse (five
    shifted reads of one halo), so the pass STAGES the halo — and the
    certificate stays exact by key, because erasure strips the stage."""
    region = stencil_tile().naive
    (g,) = plan_region(region).groups
    assert g.template == "stencil"
    assert g.certificate.verdict == "proved-exact"
    assert dict(g.params)["staged"] == (0,)  # the halo param
    assert any(n.op == "tl.stage" for n in _walk(g.kernel))
    got = run_region(g.kernel, [stencil_tile().inputs["u"]]).to_numpy()
    want = run_region(region, [stencil_tile().inputs["u"]]).to_numpy()
    np.testing.assert_array_equal(got, want)


def test_reuse_free_chains_stay_map_chains():
    """Template 1 still claims chains with NO neighborhood overlap — a
    single shifted read has nothing to reuse, so nothing stages."""
    b = Builder(OPS)
    x = b.param(0, tensor_type_of_layout(_t(np.zeros((4, 6)), ("x", "y")).layout))
    sh = b.emit("tl.shift", x, deltas=(("x", 1),))
    sl = b.emit("tl.slice", sh, ranges=(("x", (1, 5)),))
    y = b.emit("tl.pointwise", sl, f="exp")
    region = Region(params=(x,), body=(b.emit("core.yield", y),))
    (g,) = plan_region(region).groups
    assert g.template == "map-chain"
    assert g.certificate.verdict == "proved-exact"


def test_bare_row_normalization_stages_its_rows():
    """Template 3a: softmax over a parameter — the row inputs stage once
    (three passes, one load), proved-exact by erasure."""
    rng = np.random.default_rng(9)
    X = _t(rng.standard_normal((4, 6)), ("t", "s"))
    b = Builder(OPS)
    p = b.param(0, tensor_type_of_layout(X.layout))
    mx = b.emit("tl.reduce", p, dims=("s",), f="max")
    e = b.emit("tl.pointwise", b.emit("tl.pointwise", p, b.emit("tl.repeat_like", mx, p), f="sub"), f="exp")
    sm = b.emit("tl.reduce", e, dims=("s",), f="sum")
    pr = b.emit("tl.pointwise", e, b.emit("tl.repeat_like", sm, p), f="div")
    region = Region(params=(p,), body=(b.emit("core.yield", pr),))
    (g,) = plan_region(region).groups
    assert g.template == "row-normalization"
    assert g.certificate.verdict == "proved-exact"
    assert any(n.op == "tl.stage" for n in _walk(g.kernel))
    got = run_region(g.kernel, [X]).to_numpy(order=("t", "s"))
    want = run_region(region, [X]).to_numpy(order=("t", "s"))
    np.testing.assert_array_equal(got, want)


def test_the_flash_composition_is_recognized_and_licensed():
    """Template 3b: contraction -> closed-form mask -> softmax ->
    contraction becomes the online-softmax fold, certified under the
    declared license with the template's own adversarial families."""
    f = flash_tile()
    (g,) = plan_region(f.naive).groups
    assert g.template == "flash"
    assert g.certificate.verdict == "licensed-differential"
    assert g.certificate.licenses == ("flash.online-softmax",)
    assert g.certificate.families == ("gaussian", "wide-scores", "dominant-key")
    assert sum(1 for n in _walk(g.kernel) if n.op == "tl.fold") == 2  # o and den finals
    vals = list(f.inputs.values())
    got = run_region(g.kernel, vals).to_numpy(order=("t", "o"))
    want = run_region(f.naive, vals).to_numpy(order=("t", "o"))
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)


def _walk(region):
    from pdum.tl.dialect import walk_region

    for n in walk_region(region):
        yield n
        for r in n.regions:
            yield from _walk(r)


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


def test_certification_caps_at_the_program_scale():
    """§7.7: a T=4096 flash region — the size whose differential
    certification ran T^2 softmax through the python reference and
    killed sessions — certifies through a SHRUNK TWIN: same match, same
    generator, clamped extents. The certificate is about the program,
    not the size."""
    f = flash_tile(T=4096, E=16, OD=16, SI=2)
    (g,) = plan_region(f.naive).groups
    assert g.template == "flash" and g.confidence == "yellow"
    assert g.certificate is not None
