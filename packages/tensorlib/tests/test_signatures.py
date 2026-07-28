"""The carrier/unit signature pass: inference, refusals, target_unit."""

import numpy as np
import pytest
from pdum.tl import (
    SignatureError,
    Tensor,
    VInfo,
    defmarker,
    defreducer,
    infer_signatures,
    marker_signature,
    pointwise,
    pw,
    red,
    reduce,
    scan,
    u,
)
from pdum.tl.autodiff import grad
from pdum.tl.dialect import run_named, run_region
from pdum.tl.lifting import lift_step
from pdum.tl.markers import exp
from pdum.tl.units import ONE


def T(arr, names, unit=None):
    t = Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)
    return t.with_value_units(u.parse_unit(unit)) if unit else t


M = u.parse_unit("m")
S = u.parse_unit("s")


def _mul_sum_step(x, w):
    def step(x, w):
        p = x * w
        y = reduce(red.sum, p, "i")
        return y

    return lift_step(step, x=x.layout, w=w.layout)


def test_units_flow_through_mul_and_sum():
    inputs = {"x": T([1.0], "i", "m"), "w": T([1.0], "i", "s")}
    ls = _mul_sum_step(inputs["x"], inputs["w"])
    sigs = infer_signatures(ls.region, inputs, names=ls.names)
    assert sigs["p"].unit == M * S
    assert sigs["y"].unit == M * S


def test_unlabeled_programs_stay_unknown():
    inputs = {"x": T([1.0], "i"), "w": T([1.0], "i")}
    ls = _mul_sum_step(inputs["x"], inputs["w"])
    sigs = infer_signatures(ls.region, inputs, names=ls.names)
    assert sigs["y"] == VInfo("real", None)


def test_grad_infers_target_unit():
    # CONCERNS #16's payoff: no target_unit argument — the pass reads it
    inputs = {"x": T([2.0, 3.0], "i", "m"), "w": T([5.0, 7.0], "i", "s")}
    ls = _mul_sum_step(inputs["x"], inputs["w"])
    rg = grad(ls.region, "y", inputs, names=ls.names)
    env = run_named(rg.region, inputs, rg.names)
    assert env[rg.grads["x"]].value_units == S  # (m·s)/m
    assert env[rg.grads["w"]].value_units == M
    np.testing.assert_allclose(env[rg.grads["x"]].to_numpy(), [5.0, 7.0])


def test_conflicting_units_refuse_in_pass_and_in_grad():
    def step(x, w):
        p = x + w
        y = reduce(red.sum, p, "i")
        return y

    inputs = {"x": T([1.0], "i", "m"), "w": T([1.0], "i", "s")}
    ls = lift_step(step, x=inputs["x"].layout, w=inputs["w"].layout)
    with pytest.raises(SignatureError, match="unit mismatch"):
        infer_signatures(ls.region, inputs, names=ls.names)
    with pytest.raises(SignatureError):
        grad(ls.region, "y", inputs, names=ls.names)


def test_exp_of_dimensioned_refuses_statically():
    def step(x):
        e = pointwise(exp, x)
        return e

    ls = lift_step(step, x=T([1.0], "i").layout)
    with pytest.raises(SignatureError, match="dimensionless"):
        infer_signatures(ls.region, {"x": T([1.0], "i", "m")}, names=ls.names)


def test_exp_of_dimensioned_refuses_at_runtime():
    with pytest.raises(SignatureError, match="dimensionless"):
        pointwise(pw.exp, T([1.0], "i", "m"))
    assert pointwise(pw.exp, T([1.0], "i")).to_numpy() == pytest.approx(np.e)


def test_zero_constant_is_unit_polymorphic():
    plus0 = defmarker(None, 1, lambda x: x + 0)
    plus1 = defmarker(None, 1, lambda x: x + 1)
    assert marker_signature(plus0, [VInfo(None, M)]).unit == M
    with pytest.raises(SignatureError, match="unit mismatch"):
        marker_signature(plus1, [VInfo(None, M)])


def test_comparison_units_must_match():
    with pytest.raises(SignatureError, match="unit mismatch"):
        marker_signature("lt", [VInfo(None, M), VInfo(None, S)])
    assert marker_signature("lt", [VInfo(None, M), VInfo(None, M)]) == VInfo("bool", None)


def test_where_condition_must_be_bool():
    with pytest.raises(SignatureError, match="bool"):
        marker_signature("where", [VInfo("real"), VInfo(), VInfo()])
    ok = marker_signature("where", [VInfo("bool"), VInfo("int", M), VInfo("real")])
    assert ok == VInfo("real", M)


def test_carriers_join_up_the_tower():
    assert marker_signature("div", [VInfo("int"), VInfo("int")]).carrier == "rat"
    assert marker_signature("exp", [VInfo("int")]) == VInfo("real", ONE)
    assert marker_signature("mul", [VInfo("bool"), VInfo("real")]).carrier == "real"


def test_composite_reducer_signature_reaches_fixed_point():
    # linrec with unitless decay and metre-valued drive infers metres
    # (same declaration as test_mdsl's — content-identical re-registration)
    linrec_t = defreducer(
        "linrec_t",
        state=2,
        element=2,
        lift=lambda a, b: (a, b),
        combine=lambda left, right: (left[0] * right[0], right[0] * left[1] + right[1]),
        init=(1.0, 0.0),
        project=lambda A, B: B,
    )

    def step(a, b):
        h = scan(linrec_t, (a, b), "t")
        return h

    inputs = {"a": T([0.5], "t"), "b": T([1.0], "t", "m")}
    ls = lift_step(step, a=inputs["a"].layout, b=inputs["b"].layout)
    sigs = infer_signatures(ls.region, inputs, names=ls.names)
    assert sigs["h"].unit == M


def test_prod_of_dimensioned_refuses():
    def step(x):
        y = reduce(red.prod, x, "i")
        return y

    ls = lift_step(step, x=T([1.0], "i").layout)
    with pytest.raises(SignatureError, match="static extent"):
        infer_signatures(ls.region, {"x": T([1.0], "i", "m")}, names=ls.names)
    assert infer_signatures(ls.region, {"x": T([1.0], "i")}, names=ls.names)["y"].unit is None


def test_content_addressed_defmarker_dedupes():
    m1 = defmarker(None, 1, lambda x: x * x + 1)
    m2 = defmarker(None, 1, lambda x: x * x + 1)
    assert m1 is m2
    assert m1.name.startswith("m_")


# --- the Region face (the excavation, LEVELS) -------------------------------


def test_region_face_reports_by_name():
    def step(x, w):
        p = x * w
        q = p / w
        m = reduce(red.mean, q, "i")
        return m

    inputs = {"x": T([1.0, 2.0], "i", "m"), "w": T([3.0, 4.0], "i", "s")}
    ls = lift_step(step, **{k: v.layout for k, v in inputs.items()})
    sigs = infer_signatures(ls.region, inputs, names=ls.names)
    assert sigs["p"].unit == M * S
    assert sigs["q"].unit == M
    assert sigs["m"] == VInfo("real", M)


def test_region_face_consts_are_plumbing():
    def step(x):
        y = x * 2.0
        z = y + x
        return z

    inputs = {"x": T([1.0, 2.0], "i", "m")}
    ls = lift_step(step, x=inputs["x"].layout)
    sigs = infer_signatures(ls.region, inputs, names=ls.names)
    # const materializations are deferred scalars: no report row
    assert set(sigs) == {"x", "y", "z"}
    assert sigs["z"].unit == M


def test_region_face_refuses_dimensioned_exp():
    def step(x):
        return pointwise(exp, x)

    inputs = {"x": T([1.0], "i", "m")}
    ls = lift_step(step, x=inputs["x"].layout)
    with pytest.raises(SignatureError, match="exp: argument must be dimensionless"):
        infer_signatures(ls.region, inputs, names=ls.names)


def test_region_grad_annotates_units():
    """unit(dL/dv) = unit(L)/unit(v), inferred on the gradient outputs."""

    def step(x, w):
        p = x * w
        m = reduce(red.sum, p, "i")
        return m

    inputs = {"x": T([2.0, 3.0], "i", "m"), "w": T([5.0, 7.0], "i", "s")}
    ls = lift_step(step, x=inputs["x"].layout, w=inputs["w"].layout)
    rg = grad(ls.region, ls.outputs[0], inputs, wrt=("x",), names=ls.names)
    _, got = run_region(rg.region, [inputs["x"], inputs["w"]])
    assert got.value_units == M * S / M
