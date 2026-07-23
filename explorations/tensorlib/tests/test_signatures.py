"""The carrier/unit signature pass: inference, refusals, target_unit."""

import numpy as np
import pytest
from tensorlib import (
    SignatureError,
    Tensor,
    VInfo,
    defmarker,
    infer_signatures,
    marker_signature,
    pointwise,
    pw,
    u,
)
from tensorlib.autodiff import grad
from tensorlib.ir import Instr, Program, run
from tensorlib.units import ONE


def I(var, op, operands=(), **params):  # noqa: E743
    return Instr(var, op, tuple(operands), params)


def T(arr, names, unit=None):
    t = Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)
    return t.with_value_units(u.parse_unit(unit)) if unit else t


M = u.parse_unit("m")
S = u.parse_unit("s")


def _mul_sum_prog():
    return Program(
        (
            I("x", "input"),
            I("w", "input"),
            I("p", "pointwise", ["x", "w"], f="mul"),
            I("y", "reduce", ["p"], f="sum", dims=("i",)),
        )
    )


def test_units_flow_through_mul_and_sum():
    sigs = infer_signatures(_mul_sum_prog(), {"x": T([1.0], "i", "m"), "w": T([1.0], "i", "s")})
    assert sigs["p"].unit == M * S
    assert sigs["y"].unit == M * S


def test_unlabeled_programs_stay_unknown():
    sigs = infer_signatures(_mul_sum_prog(), {"x": T([1.0], "i"), "w": T([1.0], "i")})
    assert sigs["y"] == VInfo("real", None)


def test_grad_infers_target_unit():
    # CONCERNS #16's payoff: no target_unit argument — the pass reads it
    inputs = {"x": T([2.0, 3.0], "i", "m"), "w": T([5.0, 7.0], "i", "s")}
    jp, grads = grad(_mul_sum_prog(), "y", inputs)
    env = run(jp, inputs)
    assert env[grads["x"]].value_units == S  # (m·s)/m
    assert env[grads["w"]].value_units == M
    np.testing.assert_allclose(env[grads["x"]].to_numpy(), [5.0, 7.0])


def test_conflicting_units_refuse_in_pass_and_in_grad():
    prog = Program(
        (
            I("x", "input"),
            I("w", "input"),
            I("p", "pointwise", ["x", "w"], f="add"),
            I("y", "reduce", ["p"], f="sum", dims=("i",)),
        )
    )
    inputs = {"x": T([1.0], "i", "m"), "w": T([1.0], "i", "s")}
    with pytest.raises(SignatureError, match="unit mismatch"):
        infer_signatures(prog, inputs)
    with pytest.raises(SignatureError):
        grad(prog, "y", inputs)


def test_exp_of_dimensioned_refuses_statically():
    prog = Program((I("x", "input"), I("e", "pointwise", ["x"], f="exp")))
    with pytest.raises(SignatureError, match="dimensionless"):
        infer_signatures(prog, {"x": T([1.0], "i", "m")})


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
    prog = Program(
        (
            I("a", "input"),
            I("b", "input"),
            I("h", "scan", ["a", "b"], f="linrec_t", dim="t"),
        )
    )
    import tests.test_mdsl  # noqa: F401  (registers linrec_t)

    sigs = infer_signatures(prog, {"a": T([0.5], "t"), "b": T([1.0], "t", "m")})
    assert sigs["h"].unit == M


def test_prod_of_dimensioned_refuses():
    prog = Program((I("x", "input"), I("y", "reduce", ["x"], f="prod", dims=("i",))))
    with pytest.raises(SignatureError, match="static extent"):
        infer_signatures(prog, {"x": T([1.0], "i", "m")})
    assert infer_signatures(prog, {"x": T([1.0], "i")})["y"].unit is None


def test_content_addressed_defmarker_dedupes():
    m1 = defmarker(None, 1, lambda x: x * x + 1)
    m2 = defmarker(None, 1, lambda x: x * x + 1)
    assert m1 is m2
    assert m1.name.startswith("m_")
