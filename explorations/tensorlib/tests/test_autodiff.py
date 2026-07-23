"""Reverse-mode AD: every adjoint rule validated by finite differences."""

import numpy as np
import pytest
from tensorlib import Tensor
from tensorlib.autodiff import grad, numeric_grad
from tensorlib.ir import Instr, Program, run


def I(var, op, operands=(), **params):  # noqa: E743
    return Instr(var, op, tuple(operands), params)


def check(instrs, inputs, wrt, target=None, rtol=1e-4, atol=1e-7):
    prog = Program(tuple(instrs))
    target = target or prog.instrs[-1].var
    joint, grads = grad(prog, target, dict(inputs))
    assert grads[wrt] is not None
    env = run(joint, inputs)
    got = env[grads[wrt]].to_numpy(order=inputs[wrt].names)
    want = numeric_grad(prog, target, wrt, inputs)
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)
    return joint, grads


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


RNG = np.random.default_rng(7)


# ----------------------------------------------------------------------
# pointwise markers
# ----------------------------------------------------------------------


def test_grad_mul_div_exp_log():
    ins = [
        I("x", "input"),
        I("w", "input"),
        I("p", "pointwise", ["x", "w"], f="mul"),
        I("e", "pointwise", ["p"], f="exp"),
        I("l", "pointwise", ["e"], f="log"),
        I("d", "pointwise", ["l", "w"], f="div"),
        I("y", "reduce", ["d"], f="sum", dims=("i",)),
    ]
    inputs = {"x": T(RNG.uniform(1, 2, 4), ("i",)), "w": T(RNG.uniform(1, 2, 4), ("i",))}
    check(ins, inputs, "x")
    check(ins, inputs, "w")


def test_grad_maximum_and_where():
    ins = [
        I("a", "input"),
        I("bb", "input"),
        I("m", "pointwise", ["a", "bb"], f="maximum"),
        I("c", "pointwise", ["a", "bb"], f="lt"),
        I("w", "pointwise", ["c", "m", "a"], f="where"),
        I("y", "reduce", ["w"], f="sum", dims=("i",)),
    ]
    inputs = {"a": T([1.0, 5.0, 2.0, 7.0], ("i",)), "bb": T([3.0, 1.0, 6.0, 2.0], ("i",))}
    check(ins, inputs, "a")
    check(ins, inputs, "bb")


def test_fanout_accumulates():
    ins = [
        I("x", "input"),
        I("sq", "pointwise", ["x", "x"], f="mul"),
        I("y", "reduce", ["sq"], f="sum", dims=("i",)),
    ]
    inputs = {"x": T([1.0, -2.0, 3.0], ("i",))}
    joint, grads = check(ins, inputs, "x")
    env = run(joint, inputs)
    np.testing.assert_allclose(env[grads["x"]].to_numpy(), 2 * inputs["x"].to_numpy())


# ----------------------------------------------------------------------
# layout adjoints
# ----------------------------------------------------------------------


def test_grad_slice_and_pad():
    ins = [
        I("x", "input"),
        I("s", "slice", ["x"], ranges={"i": (1, 4)}),
        I("y", "reduce", ["s"], f="sum", dims=("i",)),
    ]
    joint, grads = check(ins, {"x": T(RNG.standard_normal(5), ("i",))}, "x")
    env = run(joint, {"x": T([1.0, 2, 3, 4, 5], ("i",))})
    np.testing.assert_allclose(env[grads["x"]].to_numpy(), [0, 1, 1, 1, 0])

    ins = [
        I("x", "input"),
        I("p", "pad", ["x"], fill=0.0, extents={"i": (-1, 4)}),
        I("w", "input"),
        I("m", "pointwise", ["p", "w"], f="mul"),
        I("y", "reduce", ["m"], f="sum", dims=("i",)),
    ]
    inputs = {"x": T(RNG.standard_normal(3), ("i",)), "w": T(RNG.standard_normal(5), ("i",)).shift(i=-1)}
    check(ins, {"x": inputs["x"], "w": T(RNG.standard_normal(5), ("i",)).shift(i=-1)}, "x")


def test_grad_relabelings():
    ins = [
        I("x", "input"),
        I("f", "flip", ["x"], name="i"),
        I("sh", "shift", ["f"], deltas={"i": 3}),
        I("r", "rename", ["sh"], mapping={"i": "j"}),
        I("w", "input"),
        I("wr", "rename", ["w"], mapping={"i": "j"}),
        I("ws", "shift", ["wr"], deltas={"j": 3}),
        I("m", "pointwise", ["r", "ws"], f="mul"),
        I("y", "reduce", ["m"], f="sum", dims=("j",)),
    ]
    inputs = {"x": T(RNG.standard_normal(4), ("i",)), "w": T(RNG.standard_normal(4), ("i",))}
    check(ins, inputs, "x")


def test_grad_repeat_and_select():
    ins = [
        I("x", "input"),
        I("r", "repeat", ["x"], name="n", extent=(0, 3)),
        I("w", "input"),
        I("m", "pointwise", ["r", "w"], f="mul"),
        I("y", "reduce", ["m"], f="sum", dims=("i", "n")),
    ]
    inputs = {
        "x": T(RNG.standard_normal(4), ("i",)),
        "w": T(RNG.standard_normal((4, 3)), ("i", "n")),
    }
    check(ins, inputs, "x")

    ins = [
        I("x", "input"),
        I("s", "select", ["x"], coords={"i": 2}),
        I("y", "reduce", ["s"], f="sum", dims=("j",)),
    ]
    joint, grads = check(ins, {"x": T(RNG.standard_normal((4, 3)), ("i", "j"))}, "x")


def test_grad_split_merge():
    ins = [
        I("x", "input"),
        I("b", "split", ["x"], name="i", parts={"io": 2, "ii": 3}),
        I("w", "input"),
        I("m", "pointwise", ["b", "w"], f="mul"),
        I("mo", "materialize", ["m"], order=("io", "ii")),
        I("g", "merge", ["mo"], parts=("io", "ii"), name="i"),
        I("y", "reduce", ["g"], f="sum", dims=("i",)),
    ]
    inputs = {
        "x": T(RNG.standard_normal(6), ("i",)),
        "w": T(RNG.standard_normal((2, 3)), ("io", "ii")),
    }
    check(ins, inputs, "x")
    check(ins, inputs, "w")


def test_grad_window_conv():
    ins = [
        I("x", "input"),
        I("xw", "window", ["x"], name="i", k_name="k", k=3),
        I("w", "input"),
        I("wr", "repeat", ["w"], name="i", extent=(0, 4)),
        I("m", "pointwise", ["xw", "wr"], f="mul"),
        I("y", "reduce", ["m"], f="sum", dims=("i", "k")),
    ]
    inputs = {
        "x": T(RNG.standard_normal(6), ("i",)),
        "w": T(RNG.standard_normal(3), ("k",)),
    }
    check(ins, inputs, "x")
    check(ins, inputs, "w")


def test_grad_stencil_discards_fill_cotangent():
    ins = [
        I("x", "input"),
        I("xs", "stencil", ["x"], name="i", k=(-1, 1), fill=0.0),
        I("w", "input"),
        I("wr", "repeat", ["w"], name="i", extent=(0, 5)),
        I("m", "pointwise", ["xs", "wr"], f="mul"),
        I("y", "reduce", ["m"], f="sum", dims=("i", "i_k")),
    ]
    inputs = {
        "x": T(RNG.standard_normal(5), ("i",)),
        "w": T(RNG.standard_normal(3), ("i_k",)).shift(i_k=-1),
    }
    check(ins, inputs, "x")


def test_grad_dilated_stencil():
    ins = [
        I("x", "input"),
        I("xs", "stencil", ["x"], name="i", k=(-1, 1), fill=0.0, dilation=2),
        I("y", "reduce", ["xs"], f="sum", dims=("i", "i_k")),
    ]
    check(ins, {"x": T(RNG.standard_normal(6), ("i",))}, "x")


def test_grad_decimate():
    ins = [
        I("x", "input"),
        I("d", "decimate", ["x"], name="i", factor=2, phase=1),
        I("w", "input"),
        I("m", "pointwise", ["d", "w"], f="mul"),
        I("y", "reduce", ["m"], f="sum", dims=("i",)),
    ]
    inputs = {
        "x": T(RNG.standard_normal(6), ("i",)),
        "w": T(RNG.standard_normal(3), ("i",)),
    }
    joint, grads = check(ins, inputs, "x")
    env = run(joint, inputs)
    g = env[grads["x"]].to_numpy()
    np.testing.assert_allclose(g[::2], 0.0)  # untouched phase gets zero


def test_grad_diagonal():
    ins = [
        I("x", "input"),
        I("d", "diagonal", ["x"], parts=("i", "j"), name="z"),
        I("w", "input"),
        I("m", "pointwise", ["d", "w"], f="mul"),
        I("y", "reduce", ["m"], f="sum", dims=("z",)),
    ]
    inputs = {
        "x": T(RNG.standard_normal((3, 3)), ("i", "j")),
        "w": T(RNG.standard_normal(3), ("z",)),
    }
    check(ins, inputs, "x")


# ----------------------------------------------------------------------
# reduce / scan
# ----------------------------------------------------------------------


def test_grad_reduce_mean_and_max():
    ins = [
        I("x", "input"),
        I("mu", "reduce", ["x"], f="mean", dims=("j",)),
        I("mx", "reduce", ["x"], f="max", dims=("j",)),
        I("s", "pointwise", ["mu", "mx"], f="mul"),
        I("y", "reduce", ["s"], f="sum", dims=("i",)),
    ]
    # distinct values: keep max-ties away from finite differences
    x = np.array([[1.0, 5.0, 2.0], [9.0, 3.0, 4.0]])
    check(ins, {"x": T(x, ("i", "j"))}, "x")


def test_grad_scan_sum():
    ins = [
        I("x", "input"),
        I("cs", "scan", ["x"], f="sum", dim="i"),
        I("w", "input"),
        I("m", "pointwise", ["cs", "w"], f="mul"),
        I("y", "reduce", ["m"], f="sum", dims=("i",)),
    ]
    inputs = {
        "x": T(RNG.standard_normal(5), ("i",)),
        "w": T(RNG.standard_normal(5), ("i",)),
    }
    check(ins, inputs, "x")


# ----------------------------------------------------------------------
# contracts and end-to-end
# ----------------------------------------------------------------------


def test_seed_contract():
    prog = Program((I("x", "input"), I("d", "pointwise", ["x", "x"], f="mul")))
    with pytest.raises(ValueError):
        grad(prog, "d", {"x": T([1.0, 2.0], ("i",))})  # non-scalar, no seed
    joint, grads = grad(prog, "d", {"x": T([1.0, 2.0], ("i",))}, seed="dY")
    env = run(joint, {"x": T([3.0, 4.0], ("i",)), "dY": T([1.0, 1.0], ("i",))})
    np.testing.assert_allclose(env[grads["x"]].to_numpy(), [6.0, 8.0])


def test_unreachable_vars_get_none():
    prog = Program(
        (
            I("x", "input"),
            I("u", "input"),
            I("y", "reduce", ["x"], f="sum", dims=("i",)),
        )
    )
    _, grads = grad(prog, "y", {"x": T([1.0], ("i",)), "u": T([1.0], ("i",))})
    assert grads["u"] is None and grads["x"] is not None


def test_grad_matmul_end_to_end():
    a = RNG.standard_normal((2, 3))
    bm = RNG.standard_normal((3, 4))
    ins = [
        I("A", "input"),
        I("B", "input"),
        I("A3", "repeat", ["A"], name="n", extent=(0, 4)),
        I("B3", "repeat", ["B"], name="m", extent=(0, 2)),
        I("P", "pointwise", ["A3", "B3"], f="mul"),
        I("C", "reduce", ["P"], f="sum", dims=("k",)),
        I("L", "reduce", ["C"], f="sum", dims=("m", "n")),
    ]
    inputs = {"A": T(a, ("m", "k")), "B": T(bm, ("k", "n"))}
    joint, grads = check(ins, inputs, "A")
    check(ins, inputs, "B")
    # analytic check: dL/dA = ones @ B^T
    env = run(joint, inputs)
    np.testing.assert_allclose(
        env[grads["A"]].to_numpy(order=("m", "k")),
        np.ones((2, 4)) @ bm.T,
        rtol=1e-6,
    )


def test_grad_softmax_cross_entropy():
    s = RNG.standard_normal((2, 4))
    onehot = np.zeros((2, 4))
    onehot[0, 1] = onehot[1, 3] = 1.0
    ins = [
        I("S", "input"),
        I("t", "input"),
        I("mx", "reduce", ["S"], f="max", dims=("v",)),
        I("mr", "repeat", ["mx"], name="v", extent=(0, 4)),
        I("sh", "pointwise", ["S", "mr"], f="sub"),
        I("e", "pointwise", ["sh"], f="exp"),
        I("z", "reduce", ["e"], f="sum", dims=("v",)),
        I("zr", "repeat", ["z"], name="v", extent=(0, 4)),
        I("lz", "pointwise", ["zr"], f="log"),
        I("lp", "pointwise", ["sh", "lz"], f="sub"),
        I("nll", "pointwise", ["lp", "t"], f="mul"),
        I("sum1", "reduce", ["nll"], f="sum", dims=("v",)),
        I("negl", "pointwise", ["sum1"], f="neg"),
        I("y", "reduce", ["negl"], f="sum", dims=("i",)),
    ]
    inputs = {"S": T(s, ("i", "v")), "t": T(onehot, ("i", "v"))}
    joint, grads = check(ins, inputs, "S", rtol=1e-3, atol=1e-5)
    # analytic: dL/dS = softmax(S) - onehot
    env = run(joint, inputs)
    sm = np.exp(s - s.max(1, keepdims=True))
    sm /= sm.sum(1, keepdims=True)
    np.testing.assert_allclose(env[grads["S"]].to_numpy(order=("i", "v")), sm - onehot, rtol=1e-5, atol=1e-8)


# ----------------------------------------------------------------------
# review regressions (adversarial + cleanliness findings)
# ----------------------------------------------------------------------


def test_grad_stencil_tap_entirely_outside_source():
    # far-top dilated taps used to emit an un-runnable pad; now: zero contribution
    ins = [
        I("x", "input"),
        I("xs", "stencil", ["x"], name="i", k=(0, 3), fill=0.0, dilation=2),
        I("y", "reduce", ["xs"], f="sum", dims=("i", "i_k")),
    ]
    check(ins, {"x": T(RNG.standard_normal(4), ("i",))}, "x")


def test_grad_diagonal_disjoint_parts_is_zero():
    ins = [
        I("x", "input"),
        I("sh", "shift", ["x"], deltas={"j": 5}),
        I("d", "diagonal", ["sh"], parts=("i", "j"), name="z"),
        I("y", "reduce", ["d"], f="sum", dims=("z",)),
    ]
    prog = Program(tuple(ins))
    x = T(RNG.standard_normal((3, 3)), ("i", "j"))
    joint, grads = grad(prog, "y", {"x": x})
    env = run(joint, {"x": x})
    np.testing.assert_allclose(env[grads["x"]].to_numpy(), np.zeros((3, 3)))


def test_reduce_max_tie_overcount_is_pinned():
    # documented caveat: every tied element receives the full cotangent
    ins = [
        I("x", "input"),
        I("m", "reduce", ["x"], f="max", dims=("i",)),
    ]
    prog = Program(tuple(ins))
    x = T([3.0, 3.0, 1.0], ("i",))
    joint, grads = grad(prog, "m", {"x": x})
    env = run(joint, {"x": x})
    np.testing.assert_allclose(env[grads["x"]].to_numpy(), [1.0, 1.0, 0.0])


def test_unknown_differentiable_marker_raises_not_silent_zero():
    from tensorlib.compute import Marker
    from tensorlib.ir import PW

    PW["sqrt_tmp"] = Marker("sqrt_tmp", np.sqrt)
    try:
        ins = [
            I("x", "input"),
            I("s", "pointwise", ["x"], f="sqrt_tmp"),
            I("y", "reduce", ["s"], f="sum", dims=("i",)),
        ]
        prog = Program(tuple(ins))
        with pytest.raises(NotImplementedError):
            grad(prog, "y", {"x": T([1.0, 4.0], ("i",))})
    finally:
        del PW["sqrt_tmp"]


def test_grad_contract_errors_are_clear():
    prog = Program((I("x", "input"), I("y", "reduce", ["x"], f="sum", dims=("i",))))
    with pytest.raises(KeyError):
        grad(prog, "nope", {"x": T([1.0], ("i",))})
    with pytest.raises(ValueError, match="collides"):
        grad(prog, "y", {"x": T([1.0], ("i",))}, seed="x")


def test_instr_params_are_snapshotted():
    shared = {"f": "sum", "dims": ("i",)}
    a = I("x", "input")
    bb = Instr("y", "reduce", ("x",), shared)
    shared["dims"] = ("corrupted",)  # caller mutation must not leak in
    prog = Program((a, bb))
    env = run(prog, {"x": T([1.0, 2.0], ("i",))})
    assert env["y"].item() == 3.0
    with pytest.raises(TypeError):
        bb.params["dims"] = ("nope",)  # mappingproxy refuses


def test_second_order_differentiation():
    ins = (
        I("x", "input"),
        I("x2", "pointwise", ["x", "x"], f="mul"),
        I("x3", "pointwise", ["x2", "x"], f="mul"),
        I("y", "reduce", ["x3"], f="sum", dims=("i",)),
    )
    x = T([1.0, -2.0, 3.0], ("i",))
    j1, g1 = grad(Program(ins), "y", {"x": x})
    p2 = Program(j1.instrs + (Instr("gy", "reduce", (g1["x"],), {"f": "sum", "dims": ("i",)}),))
    j2, g2 = grad(p2, "gy", {"x": x})
    env = run(j2, {"x": x})
    np.testing.assert_allclose(env[g1["x"]].to_numpy(), 3 * x.to_numpy() ** 2)
    np.testing.assert_allclose(env[g2["x"]].to_numpy(), 6 * x.to_numpy())


# ----------------------------------------------------------------------
# units in gradients (CONCERNS #19 resolution)
# ----------------------------------------------------------------------


def test_gradients_carry_primal_charts():
    from tensorlib import q

    x = T(RNG.standard_normal(5), ("i",)).with_charts(i=("0 nm", "25 nm"))
    ins = [
        I("x", "input"),
        I("xs", "stencil", ["x"], name="i", k=(-1, 1), fill=0.0),
        I("y", "reduce", ["xs"], f="sum", dims=("i", "i_k")),
    ]
    prog = Program(tuple(ins))
    joint, grads = grad(prog, "y", {"x": x})
    env = run(joint, {"x": x})
    g = env[grads["x"]]
    assert g.layout.dim("i").chart == x.layout.dim("i").chart
    # physical indexing of the gradient works: entry FOR the sample at 50 nm
    assert g.item(i=q("50 nm")) == g.item(i=2)
    want = numeric_grad(prog, "y", "x", {"x": x})  # chart-preserving FD
    np.testing.assert_allclose(g.to_numpy(), want, rtol=1e-4, atol=1e-7)


def test_select_compensation_paths_accumulate_when_charted():
    # the crux: one contribution flows through split+select (whose forward
    # promotes/compensates sibling charts), another directly — both must be
    # restamped onto x's chart and add cleanly
    x = T(RNG.standard_normal(6), ("i",)).with_charts(i=("0 um", "0.25 um"))
    ins = [
        I("x", "input"),
        I("b", "split", ["x"], name="i", parts={"ib": 3, "ii": 2}),
        I("s", "select", ["b"], coords={"ii": 1}),
        I("p", "reduce", ["s"], f="sum", dims=("ib",)),
        I("d", "reduce", ["x"], f="sum", dims=("i",)),
        I("y", "pointwise", ["p", "d"], f="add"),
    ]
    prog = Program(tuple(ins))
    joint, grads = grad(prog, "y", {"x": x})
    env = run(joint, {"x": x})
    g = env[grads["x"]]
    assert g.layout.dim("i").chart == x.layout.dim("i").chart
    np.testing.assert_allclose(g.to_numpy(), [1, 2, 1, 2, 1, 2])  # 1 + phase-hit
    want = numeric_grad(prog, "y", "x", {"x": x})
    np.testing.assert_allclose(g.to_numpy(), want, rtol=1e-4, atol=1e-7)


def test_gradients_carry_primal_labels():
    x = T(RNG.standard_normal((3, 4)), ("c", "i")).with_labels(c=("R", "G", "B"))
    ins = [
        I("x", "input"),
        I("sq", "pointwise", ["x", "x"], f="mul"),
        I("y", "reduce", ["sq"], f="sum", dims=("c", "i")),
    ]
    joint, grads = grad(Program(tuple(ins)), "y", {"x": x})
    env = run(joint, {"x": x})
    g = env[grads["x"]]
    assert g.layout.dim("c").labels == ("R", "G", "B")
    np.testing.assert_allclose(g.item(c="G", i=1), 2 * x.item(c="G", i=1), rtol=1e-12)


def test_gradient_value_units():
    from tensorlib import u

    volt = u.define("V_ad", dim="voltage_ad")
    x = T(RNG.standard_normal(4), ("i",)).with_value_units(volt)
    w = T(RNG.standard_normal(4), ("i",))  # no declared unit
    ins = [
        I("x", "input"),
        I("w", "input"),
        I("p", "pointwise", ["x", "w"], f="mul"),
        I("L", "reduce", ["p"], f="sum", dims=("i",)),
    ]
    joint, grads = grad(Program(tuple(ins)), "L", {"x": x, "w": w}, target_unit=volt * volt)
    env = run(joint, {"x": x, "w": w})
    assert env[grads["x"]].value_units == volt  # V²/V
    assert env[grads["w"]].value_units == volt * volt  # V²/1
    # without target_unit: no annotation
    j2, g2 = grad(Program(tuple(ins)), "L", {"x": x, "w": w})
    assert run(j2, {"x": x, "w": w})[g2["x"]].value_units is None


def test_charted_window_gradient_charts_flow_naturally():
    x = T(RNG.standard_normal(6), ("t",)).with_charts(t=("0 ms", "1 ms"))
    ins = [
        I("x", "input"),
        I("w", "window", ["x"], name="t", k_name="k", k=3),
        I("y", "reduce", ["w"], f="sum", dims=("t", "k")),
    ]
    prog = Program(tuple(ins))
    joint, grads = grad(prog, "y", {"x": x})
    env = run(joint, {"x": x})
    g = env[grads["x"]]
    assert g.layout.dim("t").chart == x.layout.dim("t").chart
    np.testing.assert_allclose(g.to_numpy(), [1, 2, 3, 3, 2, 1])  # overlap counts
