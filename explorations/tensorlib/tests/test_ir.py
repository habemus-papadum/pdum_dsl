"""The linear SSA IR: validation, interpretation, layout inference."""

import numpy as np
import pytest
from tensorlib import Tensor
from tensorlib.ir import Instr, Program, infer, run


def I(var, op, operands=(), **params):  # noqa: E743 - terse test helper
    return Instr(var, op, tuple(operands), params)


def test_ssa_validation():
    with pytest.raises(ValueError):
        Program((I("x", "input"), I("x", "input")))  # double assignment
    with pytest.raises(ValueError):
        Program((I("y", "flip", ["x"], name="i"),))  # undefined operand
    with pytest.raises(ValueError):
        Program((I("x", "frobnicate"),))  # unknown op


def test_matmul_program():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    bm = np.arange(12, dtype=np.float64).reshape(3, 4)
    prog = Program(
        (
            I("A", "input"),
            I("B", "input"),
            I("A3", "repeat", ["A"], name="n", extent=(0, 4)),
            I("B3", "repeat", ["B"], name="m", extent=(0, 2)),
            I("P", "pointwise", ["A3", "B3"], f="mul"),
            I("C", "reduce", ["P"], f="sum", dims=("k",)),
        )
    )
    env = run(
        prog,
        {"A": Tensor.from_numpy(a, ("m", "k")), "B": Tensor.from_numpy(bm, ("k", "n"))},
    )
    np.testing.assert_allclose(env["C"].to_numpy(order=("m", "n")), a @ bm)


def test_padded_conv_program_with_iota_free_masking():
    x = np.arange(1.0, 6.0)
    w = np.array([1.0, -1.0, 2.0])
    prog = Program(
        (
            I("x", "input"),
            I("w", "input"),
            I("xs", "stencil", ["x"], name="x", k=(-1, 1), fill=0.0),
            I("wk", "rename", ["w"], mapping={"k": "x_k"}),
            I("ws", "shift", ["wk"], deltas={"x_k": -1}),
            I("wr", "repeat", ["ws"], name="x", extent=(0, 5)),
            I("p", "pointwise", ["xs", "wr"], f="mul"),
            I("y", "reduce", ["p"], f="sum", dims=("x_k",)),
        )
    )
    env = run(prog, {"x": Tensor.from_numpy(x, ("x",)), "w": Tensor.from_numpy(w, ("k",))})
    expect = np.convolve(x, w[::-1], mode="same")
    np.testing.assert_allclose(env["y"].to_numpy(), expect)


def test_infer_matches_run_layouts():
    x = np.arange(8, dtype=np.float64)
    prog = Program(
        (
            I("x", "input"),
            I("s", "stencil", ["x"], name="x", k=(-1, 1), fill=0.0),
            I("i", "iota", ["s"], name="x_k"),
            I("b", "split", ["x"], name="x", parts={"xo": 2, "xi": 4}),
            I("r", "reduce", ["s"], f="sum", dims=("x_k",)),
        )
    )
    t = Tensor.from_numpy(x, ("x",))
    env = run(prog, {"x": t})
    shadows = infer(prog, {"x": t})
    for v in prog.vars:
        got = {(d.name, d.start, d.stop) for d in shadows[v].dims}
        want = {(d.name, d.start, d.stop) for d in env[v].layout.dims}
        assert got == want, v


def test_materialize_controls_order_and_enables_merge():
    arr = np.arange(8, dtype=np.float64)
    prog = Program(
        (
            I("x", "input"),
            I("b", "split", ["x"], name="x", parts={"xo": 2, "xi": 4}),
            I("c", "pointwise", ["b", "b"], f="add"),
            I("m", "materialize", ["c"], order=("xo", "xi")),
            I("y", "merge", ["m"], parts=("xo", "xi"), name="x"),
        )
    )
    env = run(prog, {"x": Tensor.from_numpy(arr, ("x",))})
    np.testing.assert_allclose(env["y"].to_numpy(), 2 * arr)
