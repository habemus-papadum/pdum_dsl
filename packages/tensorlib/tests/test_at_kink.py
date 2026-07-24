"""The at-kink law, pinned at the kink points (200 §S.2 — frozen contract).

The derivative table is one-sided and PARTITIONS: at a tie exactly one
operand receives the cotangent — first-wins. Reduce adjoints derive through
the pairwise combine and inherit the partition. These are literal
expectations; a drifted selection is an API break.
"""

import numpy as np
from pdum.tl import Tensor
from pdum.tl.autodiff import grad
from pdum.tl.ir import Instr, Program, run


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def I(var, op, operands=(), **params):  # noqa: E743
    return Instr(var, op, tuple(operands), params)


def _grad_of(prog, target, inputs):
    joint, grads = grad(prog, target, inputs)
    env = run(joint, inputs)
    return {v: (None if gv is None else env[gv].to_numpy()) for v, gv in grads.items()}


def test_pointwise_maximum_tie_goes_left():
    prog = Program(
        (
            I("a", "input"),
            I("b", "input"),
            I("y", "pointwise", ("a", "b"), f="maximum"),
            I("s", "reduce", ("y",), f="sum", dims=("i",)),
        )
    )
    inputs = {"a": T([2.0, 5.0, 1.0], ("i",)), "b": T([2.0, 3.0, 4.0], ("i",))}
    g = _grad_of(prog, "s", inputs)
    np.testing.assert_allclose(g["a"], [1.0, 1.0, 0.0])  # the tie at i=0 goes LEFT
    np.testing.assert_allclose(g["b"], [0.0, 0.0, 1.0])  # never double-counted


def test_pointwise_minimum_tie_goes_left():
    prog = Program(
        (
            I("a", "input"),
            I("b", "input"),
            I("y", "pointwise", ("a", "b"), f="minimum"),
            I("s", "reduce", ("y",), f="sum", dims=("i",)),
        )
    )
    inputs = {"a": T([2.0, 5.0], ("i",)), "b": T([2.0, 3.0], ("i",))}
    g = _grad_of(prog, "s", inputs)
    np.testing.assert_allclose(g["a"], [1.0, 0.0])
    np.testing.assert_allclose(g["b"], [0.0, 1.0])


def test_reduce_max_all_ties_one_winner():
    prog = Program((I("x", "input"), I("m", "reduce", ("x",), f="max", dims=("i",))))
    x = T([7.0, 7.0, 7.0], ("i",))
    g = _grad_of(prog, "m", {"x": x})
    np.testing.assert_allclose(g["x"], [1.0, 0.0, 0.0])  # first along i, alone


def test_reduce_min_tie_first_wins():
    prog = Program((I("x", "input"), I("m", "reduce", ("x",), f="min", dims=("i",))))
    x = T([4.0, 1.0, 1.0], ("i",))
    g = _grad_of(prog, "m", {"x": x})
    np.testing.assert_allclose(g["x"], [0.0, 1.0, 0.0])


def test_reduce_max_two_dims_lexicographic_first():
    """Multi-dim reduce derives as a chain of single-dim reduces in declared
    order — the winner of an all-tie square is the lexicographically first
    element, and the partition never splits the cotangent."""
    prog = Program((I("x", "input"), I("m", "reduce", ("x",), f="max", dims=("i", "j"))))
    x = T(np.full((2, 2), 3.0), ("i", "j"))
    g = _grad_of(prog, "m", {"x": x})
    np.testing.assert_allclose(g["x"], [[1.0, 0.0], [0.0, 0.0]])
    assert float(np.sum(g["x"])) == 1.0  # a partition: total mass is exactly one


def test_reduce_max_gradient_mass_is_conserved():
    """Sum of the gradient equals the seed regardless of tie structure —
    the partition law's conservation face."""
    prog = Program((I("x", "input"), I("m", "reduce", ("x",), f="max", dims=("i",))))
    for data in ([3.0, 3.0, 1.0], [1.0, 3.0, 3.0], [3.0, 1.0, 3.0]):
        g = _grad_of(prog, "m", {"x": T(data, ("i",))})
        assert float(np.sum(g["x"])) == 1.0
