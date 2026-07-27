"""The ops-count model: named buckets, cost models, MAC fusion."""

import numpy as np
from pdum.tl import Tensor, defmarker, defreducer, ops_count
from pdum.tl.ir import Instr, Program
from pdum.tl.mdsl import exp as sym_exp


def I(var, op, operands=(), **params):  # noqa: E743
    return Instr(var, op, tuple(operands), params)


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


M_, K_, N_ = 3, 4, 5


def _matmul():
    prog = Program(
        (
            I("x", "input"),
            I("w", "input"),
            I("xr", "repeat", ["x"], name="n", extent=(0, N_)),
            I("wr", "repeat", ["w"], name="m", extent=(0, M_)),
            I("p", "pointwise", ["xr", "wr"], f="mul"),
            I("y", "reduce", ["p"], f="sum", dims=("k",)),
        )
    )
    inputs = {"x": T(np.zeros((M_, K_)), ("m", "k")), "w": T(np.zeros((K_, N_)), ("k", "n"))}
    return prog, inputs


def test_matmul_counts_muls_and_adds_separately():
    ops = ops_count(*_matmul())
    assert ops.total["mul"] == M_ * K_ * N_
    assert ops.total["add"] == M_ * N_ * (K_ - 1)


def test_mac_fusion_recognizes_the_contraction_pattern():
    ops = ops_count(*_matmul(), fuse_mac=True)
    assert ops.total["mac"] == M_ * K_ * N_
    assert ops.total["mul"] == 0 and ops.total["add"] == 0


def test_mac_fusion_refuses_when_the_product_is_observed_elsewhere():
    prog, inputs = _matmul()
    prog = Program(prog.instrs + (I("p2", "pointwise", ["p", "p"], f="add"),))
    ops = ops_count(prog, inputs, fuse_mac=True)
    assert ops.total["mac"] == 0  # p has two consumers; no fusion


def test_composite_pointwise_counts_by_tree():
    defmarker("sig_oc", 1, lambda x: 1 / (1 + sym_exp(-x)))
    n = 4
    prog = Program((I("x", "input"), I("s", "pointwise", ["x"], f="sig_oc")))
    ops = ops_count(prog, {"x": T(np.zeros(n), ("i",))})
    assert ops.per_var["s"] == {"div": n, "add": n, "exp": n, "neg": n}
    # exp's cost is the model's opinion, not the count's
    assert ops.weighted({"exp": 20.0}) == 3 * n + 20.0 * n


def test_scan_counts_folds_not_elements():
    prog = Program((I("x", "input"), I("s", "scan", ["x"], f="sum", dim="t")))
    ops = ops_count(prog, {"x": T(np.zeros((3, 5)), ("b", "t"))})
    assert ops.total["add"] == 3 * (5 - 1)


def test_composite_scan_counts_lift_combine_project():
    defreducer(
        "lr_oc",
        state=2,
        element=2,
        lift=lambda a, b: (a, b),
        combine=lambda left, right: (left[0] * right[0], right[0] * left[1] + right[1]),
        init=(1.0, 0.0),
        project=lambda A, B: B,
    )
    n = 6
    prog = Program(
        (
            I("a", "input"),
            I("b", "input"),
            I("h", "scan", ["a", "b"], f="lr_oc", dim="t"),
        )
    )
    ops = ops_count(prog, {"a": T(np.zeros(n), ("t",)), "b": T(np.zeros(n), ("t",))})
    # combine = 2 muls + 1 add per fold; identity lift/project cost nothing
    assert ops.per_var["h"] == {"mul": 2 * (n - 1), "add": n - 1}


def test_materialize_counts_copies_in_their_own_bucket():
    prog = Program((I("x", "input"), I("m", "materialize", ["x"], order=("i",))))
    ops = ops_count(prog, {"x": T(np.zeros(7), ("i",))})
    assert ops.total == {"copy": 7}
    assert ops.weighted({"copy": 0.0}) == 0.0
