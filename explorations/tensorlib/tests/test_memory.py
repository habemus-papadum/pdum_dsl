"""L1 peak-memory simulator: aliases, closed forms, schedules."""

import numpy as np
import pytest
from tensorlib import Tensor
from tensorlib.autodiff import grad
from tensorlib.ir import Instr, Program
from tensorlib.memory import peak_memory
from tensorlib.zoo import gpt2, heat2d


def I(var, op, operands=(), **params):  # noqa: E743
    return Instr(var, op, tuple(operands), params)


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


X8 = {"x": T(np.zeros(8), ("i",))}  # 64 bytes


def test_views_are_free_and_keep_their_root_alive():
    prog = Program(
        (
            I("x", "input"),
            I("a", "pointwise", ["x", "x"], f="mul"),  # 64
            I("s", "slice", ["a"], ranges={"i": (0, 4)}),  # view: 0 bytes
            I("c", "pointwise", ["s", "s"], f="mul"),  # 32; a must survive to here
        )
    )
    r = peak_memory(prog, X8)
    assert "s" not in r.alloc_bytes
    assert r.peak_bytes == 64 + 64 + 32 and r.peak_at == "c"
    rf = peak_memory(prog, X8, free_inputs=True)
    assert rf.peak_bytes == 64 + 64  # x dies after a; peak is at a


def test_masks_and_positions_cost_nothing():
    prog = Program(
        (
            I("x", "input"),
            I("it", "iota", ["x"], name="i"),
            I("c4", "const", [], value=4, dims=(("i", (0, 8)),), dtype="int64"),
            I("m", "pointwise", ["it", "c4"], f="lt"),
            I("z", "const", [], value=0.0, dims=(("i", (0, 8)),)),
            I("y", "pointwise", ["m", "x", "z"], f="where"),
        )
    )
    r = peak_memory(prog, X8)
    assert set(r.alloc_bytes) == {"x", "m", "y"}  # iota/consts absent entirely
    assert r.peak_bytes == 64 * 3


def test_the_schedule_moves_the_peak():
    prog = Program(
        (
            I("x", "input"),
            I("a", "pointwise", ["x", "x"], f="mul"),  # 64
            I("ra", "reduce", ["a"], f="sum", dims=("i",)),  # 8
            I("b", "pointwise", ["x", "x"], f="add"),  # 64
            I("rb", "reduce", ["b"], f="sum", dims=("i",)),  # 8
            I("out", "pointwise", ["ra", "rb"], f="mul"),  # 8
        )
    )
    good = peak_memory(prog, X8)
    bad = peak_memory(prog, X8, order=["x", "a", "b", "ra", "rb", "out"])
    assert good.peak_bytes == 144  # x + ra + b + rb, at rb
    assert bad.peak_bytes == 200  # x + a + b + ra live together
    assert bad.peak_at == "ra"
    with pytest.raises(ValueError, match="topological"):
        peak_memory(prog, X8, order=["x", "ra", "a", "b", "rb", "out"])


def test_fold_transient_counts_the_step():
    m = heat2d()
    r = peak_memory(m.program, m.inputs)
    u_bytes = 5 * 5 * 8
    assert r.peak_at == "uf"
    assert r.peak_bytes > 3 * u_bytes  # u0 + carry + step internals + out
    assert any(k == "(fold transient)" for k, _ in r.live_at_peak)


def test_gpt2_backward_needs_more_than_forward():
    m = gpt2()
    fwd = peak_memory(m.program, m.inputs)
    jp, _ = grad(m.program, m.out, m.inputs, seed="dL")
    joint = peak_memory(jp, {**m.inputs, "dL": T(np.zeros((4, 5)), ("t", "v"))})
    assert fwd.peak_bytes > fwd.input_bytes  # activations dominate inputs? sanity
    assert joint.peak_bytes > fwd.peak_bytes
    assert joint.input_bytes == fwd.input_bytes + 4 * 5 * 8  # + the seed
