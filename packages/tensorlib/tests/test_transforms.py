"""L1 transformations: requested-gradients DCE + min-cut checkpointing."""

import numpy as np
import pytest
from pdum.tl import Tensor, peak_memory
from pdum.tl.autodiff import grad
from pdum.tl.ir import Instr, Program, run
from pdum.tl.transforms import checkpoint, dce
from pdum.tl.zoo import gpt2


def I(var, op, operands=(), **params):  # noqa: E743
    return Instr(var, op, tuple(operands), params)


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def _loss_prog(m):
    return Program(
        m.program.instrs
        + (
            I("zsq", "pointwise", (m.out, m.out), f="mul"),
            I("zloss", "reduce", ("zsq",), f="sum", dims=("t", "v")),
        )
    )


def test_dce_keeps_exactly_the_requested_slice():
    prog = Program(
        (
            I("x", "input"),
            I("a", "pointwise", ["x", "x"], f="mul"),
            I("b", "pointwise", ["x", "x"], f="add"),  # dead wrt 'ra'
            I("ra", "reduce", ["a"], f="sum", dims=("i",)),
            I("rb", "reduce", ["b"], f="sum", dims=("i",)),
        )
    )
    pruned = dce(prog, ("ra",))
    assert pruned.vars == ("x", "a", "ra")
    x = {"x": T(np.arange(4.0), ("i",))}
    assert run(pruned, x)["ra"].item() == run(prog, x)["ra"].item()
    with pytest.raises(KeyError):
        dce(prog, ("nope",))


def test_dce_prunes_unrequested_gradient_work():
    m = gpt2()
    prog = _loss_prog(m)
    jp, grads = grad(prog, "zloss", m.inputs)
    keep_x = dce(jp, (grads["x"], "zloss"))
    assert len(keep_x.instrs) < len(jp.instrs)  # weight-grad work is gone
    ex, ej = run(keep_x, m.inputs), run(jp, m.inputs)
    np.testing.assert_allclose(
        ex[grads["x"]].to_numpy(order=("t", "d")), ej[grads["x"]].to_numpy(order=("t", "d")), rtol=1e-12
    )
    assert peak_memory(keep_x, m.inputs).peak_bytes <= peak_memory(jp, m.inputs).peak_bytes


def _chain():
    # x -> sq -> e (exp, big) -> r (reduce, banned) ; loss = r*r
    prog = Program(
        (
            I("x", "input"),
            I("sq", "pointwise", ["x", "x"], f="mul"),
            I("e", "pointwise", ["sq"], f="exp"),
            I("r", "reduce", ["e"], f="sum", dims=("i",)),
            I("loss", "pointwise", ["r", "r"], f="mul"),
        )
    )
    return prog, {"x": T(0.1 * np.arange(8.0), ("i",))}


def test_checkpoint_recomputes_the_cheap_chain():
    prog, inputs = _chain()
    jp, grads = grad(prog, "loss", inputs)
    ck = checkpoint(jp, "loss", inputs)
    # pointwise sq/e recompute from the (free-to-keep) input; only the
    # banned reduce output r must be saved — and it is 8 bytes
    assert [v for v, _ in ck.saved] == ["r"]
    assert ck.bytes_after == 8
    assert ck.bytes_before > ck.bytes_after
    assert any(v in ck.recomputed for v in ("sq", "e"))
    ej, ec = run(jp, inputs), run(ck.program, inputs)
    np.testing.assert_allclose(ec[grads["x"]].to_numpy(), ej[grads["x"]].to_numpy(), rtol=1e-12)


def test_checkpoint_recompute_everything_mode():
    prog, inputs = _chain()
    jp, grads = grad(prog, "loss", inputs)
    ck = checkpoint(jp, "loss", inputs, ban=())
    assert ck.bytes_after == 0  # everything re-derives from the input
    ej, ec = run(jp, inputs), run(ck.program, inputs)
    np.testing.assert_allclose(ec[grads["x"]].to_numpy(), ej[grads["x"]].to_numpy(), rtol=1e-12)


def test_banned_ops_read_by_backward_are_saved():
    prog = Program(
        (
            I("x", "input"),
            I("s", "scan", ["x"], f="sum", dim="i"),
            I("ss", "pointwise", ["s", "s"], f="mul"),
            I("loss", "reduce", ["ss"], f="sum", dims=("i",)),
        )
    )
    inputs = {"x": T(np.arange(5.0), ("i",))}
    jp, grads = grad(prog, "loss", inputs)
    ck = checkpoint(jp, "loss", inputs)
    assert "s" in [v for v, _ in ck.saved]  # scan is banned from recompute
    ej, ec = run(jp, inputs), run(ck.program, inputs)
    np.testing.assert_allclose(ec[grads["x"]].to_numpy(), ej[grads["x"]].to_numpy(), rtol=1e-12)


def test_checkpoint_gpt2_shrinks_the_boundary_and_the_peak():
    m = gpt2()
    prog = _loss_prog(m)
    jp, grads = grad(prog, "zloss", m.inputs)
    ck = checkpoint(jp, "zloss", m.inputs)
    assert ck.bytes_after < ck.bytes_before
    before = peak_memory(jp, m.inputs).peak_bytes
    after = peak_memory(ck.program, m.inputs).peak_bytes
    assert after <= before
    ej, ec = run(jp, m.inputs), run(ck.program, m.inputs)
    for v in ("x", "h.0.attn.wq"):
        np.testing.assert_allclose(
            ec[grads[v]].to_numpy(order=m.inputs[v].names),
            ej[grads[v]].to_numpy(order=m.inputs[v].names),
            rtol=1e-10,
        )


def test_dce_then_checkpoint_compose():
    m = gpt2()
    prog = _loss_prog(m)
    jp, grads = grad(prog, "zloss", m.inputs)
    pruned = dce(jp, (grads["x"], "zloss"))
    ck = checkpoint(pruned, "zloss", m.inputs)
    ej, ec = run(jp, m.inputs), run(ck.program, m.inputs)
    np.testing.assert_allclose(
        ec[grads["x"]].to_numpy(order=("t", "d")), ej[grads["x"]].to_numpy(order=("t", "d")), rtol=1e-10
    )
    # the composition is the point: prune first, then plan what remains
    assert ck.bytes_before <= checkpoint(jp, "zloss", m.inputs).bytes_before
