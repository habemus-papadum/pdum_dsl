"""The model zoo: every entry's program matches its numpy denotation."""

import numpy as np
import pytest
from pdum.tl.autodiff import grad, numeric_grad
from pdum.tl.ir import Instr, Program, infer, run
from pdum.tl.zoo import (
    fdtd1d_staggered,
    flash_attention,
    gated_attention,
    gpt2,
    heat2d,
    llama_block,
    qknorm_attention,
    sliding_attention,
)

ENTRIES = {
    "gpt2": gpt2,
    "llama": llama_block,
    "sliding": sliding_attention,
    "gated": gated_attention,
    "qknorm": qknorm_attention,
    "flash": flash_attention,
    "flash_naive": lambda: flash_attention(naive=True),
    "heat2d": heat2d,
    "fdtd": fdtd1d_staggered,
}


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_zoo_forward_matches_numpy(name):
    m = ENTRIES[name]()
    env = run(m.program, m.inputs)
    got = env[m.out].to_numpy(order=m.order)
    np.testing.assert_allclose(got, m.ref(m.numpy_inputs()), rtol=1e-9, atol=1e-12)


def _with_loss(m):
    shadows = infer(m.program, m.inputs)
    dims = tuple(d.name for d in shadows[m.out].dims)
    return Program(
        m.program.instrs
        + (
            Instr("zsq", "pointwise", (m.out, m.out), {"f": "mul"}),
            Instr("zloss", "reduce", ("zsq",), {"f": "sum", "dims": dims}),
        )
    )


@pytest.mark.parametrize(
    ("name", "wrt"),
    [("gpt2", ("x", "L0.wq")), ("llama", ("x",)), ("heat2d", ("u0",)), ("fdtd", ("E0",))],
)
def test_zoo_gradients_match_fd(name, wrt):
    m = ENTRIES[name]()
    prog = _with_loss(m)
    jp, grads = grad(prog, "zloss", m.inputs)
    env = run(jp, m.inputs)
    for v in wrt:
        fd = numeric_grad(prog, "zloss", v, m.inputs)
        got = env[grads[v]].to_numpy(order=m.inputs[v].names)
        np.testing.assert_allclose(got, fd, rtol=3e-4, atol=1e-6)


def test_flash_equals_naive_forward_and_backward():
    # the online-softmax reducer IS softmax-then-contract — including its
    # DERIVED backward pass (composite-reducer BPTT, no hand rule anywhere)
    fl, nv = flash_attention(), flash_attention(naive=True)
    outs, gs = [], []
    for m in (fl, nv):
        prog = _with_loss(m)
        jp, grads = grad(prog, "zloss", m.inputs)
        env = run(jp, m.inputs)
        outs.append(env[m.out].to_numpy(order=m.order))
        gs.append({v: env[grads[v]].to_numpy(order=m.inputs[v].names) for v in ("q", "k", "v")})
    np.testing.assert_allclose(outs[0], outs[1], rtol=1e-9)
    for v in ("q", "k", "v"):
        np.testing.assert_allclose(gs[0][v], gs[1][v], rtol=1e-6, atol=1e-10)


def test_fdtd_gradient_carries_the_staggered_chart():
    m = fdtd1d_staggered()
    prog = _with_loss(m)
    jp, grads = grad(prog, "zloss", m.inputs)
    env = run(jp, m.inputs)
    gE = env[grads["E0"]]
    (xd,) = gE.layout.dims
    assert xd.chart == m.inputs["E0"].layout.dim("x").chart  # integer grid
    gH = env[grads["H0"]]
    assert gH.layout.dim("x").chart == m.inputs["H0"].layout.dim("x").chart  # half grid
