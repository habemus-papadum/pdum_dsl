"""The tl dialect's battery (240 C4.1) — promoted from the C3 spike.

The dialect now lives in ``pdum/tl/dialect.py``; these are its pins, kept
verbatim from the spike where possible: the kernel differential against
the incumbent machinery (bit-identical), alignment as a type rule with
source locations, content-keyed lowering, the per-kind yield protocol,
the two-pass refusal, the fold forward/adjoint differentials, and the
recompute theorem through the dialect (revolve ≡ store-all,
bit-identical, dropout on)."""

import numpy as np
import pytest
from pdum.dsl.ir import Builder, Region
from pdum.tl import Tensor, compute, thread_idx  # noqa: F401 — thread_idx: bodies' global
from pdum.tl.compute import const_like, pointwise  # noqa: F401 — S.1 spellings in bodies
from pdum.tl.dialect import (
    CORE_OPS,
    TL_OPS,
    TensorType,
    check_fold_step_supported,
    derive_step_vjp,
    fold_grad,
    lower_body,
    run_region,
    tensor_type,
)
from pdum.tl.markers import maximum, tanh, where  # noqa: F401 — bare in bodies, BOTH engines


@compute
def spike_kernel(img):
    y, x = thread_idx("y", "x")
    v = maximum(y, x) * 0.25 + (y - x) * 0.5
    img[y, x] = v * v + 1.5


def test_the_dialect_differential_is_bit_identical():
    """The same body, two machines: today's _KernelLowerer + ir.run vs the
    dsl Lowerer with the tl pack + run_region — identical bytes."""
    a = Tensor.from_numpy(np.zeros((5, 7)), ("y", "x"))
    spike_kernel(a)  # the incumbent tl machinery
    b = Tensor.from_numpy(np.zeros((5, 7)), ("y", "x"))
    region = lower_body(spike_kernel.fn, (tensor_type(b),), kind="compute")
    run_region(region, [b])
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_alignment_is_a_type_rule_and_refuses_with_locations():
    """tl's alignment law enforced by the dsl TYPE machinery: a misaligned
    store refuses AT EMISSION, with the source point in the message."""

    def bad(img, other):
        y, x = thread_idx("y", "x")
        img[y, x] = other * 1.0  # a different lattice: misaligned

    img = Tensor.from_numpy(np.zeros((4, 4)), ("y", "x"))
    other = Tensor.from_numpy(np.zeros((3, 3)), ("y", "x"))
    with pytest.raises(TypeError, match=r"aligned to the target.*\[.*test_dialect"):
        lower_body(bad, (tensor_type(img), tensor_type(other)), kind="compute")


def test_lowering_is_content_keyed():
    """fp-keyed IR: two independent lowerings of the same body produce
    content-identical Regions — the cache-efficiency premise holds."""
    a = Tensor.from_numpy(np.zeros((5, 7)), ("y", "x"))
    r1 = lower_body(spike_kernel.fn, (tensor_type(a),), kind="compute")
    r2 = lower_body(spike_kernel.fn, (tensor_type(a),), kind="compute")
    assert r1.key == r2.key


def test_the_yield_protocol_refuses_kernel_returns():
    """Kinds declare their yields: a compute body yields its token; a
    ``return`` in one refuses with the pinned voice."""

    def bad(img):
        (y,) = thread_idx("y")
        return img

    img = Tensor.from_numpy(np.zeros(3), ("y",))
    with pytest.raises(ValueError, match="kernels return nothing"):
        lower_body(bad, (tensor_type(img),), kind="compute")


# --- fold: the hard part, promoted -------------------------------------------


def _theorem_step(s, m):
    # ONE source, two engines — spelled in the ratified S.1 STEP style
    kept = pointwise(where, m < 0.4, const_like(s, 0.0), s)
    return s * 0.8 + pointwise(tanh, kept) * 0.3


def _theorem_setup():
    from pdum.tl import fold_in, uniform

    N, TM = 4, 6
    rng = np.random.default_rng(2)
    s0 = Tensor.from_numpy(rng.standard_normal(N), ("x",))
    lattice = Tensor.from_numpy(np.zeros((TM, N)), ("tm", "x"))
    mask = uniform(fold_in(5, "train.drop"), lattice.layout)  # zero-memory FIELD
    return s0, mask, TM


def _incumbent_fold_program(s0):
    from pdum.tl.ir import Instr, Program
    from pdum.tl.lifting import lift_step

    ls = lift_step(_theorem_step, s=s0.layout, m=Tensor.from_numpy(np.zeros(4), ("x",)).layout)
    fold_params = {
        "step": ls.program,
        "dim": "tm",
        "state": ("s",),
        "element": ("m",),
        "carry": {"s": ls.outputs[0]},
        "out": ("final", ls.outputs[0]),
    }
    return Program(
        (
            Instr("s0", "input", (), {}),
            Instr("mask", "input", (), {}),
            Instr("sf", "fold", ("s0", "mask"), fold_params),
            Instr("zloss", "reduce", ("sf",), {"f": "sum", "dims": ("x",)}),
        )
    )


def test_b1_fold_forward_differential():
    """The fold forward through the dialect ≡ the tl fold under ir.run —
    the same single-source step function, two engines, identical bytes."""
    from pdum.tl.ir import run

    s0, mask, _ = _theorem_setup()
    step = lower_body(_theorem_step, (tensor_type(s0), tensor_type(s0)), kind="step")
    check_fold_step_supported(step)  # pass 1
    b = Builder({**CORE_OPS, **TL_OPS})
    src_tt = TensorType(tuple((d.name, d.start, d.stop) for d in mask.layout.dims))
    p_s, p_src = b.param(0, tensor_type(s0)), b.param(1, src_tt)
    fold = b.emit("tl.fold", p_s, p_src, regions=(step,), dim="tm")
    region = Region(params=(p_s, p_src), body=(b.emit("core.yield", fold),))
    got = run_region(region, [s0, mask])
    want = run(_incumbent_fold_program(s0), {"s0": s0, "mask": mask})["sf"]
    np.testing.assert_array_equal(got.to_numpy(order=("x",)), want.to_numpy(order=("x",)))


def test_b2_store_all_adjoint_differential():
    """The region-derived VJP under store-all ≡ tl's autodiff gradient —
    same step, same field, gradient wrt the initial state."""
    from pdum.tl.autodiff import grad
    from pdum.tl.ir import run

    s0, mask, _ = _theorem_setup()
    step = lower_body(_theorem_step, (tensor_type(s0), tensor_type(s0)), kind="step")
    vjp = derive_step_vjp(step)
    got = fold_grad(step, vjp, s0, mask, "tm")
    prog = _incumbent_fold_program(s0)
    joint, grads = grad(prog, "zloss", {"s0": s0, "mask": mask}, fold_segments=1)
    want = run(joint, {"s0": s0, "mask": mask})[grads["s0"]]
    np.testing.assert_allclose(got.to_numpy(order=("x",)), want.to_numpy(order=("x",)), rtol=1e-12)


def test_b3_revolve_is_bit_identical_with_the_field_on():
    """THE THEOREM through the dialect: revolve (recompute from checkpoints,
    re-selecting the zero-memory field at absolute coordinates) and
    store-all produce BIT-IDENTICAL gradients — dropout on, no mask stored."""
    from pdum.tl.random import RandomBuffer

    s0, mask, _ = _theorem_setup()
    assert isinstance(mask.buffer, RandomBuffer) and mask.buffer.data is None  # zero bytes
    step = lower_body(_theorem_step, (tensor_type(s0), tensor_type(s0)), kind="step")
    vjp = derive_step_vjp(step)
    g_all = fold_grad(step, vjp, s0, mask, "tm")
    g_rev = fold_grad(step, vjp, s0, mask, "tm", slots=2)
    np.testing.assert_array_equal(g_rev.to_numpy(order=("x",)), g_all.to_numpy(order=("x",)))
    assert not np.array_equal(g_all.to_numpy(order=("x",)), np.zeros(4))


def test_b_pass1_refuses_unsupported_shapes_with_the_reason():
    """The two-pass mechanism's first pass: an unsupported op in a step
    refuses NAMING the op, the supported set, and why."""
    b = Builder({**CORE_OPS, **TL_OPS})
    tt = TensorType((("x", 0, 4),))
    p = b.param(("st", 0), tt)
    rogue = b.emit("tl.iota", p, name="x")
    bad = Region(params=(p, b.param(("st", 1), tt)), body=(b.emit("core.yield", rogue),))
    with pytest.raises(TypeError, match=r"contains 'tl.iota'.*does not support yet.*no region rule"):
        check_fold_step_supported(bad)
