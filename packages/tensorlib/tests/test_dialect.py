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
from pdum.tl import Tensor, compute, f32, thread_idx  # noqa: F401 — ambient vocabulary: bodies' globals
from pdum.tl.compute import const_like, pointwise, reduce, repeat_like  # noqa: F401 — S.1 spellings in bodies
from pdum.tl.dialect import (
    CORE_OPS,
    TL_OPS,
    check_fold_step_supported,
    derive_step_vjp,
    fold_grad,
    lower_body,
    run_region,
    tensor_type,
)
from pdum.tl.markers import maximum, red, tanh, where  # noqa: F401 — bare in bodies, BOTH engines


@compute
def spike_kernel(img):
    y, x = thread_idx("y", "x")
    v = maximum(f32(y), f32(x)) * 0.25 + (f32(y) - f32(x)) * 0.5
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


def test_the_labeling_frame_is_identity_and_misaligns_at_emission():
    """C4 tail: charts/labels/levels are IN type identity — same dims,
    different frames refuse AT EMISSION, quoting the incumbent alignment
    diagnosis (tl's own fix recipes), with the source point."""

    def bad(img, plain, charted):
        y, x = thread_idx("y", "x")
        img[y, x] = plain * charted

    img = Tensor.from_numpy(np.zeros((2, 3)), ("y", "x"))
    plain = Tensor.from_numpy(np.ones((2, 3)), ("y", "x"))
    charted = plain.with_charts(x=("0 um", "0.25 um"))
    with pytest.raises(TypeError, match=r"(?s)ALIGNED.*charted.*\[.*test_dialect"):
        lower_body(bad, (tensor_type(img), tensor_type(plain), tensor_type(charted)), kind="compute")


def test_content_keys_distinguish_frames():
    """One body, same dims, different labeling frames: DIFFERENT region
    keys — so a content cache can never serve a charted lattice an
    artifact lowered for a plain one."""
    plain = Tensor.from_numpy(np.zeros((5, 7)), ("y", "x"))
    charted = Tensor.from_numpy(np.zeros((5, 7)), ("y", "x")).with_charts(x=("0 um", "1 um"))
    r1 = lower_body(spike_kernel.fn, (tensor_type(plain),), kind="compute")
    r2 = lower_body(spike_kernel.fn, (tensor_type(charted),), kind="compute")
    assert r1.key != r2.key


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
    src_tt = tensor_type(mask)
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
    tt = tensor_type(Tensor.from_numpy(np.zeros(4), ("x",)))
    p = b.param(("st", 0), tt)
    rogue = b.emit("tl.iota", p, name="x")
    bad = Region(params=(p, b.param(("st", 1), tt)), body=(b.emit("core.yield", rogue),))
    with pytest.raises(TypeError, match=r"contains 'tl.iota'.*does not support yet.*no region rule"):
        check_fold_step_supported(bad)


# --- C4.3a: the step-tier op families, bridged -------------------------------


def test_step_layernorm_differential_single_source():
    """THE flagship C4.3a differential: the ZOO's layernorm — verbatim,
    untouched — lowered through the dialect (reduce/repeat_like/pointwise
    bridged onto infer_instr/eval_instr) vs the same function run EAGERLY
    on tensors (the S.1 denotational reference). Identical bytes."""
    from pdum.tl.zoo.zoo_common import layernorm

    rng = np.random.default_rng(7)
    x = Tensor.from_numpy(rng.standard_normal((5, 8)), ("t", "d"))
    g = Tensor.from_numpy(rng.standard_normal(8), ("d",))
    b = Tensor.from_numpy(rng.standard_normal(8), ("d",))
    want = layernorm(x, g, b, feat="d", eps=1e-5)  # eager: the reference
    region = lower_body(
        layernorm,
        (tensor_type(x), tensor_type(g), tensor_type(b)),
        kind="step",
        host={"feat": "d", "eps": 1e-5},
    )
    got = run_region(region, [x, g, b])
    np.testing.assert_array_equal(got.to_numpy(order=("t", "d")), want.to_numpy(order=("t", "d")))


def test_step_layout_chain_differential():
    """The layout-method family through the dialect: shift/slice/pad as
    tl.* ops whose type rules ARE the incumbent shadow inference — vs the
    same chain evaluated eagerly. Identical bytes."""

    def stencil(E):
        dE = E.shift(x=-1).slice(x=(0, 9)) - E.slice(x=(0, 9))
        return dE.pad(x=(0, 10), fill=0.0)

    rng = np.random.default_rng(3)
    E = Tensor.from_numpy(rng.standard_normal(10), ("x",))
    want = stencil(E)  # eager: Tensor methods
    region = lower_body(stencil, (tensor_type(E),), kind="step")
    got = run_region(region, [E])
    np.testing.assert_array_equal(got.to_numpy(order=("x",)), want.to_numpy(order=("x",)))


def test_structural_slots_refuse_tensors_with_the_annotation_fix():
    """A tensor reaching a structural slot (a method parameter) refuses
    with the lifting doctrine's message — the same law, the new engine."""

    def bad(E):
        return E.slice(x=(0, E))  # a tensor in a structural slot

    E = Tensor.from_numpy(np.zeros(4), ("x",))
    with pytest.raises(ValueError, match="STRUCTURAL slot"):
        lower_body(bad, (tensor_type(E),), kind="step")


# --- C4.3b: the general region VJP -------------------------------------------


def test_step_layernorm_grad_differential():
    """THE C4.3b flagship: the GENERAL region VJP (pointwise via the one
    table + reduce sum/mean + repeat_like adjoints) differentiates the
    zoo's layernorm — gradients wrt x, g, AND b match the incumbent
    autodiff engine to 1e-12 (cross-engine summation order)."""
    from pdum.tl.autodiff import grad
    from pdum.tl.dialect import derive_vjp
    from pdum.tl.ir import Instr, Program, run
    from pdum.tl.lifting import lift_step
    from pdum.tl.zoo.zoo_common import layernorm

    def ln(x, g, b):
        return layernorm(x, g, b, feat="d", eps=1e-5)

    rng = np.random.default_rng(11)
    x = Tensor.from_numpy(rng.standard_normal((5, 8)), ("t", "d"))
    g = Tensor.from_numpy(rng.standard_normal(8), ("d",))
    b = Tensor.from_numpy(rng.standard_normal(8), ("d",))
    # the incumbent: lift + autodiff over the Program world
    ls = lift_step(ln, x=x.layout, g=g.layout, b=b.layout)
    prog = Program((*ls.program.instrs, Instr("zloss", "reduce", (ls.outputs[0],), {"f": "sum", "dims": ("t", "d")})))
    joint, grads = grad(prog, "zloss", {"x": x, "g": g, "b": b})
    env = run(joint, {"x": x, "g": g, "b": b})
    # the dialect: the same function, the general VJP, a ones seed
    region = lower_body(
        layernorm,
        (tensor_type(x), tensor_type(g), tensor_type(b)),
        kind="step",
        host={"feat": "d", "eps": 1e-5},
    )
    vjp = derive_vjp(region)
    ones = Tensor.from_numpy(np.ones((5, 8)), ("t", "d"))
    dx, dg, db = run_region(vjp, [x, g, b, ones])
    for got, name in ((dx, "x"), (dg, "g"), (db, "b")):
        want = env[grads[name]]
        order = tuple(d.name for d in want.layout.dims)
        np.testing.assert_allclose(got.to_numpy(order=order), want.to_numpy(order=order), rtol=1e-12)


def test_vjp_pass1_refuses_unsupported_reducers():
    """The engine grows per-op, by declaration: a max-reduce refuses with
    the arriving-slice reason, never a silent wrong gradient."""
    from pdum.tl.dialect import check_vjp_supported

    def peak(x):
        return reduce(red.max, x, "d")

    x = Tensor.from_numpy(np.zeros((3, 4)), ("t", "d"))
    region = lower_body(peak, (tensor_type(x),), kind="step")
    with pytest.raises(TypeError, match=r"reducer 'max' adjoint arrives.*sum/mean today"):
        check_vjp_supported(region)


# --- C4.3c: the migration view (region -> Program) ---------------------------


def test_export_round_trips_the_theorem_step():
    """A dialect region rendered as an incumbent Program runs bit-identical
    to the incumbent lift_step of the same function."""
    from pdum.tl.dialect import export_program
    from pdum.tl.ir import run
    from pdum.tl.lifting import lift_step

    s0, mask, _ = _theorem_setup()
    m = Tensor.from_numpy(np.zeros(4), ("x",))
    region = lower_body(_theorem_step, (tensor_type(s0), tensor_type(m)), kind="step")
    prog, outs = export_program(region, ("s", "m"))
    got = run(prog, {"s": s0, "m": m})[outs[0]]
    ls = lift_step(_theorem_step, s=s0.layout, m=m.layout)
    want = run(ls.program, {"s": s0, "m": m})[ls.outputs[0]]
    np.testing.assert_array_equal(got.to_numpy(order=("x",)), want.to_numpy(order=("x",)))


def test_export_grad_max_reduce_and_layout_ops_bit_identical():
    """THE C4.3c crown: the INCUMBENT autodiff — first-occurrence masks
    (the partition law, ties included), layout-op adjoints — runs over an
    EXPORTED region and matches the incumbent-lifted path bit-for-bit.
    Adjoint knowledge stays single-copy; regions get it through the view."""
    from pdum.tl.autodiff import grad
    from pdum.tl.dialect import export_program
    from pdum.tl.ir import Instr, Program, run
    from pdum.tl.lifting import lift_step

    def spiky(x):
        m = reduce(red.max, x, "d")  # ties below: the partition law must hold
        e = x - repeat_like(m, x)
        s = e.shift(d=1).slice(d=(1, 8)).pad(d=(0, 8), fill=0.0)
        return s * 2.0

    arr = np.arange(40.0).reshape(5, 8)
    arr[2, 3] = arr[2, 7] = 99.0  # a TIE along the reduced dim
    x = Tensor.from_numpy(arr, ("t", "d"))

    def with_loss(prog, out):
        return Program((*prog.instrs, Instr("zloss", "reduce", (out,), {"f": "sum", "dims": ("t", "d")})))

    region = lower_body(spiky, (tensor_type(x),), kind="step")
    prog_a, outs = export_program(region, ("x",))
    ja, ga = grad(with_loss(prog_a, outs[0]), "zloss", {"x": x})
    got = run(ja, {"x": x})[ga["x"]]
    ls = lift_step(spiky, x=x.layout)
    jb, gb = grad(with_loss(ls.program, ls.outputs[0]), "zloss", {"x": x})
    want = run(jb, {"x": x})[gb["x"]]
    np.testing.assert_array_equal(got.to_numpy(order=("t", "d")), want.to_numpy(order=("t", "d")))
    assert not np.array_equal(got.to_numpy(), np.zeros((5, 8)))


def test_export_multi_output_step():
    """A two-output step (the FDTD shape) exports with both outputs and
    runs bit-identical to the incumbent lift."""
    from pdum.tl.dialect import export_program
    from pdum.tl.ir import run
    from pdum.tl.lifting import lift_step

    def leap(E, H):
        H1 = H + E.shift(x=-1).slice(x=(0, 9)).pad(x=(0, 10), fill=0.0) * 0.5
        E1 = E + H1 * 0.25
        return E1, H1

    rng = np.random.default_rng(9)
    E = Tensor.from_numpy(rng.standard_normal(10), ("x",))
    H = Tensor.from_numpy(rng.standard_normal(10), ("x",))
    region = lower_body(leap, (tensor_type(E), tensor_type(H)), kind="step")
    prog, outs = export_program(region, ("E", "H"))
    env = run(prog, {"E": E, "H": H})
    ls = lift_step(leap, E=E.layout, H=H.layout)
    env_b = run(ls.program, {"E": E, "H": H})
    for got_v, want_v in zip(outs, ls.outputs):
        np.testing.assert_array_equal(env[got_v].to_numpy(order=("x",)), env_b[want_v].to_numpy(order=("x",)))
