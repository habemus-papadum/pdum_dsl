"""The random-field primitives (200 §1.8): counter-based closed-form fields,
exact under views, regenerating bit-identically — the recompute theorem's
foundation — with the dropout idiom and gradient-free AD behavior."""

import numpy as np
import pytest
from pdum.tl import Tensor, fold_in, normal, uniform
from pdum.tl.autodiff import grad
from pdum.tl.ir import Instr, Program, run
from pdum.tl.random import RandomBuffer, _philox2x32


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def I(var, op, operands=(), **params):  # noqa: E743
    return Instr(var, op, tuple(operands), params)


def test_the_field_regenerates_bit_identically():
    lay = T(np.zeros((3, 4)), ("i", "j")).layout
    a = uniform(7, lay).to_numpy()
    b = uniform(7, lay).to_numpy()
    np.testing.assert_array_equal(a, b)  # the recompute theorem's foundation
    c = uniform(8, lay).to_numpy()
    assert not np.array_equal(a, c)  # a different stream is a different field


def test_exact_under_views_zero_memory():
    base = T(np.zeros(10), ("x",))
    u = uniform(3, base.layout)
    assert isinstance(u.buffer, RandomBuffer) and u.buffer.data is None  # no memory
    full = u.to_numpy()
    sl = u.slice(x=(2, 7)).to_numpy()
    np.testing.assert_array_equal(sl, full[2:7])  # a view reads the SAME lattice
    sh = u.shift(x=1).slice(x=(1, 10)).to_numpy()
    np.testing.assert_array_equal(sh, full[:9])


def test_uniform_bits_are_exact_rationals_in_range():
    u = uniform(11, T(np.zeros(4096), ("x",)).layout)
    vals = u.to_numpy()
    assert ((vals >= 0.0) & (vals < 1.0)).all()
    assert abs(vals.mean() - 0.5) < 0.02  # Philox-quality bits, smoke-checked
    assert u.carrier == "rat"  # u32 / 2^32 IS a rational — carrier-consistent


def test_normal_field_moments():
    n = normal(13, T(np.zeros(8192), ("x",)).layout)
    vals = n.to_numpy()
    assert abs(vals.mean()) < 0.05 and abs(vals.std() - 1.0) < 0.05
    assert n.carrier == "real"


def test_fold_in_streams_are_stable_and_distinct():
    root = 42
    assert fold_in(root, "h.0.attn_drop") == fold_in(root, "h.0.attn_drop")  # process-stable
    assert fold_in(root, "h.0.attn_drop") != fold_in(root, "h.1.attn_drop")
    assert fold_in(root, 0) != fold_in(root, 1)  # step streams
    with pytest.raises(TypeError, match="path string or a step index"):
        fold_in(root, 1.5)


def test_philox_reference_vector_is_frozen():
    """The generator is FROZEN contract: device lowerings must reproduce
    these exact words. A drifted constant is silent nonreproducibility."""
    assert _philox2x32(0, 0) == _philox2x32(0, 0)
    pinned = [_philox2x32(0, 0), _philox2x32(0, 1), _philox2x32(1, 0), _philox2x32(0xDEADBEEF, 12345)]
    assert pinned == [4280135257, 3705464917, 2473546483, 244130200]


def test_dropout_idiom_in_ir_with_gradient():
    """where(u < p, 0, x/(1-p)) — the mask acts as a constant field; the
    gradient is exactly the kept-mask scaling, via existing rules only."""
    p, key = 0.25, fold_in(9, "drop_site")
    prog = Program(
        (
            I("x", "input"),
            I("u", "random", ("x",), dist="uniform", key=key),
            I("pc", "const", (), value=p, dims=(("i", (0, 64)),)),
            I("kc", "const", (), value=1.0 - p, dims=(("i", (0, 64)),)),
            I("m", "pointwise", ("u", "pc"), f="lt"),
            I("z", "const", (), value=0.0, dims=(("i", (0, 64)),)),
            I("xs", "pointwise", ("x", "kc"), f="div"),
            I("y", "pointwise", ("m", "z", "xs"), f="where"),
            I("s", "reduce", ("y",), f="sum", dims=("i",)),
        )
    )
    x = T(np.random.default_rng(0).standard_normal(64), ("i",))
    env = run(prog, {"x": x})
    mask = env["m"].to_numpy()
    np.testing.assert_allclose(env["y"].to_numpy(), np.where(mask, 0.0, x.to_numpy() / (1 - p)))
    joint, grads = grad(prog, "s", {"x": x})
    genv = run(joint, {"x": x})
    np.testing.assert_allclose(genv[grads["x"]].to_numpy(), np.where(mask, 0.0, 1.0 / (1 - p)))
    assert grads["u"] is None  # the field is gradient-free: a constant mask


def test_random_regenerates_inside_reruns_identically():
    key = fold_in(1, "site")
    prog = Program((I("x", "input"), I("u", "random", ("x",), dist="uniform", key=key)))
    x = T(np.zeros(16), ("i",))
    a = run(prog, {"x": x})["u"].to_numpy()
    b = run(prog, {"x": x})["u"].to_numpy()
    np.testing.assert_array_equal(a, b)


def test_the_recompute_theorem_revolve_equals_store_all_with_dropout_on():
    """THE FULL PIN (200 §1.8.4, P7): a revolve-checkpointed fold and the
    store-all schedule produce BIT-IDENTICAL gradients with dropout on.
    The mask is a zero-memory closed-form field over (tm, x) consumed as a
    fold element — recompute re-reads the same coordinates and regenerates
    the same bits BY CONSTRUCTION; no mask is ever stored anywhere."""
    from pdum.dsl.types import Literal
    from pdum.tl.autodiff import grad
    from pdum.tl.compute import const_like, pointwise
    from pdum.tl.lifting import lift_step
    from pdum.tl.markers import lt, tanh, where
    from pdum.tl.random import RandomBuffer

    N, TM, p_drop = 4, 6, 0.4
    key = fold_in(5, "train.drop")
    rng = np.random.default_rng(2)
    s0 = T(rng.standard_normal(N), ("x",))
    mask_lattice = T(np.zeros((TM, N)), ("tm", "x"))
    mask = uniform(key, mask_lattice.layout)
    assert isinstance(mask.buffer, RandomBuffer) and mask.buffer.data is None  # zero bytes

    def step(s, m, p: Literal[float]):
        kept = pointwise(where, pointwise(lt, m, const_like(m, p)), const_like(s, 0.0), s)
        return s * 0.8 + pointwise(tanh, kept) * 0.3

    ls = lift_step(step, s=s0.layout, m=T(np.zeros(N), ("x",)).layout, p=p_drop)
    fold_params = {
        "step": ls.program,
        "dim": "tm",
        "state": ("s",),
        "element": ("m",),
        "carry": {"s": ls.outputs[0]},
        "out": ("final", ls.outputs[0]),
    }
    prog = Program(
        (
            I("s0", "input"),
            I("mask", "input"),
            I("sf", "fold", ("s0", "mask"), **fold_params),
            I("zloss", "reduce", ("sf",), f="sum", dims=("x",)),
        )
    )
    inputs = {"s0": s0, "mask": mask}
    j_all, g_all = grad(prog, "zloss", dict(inputs), fold_segments=1)
    j_rev, g_rev = grad(prog, "zloss", dict(inputs), fold_slots=2)
    e_all = run(j_all, inputs)
    e_rev = run(j_rev, inputs)
    got_all = e_all[g_all["s0"]].to_numpy(order=("x",))
    got_rev = e_rev[g_rev["s0"]].to_numpy(order=("x",))
    np.testing.assert_array_equal(got_rev, got_all)  # BIT-identical, dropout on
    assert not np.array_equal(got_all, np.zeros(N))
