"""The marker DSL: lowered composites, derived partials, structured reducers."""

import numpy as np
import pytest
from pdum.tl import Tensor, defmarker, defreducer, pointwise, red, reduce, scan
from pdum.tl.autodiff import grad, numeric_grad
from pdum.tl.dialect import run_named
from pdum.tl.lifting import lift_step
from pdum.tl.mdsl import diff, exp, gt, log, tanh, where
from pdum.tl.nodes import Arg, Const, Prim
from pdum.tl.producer import lower, scalars
from pdum.tl.registry import MARKERS


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


RNG = np.random.default_rng(11)

sigmoid = defmarker("sigmoid_t", 1, lambda x: 1 / (1 + exp(-x)))
softplus = defmarker("softplus_t", 1, lambda x: log(1 + exp(x)))
GELU_C = 0.7978845608028654  # sqrt(2/pi)
gelu = defmarker("gelu_t", 1, lambda x: 0.5 * x * (1 + tanh(GELU_C * (x + 0.044715 * x * x * x))))
relu = defmarker("relu_t", 1, lambda x: where(gt(x, 0), x, 0 * x))

_COMPOSITES = {"sigmoid_t": sigmoid, "softplus_t": softplus, "gelu_t": gelu, "relu_t": relu}


# ----------------------------------------------------------------------
# tracing and evaluation
# ----------------------------------------------------------------------


def test_the_producer_builds_trees():
    (body,) = lower(lambda x: 2 * x + 1, scalars(1))
    assert body == Prim("add", (Prim("mul", (Const(2), Arg(0))), Const(1)))


def test_python_control_flow_is_refused():
    # straight-line detection at LOWERING (P4) — the producer refuses by
    # inspection, where the tracer used to trip at trace time
    with pytest.raises(ValueError, match=r"straight-line.*where\(cond, a, b\)"):
        defmarker("bad_t", 1, lambda x: x if x else -x)
    with pytest.raises(ValueError, match=r"straight-line"):
        defmarker("bad_loop_t", 1, _looped)


def _looped(x):
    for _ in range(3):
        x = x * x
    return x


def test_composite_eval_matches_numpy():
    x = RNG.standard_normal(7)
    t = T(x, ("i",))
    np.testing.assert_allclose(pointwise(sigmoid, t).to_numpy(), 1 / (1 + np.exp(-x)))
    np.testing.assert_allclose(pointwise(softplus, t).to_numpy(), np.log1p(np.exp(x)), rtol=1e-12)
    np.testing.assert_allclose(
        pointwise(gelu, t).to_numpy(),
        0.5 * x * (1 + np.tanh(GELU_C * (x + 0.044715 * x**3))),
    )
    np.testing.assert_allclose(pointwise(relu, t).to_numpy(), np.maximum(x, 0))


def test_composites_work_in_ir_by_name():
    def step(x):
        s = pointwise(sigmoid, x)
        y = reduce(red.sum, s, "i")
        return s, y

    x = T([0.0, 1.0, -1.0], ("i",))
    ls = lift_step(step, x=x.layout)
    env = run_named(ls.region, {"x": x}, ls.names)
    np.testing.assert_allclose(env["s"].to_numpy(), 1 / (1 + np.exp(-x.to_numpy())))


# ----------------------------------------------------------------------
# derived partials — no hand-written gradient rules anywhere below
# ----------------------------------------------------------------------


def test_partial_is_derived_and_registered():
    ds = sigmoid.partial(0)
    assert ds.name == "sigmoid_t.d0" and ds.name in MARKERS
    x = RNG.standard_normal(5)
    got = pointwise(ds, T(x, ("i",))).to_numpy()
    s = 1 / (1 + np.exp(-x))
    np.testing.assert_allclose(got, s * (1 - s), rtol=1e-10)


@pytest.mark.parametrize("mname", ["sigmoid_t", "softplus_t", "gelu_t", "relu_t"])
def test_composites_differentiate_automatically(mname):
    m = _COMPOSITES[mname]

    def step(x, w):
        p = x * w
        a = pointwise(m, p)
        y = reduce(red.sum, a, "i")
        return y

    inputs = {
        "x": T(RNG.uniform(0.2, 1.5, 5), ("i",)),
        "w": T(RNG.uniform(0.2, 1.5, 5), ("i",)),
    }
    ls = lift_step(step, x=inputs["x"].layout, w=inputs["w"].layout)
    rg = grad(ls.region, "y", dict(inputs), names=ls.names)
    env = run_named(rg.region, inputs, rg.names)
    for wrt in ("x", "w"):
        got = env[rg.grads[wrt]].to_numpy(order=inputs[wrt].names)
        want = numeric_grad(ls.region, "y", wrt, inputs, ls.names)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-7)


def test_tanh_primitive_gradient():
    def step(x):
        t = pointwise(tanh, x)
        y = reduce(red.sum, t, "i")
        return y

    inputs = {"x": T(RNG.standard_normal(5), ("i",))}
    ls = lift_step(step, x=inputs["x"].layout)
    rg = grad(ls.region, "y", dict(inputs), names=ls.names)
    env = run_named(rg.region, inputs, rg.names)
    np.testing.assert_allclose(
        env[rg.grads["x"]].to_numpy(),
        numeric_grad(ls.region, "y", "x", inputs, ls.names),
        rtol=1e-4,
        atol=1e-7,
    )


def test_gradient_free_composite_contributes_nothing():
    mask = defmarker("mask_t", 2, lambda x, y: gt(x, y))
    assert isinstance(diff(mask.body, 0), Const)


# ----------------------------------------------------------------------
# structured-state reducers: the SSM flagship
# ----------------------------------------------------------------------

linrec = defreducer(
    "linrec_t",
    state=2,
    element=2,
    lift=lambda a, b: (a, b),
    combine=lambda left, right: (left[0] * right[0], right[0] * left[1] + right[1]),
    init=(1.0, 0.0),
    project=lambda A, B: B,
)


def test_ssm_scan_matches_the_recurrence():
    n = 9
    a = RNG.uniform(0.5, 1.1, n)
    bb = RNG.standard_normal(n)
    h_ref = np.empty(n)
    h = 0.0
    for t in range(n):
        h = a[t] * h + bb[t]
        h_ref[t] = h
    out = scan(linrec, (T(a, ("t",)), T(bb, ("t",))), "t")
    np.testing.assert_allclose(out.to_numpy(), h_ref, rtol=1e-10)


def test_ssm_combine_is_associative():
    from pdum.tl.compute import _eval_tree

    rng = np.random.default_rng(3)
    for _ in range(20):
        s1, s2, s3 = (list(rng.standard_normal(2)) for _ in range(3))
        ab = [_eval_tree(n, s1 + s2) for n in linrec.combine]
        left = [_eval_tree(n, ab + s3) for n in linrec.combine]
        bc = [_eval_tree(n, s2 + s3) for n in linrec.combine]
        right = [_eval_tree(n, s1 + bc) for n in linrec.combine]
        np.testing.assert_allclose(left, right, rtol=1e-10)


def test_composite_reduce_gives_the_final_state():
    a = np.array([0.5, 2.0, 0.25])
    bb = np.array([1.0, -1.0, 3.0])
    h = 0.0
    for t in range(3):
        h = a[t] * h + bb[t]
    out = reduce(linrec, (T(a, ("t",)), T(bb, ("t",))), ("t",))
    assert out.layout.dims == ()
    np.testing.assert_allclose(float(out.item()), h)


def test_composite_reduce_empty_dim_is_the_identity_state():
    a = T(np.zeros(0), ("t",))
    out = reduce(linrec, (a, a), ("t",))
    np.testing.assert_allclose(float(out.item()), 0.0)  # project(init) = B = 0


def test_cumsum_as_composite_reducer():
    csum = defreducer(
        "csum_t",
        state=1,
        element=1,
        lift=lambda x: (x,),
        combine=lambda left, right: (left[0] + right[0],),
        init=(0.0,),
    )
    x = RNG.standard_normal(6)
    out = scan(csum, T(x, ("i",)), "i")
    np.testing.assert_allclose(out.to_numpy(), np.cumsum(x), rtol=1e-12)


def test_multi_operand_scan_in_ir_and_misalignment_refused():
    a = T(np.full(4, 0.5), ("t",))
    bb = T(np.arange(4.0), ("t",))

    def step(a, b):
        h = scan(linrec, (a, b), "t")
        return h

    ls = lift_step(step, a=a.layout, b=bb.layout)
    env = run_named(ls.region, {"a": a, "b": bb}, ls.names)
    assert env["h"].to_numpy().shape == (4,)
    with pytest.raises(ValueError, match="aligned"):
        scan(linrec, (a, bb.shift(t=1)), "t")


# ----------------------------------------------------------------------
# the SSM backward pass: BPTT emitted as IR
# ----------------------------------------------------------------------


def _ssm_step(loss="sum"):
    if loss == "sum":

        def step(a, b):
            h = scan(linrec, (a, b), "t")
            y = reduce(red.sum, h, "t")
            return y

    else:

        def step(a, b):
            h = scan(linrec, (a, b), "t")
            hh = h * h
            y = reduce(red.sum, hh, "t")
            return y

    return step


def test_ssm_gradient_analytic_structure():
    # for L = sum_t h_t: h̄_t = 1 + a_{t+1}·h̄_{t+1}, b̄ = h̄, ā_t = h_{t-1}·h̄_t
    a = np.array([0.9, 1.1, 0.7, 1.3, 0.8])
    bv = np.arange(1.0, 6.0)
    inputs = {"a": T(a, ("t",)), "b": T(bv, ("t",))}
    ls = lift_step(_ssm_step("sum"), a=inputs["a"].layout, b=inputs["b"].layout)
    rg = grad(ls.region, "y", dict(inputs), names=ls.names)
    env = run_named(rg.region, inputs, rg.names)
    h = np.empty(5)
    acc = 0.0
    for t in range(5):
        acc = a[t] * acc + bv[t]
        h[t] = acc
    hbar = np.empty(5)
    hbar[-1] = 1.0
    for t in range(3, -1, -1):
        hbar[t] = 1.0 + a[t + 1] * hbar[t + 1]
    np.testing.assert_allclose(env[rg.grads["b"]].to_numpy(), hbar, rtol=1e-10)
    np.testing.assert_allclose(env[rg.grads["a"]].to_numpy(), hbar * np.concatenate([[0.0], h[:-1]]), rtol=1e-10)


def test_ssm_scan_gradients_match_fd():
    # quadratic loss: the scan cotangent is NON-uniform, exercising the ȳ path
    n = 6
    inputs = {"a": T(RNG.uniform(0.4, 1.2, n), ("t",)), "b": T(RNG.standard_normal(n), ("t",))}
    ls = lift_step(_ssm_step("quadratic"), a=inputs["a"].layout, b=inputs["b"].layout)
    rg = grad(ls.region, "y", dict(inputs), names=ls.names)
    env = run_named(rg.region, inputs, rg.names)
    for v in ("a", "b"):
        fd = numeric_grad(ls.region, "y", v, inputs, ls.names)
        np.testing.assert_allclose(env[rg.grads[v]].to_numpy(), fd, rtol=1e-5, atol=1e-8)


def test_ssm_reduce_gradient_matches_fd():
    # reduce = select-last of the scan; its adjoint embeds at the last slot
    def step(a, b):
        hf = reduce(linrec, (a, b), "t")
        return hf

    n = 5
    inputs = {"a": T(RNG.uniform(0.4, 1.2, n), ("t",)), "b": T(RNG.standard_normal(n), ("t",))}
    ls = lift_step(step, a=inputs["a"].layout, b=inputs["b"].layout)
    rg = grad(ls.region, "hf", dict(inputs), names=ls.names)
    env = run_named(rg.region, inputs, rg.names)
    for v in ("a", "b"):
        fd = numeric_grad(ls.region, "hf", v, inputs, ls.names)
        np.testing.assert_allclose(env[rg.grads[v]].to_numpy(), fd, rtol=1e-5, atol=1e-8)


def test_batched_ssm_gradient_matches_fd():
    inputs = {
        "a": T(RNG.uniform(0.4, 1.2, (2, 4)), ("k", "t")),
        "b": T(RNG.standard_normal((2, 4)), ("k", "t")),
    }

    def step(a, b):
        h = scan(linrec, (a, b), "t")
        y = reduce(red.sum, h, ("k", "t"))
        return y

    ls = lift_step(step, a=inputs["a"].layout, b=inputs["b"].layout)
    rg = grad(ls.region, "y", dict(inputs), names=ls.names)
    env = run_named(rg.region, inputs, rg.names)
    for v in ("a", "b"):
        fd = numeric_grad(ls.region, "y", v, inputs, ls.names)
        np.testing.assert_allclose(env[rg.grads[v]].to_numpy(), fd, rtol=1e-5, atol=1e-8)


def test_cumsum_composite_adjoint_matches_plain_scan_sum():
    defreducer(
        "csum_g",
        state=1,
        element=1,
        lift=lambda x: (x,),
        combine=lambda left, right: (left[0] + right[0],),
        init=(0.0,),
    )
    x = T(RNG.standard_normal(5), ("i",))

    def build(fname):
        def step(x):
            s = scan(fname, x, "i")
            ss = s * s
            y = reduce(red.sum, ss, "i")
            return y

        ls = lift_step(step, x=x.layout)
        rg = grad(ls.region, "y", {"x": x}, names=ls.names)
        return run_named(rg.region, {"x": x}, rg.names)[rg.grads["x"]].to_numpy()

    np.testing.assert_allclose(build("csum_g"), build("sum"), rtol=1e-10)


def test_empty_composite_scan_gradient_is_zeros():
    e = T(np.zeros(0), ("t",))
    ls = lift_step(_ssm_step("sum"), a=e.layout, b=e.layout)
    rg = grad(ls.region, "y", {"a": e, "b": e}, names=ls.names)
    env = run_named(rg.region, {"a": e, "b": e}, rg.names)
    assert env[rg.grads["a"]].to_numpy().shape == (0,)
    assert env[rg.grads["b"]].to_numpy().shape == (0,)


def test_defmarker_refuses_primitive_names():
    with pytest.raises(ValueError):
        defmarker("mul", 2, lambda a, b: a * b)
