"""Reverse-mode AD: every adjoint rule validated by finite differences.

Region-first since the excavation (LEVELS): bodies author through
lift_step, gradients through grad's RegionGrad, execution through
run_named, FD through numeric_grad — the same pins, the surviving
representation."""

import numpy as np
import pytest
from pdum.dsl.ir import Builder, Region
from pdum.dsl.ops import CORE_OPS
from pdum.tl import Tensor, red, reduce, scan
from pdum.tl.autodiff import grad, numeric_grad
from pdum.tl.compute import pointwise
from pdum.tl.dialect import TL_OPS, run_named
from pdum.tl.lifting import lift_step
from pdum.tl.markers import exp, log, maximum, neg, where


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def lift(fn, inputs):
    return lift_step(fn, **{k: v.layout for k, v in inputs.items()})


def check(fn, inputs, wrt, rtol=1e-4, atol=1e-7):
    ls = lift(fn, inputs)
    target = ls.outputs[0]
    rg = grad(ls.region, target, dict(inputs), names=ls.names)
    assert rg.grads[wrt] is not None
    vals = run_named(rg.region, inputs, rg.names)
    got = vals[rg.grads[wrt]].to_numpy(order=inputs[wrt].names)
    want = numeric_grad(ls.region, target, wrt, inputs, ls.names)
    np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)
    return rg, vals


RNG = np.random.default_rng(7)


# ----------------------------------------------------------------------
# pointwise markers
# ----------------------------------------------------------------------


def test_grad_mul_div_exp_log():
    def f(x, w):
        p = x * w
        e = pointwise(exp, p)
        ll = pointwise(log, e)
        d = ll / w
        return reduce(red.sum, d, "i")

    inputs = {"x": T(RNG.uniform(1, 2, 4), ("i",)), "w": T(RNG.uniform(1, 2, 4), ("i",))}
    check(f, inputs, "x")
    check(f, inputs, "w")


def test_grad_maximum_and_where():
    def f(a, bb):
        m = pointwise(maximum, a, bb)
        c = a < bb
        w = pointwise(where, c, m, a)
        return reduce(red.sum, w, "i")

    inputs = {"a": T([1.0, 5.0, 2.0, 7.0], ("i",)), "bb": T([3.0, 1.0, 6.0, 2.0], ("i",))}
    check(f, inputs, "a")
    check(f, inputs, "bb")


def test_fanout_accumulates():
    def f(x):
        sq = x * x
        return reduce(red.sum, sq, "i")

    inputs = {"x": T([1.0, -2.0, 3.0], ("i",))}
    rg, vals = check(f, inputs, "x")
    np.testing.assert_allclose(vals[rg.grads["x"]].to_numpy(), 2 * inputs["x"].to_numpy())


# ----------------------------------------------------------------------
# layout adjoints
# ----------------------------------------------------------------------


def test_grad_slice_and_pad():
    def f(x):
        s = x.slice(i=(1, 4))
        return reduce(red.sum, s, "i")

    ls = lift(f, {"x": T(RNG.standard_normal(5), ("i",))})
    rg = grad(ls.region, ls.outputs[0], names=ls.names)
    vals = run_named(rg.region, {"x": T([1.0, 2, 3, 4, 5], ("i",))}, rg.names)
    np.testing.assert_allclose(vals[rg.grads["x"]].to_numpy(), [0, 1, 1, 1, 0])

    def g(x, w):
        p = x.pad(0.0, i=(-1, 4))
        m = p * w
        return reduce(red.sum, m, "i")

    inputs = {"x": T(RNG.standard_normal(3), ("i",)), "w": T(RNG.standard_normal(5), ("i",)).shift(i=-1)}
    check(g, inputs, "x")


def test_grad_relabelings():
    def f(x, w):
        r = x.flip("i").shift(i=3).rename(i="j")
        ws = w.rename(i="j").shift(j=3)
        m = r * ws
        return reduce(red.sum, m, "j")

    inputs = {"x": T(RNG.standard_normal(4), ("i",)), "w": T(RNG.standard_normal(4), ("i",))}
    check(f, inputs, "x")


def test_grad_repeat_and_select():
    def f(x, w):
        r = x.repeat("n", (0, 3))
        m = r * w
        return reduce(red.sum, m, ("i", "n"))

    inputs = {
        "x": T(RNG.standard_normal(4), ("i",)),
        "w": T(RNG.standard_normal((4, 3)), ("i", "n")),
    }
    check(f, inputs, "x")

    def g(x):
        s = x.select(i=2)
        return reduce(red.sum, s, "j")

    check(g, {"x": T(RNG.standard_normal((4, 3)), ("i", "j"))}, "x")


def test_grad_split_merge():
    def f(x, w):
        b = x.split("i", io=2, ii=3)
        m = b * w
        g = m.merge(("io", "ii"), "i")
        return reduce(red.sum, g, "i")

    inputs = {
        "x": T(RNG.standard_normal(6), ("i",)),
        "w": T(RNG.standard_normal((2, 3)), ("io", "ii")),
    }
    check(f, inputs, "x")
    check(f, inputs, "w")


def test_grad_window_conv():
    def f(x, w):
        xw = x.window("i", "k", 3)
        wr = w.repeat("i", (0, 4))
        m = xw * wr
        return reduce(red.sum, m, ("i", "k"))

    inputs = {
        "x": T(RNG.standard_normal(6), ("i",)),
        "w": T(RNG.standard_normal(3), ("k",)),
    }
    check(f, inputs, "x")
    check(f, inputs, "w")


def test_grad_stencil_discards_fill_cotangent():
    def f(x, w):
        xs = x.stencil("i", (-1, 1), None, 0.0)
        wr = w.repeat("i", (0, 5))
        m = xs * wr
        return reduce(red.sum, m, ("i", "i_k"))

    inputs = {
        "x": T(RNG.standard_normal(5), ("i",)),
        "w": T(RNG.standard_normal(3), ("i_k",)).shift(i_k=-1),
    }
    check(f, inputs, "x")


def test_grad_dilated_stencil():
    def f(x):
        xs = x.stencil("i", (-1, 1), None, 0.0, 2)
        return reduce(red.sum, xs, ("i", "i_k"))

    check(f, {"x": T(RNG.standard_normal(6), ("i",))}, "x")


def test_grad_decimate():
    def f(x, w):
        d = x.decimate("i", 2, 1)
        m = d * w
        return reduce(red.sum, m, "i")

    inputs = {
        "x": T(RNG.standard_normal(6), ("i",)),
        "w": T(RNG.standard_normal(3), ("i",)),
    }
    rg, vals = check(f, inputs, "x")
    g = vals[rg.grads["x"]].to_numpy()
    np.testing.assert_allclose(g[::2], 0.0)  # untouched phase gets zero


def test_grad_diagonal():
    def f(x, w):
        d = x.diagonal(("i", "j"), "z")
        m = d * w
        return reduce(red.sum, m, "z")

    inputs = {
        "x": T(RNG.standard_normal((3, 3)), ("i", "j")),
        "w": T(RNG.standard_normal(3), ("z",)),
    }
    check(f, inputs, "x")


# ----------------------------------------------------------------------
# reduce / scan
# ----------------------------------------------------------------------


def test_grad_reduce_mean_and_max():
    def f(x):
        mu = reduce(red.mean, x, "j")
        mx = reduce(red.max, x, "j")
        s = mu * mx
        return reduce(red.sum, s, "i")

    # distinct values: keep max-ties away from finite differences
    x = np.array([[1.0, 5.0, 2.0], [9.0, 3.0, 4.0]])
    check(f, {"x": T(x, ("i", "j"))}, "x")


def test_grad_scan_sum():
    def f(x, w):
        cs = scan(red.sum, x, "i")
        m = cs * w
        return reduce(red.sum, m, "i")

    inputs = {
        "x": T(RNG.standard_normal(5), ("i",)),
        "w": T(RNG.standard_normal(5), ("i",)),
    }
    check(f, inputs, "x")


# ----------------------------------------------------------------------
# contracts and end-to-end
# ----------------------------------------------------------------------


def test_seed_contract():
    def f(x):
        d = x * x
        return d

    ls = lift(f, {"x": T([1.0, 2.0], ("i",))})
    with pytest.raises(ValueError):
        grad(ls.region, ls.outputs[0], names=ls.names)  # non-scalar, no seed
    rg = grad(ls.region, ls.outputs[0], seed="dY", names=ls.names)
    vals = run_named(rg.region, {"x": T([3.0, 4.0], ("i",)), "dY": T([1.0, 1.0], ("i",))}, rg.names)
    np.testing.assert_allclose(vals[rg.grads["x"]].to_numpy(), [6.0, 8.0])


def test_unreachable_vars_get_none():
    def f(x, u):
        y = reduce(red.sum, x, "i")
        return y

    ls = lift(f, {"x": T([1.0], ("i",)), "u": T([1.0], ("i",))})
    rg = grad(ls.region, ls.outputs[0], names=ls.names)
    assert rg.grads["u"] is None and rg.grads["x"] is not None


def test_grad_matmul_end_to_end():
    a = RNG.standard_normal((2, 3))
    bm = RNG.standard_normal((3, 4))

    def f(A, B):
        A3 = A.repeat("n", (0, 4))
        B3 = B.repeat("m", (0, 2))
        P = A3 * B3
        C = reduce(red.sum, P, "k")
        return reduce(red.sum, C, ("m", "n"))

    inputs = {"A": T(a, ("m", "k")), "B": T(bm, ("k", "n"))}
    rg, vals = check(f, inputs, "A")
    check(f, inputs, "B")
    # analytic check: dL/dA = ones @ B^T
    np.testing.assert_allclose(
        vals[rg.grads["A"]].to_numpy(order=("m", "k")),
        np.ones((2, 4)) @ bm.T,
        rtol=1e-6,
    )


def test_grad_softmax_cross_entropy():
    s = RNG.standard_normal((2, 4))
    onehot = np.zeros((2, 4))
    onehot[0, 1] = onehot[1, 3] = 1.0

    def f(S, t):
        mx = reduce(red.max, S, "v")
        sh = S - mx.repeat("v", (0, 4))
        e = pointwise(exp, sh)
        z = reduce(red.sum, e, "v")
        lz = pointwise(log, z.repeat("v", (0, 4)))
        lp = sh - lz
        nll = lp * t
        sum1 = reduce(red.sum, nll, "v")
        negl = pointwise(neg, sum1)
        return reduce(red.sum, negl, "i")

    inputs = {"S": T(s, ("i", "v")), "t": T(onehot, ("i", "v"))}
    rg, vals = check(f, inputs, "S", rtol=1e-3, atol=1e-5)
    # analytic: dL/dS = softmax(S) - onehot
    sm = np.exp(s - s.max(1, keepdims=True))
    sm /= sm.sum(1, keepdims=True)
    np.testing.assert_allclose(vals[rg.grads["S"]].to_numpy(order=("i", "v")), sm - onehot, rtol=1e-5, atol=1e-8)


# ----------------------------------------------------------------------
# review regressions (adversarial + cleanliness findings)
# ----------------------------------------------------------------------


def test_grad_stencil_tap_entirely_outside_source():
    # far-top dilated taps used to emit an un-runnable pad; now: zero contribution
    def f(x):
        xs = x.stencil("i", (0, 3), None, 0.0, 2)
        return reduce(red.sum, xs, ("i", "i_k"))

    check(f, {"x": T(RNG.standard_normal(4), ("i",))}, "x")


def test_grad_diagonal_disjoint_parts_is_zero():
    def f(x):
        sh = x.shift(j=5)
        d = sh.diagonal(("i", "j"), "z")
        return reduce(red.sum, d, "z")

    x = T(RNG.standard_normal((3, 3)), ("i", "j"))
    ls = lift(f, {"x": x})
    rg = grad(ls.region, ls.outputs[0], names=ls.names)
    vals = run_named(rg.region, {"x": x}, rg.names)
    np.testing.assert_allclose(vals[rg.grads["x"]].to_numpy(), np.zeros((3, 3)))


def test_reduce_max_tie_partitions_first_wins():
    # THE AT-KINK LAW (200 §S.2, re-pinned at P4): at a tie exactly ONE
    # element receives the cotangent — the first along the reduced dim.
    def f(x):
        m = reduce(red.max, x, "i")
        return m

    x = T([3.0, 3.0, 1.0], ("i",))
    ls = lift(f, {"x": x})
    rg = grad(ls.region, ls.outputs[0], names=ls.names)
    vals = run_named(rg.region, {"x": x}, rg.names)
    np.testing.assert_allclose(vals[rg.grads["x"]].to_numpy(), [1.0, 0.0, 0.0])


def test_unknown_differentiable_marker_raises_not_silent_zero():
    from pdum.tl.compute import Marker
    from pdum.tl.markers import PW

    tmp = Marker("sqrt_tmp", np.sqrt)
    PW["sqrt_tmp"] = tmp
    try:

        def f(x):
            s = pointwise(tmp, x)
            return reduce(red.sum, s, "i")

        ls = lift(f, {"x": T([1.0, 4.0], ("i",))})
        with pytest.raises(NotImplementedError):
            grad(ls.region, ls.outputs[0], names=ls.names)
    finally:
        del PW["sqrt_tmp"]


def test_grad_contract_errors_are_clear():
    def f(x):
        y = reduce(red.sum, x, "i")
        return y

    ls = lift(f, {"x": T([1.0], ("i",))})
    with pytest.raises(KeyError):
        grad(ls.region, "nope", names=ls.names)
    with pytest.raises(ValueError, match="collides"):
        grad(ls.region, ls.outputs[0], seed="x", names=ls.names)


def test_second_order_differentiation():
    def f(x):
        x2 = x * x
        x3 = x2 * x
        return reduce(red.sum, x3, "i")

    x = T([1.0, -2.0, 3.0], ("i",))
    ls = lift(f, {"x": x})
    rg1 = grad(ls.region, ls.outputs[0], names=ls.names)
    # differentiate the gradient: re-yield sum(dy/dx) and grad again — the
    # joint is ordinary IR, differentiable like any region
    b = Builder({**CORE_OPS, **TL_OPS})
    gx = rg1.nodes[rg1.grads["x"]]
    gy = b.emit("tl.reduce", gx, f="sum", dims=("i",))
    r2 = Region(params=rg1.region.params, body=(b.emit("core.yield", gy),))
    names2 = dict(rg1.names)
    names2[id(gy)] = "gy"
    rg2 = grad(r2, "gy", names=names2)
    v1 = run_named(rg1.region, {"x": x}, rg1.names)
    v2 = run_named(rg2.region, {"x": x}, rg2.names)
    np.testing.assert_allclose(v1[rg1.grads["x"]].to_numpy(), 3 * x.to_numpy() ** 2)
    np.testing.assert_allclose(v2[rg2.grads["x"]].to_numpy(), 6 * x.to_numpy())


# ----------------------------------------------------------------------
# units in gradients (CONCERNS #19 resolution)
# ----------------------------------------------------------------------


def test_gradients_carry_primal_charts():
    from pdum.tl import q

    x = T(RNG.standard_normal(5), ("i",)).with_charts(i=("0 nm", "25 nm"))

    def f(x):
        xs = x.stencil("i", (-1, 1), None, 0.0)
        return reduce(red.sum, xs, ("i", "i_k"))

    ls = lift(f, {"x": x})
    rg = grad(ls.region, ls.outputs[0], names=ls.names)
    vals = run_named(rg.region, {"x": x}, rg.names)
    g = vals[rg.grads["x"]]
    assert g.layout.dim("i").chart == x.layout.dim("i").chart
    # physical indexing of the gradient works: entry FOR the sample at 50 nm
    assert g.item(i=q("50 nm")) == g.item(i=2)
    want = numeric_grad(ls.region, ls.outputs[0], "x", {"x": x}, ls.names)  # chart-preserving FD
    np.testing.assert_allclose(g.to_numpy(), want, rtol=1e-4, atol=1e-7)


def test_select_compensation_paths_accumulate_when_charted():
    # the crux: one contribution flows through split+select (whose forward
    # promotes/compensates sibling charts), another directly — both must be
    # restamped onto x's chart and add cleanly
    x = T(RNG.standard_normal(6), ("i",)).with_charts(i=("0 um", "0.25 um"))

    def f(x):
        b = x.split("i", ib=3, ii=2)
        s = b.select(ii=1)
        p = reduce(red.sum, s, "ib")
        d = reduce(red.sum, x, "i")
        y = p + d
        return y

    ls = lift(f, {"x": x})
    rg = grad(ls.region, ls.outputs[0], names=ls.names)
    vals = run_named(rg.region, {"x": x}, rg.names)
    g = vals[rg.grads["x"]]
    assert g.layout.dim("i").chart == x.layout.dim("i").chart
    np.testing.assert_allclose(g.to_numpy(), [1, 2, 1, 2, 1, 2])  # 1 + phase-hit
    want = numeric_grad(ls.region, ls.outputs[0], "x", {"x": x}, ls.names)
    np.testing.assert_allclose(g.to_numpy(), want, rtol=1e-4, atol=1e-7)


def test_gradients_carry_primal_labels():
    x = T(RNG.standard_normal((3, 4)), ("c", "i")).with_labels(c=("R", "G", "B"))

    def f(x):
        sq = x * x
        return reduce(red.sum, sq, ("c", "i"))

    ls = lift(f, {"x": x})
    rg = grad(ls.region, ls.outputs[0], names=ls.names)
    vals = run_named(rg.region, {"x": x}, rg.names)
    g = vals[rg.grads["x"]]
    assert g.layout.dim("c").labels == ("R", "G", "B")
    np.testing.assert_allclose(g.item(c="G", i=1), 2 * x.item(c="G", i=1), rtol=1e-12)


def test_gradient_value_units():
    from pdum.tl import u

    volt = u.define("V_ad", dim="voltage_ad")
    x = T(RNG.standard_normal(4), ("i",)).with_value_units(volt)
    w = T(RNG.standard_normal(4), ("i",))  # no declared unit

    def f(x, w):
        p = x * w
        return reduce(red.sum, p, "i")

    inputs = {"x": x, "w": w}
    ls = lift(f, inputs)
    rg = grad(ls.region, ls.outputs[0], inputs, target_unit=volt * volt, names=ls.names)
    vals = run_named(rg.region, inputs, rg.names)
    assert vals[rg.grads["x"]].value_units == volt  # V²/V
    assert vals[rg.grads["w"]].value_units == volt * volt  # V²/1
    # without target_unit: no annotation
    rg2 = grad(ls.region, ls.outputs[0], inputs, names=ls.names)
    assert run_named(rg2.region, inputs, rg2.names)[rg2.grads["x"]].value_units is None


def test_charted_window_gradient_charts_flow_naturally():
    x = T(RNG.standard_normal(6), ("t",)).with_charts(t=("0 ms", "1 ms"))

    def f(x):
        w = x.window("t", "k", 3)
        return reduce(red.sum, w, ("t", "k"))

    ls = lift(f, {"x": x})
    rg = grad(ls.region, ls.outputs[0], names=ls.names)
    vals = run_named(rg.region, {"x": x}, rg.names)
    g = vals[rg.grads["x"]]
    assert g.layout.dim("t").chart == x.layout.dim("t").chart
    np.testing.assert_allclose(g.to_numpy(), [1, 2, 3, 3, 2, 1])  # overlap counts
