"""fold — the tensor-state scan: regions as step functions, derived BPTT."""

import numpy as np
import pytest

from pdum.dsl.ir import Builder, Region
from pdum.dsl.ops import CORE_OPS
from pdum.tl import Tensor, defreducer, ops_count, peak_memory, red, reduce, scan
from pdum.tl.autodiff import grad, numeric_grad
from pdum.tl.compute import repeat_like
from pdum.tl.dialect import (
    TL_OPS,
    fold_region,
    region_names,
    run_named,
    run_region,
    tensor_type_of_layout,
)
from pdum.tl.lifting import lift_step


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def _loss_region(inner, fold, out_name, dims, param_names):
    """Extend a fold-bearing region with loss = sum(out^2) over ``dims``,
    yielding (out, loss) under the naming law."""
    b = Builder({**CORE_OPS, **TL_OPS})
    sq = b.emit("tl.pointwise", fold, fold, f="mul")
    loss = b.emit("tl.reduce", sq, f="sum", dims=dims)
    region = Region(params=inner.params, body=(b.emit("core.yield", b.emit("core.tuple", fold, loss)),))
    names = region_names(region, param_names, {id(fold): out_name, id(loss): "loss"})
    return region, names


RNG = np.random.default_rng(23)

# ----------------------------------------------------------------------
# gated linear attention (Mamba-2/DeltaNet-lite): S_t = a_t S + k_t v_t^T
# ----------------------------------------------------------------------

DK, DV, TN = 3, 2, 4


def _gla_step(S, a, kk, vv, qq):
    S1 = repeat_like(a, S) * S + repeat_like(kk, S) * repeat_like(vv, S)
    y = reduce(red.sum, S1 * repeat_like(qq, S), ("p",))
    return S1, y


def _gla_model():
    """The GLA fold + loss = sum(ys^2) over (t, r), on the region face.

    The state is authored (r, p)-ordered: the adjoint of the emit path
    (reduce over "p") derives an (r, p)-ordered state cotangent, and the
    region fold's positional carry check compares types ORDER-SENSITIVELY —
    a (p, r) state refuses at build ("changes the state type")."""
    ls = lift_step(
        _gla_step,
        S=T(np.zeros((DV, DK)), ("r", "p")).layout,
        a=T(0.0, ()).layout,
        kk=T(np.zeros(DK), ("p",)).layout,
        vv=T(np.zeros(DV), ("r",)).layout,
        qq=T(np.zeros(DK), ("p",)).layout,
    )
    inner, fold = fold_region(
        ls.region,
        dim="t",
        state=("S",),
        element=("a", "kk", "vv", "qq"),
        out=("emit",),
        init_types=(tensor_type_of_layout(T(np.zeros((DV, DK)), ("r", "p")).layout),),
        src_types=(
            tensor_type_of_layout(T(np.zeros(TN), ("t",)).layout),
            tensor_type_of_layout(T(np.zeros((TN, DK)), ("t", "p")).layout),
            tensor_type_of_layout(T(np.zeros((TN, DV)), ("t", "r")).layout),
            tensor_type_of_layout(T(np.zeros((TN, DK)), ("t", "p")).layout),
        ),
    )
    return _loss_region(inner, fold, "ys", ("t", "r"), ("S0", "a", "k", "v", "q"))


def _gla_inputs():
    return {
        "S0": T(RNG.standard_normal((DV, DK)), ("r", "p")),
        "a": T(RNG.uniform(0.5, 1.0, TN), ("t",)),
        "k": T(RNG.standard_normal((TN, DK)), ("t", "p")),
        "v": T(RNG.standard_normal((TN, DV)), ("t", "r")),
        "q": T(RNG.standard_normal((TN, DK)), ("t", "p")),
    }


def _gla_ref(inputs):
    S = inputs["S0"].to_numpy(order=("p", "r")).copy()
    a, k, v, q = (inputs[n].to_numpy() for n in ("a", "k", "v", "q"))
    ys = []
    for t in range(TN):
        S = a[t] * S + np.outer(k[t], v[t])
        ys.append(S.T @ q[t])
    return np.stack(ys)


def test_gla_fold_matches_the_recurrence():
    inputs = _gla_inputs()
    region, names = _gla_model()
    vals = run_named(region, inputs, names)
    np.testing.assert_allclose(vals["ys"].to_numpy(order=("t", "r")), _gla_ref(inputs), rtol=1e-12)


def test_gla_fold_gradients_match_fd():
    inputs = _gla_inputs()
    region, names = _gla_model()
    rg = grad(region, "loss", inputs, names=names)
    vals = run_named(rg.region, inputs, rg.names)
    for wrt in ("S0", "a", "k", "v", "q"):
        fd = numeric_grad(region, "loss", wrt, inputs, names)
        got = vals[rg.grads[wrt]].to_numpy(order=inputs[wrt].names)
        np.testing.assert_allclose(got, fd, rtol=2e-5, atol=1e-7)


# ----------------------------------------------------------------------
# 1D FDTD leapfrog: two field states, no per-step elements (extent-driven)
# ----------------------------------------------------------------------

N, NT, C = 6, 4, 0.3


def _fdtd_step(E, H):
    H1 = H + C * (E.shift(x=-1).slice(x=(0, N - 1)) - E.slice(x=(0, N - 1)))
    dH = (H1.slice(x=(1, N - 1)) - H1.shift(x=1).slice(x=(1, N - 1))).pad(0.0, x=(0, N))
    E1 = E + C * dH
    return E1, H1


def _fdtd_step_emit(E, H):
    E1, H1 = _fdtd_step(E, H)
    return E1, H1, E1


def _fdtd_model(out=("final", 0), steps=NT):
    E0, H0 = _fdtd_ref()
    step = _fdtd_step_emit if out[0] == "emit" else _fdtd_step
    ls = lift_step(step, E=E0.layout, H=H0.layout)
    inner, fold = fold_region(
        ls.region,
        dim="t",
        state=("E", "H"),
        element=(),
        out=out,
        init_types=(tensor_type_of_layout(E0.layout), tensor_type_of_layout(H0.layout)),
        extent=(0, steps),
    )
    dims = ("x",) if out[0] == "final" else ("t", "x")
    return _loss_region(inner, fold, "Ef", dims, ("E0", "H0"))


def _fdtd_ref():
    E = np.zeros(N)
    E[2] = 1.0  # a pulse
    H = np.zeros(N - 1)
    return T(E, ("x",)), T(H, ("x",))


def _fdtd_loop(E0, H0, steps=NT):
    E, H = E0.copy(), H0.copy()
    traj = []
    for _ in range(steps):
        H = H + C * (E[1:] - E[:-1])
        dH = np.zeros(N)
        dH[1 : N - 1] = H[1 : N - 1] - H[0 : N - 2]
        E = E + C * dH
        traj.append(E.copy())
    return E, np.stack(traj)


def test_fdtd_fold_matches_the_time_loop():
    E0, H0 = _fdtd_ref()
    inputs = {"E0": E0, "H0": H0}
    Ef, traj = _fdtd_loop(E0.to_numpy(), H0.to_numpy())
    region, names = _fdtd_model()
    np.testing.assert_allclose(run_named(region, inputs, names)["Ef"].to_numpy(), Ef, rtol=1e-12)
    region2, names2 = _fdtd_model(out=("emit",))
    np.testing.assert_allclose(run_named(region2, inputs, names2)["Ef"].to_numpy(order=("t", "x")), traj, rtol=1e-12)


def test_fdtd_adjoint_matches_fd():
    E0, H0 = _fdtd_ref()
    inputs = {"E0": E0, "H0": H0}
    for out in (("final", 0), ("emit",)):
        region, names = _fdtd_model(out=out)
        rg = grad(region, "loss", inputs, names=names)
        vals = run_named(rg.region, inputs, rg.names)
        for wrt in ("E0", "H0"):
            fd = numeric_grad(region, "loss", wrt, inputs, names)
            np.testing.assert_allclose(vals[rg.grads[wrt]].to_numpy(), fd, rtol=2e-5, atol=1e-8)


# ----------------------------------------------------------------------
# consistency, edges, refusals
# ----------------------------------------------------------------------


def _lin_step(h, av, bv):
    h1 = av * h + bv
    return h1, h1


def _lin_scan(a, b):
    h = scan("linrec_f", (a, b), "t")
    loss = reduce(red.sum, h * h, ("t",))
    return h, loss


def test_scalar_fold_matches_composite_linrec():
    defreducer(
        "linrec_f",
        state=2,
        element=2,
        lift=lambda a, b: (a, b),
        combine=lambda left, right: (left[0] * right[0], right[0] * left[1] + right[1]),
        init=(1.0, 0.0),
        project=lambda A, B: B,
    )
    n = 5
    inputs = {
        "h0": T(0.0, ()),
        "a": T(RNG.uniform(0.5, 1.1, n), ("t",)),
        "b": T(RNG.standard_normal(n), ("t",)),
    }

    def build(kind):
        if kind == "fold":
            ls = lift_step(_lin_step, h=T(0.0, ()).layout, av=T(0.0, ()).layout, bv=T(0.0, ()).layout)
            inner, fold = fold_region(
                ls.region,
                dim="t",
                state=("h",),
                element=("av", "bv"),
                out=("emit",),
                init_types=(tensor_type_of_layout(inputs["h0"].layout),),
                src_types=(tensor_type_of_layout(inputs["a"].layout), tensor_type_of_layout(inputs["b"].layout)),
            )
            region, names = _loss_region(inner, fold, "h", ("t",), ("h0", "a", "b"))
            ins = inputs
        else:
            ls = lift_step(_lin_scan, a=inputs["a"].layout, b=inputs["b"].layout)
            region, names = ls.region, ls.names
            ins = {k: inputs[k] for k in ("a", "b")}
        rg = grad(region, "loss", ins, names=names)
        vals = run_named(rg.region, ins, rg.names)
        h = run_named(region, ins, names)["h"]
        return h.to_numpy(), vals[rg.grads["a"]].to_numpy(), vals[rg.grads["b"]].to_numpy()

    hf, gaf, gbf = build("fold")
    hs, gas, gbs = build("scan")
    np.testing.assert_allclose(hf, hs, rtol=1e-10)
    np.testing.assert_allclose(gaf, gas, rtol=1e-8)
    np.testing.assert_allclose(gbf, gbs, rtol=1e-8)


def test_empty_fold_is_the_identity_and_grads_pass_through():
    E0, H0 = _fdtd_ref()
    inputs = {"E0": E0, "H0": H0}
    region, names = _fdtd_model(steps=0)
    vals = run_named(region, inputs, names)
    np.testing.assert_allclose(vals["Ef"].to_numpy(), E0.to_numpy())
    rg = grad(region, "loss", inputs, names=names)
    valsj = run_named(rg.region, inputs, rg.names)
    np.testing.assert_allclose(valsj[rg.grads["E0"]].to_numpy(), 2 * E0.to_numpy())  # d(sum E0^2)/dE0
    np.testing.assert_allclose(valsj[rg.grads["H0"]].to_numpy(), np.zeros(N - 1))


def test_fold_carry_drift_refused():
    # the region face refuses at BUILD (the type rule): a carry that changes
    # the state type never constructs (the incumbent refused at run time
    # with "state layout")
    def drift(E):
        return E.slice(x=(0, N - 1))

    ls = lift_step(drift, E=_fdtd_ref()[0].layout)
    with pytest.raises(TypeError, match="changes the state type"):
        fold_region(
            ls.region,
            dim="t",
            state=("E",),
            element=(),
            out=("final", 0),
            init_types=(tensor_type_of_layout(_fdtd_ref()[0].layout),),
            extent=(0, 2),
        )


# test_fold_final_must_be_a_carry: DELETED — the positional contract's
# out=("final", i) can only index a state; a non-carry final output is
# unrepresentable on the region face (the refusal's subject is gone).


def test_fold_ops_count_scales_with_steps():
    region, names = _gla_model()
    ops = ops_count(region, names=names)
    # per step: muls Sa+kv+Sq = 3*(DK*DV); adds S1 (DK*DV) + reduce (DK-1)*DV
    per_mul = 3 * DK * DV
    per_add = DK * DV + (DK - 1) * DV
    assert ops.per_var["ys"]["mul"] == per_mul * TN
    assert ops.per_var["ys"]["add"] == per_add * TN


# ----------------------------------------------------------------------
# segmented (checkpointed) fold adjoints: the memory/recompute curve
# ----------------------------------------------------------------------


def test_segmented_fold_adjoint_matches_and_trades_memory_for_ops():
    E0, H0 = _fdtd_ref()
    inputs = {"E0": E0, "H0": H0}
    region, names = _fdtd_model(out=("emit",), steps=12)
    results, peaks, costs = {}, {}, {}
    for K in (None, 2, 3, 6):
        rg = grad(region, "loss", inputs, fold_segments=K, names=names)
        vals = run_named(rg.region, inputs, rg.names)
        results[K] = {v: vals[rg.grads[v]].to_numpy() for v in ("E0", "H0")}
        peaks[K] = peak_memory(rg.region, inputs, names=rg.names).peak_bytes
        costs[K] = ops_count(rg.region, names=rg.names).weighted()
    for K in (2, 3, 6):
        for v in ("E0", "H0"):
            np.testing.assert_allclose(results[K][v], results[None][v], rtol=1e-9)
        assert costs[K] > costs[None]  # segments pay recompute...
    assert peaks[3] < peaks[None]  # ...to buy peak memory
    with pytest.raises(ValueError, match="divide"):
        grad(region, "loss", inputs, fold_segments=5, names=names)  # 12 % 5 != 0


def test_segmented_gla_gradients_match_store_all():
    inputs = _gla_inputs()
    region, names = _gla_model()
    rg0 = grad(region, "loss", inputs, names=names)
    rg2 = grad(region, "loss", inputs, fold_segments=2, names=names)
    e0, e2 = run_named(rg0.region, inputs, rg0.names), run_named(rg2.region, inputs, rg2.names)
    for v in ("S0", "a", "k", "v", "q"):
        np.testing.assert_allclose(
            e2[rg2.grads[v]].to_numpy(order=inputs[v].names),
            e0[rg0.grads[v]].to_numpy(order=inputs[v].names),
            rtol=1e-9,
        )


# ----------------------------------------------------------------------
# binomial revolve (Griewank & Walther): the same pieces, a log-T schedule
# ----------------------------------------------------------------------


def test_revolve_split_is_the_optimal_offline_schedule():
    # the DP split minimizes recompute; cross-check against a brute force,
    # and against the binomial invariant beta(s, r) = C(s+r, s)
    from functools import lru_cache
    from math import comb

    from pdum.tl.autodiff import _revolve_cost, _revolve_split

    @lru_cache(maxsize=None)
    def brute(s, length):  # same recurrence, independently memoized
        if length <= 1 or s >= length:
            return 0.0
        if s < 1:
            return float("inf")
        return min(m + brute(s - 1, length - m) + brute(s, m) for m in range(1, length))

    for s in range(1, 6):
        for length in range(2, 30):
            if s >= length:
                continue  # a leaf (fits in the slots): never split
            m = _revolve_split(s, length)
            assert 1 <= m < length
            assert m + _revolve_cost(s - 1, length - m) + _revolve_cost(s, m) == brute(s, length)
        # the binomial invariant: with s slots and r recomputes you reverse up
        # to beta = C(s+r, s) steps, and such a full chain re-advances exactly
        # r times per step at the extreme — the cost is finite and matches
        for r in range(1, 5):
            assert _revolve_cost(s, comb(s + r, s)) < float("inf")
    assert _revolve_split(1, 8) == 7  # one slot: forced triangular (advance to hi-1)


def test_revolve_fold_adjoint_matches_store_all_no_divisibility():
    # FDTD with T=16 (divisible) and T=13 (NOT divisible by any of 2,3,4 —
    # fold_segments would refuse; revolve does not care)
    E0, H0 = _fdtd_ref()
    inputs = {"E0": E0, "H0": H0}
    for steps in (16, 13):
        region, names = _fdtd_model(out=("final", 0), steps=steps)
        rg0 = grad(region, "loss", inputs, names=names)
        e0 = run_named(rg0.region, inputs, rg0.names)
        for S in (2, 3, 4):
            rg = grad(region, "loss", inputs, fold_slots=S, names=names)
            e = run_named(rg.region, inputs, rg.names)
            for v in ("E0", "H0"):
                np.testing.assert_allclose(e[rg.grads[v]].to_numpy(), e0[rg0.grads[v]].to_numpy(), rtol=1e-9)
    # 13 is prime: fold_segments has no interior divisor, revolve reversed it
    region13, names13 = _fdtd_model(out=("final", 0), steps=13)
    with pytest.raises(ValueError, match="divide"):
        grad(region13, "loss", inputs, fold_segments=4, names=names13)


def test_revolve_gla_gradients_match_store_all():
    inputs = _gla_inputs()
    region, names = _gla_model()  # TN=4 elements present, emit-trajectory output
    rg0 = grad(region, "loss", inputs, names=names)
    e0 = run_named(rg0.region, inputs, rg0.names)
    for S in (1, 2, 3):
        rg = grad(region, "loss", inputs, fold_slots=S, names=names)
        e = run_named(rg.region, inputs, rg.names)
        for v in ("S0", "a", "k", "v", "q"):
            np.testing.assert_allclose(
                e[rg.grads[v]].to_numpy(order=inputs[v].names),
                e0[rg0.grads[v]].to_numpy(order=inputs[v].names),
                rtol=1e-9,
            )


def test_revolve_three_way_memory_table():
    E0, H0 = _fdtd_ref()
    inputs = {"E0": E0, "H0": H0}
    # out=final: the trajectory is NOT the output, so holding it is a pure
    # backward cost — exactly what checkpointing removes (the emit variant
    # materializes the whole space-time output regardless, masking the win)
    region, names = _fdtd_model(out=("final", 0), steps=24)

    def peak_ops(**kw):
        rg = grad(region, "loss", inputs, names=names, **kw)
        peak = peak_memory(rg.region, inputs, names=rg.names).peak_bytes
        return peak, ops_count(rg.region, names=rg.names).weighted()

    store_peak, store_ops = peak_ops()
    unif_peak = min(peak_ops(fold_segments=K)[0] for K in (4, 6, 8))  # K≈√24
    for S in (1, 2, 3, 4):
        rev_peak, rev_ops = peak_ops(fold_slots=S)
        assert rev_peak < store_peak  # revolve buys peak vs store-all...
        assert rev_ops > store_ops  # ...by paying recompute
        assert rev_peak <= unif_peak  # and undercuts uniform's √T minimum
    # the tradeoff is monotone: more slots -> more peak, less recompute
    peaks = [peak_ops(fold_slots=S)[0] for S in (1, 2, 3, 4, 5)]
    opses = [peak_ops(fold_slots=S)[1] for S in (1, 2, 3, 4, 5)]
    assert peaks == sorted(peaks)
    assert opses == sorted(opses, reverse=True)


def test_revolve_knob_exclusivity_and_degenerate_slots():
    E0, H0 = _fdtd_ref()
    inputs = {"E0": E0, "H0": H0}
    region, names = _fdtd_model(out=("final", 0), steps=8)
    rg0 = grad(region, "loss", inputs, names=names)
    e0 = run_named(rg0.region, inputs, rg0.names)
    # both knobs at once is refused
    with pytest.raises(ValueError, match="not both"):
        grad(region, "loss", inputs, fold_segments=2, fold_slots=2, names=names)
    with pytest.raises(ValueError, match="must be >= 1"):
        grad(region, "loss", inputs, fold_slots=0, names=names)
    # S=1 works (degenerate, recompute-heavy triangular schedule)
    rg1 = grad(region, "loss", inputs, fold_slots=1, names=names)
    e1 = run_named(rg1.region, inputs, rg1.names)
    for v in ("E0", "H0"):
        np.testing.assert_allclose(e1[rg1.grads[v]].to_numpy(), e0[rg0.grads[v]].to_numpy(), rtol=1e-9)
    # S >= T: enough slots to hold everything -> collapses to store-all, and
    # its peak equals the store-all peak exactly (a single full leaf)
    rgb = grad(region, "loss", inputs, fold_slots=99, names=names)
    eb = run_named(rgb.region, inputs, rgb.names)
    for v in ("E0", "H0"):
        np.testing.assert_allclose(eb[rgb.grads[v]].to_numpy(), e0[rg0.grads[v]].to_numpy(), rtol=1e-9)
    assert (
        peak_memory(rgb.region, inputs, names=rgb.names).peak_bytes
        == peak_memory(rg0.region, inputs, names=rg0.names).peak_bytes
    )


# --- the Region face (the excavation, LEVELS) -------------------------------


def test_region_fold_runs_the_physics_entries():
    """Region-first: the fold-bearing zoo regions under run_region (the
    positional door) match their numpy denotations — multi-state (FDTD)
    and elementless (both) covered."""
    from pdum.tl.zoo import fdtd1d_staggered, heat2d

    for m, ins in ((heat2d(), ("u0",)), (fdtd1d_staggered(), ("E0", "H0"))):
        got = run_region(m.region, [m.inputs[k] for k in ins])
        np.testing.assert_allclose(got.to_numpy(order=m.order), m.ref(m.numpy_inputs()), rtol=1e-9, atol=1e-12)


def test_region_fold_emit_with_element_matches_the_recurrence():
    """An emit fold with an element source: region authoring (positional
    yield contract) against the recurrence computed by hand.
    (The incumbent-Program comparison and the export_program round-trip
    died with the Program IR — export_program is deleted.)"""

    def step(s, m):
        s1 = s + m
        e = s1 * s1
        return s1, e

    s0 = Tensor.from_numpy(np.arange(3.0), ("x",))
    src = Tensor.from_numpy(np.arange(12.0).reshape(4, 3), ("tm", "x"))
    elem = Tensor.from_numpy(np.zeros(3), ("x",))
    ls = lift_step(step, s=s0.layout, m=elem.layout)
    region, fold = fold_region(
        ls.region,
        dim="tm",
        state=("s",),
        element=("m",),
        out=("emit",),
        init_types=(tensor_type_of_layout(s0.layout),),
        src_types=(tensor_type_of_layout(src.layout),),
    )
    got = run_region(region, [s0, src])
    s = s0.to_numpy().copy()
    rows = []
    for t in range(4):
        s = s + src.to_numpy()[t]
        rows.append(s * s)
    np.testing.assert_array_equal(got.to_numpy(order=("tm", "x")), np.stack(rows))


def test_region_analyses_cover_folds():
    """The four analyses serve fold-bearing regions (elementless,
    multi-state, charted states) — direct coverage pins now that the
    Program-face differential is gone."""
    from pdum.tl import infer_signatures, mesh, traffic
    from pdum.tl.zoo import fdtd1d_staggered, heat2d

    for m in (heat2d(), fdtd1d_staggered()):
        ro = ops_count(m.region, names=m.names)
        assert m.out in ro.per_var and sum(ro.total.values()) > 0
        rm = peak_memory(m.region, m.inputs, names=m.names)
        assert rm.peak_bytes > 0
        assert rm.input_bytes == sum(v.to_numpy().size * 8 for v in m.inputs.values())
        rs = infer_signatures(m.region, m.inputs, names=m.names)
        assert m.out in rs
        tr = traffic(m.region, None, mesh(2), names=m.names)
        assert tr.collectives == ()  # nothing bound: the fold moves no traffic


def test_fold_adjoint_tolerates_permuted_carry_dims():
    """Regression (the excavation): the derived state cotangent may carry
    its dims in a different ORDER than the state — presentation order is
    never semantics (220 §11), so the carry check compares frames
    order-insensitively and the gradient still derives."""
    from pdum.dsl.ir import Builder, Region
    from pdum.dsl.ops import CORE_OPS
    from pdum.tl.autodiff import grad, numeric_grad
    from pdum.tl.dialect import TL_OPS, fold_region, region_names, tensor_type_of_layout
    from pdum.tl.lifting import lift_step

    def step(S, m):
        S1 = S * 0.9 + m.repeat("p", (0, 3)) * 0.1
        return S1

    S0 = T(np.arange(6.0).reshape(3, 2), ("p", "r"))  # state ordered (p, r)
    src = T(np.arange(8.0).reshape(4, 2), ("tm", "r"))
    elem = T(np.zeros(2), ("r",))
    ls = lift_step(step, S=S0.layout, m=elem.layout)
    region, fold = fold_region(
        ls.region,
        dim="tm",
        state=("S",),
        element=("m",),
        out=("final", 0),
        init_types=(tensor_type_of_layout(S0.layout),),
        src_types=(tensor_type_of_layout(src.layout),),
    )
    b = Builder({**CORE_OPS, **TL_OPS})
    loss = b.emit("tl.reduce", fold, f="sum", dims=("p", "r"))
    r2 = Region(params=region.params, body=(b.emit("core.yield", loss),))
    names = region_names(r2, ("S0", "src"), {id(fold): "Sf", id(loss): "loss"})
    rg = grad(r2, "loss", {"S0": S0, "src": src}, wrt=("S0", "src"), names=names)
    from pdum.tl.dialect import run_named

    vals = run_named(rg.region, {"S0": S0, "src": src}, rg.names)
    for v in ("S0", "src"):
        got = vals[rg.grads[v]].to_numpy(order={"S0": ("p", "r"), "src": ("tm", "r")}[v])
        want = numeric_grad(r2, "loss", v, {"S0": S0, "src": src}, names)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-7)
