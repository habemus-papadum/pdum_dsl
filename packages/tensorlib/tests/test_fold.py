"""fold — the tensor-state scan: programs as step functions, derived BPTT."""

import numpy as np
import pytest
from pdum.tl import Tensor, defreducer, ops_count
from pdum.tl.autodiff import grad, numeric_grad
from pdum.tl.ir import Instr, Program, run


def I(var, op, operands=(), **params):  # noqa: E743
    return Instr(var, op, tuple(operands), params)


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


RNG = np.random.default_rng(23)

# ----------------------------------------------------------------------
# gated linear attention (Mamba-2/DeltaNet-lite): S_t = a_t S + k_t v_t^T
# ----------------------------------------------------------------------

DK, DV, TN = 3, 2, 4

GLA_STEP = Program(
    (
        I("S", "input"),
        I("a", "input"),
        I("kk", "input"),
        I("vv", "input"),
        I("qq", "input"),
        I("a1", "repeat", ["a"], name="p", extent=(0, DK)),
        I("ar", "repeat", ["a1"], name="r", extent=(0, DV)),
        I("kr", "repeat", ["kk"], name="r", extent=(0, DV)),
        I("vr", "repeat", ["vv"], name="p", extent=(0, DK)),
        I("Sa", "pointwise", ["ar", "S"], f="mul"),
        I("kv", "pointwise", ["kr", "vr"], f="mul"),
        I("S1", "pointwise", ["Sa", "kv"], f="add"),
        I("qr", "repeat", ["qq"], name="r", extent=(0, DV)),
        I("Sq", "pointwise", ["S1", "qr"], f="mul"),
        I("y", "reduce", ["Sq"], f="sum", dims=("p",)),
    )
)


def _gla_prog():
    return Program(
        (
            I("S0", "input"),
            I("a", "input"),
            I("k", "input"),
            I("v", "input"),
            I("q", "input"),
            I(
                "ys",
                "fold",
                ["S0", "a", "k", "v", "q"],
                step=GLA_STEP,
                dim="t",
                state=("S",),
                element=("a", "kk", "vv", "qq"),
                carry={"S": "S1"},
                out=("emit", "y"),
            ),
            I("y2", "pointwise", ["ys", "ys"], f="mul"),
            I("loss", "reduce", ["y2"], f="sum", dims=("t", "r")),
        )
    )


def _gla_inputs():
    return {
        "S0": T(RNG.standard_normal((DK, DV)), ("p", "r")),
        "a": T(RNG.uniform(0.5, 1.0, TN), ("t",)),
        "k": T(RNG.standard_normal((TN, DK)), ("t", "p")),
        "v": T(RNG.standard_normal((TN, DV)), ("t", "r")),
        "q": T(RNG.standard_normal((TN, DK)), ("t", "p")),
    }


def _gla_ref(inputs):
    S = inputs["S0"].to_numpy().copy()
    a, k, v, q = (inputs[n].to_numpy() for n in ("a", "k", "v", "q"))
    ys = []
    for t in range(TN):
        S = a[t] * S + np.outer(k[t], v[t])
        ys.append(S.T @ q[t])
    return np.stack(ys)


def test_gla_fold_matches_the_recurrence():
    inputs = _gla_inputs()
    env = run(_gla_prog(), inputs)
    np.testing.assert_allclose(env["ys"].to_numpy(order=("t", "r")), _gla_ref(inputs), rtol=1e-12)


def test_gla_fold_gradients_match_fd():
    inputs = _gla_inputs()
    prog = _gla_prog()
    jp, grads = grad(prog, "loss", inputs)
    env = run(jp, inputs)
    for wrt in ("S0", "a", "k", "v", "q"):
        fd = numeric_grad(prog, "loss", wrt, inputs)
        np.testing.assert_allclose(env[grads[wrt]].to_numpy(), fd, rtol=2e-5, atol=1e-7)


# ----------------------------------------------------------------------
# 1D FDTD leapfrog: two field states, no per-step elements (extent-driven)
# ----------------------------------------------------------------------

N, NT, C = 6, 4, 0.3

FDTD_STEP = Program(
    (
        I("E", "input"),
        I("H", "input"),
        I("Es", "shift", ["E"], deltas={"x": -1}),
        I("Ea", "slice", ["Es"], ranges={"x": (0, N - 1)}),
        I("Eb", "slice", ["E"], ranges={"x": (0, N - 1)}),
        I("dE", "pointwise", ["Ea", "Eb"], f="sub"),
        I("cH", "const", [], value=C, dims=(("x", (0, N - 1)),)),
        I("dHs", "pointwise", ["cH", "dE"], f="mul"),
        I("H1", "pointwise", ["H", "dHs"], f="add"),
        I("Hs", "shift", ["H1"], deltas={"x": 1}),
        I("Ha", "slice", ["H1"], ranges={"x": (1, N - 1)}),
        I("Hb", "slice", ["Hs"], ranges={"x": (1, N - 1)}),
        I("dH", "pointwise", ["Ha", "Hb"], f="sub"),
        I("pdH", "pad", ["dH"], fill=0.0, extents={"x": (0, N)}),
        I("cE", "const", [], value=C, dims=(("x", (0, N)),)),
        I("dEs", "pointwise", ["cE", "pdH"], f="mul"),
        I("E1", "pointwise", ["E", "dEs"], f="add"),
    )
)


def _fdtd_prog(out=("final", "E1"), steps=NT):
    return Program(
        (
            I("E0", "input"),
            I("H0", "input"),
            I(
                "Ef",
                "fold",
                ["E0", "H0"],
                step=FDTD_STEP,
                dim="t",
                state=("E", "H"),
                element=(),
                carry={"E": "E1", "H": "H1"},
                out=out,
                extent=(0, steps),
            ),
            I("E2", "pointwise", ["Ef", "Ef"], f="mul"),
            I("loss", "reduce", ["E2"], f="sum", dims=("x",) if out[0] == "final" else ("t", "x")),
        )
    )


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
    env = run(_fdtd_prog(), inputs)
    np.testing.assert_allclose(env["Ef"].to_numpy(), Ef, rtol=1e-12)
    env2 = run(_fdtd_prog(out=("emit", "E1")), inputs)
    np.testing.assert_allclose(env2["Ef"].to_numpy(order=("t", "x")), traj, rtol=1e-12)


def test_fdtd_adjoint_matches_fd():
    E0, H0 = _fdtd_ref()
    inputs = {"E0": E0, "H0": H0}
    for out in (("final", "E1"), ("emit", "E1")):
        prog = _fdtd_prog(out=out)
        jp, grads = grad(prog, "loss", inputs)
        env = run(jp, inputs)
        for wrt in ("E0", "H0"):
            fd = numeric_grad(prog, "loss", wrt, inputs)
            np.testing.assert_allclose(env[grads[wrt]].to_numpy(), fd, rtol=2e-5, atol=1e-8)


# ----------------------------------------------------------------------
# consistency, edges, refusals
# ----------------------------------------------------------------------


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
    step = Program(
        (
            I("h", "input"),
            I("av", "input"),
            I("bv", "input"),
            I("ah", "pointwise", ["av", "h"], f="mul"),
            I("h1", "pointwise", ["ah", "bv"], f="add"),
        )
    )
    n = 5
    inputs = {
        "h0": T(0.0, ()),
        "a": T(RNG.uniform(0.5, 1.1, n), ("t",)),
        "b": T(RNG.standard_normal(n), ("t",)),
    }

    def build(kind):
        if kind == "fold":
            head = I(
                "h",
                "fold",
                ["h0", "a", "b"],
                step=step,
                dim="t",
                state=("h",),
                element=("av", "bv"),
                carry={"h": "h1"},
                out=("emit", "h1"),
            )
            ins = (I("h0", "input"), I("a", "input"), I("b", "input"), head)
        else:
            ins = (I("a", "input"), I("b", "input"), I("h", "scan", ["a", "b"], f="linrec_f", dim="t"))
        prog = Program(
            ins
            + (
                I("hh", "pointwise", ["h", "h"], f="mul"),
                I("loss", "reduce", ["hh"], f="sum", dims=("t",)),
            )
        )
        jp, grads = grad(prog, "loss", inputs)
        env = run(jp, inputs)
        return env["h"].to_numpy(), env[grads["a"]].to_numpy(), env[grads["b"]].to_numpy()

    hf, gaf, gbf = build("fold")
    hs, gas, gbs = build("scan")
    np.testing.assert_allclose(hf, hs, rtol=1e-10)
    np.testing.assert_allclose(gaf, gas, rtol=1e-8)
    np.testing.assert_allclose(gbf, gbs, rtol=1e-8)


def test_empty_fold_is_the_identity_and_grads_pass_through():
    E0, H0 = _fdtd_ref()
    inputs = {"E0": E0, "H0": H0}
    prog = _fdtd_prog(steps=0)
    env = run(prog, inputs)
    np.testing.assert_allclose(env["Ef"].to_numpy(), E0.to_numpy())
    jp, grads = grad(prog, "loss", inputs)
    envj = run(jp, inputs)
    np.testing.assert_allclose(envj[grads["E0"]].to_numpy(), 2 * E0.to_numpy())  # d(sum E0^2)/dE0
    np.testing.assert_allclose(envj[grads["H0"]].to_numpy(), np.zeros(N - 1))


def test_fold_carry_drift_refused():
    step = Program(
        (
            I("E", "input"),
            I("E1", "slice", ["E"], ranges={"x": (0, N - 1)}),
        )
    )
    prog = Program(
        (
            I("E0", "input"),
            I(
                "Ef",
                "fold",
                ["E0"],
                step=step,
                dim="t",
                state=("E",),
                element=(),
                carry={"E": "E1"},
                out=("final", "E1"),
                extent=(0, 2),
            ),
        )
    )
    with pytest.raises(ValueError, match="state layout"):
        run(prog, {"E0": _fdtd_ref()[0]})


def test_fold_final_must_be_a_carry():
    with pytest.raises(ValueError, match="carry output"):
        run(
            Program(
                (
                    I("S0", "input"),
                    I("a", "input"),
                    I("k", "input"),
                    I("v", "input"),
                    I("q", "input"),
                    I(
                        "ys",
                        "fold",
                        ["S0", "a", "k", "v", "q"],
                        step=GLA_STEP,
                        dim="t",
                        state=("S",),
                        element=("a", "kk", "vv", "qq"),
                        carry={"S": "S1"},
                        out=("final", "y"),
                    ),
                )
            ),
            _gla_inputs(),
        )


def test_fold_ops_count_scales_with_steps():
    inputs = _gla_inputs()
    ops = ops_count(_gla_prog(), inputs)
    # per step: muls Sa+kv+Sq = 3*(DK*DV); adds S1 (DK*DV) + reduce (DK-1)*DV
    per_mul = 3 * DK * DV
    per_add = DK * DV + (DK - 1) * DV
    assert ops.per_var["ys"]["mul"] == per_mul * TN
    assert ops.per_var["ys"]["add"] == per_add * TN


# ----------------------------------------------------------------------
# segmented (checkpointed) fold adjoints: the memory/recompute curve
# ----------------------------------------------------------------------


def test_segmented_fold_adjoint_matches_and_trades_memory_for_ops():
    from pdum.tl.memory import peak_memory
    from pdum.tl.opcount import ops_count

    E0, H0 = _fdtd_ref()
    inputs = {"E0": E0, "H0": H0}
    prog = _fdtd_prog(out=("emit", "E1"), steps=12)
    results, peaks, costs = {}, {}, {}
    for K in (None, 2, 3, 6):
        jp, grads = grad(prog, "loss", inputs, fold_segments=K)
        env = run(jp, inputs)
        results[K] = {v: env[grads[v]].to_numpy() for v in ("E0", "H0")}
        peaks[K] = peak_memory(jp, inputs).peak_bytes
        costs[K] = ops_count(jp, inputs).weighted()
    for K in (2, 3, 6):
        for v in ("E0", "H0"):
            np.testing.assert_allclose(results[K][v], results[None][v], rtol=1e-9)
        assert costs[K] > costs[None]  # segments pay recompute...
    assert peaks[3] < peaks[None]  # ...to buy peak memory
    with pytest.raises(ValueError, match="divide"):
        grad(prog, "loss", inputs, fold_segments=5)  # 12 % 5 != 0


def test_segmented_gla_gradients_match_store_all():
    inputs = _gla_inputs()
    prog = _gla_prog()
    jp0, g0 = grad(prog, "loss", inputs)
    jp2, g2 = grad(prog, "loss", inputs, fold_segments=2)
    e0, e2 = run(jp0, inputs), run(jp2, inputs)
    for v in ("S0", "a", "k", "v", "q"):
        np.testing.assert_allclose(
            e2[g2[v]].to_numpy(order=inputs[v].names),
            e0[g0[v]].to_numpy(order=inputs[v].names),
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
        prog = _fdtd_prog(out=("final", "E1"), steps=steps)
        jp0, g0 = grad(prog, "loss", inputs)
        e0 = run(jp0, inputs)
        for S in (2, 3, 4):
            jp, g = grad(prog, "loss", inputs, fold_slots=S)
            e = run(jp, inputs)
            for v in ("E0", "H0"):
                np.testing.assert_allclose(e[g[v]].to_numpy(), e0[g0[v]].to_numpy(), rtol=1e-9)
    # 13 is prime: fold_segments has no interior divisor, revolve reversed it
    with pytest.raises(ValueError, match="divide"):
        grad(_fdtd_prog(out=("final", "E1"), steps=13), "loss", inputs, fold_segments=4)


def test_revolve_gla_gradients_match_store_all():
    inputs = _gla_inputs()
    prog = _gla_prog()  # TN=4 elements present, emit-trajectory output
    jp0, g0 = grad(prog, "loss", inputs)
    e0 = run(jp0, inputs)
    for S in (1, 2, 3):
        jp, g = grad(prog, "loss", inputs, fold_slots=S)
        e = run(jp, inputs)
        for v in ("S0", "a", "k", "v", "q"):
            np.testing.assert_allclose(
                e[g[v]].to_numpy(order=inputs[v].names),
                e0[g0[v]].to_numpy(order=inputs[v].names),
                rtol=1e-9,
            )


def test_revolve_three_way_memory_table():
    from pdum.tl.memory import peak_memory
    from pdum.tl.opcount import ops_count

    E0, H0 = _fdtd_ref()
    inputs = {"E0": E0, "H0": H0}
    # out=final: the trajectory is NOT the output, so holding it is a pure
    # backward cost — exactly what checkpointing removes (the emit variant
    # materializes the whole space-time output regardless, masking the win)
    prog = _fdtd_prog(out=("final", "E1"), steps=24)

    def peak_ops(**kw):
        jp, _ = grad(prog, "loss", inputs, **kw)
        return peak_memory(jp, inputs).peak_bytes, ops_count(jp, inputs).weighted()

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
    prog = _fdtd_prog(out=("final", "E1"), steps=8)
    jp0, g0 = grad(prog, "loss", inputs)
    e0 = run(jp0, inputs)
    # both knobs at once is refused
    with pytest.raises(ValueError, match="not both"):
        grad(prog, "loss", inputs, fold_segments=2, fold_slots=2)
    with pytest.raises(ValueError, match="must be >= 1"):
        grad(prog, "loss", inputs, fold_slots=0)
    # S=1 works (degenerate, recompute-heavy triangular schedule)
    jp1, g1 = grad(prog, "loss", inputs, fold_slots=1)
    e1 = run(jp1, inputs)
    for v in ("E0", "H0"):
        np.testing.assert_allclose(e1[g1[v]].to_numpy(), e0[g0[v]].to_numpy(), rtol=1e-9)
    # S >= T: enough slots to hold everything -> collapses to store-all, and
    # its peak equals the store-all peak exactly (a single full leaf)
    from pdum.tl.memory import peak_memory

    jpb, gb = grad(prog, "loss", inputs, fold_slots=99)
    eb = run(jpb, inputs)
    for v in ("E0", "H0"):
        np.testing.assert_allclose(eb[gb[v]].to_numpy(), e0[g0[v]].to_numpy(), rtol=1e-9)
    assert peak_memory(jpb, inputs).peak_bytes == peak_memory(jp0, inputs).peak_bytes
