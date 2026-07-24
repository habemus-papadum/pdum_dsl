"""Tensor-typed lifting (S.2): plain functions over tensor-typed parameters
lower to step Programs; Literal-annotated parameters are structural; the
structural-slot refusal names the annotation fix."""

from fractions import Fraction

import numpy as np
import pytest
from pdum.dsl.types import Literal
from pdum.tl import Tensor
from pdum.tl.chart import chart
from pdum.tl.ir import Instr, Program, run
from pdum.tl.lifting import lift_step

e_chart = chart(0, 1, axis="x")
h_chart = chart(Fraction(1, 2), 1, axis="x")
C = 0.4


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def _ghost(u, dim, extent, delta):
    # u[i - delta] with a zero ghost outside: shift, slice, pad(0)
    lo, hi = extent
    sh = u.shift(**{dim: delta})
    sl = sh.slice(**{dim: (lo + max(delta, 0), hi + min(delta, 0))})
    return sl.pad(fill=0.0, **{dim: extent})


def heat_step(u, n: Literal[int], alpha: Literal[float]):
    lap = _ghost(u, "x", (0, n), 1) + _ghost(u, "x", (0, n), -1) - 2.0 * u
    return u + alpha * lap


def test_heat_step_lifts_and_folds():
    n, alpha, steps = 6, 0.1, 4
    u0 = T(np.random.default_rng(2).standard_normal(n), ("x",))
    ls = lift_step(heat_step, u=u0.layout, n=n, alpha=alpha)
    assert ls.inputs == ("u",)
    prog = Program(
        (
            Instr("u0", "input"),
            Instr(
                "uf",
                "fold",
                ("u0",),
                {
                    "step": ls.program,
                    "dim": "t",
                    "state": ("u",),
                    "element": (),
                    "carry": {"u": ls.outputs[0]},
                    "out": ("final", ls.outputs[0]),
                    "extent": (0, steps),
                },
            ),
        )
    )
    got = run(prog, {"u0": u0})["uf"].to_numpy()
    ref = u0.to_numpy().copy()
    for _ in range(steps):
        up = np.zeros(n + 2)
        up[1:-1] = ref
        ref = ref + alpha * (up[:-2] + up[2:] - 2 * ref)
    np.testing.assert_allclose(got, ref, rtol=1e-12)


def fdtd_step(E, H, n: Literal[int]):
    # dE_i = E_{i+1} - E_i lives at i + 1/2 — each operand SAYS so, exactly,
    # before combining (the recharting is the discretization's honesty)
    Ea = E.shift(x=-1).slice(x=(0, n - 1)).with_charts(x=h_chart)
    Eb = E.slice(x=(0, n - 1)).with_charts(x=h_chart)
    H1 = H + C * (Ea - Eb)
    Ha = H1.slice(x=(1, n - 1)).with_charts(x=e_chart)
    Hb = H1.shift(x=1).slice(x=(1, n - 1)).with_charts(x=e_chart)
    E1 = E + C * (Ha - Hb).pad(x=(0, n), fill=0.0)
    return E1, H1


def test_fdtd_step_the_spec_example_lifts_with_charts():
    """The S.2 worked example, nearly verbatim: staggered charts carried by
    the lifted step; c lifts inheriting dims AND charts; layout preserved."""
    n = 6
    rng = np.random.default_rng(7)
    E0 = T(rng.standard_normal(n), ("x",)).with_charts(x=e_chart)
    H0 = T(rng.standard_normal(n - 1), ("x",)).with_charts(x=h_chart)
    ls = lift_step(fdtd_step, E=E0.layout, H=H0.layout, n=n)
    assert ls.inputs == ("E", "H") and len(ls.outputs) == 2
    env = run(ls.program, {"E": E0, "H": H0})
    E1, H1 = env[ls.outputs[0]], env[ls.outputs[1]]
    # denotation vs plain numpy leapfrog
    e, h = E0.to_numpy(), H0.to_numpy()
    h1 = h + C * (e[1:] - e[:-1])
    e1 = e.copy()  # boundary entries hold: the pad(0) IS the boundary condition
    e1[1:-1] = e[1:-1] + C * (h1[1:] - h1[:-1])
    np.testing.assert_allclose(H1.to_numpy(), h1, rtol=1e-12)
    np.testing.assert_allclose(E1.to_numpy(), e1, rtol=1e-12)
    # the staggering survived: E1 on the integer grid, H1 on the half grid
    assert E1.layout.dim("x").chart == e_chart
    assert H1.layout.dim("x").chart == h_chart


def test_structural_slot_refuses_a_tensor_naming_the_fix():
    def bad(u, k):
        return u.slice(x=(0, k))  # k is tensor-typed here: a runtime value shaping the lattice

    with pytest.raises(ValueError, match=r"STRUCTURAL slot.*Literal\[int\].*200 §1.5"):
        lift_step(bad, u=T([1.0, 2.0], ("x",)).layout, k=T([1.0], ("x",)).layout)


def test_unannotated_parameter_refuses_a_plain_int_naming_the_annotation():
    def step(u, n):
        return u.slice(x=(0, n))

    with pytest.raises(ValueError, match=r"structural parameters declare themselves.*Literal\[int\]"):
        lift_step(step, u=T([1.0, 2.0], ("x",)).layout, n=2)


def test_literal_annotation_type_checks():
    def step(u, n: Literal[int]):
        return u.slice(x=(0, n))

    with pytest.raises(ValueError, match=r"Literal\[int\]; got 2.5"):
        lift_step(step, u=T([1.0, 2.0], ("x",)).layout, n=2.5)


def test_step_bodies_are_straight_line():
    def bad(u):
        for _ in range(3):
            u = u + u
        return u

    with pytest.raises(ValueError, match="straight-line"):
        lift_step(bad, u=T([1.0], ("x",)).layout)
