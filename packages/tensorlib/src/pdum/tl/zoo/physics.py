"""Physics entries: time-stepped fields via fold, boundaries via guards,
staggering via exact charts — step bodies in the SHARED SYNTAX (S.2): plain
functions over tensor-typed parameters, lifted to step Programs.

`heat2d` — explicit Euler with Dirichlet-0 ghosts: every neighbor access
is shift+slice+pad(0) (the ghost helper INLINES at lifting), so the
boundary condition IS the guard fill.

`fdtd1d_staggered` — 1D leapfrog on a Yee grid with the staggering carried
by CHARTS: E lives at integer x, H at half-integer x (exact Fraction(1,2)
origins). Differencing E produces values that physically live on the H
grid — and alignment (D17) refuses to combine them until the program SAYS
so via with_charts. The recharting is the discretization's honesty made
syntax: every half-step move is explicit and exact."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from pdum.dsl.types import Literal

from ..build import Build
from ..chart import chart
from ..lifting import lift_step
from .zoo_common import ZooModel, t_in


def _ghost(u, dim, extent, delta):
    # u[i - delta] with a zero ghost outside: shift, slice, pad(0)
    lo, hi = extent
    sh = u.shift(**{dim: delta})
    sl = sh.slice(**{dim: (lo + max(delta, 0), hi + min(delta, 0))})
    return sl.pad(fill=0.0, **{dim: extent})


def _heat_step(u, n: Literal[int], m: Literal[int], alpha: Literal[float]):
    nsum = _ghost(u, "x", (0, n), 1) + _ghost(u, "x", (0, n), -1) + _ghost(u, "y", (0, m), 1) + _ghost(
        u, "y", (0, m), -1
    )
    lap = nsum - 4.0 * u
    return u + alpha * lap


def heat2d(N=5, M=5, T=3, alpha=0.1, seed=13) -> ZooModel:
    rng = np.random.default_rng(seed)
    b = Build()
    inputs: dict = {}
    b.input(t_in(inputs, "u0", rng.standard_normal((N, M)), ("x", "y")))
    ls = lift_step(_heat_step, u=inputs["u0"].layout, n=N, m=M, alpha=alpha)
    uf = b.emit(
        "fold",
        ("u0",),
        hint="uf",
        step=ls.program,
        dim="tm",
        state=("u",),
        element=(),
        carry={"u": ls.outputs[0]},
        out=("final", ls.outputs[0]),
        extent=(0, T),
    )

    def ref(inp):
        u = inp["u0"].copy()
        for _ in range(T):
            up = np.zeros((N + 2, M + 2))
            up[1:-1, 1:-1] = u
            lap = up[:-2, 1:-1] + up[2:, 1:-1] + up[1:-1, :-2] + up[1:-1, 2:] - 4 * u
            u = u + alpha * lap
        return u

    return ZooModel(b.program(), inputs, uf, ref, ("x", "y"))


def fdtd1d_staggered(N=6, T=3, c=0.4, seed=17) -> ZooModel:
    e_chart = chart(0, 1, axis="x")
    h_chart = chart(Fraction(1, 2), 1, axis="x")
    rng = np.random.default_rng(seed)
    E0 = np.zeros(N)
    E0[N // 2] = 1.0
    E0 += 0.1 * rng.standard_normal(N)
    H0 = 0.1 * rng.standard_normal(N - 1)
    inputs = {
        "E0": Tensor_from(E0, e_chart),
        "H0": Tensor_from(H0, h_chart),
    }
    b = Build()
    b.input("E0")
    b.input("H0")

    def step(E, H, n: Literal[int]):
        # dE_i = E_{i+1} - E_i lives at i + 1/2 — each operand SAYS so,
        # exactly, before combining; c lifts inheriting dims AND charts
        Ea = E.shift(x=-1).slice(x=(0, n - 1)).with_charts(x=h_chart)
        Eb = E.slice(x=(0, n - 1)).with_charts(x=h_chart)
        H1 = H + c * (Ea - Eb)
        # dH_i = H1_{i+1/2} - H1_{i-1/2} lives back at integer i
        Ha = H1.slice(x=(1, n - 1)).with_charts(x=e_chart)
        Hb = H1.shift(x=1).slice(x=(1, n - 1)).with_charts(x=e_chart)
        E1 = E + c * (Ha - Hb).pad(x=(0, n), fill=0.0)
        return E1, H1

    ls = lift_step(step, E=inputs["E0"].layout, H=inputs["H0"].layout, n=N)
    ef = b.emit(
        "fold",
        ("E0", "H0"),
        hint="Ef",
        step=ls.program,
        dim="tm",
        state=("E", "H"),
        element=(),
        carry={"E": ls.outputs[0], "H": ls.outputs[1]},
        out=("final", ls.outputs[0]),
        extent=(0, T),
    )

    def ref(inp):
        E, H = inp["E0"].copy(), inp["H0"].copy()
        for _ in range(T):
            H = H + c * (E[1:] - E[:-1])
            dH = np.zeros(N)
            dH[1 : N - 1] = H[1 : N - 1] - H[0 : N - 2]
            E = E + c * dH
        return E

    return ZooModel(b.program(), inputs, ef, ref, ("x",))


def Tensor_from(arr, ch):
    from ..tensor import Tensor

    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), ("x",)).with_charts(x=ch)
