"""Physics entries: time-stepped fields via fold, boundaries via guards,
staggering via exact charts.

`heat2d` — explicit Euler with Dirichlet-0 ghosts: every neighbor access
is shift+slice+pad(0), so the boundary condition IS the guard fill.

`fdtd1d_staggered` — 1D leapfrog on a Yee grid with the staggering carried
by CHARTS: E lives at integer x, H at half-integer x (exact Fraction(1,2)
origins). Differencing E produces values that physically live on the H
grid — and alignment (D17) refuses to combine them until the program SAYS
so via with_charts. The recharting is the discretization's honesty made
syntax: every half-step move is explicit and exact."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..build import Build
from ..chart import chart
from ..tensor import Tensor
from .zoo_common import ZooModel, t_in


def heat2d(N=5, M=5, T=3, alpha=0.1, seed=13) -> ZooModel:
    rng = np.random.default_rng(seed)
    b = Build()
    inputs: dict = {}
    b.input(t_in(inputs, "u0", rng.standard_normal((N, M)), ("x", "y")))
    shape = [("x", (0, N)), ("y", (0, M))]

    sb = Build()
    u = sb.input("u")

    def ghost(dim, extent, delta):
        # u[i - delta] with a zero ghost outside: shift, slice, pad(0)
        sh = sb.emit("shift", (u,), hint="sh", deltas={dim: delta})
        lo, hi = extent
        rng_ = (lo + max(delta, 0), hi + min(delta, 0))
        sl = sb.emit("slice", (sh,), hint="sl", ranges={dim: rng_})
        return sb.emit("pad", (sl,), hint="gh", fill=0.0, extents={dim: extent})

    nsum = sb.pw(
        "add",
        sb.pw("add", ghost("x", (0, N), 1), ghost("x", (0, N), -1)),
        sb.pw("add", ghost("y", (0, M), 1), ghost("y", (0, M), -1)),
        hint="nsum",
    )
    lap = sb.pw("sub", nsum, sb.pw("mul", sb.const(4.0, shape, hint="four"), u), hint="lap")
    u1 = sb.pw("add", u, sb.pw("mul", sb.const(alpha, shape, hint="alpha"), lap), hint="u1")

    uf = b.emit(
        "fold",
        ("u0",),
        hint="uf",
        step=sb.program(),
        dim="tm",
        state=("u",),
        element=(),
        carry={"u": u1},
        out=("final", u1),
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
        "E0": Tensor.from_numpy(E0, ("x",)).with_charts(x=e_chart),
        "H0": Tensor.from_numpy(H0, ("x",)).with_charts(x=h_chart),
    }
    b = Build()
    b.input("E0")
    b.input("H0")

    sb = Build()
    E = sb.input("E")
    H = sb.input("H")

    def rechart(v, ch):
        return sb.emit("with_charts", (v,), hint="rc", charts={"x": ch})

    def cst(value, extent, ch):
        return rechart(sb.const(value, [("x", extent)], hint="cc"), ch)

    # dE_i = E_{i+1} - E_i lives at i + 1/2 — say so, exactly
    Ea = sb.emit("slice", (sb.emit("shift", (E,), hint="Es", deltas={"x": -1}),), hint="Ea", ranges={"x": (0, N - 1)})
    Eb = sb.emit("slice", (E,), hint="Eb", ranges={"x": (0, N - 1)})
    dE = sb.pw("sub", rechart(Ea, h_chart), rechart(Eb, h_chart), hint="dE")
    H1 = sb.pw("add", H, sb.pw("mul", cst(c, (0, N - 1), h_chart), dE), hint="H1")
    # dH_i = H1_{i+1/2} - H1_{i-1/2} lives back at integer i
    Ha = sb.emit("slice", (H1,), hint="Ha", ranges={"x": (1, N - 1)})
    Hb = sb.emit("slice", (sb.emit("shift", (H1,), hint="Hs", deltas={"x": 1}),), hint="Hb", ranges={"x": (1, N - 1)})
    dH = sb.pw("sub", rechart(Ha, e_chart), rechart(Hb, e_chart), hint="dH")
    pdH = sb.emit("pad", (dH,), hint="pdH", fill=0.0, extents={"x": (0, N)})
    E1 = sb.pw("add", E, sb.pw("mul", cst(c, (0, N), e_chart), pdH), hint="E1")

    ef = b.emit(
        "fold",
        ("E0", "H0"),
        hint="Ef",
        step=sb.program(),
        dim="tm",
        state=("E", "H"),
        element=(),
        carry={"E": E1, "H": H1},
        out=("final", E1),
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
