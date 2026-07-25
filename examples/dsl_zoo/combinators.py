"""The combinator algebra from the markup — running TODAY, zero engine changes.

scale/translate are FACTORIES returning combinators (shader -> shader);
`|` composes combinators left-to-right; `>>` applies a shader to a
combinator chain (the pdum_plumbum execution operator). Both marked-up
spellings work:

    f = circle(...) >> scale(0.5) | translate(2.0, 3.0)
    op = scale(0.5) | translate(2.0, 3.0); f = circle(...) >> op

Each application builds a @jit handle capturing the inner shader —
capture-and-call inlining, guards recursing, type-keyed caching: fresh
combinator VALUES are warm hits, new chain SHAPES are new artifacts.
"""

import numpy as np
from pdum.dsl import jit
from pdum.tl import Tensor, compute, thread_idx
from viewing import show


class Comb:
    """A shader combinator: shader -> shader. `|` composes (left applies
    first), `shader >> comb` applies. ~10 lines; promote to the dsl tier
    if blessed."""

    def __init__(self, apply):
        self.apply = apply

    def __or__(self, other: "Comb") -> "Comb":
        return Comb(lambda f: other.apply(self.apply(f)))

    def __rrshift__(self, shader):  # shader >> comb
        return self.apply(shader)

    def __ror__(self, shader):  # (shader >> a) | b  — precedence-friendly
        return self.apply(shader)


def scale(s):
    def apply(f):
        @jit()
        def go(y, x):
            return f(y * s, x * s)

        return go

    return Comb(apply)


def translate(dy, dx):
    def apply(f):
        @jit()
        def go(y, x):
            return f(y - dy, x - dx)

        return go

    return Comb(apply)


def circle(cy, cx, r):
    @jit()
    def go(y, x):
        d = sqrt((y - cy) * (y - cy) + (x - cx) * (x - cx))  # noqa: F821
        return 1.0 if d > r else 0.0

    return go


@compute
def shader(f, img):
    y, x = thread_idx("y", "x")
    img[y, x] = f(y, x)


# --- fuzz, AT THE KERNEL LEVEL: works today ---------------------------------
# The combinator FORM of fuzz (a Comb reading a texture inside the f tier)
# is the open door — value-language kernels cannot read tensors. But the
# EFFECT runs now: the kernel displaces the coordinates by a precomputed
# noise texture and evaluates f at the displaced points (the fn-arg path
# evaluates f at ARBITRARY per-element coordinates, not just iotas).


@compute
def fuzzed(f, ny, nx, img):
    y, x = thread_idx("y", "x")
    img[y, x] = f(y + ny[y, x], x + nx[y, x])


if __name__ == "__main__":
    img = Tensor.from_numpy(np.zeros((24, 40)), ("y", "x"))

    f = circle(12.0, 20.0, 8.0) >> scale(0.5) | translate(2.0, 3.0)
    shader(f, img)
    show(img, "-- circle >> scale(0.5) | translate(2, 3)  (the one-liner):")

    op = scale(0.5) | translate(2.0, 3.0)
    f2 = circle(12.0, 20.0, 8.0) >> op
    img2 = Tensor.from_numpy(np.zeros((24, 40)), ("y", "x"))
    shader(f2, img2)
    assert np.array_equal(img.to_numpy(), img2.to_numpy())
    print("-- two-liner (op = ... ; f = circle >> op): identical image ✓")

    rng = np.random.default_rng(4)
    amp = 1.8
    ny = Tensor.from_numpy(amp * rng.standard_normal((24, 40)), ("y", "x"))
    nx = Tensor.from_numpy(amp * rng.standard_normal((24, 40)), ("y", "x"))
    fuzzed(circle(12.0, 20.0, 8.0), ny, nx, img)
    show(img, "-- fuzz via precomputed noise texture (kernel-level):")
