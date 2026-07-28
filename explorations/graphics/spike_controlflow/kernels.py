"""The Part-2 subjects: @compute kernels whose fn-argument is a @jit
device function containing REAL Python `if` statements.

Three shapes, chosen to stress the exclusive-use analysis rather than to
be pretty:

  band      one `if`, cheap arms -- the smoke test.
  fat       one `if`, EXPENSIVE arms (transcendental chains) with a
            deliberately SHARED subexpression used by both arms plus the
            condition. The shared node must stay hoisted; only the
            exclusive tails may sink.
  nested    an `if` inside an `if` inside the else-arm, so the sunk sets
            nest and the emitted scopes form a tree, not a list.

Every capture (lo/hi/k...) rides an abi slot, so the arms also contain
leaf nodes (abi.slot) that are exclusive to one arm -- exactly the case
where sinking wins on a real device: an unconditional uniform load in
the flat form becomes a conditional one.
"""

from __future__ import annotations

import _paths  # noqa: F401  (sys.path side effect, must precede pdum imports)
import numpy as np
from pdum.dsl import jit
from pdum.tl import Tensor, compute, f32, global_idx
from pdum.tl.markers import exp, sqrt, tanh


def T(a, names=("y", "x")):
    return Tensor.from_numpy(np.asarray(a, dtype=np.float64), names)


@compute
def shader(f, img):
    """The one kernel shape all three device functions launch under."""
    i, j = global_idx("y", "x")
    img[i, j] = f(f32(i) * 0.35 - 1.7 + f32(j) * 0.11)


def band(lo, hi):
    @jit()
    def go(v):
        if v > lo:
            y = v * hi
        else:
            y = 0.0
        return y

    return go


def fat(lo, ka, kb):
    @jit()
    def go(v):
        s = tanh(v * 0.5) + sqrt(v * v + 1.0)  # SHARED: cond and both arms read it
        if s > lo:
            y = exp(s * ka) * sqrt(s * s + ka) + tanh(s * ka - 1.0) * ka
        else:
            y = exp(-s * kb) * sqrt(s * s + kb) - tanh(s * kb + 1.0) * kb
        return y

    return go


def nested(lo, mid, ka, kb):
    @jit()
    def go(v):
        s = tanh(v * 0.5)
        if s > lo:
            y = exp(s * ka) + s
        else:
            if s > mid:
                y = sqrt(s * s + kb) * kb
            else:
                y = tanh(s * kb) - exp(s * 0.25)
        return y

    return go


SUBJECTS = {
    "band": (band(-0.2, 5.0), (5, 7)),
    "fat": (fat(1.3, 0.37, 0.61), (9, 11)),
    "nested": (nested(0.2, -0.4, 0.37, 0.61), (9, 11)),
}


def artifact(f, shape):
    """The compiled artifact, no launch needed (kernel._compile)."""
    from pdum.tl.kernel import _compile

    img = T(np.zeros(shape))
    return _compile(shader.fn, (f, img)), img


if __name__ == "__main__":
    from pdum.tl.dialect import walk_region

    for name, (f, shape) in SUBJECTS.items():
        art, _ = artifact(f, shape)
        ops = [n.op for n in walk_region(art.region)]
        pw = [dict(n.attrs)["f"] for n in walk_region(art.region) if n.op == "tl.pointwise"]
        print(f"\n--- {name} ---")
        print("  region ops :", ops)
        print("  pointwise f:", pw)
        print("  core.if    :", ops.count("core.if"), " where/select:",
              pw.count("where") + ops.count("core.select"))
        print("  arg_slots  :", [(p, b) for p, _, _, b, _, _ in art.arg_slots])
