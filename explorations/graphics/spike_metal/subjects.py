"""The differential subjects -- lifted from the conformance battery.

Deliberately NOT new kernels: these are the same subjects
`conformance/test_conformance_kernels.py` already runs reference-vs-WGSL,
so the Metal column slots into an existing differential rather than
inventing its own notion of correct. Seven of the eight are that file's
kernels verbatim; `banded` is added because the battery has no
`where`/select subject at all (a coverage hole worth naming -- select is
the normal form the whole control-flow position rests on, and no
conformance kernel exercises it on device).

Each subject names what it is evidence FOR:

  ambient   pure `tl.iota` -- the ambient row, no buffer reads.
  copy      pointwise arithmetic over a loaded buffer.
  markers   maximum/tanh/sqrt -- the transcendental marker rows, where
            f32 agreement between two different vendors' math libraries
            is least guaranteed and therefore most interesting.
  gather    computed-index `tl.read` (tex[i32(i), i32(i)]) -- the row
            that emits an unguarded index expression.
  box3      compile-time-unrolled neighborhood: several reads at
            distinct computed offsets.
  uniform   a captured scalar riding an `abi.slot` -- the uniform-slot
            channel, one value.
  spliced   a jit fn-argument chain (`twill | zoom`) -- the fn-arg
            capture blocks, i.e. MULTIPLE slots at plan-assigned offsets.
  banded    a jit fn with a real Python `if`, which the splicer flattens
            to `tl.pointwise f="where"` -- the select row.
"""

from __future__ import annotations

import _paths  # noqa: F401  (sys.path side effect, must precede pdum imports)
import numpy as np
from pdum.dsl import jit, op
from pdum.tl import Tensor, compute, f32, global_idx, i32  # noqa: F401 -- bodies' globals
from pdum.tl.markers import maximum, sqrt, tanh  # noqa: F401 -- bare in kernel bodies


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


@compute
def ambient_field(img):
    i, j = global_idx("y", "x")
    img[i, j] = f32(i) * 0.25 + f32(j) * 0.5 + 1.5


@compute
def scaled_copy(src, dst):
    i, j = global_idx("y", "x")
    dst[i, j] = src[i, j] * 2.0 + 1.0


@compute
def spiky(img):
    i, j = global_idx("y", "x")
    v = maximum(f32(i), f32(j)) * 0.25 + (f32(i) - f32(j)) * 0.5
    img[i, j] = tanh(v) + sqrt(v * v + 1.0)


@compute
def gather_diag(tex, img):
    (i,) = global_idx("y")
    img[i] = tex[i32(i), i32(i)]


@compute
def box3(tex, img):
    (i,) = global_idx("y")
    acc = 0.0
    for di in range(3):
        acc = acc + tex[i32(i) + di]
    img[i] = acc / 3.0


_GAIN = 3.5


@compute
def gained(img):
    i, j = global_idx("y", "x")
    img[i, j] = (f32(i) + f32(j)) * _GAIN


@op
def twill(a, b):
    @jit()
    def go(x):
        return x * a + b

    return go


@op
def zoom(s):
    @jit()
    def go(x):
        return x * s

    return go


@compute
def shader(f, img):
    i, j = global_idx("y", "x")
    img[i, j] = f(f32(i) + f32(j))


@compute
def tanh_ramp(img):
    """A ramp straight through Metal's tanh overflow cliff: the argument
    reaches 0.75*(H-1), which crosses 44.36 at H = 60."""
    i, j = global_idx("y", "x")
    img[i, j] = tanh(f32(i) * 0.75 + f32(j) * 0.0)


def band(lo, hi):
    """A real Python `if` in a jit device fn: the splicer flattens it to
    `tl.pointwise f="where"`, which is the select row on both targets."""

    @jit()
    def go(v):
        if v > lo:
            y = v * hi
        else:
            y = v * 0.5 - 1.0
        return y

    return go


def _rng(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape)


# name -> (kernel, make_args). make_args returns FRESH tensors every call so
# each executor sees identical inputs (the battery's discipline).
SUBJECTS = {
    "ambient": (ambient_field, lambda: (T(np.zeros((5, 7)), ("y", "x")),)),
    "copy": (
        scaled_copy,
        lambda: (T(_rng((4, 6), 7), ("y", "x")), T(np.zeros((4, 6)), ("y", "x"))),
    ),
    "markers": (spiky, lambda: (T(np.zeros((5, 7)), ("y", "x")),)),
    "gather": (
        gather_diag,
        lambda: (T(np.arange(9.0).reshape(3, 3), ("y", "x")), T(np.zeros(3), ("y",))),
    ),
    "box3": (box3, lambda: (T(np.arange(5.0), ("y",)), T(np.zeros(3), ("y",)))),
    "uniform": (gained, lambda: (T(np.zeros((3, 4)), ("y", "x")),)),
    "spliced": (
        shader,
        lambda: (twill(4.0, 3.0) | zoom(0.5), T(np.zeros((3, 4)), ("y", "x"))),
    ),
    "banded": (shader, lambda: (band(-0.2, 5.0), T(np.zeros((9, 11)), ("y", "x")))),
    # Deliberately NOT multiples of the 8x8 threadgroup, and big enough that
    # the overhang is many whole threadgroups: this is what actually exercises
    # the bounds guard (and its omission under dispatchThreads:). 255x129 has
    # a ragged edge on both axes; 301x203 does too, over the marker rows.
    "copy_big": (
        scaled_copy,
        lambda: (T(_rng((255, 129), 11), ("y", "x")), T(np.zeros((255, 129)), ("y", "x"))),
    ),
    # 0.75*(H-1) is spiky's largest argument to tanh, and Metal's tanh returns
    # NaN past 44.36 (see tanh_wide below), so H is capped at 60 here. Both
    # extents stay ragged mod 8 (59 = 7*8+3, 177 = 22*8+1).
    "markers_big": (spiky, lambda: (T(np.zeros((59, 177)), ("y", "x")),)),
    # THE MATH-LIBRARY FINDING, as a subject. tanh saturates to 1.0 in exact
    # arithmetic, and the f64 reference says so -- but Metal computes it from
    # exp(2x), which overflows f32 at x = log(FLT_MAX)/2 = 44.36, so the device
    # returns NaN where the reference returns 1.0. Both device columns fail
    # IDENTICALLY (wgpu also lowers to Metal here), and they fail as a REFUSAL:
    # Tensor.from_numpy rejects the nan bit pattern at decode (200 section 4).
    # Invisible in the conformance battery today only because its `spiky` runs
    # at 5x7, where the argument never exceeds ~2.
    "tanh_wide": (tanh_ramp, lambda: (T(np.zeros((96, 3)), ("y", "x")),)),
}


def artifact(kernel, args):
    """The compiled artifact WITHOUT launching it. `_compile` is the
    private door; spike_runner's H2 is that no public one exists."""
    from pdum.tl.kernel import _compile

    return _compile(kernel.fn, args)
