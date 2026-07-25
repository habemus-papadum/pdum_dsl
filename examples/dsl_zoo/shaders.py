"""The shader zoo: one @compute kernel, four f's — in TODAY'S working syntax.

An f is a VALUE-LANGUAGE kernel (@jit: branches, loops, intrinsics),
applied per element through the spelled oracle. Three findings this zoo
surfaced, left in place for markup:

FINDING 1 — the pipe cannot carry 2-D coordinates today. `|` threads
exactly one value and the entry refuses two ("pipe threads exactly one
value; got 2 arg types"), and composite (tuple) ARGUMENTS are not
marshalable at the dsl entry ("pass fields as separate arguments"). So
    f = circle(r=8) | scale(2.0)          # the sketched form REFUSES today
Decision doors: pipe entry arity, composite-arg marshaling, or a
different composition spelling for coordinate functions.

FINDING 2 — composition WORKS today as capture-and-call: a handle that
captures another handle inlines it (guards recurse). `scaled(circle(...),
2.0)` below is real, running composition — just not the `|` spelling.

FINDING 3 — texture as a pipeable f does not exist: value-language
kernels cannot read tensors. General sampling is `take` (P9); an
integer-factor scale works NOW as a layout view sampled at the thread
coordinates (`sample_texture`).
"""

import numpy as np
from pdum.dsl import jit
from pdum.tl import Tensor, compute, thread_idx
from viewing import pgm, show


@compute
def shader(f, img):
    y, x = thread_idx("y", "x")
    img[y, x] = f(y, x)


# --- the f's: black shapes on white background ------------------------------


def circle(cy, cx, r):
    @jit()
    def go(y, x):
        d = sqrt((y - cy) * (y - cy) + (x - cx) * (x - cx))  # noqa: F821
        return 1.0 if d > r else 0.0

    return go


def square(cy, cx, r):
    @jit()
    def go(y, x):
        inside = abs(y - cy) < r and abs(x - cx) < r  # noqa: F821
        return 0.0 if inside else 1.0

    return go


def scaled(f, k, cy, cx):
    """Composition TODAY: capture the inner handle, transform the
    coordinates, call it — inlined at compile, guards recurse (FINDING 2).
    The `|` spelling of this is FINDING 1's refusal."""

    @jit()
    def go(y, x):
        return f(cy + (y - cy) / k, cx + (x - cx) / k)

    return go


# --- texture: the nearest WORKING spelling (FINDING 3) ----------------------


@compute
def sample_texture(tex, img):
    y, x = thread_idx("y", "x")
    img[y, x] = tex[y, x]


if __name__ == "__main__":
    img = Tensor.from_numpy(np.zeros((24, 40)), ("y", "x"))

    shader(circle(12.0, 20.0, 8.0), img)
    show(img, "-- circle (black on white):")

    shader(square(12.0, 20.0, 6.0), img)
    show(img, "-- square:")

    shader(scaled(circle(12.0, 20.0, 8.0), 0.5, 12.0, 20.0), img)
    show(img, "-- circle through scale 0.5 (half radius on screen):")

    # texture: draw a circle, then resample it through a decimate view
    small = Tensor.from_numpy(np.zeros((24, 40)), ("y", "x"))
    shader(circle(12.0, 20.0, 9.0), small)
    out = Tensor.from_numpy(np.zeros((12, 20)), ("y", "x"))
    sample_texture(small.decimate("y", 2).decimate("x", 2), out)
    show(out, "-- texture via a decimate view (integer scale only):")

    pgm(img, "/tmp/claude-501/zoo_scaled_circle.pgm")
    print("wrote /tmp/claude-501/zoo_scaled_circle.pgm")


########### Notes
# (owner markup preserved verbatim below; commented so the file runs —
#  the living version of this sketch is combinators.py)
# My mistake around the piping syntax. This is an instance where we actually need
# the execution operator (>>) that was in the original piping reference library
# that we used (pdum_plumbum).

# Let me explain by working backwards.

# scale (and rotate and translate) are factories that create shader combinators.
# That is they take in data, and then return a function that takes in a shader and returns a shader.  Ignore texture for now -- i think that was a mistake

# def scale(s):
#     def comb(f):
#         def go(y, x):
#             yprime = y*s
#             xprime = x*s
#             return f(yprime, xprime)
#         return go
#     return comb


# def translate(dy, dx):
#     def comb(f):
#         def go(y, x):
#             return f(y - dy, x - dx)

#         return go

#     return comb


# f = circle(12.0, 20.0, 8.0) >> scale(0.5) | translate(2.0, 3.0)

##or

# op = scale(0.5) | translate(2.0, 3.0)

# f = circle(12.0, 20.0, 8.0) >> op

# The next question, has to do with how tapping would work in this environment.
# For instance, if I wanted to tap the value of y prime in the scale function

# The idea is that TAPs wouldn't create output values. You wouldn't have the function return a value when you use TAPs. Instead, you would pass the name and the tensor to write the TAP into. kernel[taps={'yprime': yprime_tensor}, block, thread].

# Taps would have to be isBits values that would create a tensor with the layout of the block and block the grid, basically not the blocks and the threads all together.

# And in our examples right now, we don't have any notion of creating grid configurations. When we invoke, we kind of directly call without creating the grid parameters, but in this case we should try to do that.

# And then I still think there's a question of, when we integrate this in the assemblage layer, how would we specify things like grid parameters?

# And ideally, this should be flexible enough to have things like shared memory configuration and barriers, synchronizations, and other kinds of things that let you write high-performance compute kernels, whether they're going to be in WebGPU or CUDA.
