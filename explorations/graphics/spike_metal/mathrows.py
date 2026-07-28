"""The target NUMERIC CONTRACT -- a marker row that is right and wrong.

This exists because the differential turned up something neither the
backend nor the runtime definition has a place for, and it deserves to
be reproducible on its own rather than buried in a table cell.

`tanh` translates to the identical spelling in WGSL and MSL, and that
row is correct by every standard we have: same name, same signature,
same IEEE f32 discipline 210 imposes on both sides. It is nonetheless
WRONG on this device for large arguments, because Metal computes tanh
from exp(2x), which overflows f32 at

    x = log(FLT_MAX) / 2 = 88.7228 / 2 = 44.3614

and inf/inf is NaN. So the device returns NaN where the reference
returns 1.0 -- for a function that is bounded in [-1, 1] and whose exact
value at x = 44 is 1.0 to well beyond f32 precision.

Three things this script pins down, each of which matters to 280:

  1. The threshold is exactly log(FLT_MAX)/2, confirming the exp(2x)
     mechanism rather than a driver quirk.
  2. MTLMathMode/fastMathEnabled does NOT change it. This is not a
     fast-math tradeoff to be opted out of; it is the math library.
  3. A ONE-ROW backend fix is exact: tanh(clamp(x, -20, 20)). tanh(20)
     rounds to 1.0f, so clamping changes no representable result while
     removing the overflow entirely -- verified below against numpy.

Whose job is this? Not the backend's (the row is a faithful
translation), not the runtime's (it never touches arithmetic), not the
shared tier's (it is target-specific). It is a fourth category the
runtime/backend vocabulary has no word for: the TARGET NUMERIC
CONTRACT -- the set of per-target facts about where the math library
stops agreeing with the reference. 210 has the right section for it and
no mechanism. The conformance battery cannot find these on its own
either: its `spiky` subject runs at 5x7, where tanh's argument never
exceeds ~2, so this has been invisible.

The general worry, which this spike did NOT survey: tanh is unlikely to
be alone. exp, sinh/cosh, and pow are the usual company.
"""

from __future__ import annotations

import _paths  # noqa: F401
import Metal
import numpy as np
from metal_runtime import runtime

_PROBE = """
#include <metal_stdlib>
using namespace metal;
kernel void main0(const device float* x [[buffer(0)]],
                  device float* o [[buffer(1)]],
                  uint3 gid [[thread_position_in_grid]]) {
  o[gid.x] = %s;
}
"""


def _eval(expr: str, xs: np.ndarray, options=None) -> np.ndarray:
    rt = runtime()
    src = _PROBE % expr
    if options is None:
        pso = rt.pipeline(src)
    else:  # a distinct compile: bypass the source-keyed cache deliberately
        lib, err = rt.device.newLibraryWithSource_options_error_(src, options, None)
        if lib is None:
            raise RuntimeError(f"compile failed: {err}")
        pso, err = rt.device.newComputePipelineStateWithFunction_error_(
            lib.newFunctionWithName_("main0"), None
        )
    a = rt.buffer_copy(xs)
    b = rt.buffer_empty(xs.nbytes)
    rt.dispatch(pso, [a, b], (xs.size, 1, 1), (64, 1, 1))
    return rt.read(b, xs.size)


def main():
    rt = runtime()
    print(f"device: {rt.name}")
    fmax = np.finfo(np.float32).max
    predicted = float(np.log(fmax) / 2)
    print(f"predicted overflow threshold  log(FLT_MAX)/2 = {predicted:.4f}\n")

    xs = np.linspace(40.0, 50.0, 100001).astype(np.float32)
    got = _eval("tanh(x[gid.x])", xs)
    bad = ~np.isfinite(got)
    first = float(xs[bad][0]) if bad.any() else None
    last_ok = float(xs[~bad][-1]) if (~bad).any() else None
    print("1) DEFAULT compile options")
    print(f"   non-finite results          : {bad.sum()} / {xs.size}")
    print(f"   last finite x               : {last_ok:.4f}")
    print(f"   first non-finite x          : {first:.4f}")
    print(f"   matches log(FLT_MAX)/2      : {abs(first - predicted) < 2e-3}")
    print(f"   value there on the reference: {np.tanh(np.float64(first))!r}")

    print("\n2) explicit math mode")
    opts = Metal.MTLCompileOptions.alloc().init()
    try:
        opts.setMathMode_(Metal.MTLMathModeSafe)
        label = "MTLMathModeSafe"
    except Exception:
        opts.setFastMathEnabled_(False)
        label = "fastMathEnabled=False"
    got2 = _eval("tanh(x[gid.x])", xs, options=opts)
    bad2 = ~np.isfinite(got2)
    print(f"   {label}: non-finite {bad2.sum()} / {xs.size}  "
          f"-> {'UNCHANGED, not a fast-math tradeoff' if bad2.sum() == bad.sum() else 'CHANGED'}")

    print("\n3) the one-row fix: tanh(clamp(x, -20, 20))")
    wide = np.concatenate(
        [np.linspace(-200, 200, 50001), np.array([0.0, 1e-8, -1e-8, 19.9, 20.1, 44.36])]
    ).astype(np.float32)
    fixed = _eval("tanh(clamp(x[gid.x], -20.0f, 20.0f))", wide)
    ref = np.tanh(wide.astype(np.float64)).astype(np.float32)
    finite = np.isfinite(fixed)
    print(f"   non-finite results          : {(~finite).sum()} / {wide.size}")
    print(f"   max abs err vs numpy tanh   : {np.max(np.abs(fixed - ref)):.3e}")
    print(f"   tanh(20.0f) == 1.0f exactly : {float(_eval('tanh(x[gid.x])', np.array([20.0], np.float32))[0]) == 1.0}")

    # Does the clamp change any result it did not have to? Compare clamped vs
    # unclamped on the range where the unclamped form still works.
    inr = np.linspace(-20.0, 20.0, 50001).astype(np.float32)
    plain, clamped = _eval("tanh(x[gid.x])", inr), _eval("tanh(clamp(x[gid.x], -20.0f, 20.0f))", inr)
    print(f"   clamped == unclamped on |x|<=20 : BITWISE {np.array_equal(plain, clamped)}")
    base = np.max(np.abs(plain - np.tanh(inr.astype(np.float64)).astype(np.float32)))
    print(f"   (the {np.max(np.abs(fixed - ref)):.3e} above is Metal's baseline tanh ulp,")
    print(f"    not the clamp: unclamped tanh on |x|<=20 already differs by {base:.3e})")


if __name__ == "__main__":
    main()
