"""The headless checks, and how to read their numbers.

Pulled out of ``mouse_ripple.py`` so the demo reads as one story. Every
function here takes DATA (frames, sources, a callable that produces a
golden) and prints; none of them knows what the scene is.

Three comparisons, with three different bars, and the bars are not
interchangeable:

  vs THE NUMPY REFERENCE -- a tolerance, never equality. The reference
  interior is f64 and both devices are f32, so the two are answering
  slightly different questions. The conformance cylinder golden's
  atol=2e-3 is calibrated to ONE angle; a frame loop sweeping angles
  will find pixels outside it, and finding them is not a regression.
  ``local_slope`` exists to tell the two cases apart: if the reference's
  own step to its neighbour is the same size as the disagreement, the
  pixel is on an antialiasing ramp and one f32 ulp of interpolated ``u``
  moved it. If the disagreement were large where the image is FLAT, that
  would be a bug.

  vs render_wgpu -- bit-exactness, because it is the same device, the
  same f32, and the same program. Any difference at all would be this
  demo's warm factoring being wrong. (It only earns that bar if the
  ripple runs on the device on both sides; see the note in
  ``mouse_ripple.render_wgpu_frame``.)

  webgpu vs metal -- report whatever it is. Compute was bitwise-equal in
  spike_metal, but rasterization involves fixed-function coverage and
  interpolation hardware reached through two different drivers, so there
  was no reason to expect equality and the honest thing is to measure.
  Read the caveat printed alongside the number: wgpu lowers WGSL through
  Naga to MSL on this machine, so the two columns share a GPU, a
  rasterizer and a math library. Equality here is evidence about our two
  translators, not about portability across vendors.
"""

from __future__ import annotations

import numpy as np


def local_slope(img: np.ndarray) -> np.ndarray:
    """The largest absolute step to a 4-neighbour, per pixel -- how steep
    the reference image itself is where the device disagrees with it."""
    p = np.pad(img, 1, mode="edge")
    return np.maximum.reduce(
        [
            np.abs(img - p[:-2, 1:-1]),
            np.abs(img - p[2:, 1:-1]),
            np.abs(img - p[1:-1, :-2]),
            np.abs(img - p[1:-1, 2:]),
        ]
    )


def against_reference(results: dict, golden, n_frames: int, atol: float) -> None:
    """``golden(i)`` -> the numpy reference image for frame i."""
    print(f"\nvs the numpy reference render (atol {atol:.0e}), all {n_frames} frames:")
    goldens = [golden(i) for i in range(n_frames)]
    for name, frames in results.items():
        worst_px, worst_err, tot_over, slopes = 0, 0.0, 0, []
        for i, got in enumerate(frames):
            ref = goldens[i]
            err = np.abs(got - ref)
            over = err > atol
            tot_over += int(over.sum())
            worst_px = max(worst_px, int(over.sum()))
            worst_err = max(worst_err, float(err.max()))
            if over.any():
                slopes += list(local_slope(ref)[over])
            if int((ref != 0).sum()) != int((got != 0).sum()):
                print(f"  {name} frame {i}: COVERAGE DIFFERS ref {(ref != 0).sum()} dev {(got != 0).sum()}")
        med = float(np.median(slopes)) if slopes else 0.0
        print(
            f"  {name:11s} worst max|err| {worst_err:.3e}   over-atol {tot_over} px total "
            f"(worst frame {worst_px}/{frames[0].size})   median reference slope there {med:.3f}"
        )


def against_one_shot(results: dict, ref0, today, atol: float) -> None:
    """``today`` is the frame the existing one-shot device path produced."""
    err = np.abs(today - ref0)
    first = next(iter(results.values()))[0]
    print(
        f"  {'render_wgpu':11s} (the one-shot path, for scale)   max|err| {float(err.max()):.3e}   "
        f"over-atol {int((err > atol).sum())} px   |render_wgpu - this demo| = "
        f"{float(np.abs(today - first).max()):.3e}"
    )


def cross_backend(results: dict) -> None:
    if len(results) != 2:
        return
    a, b = results["webgpu"], results["metal"]
    diffs = [float(np.abs(x - y).max()) for x, y in zip(a, b)]
    exact = sum(1 for x, y in zip(a, b) if np.array_equal(x, y))
    print("\ncross-backend (identical mouse path, identical program, identical frames):")
    print(f"  per-frame max |webgpu - metal|: {['%.3e' % d for d in diffs]}")
    print(f"  bitwise-identical frames: {exact}/{len(diffs)}   worst: {max(diffs):.3e}")
    print("  caveat: wgpu lowers WGSL through Naga to MSL here, so both columns end up")
    print("  on one GPU with one rasterizer and one math library. A second VENDOR is the")
    print("  first honest portability test; this is a test of our two translators.")


def source_diff(engines: dict) -> None:
    if len(engines) != 2:
        return
    import rowdiff

    w, m = engines["webgpu"], engines["metal"]
    rowdiff.print_report(
        [
            rowdiff.report("compute", w.c_source, m.c_source),
            rowdiff.report("vertex+fragment", w.r_source, m.r_source),
        ]
    )
