"""The Part-2 differential: reference vs flat WGSL vs reconstructed WGSL.

Three executions of the SAME artifact over identical fresh inputs:
  ref    -- run_region, the numpy interpreter (the artifact's own executor)
  flat   -- wgsl_executor.wgpu_artifact (today's translation, unmodified)
  recon  -- ifrecon.recon_artifact (the pass under test)

flat and recon must agree with ref to f32 tolerance (the device computes
f32, the reference f64 -- 210's numeric policy), and recon must agree with
flat EXACTLY: both are f32 evaluations of the same expression DAG in the
same order, so any difference at all is a bug in the pass, not rounding.
"""

from __future__ import annotations

import sys

import _paths  # noqa: F401
import numpy as np
from ifrecon import recon_artifact, translate
from kernels import SUBJECTS, T, artifact

from wgsl_executor import _translate as flat_translate
from wgsl_executor import wgpu_artifact


def one(name, f, shape, show=False):
    art, _ = artifact(f, shape)
    img_ref, img_flat, img_recon = (T(np.zeros(shape)) for _ in range(3))

    art.launch((f, img_ref), {})  # the numpy reference executor
    wgpu_artifact(art).launch((f, img_flat), {})
    recon_artifact(art).launch((f, img_recon), {})

    ref, fl, rc = (t.to_numpy() for t in (img_ref, img_flat, img_recon))
    src, meta = translate(art)
    fsrc, _ = flat_translate(art)
    st = meta["stats"]

    np.testing.assert_allclose(fl, ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(rc, ref, rtol=1e-5, atol=1e-6)
    exact = np.array_equal(rc, fl)
    print(
        f"{name:<8} nodes {st['nodes']:>3}  selects {st['selects']}  "
        f"sunk {st['sunk']:>3} ({100 * st['sunk'] / st['nodes']:4.1f}%)  "
        f"scopes {st['scopes']}  depth {st['max_depth']}  |  "
        f"vs ref max|d| flat {np.abs(fl - ref).max():.2e} recon {np.abs(rc - ref).max():.2e}"
        f"  |  recon==flat bitwise: {exact}"
    )
    if not exact:
        print(f"   !! recon differs from flat: max |d| {np.abs(rc - fl).max():.3e}")
    if show:
        print(f"\n----- {name}: FLAT ({len(fsrc.splitlines())} lines) -----\n{fsrc}")
        print(f"\n----- {name}: RECONSTRUCTED ({len(src.splitlines())} lines) -----\n{src}")
    return exact


if __name__ == "__main__":
    show = "--show" in sys.argv
    ok = True
    for name, (f, shape) in SUBJECTS.items():
        ok &= one(name, f, shape, show=show and name in ("band", "nested"))
    print("\nALL EXACT" if ok else "\nMISMATCH -- see above")
