"""The differential: numpy reference vs WGSL/wgpu vs MSL/Metal.

Four columns over identical fresh inputs, per the conformance battery's
discipline (`make_args()` per column, never a reused tensor):

  ref          the f64 numpy interpreter (`kernel(*args)` -- the normal call)
  wgsl         `wgpu_artifact(art).launch(...)`, the existing conformance path
  metal        `metal_artifact(art).launch(...)`, guard + dispatchThreadgroups:
  metal/exact  the same, guard OMITTED + dispatchThreads: (non-uniform
               threadgroups) -- the runtime/backend negotiation, checked

Two comparisons matter and they are different questions:

  * device vs reference is APPROXIMATE and expected to be: the reference
    computes f64, both devices compute f32 (210's policy on both sides).
    The battery's tolerance is rtol=1e-5/atol=1e-6.
  * device vs device is the interesting one. It has no a-priori reason to
    be exact -- two vendors' transcendental implementations may differ in
    the last ulp -- so whatever it comes out to is a measurement, and the
    report states which subjects are bit-exact and which are not.

`metal` vs `metal/exact` SHOULD be bitwise identical: same shader modulo
a dead guard, same arithmetic. If it ever isn't, the non-uniform
threadgroup dispatch is doing something we did not ask for.

THE CAVEAT ON BIT-EXACTNESS, which must travel with the result. On this
machine wgpu reports `backend_type: 'Metal'` -- wgpu-native lowers our
WGSL through Naga to MSL and hands it to the same Metal compiler and the
same M3 Ultra that our own MSL goes to. So the two device columns share
a math library and a GPU by construction. What the exactness therefore
PROVES is that two independently written translators emit arithmetically
equivalent programs -- a real and useful statement about our codegen.
What it does NOT prove is anything about cross-vendor numerics; a CUDA
or a Vulkan/Windows column would be the first honest test of that, and
`tanh_wide` below is the standing evidence that such a test would find
something.
"""

from __future__ import annotations

import _paths  # noqa: F401
import numpy as np
from metal_executor import metal_artifact
from msl_backend import Untranslatable as MSLUntranslatable
from subjects import SUBJECTS, artifact


def _outs(art, args):
    """Every writable parameter's array, in a stable order."""
    return [(n, args[art.params.index(n)].to_numpy()) for n in art.writable]


def _maxabs(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)))) if a.size else 0.0


def run(verbose: bool = False):
    from wgsl_executor import Untranslatable as WGSLUntranslatable
    from wgsl_executor import wgpu_artifact

    rows = []
    for name, (kernel, mk) in SUBJECTS.items():
        ref_args = mk()
        kernel(*ref_args)  # the reference launch
        art = artifact(kernel, mk())
        ref = _outs(art, ref_args)

        cols = {}
        for col, build in (
            ("wgsl", lambda a: wgpu_artifact(a)),
            ("metal", lambda a: metal_artifact(a, mode="copy")),
            ("metal/exact", lambda a: metal_artifact(a, mode="copy", exact_grid=True)),
            ("metal/adopt", lambda a: metal_artifact(a, mode="adopt")),
        ):
            args = mk()
            try:
                build(art).launch(args, {})
            except (WGSLUntranslatable, MSLUntranslatable) as exc:
                cols[col] = ("SKIP", str(exc))
                continue
            except ValueError as exc:
                # The device produced a bit pattern the boundary refuses --
                # today that means inf/nan (200 section 4). NOT a plumbing
                # failure: it is the device disagreeing with the reference in a
                # way the type system catches. Recorded, not swallowed.
                if "inf/nan" not in str(exc):
                    raise
                cols[col] = ("REFUSED", "nonfinite at decode")
                continue
            cols[col] = ("OK", _outs(art, args))

        row = {"subject": name, "writables": [n for n, _ in ref], "shape": ref[0][1].shape}
        for col, (status, payload) in cols.items():
            if status != "OK":
                row[col] = None
                row[f"{col}_note"] = f"{status}: {payload}"
                continue
            row[col] = max(_maxabs(g, r) for (_, g), (_, r) in zip(payload, ref))
        # device-vs-device: the question the reference cannot answer
        if cols["wgsl"][0] == "OK" and cols["metal"][0] == "OK":
            row["wgsl_vs_metal"] = max(
                _maxabs(m, w) for (_, m), (_, w) in zip(cols["metal"][1], cols["wgsl"][1])
            )
        if cols["metal"][0] == "OK" and cols["metal/exact"][0] == "OK":
            row["guard_vs_exact"] = max(
                _maxabs(m, e) for (_, m), (_, e) in zip(cols["metal"][1], cols["metal/exact"][1])
            )
        rows.append(row)
        if verbose:
            print(row)
    return rows


def _fmt(v):
    if v is None:
        return "--"
    return "0 (exact)" if v == 0.0 else f"{v:.3e}"


def main():
    rows = run()
    hdr = ("subject", "shape", "wgsl-ref", "metal-ref", "exact-ref", "adopt-ref", "wgsl~metal", "guard~exact")
    keys = ("subject", "shape", "wgsl", "metal", "metal/exact", "metal/adopt", "wgsl_vs_metal", "guard_vs_exact")
    widths = [max(len(h), 11) for h in hdr]
    print("  ".join(h.ljust(w) for h, w in zip(hdr, widths)))
    print("  ".join("-" * w for w in widths))
    worst_dd = 0.0
    for r in rows:
        cells = []
        for k, w in zip(keys, widths):
            v = r.get(k)
            s = str(v) if k in ("subject", "shape") else _fmt(v)
            cells.append(s.ljust(w))
        print("  ".join(cells))
        if r.get("wgsl_vs_metal") is not None:
            worst_dd = max(worst_dd, r["wgsl_vs_metal"])
    print()
    print(f"worst device-vs-device disagreement over all subjects: {_fmt(worst_dd)}")
    exact = [r["subject"] for r in rows if r.get("wgsl_vs_metal") == 0.0]
    approx = [(r["subject"], r["wgsl_vs_metal"]) for r in rows if r.get("wgsl_vs_metal")]
    print(f"bit-exact WGSL==MSL: {len(exact)}/{len(rows)} -> {exact}")
    if approx:
        print("differing:", ", ".join(f"{n} ({v:.2e})" for n, v in approx))
    bad = [r["subject"] for r in rows if r.get("guard_vs_exact")]
    print(f"guard vs dispatchThreads: {'ALL BITWISE EQUAL' if not bad else 'DIFFER: ' + str(bad)}")
    notes = [
        (r["subject"], c, r[f"{c}_note"])
        for r in rows
        for c in ("wgsl", "metal", "metal/exact", "metal/adopt")
        if r.get(f"{c}_note")
    ]
    if notes:
        print("\nnon-OK columns:")
        for s, c, n in notes:
            print(f"  {s}/{c}: {n}")
    return rows


if __name__ == "__main__":
    main()
