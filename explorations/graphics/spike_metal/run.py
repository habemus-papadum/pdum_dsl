"""spike_metal -- the Metal twin of the WGSL compute path. One entry point.

    /Users/nehal/src/pdum_dsl/.venv/bin/python -B run.py [section ...]

Sections (default: all, in this order):

    diff      three-way differential: numpy reference vs WGSL/wgpu vs MSL/Metal
              (+ the guard-vs-dispatchThreads bitwise check)
    rows      the mechanical WGSL-vs-MSL row diff and translator duplication
    math      the tanh / target-numeric-contract exhibit
    mem       unified memory: zero-copy proof, protocol benchmark, the deflation
    seam      the seam ledger and the user-visible program diff

WHAT THIS SPIKE IS. 280 proposes runtime = device/transfer/launch/timing,
backend = IR -> executable code. This builds the Metal twin of the
existing WGSL conformance path to test whether those definitions carve
the code at its joints, and to find out what the user-visible diff
between "run on WebGPU" and "run on Metal" actually is.

FILES

    msl_backend.py     IR -> MSL. Imports no Metal, touches no device.
    metal_runtime.py   device/queue/buffers/dispatch/sync/timing via PyObjC.
                       Imports no pdum.
    metal_executor.py  the glue: the twin of wgpu_artifact, ~40 lines,
                       every block tagged BACKEND / RUNTIME / SHARED.
    subjects.py        11 differential subjects, 8 of them the conformance
                       battery's own kernels verbatim.
    differential.py    the four-column differential.
    rowdiff.py         mechanical WGSL-vs-MSL comparison, two ways.
    mathrows.py        the tanh overflow exhibit and its one-row fix.
    unified.py         zero-copy proof + protocol benchmark + the deflation.
    seam_ledger.py     THE DELIVERABLE: the a/b/c classification, the five
                       places the definitions failed to carve, and the
                       program-diff sketch. Read this one first.

DEPENDENCY ADDED. `pyobjc-framework-Metal` (with pyobjc-core and
pyobjc-framework-cocoa) was installed into the repo venv by this spike.
Additive -- no existing package changed. Metal's RUNTIME shader compiler
(newLibraryWithSource:) means no Xcode and no offline `xcrun metal` is
needed, which matters: `xcrun metal` is in fact absent on this machine.
"""

from __future__ import annotations

import sys
import time

import _paths  # noqa: F401

SECTIONS = ("diff", "rows", "math", "mem", "seam")


def _banner(title: str):
    print("\n" + "#" * 78)
    print(f"# {title}")
    print("#" * 78 + "\n")


def main(argv):
    want = [a for a in argv[1:] if not a.startswith("-")] or list(SECTIONS)
    bad = [w for w in want if w not in SECTIONS]
    if bad:
        raise SystemExit(f"unknown section(s) {bad}; choose from {SECTIONS}")

    t0 = time.perf_counter()
    for name in want:
        if name == "diff":
            _banner("DIFFERENTIAL -- reference vs WGSL/wgpu vs MSL/Metal")
            import differential

            differential.main()
        elif name == "rows":
            _banner("ROW DIFF -- how much of WGSL and MSL is the same language")
            import rowdiff

            rowdiff.main()
        elif name == "math":
            _banner("TARGET NUMERIC CONTRACT -- the tanh row that is right and wrong")
            import mathrows

            mathrows.main()
        elif name == "mem":
            _banner("UNIFIED MEMORY -- zero-copy proof, protocol, and the deflation")
            import unified

            unified.main()
        elif name == "seam":
            _banner("SEAM LEDGER -- did runtime-vs-backend carve at the joints?")
            import seam_ledger

            seam_ledger.main()
    print(f"\n[run.py: {', '.join(want)} in {time.perf_counter() - t0:.1f}s]")


if __name__ == "__main__":
    main(sys.argv)
