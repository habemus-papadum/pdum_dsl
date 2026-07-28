"""The windowed presenter — the same encode path, a different view.

    python window.py            (from this directory, packages on PYTHONPATH)
    ./run.sh window

Uses ``rendercanvas`` (present in the venv, 2.6.3, glfw backend available).
The ONLY things this file does that ``offscreen.py`` does not:

  * it asks the canvas for the current swap-chain texture instead of owning
    an r32float attachment, and
  * it never reads anything back.

``FrameRunner.encode`` is called identically by both. What is NOT shared is
the compiled pipeline: the swap chain's format is bgra8unorm-srgb and the
offscreen target is r32float, so each needs its own ``CompiledRender`` — the
fragment's color0 is a scalar and the surface expansion (one channel -> four)
is chosen at WGSL generation time. See FINDINGS.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scene  # noqa: E402
from device import make_device  # noqa: E402
from residency import Residency  # noqa: E402
from runner import CompiledCompute, CompiledRender, FrameRunner  # noqa: E402


def main(width=900, height=600):
    try:
        from rendercanvas.auto import RenderCanvas, loop
    except ImportError:
        print(
            "rendercanvas is not importable. Install it (it carries a glfw backend):\n"
            "    uv pip install rendercanvas glfw\n"
            "then re-run. The offscreen path (python offscreen.py) needs nothing extra."
        )
        return 1

    device, _ = make_device()
    canvas = RenderCanvas(size=(width, height), title="pdum spike_runner — cylinder (warm frame loop)")
    ctx = canvas.get_context("wgpu")
    fmt = ctx.get_preferred_format(device.adapter)
    ctx.configure(device=device, format=fmt)

    res = Residency(device)
    mesh, rippled = scene.make_buffers()
    pso, g = scene.make_pso(scene.angle_of(0))
    scene.set_phase(scene.phase_of(0))
    comp = CompiledCompute(
        device,
        res,
        scene.ripple,
        (mesh.field("theta"), mesh.field("h"), rippled.field("theta"), rippled.field("h")),
        labels=["mesh.theta", "mesh.h", "rippled.theta", "rippled.h"],
    )
    rend = CompiledRender(device, res, pso, (rippled,), (g,), target_format=fmt)
    runner = FrameRunner(device, res, comp, rend)
    print(f"surface format {fmt}; one compiled pipeline, {rend.prog.draw_count} vertices, no per-frame re-lowering")

    state = {"i": 0, "t0": time.perf_counter(), "n": 0}

    @canvas.request_draw
    def draw():
        i = state["i"]
        state["i"] += 1
        scene.set_phase(scene.phase_of(i))
        runner.update(vs_fn=scene.vs_fn_for(scene.angle_of(i)))

        enc = device.create_command_encoder()
        runner.encode(enc, ctx.get_current_texture().create_view(), clear=(0.02, 0.02, 0.03, 1.0))
        device.queue.submit([enc.finish()])  # the host owns the submit

        state["n"] += 1
        dt = time.perf_counter() - state["t0"]
        if dt >= 2.0:
            print(f"{state['n'] / dt:6.1f} fps  (frame {i})")
            state["t0"], state["n"] = time.perf_counter(), 0

    loop.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
