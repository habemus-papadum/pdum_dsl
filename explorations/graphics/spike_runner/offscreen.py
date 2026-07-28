"""Headless: build once, render a sequence, verify frame 0, time the loop.

    python -m offscreen            (from this directory, with the packages on PYTHONPATH)

Does four things:
  1. builds the frame runner once;
  2. renders N frames offscreen through the SHARED encode path, varying
     angle and ripple phase, and writes them as PNGs;
  3. compares frame 0 against ``graphics.render``'s numpy reference at the
     conformance cylinder golden's tolerance (atol=2e-3);
  4. reports the per-frame cost breakdown and the copy ledger, against one
     cold ``render_wgpu`` call for scale.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "conformance"))

import scene  # noqa: E402
from copies import counting  # noqa: E402
from device import PassTimer, make_device  # noqa: E402
from pngw import write_gray  # noqa: E402
from residency import Residency  # noqa: E402
from runner import CompiledCompute, CompiledRender, FrameRunner, artifact_of  # noqa: E402

ATOL = 2e-3  # conformance/test_conformance_render.py::test_the_cylinder_renders_on_the_device


def cold_frame_today(mesh, angle, phase):
    """One frame the way the machinery does it now, entirely on the device:
    the conformance compute executor for the ripple, then ``render_wgpu``.

    Both are one-shot. The compute executor uploads every tensor argument
    and reads every writable back to the host (wgsl_executor.py:258-290);
    ``render_wgpu`` then re-lowers both shader stages, regenerates WGSL,
    creates a shader module and a render pipeline, re-uploads the geometry
    it just read back, renders, and maps the result (wgsl_executor.py:
    391-616). Nothing is retained between calls."""
    from wgsl_executor import render_wgpu, wgpu_artifact

    scene.set_phase(phase)
    _mesh, rippled = mesh, scene.make_buffers()[1]
    args = (mesh.field("theta"), mesh.field("h"), rippled.field("theta"), rippled.field("h"))
    art = artifact_of(scene.ripple, *args)
    wgpu_artifact(art).launch(args)
    pso, g = scene.make_pso(angle)
    return render_wgpu(pso, rippled, g, shape=scene.SIZE)


class Offscreen:
    """The r32float attachment + its readback staging buffer, created ONCE.

    ``render_wgpu`` creates both per call (wgsl_executor.py:560, :610)."""

    def __init__(self, device, size):
        import wgpu

        self.device = device
        self.H, self.W = size
        self.tex = device.create_texture(
            size=(self.W, self.H, 1),
            format=wgpu.TextureFormat.r32float,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.view = self.tex.create_view()
        self.bpr = (self.W * 4 + 255) // 256 * 256
        self.staging = device.create_buffer(
            size=self.bpr * self.H, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        )

    def encode_readback(self, enc) -> None:
        enc.copy_texture_to_buffer(
            {"texture": self.tex},
            {"buffer": self.staging, "bytes_per_row": self.bpr, "rows_per_image": self.H},
            (self.W, self.H, 1),
        )

    def read(self) -> np.ndarray:
        raw = np.frombuffer(self.device.queue.read_buffer(self.staging), dtype=np.float32)
        img = raw.reshape(self.H, self.bpr // 4)[:, : self.W]
        return np.flipud(img).astype(np.float64)  # WebGPU y-up -> the reference row convention


def build(size=scene.SIZE):
    device, has_ts = make_device()
    res = Residency(device)
    mesh, rippled = scene.make_buffers()
    pso, g = scene.make_pso(scene.angle_of(0))
    scene.set_phase(scene.phase_of(0))

    t0 = time.perf_counter()
    comp = CompiledCompute(
        device,
        res,
        scene.ripple,
        (mesh.field("theta"), mesh.field("h"), rippled.field("theta"), rippled.field("h")),
        labels=["mesh.theta", "mesh.h", "rippled.theta", "rippled.h"],
    )
    rend = CompiledRender(device, res, pso, (rippled,), (g,), target_format="r32float")
    build_s = time.perf_counter() - t0
    return device, res, mesh, rippled, FrameRunner(device, res, comp, rend), build_s, has_ts


def frame(runner, off, i, timer=None):
    """One frame, host side. The ONLY device traffic before encode is the
    slot-channel write in ``update``."""
    scene.set_phase(scene.phase_of(i))
    t_a = time.perf_counter()
    runner.update(vs_fn=scene.vs_fn_for(scene.angle_of(i)))
    t_b = time.perf_counter()
    enc = runner.device.create_command_encoder()
    runner.encode(enc, off.view, timer=timer)
    off.encode_readback(enc)
    if timer:
        timer.resolve_into(enc)
    cmd = enc.finish()
    t_c = time.perf_counter()
    runner.device.queue.submit([cmd])
    t_d = time.perf_counter()
    img = off.read()
    t_e = time.perf_counter()
    return img, (t_b - t_a, t_c - t_b, t_d - t_c, t_e - t_d)


def main(n_frames=8, n_timed=200):
    device, res, mesh, rippled, runner, build_s, has_ts = build()
    off = Offscreen(device, scene.SIZE)
    outdir = HERE / "frames"
    outdir.mkdir(exist_ok=True)

    print(f"device: timestamps={has_ts}  build (lower+translate+pipelines+upload) = {build_s * 1e3:.1f} ms")
    print(f"resident buffers after build: {res.copy_ledger()}")
    print(f"compute WGSL:\n{runner.compute.prog.source}\n")

    # --- 2. the sequence -----------------------------------------------------
    frames = []
    for i in range(n_frames):
        img, _ = frame(runner, off, i)
        frames.append(img)
        write_gray(outdir / f"frame{i:02d}.png", np.flipud(img))  # PNG row 0 = top
    print(f"wrote {n_frames} PNGs to {outdir}")

    # The SAME compiled pipeline into a bigger attachment: no recompile, no
    # re-lowering. Resolution never entered the pipeline -- though
    # _lower_fragment still DEMANDS a lattice-shaped varying_types dict it
    # does not use (see FINDINGS.md), which is why render_wgpu takes `shape`.
    big = Offscreen(device, (256, 384))
    frame(runner, big, 3)
    write_gray(outdir / "hero_256x384.png", np.flipud(big.read()))
    print(f"  + hero_256x384.png from the same pipeline (render pipeline id {id(runner.render.pipeline):#x})")

    # --- 3. the golden -------------------------------------------------------
    # Two comparisons. The one that validates the RUNNER is against
    # render_wgpu: same device, same f32, so any difference is the warm
    # factoring's fault and the bar is bit-exactness. The one against numpy
    # is the scene's f32-vs-f64 differential and carries the stated
    # tolerance -- the conformance cylinder golden's atol.
    print(f"\nEVERY frame: vs render_wgpu (bar: bit-exact) and vs numpy reference (atol {ATOL:.0e}):")
    ok = True
    for i, got in enumerate(frames):
        today = cold_frame_today(mesh, scene.angle_of(i), scene.phase_of(i))
        exact = np.array_equal(got, today)
        ok = ok and exact
        ref = scene.reference_frame(mesh, scene.angle_of(i), scene.phase_of(i))
        err, err_today = np.abs(got - ref), np.abs(today - ref)
        bad, bad_today = int((err > ATOL).sum()), int((err_today > ATOL).sum())
        cov = int((ref != 0).sum()), int((got != 0).sum())
        print(
            f"  frame {i}: vs render_wgpu {'BIT-EXACT' if exact else 'DIFFERS'}  |  "
            f"vs numpy max|err| {err.max():.3e} over-atol {bad}px "
            f"(render_wgpu itself: {bad_today}px)  coverage ref/dev {cov[0]}/{cov[1]}"
        )
    print(f"  runner verification (bit-exact against the existing device path): {'PASS' if ok else 'FAIL'}")
    distinct = len({f.tobytes() for f in frames})
    print(f"  distinct frames: {distinct}/{n_frames} (the loop is animating, not replaying one image)")
    print(
        f"  ONE pipeline served all {n_frames} angles: render pipeline id "
        f"{id(runner.render.pipeline):#x}, compute pipeline id {id(runner.compute.pipeline):#x}"
    )

    # --- 4. the copy ledger --------------------------------------------------
    res.reset_ledger()
    with counting(device) as warm:
        frame(runner, off, 3)
    print("\ncopies per WARM frame (device instrumented):")
    print(f"  {warm.summary()}")
    print(f"  residency ledger (geometry uploads/downloads): {res.copy_ledger()}")
    print(f"  slot-channel bytes written per frame: {runner.uniform_bytes_last}")
    print(f"  breakdown: {warm.detail}")

    print("\ncopies per frame the way the machinery does it TODAY (same device, same scene):")
    with counting(device) as cold:
        cold_frame_today(mesh, scene.angle_of(3), scene.phase_of(3))
    print(f"  {cold.summary()}")
    print(f"  breakdown: {cold.detail}")

    # --- 5. timing -----------------------------------------------------------
    timer = PassTimer(device, 2) if has_ts else None
    for i in range(20):  # warmup
        frame(runner, off, i)
    rows = []
    gpu_compute, gpu_render = [], []
    for i in range(n_timed):
        _, parts = frame(runner, off, i % 64, timer=timer)
        rows.append(parts)
        if timer:
            a, b = timer.read_ms()
            gpu_compute.append(a)
            gpu_render.append(b)
    cols = list(zip(*rows))
    labels = ["update (staging+write_buffer)", "encode (2 passes + copy + finish)", "submit", "readback (sync map)"]
    print(f"\nwarm frame breakdown, {n_timed} samples, {scene.SIZE[0]}x{scene.SIZE[1]} r32float:")
    print(f"  {'phase':38s} {'min ms':>9s} {'median ms':>10s}")
    for lab, c in zip(labels, cols):
        print(f"  {lab:38s} {min(c) * 1e3:9.3f} {statistics.median(c) * 1e3:10.3f}")
    tot = [sum(r) for r in rows]
    print(f"  {'TOTAL (incl. readback)':38s} {min(tot) * 1e3:9.3f} {statistics.median(tot) * 1e3:10.3f}")
    no_rb = [sum(r[:3]) for r in rows]
    med_no_rb = statistics.median(no_rb) * 1e3
    print(f"  {'TOTAL (no readback = the window path)':38s} {min(no_rb) * 1e3:9.3f} {med_no_rb:10.3f}")
    if timer:
        med_gc = statistics.median(gpu_compute)
        print(f"  {'GPU compute pass (timestamps)':38s} {min(gpu_compute) * 1:9.3f} {med_gc:10.3f}")
        med_gr = statistics.median(gpu_render)
        print(f"  {'GPU render pass (timestamps)':38s} {min(gpu_render) * 1:9.3f} {med_gr:10.3f}")

    # --- 6. the cold call for scale -----------------------------------------
    from pdum.tl.zoo.cylinder import ripple as ref_ripple
    from wgsl_executor import render_wgpu

    scene.set_phase(scene.phase_of(0))
    pso0, g0 = scene.make_pso(scene.angle_of(0))
    host_rippled = scene.make_buffers()[1]
    colds, colds_full = [], []
    for _ in range(12):
        t = time.perf_counter()
        ref_ripple(mesh.field("theta"), mesh.field("h"), host_rippled.field("theta"), host_rippled.field("h"))
        t_mid = time.perf_counter()
        render_wgpu(pso0, host_rippled, g0, shape=scene.SIZE)
        t_end = time.perf_counter()
        colds.append(t_end - t_mid)
        colds_full.append(t_end - t)
    t_ref = []
    for _ in range(3):
        t = time.perf_counter()
        scene.reference_frame(mesh, scene.angle_of(0), scene.phase_of(0))
        t_ref.append(time.perf_counter() - t)
    print("\nfor scale (min of 12 / min of 3):")
    print(f"  cold render_wgpu (re-lower, re-translate, new pipeline, re-upload): {min(colds) * 1e3:8.2f} ms")
    print(f"  cold numpy ripple + cold render_wgpu (the whole frame, today):      {min(colds_full) * 1e3:8.2f} ms")
    print(f"  numpy reference render (graphics.render):                          {min(t_ref) * 1e3:8.2f} ms")
    print(f"  warm loop total (min, incl. readback):                              {min(tot) * 1e3:8.2f} ms")
    print(f"  speedup, warm vs whole cold frame:                                  {min(colds_full) / min(tot):8.1f}x")
    return 0 if ok else 1


if __name__ == "__main__":
    for a in sys.argv[1:]:
        if a.startswith("--size="):
            h, w = a.split("=", 1)[1].split("x")
            scene.SIZE = (int(h), int(w))
    raise SystemExit(main())
