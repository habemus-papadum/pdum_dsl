"""Timing at a given size, no numpy-reference verification (the reference
rasterizer is O(pixels) in Python and takes ~24 s/frame at 512x768)."""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offscreen
import scene
from device import PassTimer

H, W = (int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "512x768").split("x"))
scene.SIZE = (H, W)
device, res, mesh, rip, runner, build_s, has_ts = offscreen.build()
off = offscreen.Offscreen(device, (H, W))
timer = PassTimer(device, 2) if has_ts else None
for i in range(30):
    offscreen.frame(runner, off, i, timer=timer)
rows, gc, gr = [], [], []
for i in range(200):
    _, parts = offscreen.frame(runner, off, i % 64, timer=timer)
    rows.append(parts)
    if timer:
        a, b = timer.read_ms()
        gc.append(a)
        gr.append(b)
cols = list(zip(*rows))
labels = ["update (staging+write_buffer)", "encode (2 passes + copy + finish)", "submit", "readback (sync map)"]
print(f"warm frame breakdown, 200 samples, {H}x{W} r32float:")
print(f"  {'phase':38s} {'min ms':>9s} {'median ms':>10s}")
for lab, c in zip(labels, cols):
    print(f"  {lab:38s} {min(c)*1e3:9.3f} {statistics.median(c)*1e3:10.3f}")
tot = [sum(r) for r in rows]
norb = [sum(r[:3]) for r in rows]
print(f"  {'TOTAL (incl. readback)':38s} {min(tot)*1e3:9.3f} {statistics.median(tot)*1e3:10.3f}")
print(f"  {'TOTAL (no readback = window path)':38s} {min(norb)*1e3:9.3f} {statistics.median(norb)*1e3:10.3f}")
if timer:
    print(f"  {'GPU compute pass (timestamps)':38s} {min(gc):9.3f} {statistics.median(gc):10.3f}")
    print(f"  {'GPU render pass (timestamps)':38s} {min(gr):9.3f} {statistics.median(gr):10.3f}")
