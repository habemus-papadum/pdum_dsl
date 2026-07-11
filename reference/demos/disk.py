"""M0.3 demo — a disk whose center orbits, rendered in a glfw window.

The shader closure is rebuilt every frame with new ``cx, cy`` *values* (same
*types*), so the pipeline compiles exactly once and each frame only rewrites the
uniform buffer. Watch the printed ``compiles=1`` stay at 1 while ``frames`` climbs.

    uv run python reference/demos/disk.py              # interactive window
    uv run python reference/demos/disk.py --frames 120 # render N frames, print stats, exit
"""

from __future__ import annotations

import math
import sys

from pdum.dsl_reference import builtins, jit
from pdum.dsl_reference.webgpu import Context


def disk(cx, cy, radius):
    @jit(kind="fragment")
    def shader():
        x, y = builtins.FragCoord.xy
        dx = x - cx
        dy = y - cy
        d2 = dx * dx + dy * dy
        return (1.0, 0.5, 0.0) if d2 < radius * radius else (0.05, 0.05, 0.12)

    return shader


def main() -> None:
    ctx = Context()
    canvas, drawer = ctx.window_drawer(size=(640, 480), title="pdum.dsl — orbiting disk (compiles once)")
    t = [0.0]

    def frame() -> None:
        t[0] += 0.03
        w, h = drawer.target.size
        cx = w / 2 + (w * 0.28) * math.cos(t[0])
        cy = h / 2 + (h * 0.28) * math.sin(t[0] * 1.3)
        drawer.update(disk(cx, cy, min(w, h) * 0.12))
        drawer.show()
        if drawer.uniform_writes % 30 == 0:
            print(f"frames={drawer.uniform_writes:4d}  compiles={drawer.compile_count}")

    if "--frames" in sys.argv:
        n = int(sys.argv[sys.argv.index("--frames") + 1])
        canvas.request_draw(frame)
        for _ in range(n):
            canvas.force_draw()
        print(f"DONE frames={drawer.uniform_writes} compiles={drawer.compile_count}")
        return

    # Continuous: re-arm the draw each frame.
    def animate() -> None:
        frame()
        canvas.request_draw(animate)

    ctx.run(canvas, animate)


if __name__ == "__main__":
    main()
