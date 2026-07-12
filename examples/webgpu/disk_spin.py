"""A spinning disk, computed per frame by a WGSL kernel — the thesis live.

Every frame builds a FRESH closure over new capture values and calls it;
the specialization cache turns that into one compile + N uniform writes.
Run: `uv run python examples/webgpu/disk_spin.py` (exits politely without a
GPU). The interactive window version arrives with the graphics
`draw(target)` surface; this one animates in the terminal.
"""

import math
import sys
import time

from pdum.dsl import jit, no_compile
from pdum.dsl.demo.simple_shader import wgsl
from pdum.dsl.kernel.registry import DEFAULT

if not wgsl.is_available():
    print("no wgpu adapter here — nothing to demo (this is the polite exit)")
    sys.exit(0)


def make_disk(cx, cy, r):
    @jit(kind="simple_shader.compute")
    def disk(i, j):
        x = i / 24.0 - 1.0
        y = j / 24.0 - 1.0
        dx = x - cx
        dy = y - cy
        return 1.0 if dx * dx + dy * dy < r * r else 0.0

    return disk


W = H = 48
SHADES = " ░▒▓█"
make_disk(0.0, 0.0, 0.4)(out=(W, H))  # frame 0 pays the one compile
c0 = DEFAULT.specializations.compiles

with no_compile():  # every animation frame MUST be a cache hit
    for f in range(120):
        a = f / 120 * 2 * math.pi
        img = make_disk(0.5 * math.cos(a), 0.5 * math.sin(a), 0.35)(out=(W, H))
        frame = "\n".join(
            "".join(SHADES[int(img[j * W + i] * (len(SHADES) - 1))] for i in range(0, W, 1)) for j in range(0, H, 2)
        )
        sys.stdout.write("\x1b[H\x1b[2J" + frame + "\n")
        sys.stdout.flush()
        time.sleep(1 / 30)

print(f"120 frames, {DEFAULT.specializations.compiles - c0} new compiles — the thesis, animated")
