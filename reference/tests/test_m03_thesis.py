"""M0.3 — the thesis, on real hardware. A render loop that changes capture *values*
every frame compiles the pipeline exactly once and only rewrites the uniform buffer.

Requires a working GPU (Metal/Vulkan/DX12); these are integration tests.
"""

from pdum.dsl import builtins, jit
from pdum.dsl.webgpu import Context


def make_disk(cx, cy, radius):
    @jit(kind="fragment")
    def shader():
        x, y = builtins.FragCoord.xy
        dx = x - cx
        dy = y - cy
        d2 = dx * dx + dy * dy
        return (1.0, 0.5, 0.0) if d2 < radius * radius else (0.0, 0.0, 0.0)

    return shader


def test_one_compile_many_uniform_writes():
    ctx = Context()
    drawer = ctx.offscreen_drawer(size=(64, 64), format="rgba8unorm")
    n = 8
    for k in range(n):
        drawer.update(make_disk(10.0 + k, 32.0, 20.0))  # moving center, type stays float
        drawer.show()
    assert drawer.compile_count == 1  # THE THESIS: build once...
    assert drawer.uniform_writes == n  # ...rewrite uniforms every frame
    assert drawer.cache.hit_count == n - 1


def test_shader_actually_renders():
    ctx = Context()
    drawer = ctx.offscreen_drawer(size=(64, 64), format="rgba8unorm")
    drawer.update(make_disk(32.0, 32.0, 20.0))  # centered disk
    drawer.show()
    px = drawer.target.read_pixels()  # tightly-packed RGBA, row 0 at top

    def pixel(col, row):
        i = (row * 64 + col) * 4
        return px[i], px[i + 1], px[i + 2], px[i + 3]

    cr, cg, cb, ca = pixel(32, 32)  # inside the disk -> orange
    assert cr > 200 and 110 <= cg <= 145 and cb < 40 and ca == 255
    kr, kg, kb, _ = pixel(0, 0)  # corner -> outside -> black
    assert kr < 10 and kg < 10 and kb < 10


def test_type_change_forces_recompile():
    ctx = Context()
    drawer = ctx.offscreen_drawer(size=(32, 32))
    drawer.update(make_disk(10.0, 10.0, 5.0))
    drawer.update(make_disk(10.0, 10.0, 5.0))  # same types -> hit
    assert drawer.compile_count == 1
    drawer.update(make_disk(10, 10, 5))  # int captures -> different FnType -> miss
    assert drawer.compile_count == 2
