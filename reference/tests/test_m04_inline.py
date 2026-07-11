"""M0.4 — device functions + monomorphic inlining. Reproduces the demo.py shape:
a higher-order ``shader(img)`` where ``img`` calls ``weave`` and ``weave`` captures
an int uniform ``k``. Everything inlines into one WGSL function; the pipeline still
compiles once across changing ``k``."""

from pdum.dsl_reference import builtins, jit
from pdum.dsl_reference.backends.wgsl import compile_fragment
from pdum.dsl_reference.webgpu import Context


@jit(kind="fragment")
def shader(f):
    i, j = builtins.FragCoord.xy
    v = f(i, j)
    return (v / 800.0, 0.3, 0.6)


def make_img(k):
    @jit(kind="device")
    def weave(x, y):
        return x + y + k  # k captured -> uniform

    @jit(kind="device")
    def img(x, y):
        return weave(x, y) + 10  # nested device call

    return img


def test_device_fns_are_inlined():
    w = compile_fragment(shader(make_img(3))).wgsl
    assert "fn vs_main" in w and "fn fs_main" in w
    assert "fn weave" not in w and "fn img" not in w  # inlined away
    assert "_k" in w and "u." in w  # k flowed in as a uniform
    assert "+ f32(10)" in w  # img's nested arithmetic, inlined


def test_higher_order_one_compile_across_values():
    ctx = Context()
    drawer = ctx.offscreen_drawer(size=(64, 64))
    n = 6
    for k in range(n):
        drawer.update(shader(make_img(k)))  # k changes (int), types stable
        drawer.show()
    assert drawer.compile_count == 1  # inlined chain still compiles once
    assert drawer.uniform_writes == n


def test_renders_nonblack():
    ctx = Context()
    drawer = ctx.offscreen_drawer(size=(64, 64))
    drawer.update(shader(make_img(5)))
    drawer.show()
    px = drawer.target.read_pixels()
    assert any(px[i] > 0 for i in range(0, len(px), 4))  # some red from the gradient
    assert px[1] > 50  # green channel is a constant 0.3
