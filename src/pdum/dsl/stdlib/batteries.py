"""The first batteries: math intrinsics + DSL-written helpers + Color.

Two economics, deliberately mixed (the numba 2:1 lesson, architecture risk
#4): ops that GPUs have natively are *intrinsics* (an op + a per-target
spelling — hand-spelled once per target), while everything composable is
*DSL-written* (an ``@overload`` body inlined at call sites — portable to
every target for free, including targets that don't exist yet). The step-10
exit gate counts the ratio.

The DSL-written batteries are MODULE-LEVEL on purpose: their bodies call
each other by bare name (``smoothstep`` uses ``clamp``), and at module level
those names are *globals* — invisible to capture (env stays empty), resolved
through the overload table at lower time. Defined inside a function they
would close over each other and trip the capture-free rule.

Nothing here touches the kernel; ``install(registry)`` is the whole API.
"""

from __future__ import annotations

from dataclasses import dataclass

from .surfaces import defop, intrinsic, overload, record, spell

_PY = "demo.simple_shader.python"
_WGSL = ("demo.simple_shader.wgsl.compute", "demo.simple_shader.wgsl.fragment")


def _unary_float(args, attrs, regions):
    from ..kernel.types import Scalar

    (t,) = args
    if not (isinstance(t, Scalar) and t.kind.startswith("f")):
        raise TypeError(f"expected a float operand, got {t!r}")
    return t


def _binary_same(args, attrs, regions):
    if args[0] != args[1]:
        raise TypeError(f"strict operands: {args[0]!r} vs {args[1]!r}")
    return args[0]


_OPS = {  # op -> (type_rule, python spelling, wgsl spelling)
    "math.sqrt": (_unary_float, "math.sqrt({0})", "sqrt({0})"),
    "math.exp": (_unary_float, "math.exp({0})", "exp({0})"),
    "math.sin": (_unary_float, "math.sin({0})", "sin({0})"),
    "math.cos": (_unary_float, "math.cos({0})", "cos({0})"),
    "math.floor": (_unary_float, "float(math.floor({0}))", "floor({0})"),
    "math.abs": (_unary_float, "abs({0})", "abs({0})"),
    "math.min": (_binary_same, "min({0}, {1})", "min({0}, {1})"),
    "math.max": (_binary_same, "max({0}, {1})", "max({0}, {1})"),
}


# --- DSL-written batteries: portable by construction ---------------------------
# (Cross-references are bare-name calls resolved via the overload table.)


def clamp(x, lo, hi):
    return min(max(x, lo), hi)


def mix(a, b, t):
    return a * (1.0 - t) + b * t


def step(edge, x):
    return 1.0 if x >= edge else 0.0


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fract(x):
    return x - floor(x)


def dot2(a, b):
    return a[0] * b[0] + a[1] * b[1]


def length2(v):
    return sqrt(dot2(v, v))


def lerp2(a, b, t):
    return (mix(a[0], b[0], t), mix(a[1], b[1], t))


_DSL_BATTERIES = (clamp, mix, step, smoothstep, fract, dot2, length2, lerp2)


@dataclass(frozen=True)
class Color:
    r: float
    g: float
    b: float

    def luminance(self):
        return 0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b

    def scaled(self, k):
        return (self.r * k, self.g * k, self.b * k)


def install(registry) -> None:
    for op, (rule, py, wgsl) in _OPS.items():
        defop(registry, op, rule)
        intrinsic(registry, op.split(".", 1)[1], op)  # `sqrt(x)` in DSL source -> the math.sqrt op
        if _PY in registry.backends:
            spell(registry, _PY, op, py)
        for name in _WGSL:
            if name in registry.backends:
                spell(registry, name, op, wgsl)
    for fn in _DSL_BATTERIES:
        overload(registry, fn.__name__)(fn)
    record(registry, Color)
