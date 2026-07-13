"""Demo vocabulary: a toy ``Color`` record + 2D vector helpers.

This is a DEMO on purpose (090's stdlib-minimalism policy). Real color
modeling — spaces, RGB vs Lab vs OkLab, gamma — is a domain library's
ground, and the five surfaces make such a library an ordinary pip install
(an ``install(registry)`` plus an entry point). The stdlib must not squat
on that ground; it proves extension is cheap by staying small. Consuming
demo vocabulary is therefore one explicit import::

    from pdum.dsl.demo import graphics  # wires Color + helpers into DEFAULT

Everything here is portable by construction: the helpers are ``@overload``
DSL bodies (inlined per call site, zero per-target spellings) and ``Color``
is an ``@record`` dataclass (fields → uniform slots, methods inline).

The helpers are MODULE-LEVEL on purpose: cross-references (``length2`` →
``dot2``/``sqrt``, ``lerp2`` → ``mix``) are globals — invisible to capture,
resolved through the overload table at lower time — even when the referent
lives in another package, as ``mix`` (stdlib battery) does here.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..kernel.registry import DEFAULT
from ..stdlib.surfaces import overload, record


def dot2(a, b):
    return a[0] * b[0] + a[1] * b[1]


def length2(v):
    return sqrt(dot2(v, v))


def lerp2(a, b, t):
    return (mix(a[0], b[0], t), mix(a[1], b[1], t))


# GL derivative vocabulary (110 §3): one-line sugar over the analytic `D`.
# GLSL's dFdx is a finite difference across the pixel QUAD; ours is exact —
# and compute shaders have no quads, so analytic is the only derivative there.
def ddx(v):
    return D(v)[0]


def ddy(v):
    return D(v)[1]


def fwidth(v):
    d = D(v)
    return abs(d[0]) + abs(d[1])


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
    for fn in (dot2, length2, lerp2, ddx, ddy, fwidth):
        overload(registry, fn.__name__)(fn)
    record(registry, Color)


install(DEFAULT)
