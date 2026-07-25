"""The batteries: the marker vocabulary's ops + DSL-written scalar helpers.

Two economics, deliberately mixed (the numba 2:1 lesson, architecture risk
#4): PRIMITIVES are markers (numpy-named, numpy-executed — the vocabulary in
``markers.py``) whose value-tier ops, type rules, and reference spellings
are DERIVED here from the declarations — one primitive, one registration
walk, no hand-kept parallel lists. Everything composable is *DSL-written*
(an ``@overload`` body inlined at call sites — portable to every target for
free, derivative for free through inlining). Batteries keep their DOMAIN
names (clamp/mix/smoothstep are GLSL heritage); only primitives take
numpy's names.

The two-door process for adding vocabulary (200 §S.2): can it be written
over existing primitives? Then it is a battery — no op, no spelling, no
table row. Must it be primitive? Then ONE marker declaration plus its
derivative row (or an explicit gradient-free row) — a float primitive
missing its derivative decision refuses to differentiate, never guesses.

The DSL-written batteries are MODULE-LEVEL on purpose: their bodies call
each other by bare name, resolved through the overload table at lower
time. The stdlib stays deliberately SMALL (090): scalar math only —
domain vocabulary (Bessel-class functions, a Color record) belongs to
ecosystem packages, an ordinary import away through the same surfaces.

Nothing here touches the kernel; ``install(registry)`` is the whole API.
"""

from __future__ import annotations

from .derivative import TABLE
from .markers import Marker, floor, maximum, minimum, pw, value_op
from .surfaces import defop, overload, spell

_REF = "reference"


def _unary_float(args, attrs, regions):
    from .types import Scalar

    (t,) = args
    if not (isinstance(t, Scalar) and t.kind.startswith("f")):
        raise TypeError(f"expected a float operand, got {t!r}")
    return t


def _binary_same(args, attrs, regions):
    if args[0] != args[1]:
        raise TypeError(f"strict operands: {args[0]!r} vs {args[1]!r}")
    return args[0]


_TYPE_RULES = {1: _unary_float, 2: _binary_same}
_PREDS = {"eq", "ne", "le", "lt", "ge", "gt"}  # comparisons ride core.cmp


def _pw_markers() -> list[Marker]:
    """The function primitives: every marker whose value-tier op is pw.*
    (operators and where stay core-owned; predicates ride core.cmp)."""
    out = []
    for m in vars(pw).values():
        if isinstance(m, Marker) and value_op(m).startswith("pw.") and m.name not in _PREDS:
            out.append(m)
    return out


# --- DSL-written batteries: portable by construction, domain-named -----------
# (Cross-references are bare-name calls resolved via the overload table.)


def clamp(x, lo, hi):
    # the marker globals make batteries REAL functions: host-callable on
    # scalars, dsl-lowered via the overload table, tl-inlined into kernels
    return minimum(maximum(x, lo), hi)


def mix(a, b, t):
    return a * (1.0 - t) + b * t


def step(edge, x):
    return 1.0 if x >= edge else 0.0


def smoothstep(e0, e1, x):
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fract(x):
    return x - floor(x)


_DSL_BATTERIES = (clamp, mix, step, smoothstep, fract)


def install(registry) -> None:
    for m in _pw_markers():
        op, arity = value_op(m), len(TABLE[m.name])
        defop(registry, op, _TYPE_RULES[arity])
        registry.overloads[m.name] = m  # the MARKER is the overload value
        if _REF in registry.backends:
            args = ", ".join("{%d}" % i for i in range(arity))
            spell(registry, _REF, op, f"np.{m.fn.__name__}({args})")
    registry.overloads["where"] = pw.where  # op = core.select (structure stays core)
    for fn in _DSL_BATTERIES:
        overload(registry, fn.__name__)(fn)
