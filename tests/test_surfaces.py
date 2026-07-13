"""Step 10 — the five surfaces: every extension lands through a registration,
and the extension-locality law (zero kernel diffs) is enforced by construction:
this file touches ONLY public registry surfaces."""

from dataclasses import dataclass

import pytest

import pdum.dsl  # noqa: F401  — batteries
from pdum.dsl.backends._emit import emit_dominated  # noqa: F401  (namespace pkg importable)
from pdum.dsl.demo.simple_shader.python import install as install_python
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.registry import Registry
from pdum.dsl.stdlib import install as install_lang
from pdum.dsl.stdlib.surfaces import Dialect, defop, intrinsic, overload, record, spell


def fresh() -> Registry:
    reg = Registry()
    install_python(reg)  # backends BEFORE the dialect: batteries spell onto them
    install_lang(reg)
    return reg


# --- surface A + D: defop + spell, end to end -----------------------------------


def test_defop_and_spell_add_an_op_with_zero_kernel_diffs():
    reg = fresh()
    defop(reg, "math.tanh", lambda args, attrs, regions: args[0])
    intrinsic(reg, "tanh", "math.tanh")
    spell(reg, "demo.simple_shader.python", "math.tanh", "math.tanh({0})")

    @jit()
    def f(x):
        return tanh(x) * 2.0  # noqa: F821 — resolved through the overload table

    import math

    assert abs(reg.dispatch(f, (0.5,)) - math.tanh(0.5) * 2.0) < 1e-12


def test_decomposition_gates_on_the_target_op_set():
    """A backend WITHOUT the native op gets the decomposition for free; one
    WITH it never pays (§2.10) — asserted via the rendered source."""
    from pdum.dsl.kernel.rewrite import Pat

    reg = fresh()
    defop(reg, "math.sinh", lambda args, attrs, regions: args[0])
    intrinsic(reg, "sinh", "math.sinh")
    # no spelling registered -> the decomposition must fire:
    reg.decompositions.append(
        (
            "math.sinh",
            (
                Pat("math.sinh"),
                lambda b, m: b.emit(
                    "core.div",
                    b.emit(
                        "core.sub",
                        b.emit("math.exp", m["root"].args[0]),
                        b.emit("math.exp", b.emit("core.neg", m["root"].args[0])),
                    ),
                    b.emit("core.const", type=m["root"].type, value=2.0),
                ),
            ),
        )
    )

    @jit()
    def f(x):
        return sinh(x)  # noqa: F821

    import math

    assert abs(reg.dispatch(f, (0.3,)) - math.sinh(0.3)) < 1e-12
    src = next(iter(reg.specializations._ready.values())).artifact.__pdum_source__
    assert "math.exp(" in src and "sinh" not in src  # decomposed away
    # now give the backend a native spelling: fresh registry, same kernel shape
    reg2 = fresh()
    defop(reg2, "math.sinh", lambda args, attrs, regions: args[0])
    intrinsic(reg2, "sinh", "math.sinh")
    spell(reg2, "demo.simple_shader.python", "math.sinh", "math.sinh({0})")
    reg2.decompositions.append(reg.decompositions[-1])

    @jit()
    def g(x):
        return sinh(x)  # noqa: F821

    reg2.dispatch(g, (0.3,))
    src2 = next(iter(reg2.specializations._ready.values())).artifact.__pdum_source__
    assert "math.sinh(" in src2  # native op survived: the gate skipped the decomposition


# --- surface B: DSL-written batteries ---------------------------------------------


def test_batteries_are_inlined_and_compose():
    @jit()
    def f(x):
        return smoothstep(0.0, 1.0, clamp(x, 0.0, 1.0))  # noqa: F821

    assert f(0.5) == 0.5
    assert f(-3.0) == 0.0 and f(9.0) == 1.0


def test_overload_must_be_capture_free():
    reg = fresh()
    k = 3.0

    def leaky(x):
        return x * k

    with pytest.raises(TypeError, match="capture-free"):
        overload(reg, "leaky")(leaky)


def test_tuple_batteries_run_on_both_backends():
    from pdum.dsl.demo import graphics  # noqa: F401  — length2 is demo vocabulary, explicitly imported
    from pdum.dsl.demo.simple_shader import wgsl

    @jit()
    def f(x):
        v = (x, x * 2.0)
        return length2(v)  # noqa: F821

    assert abs(f(3.0) - (9 + 36) ** 0.5) < 1e-12
    if wgsl.is_available():

        @jit(kind="simple_shader.compute")
        def g(i):
            v = (i, i * 2.0)
            return length2(v)  # noqa: F821

        out = g(out=4)
        assert abs(out[3] - (9 + 36) ** 0.5) < 1e-5  # tuples ELIMINATED before WGSL sees them


# --- surface C: records --------------------------------------------------------------


def test_record_roundtrip_fields_methods_and_thesis():
    from pdum.dsl.demo.graphics import Color
    from pdum.dsl.kernel.cache import no_compile

    def tint(c):
        @jit()
        def f(x):
            return c.luminance() + c.scaled(2.0)[1] * x

        return f

    c = Color(0.5, 0.25, 0.125)
    expected = 0.2126 * 0.5 + 0.7152 * 0.25 + 0.0722 * 0.125 + 0.5 * 1.0
    assert abs(tint(c)(1.0) - expected) < 1e-12
    tint(c)(1.0)
    with no_compile():  # new color VALUES: same Record type, cache hit
        tint(Color(0.9, 0.8, 0.7))(1.0)


def test_record_wants_a_dataclass():
    reg = fresh()

    class NotADataclass:
        pass

    with pytest.raises(TypeError, match="dataclass"):
        record(reg, NotADataclass)


# --- surface E: layering + dialects + entry points --------------------------------


def test_extend_layers_without_touching_the_parent():
    reg = fresh()
    child = reg.extend()
    defop(child, "math.cbrt", lambda args, attrs, regions: args[0])
    intrinsic(child, "cbrt", "math.cbrt")
    spell(child, "demo.simple_shader.python", "math.cbrt", "({0} ** (1.0/3.0))")

    @jit()
    def f(x):
        return cbrt(x)  # noqa: F821

    assert abs(child.dispatch(f, (8.0,)) - 2.0) < 1e-12
    assert "math.cbrt" not in reg.ops  # the parent never saw it

    @jit()
    def g(x):
        return cbrt(x)  # noqa: F821

    from pdum.dsl.kernel.lower import MissingRule

    with pytest.raises(MissingRule, match="cbrt"):
        reg.dispatch(g, (8.0,))


def test_dialect_is_bundling_sugar():
    reg = fresh()
    d = Dialect(
        "cbrt-pack",
        (
            lambda r: defop(r, "math.cbrt", lambda a, at, rg: a[0]),
            lambda r: intrinsic(r, "cbrt", "math.cbrt"),
            lambda r: spell(r, "demo.simple_shader.python", "math.cbrt", "({0} ** (1.0/3.0))"),
        ),
    )
    d.install(reg)

    @jit()
    def f(x):
        return cbrt(x)  # noqa: F821

    assert abs(reg.dispatch(f, (27.0,)) - 3.0) < 1e-9


def test_entry_points_loader_calls_installers():
    calls = []

    class FakeEP:
        name = "fake-backend"

        def load(self):
            return lambda registry: calls.append(registry)

    reg = Registry()
    assert reg.load_entry_points(entries=[FakeEP()]) == ["fake-backend"]
    assert calls == [reg]


# --- the extension-locality law -----------------------------------------------------


def test_extension_locality_zero_kernel_diffs():
    """The whole file is the test: a new op, a new battery, a new record, and
    a layered registry — all landed above with imports from stdlib/demo ONLY.
    This assertion pins it structurally: nothing in pdum.dsl.kernel exposes a
    mutable module-global registry a satellite could have needed to touch."""
    import pdum.dsl.kernel.api as api
    import pdum.dsl.kernel.registry as kreg

    assert kreg.DEFAULT.__class__ is Registry  # one explicit object, not module state
    assert not hasattr(api, "OVERLOADS") and not hasattr(kreg, "OVERLOADS")

    @dataclass(frozen=True)
    class Pixel:
        u: float
        v: float

        def manhattan(self):
            return abs(self.u) + abs(self.v)  # noqa: F821 — the abs BATTERY, not the builtin

    reg = fresh()
    record(reg, Pixel)

    def probe(p):
        @jit()
        def f(x):
            return p.manhattan() * x

        return f

    assert reg.dispatch(probe(Pixel(-1.5, 2.0)), (2.0,)) == 7.0


def test_extend_does_not_share_spelling_tables():
    """A child's spell() must never mutate the parent's backend (the layering
    promise) — Backend records share everything EXCEPT code_for_op."""
    reg = fresh()
    child = reg.extend()
    spell(child, "demo.simple_shader.python", "math.leaktest", "LEAK({0})")
    assert "math.leaktest" not in reg.backends["demo.simple_shader.python"].code_for_op
