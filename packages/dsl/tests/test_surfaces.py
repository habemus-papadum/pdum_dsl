"""Step 10 — the five surfaces: every extension lands through a registration,
and the extension-locality law (zero kernel diffs) is enforced by construction:
this file touches ONLY public registry surfaces."""

from dataclasses import dataclass

import pdum.dsl  # noqa: F401  — batteries
import pytest
from pdum.dsl import install as install_lang
from pdum.dsl.api import jit
from pdum.dsl.reference import install as install_reference
from pdum.dsl.reference import reference
from pdum.dsl.registry import Registry
from pdum.dsl.render import emit_dominated  # noqa: F401  (namespace pkg importable)
from pdum.dsl.surfaces import Dialect, defop, intrinsic, overload, record, spell


def _length2(v):  # module-level battery: cross-refs are globals, not captures
    return sqrt(v[0] * v[0] + v[1] * v[1])  # noqa: F821


def fresh() -> Registry:
    reg = Registry()
    install_reference(reg)  # backends BEFORE the dialect: batteries spell onto them
    install_lang(reg)
    return reg


# --- surface A + D: defop + spell, end to end -----------------------------------


def test_defop_and_spell_add_an_op_with_zero_kernel_diffs():
    reg = fresh()
    defop(reg, "math.tanh", lambda args, attrs, regions: args[0])
    intrinsic(reg, "tanh", "math.tanh")
    spell(reg, "reference", "math.tanh", "math.tanh({0})")

    @jit()
    def f(x):
        return tanh(x) * 2.0  # noqa: F821 — resolved through the overload table

    import math

    assert abs(reg.dispatch(f, (0.5,), backend="reference") - math.tanh(0.5) * 2.0) < 1e-12


def test_decomposition_gates_on_the_target_op_set():
    """A backend WITHOUT the native op gets the decomposition for free; one
    WITH it never pays (§2.10) — asserted via the rendered source."""
    from pdum.dsl.rewrite import Pat

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

    assert abs(reg.dispatch(f, (0.3,), backend="reference") - math.sinh(0.3)) < 1e-12
    src = next(iter(reg.specializations._ready.values())).artifact.__pdum_source__
    assert "math.exp(" in src and "sinh" not in src  # decomposed away
    # now give the backend a native spelling: fresh registry, same kernel shape
    reg2 = fresh()
    defop(reg2, "math.sinh", lambda args, attrs, regions: args[0])
    intrinsic(reg2, "sinh", "math.sinh")
    spell(reg2, "reference", "math.sinh", "math.sinh({0})")
    reg2.decompositions.append(reg.decompositions[-1])

    @jit()
    def g(x):
        return sinh(x)  # noqa: F821

    reg2.dispatch(g, (0.3,), backend="reference")
    src2 = next(iter(reg2.specializations._ready.values())).artifact.__pdum_source__
    assert "math.sinh(" in src2  # native op survived: the gate skipped the decomposition


# --- surface B: DSL-written batteries ---------------------------------------------


def test_batteries_are_inlined_and_compose():
    @jit()
    def f(x):
        return smoothstep(0.0, 1.0, clamp(x, 0.0, 1.0))  # noqa: F821

    assert reference(f)(0.5) == 0.5
    assert reference(f)(-3.0) == 0.0 and reference(f)(9.0) == 1.0


def test_overload_must_be_capture_free():
    reg = fresh()
    k = 3.0

    def leaky(x):
        return x * k

    with pytest.raises(TypeError, match="capture-free"):
        overload(reg, "leaky")(leaky)


def test_tuple_batteries_through_an_overload():
    """Domain vocabulary is an ordinary registered overload away (090
    minimalism: length2 was demo vocabulary; here it is test vocabulary)."""
    from pdum.dsl.registry import DEFAULT
    from pdum.dsl.surfaces import overload

    if "length2" not in DEFAULT.overloads:
        overload(DEFAULT, "length2")(_length2)

    @jit()
    def f(x):
        v = (x, x * 2.0)
        return length2(v)  # noqa: F821

    assert abs(reference(f)(3.0) - (9 + 36) ** 0.5) < 1e-12


# --- surface C: records --------------------------------------------------------------


def test_record_roundtrip_fields_methods_and_thesis():
    from dataclasses import dataclass

    from pdum.dsl.cache import no_compile
    from pdum.dsl.registry import DEFAULT

    @dataclass(frozen=True)
    class Color:
        r: float
        g: float
        b: float

        def luminance(self):
            return 0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b

        def scaled(self, k):
            return (self.r * k, self.g * k, self.b * k)

    if "Color" not in {rec.__name__ for rec in getattr(DEFAULT, "_test_records", [])}:
        Color = record(DEFAULT, Color)
        DEFAULT._test_records = [Color]

    def tint(c):
        @jit()
        def f(x):
            return c.luminance() + c.scaled(2.0)[1] * x

        return f

    c = Color(0.5, 0.25, 0.125)
    expected = 0.2126 * 0.5 + 0.7152 * 0.25 + 0.0722 * 0.125 + 0.5 * 1.0
    assert abs(reference(tint(c))(1.0) - expected) < 1e-12
    reference(tint(c))(1.0)
    with no_compile():  # new color VALUES: same Record type, cache hit
        reference(tint(Color(0.9, 0.8, 0.7)))(1.0)


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
    spell(child, "reference", "math.cbrt", "({0} ** (1.0/3.0))")

    @jit()
    def f(x):
        return cbrt(x)  # noqa: F821

    assert abs(child.dispatch(f, (8.0,), backend="reference") - 2.0) < 1e-12
    assert "math.cbrt" not in reg.ops  # the parent never saw it

    @jit()
    def g(x):
        return cbrt(x)  # noqa: F821

    from pdum.dsl.lower import MissingRule

    with pytest.raises(MissingRule, match="cbrt"):
        reg.dispatch(g, (8.0,), backend="reference")


def test_dialect_is_bundling_sugar():
    reg = fresh()
    d = Dialect(
        "cbrt-pack",
        (
            lambda r: defop(r, "math.cbrt", lambda a, at, rg: a[0]),
            lambda r: intrinsic(r, "cbrt", "math.cbrt"),
            lambda r: spell(r, "reference", "math.cbrt", "({0} ** (1.0/3.0))"),
        ),
    )
    d.install(reg)

    @jit()
    def f(x):
        return cbrt(x)  # noqa: F821

    assert abs(reg.dispatch(f, (27.0,), backend="reference") - 3.0) < 1e-9


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
    import pdum.dsl.api as api
    import pdum.dsl.registry as kreg

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

    assert reg.dispatch(probe(Pixel(-1.5, 2.0)), (2.0,), backend="reference") == 7.0


def test_extend_does_not_share_spelling_tables():
    """A child's spell() must never mutate the parent's backend (the layering
    promise) — Backend records share everything EXCEPT code_for_op."""
    reg = fresh()
    child = reg.extend()
    spell(child, "reference", "math.leaktest", "LEAK({0})")
    assert "math.leaktest" not in reg.backends["reference"].code_for_op
