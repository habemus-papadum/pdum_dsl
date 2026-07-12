"""The hot path, end to end: THE THESIS TEST. A render loop that changes
capture values must compile once and only move bytes thereafter."""

import pytest

import pdum.dsl  # noqa: F401  — batteries: base dialect + python backend into DEFAULT
from pdum.dsl.combinators import collect, op
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.cache import CompileForbidden, no_compile
from pdum.dsl.kernel.registry import DEFAULT, Registry

_SOURCES = {
    "compiles": lambda: DEFAULT.specializations.compiles,
    "hits": lambda: DEFAULT.specializations.hits,
    "guard_misses": lambda: DEFAULT.specializations.guard_misses,
    "artifact_compiles": lambda: DEFAULT.artifacts.compiles,
}


@pytest.fixture(autouse=True)
def counters():
    DEFAULT.specializations.bump_generation()  # tier 1 cold per test (tier 2 content-addressed, untouched)
    before = {name: read() for name, read in _SOURCES.items()}

    class Delta:
        def __getattr__(self, name):  # one name list; a typo is a loud KeyError
            return _SOURCES[name]() - before[name]

    return Delta()


def make_shader(cx, gain):
    @jit()
    def shader(x):
        d = x - cx
        return gain * (1.0 if d < 0.5 else 0.0)

    return shader


def test_a_kernel_actually_runs():
    f = make_shader(0.3, 2.0)
    assert f(0.5) == 2.0  # d = 0.2 < 0.5 -> gain
    assert f(0.9) == 0.0  # d = 0.6 -> other branch


def test_the_thesis_300_frames_one_compile(counters):
    make_shader(0.0, 1.0)(0.5)  # frame 0: the one compile
    with no_compile():  # every further frame MUST be a hit — enforced, not hoped
        for i in range(1, 300):
            assert make_shader(i * 0.001, 1.0 + i)(0.5) == (1.0 + i)
    assert counters.compiles == 1
    assert counters.hits == 299
    assert counters.guard_misses == 0


def test_arg_type_change_is_a_new_specialization(counters):
    f = make_shader(0.0, 2.0)
    f(0.25)
    f(0.25)  # same types: hit
    with pytest.raises(TypeError):  # strict core: i64 - f64 refused, loudly, with locs
        f(1)
    assert counters.compiles == 1


def test_cold_key_under_no_compile_is_forbidden_and_named():
    f = make_shader(-1.0, 5.0)
    with no_compile(), pytest.raises(CompileForbidden):
        f(0.125)


def test_identical_bodies_share_the_artifact(counters):
    def site_a(k):
        @jit()
        def go(x):
            return x * k + 0.03125  # distinctive body: this test owns its artifact

        return go

    def site_b(k):
        @jit()
        def go(x):
            return x * k + 0.03125

        return go

    assert site_a(2.0)(3.0) == 6.03125
    assert site_b(5.0)(3.0) == 15.03125
    assert counters.compiles == 2  # two templates, two specializations...
    assert counters.artifact_compiles == 1  # ...ONE artifact: content-addressed tier proven


def test_guard_catches_a_rebound_capture(counters):
    def outer():
        k = 2.0

        @jit()
        def f(x):
            return x * k

        k = 3.0  # rebinding after decoration: the cell drifts from the captured env
        return f

    f = outer()
    assert f(1.0) == 2.0  # decoration-time semantics: env was captured at @jit
    assert f(1.0) == 2.0  # never silently stale — the drift is COUNTED, loudly
    assert counters.guard_misses >= 1


def test_pipelines_execute_fused(counters):
    @op
    def scale(k):
        @jit()
        def go(x):
            return x * k

        return go

    @op
    def shift(b):
        @jit()
        def go(x):
            return x + b

        return go

    assert (2.0 > (scale(3.0) | shift(1.0))) == 7.0
    assert (2.0 > (scale(3.0) | shift(1.0) | collect)) == 7.0
    with no_compile():  # new stage configs, same types: the pipe specialization holds
        assert (5.0 > (scale(4.0) | shift(2.0))) == 22.0


def test_rendered_source_reads_the_staging_abi():
    f = make_shader(0.7, 9.0)
    f(0.5)
    key = DEFAULT.specializations.key_for(f, (DEFAULT.table.fingerprint(0.5),), ("demo.simple_shader.python", 1))
    src = DEFAULT.specializations._ready[key].artifact.__pdum_source__
    assert "def kernel(staging, leaves):" in src
    assert "_u('<d', staging," in src and "return v" in src
    assert "env" not in src  # no logical capture survives into the artifact


def test_generation_bump_recompiles_but_artifacts_survive(counters):
    f = make_shader(0.1, 1.5)
    f(0.5)
    arts_before_bump = DEFAULT.artifacts.compiles
    DEFAULT.specializations.bump_generation()
    f(0.5)
    assert counters.compiles == 2
    assert DEFAULT.artifacts.compiles == arts_before_bump  # tier 2 is generation-free by design


def test_bare_registry_is_loud():
    from pdum.dsl.kernel.registry import NoBackend

    bare = Registry()
    f = make_shader(0.0, 1.0)
    with pytest.raises(NoBackend, match="batteries"):
        bare.dispatch(f, (1.0,))


def test_lazy_branches_guard_then_divide():
    """The step-8 review found the eager first draft crashing on exactly the
    guarded input. Branches are real if/else statements now — lazy."""

    @jit()
    def guarded(x):
        return (1.0 / x) if x > 0.0 else 0.0

    assert guarded(2.0) == 0.5
    assert guarded(0.0) == 0.0  # the eager renderer raised ZeroDivisionError here

    def make(cx):
        @jit()
        def shared(x):
            d = x - cx
            return (d if x > 0.0 else 0.0) + d  # d used in-branch AND at the join: hoists

        return shared

    assert make(0.5)(2.0) == 3.0
    assert make(0.5)(-1.0) == -1.5


def test_guard_reaches_into_nested_handles(counters):
    """An inlined callee is baked into the artifact; ITS drift must guard the
    entry too (review-caught hole)."""

    def outer():
        k = 2.0

        @jit()
        def inner(v):
            return v * k

        k = 5.0  # inner's cell drifts

        @jit()
        def f(x):
            return inner(x) + 1.0

        return f

    f = outer()
    assert f(1.0) == 3.0  # decoration-time k=2.0
    f(1.0)
    assert counters.guard_misses >= 1  # the OUTER entry noticed the INNER drift


def test_per_role_routing_with_two_backends():
    """Step 9: routes map kinds to backends; unrouted kinds use the default."""
    from pdum.dsl.demo.simple_shader.python import PYTHON, install
    from pdum.dsl.kernel.registry import Backend
    from pdum.dsl.stdlib import install as install_lang

    reg = install_lang(Registry())
    install(reg)  # python, default
    hits = []

    def spy_compile(source, name="kernel"):
        hits.append(name)
        return PYTHON.compile(source, name)

    other = Backend(name="other", render=PYTHON.render, compile=spy_compile, fp=("other", 1))
    reg.register_backend(other, kinds=("spykind",))
    assert reg.dispatch(make_shader(0.0, 2.0), (0.25,)) == 2.0  # "device": default python, no spy
    assert hits == []

    @jit(kind="spykind")
    def k(x):
        return x + 1.0

    assert reg.dispatch(k, (1.0,)) == 2.0  # routed kind: compiled by the spy backend
    assert hits == ["kernel"]


def test_fresh_registry_via_install_seams():
    """Surface E is not a singleton in disguise: a hand-built Registry gets
    the same batteries through the explicit install() seams."""
    from pdum.dsl.demo.simple_shader.python import install as install_backend
    from pdum.dsl.stdlib import install as install_lang

    reg = install_lang(Registry())
    install_backend(reg)
    f = make_shader(0.25, 4.0)
    assert reg.dispatch(f, (0.5,)) == 4.0
    assert DEFAULT.specializations is not reg.specializations  # independent worlds
