"""Design 120 §5 — the anti-drift gate, written BEFORE the twin: with a sink
armed, dispatch must route through `_dispatch_traced`, produce IDENTICAL
results across a matrix of target shapes, and emit a phase for every step
the untraced body performs."""

import pytest

np = pytest.importorskip("numpy")

import pdum.dsl  # noqa: F401, E402
from pdum.dsl.kernel import events  # noqa: E402
from pdum.dsl.kernel.api import jit  # noqa: E402
from pdum.dsl.kernel.registry import DEFAULT  # noqa: E402
from pdum.dsl.stdlib.arrays import Named  # noqa: E402

PHASES = {"dispatch.probe", "dispatch.extract", "dispatch.pack", "dispatch.launch"}


@pytest.fixture()
def sink():
    seen: list = []
    events.SINKS.append(lambda *e: seen.append(e))
    yield seen
    events.SINKS.clear()


def matrix():
    table = np.arange(6.0).reshape(2, 3)
    nt = Named(np.arange(6.0).reshape(2, 3), ("y", "x"))

    def scalar(c):
        @jit()
        def f(x):
            return c * (x - 0.25)

        return f

    def carries(g):
        @jit()
        def f(x):
            a = x
            b = 1.0
            for i in range(4):
                if a > b:
                    b = b + a * g
                else:
                    a = a + 0.5
            return a + b

        return f

    def arrays(t):
        @jit()
        def f(i, j):
            return t[i, j] * 2.0

        return f

    def named(t):
        @jit()
        def f(i, j):
            return t.isel(x=j, y=i) * 2.0

        return f

    return [
        (scalar(2.0), (1.5,)),
        (carries(0.3), (0.75,)),
        (arrays(table), (1, 2)),
        (named(nt), (1, 2)),
    ]


def test_traced_dispatch_agrees(sink):
    for target, args in matrix():
        dark = DEFAULT.dispatch(target, args)  # warm/compile through... sink armed: traced
        events.SINKS.clear()
        untraced = DEFAULT.dispatch(target, args)  # dark: today's body
        events.SINKS.append(lambda *e: sink.append(e))
        traced = DEFAULT.dispatch(target, args)
        assert dark == untraced == traced, f"{target.fntype.template.label}: traced twin drifted"
        names = {e[0] for e in sink if e[0].startswith("dispatch.")}
        assert names == PHASES, f"{target.fntype.template.label}: phases {names} != {PHASES}"
        sink.clear()


def test_traced_dispatch_agrees_on_c_backend(sink):
    from pdum.dsl.backends import c

    if not c.is_available():
        pytest.skip("no C compiler")
    ext = DEFAULT.extend()
    c.install(ext, default=True)

    def make(k):
        @jit()
        def f(x):
            acc = x
            for i in range(3):
                acc = acc * k
            return acc

        return f

    f = make(1.5)
    traced = ext.dispatch(f, (2.0,))
    events.SINKS.clear()
    dark = ext.dispatch(f, (2.0,))
    assert traced == dark == 2.0 * 1.5**3


def test_miss_path_phases_appear_under_the_spans(sink):
    def make(c):
        @jit()
        def f(x):
            return c + x

        return f

    make(9.125)(1.0)  # cold: the miss path runs with the sink armed
    names = [e[0] for e in sink]
    for expected in ("spec.miss", "spec.compile", "lower", "rewrite", "render", "artifact.compile"):
        assert expected in names, f"{expected} missing from {names}"
    by_name = {e[0]: e for e in sink}
    assert by_name["lower"][3] > by_name["spec.compile"][3]  # nested deeper


def test_out_tag_still_flows_when_traced(sink):
    from pdum.dsl.demo.simple_shader import wgsl

    if not wgsl.is_available():
        pytest.skip("no wgpu adapter")

    @jit(kind="simple_shader.compute")
    def k(i):
        return i * 0.5

    out = DEFAULT.dispatch(k, (), 4)
    assert len(out) == 4  # the traced twin must pass out= through identically
