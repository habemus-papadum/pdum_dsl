"""dispatch and its traced twin are ONE behavior (design 120 §5): same
results, same cache movement, events the only difference. The matrix runs
every target shape that exists at P1 — Handles (scalar and statement-heavy)
and Pipelines — through both paths via the spelled oracle door. (Transform
targets rejoin the matrix with the tensor tier.)"""

import pdum.dsl  # noqa: F401
import pytest
from pdum.dsl import events  # the SEAM (emit/span/SINKS), not the recorder
from pdum.dsl.api import jit
from pdum.dsl.pipe import op
from pdum.dsl.registry import DEFAULT

PHASES = {"dispatch.probe", "dispatch.extract", "dispatch.pack", "dispatch.launch"}


@pytest.fixture()
def sink():
    seen: list = []
    events.SINKS.append(lambda *e: seen.append(e))
    yield seen
    events.SINKS.clear()


def matrix():
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

    @op
    def stage(k):
        @jit()
        def s(x):
            return x + k

        return s

    return [
        (scalar(3.0), (1.0,)),
        (carries(0.5), (2.0,)),
        (stage(1.0) | stage(2.0), (4.0,)),  # DerivedValue target: the traced twin
        # must handle wrapper captures/kinds identically to Handles
    ]


def _run(target, args):
    return DEFAULT.dispatch(target, args, backend="reference")


def test_traced_dispatch_agrees(sink):
    for target, args in matrix():
        dark = _run(target, args)  # sink armed: traced body
        events.SINKS.clear()
        untraced = _run(target, args)  # dark: the plain body
        events.SINKS.append(lambda *e: sink.append(e))
        traced = _run(target, args)
        assert dark == untraced == traced, f"{target.fntype.template.label}: traced twin drifted"
        names = {e[0] for e in sink if e[0].startswith("dispatch.")}
        assert names == PHASES, f"{target.fntype.template.label}: phases {names} != {PHASES}"
        sink.clear()


def test_miss_path_phases_appear_under_the_spans(sink):
    def make(c):
        @jit()
        def f(x):
            return c + x

        return f

    _run(make(9.125), (1.0,))  # cold: the miss path runs with the sink armed
    names = [e[0] for e in sink]
    for expected in ("spec.miss", "spec.compile", "lower", "rewrite", "render", "artifact.compile"):
        assert expected in names, f"{expected} missing from {names}"
    by_name = {e[0]: e for e in sink}
    assert by_name["lower"][3] > by_name["spec.compile"][3]  # nested deeper
