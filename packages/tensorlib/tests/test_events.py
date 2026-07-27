"""The tl side of the events seam (200 §1.10, P3): Program build and adjoint
derivation are compile-ish acts that announce themselves, so forbid() can pin
"this loop builds zero Programs" — and the cache-backed registries make
re-registration a HIT, pinnable the same way (the idempotence gate)."""

import numpy as np
import pytest
from pdum.dsl import events
from pdum.dsl.cache import CompileForbidden
from pdum.tl import Tensor, defmarker, defreducer
from pdum.tl.autodiff import grad
from pdum.tl.ir import Instr, Program, run
from pdum.tl.mdsl import exp
from pdum.tl.registry import MARKERS


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def I(var, op, operands=(), **params):  # noqa: E743
    return Instr(var, op, tuple(operands), params)


@pytest.fixture()
def sink():
    seen: list = []
    events.SINKS.append(lambda *e: seen.append(e))
    yield seen
    events.SINKS.clear()


def _prog() -> Program:
    return Program((I("x", "input"), I("y", "pointwise", ("x",), f="exp")))


def test_building_a_program_announces_itself(sink):
    _prog()
    assert [e[0] for e in sink] == ["program.build"]


def test_a_hot_loop_builds_zero_programs():
    """THE pin the seam exists for: running cached Programs is not building
    Programs — forbid proves it, structurally."""
    prog = _prog()
    x = T([0.0, 1.0], ("i",))
    with events.forbid("program.build"):
        for _ in range(5):
            run(prog, {"x": x})
    with events.forbid("program.build"), pytest.raises(CompileForbidden):
        _prog()


def test_adjoint_derivation_is_a_span_over_its_builds(sink):
    prog = Program(
        (
            I("x", "input"),
            I("y", "pointwise", ("x",), f="exp"),
            I("s", "reduce", ("y",), f="sum", dims=("i",)),
        )
    )
    sink.clear()
    grad(prog, "s", {"x": T([0.0, 1.0], ("i",))})
    names = [e[0] for e in sink]
    assert "adjoint.derive" in names and "program.build" in names
    by_name = {e[0]: e for e in sink}
    assert by_name["program.build"][3] > by_name["adjoint.derive"][3]  # nested deeper


def test_identical_marker_reregistration_is_one_entry_and_a_hit():
    """THE P3 GATE PIN: re-registering an identical marker yields one entry —
    the second registration is a cache hit (no marker.miss fires)."""
    m1 = defmarker("gate_sigmoid", 1, lambda x: 1 / (1 + exp(-x)))
    with events.forbid("marker.miss"):
        m2 = defmarker("gate_sigmoid", 1, lambda x: 1 / (1 + exp(-x)))
    assert m2 is m1  # one entry, the same object


def test_identical_reducer_reregistration_is_one_entry_and_a_hit():
    def declare():
        return defreducer(
            "gate_linrec",
            state=2,
            element=2,
            lift=lambda a, b: (a, b),
            combine=lambda left, right: (left[0] * right[0], right[0] * left[1] + right[1]),
            init=(1.0, 0.0),
            project=lambda A, B: B,
        )

    r1 = declare()
    with events.forbid("reducer.miss"):
        r2 = declare()
    assert r2 is r1


def test_derivation_under_cache_the_rewrite_runs_once():
    """partial(i) is a cache entry computed on demand from a cache entry:
    the first request derives (marker.miss fires); every later request is a
    hit — pinned with forbid, not with counters."""
    m = defmarker("gate_softplus", 1, lambda x: exp(x) / (1 + exp(x)))
    d = m.partial(0)
    assert d.name == "gate_softplus.d0" and d.name in MARKERS
    with events.forbid("marker.miss"):
        assert m.partial(0) is d
