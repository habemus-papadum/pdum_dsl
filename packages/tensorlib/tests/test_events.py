"""The tl side of the events seam (200 §1.10, P3): assemblage builds and
adjoint derivation are compile-ish acts that announce themselves, so forbid()
can pin "this loop builds nothing" — and the cache-backed registries make
re-registration a HIT, pinnable the same way (the idempotence gate)."""

from itertools import count

import numpy as np
import pytest

from pdum.dsl import events
from pdum.dsl.cache import CompileForbidden
from pdum.tl import Tensor, defmarker, defreducer
from pdum.tl.assemblage import assemblage, unit
from pdum.tl.autodiff import grad
from pdum.tl.compute import pointwise, red, reduce
from pdum.tl.dialect import run_named
from pdum.tl.lifting import lift_step
from pdum.tl.mdsl import exp
from pdum.tl.registry import MARKERS


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


@pytest.fixture()
def sink():
    seen: list = []
    events.SINKS.append(lambda *e: seen.append(e))
    yield seen
    events.SINKS.clear()


_FRESH = count()


def _fresh_assemblage():
    """A never-before-seen build identity: the captured tag rides the
    assemblage fingerprint, so every call is a COLD build."""
    tag = 1.0 + next(_FRESH)

    @unit
    def f(h):
        return h * tag

    return assemblage(f, h=T([0.0, 1.0], ("i",)).layout)


def test_building_an_assemblage_announces_itself(sink):
    # the program.build event died with the Program IR; the surviving
    # compile-ish announcement is the assemblage cache seam
    _fresh_assemblage()
    assert [e[0] for e in sink] == ["assemblage.miss", "assemblage.compile"]


def test_a_hot_loop_builds_zero_assemblages():
    """THE pin the seam exists for: running a cached build is not building
    — forbid proves it, structurally."""
    a = _fresh_assemblage()
    x = T([0.0, 1.0], ("i",))
    with events.forbid("assemblage.miss"):
        for _ in range(5):
            run_named(a.region, {"h": x}, a.names)
    with events.forbid("assemblage.miss"), pytest.raises(CompileForbidden):
        _fresh_assemblage()


def test_adjoint_derivation_is_a_span(sink):
    from pdum.tl.markers import exp as m_exp

    def step(x):
        y = pointwise(m_exp, x)
        return reduce(red.sum, y, "i")

    ls = lift_step(step, x=T([0.0, 1.0], ("i",)).layout)
    sink.clear()
    grad(ls.region, ls.outputs[0], {"x": T([0.0, 1.0], ("i",))}, names=ls.names)
    assert "adjoint.derive" in [e[0] for e in sink]
    # the "program.build nests deeper than adjoint.derive" clause died with
    # the Program IR — the derivation builds no Program inside its span


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
