"""Design 120 §6 — the recorder: exact counts under sampling, interned
traces naming the user's loop, the drift canary, expect(), non-reentrancy."""

import pytest

import pdum.dsl  # noqa: F401
from pdum.dsl import events
from pdum.dsl.kernel.api import jit


def drifty_pair():
    """A kernel whose captured CELL can be rebound after decoration — the
    canonical guard-drift shape (same cell, new contents)."""
    c = 1.0

    @jit()
    def g(x):
        return c * x

    def rebind(v):
        nonlocal c
        c = v

    return g, rebind


def test_drifting_capture_is_counted_attributed_and_traced():
    """The motivating bug (120 §1.2): a capture rebound every frame recompiles
    every frame with a correct answer and no symptom — until now."""
    g, rebind = drifty_pair()
    g(1.0)
    n = 30
    with events.record() as ev:
        for i in range(n):
            rebind(float(i))  # SAME cell, new contents: drift on every call
            g(1.0)  # (captures are FROZEN at decoration: the value stays 1.0 —
            # the drift melts the cache, not the answer; that is the whole point)
    drift = ev["guard.drift"]
    assert drift.count == n  # exact, unsampled
    assert ev["spec.compile"].count == n  # each drift recompiled (the melt, visible)
    t = drift.exemplars[0].trace
    assert any("test_recorder" in fr.filename for fr in t.user_frames)  # names OUR loop


def test_counts_stay_exact_under_harsh_sampling():
    @jit()
    def f(x):
        return x * 2.5

    f(1.0)
    with events.record(policy={"dispatch.probe": events.Sampling(first=0, then=0)}) as ev:
        for _ in range(50):
            f(1.0)
    assert ev["dispatch.probe"].count == 50  # sampling gates traces, never counts
    assert ev["dispatch.probe"].traces == []  # nothing admitted


def test_report_renders_a_tree_and_top():
    def fresh(c):
        @jit()
        def f(x):
            return c + x * c * c

        return f

    with events.record() as ev:
        fresh(7.25)(1.0)  # a cold compile: the full miss-path tree
    text = str(ev)
    assert "spec.compile" in text and "lower" in text
    # dispatch.probe CONTAINS the compile on a cold call; both dominate:
    assert ev.top(2)[0].name in ("dispatch.probe", "spec.compile")
    assert ev["lower"].depth > ev["spec.compile"].depth


def test_expect_budgets():
    @jit()
    def f(x):
        return x + 41.5

    with events.expect(**{"spec.compile": 1, "guard.drift": 0}):
        f(1.0)  # cold: one compile
        f(2.0)  # same types: hit
    with pytest.raises(AssertionError, match="spec.compile: got 0, expected 1"):
        with events.expect(**{"spec.compile": 1}):
            f(3.0)  # all hits: no compile happens


def test_record_refuses_to_nest():
    with events.record():
        with pytest.raises(RuntimeError, match="do not nest"):
            with events.record():
                pass
    with events.record():  # re-armable after a clean exit
        pass


def test_forbid_guard_drift_is_now_writable():
    """The assertion 120 §6.4 says you cannot write today."""
    g, rebind = drifty_pair()
    g(1.0)
    with events.forbid("guard.drift"):
        g(2.0)  # stable capture: fine
    rebind(9.0)
    with pytest.raises(events.EventForbidden, match="guard.drift"):
        with events.forbid("guard.drift"):
            g(1.0)


def test_miss_exemplar_carries_explain():
    @jit()
    def f(x):
        return x + x  # valid for i64 AND f64: the arg TYPE is the axis

    f(1.0)
    with events.record() as ev:
        f(1)  # i64 signature: a legitimate spec.miss
    ex = ev["spec.miss"].exemplars[0]
    assert "differs in" in ex.explain or "first sight" in ex.explain


def test_interning_collapses_one_loop_to_one_trace():
    def fresh(c):
        @jit()
        def f(x):
            return x - c

        return f

    fresh(0.25)(1.0)  # compile once outside the recording
    with events.record(default=events.Sampling(every=1)) as ev:
        for i in range(12):
            fresh(float(i) + 0.5)(1.0)  # 12 fresh closures: hits, one call site
    probe = ev["dispatch.probe"]
    assert probe.count == 12
    assert len(probe.traces) == 1  # every event sampled, ONE interned stack
