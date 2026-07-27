"""Design 120 — the kernel event seam: dark no-op, sink protocol, forbid
wildcards, exception-path spans, depth nesting, and no_compile equivalence."""

import pytest
from pdum.dsl import events
from pdum.dsl.events import EventForbidden, emit, forbid, span


@pytest.fixture()
def sink():
    seen: list = []
    events.SINKS.append(lambda *e: seen.append(e))
    yield seen
    events.SINKS.clear()


def test_dark_path_is_a_noop():
    emit("spec.miss", key=("k",))  # no sinks, no forbid: nothing happens
    with span("spec.compile"):
        pass


def test_sink_receives_the_protocol_tuple(sink):
    emit("guard.drift", key=("K",), dur_ns=7)
    assert sink == [("guard.drift", ("K",), 7, 0, None)]


def test_span_times_and_nests(sink):
    with span("outer", "a"):
        with span("inner", "b"):
            pass
    (n1, k1, d1, depth1, _), (n2, k2, d2, depth2, _) = sink
    assert (n1, depth1) == ("inner", 1) and (n2, depth2) == ("outer", 0)
    assert d1 >= 0 and d2 >= d1  # the outer span contains the inner


def test_span_records_on_the_exception_path(sink):
    with pytest.raises(ValueError):
        with span("spec.compile", "K"):
            raise ValueError("a 40ms compile that then raises is a perf event")
    assert sink and sink[0][0] == "spec.compile"


def test_forbid_exact_and_wildcard():
    with forbid("guard.drift"):
        with pytest.raises(EventForbidden, match="guard.drift"):
            emit("guard.drift")
    with forbid("spec.*"):
        with pytest.raises(EventForbidden, match=r"spec\.miss"):
            emit("spec.miss")
        emit("guard.drift")  # not matched: fine
    emit("spec.miss")  # scope ended


def test_forbid_carries_the_lazy_detail():
    with forbid("spec.miss"):
        with pytest.raises(EventForbidden, match="nearest entry differs in: env_types"):
            emit("spec.miss", key=("K",), detail=lambda: "nearest entry differs in: env_types")


def test_forbid_works_dark_and_span_checks_at_entry():
    assert not events.SINKS
    with forbid("artifact.*"):
        with pytest.raises(EventForbidden):
            with span("artifact.compile"):
                raise AssertionError("must refuse BEFORE doing the work")


def test_no_compile_is_one_call_on_the_seam():
    """The proposal's design test (120 §1.2): the old context manager must be
    expressible as one forbid() call, alias intact."""
    from pdum.dsl.cache import CompileForbidden, no_compile

    assert CompileForbidden is EventForbidden
    with no_compile():
        with pytest.raises(CompileForbidden):
            emit("spec.miss")
