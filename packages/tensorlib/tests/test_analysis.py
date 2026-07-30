"""The analysis cache (330 §4): compute-once, the DAG, the warmth law, the
measurement ledger, and the refusals."""

import numpy as np
import pytest

import pdum.tl.analysis as analysis
from pdum.dsl import events
from pdum.dsl.cache import Memo
from pdum.tl.analysis import defanalysis, load_ledger, no_reanalysis, save_ledger
from pdum.tl.certify import certify
from pdum.tl.licenses import GEMM_F16_TILES
from pdum.tl.zoo.tiles import gemm_tile


@pytest.fixture()
def fresh(monkeypatch):
    """An isolated cache world per test — the module globals are the seam."""
    monkeypatch.setattr(analysis, "ANALYSES", Memo("analysis", capacity=1 << 30))
    monkeypatch.setattr(analysis, "LEDGER", {})


@pytest.fixture()
def sink():
    seen: list = []
    events.SINKS.append(lambda *e: seen.append(e))
    yield seen
    events.SINKS.clear()


def test_analysis_computes_once_and_then_hits(fresh, sink):
    calls = []

    @defanalysis("t.double", 1)
    def double(x):
        calls.append(x)
        return x * 2

    a = double(21)
    b = double(21)
    assert a.value == b.value == 42 and a.key == b.key
    assert calls == [21]  # slow once
    assert sum(1 for e in sink if e[0] == "analysis.miss") == 1


def test_the_warmth_law_extends_to_analysis(fresh):
    @defanalysis("t.id", 1)
    def ident(x):
        return x

    ident(7)  # cold, outside the pin
    with no_reanalysis():
        ident(7)  # warm: a hit, permitted
    with pytest.raises(events.EventForbidden):
        with no_reanalysis():
            ident(8)  # cold inside the pin: the law bites


def test_facts_compose_into_a_dag(fresh):
    """A Fact input contributes its KEY (the DAG edge) and unwraps to its
    VALUE for the function; bumping the upstream version re-keys every
    downstream fact."""
    seen = []

    @defanalysis("t.up", 1)
    def up1(x):
        return x + 1

    @defanalysis("t.up", 2)
    def up2(x):
        return x + 1

    @defanalysis("t.down", 1)
    def down(v):
        seen.append(v)
        return v * 10

    d1 = down(up1(4))
    assert seen == [5]  # the fact unwrapped
    d2 = down(up2(4))
    assert d1.value == d2.value == 50
    assert d1.key != d2.key  # same value, different provenance: different fact


def test_params_and_versions_enter_the_key(fresh):
    @defanalysis("t.f", 1)
    def f(x, *, mode):
        return (x, mode)

    assert f(1, mode="a").key != f(1, mode="b").key
    assert f(1, mode="a").key == f(1, mode="a").key


def test_unkeyable_inputs_refuse(fresh):
    @defanalysis("t.f", 1)
    def f(x):
        return x

    with pytest.raises(TypeError, match="content-keyable"):
        f(object())


def test_ndarray_inputs_key_by_content(fresh):
    calls = []

    @defanalysis("t.sum", 1)
    def total(a):
        calls.append(1)
        return float(a.sum())

    x = np.arange(6.0).reshape(2, 3)
    assert total(x).key == total(x.copy()).key  # content, not identity
    assert len(calls) == 1


def test_the_measurement_ledger_round_trips(fresh, tmp_path):
    """A measurement is an analysis whose evaluator is the machine: plain
    JSON data, persisted, and NEVER re-run for an identical key — across
    sessions included."""
    calls = []

    @defanalysis("bench.fake", 1, ledger=True)
    def measure(name, *, machine):
        calls.append(name)
        return {"ms": 1.5, "reps": [3, 4]}

    fact = measure("gemm", machine="rtx4090")
    assert fact.value == {"ms": 1.5, "reps": [3, 4]}
    path = tmp_path / "ledger.json"
    assert save_ledger(path) == 1

    # a fresh session: empty caches, the ledger reloads, the machine is idle
    analysis.ANALYSES = Memo("analysis", capacity=1 << 30)
    analysis.LEDGER = {}
    assert load_ledger(path) == 1
    again = measure("gemm", machine="rtx4090")
    assert again.value == {"ms": 1.5, "reps": [3, 4]}
    assert calls == ["gemm"]  # measured exactly once, ever


def test_ledger_values_must_be_plain_data(fresh):
    @defanalysis("bench.bad", 1, ledger=True)
    def bad():
        return object()

    with pytest.raises(TypeError, match="plain data"):
        bad()


def test_certification_is_warm_on_repeat(fresh):
    """The seam's first production consumer: a repeated certify of the same
    flagship is all hits — erasure and normalization never recompute."""
    f = gemm_tile()
    lics = tuple(x for x in GEMM_F16_TILES if x.kind == "reassociation")
    certify(f.region, f.naive, licenses=lics)  # cold
    with no_reanalysis():
        cert = certify(f.region, f.naive, licenses=lics)  # warm: zero misses
    assert cert.verdict == "proved-licensed"
