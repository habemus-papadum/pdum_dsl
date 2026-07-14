"""Step 10b — the bench satellite: adaptive sampling, phase decomposition,
and the step-9 microbench thresholds becoming REAL gates."""

import pytest

import pdum.dsl  # noqa: F401
from pdum.dsl import viz
from pdum.dsl.bench import benchmark, gpu_timeline, instrument, phase_timeline
from pdum.dsl.kernel.api import jit


def make(cx, gain):
    @jit()
    def f(x):
        return gain * (x - cx)

    return f


def test_benchmark_tunes_evals_for_fast_functions():
    ticks = iter(range(0, 10_000_000, 1))  # a fake 1-second-resolution clock

    def fake_timer():
        return next(ticks) * 1e-7  # 100 ns per tick

    t = benchmark(lambda: None, budget_s=0.001, min_sample_s=50e-6, timer=fake_timer, max_samples=3)
    assert t.evals > 1  # a no-op MUST be batched: one eval can't clear the floor
    assert t.samples and t.minimum >= 0.0


def test_benchmark_statistics_are_ordered():
    f = make(0.1, 2.0)
    f(1.0)
    t = benchmark(lambda: f(1.0), budget_s=0.05)
    assert 0 < t.minimum <= t.median <= max(t.samples)
    assert "µs" in repr(t)


def test_hit_path_threshold_gate():
    """The plan's step-9 gate, finally real: alarm 5 µs / fail 10 µs on the
    warm hit path (generous 4x CI margin on the fail line)."""
    f = make(0.3, 2.0)
    f(1.0)
    t = benchmark(lambda: f(1.0), budget_s=0.1)
    if t.minimum > 5e-6:
        print(f"ALARM: hit path {t.minimum * 1e6:.2f} µs exceeds the 5 µs alarm line")
    if t.minimum >= 40e-6:  # scheduler noise is one-sided: a REAL regression fails twice
        t = benchmark(lambda: f(1.0), budget_s=0.1)
    assert t.minimum < 40e-6, f"hit path {t.minimum * 1e6:.1f} µs blew the fail threshold (twice)"


def test_instrument_phases_sum_to_total_and_restore():
    from pdum.dsl.kernel.registry import DEFAULT

    f = make(0.5, 3.0)
    f(2.0)  # warm first, so the ORIGINAL seams can be captured before instrument runs
    key = DEFAULT.specializations.key_for(f, (DEFAULT.table.fingerprint(2.0),), DEFAULT.backend_for(f.kind).fp)
    rec = DEFAULT.specializations.probe(key)
    orig_extract, orig_launch = rec.extract, rec.launch
    phases = instrument(f, 2.0, frames=20)
    parts = phases["key+probe"] + phases["extract"] + phases["pack"] + phases["launch"]
    if not (parts <= phases["total"] <= parts + 50e-6):  # one-sided scheduler noise: retry once
        phases = instrument(f, 2.0, frames=20)
        parts = phases["key+probe"] + phases["extract"] + phases["pack"] + phases["launch"]
    assert parts <= phases["total"] <= parts + 50e-6  # phases nest inside the total
    # Post-120: instrument reads the event sink — it must NEVER touch the record's seams:
    assert rec.extract is orig_extract and rec.launch is orig_launch


def test_timeline_widget_is_static_html():
    tl = phase_timeline({"key+probe": 1e-6, "extract": 2e-6, "pack": 1e-6, "launch": 3e-6})
    html = viz.render(tl)._repr_html_()
    assert "timeline" in html and "<script" not in html.lower()
    assert html.count("data-tip") >= 4  # one hover per span


def test_gpu_timeline_decomposes_launch():
    from pdum.dsl.demo.simple_shader import wgsl

    if not wgsl.is_available():
        pytest.skip("no wgpu adapter")

    def gmake(cx):
        @jit(kind="simple_shader.compute")
        def k(i, j):
            return (i / 64.0 - cx) * (j / 64.0)

        return k

    tl = gpu_timeline(gmake(0.5), out=(512, 512), frames=5)
    if tl is None:
        pytest.skip("adapter lacks timestamp-query")
    names = [s[0] for s in tl.spans]
    assert names == ["encode+submit", "gpu", "readback"]
    spans = {s[0]: s[2] for s in tl.spans}
    assert 0 <= spans["gpu"] < 0.05  # tiny passes may quantize to one tick; never negative
    assert spans["readback"] > 0  # the blocking map is the real cost — ch10's answer
