"""Measurement as a satellite: adaptive microbenchmarks, phase timers, timelines.

Three tools, none of which the kernel knows about (zero kernel edits):

- ``benchmark(fn)`` — BenchmarkTools.jl-style adaptive sampling. One warmup
  call; then evals-per-sample is TUNED upward until a sample takes long
  enough to swamp timer resolution; then samples accumulate until the time
  budget. The **minimum** is the headline estimator (noise is strictly
  additive on a quiet machine); median/mean report the distribution.
  A naive ``timeit``-one-loop mean overstates fast paths — ch11b shows by
  how much.
- ``instrument(target, ...)`` — phase decomposition of the dispatch hot
  path, riding the kernel's EVENT SEAM (design 120): arming a sink routes
  dispatch through its traced twin, which stamps key+probe / extract /
  pack / launch. No cache-entry surgery — the step-10b monkeypatch this
  replaced could not see the miss path and broke on the guard drift it
  should have reported.
- ``Timeline`` — a list of labelled spans, rendered by the viz satellite as
  a static-HTML bar chart (hover for µs; one lane per phase family).

GPU decomposition rides the demo runtime's ``timed_call`` (WebGPU
``timestamp-query``: begin/end-of-pass timestamps in nanoseconds), which
splits ``launch`` into encode / GPU execution / readback — the instrument
that answers ch10's "where did the 2 ms go".
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from time import perf_counter as _pc

from .kernel import events
from .kernel.registry import DEFAULT


@dataclass
class Trial:
    """One benchmark's evidence: per-sample seconds (each averaged over
    ``evals`` inner calls), plus the tuning that produced them."""

    samples: list  # seconds per single eval (already divided by evals)
    evals: int
    warmup_s: float

    @property
    def minimum(self) -> float:
        return min(self.samples)

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples)

    def __repr__(self) -> str:
        u = 1e6
        return (
            f"Trial(min {self.minimum * u:.2f} µs · median {self.median * u:.2f} µs · "
            f"mean {self.mean * u:.2f} µs · {len(self.samples)} samples × {self.evals} evals)"
        )


def benchmark(
    fn, *, budget_s: float = 0.25, min_sample_s: float = 50e-6, max_samples: int = 10_000, timer=_pc
) -> Trial:
    """Adaptive microbenchmark. ``min_sample_s`` is the resolution floor: the
    per-sample duration is pushed above it by raising evals-per-sample, so
    quantization noise cannot masquerade as speed."""
    t0 = timer()
    fn()  # warmup: caches hot, imports done, JIT warmed
    warmup = timer() - t0
    evals = 1
    while True:  # tune: double evals until one sample clears the floor
        t0 = timer()
        for _ in range(evals):
            fn()
        if (timer() - t0) >= min_sample_s or evals >= 1 << 20:
            break
        evals <<= 1
    samples: list = []
    spent, deadline = 0.0, budget_s
    while spent < deadline and len(samples) < max_samples:
        t0 = timer()
        for _ in range(evals):
            fn()
        dt = timer() - t0
        samples.append(dt / evals)
        spent += dt
    return Trial(samples, evals, warmup)


@dataclass
class Timeline:
    """Labelled spans for the viz timeline widget: (label, start_s, dur_s, lane)."""

    spans: list = field(default_factory=list)
    title: str = "timeline"

    def add(self, label: str, start: float, dur: float, lane: str = "host") -> None:
        self.spans.append((label, start, dur, lane))

    @property
    def total(self) -> float:
        return max((s + d for _, s, d, _ in self.spans), default=0.0)


def _warm_record(registry, target, args, out):
    """Warm, then locate, the FastRecord that dispatch just used. (Used only
    by ``gpu_timeline``'s artifact-capability probe now; the CPU instrument
    rides the event seam. Retires with the step-14 runtime protocol.)"""
    registry.dispatch(target, args, out)
    arg_fp = tuple(registry.table.fingerprint(a) for a in args)
    key = registry.specializations.key_for(target, arg_fp, registry.backend_for(target.kind).fp)
    record = registry.specializations.probe(key)
    if record is None:
        raise RuntimeError("no warm record to instrument (guards drifting?)")
    return record


def instrument(target, *args, out=None, registry=None, frames: int = 50) -> dict:
    """Decompose the dispatch hot path into phases, averaged over ``frames``.

    Rides the event seam: with a sink armed, ``dispatch`` routes through its
    traced twin (120 §5). The phases are measured on the REAL entry point —
    no hand-derived keys, no record surgery — and a record rebuilt mid-loop
    is a datum in the same stream (``guard.drift``), not an error.
    """
    registry = registry or DEFAULT
    registry.dispatch(target, args, out)  # warm: misses stay out of the phase means
    phase = {
        "dispatch.probe": "key+probe",
        "dispatch.extract": "extract",
        "dispatch.pack": "pack",
        "dispatch.launch": "launch",
    }
    sums = {"key+probe": 0.0, "extract": 0.0, "pack": 0.0, "launch": 0.0, "total": 0.0}

    def sink(name, key, dur_ns, depth, detail):
        p = phase.get(name)
        if p is not None:
            sums[p] += dur_ns / 1e9
            sums["total"] += dur_ns / 1e9

    events.SINKS.append(sink)
    try:
        for _ in range(frames):
            registry.dispatch(target, args, out)
    finally:
        events.SINKS.remove(sink)
    return {k: v / frames for k, v in sums.items()}


def phase_timeline(phases: dict, title: str = "dispatch") -> Timeline:
    """Order the instrument() phases into a single-lane host timeline."""
    tl = Timeline(title=title)
    cursor = 0.0
    for name in ("key+probe", "extract", "pack", "launch"):
        tl.add(name, cursor, phases[name], "host")
        cursor += phases[name]
    return tl


def gpu_timeline(target, *, out, registry=None, frames: int = 20) -> Timeline | None:
    """Split `launch` further for the demo WGSL runtime: encode+submit / GPU
    execution (timestamp-query, ns) / readback. None when the artifact has no
    ``timed_call`` (non-GPU targets) or the adapter lacks the feature.

    The frames are driven by ``registry.dispatch`` itself, with ``launch``
    shimmed to the artifact's ``timed_call`` — the SAME seam-wrap technique as
    ``instrument``, so the timed frame is exactly the frame the hot path runs
    (no hand-replayed extract/pack that could drift from dispatch)."""
    registry = registry or DEFAULT
    record = _warm_record(registry, target, (), out)
    timed = getattr(record.artifact, "timed_call", None)
    if timed is None:
        return None
    box: dict = {}
    orig_launch = record.launch
    record.launch = lambda staging, leaves: box.__setitem__("parts", timed(staging, leaves))
    agg: dict = {}
    try:
        for _ in range(frames):
            box.clear()
            registry.dispatch(target, (), out)
            if "parts" not in box:  # dispatch served a fresh, unwrapped record
                raise RuntimeError("record was rebuilt mid-instrument (guard drift?)")
            if box["parts"] is None:
                return None  # adapter lacks timestamp-query
            for k, v in box["parts"].items():
                agg[k] = agg.get(k, 0.0) + v / frames
    finally:
        record.launch = orig_launch
    tl = Timeline(title="one GPU frame")
    cursor = 0.0
    for name in ("encode+submit", "gpu", "readback"):
        tl.add(name, cursor, agg[name], "gpu" if name == "gpu" else "host")
        cursor += agg[name]
    return tl
