"""Step 3 — the two-tier cache: the thesis proven with dummy artifacts."""

import itertools
import threading

import pytest

from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.cache import (
    ArtifactCache,
    CompileForbidden,
    FastRecord,
    ReentrantCompile,
    SpecializationCache,
    no_compile,
)
from pdum.dsl.kernel.capture import make_handle
from pdum.dsl.kernel.valuekind import fingerprint

BACKEND = ("fake-gpu", "rgba8")


def make_closure(x):
    @jit(kind="device")
    def inner(y):
        return x + y

    return inner


def _draw(cache, handle, counter, args=(), backend=BACKEND, guards=()):
    key = cache.key_for(handle, tuple(fingerprint(a) for a in args), backend)
    return cache.get_or_compile(key, lambda: FastRecord(artifact=f"artifact#{next(counter)}", guards=guards))


# --- the thesis ---------------------------------------------------------------


def test_one_compile_per_type_signature():
    cache, n = SpecializationCache(), itertools.count(1)
    a = _draw(cache, make_closure(5), n)
    b = _draw(cache, make_closure(6), n)
    assert a is b and cache.compiles == 1  # values differ; ONE artifact
    _draw(cache, make_closure(3.0), n)
    assert cache.compiles == 2  # a capture changed TYPE: its own compile


def test_the_thesis_over_a_hot_loop():
    cache, n = SpecializationCache(), itertools.count(1)
    _draw(cache, make_closure(0), n)  # frame 0 pays the compile
    with no_compile():  # every later frame must be a pure hit
        for k in range(300):
            _draw(cache, make_closure(k), n)
    assert cache.compiles == 1 and cache.hits == 300


# --- perturbation: every key component, mutated, misses with its name ----------


def test_perturbation_names_the_differing_component():
    cache, n = SpecializationCache(), itertools.count(1)
    _draw(cache, make_closure(5), n, args=(1.0,))

    def expect_miss(naming, handle=None, args=(1.0,), backend=BACKEND):
        with no_compile(), pytest.raises(CompileForbidden, match=naming):
            _draw(cache, handle or make_closure(5), n, args=args, backend=backend)

    expect_miss("env_types", handle=make_closure(5.0))  # capture type changed
    expect_miss("arg_types", args=(1,))  # argument type changed
    expect_miss("backend", backend=("other-gpu",))
    ns1, ns2 = {}, {}
    exec(compile("def f(k):\n    def g():\n        return k\n    return g\n", "<a>", "exec"), ns1)
    exec(compile("def f(k):\n    def g():\n        return k + 1\n    return g\n", "<a>", "exec"), ns2)
    _draw(cache, make_handle(ns1["f"](1), "device"), n)
    expect_miss("template", handle=make_handle(ns2["f"](1), "device"))  # edited body
    cache.bump_generation()
    expect_miss("first sight|generation")  # tier 1 was cleared by the bump


def test_generation_bump_clears_tier_one_only():
    cache, n = SpecializationCache(), itertools.count(1)
    _draw(cache, make_closure(5), n)
    assert cache.bump_generation() == 1
    assert len(cache) == 0 and cache.retirements == 1
    _draw(cache, make_closure(5), n)
    assert cache.compiles == 2  # same closure, new world


# --- guards: refuse-or-recompile, never stale -----------------------------------


def test_guard_drift_recompiles_never_serves_stale():
    cache, n = SpecializationCache(), itertools.count(1)
    frozen = {"gain": object()}
    h = make_closure(5)
    a = _draw(cache, h, n, guards=((frozen, "gain", frozen["gain"]),))
    assert _draw(cache, make_closure(6), n) is a  # guard holds: hit
    frozen["gain"] = object()  # dependency drift
    b = _draw(cache, make_closure(7), n, guards=((frozen, "gain", frozen["gain"]),))
    assert b is not a and cache.guard_misses == 1 and cache.compiles == 2


# --- concurrency ---------------------------------------------------------------


def test_concurrent_misses_compile_once():
    cache, n = SpecializationCache(), itertools.count(1)
    barrier, results = threading.Barrier(8), []

    def worker():
        barrier.wait()
        results.append(_draw(cache, make_closure(1), n).artifact)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert cache.compiles == 1 and len(set(results)) == 1


def test_reentrant_compile_is_loud():
    cache = SpecializationCache()
    h = make_closure(1)
    key = cache.key_for(h)

    def recursive():
        return cache.get_or_compile(key, recursive)

    with pytest.raises(ReentrantCompile):
        cache.get_or_compile(key, recursive)


# --- hygiene: LRU + superseded-template retirement -------------------------------


def test_lru_evicts_oldest():
    cache, n = SpecializationCache(capacity=2), itertools.count(1)
    for x in (1, 1.0, True):  # three distinct env types
        _draw(cache, make_closure(x), n)
    assert len(cache) == 2 and cache.evictions == 1
    _draw(cache, make_closure(1), n)  # the oldest was evicted: recompiles
    assert cache.compiles == 4


def test_edited_template_retires_predecessor_entries():
    cache, n = SpecializationCache(), itertools.count(1)
    src_v1 = "def f(k):\n    def g():\n        return k\n    return g\n"
    src_v2 = src_v1.replace("return k", "return k * 2")

    def handle_from(src):
        ns = {}
        exec(compile(src, "<mymodule.py>", "exec"), ns)  # same filename+qualname: same def site
        return make_handle(ns["f"](1), "device")

    _draw(cache, handle_from(src_v1), n)
    assert cache.retirements == 0
    _draw(cache, handle_from(src_v2), n)  # the edit supersedes v1's entries
    assert cache.retirements == 1 and len(cache) == 1


# --- the artifact tier -----------------------------------------------------------


def test_artifact_tier_is_content_addressed():
    art, n = ArtifactCache(), itertools.count(1)
    build = lambda: f"compiled#{next(n)}"  # noqa: E731
    a = art.get_or_compile(("sha:abc", "wgsl", ()), build)
    b = art.get_or_compile(("sha:abc", "wgsl", ()), build)  # different template, same IR
    assert a is b and art.compiles == 1
    c = art.get_or_compile(("sha:abc", "c", ()), build)  # same IR, different backend
    assert c is not a and art.compiles == 2
