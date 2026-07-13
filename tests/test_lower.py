"""Step 6 — lowering: source -> typed IR, fates, inlining, the build rule."""

import pytest

from pdum.dsl.combinators import PIPE_BUILDERS, op, register_composition, register_role
from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.capture import SourceSnapshot, make_handle
from pdum.dsl.kernel.ir import format_loc, verify
from pdum.dsl.kernel.lower import (
    MissingRule,
    NameFateError,
    NoSourceError,
    StaleSourceError,
    lower_handle,
)
from pdum.dsl.kernel.ops import CORE_OPS
from pdum.dsl.stdlib.base_lang import LOWER_RULES

# Stand-in registrations (idempotent; also done in test_combinators):
register_role("device")
register_composition("pipe", "device", "device", "fuse")


def lower(h, *arg_types, derived=None):
    region = lower_handle(h, LOWER_RULES, CORE_OPS, arg_types=tuple(arg_types), derived=derived)
    verify(region, CORE_OPS)
    return region


def dist2(cx, cy):
    @jit()
    def go(px, py):
        dx = px - cx
        dy = py - cy
        return dx * dx + dy * dy

    return go


def test_lowering_end_to_end_with_env_paths_and_locs():
    h = dist2(320.0, 240.0)
    region = lower(h, T.f64, T.f64)
    result = region.body[-1].args[0]
    assert result.op == "core.add" and result.type == T.f64
    envs = {n.attrs for n in _walk(region) if n.op == "core.env"}
    assert envs == {(("slot", (0,)),), (("slot", (1,)),)}  # cx, cy as root PATHS
    # locs are absolute file:line, rebased via the snapshot:
    snap = h.snapshot
    lines = snap.text.splitlines()
    expected = snap.firstlineno + next(i for i, ln in enumerate(lines) if "dx = px - cx" in ln)
    sub = next(n for n in _walk(region) if n.op == "core.sub")
    assert sub.loc.file.endswith("test_lower.py") and sub.loc.line == expected


def _walk(region, seen=None):
    seen = set() if seen is None else seen
    for n in region.body:
        yield from _walk_node(n, seen)


def _walk_node(n, seen):
    if id(n) in seen:
        return
    seen.add(id(n))
    yield n
    for a in n.args:
        yield from _walk_node(a, seen)
    for r in n.regions:
        for m in r.body:
            yield from _walk_node(m, seen)


def test_base_dialect_is_strict_and_errors_carry_source():
    @jit()
    def mixed(x):
        return x + 1  # i64 literal against f64 param: the base pack does NOT auto-cast

    with pytest.raises(TypeError, match=r"strict.*test_lower\.py"):
        lower(mixed, T.f64)

    @jit()
    def fixed(x):
        return x + float(1)

    region = lower(fixed, T.f64)
    assert any(n.op == "core.cast" for n in _walk(region))  # the cast is IN the IR


def test_ifexp_and_compare():
    @jit()
    def clamp01(x):
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    region = lower(clamp01, T.f64)
    ifs = [n for n in _walk(region) if n.op == "core.if"]
    assert len(ifs) == 2 and all(n.type == T.f64 for n in ifs)


def test_local_rebinding_is_ssa():
    @jit()
    def accum(x):
        t = x
        t = t * 2.0
        t = t + 1.0
        return t

    assert lower(accum, T.f64).body[-1].args[0].op == "core.add"


def test_name_fates_are_loud():
    @jit()
    def uses_global(x):
        return x + GLOBAL_JUNK  # noqa: F821

    with pytest.raises(NameFateError, match="GLOBAL_JUNK.*capture it"):
        lower(uses_global, T.f64)


def test_missing_rule_names_syntax_and_loc():
    @jit()
    def loops(x):
        while x > 0.0:  # bounded `for` landed at step 11; `while` stays a loud, located refusal
            x = x - 1.0
        return x

    with pytest.raises(MissingRule, match=r"while.*test_lower\.py"):
        lower(loops, T.f64)


def test_no_source_and_stale_source():
    ns = {}
    exec(compile("def f(k):\n    def g(x):\n        return x\n    return g\n", "<nofile>", "exec"), ns)
    with pytest.raises(NoSourceError):
        lower(make_handle(ns["f"](1), "device"), T.f64)

    h = dist2(1.0, 2.0)
    h.snapshot = SourceSnapshot(
        h.snapshot.text.replace("dx * dx", "dx * dy"), h.snapshot.filename, h.snapshot.firstlineno, h.snapshot.qualname
    )
    with pytest.raises(StaleSourceError, match="drifted"):
        lower(h, T.f64, T.f64)


def make_inner(k):
    @jit()
    def inner(y):
        t = y * k
        return t + t  # multi-statement: no M0 single-return restriction

    return inner


def make_outer(inner, c):
    @jit()
    def outer(x):
        return inner(x) + c

    return outer


def test_inlining_merges_env_paths_and_chains_provenance():
    h = make_outer(make_inner(2.0), 10.0)
    region = lower(h, T.f64)
    envs = {dict(n.attrs)["slot"] for n in _walk(region) if n.op == "core.env"}
    # outer's bound order is sorted freevars: c -> (0,), inner -> slot 1, inner's k -> (1, 0)
    assert envs == {(0,), (1, 0)}
    inlined = next(n for n in _walk(region) if n.op == "core.mul")
    assert "inlined from" in format_loc(inlined.loc)


@op
def padd(k):
    @jit()
    def go(x):
        return x + k

    return go


@op
def pmul(k):
    @jit()
    def go(x):
        return x * k

    return go


def test_pipe_build_rule_fuses_without_source():
    pipeline = padd(1.0) | pmul(2.0)
    region = lower(pipeline, T.f64, derived=PIPE_BUILDERS)
    result = region.body[-1].args[0]
    assert result.op == "core.mul" and result.args[0].op == "core.add"  # fused chain
    envs = {dict(n.attrs)["slot"] for n in _walk(region) if n.op == "core.env"}
    assert envs == {(0, 0), (1, 0)}  # stage i's capture j — matches the Pipeline env tree
    assert "<pipeline>" in format_loc(result.loc)
