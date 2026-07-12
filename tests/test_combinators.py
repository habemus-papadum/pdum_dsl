"""Step 3b — pipelines as values: the blessed combinator satellite."""

import itertools

import pytest

from pdum.dsl.combinators import (
    IncompatibleRoles,
    NotYetExecutable,
    Pipeline,
    Terminal,
    collect,
    op,
    register_composition,
    register_role,
    set_dispatcher,
)
from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.cache import FastRecord, SpecializationCache, no_compile
from pdum.dsl.kernel.valuekind import fingerprint, typeof

# Stand-in for the base-language (stdlib) package, which will own the
# "device" role and the fuse rule once lowering exists (the satellite itself
# pre-enumerates nothing beyond its own materializer concept):
register_role("device")
register_composition("pipe", "device", "device", "fuse")


@op
def add(k):
    @jit()
    def go(x):
        return x + k

    return go


@op
def mul(k):
    @jit()
    def go(x):
        return x * k

    return go


def frag():
    @jit(kind="fragment")
    def shader():
        return (0.0, 0.0)

    return shader


# --- definition is not application ---------------------------------------------


def test_defining_runs_nothing_and_identity_is_stable():
    p = add(1) | mul(2)
    q = add(9) | mul(7)  # fresh values every "frame"
    assert isinstance(p, Pipeline)
    assert p.fntype == q.fntype and p.fp == q.fp  # rebuild-stable identity
    assert p.fntype.template.label.startswith("pipe(")


def test_stage_param_type_changes_identity():
    assert (add(1) | mul(2)).fp != (add(1.0) | mul(2)).fp  # i64 -> f64 capture


def test_flattened_associativity():
    a, b, c = add(1), mul(2), add(3)
    assert ((a | b) | c).fp == (a | (b | c)).fp  # grouping vanishes syntactically
    assert len(((a | b) | c).parts) == 3


def test_config_is_conservatively_static():
    assert add(1)[64].fp != add(1).fp  # config in identity (recompile-on-change, never wrong)
    assert add(1)[64, 128].config == (64, 128)
    assert add(1)[64].fp == add(1)[64].fp


def test_config_accepts_named_mappings():
    s = add(1)[{"grid": 256, "block": 64}]
    assert s.config == (("block", 64), ("grid", 256))  # canonicalized: order-free
    assert add(1)[{"block": 64, "grid": 256}].fp == s.fp
    with pytest.raises(TypeError, match="hashable"):
        add(1)[{"grid": [1, 2]}]


# --- roles gate composition -------------------------------------------------------


def test_fragment_role_arrives_with_its_domain():
    from pdum.dsl.combinators import Stage

    # No graphics vocabulary is built in — roles ship with their packages:
    with pytest.raises(IncompatibleRoles, match="no Role registered"):
        add(1) | frag()
    # ...and once a (future WGSL) package registers it, composition is still
    # gated by rules, with the package's explanation:
    register_role("fragment", hint="orchestration arrives with the pass runtime")
    with pytest.raises(IncompatibleRoles, match="orchestration"):
        add(1) | frag()  # raw Handle operands are wrapped automatically
    with pytest.raises(IncompatibleRoles, match="no composition rule"):
        Stage(frag()) | add(1)


def test_materializer_only_ends_a_pipeline():
    p = add(1) | mul(2) | collect
    assert isinstance(p, Pipeline)
    with pytest.raises(IncompatibleRoles, match="ends a pipeline"):
        add(1) | collect | mul(2)


def test_unregistered_kind_is_loud():
    @jit(kind="compute")
    def k():
        return 0

    with pytest.raises(IncompatibleRoles, match="no Role registered"):
        add(1) | k
    register_role("compute")  # registering opens the role, but no rule yet:
    with pytest.raises(IncompatibleRoles, match="no composition rule"):
        add(1) | k


# --- application: loud without a dispatcher; the thesis with one -------------------


def test_apply_without_dispatcher_is_loud():
    with pytest.raises(NotYetExecutable):
        6 > (add(1) | mul(2))


def test_the_thesis_for_pipelines():
    cache, serial = SpecializationCache(), itertools.count(1)

    def dummy(pipeline, value):
        key = cache.key_for(pipeline, (fingerprint(value),), ("dummy",))
        rec = cache.get_or_compile(key, lambda: FastRecord(artifact=f"<fused #{next(serial)}>"))
        return ("DeviceValue", rec.artifact)

    prev = set_dispatcher(dummy)
    try:
        assert (6 > (add(1) | mul(2)))[0] == "DeviceValue"
        with no_compile():
            for k in range(300):  # rebuild the whole pipeline, fresh values, every step
                out = k > (add(k) | mul(k + 1))
        assert cache.compiles == 1 and cache.hits == 300
        assert out[1] == "<fused #1>"
    finally:
        set_dispatcher(prev)


# --- pipelines are first-class values ----------------------------------------------


def test_pipelines_are_values_of_the_type_system():
    p = add(1) | mul(2)
    assert typeof(p) == p.fntype
    assert fingerprint(p) == p.fp
    assert typeof((p, add(3) | mul(4))) == T.Tuple((p.fntype, (add(3) | mul(4)).fntype))


def test_terminal_repr_and_pipeline_repr_read_well():
    p = add(1)[64] | mul(2) | collect
    text = repr(p)
    assert "[64]" in text and "collect" in text and " | " in text
    assert isinstance(p.parts[-1], Terminal)
