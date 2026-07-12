"""Marshaling: plans from types alone, one generic packer, the ABI stages,
and the output mirror. The chapter claim: change a capture VALUE and the
plan, the artifact key, and the extractor are all untouched — only bytes move."""

import random
import struct

import pytest

from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.ir import Builder, Region, VerifyError
from pdum.dsl.kernel.lower import lower_handle
from pdum.dsl.kernel.ops import CORE_OPS
from pdum.dsl.kernel.pack import (
    _FMTS,
    ABI_OPS,
    NORMALIZE_ENV,
    LeafPath,
    ScalarLeaf,
    build_extractor,
    legalize_params,
    pack_into,
    plan_from_types,
    result_plan,
    unflatten,
    unpack_result,
)
from pdum.dsl.kernel.rewrite import Stage, run_stage
from pdum.dsl.kernel.valuekind import BUILTINS
from pdum.dsl.stdlib.base_lang import LOWER_RULES

ALL_OPS = {**CORE_OPS, **ABI_OPS}


def make_shader(cx, cy, gain):
    center = (cx, cy)

    @jit()
    def bright(v):
        return v * gain

    @jit()
    def shader(x):
        return bright(x - cy) + center[1]

    return shader


# --- leaf entries: static, from types alone -----------------------------------


def test_leaf_entries_scalar_and_tuple():
    assert BUILTINS.leaf_entries(T.f64) == (((), ScalarLeaf("f64")),)
    nested = T.Tuple((T.f64, T.Tuple((T.i64, T.boolean))))
    assert BUILTINS.leaf_entries(nested) == (
        ((0,), ScalarLeaf("f64")),
        ((1, 0), ScalarLeaf("i64")),
        ((1, 1), ScalarLeaf("bool")),
    )


def test_fntype_walker_is_the_envleaf_recursion():
    sh = make_shader(0.1, 0.2, 3.0)
    entries = BUILTINS.leaf_entries(sh.fntype)
    # env order: bright (whose env is gain), center, cy — nested paths prefixed
    paths = [p for p, _ in entries]
    assert (0, 0) in paths  # gain, THROUGH the captured kernel
    assert all(isinstance(leaf, ScalarLeaf) for _, leaf in entries)
    assert len(entries) == 4  # gain, center[0], center[1], cy


def test_unregistered_type_is_loud():
    class Odd(T.Type):
        pass

    with pytest.raises(TypeError, match="no leaves rule"):
        BUILTINS.leaf_entries(Odd())


# --- the alignment law: flatten <-> leaf_entries, fuzz-enforced ----------------


def _random_value(rng, depth=0):
    choices = [
        lambda: rng.uniform(-9, 9),
        lambda: rng.randrange(-(2**40), 2**40),
        lambda: rng.randrange(2**63, 2**64),  # the u64 bucket: `<q` would OVERFLOW on these
        lambda: rng.random() < 0.5,  # bool is an int subclass — the leaf kind must still say `bool`
    ]
    if depth < 3:
        choices.append(lambda: tuple(_random_value(rng, depth + 1) for _ in range(rng.randrange(3))))
    return rng.choice(choices)()


def test_alignment_law_fuzz():
    """flatten and leaf_entries must agree in COUNT, ORDER, and KIND. Packing
    each leaf with the format its declared kind implies is what makes kind
    drift visible: a bool that flattened as i64 (bool subclasses int, so a
    naive `<q` pack would happily accept it) fails here, as does a u64 value
    landing in a signed slot."""
    rng = random.Random(20260712)
    seen: set[str] = set()
    for _ in range(400):
        v = _random_value(rng)
        flat = BUILTINS.flatten(v)
        entries = BUILTINS.leaf_entries(BUILTINS.typeof(v))
        assert len(flat) == len(entries)
        for value, (_, leaf) in zip(flat, entries):
            struct.pack(_FMTS[leaf.kind], value)  # the DECLARED format, not a permissive stand-in
            seen.add(leaf.kind)
    assert {"f64", "i64", "u64", "bool"} <= seen  # the fuzz actually reached every builtin leaf kind


def test_handle_flatten_aligns_with_fntype_walker():
    sh = make_shader(0.5, 0.25, 2.0)
    flat = BUILTINS.flatten(sh)
    assert flat == (2.0, 0.5, 0.25, 0.25)  # gain, center[0], center[1], cy
    assert len(flat) == len(BUILTINS.leaf_entries(sh.fntype))


# --- plans and the generic packer ----------------------------------------------


def test_plan_from_types_alone_offsets_and_determinism():
    env = (T.f64, T.Tuple((T.i64, T.boolean)))
    plan = plan_from_types(env, (T.f32,), BUILTINS)
    assert plan == plan_from_types(env, (T.f32,), BUILTINS)  # types in, same plan out
    assert [s.dest.offset for s in plan.slots] == [0, 8, 16, 17]
    assert plan.staging_size == 21
    assert plan.slots[3].source == LeafPath("arg", 0, ())


def test_pack_into_bytes_and_strict_mismatch():
    plan = plan_from_types((T.f64, T.boolean), (), BUILTINS)
    staging = bytearray(plan.staging_size)
    leaves = pack_into(plan, staging, (1.5, True))
    assert leaves == ()
    assert struct.unpack_from("<d", staging, 0)[0] == 1.5
    assert struct.unpack_from("<?", staging, 8)[0] is True
    with pytest.raises(ValueError):
        pack_into(plan, staging, (1.5,))  # slot/value drift is loud, never silent


def test_convert_is_the_units_seat():
    plan = plan_from_types((T.f64,), (), BUILTINS)
    spec = plan.slots[0]
    mm_to_inch = type(spec)(spec.source, lambda v: v / 25.4, spec.dest)
    staging = bytearray(plan.staging_size)
    pack_into(type(plan)((mm_to_inch,), plan.staging_size), staging, (254.0,))
    assert struct.unpack_from("<d", staging, 0)[0] == 10.0


def test_compiled_extractor_reads_nested_env_by_path():
    sh = make_shader(0.5, 0.25, 2.0)
    plan = plan_from_types(sh.env_types, (T.f64,), BUILTINS)
    extract = build_extractor(sh.env_types, (T.f64,), plan, BUILTINS)
    assert extract(sh.captures, (7.0,)) == (2.0, 0.5, 0.25, 0.25, 7.0)


def test_compiled_extractor_agrees_with_flatten():
    """Two roads to the same leaves: `flatten` (the reference semantics the
    alignment-law fuzz checks) and the compiled per-slot getters (the hot
    path). They must never disagree — that would be silent byte corruption."""
    for sh in (make_shader(0.5, 0.25, 2.0), make_shader(-1.0, 3.5, 0.0)):
        plan = plan_from_types(sh.env_types, (), BUILTINS)
        compiled = build_extractor(sh.env_types, (), plan, BUILTINS)
        assert compiled(sh.captures, ()) == BUILTINS.flatten(sh)


def test_compiled_extractor_handles_pipelines_and_records():
    from pdum.dsl.combinators import op, register_composition, register_role

    register_role("device")
    register_composition("pipe", "device", "device", "fuse")

    @op
    def stage(k):
        @jit()
        def go(x):
            return x + k

        return go

    p = stage(1.0) | stage(2.0)  # FnType descends via .captures — Handle OR Pipeline
    plan = plan_from_types((p.fntype,), (), BUILTINS)
    extract = build_extractor((p.fntype,), (), plan, BUILTINS)
    assert extract((p,), ()) == (1.0, 2.0) == BUILTINS.flatten(p)


# --- the ABI stages -------------------------------------------------------------


def lowered(sh):
    # ch08's Subscript stand-in retired: step 10 promoted it into the base pack.
    return lower_handle(sh, LOWER_RULES, CORE_OPS, arg_types=(T.f64,))


def test_normalize_folds_extract_into_env_path():
    region = lowered(make_shader(0.5, 0.25, 2.0))
    normed = run_stage(region, NORMALIZE_ENV, ALL_OPS)
    envs = {dict(n.attrs)["slot"] for n in _walk(normed) if n.op == "core.env"}
    assert (1, 1) in envs  # center[1] became env path (1, 1)
    assert not any(n.op == "core.extract" and n.args[0].op == "core.env" for n in _walk(normed))


def test_legalize_emits_abi_slots_and_passes_legality():
    sh = make_shader(0.5, 0.25, 2.0)
    normed = run_stage(lowered(sh), NORMALIZE_ENV, ALL_OPS)
    plan = plan_from_types(sh.env_types, (T.f64,), BUILTINS)
    final = run_stage(normed, legalize_params(plan), ALL_OPS)  # legality {core, abi} enforced inside
    slots = [n for n in _walk(final) if n.op == "abi.slot"]
    assert slots and all(isinstance(dict(n.attrs)["offset"], int) for n in slots)
    assert not any(n.op == "core.env" for n in _walk(final))


def test_whole_composite_capture_is_refused():
    center = (1.0, 2.0)

    @jit()
    def uses_whole(x):
        return x + center[0] + center[1]

    region = lowered(uses_whole)  # normalize NOT run: env keeps a composite use
    plan = plan_from_types(uses_whole.env_types, (T.f64,), BUILTINS)
    with pytest.raises(VerifyError, match="composite|no slot"):
        run_stage(region, legalize_params(plan), ALL_OPS)


def test_a_surviving_core_env_is_caught_by_forbid_not_by_the_namespace_set():
    """The elimination is machine-checked, not merely a property of the rule
    set. `legal={core, abi}` CANNOT catch this — core.env is core — so a rule
    that ever became partial would silently resurrect per-frame flatten."""
    b = Builder(CORE_OPS)
    env = b.emit("core.env", type=T.f64, slot=(0,))
    region = Region(body=(b.emit("core.yield", env),))

    lax = Stage("pretend-legalize", [], legal=frozenset({"core", "abi"}))
    run_stage(region, lax, ALL_OPS)  # namespace target alone: passes, env intact

    strict = Stage("pretend-legalize", [], legal=frozenset({"core", "abi"}), forbid=frozenset({"core.env"}))
    with pytest.raises(VerifyError, match="survived the stage"):
        run_stage(region, strict, ALL_OPS)


def test_captured_kernel_as_a_value_says_so_instead_of_misdirecting():
    """A captured *Pipeline* is marshalable (it types as FnType and flattens)
    but lowering has no callee fate for it, so it lands as an FnType-typed
    core.env. NORMALIZE_ENV can never fold that — there is no extract-of-a-
    kernel — so the error must not send the user to that stage."""
    from pdum.dsl.combinators import op, register_composition, register_role

    register_role("device")
    register_composition("pipe", "device", "device", "fuse")

    @op
    def stage(k):
        @jit()
        def go(x):
            return x + k

        return go

    pipe = stage(1.0) | stage(2.0)

    @jit()
    def captures_a_pipeline(x):
        return pipe  # named, never called

    region = lower_handle(captures_a_pipeline, LOWER_RULES, CORE_OPS, arg_types=(T.f64,))
    plan = plan_from_types(captures_a_pipeline.env_types, (T.f64,), BUILTINS)
    with pytest.raises(VerifyError, match="captured kernel used as a value"):  # NOT "run NORMALIZE_ENV"
        run_stage(region, legalize_params(plan), ALL_OPS)


def test_child_and_leaves_aspects_agree_under_fuzz():
    """The compiled getters descend by the `child` aspect; the plan was built by
    the `leaves` aspect. A kind whose two aspects disagree writes the WRONG VALUE
    INTO THE RIGHT SLOT — no count check can see it. Fuzz the pair."""
    rng = random.Random(20260713)
    for _ in range(200):
        v = _random_value(rng)
        t = BUILTINS.typeof(v)
        plan = plan_from_types((t,), (), BUILTINS)
        compiled = build_extractor((t,), (), plan, BUILTINS)
        assert compiled((v,), ()) == BUILTINS.flatten(v)  # same values, same ORDER


def test_value_change_moves_bytes_not_plan_or_key():
    a, b = make_shader(0.5, 0.25, 2.0), make_shader(0.9, 0.75, 8.0)
    plan_a = plan_from_types(a.env_types, (T.f64,), BUILTINS)
    plan_b = plan_from_types(b.env_types, (T.f64,), BUILTINS)
    assert plan_a == plan_b  # same types, same plan — values shape nothing
    ra = run_stage(run_stage(lowered(a), NORMALIZE_ENV, ALL_OPS), legalize_params(plan_a), ALL_OPS)
    rb = run_stage(run_stage(lowered(b), NORMALIZE_ENV, ALL_OPS), legalize_params(plan_b), ALL_OPS)
    assert ra.key == rb.key  # artifact identity: value-free all the way to ABI
    sa, sb = bytearray(plan_a.staging_size), bytearray(plan_b.staging_size)
    extract = build_extractor(a.env_types, (T.f64,), plan_a, BUILTINS)  # ONE extractor, both entries
    pack_into(plan_a, sa, extract(a.captures, (0.0,)))
    pack_into(plan_b, sb, extract(b.captures, (0.0,)))
    assert sa != sb  # ...and the bytes are where the values went


# --- the output mirror ------------------------------------------------------------


def test_result_roundtrip_scalar_and_tuple():
    rp = result_plan(T.Tuple((T.f64, T.Tuple((T.i64, T.boolean)))), BUILTINS)
    buf = bytearray(rp.size)
    struct.pack_into("<d", buf, 0, 3.5)
    struct.pack_into("<q", buf, 8, -7)
    struct.pack_into("<?", buf, 16, True)
    assert unpack_result(rp, buf, BUILTINS) == (3.5, (-7, True))
    assert unpack_result(result_plan(T.f64, BUILTINS), struct.pack("<d", 2.25), BUILTINS) == 2.25


def test_unflatten_unknown_type_is_loud():
    with pytest.raises(TypeError, match="rebuild"):
        unflatten(T.FnType(T.Base(test_unflatten_unknown_type_is_loud.__code__), ()), (1,), BUILTINS)


def test_aspects_are_layerable_like_kinds():
    """The registries share one layering story: a child table overrides an
    aspect without touching the parent (valuekind's extend() contract)."""
    child = BUILTINS.extend()
    child.register_aspect("leaves", T.Scalar, lambda t, table: (((), ScalarLeaf("f32")),))  # narrow everything
    assert child.leaf_entries(T.f64) == (((), ScalarLeaf("f32")),)
    assert BUILTINS.leaf_entries(T.f64) == (((), ScalarLeaf("f64")),)  # parent untouched


def _walk(region):
    seen, out = set(), []

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        out.append(n)
        for a in n.args:
            visit(a)
        for r in n.regions:
            for m in r.body:
                visit(m)

    for n in region.body:
        visit(n)
    return out


def test_extract_now_types_tuples_too():
    b = Builder(CORE_OPS)
    env = b.emit("core.env", type=T.Tuple((T.f64, T.i64)), slot=(0,))
    assert b.emit("core.extract", env, index=1).type == T.i64
    with pytest.raises(TypeError, match="cannot extract"):
        b.emit("core.extract", env, index=2)
