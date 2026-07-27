"""Step 4 — the IR: programs are values; the anti-pattern is unrepresentable."""

import dataclasses
import typing

import pytest
from pdum.dsl import types as T
from pdum.dsl.ir import Builder, CallLoc, FusedLoc, Loc, Node, Region, VerifyError, format_loc, verify
from pdum.dsl.ops import CORE_OPS, UNIT
from pdum.dsl.printer import print_program

b = Builder(CORE_OPS)


def disk_body():
    """The disk-shader body, built by hand: (x-cx)^2 + (y-cy)^2 < r^2."""
    x = b.param(0, T.f64)
    y = b.param(1, T.f64)
    cx = b.emit("core.env", type=T.f64, slot=0)
    cy = b.emit("core.env", type=T.f64, slot=1)
    r = b.emit("core.env", type=T.f64, slot=2)
    dx = b.emit("core.sub", x, cx)
    dy = b.emit("core.sub", y, cy)
    d2 = b.emit("core.add", b.emit("core.mul", dx, dx), b.emit("core.mul", dy, dy))
    hit = b.emit("core.cmp", d2, b.emit("core.mul", r, r), pred="lt")
    return Region(params=(x, y), body=(b.emit("core.yield", hit),))


# --- the anti-pattern gate (architecture §6 gate 9) ---------------------------


def test_no_value_shaped_fields_reachable_from_node():
    # No field on Node/Region/Loc may be typed to hold an arbitrary runtime
    # value; `attrs` is the single, deliberate carve-out (the const/Literal
    # slot). This is the caching thesis enforced by field annotations.
    for cls in (Node, Region, Loc, CallLoc, FusedLoc):  # provenance is reachable from Node.loc
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            if f.name in ("attrs", "_key"):
                continue
            assert "object" not in str(hints[f.name]), f"{cls.__name__}.{f.name} can hold a value"
            assert "Any" not in str(hints[f.name]), f"{cls.__name__}.{f.name} can hold a value"


def test_env_carries_a_slot_never_a_value():
    env = b.emit("core.env", type=T.f64, slot=3)
    assert env.attrs == (("slot", 3),)
    with pytest.raises(TypeError):
        Node(op="core.env", type=T.f64, value=5)  # no such field exists


# --- content identity -----------------------------------------------------------


def test_content_key_stable_across_rebuilds():
    assert disk_body().key == disk_body().key
    assert len(disk_body().key) == 32  # sha256


def test_content_key_sensitive_to_structure():
    base = disk_body().key
    tweaked = Builder(CORE_OPS)
    x, y = tweaked.param(0, T.f64), tweaked.param(1, T.f64)
    cx = tweaked.emit("core.env", type=T.f64, slot=0)
    body = tweaked.emit("core.yield", tweaked.emit("core.add", x, cx))
    assert Region((x, y), (body,)).key != base


def test_loc_is_excluded_from_identity():
    n1 = b.emit("core.add", b.param(0, T.f64), b.param(0, T.f64), loc=Loc("a.py", 1))
    n2 = b.emit("core.add", b.param(0, T.f64), b.param(0, T.f64), loc=Loc("b.py", 99))
    assert n1 == n2 and n1.key == n2.key  # where code came from is not what it is


def test_provenance_algebra_is_outside_identity():
    la, lb = Loc("wave.py", 5), Loc("art.py", 40)
    n1 = b.emit("core.add", b.param(0, T.f64), b.param(0, T.f64), loc=CallLoc(la, lb))
    n2 = b.emit("core.add", b.param(0, T.f64), b.param(0, T.f64), loc=FusedLoc((la, lb)))
    assert n1 == n2 and n1.key == n2.key  # provenance never touches identity
    assert format_loc(CallLoc(la, lb)) == "wave.py:5 (inlined from art.py:40)"
    assert format_loc(FusedLoc((la, lb))) == "{wave.py:5, art.py:40}"


def test_type_errors_carry_source_points():
    i = b.emit("core.env", type=T.i64, slot=0, loc=Loc("art.py", 11))
    f = b.emit("core.env", type=T.f64, slot=1, loc=Loc("art.py", 12))
    with pytest.raises(TypeError, match=r"strict.*art\.py:13; art\.py:11; art\.py:12"):
        b.emit("core.add", i, f, loc=Loc("art.py", 13))


def test_attrs_are_canonically_sorted():
    n1 = b.emit("core.env", type=T.f64, slot=1, role="uniform")
    n2 = b.emit("core.env", type=T.f64, role="uniform", slot=1)
    assert n1.attrs == n2.attrs == (("role", "uniform"), ("slot", 1))
    assert n1.key == n2.key


# --- type rules ------------------------------------------------------------------


def test_core_arithmetic_is_strict():
    i, f = b.emit("core.env", type=T.i64, slot=0), b.emit("core.env", type=T.f64, slot=1)
    assert b.emit("core.add", i, i).type == T.i64
    with pytest.raises(TypeError, match="strict"):
        b.emit("core.add", i, f)  # NO promotion in the kernel — a dialect's lowering may insert casts
    widened = b.emit("core.cast", i, to=T.f64)
    assert b.emit("core.add", widened, f).type == T.f64  # the Julia way: cast, then same-type add
    with pytest.raises(TypeError, match="strict"):
        b.emit("core.cmp", i, f, pred="lt")
    assert b.emit("core.cmp", widened, f, pred="lt").type == T.boolean


def test_select_checks_its_types():
    i, f = b.emit("core.env", type=T.i64, slot=0), b.emit("core.env", type=T.f64, slot=1)
    cond = b.emit("core.cmp", i, i, pred="eq")
    assert b.emit("core.select", cond, f, f).type == T.f64
    with pytest.raises(TypeError, match="disagree"):
        b.emit("core.select", cond, i, f)
    with pytest.raises(TypeError, match="bool"):
        b.emit("core.select", i, f, f)


def test_vec_extract_field_cast():
    f = b.emit("core.env", type=T.f64, slot=0)
    v = b.emit("core.vec", f, f, f)
    assert v.type == T.Vec(T.f64, 3)
    assert b.emit("core.extract", v, index=2).type == T.f64
    with pytest.raises(TypeError):
        b.emit("core.extract", v, index=3)
    color = b.emit("core.env", type=T.Record("Color", (("r", T.f32), ("g", T.f32))), slot=1)
    assert b.emit("core.field", color, name="g").type == T.f32
    assert b.emit("core.cast", f, to=T.f32).type == T.f32


def test_region_ops_type_from_their_regions():
    f = b.emit("core.env", type=T.f64, slot=0)
    cond = b.emit("core.cmp", f, f, pred="lt")
    then = Region(body=(b.emit("core.yield", f),))
    other = Region(body=(b.emit("core.yield", b.emit("core.neg", f)),))
    node = b.emit("core.if", cond, regions=(then, other))
    assert node.type == T.f64
    with pytest.raises(TypeError, match="yield"):
        i = b.emit("core.env", type=T.i64, slot=1)
        b.emit("core.if", cond, regions=(then, Region(body=(b.emit("core.yield", i),))))
    assert b.emit("core.store", f, f).type == UNIT


# --- construction discipline ------------------------------------------------------


def test_builder_is_loud():
    f = b.emit("core.env", type=T.f64, slot=0)
    with pytest.raises(VerifyError, match="unknown op"):
        b.emit("core.frobnicate", f)
    with pytest.raises(VerifyError, match="explicit type"):
        b.emit("core.env", slot=0)  # env has no computable type
    with pytest.raises(VerifyError, match="region"):
        b.emit("core.if", f)  # missing its two regions


def test_verify_demands_terminated_regions():
    f = b.emit("core.env", type=T.f64, slot=0)
    with pytest.raises(VerifyError, match="yield"):
        verify(Region(body=(f,)), CORE_OPS)
    verify(disk_body(), CORE_OPS)  # the real thing passes


# --- the printer -------------------------------------------------------------------


def test_printer_golden_and_dag_sharing():
    prog = disk_body()
    text = print_program(prog, name="disk")
    assert text.startswith("disk(%p0: f64, %p1: f64) {")
    assert text.count("core.env {slot = 0}") == 1  # one definition line per node
    assert "{pred = 'lt'}" in text
    assert text.rstrip().endswith("}")
    # shared subexpression: dx*dx and dy*dy each define once, referenced once
    shared = Builder(CORE_OPS)
    e = shared.emit("core.env", type=T.f64, slot=0)
    sq = shared.emit("core.mul", e, e)
    total = shared.emit("core.add", sq, sq)  # the SAME sq object, twice
    text2 = print_program(Region(body=(shared.emit("core.yield", total),)))
    assert text2.count("core.mul") == 1  # defined once — sharing is visible
