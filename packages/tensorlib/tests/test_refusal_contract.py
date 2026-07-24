"""The joint refusal battery — seeded at P3 (200 §7, zoo gate 8).

Refusal messages are FROZEN BEHAVIOR: one shape — what happened, the
principle violated, the quoted fix (210). These tests pin the wording by
literal expectation; a drifted message is an API break, not a cleanup.
"""

import numpy as np
import pytest
from pdum.dsl.naming import NameCollision, Namer
from pdum.tl import Build, Tensor, defmarker, pointwise, pw
from pdum.tl.ir import Instr, Program, run
from pdum.tl.mdsl import exp
from pdum.tl.registry import RegistryConflict


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def test_the_shared_axis_extent_refusal():
    """Zoo gate 8: a shared dim with mismatched extents refuses with the
    diagnosis AND the exact slice() that fixes it — never silent broadcast."""
    with pytest.raises(
        ValueError,
        match=r"pointwise\(add\) requires aligned operands:\n"
        r"  operand 0, dim 'i': domain \[0, 3\) exceeds the common \[0, 2\)  ->  slice\(i=\(0, 2\)\)",
    ):
        pointwise(pw.add, T([1.0, 2.0, 3.0], ("i",)), T([1.0, 2.0], ("i",)))


def test_registry_conflict_refuses_and_names_the_fix():
    """A taken name with different content refuses — entries are immutable
    program vocabulary, never overwritten (the P3 cache-backed registry)."""
    defmarker("battery_scaled", 1, lambda x: 2 * x)
    with pytest.raises(
        RegistryConflict,
        match=r"marker 'battery_scaled' is already registered with different "
        r"content — names are program vocabulary and entries are immutable; "
        r"pick a fresh name \(or derive one from the content digest by "
        r"passing name=None\)",
    ):
        defmarker("battery_scaled", 1, lambda x: 3 * x)


def test_explicit_name_collision_refuses_never_suffixes():
    """The naming law's seed (200 §1.6): contract names are never
    auto-suffixed — a collision refuses and says so."""
    n = Namer()
    n.claim("wq")
    with pytest.raises(
        NameCollision,
        match=r"name 'wq' is already declared — contract names are never "
        r"auto-suffixed; declare it once, or address a different path",
    ):
        n.claim("wq")
    b = Build()
    b.input("x")
    with pytest.raises(NameCollision):
        b.input("x")


def test_unknown_marker_and_reducer_refuse_by_name():
    """Programs reference compute by NAME; an unregistered name refuses at
    the resolution seam, quoting the name."""
    prog = Program((Instr("a", "input"), Instr("y", "pointwise", ("a",), {"f": "no_such_marker"})))
    with pytest.raises(KeyError, match=r"unknown pointwise marker 'no_such_marker'"):
        run(prog, {"a": T([1.0], ("i",))})
    prog = Program((Instr("a", "input"), Instr("y", "reduce", ("a",), {"f": "no_such_reducer", "dims": ("i",)}))
    )
    with pytest.raises(KeyError, match=r"unknown reducer 'no_such_reducer'"):
        run(prog, {"a": T([1.0], ("i",))})


def test_primitive_names_stay_reserved():
    with pytest.raises(ValueError, match=r"'mul' is a primitive marker name"):
        defmarker("mul", 2, lambda a, b: exp(a) * b)
