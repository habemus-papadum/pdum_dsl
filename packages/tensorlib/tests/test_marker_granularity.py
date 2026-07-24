"""The marker-granularity gate (200 §S.2) — a HARD gate.

A marker (gelu's formula) and a reducer's combine are small named bodies the
AD machinery differentiates BY INSPECTION. The producer must lower one
marker to one named, inspectable body tree over primitives (captured
constants become Consts), and one combine to the same inside its structured
declaration — never inlined away, never an opaque call. The flash
derived-backward test (test_zoo) enforces the payoff; this file pins the
mechanism.
"""

import pytest
from pdum.tl import defmarker
from pdum.tl.derivative import TABLE
from pdum.tl.mdsl import exp, tanh
from pdum.tl.nodes import Arg, Const, Prim
from pdum.tl.registry import MARKERS
from pdum.tl.zoo.attention import flashsm

GELU_C = 0.7978845608028654


def _walk(node):
    yield node
    if isinstance(node, Prim):
        for a in node.args:
            yield from _walk(a)


def _gelu(x):
    return 0.5 * x * (1 + tanh(GELU_C * (x + 0.044715 * x * x * x)))


def test_one_marker_is_one_named_inspectable_tree_over_primitives():
    m = defmarker("gate.gelu", 1, _gelu)
    assert m.name in MARKERS  # named: programs reference it by this string
    ops = {n.op for n in _walk(m.body) if isinstance(n, Prim)}
    assert ops <= set(TABLE)  # every op is a primitive with a table entry
    consts = {n.value for n in _walk(m.body) if isinstance(n, Const)}
    assert GELU_C in consts  # the captured constant became a Const
    assert all(isinstance(n, (Prim, Const, Arg)) for n in _walk(m.body))


def test_the_combine_is_inspectable_inside_its_declaration():
    """flash's backward exists BECAUSE the combine is such a body: its
    component markers are trees over primitives whose partials derive."""
    cs, ls, p = flashsm.component_markers()
    for cm in (*cs, *ls, p):
        ops = {n.op for n in _walk(cm.body) if isinstance(n, Prim)}
        assert ops <= set(TABLE), cm.name
    d = cs[0].partial(0)  # ∂C_0/∂left_m — derived by rewriting, registered
    assert d.name == f"{cs[0].name}.d0" and d.name in MARKERS


def test_an_opaque_call_refuses_never_lowers():
    def helper(v):
        return exp(v)

    def body(x):
        return helper(x) + 1

    with pytest.raises(ValueError, match="opaque call would break derived differentiation"):
        defmarker("gate.opaque", 1, body)
