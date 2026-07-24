"""SCAFFOLDING (deleted with the tracer at the end of P4): the AST producer
and the Sym tracer must lower every body to the IDENTICAL Node tree. The
producer replaced the tracer as the one producer of marker bodies; this file
is the conversion's differential gate, not a lasting contract."""

from dataclasses import dataclass

import numpy as np
from pdum.tl import Tensor, defreducer, pointwise, reduce
from pdum.tl.mdsl import _trace_tuple, exp, log, maximum, sqrt, tanh, trace, where
from pdum.tl.producer import lower, scalars, tuple_binding

GELU_C = 0.7978845608028654

BODIES = [
    (1, lambda x: 1 / (1 + exp(-x))),
    (1, lambda x: log(1 + exp(x))),
    (1, lambda x: 0.5 * x * (1 + tanh(GELU_C * (x + 0.044715 * x * x * x)))),
    (1, lambda x: maximum(x, 0)),
    (1, lambda x: where(maximum(x, 0), x, -x)),
    (2, lambda a, b: sqrt(a * a + b * b)),
    (2, lambda a, b: a - 2 / b + 0.5),
]


def test_producer_and_tracer_agree_on_every_body():
    """DENOTATIONAL agreement over the whole battery. (Tree identity is too
    strong: the tracer's __radd__/__rmul__ COMMUTE operands — `1 + exp(x)`
    traced as add(exp(x), 1) — an artifact the faithful-to-source producer
    does not reproduce.)"""
    from pdum.tl.compute import _eval_tree

    rng = np.random.default_rng(3)
    for arity, fn in BODIES:
        p = lower(fn, scalars(arity))
        t = trace(fn, arity)
        args = [rng.standard_normal(9) for _ in range(arity)]
        np.testing.assert_allclose(_eval_tree(p[0], args), _eval_tree(t, args), rtol=1e-12)


def _hypot(a, b):
    return sqrt(a * a + b * b)


def _relu(x):
    return maximum(x, 0)


def test_producer_and_tracer_agree_exactly_without_reflected_operands():
    for arity, fn in [(2, _hypot), (1, _relu)]:
        assert lower(fn, scalars(arity)) == (trace(fn, arity),)


def test_producer_and_tracer_agree_on_tuple_state_bodies():
    def combine(left, right):
        return (left[0] * right[0], right[0] * left[1] + right[1])

    traced = _trace_tuple(lambda *a: combine(tuple(a[:2]), tuple(a[2:])), 4)
    assert lower(combine, tuple_binding(2) + tuple_binding(2, 2)) == traced


def test_statements_and_comparisons_lower_where_the_tracer_could_not():
    """The producer's genuine wins: assignments and comparison OPERATORS
    (the tracer needed ge(a, b) spelled as a call)."""

    def softclip(x):
        lo = 0.0
        hi = 1.0
        t = maximum(x, lo)
        return where(t >= hi, hi, t)

    body = lower(softclip, scalars(1))
    traced = trace(lambda x: where(_ge_call(maximum(x, 0.0), 1.0), 1.0, maximum(x, 0.0)), 1)
    assert body == (traced,)


def _ge_call(a, b):
    from pdum.tl.mdsl import ge

    return ge(a, b)


@dataclass(frozen=True)
class State:
    m: object
    den: object
    o: object


def test_record_state_reducer_matches_its_tuple_twin():
    """Record-typed reducer state (S.2): the record spelling and the tuple
    spelling of the SAME online-softmax combine produce identical scans."""

    def r_lift(s, v):
        return State(s, 1.0, v)

    def r_combine(L, R):
        m = maximum(L.m, R.m)
        sl, sr = exp(L.m - m), exp(R.m - m)
        return State(m, L.den * sl + R.den * sr, L.o * sl + R.o * sr)

    def r_project(s):
        return s.o / s.den

    rec = defreducer(
        "eq_flash_rec", state=State, element=2, lift=r_lift, combine=r_combine,
        init=State(-1e30, 0.0, 0.0), project=r_project,
    )
    tup = defreducer(
        "eq_flash_tup",
        state=3,
        element=2,
        lift=lambda s, v: (s, 1.0, v),
        combine=lambda L, R: (
            maximum(L[0], R[0]),
            L[1] * exp(L[0] - maximum(L[0], R[0])) + R[1] * exp(R[0] - maximum(L[0], R[0])),
            L[2] * exp(L[0] - maximum(L[0], R[0])) + R[2] * exp(R[0] - maximum(L[0], R[0])),
        ),
        init=(-1e30, 0.0, 0.0),
        project=lambda m, den, o: o / den,
    )
    rng = np.random.default_rng(5)
    s = Tensor.from_numpy(rng.standard_normal(6), ("i",))
    v = Tensor.from_numpy(rng.standard_normal(6), ("i",))
    a = reduce(rec, (s, v), ("i",)).to_numpy()
    b = reduce(tup, (s, v), ("i",)).to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-12)
    sm = np.exp(s.to_numpy() - s.to_numpy().max())
    np.testing.assert_allclose(a, (sm * v.to_numpy()).sum() / sm.sum(), rtol=1e-9)


def test_captured_constants_become_consts_in_the_body():
    from pdum.tl.nodes import Const, Prim

    k = 2.5

    def scaled(x):
        return k * x

    (body,) = lower(scaled, scalars(1))
    assert body == Prim("mul", (Const(2.5), __import__("pdum.tl.nodes", fromlist=["Arg"]).Arg(0)))


def test_relu_via_pointwise_still_evaluates():
    from pdum.tl.mdsl import defmarker

    m = defmarker("eq_relu", 1, lambda x: maximum(x, 0))
    x = np.array([-1.0, 0.0, 2.0])
    got = pointwise(m, Tensor.from_numpy(x, ("i",))).to_numpy()
    np.testing.assert_allclose(got, np.maximum(x, 0))
