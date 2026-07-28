"""The at-kink law, pinned at the kink points (200 §S.2 — frozen contract).

The derivative table is one-sided and PARTITIONS: at a tie exactly one
operand receives the cotangent — first-wins. Reduce adjoints derive through
the pairwise combine and inherit the partition. These are literal
expectations; a drifted selection is an API break.
"""

import numpy as np
from pdum.tl import Tensor, pointwise, red, reduce
from pdum.tl.autodiff import grad
from pdum.tl.dialect import run_named, run_region
from pdum.tl.lifting import lift_step
from pdum.tl.markers import maximum, minimum


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def _grad_of(step, target, inputs):
    ls = lift_step(step, **{k: v.layout for k, v in inputs.items()})
    rg = grad(ls.region, target, inputs, names=ls.names)
    env = run_named(rg.region, inputs, rg.names)
    return {v: (None if gn is None else env[gn]) for v, gn in rg.grads.items()}


def test_pointwise_maximum_tie_goes_left():
    def step(a, b):
        y = pointwise(maximum, a, b)
        s = reduce(red.sum, y, "i")
        return s

    inputs = {"a": T([2.0, 5.0, 1.0], ("i",)), "b": T([2.0, 3.0, 4.0], ("i",))}
    g = _grad_of(step, "s", inputs)
    np.testing.assert_allclose(g["a"].to_numpy(), [1.0, 1.0, 0.0])  # the tie at i=0 goes LEFT
    np.testing.assert_allclose(g["b"].to_numpy(), [0.0, 0.0, 1.0])  # never double-counted


def test_pointwise_minimum_tie_goes_left():
    def step(a, b):
        y = pointwise(minimum, a, b)
        s = reduce(red.sum, y, "i")
        return s

    inputs = {"a": T([2.0, 5.0], ("i",)), "b": T([2.0, 3.0], ("i",))}
    g = _grad_of(step, "s", inputs)
    np.testing.assert_allclose(g["a"].to_numpy(), [1.0, 0.0])
    np.testing.assert_allclose(g["b"].to_numpy(), [0.0, 1.0])


def test_reduce_max_all_ties_one_winner():
    def step(x):
        m = reduce(red.max, x, "i")
        return m

    g = _grad_of(step, "m", {"x": T([7.0, 7.0, 7.0], ("i",))})
    np.testing.assert_allclose(g["x"].to_numpy(), [1.0, 0.0, 0.0])  # first along i, alone


def test_reduce_min_tie_first_wins():
    def step(x):
        m = reduce(red.min, x, "i")
        return m

    g = _grad_of(step, "m", {"x": T([4.0, 1.0, 1.0], ("i",))})
    np.testing.assert_allclose(g["x"].to_numpy(), [0.0, 1.0, 0.0])


def test_reduce_max_two_dims_lexicographic_first():
    """Multi-dim reduce derives as a chain of single-dim reduces in declared
    order — the winner of an all-tie square is the lexicographically first
    element, and the partition never splits the cotangent."""

    def step(x):
        m = reduce(red.max, x, ("i", "j"))
        return m

    g = _grad_of(step, "m", {"x": T(np.full((2, 2), 3.0), ("i", "j"))})
    gx = g["x"].to_numpy(order=("i", "j"))
    np.testing.assert_allclose(gx, [[1.0, 0.0], [0.0, 0.0]])
    assert float(np.sum(gx)) == 1.0  # a partition: total mass is exactly one


def test_reduce_max_gradient_mass_is_conserved():
    """Sum of the gradient equals the seed regardless of tie structure —
    the partition law's conservation face."""

    def step(x):
        m = reduce(red.max, x, "i")
        return m

    for data in ([3.0, 3.0, 1.0], [1.0, 3.0, 3.0], [3.0, 1.0, 3.0]):
        g = _grad_of(step, "m", {"x": T(data, ("i",))})
        assert float(np.sum(g["x"].to_numpy())) == 1.0


def test_the_one_table_has_one_home():
    """pdum.tl re-exports the SAME objects as pdum.dsl.derivative — one
    table, one schema, both tiers (the one-body-language law made literal;
    the kink rows this file pins are those rows)."""
    import pdum.dsl.derivative as dsl_d
    import pdum.tl.derivative as tl_d
    import pdum.tl.nodes as tl_n

    assert tl_d.TABLE is dsl_d.TABLE and tl_d.diff is dsl_d.diff
    assert tl_n.Prim is dsl_d.Prim and tl_n.Const is dsl_d.Const and tl_n.Arg is dsl_d.Arg
    import pdum.dsl.markers as dsl_m
    import pdum.tl.markers as tl_m

    assert tl_m.Marker is dsl_m.Marker and tl_m.pw is dsl_m.pw  # numpy-authority: one vocabulary
    assert tl_m.sqrt is dsl_m.sqrt and tl_m.maximum is dsl_m.maximum


# --- the Region face (the excavation, LEVELS) -------------------------------


def test_region_grad_inherits_the_partition_law():
    """THE at-kink gate for the region face: heavy ties in a max-reduce —
    exactly one element per reduced line receives the cotangent
    (first-wins), pinned to the literal partition."""

    def step(x):
        m = reduce(red.max, x, "i")
        s = reduce(red.sum, m, "j")
        return s

    arr = np.array([[1.0, 3.0, 3.0, 2.0], [5.0, 5.0, 5.0, 5.0], [2.0, 2.0, 7.0, 7.0]])
    x = Tensor.from_numpy(arr, ("j", "i"))
    ls = lift_step(step, x=x.layout)
    rg = grad(ls.region, ls.outputs[0], {"x": x}, names=ls.names)
    _, got = run_region(rg.region, [x])
    got = got.to_numpy(order=("j", "i"))
    np.testing.assert_array_equal(got.sum(axis=1), np.ones(3))  # mass conserved
    np.testing.assert_array_equal(got, np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=float))
