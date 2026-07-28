"""The ops-count model: named buckets, cost models, MAC fusion."""

import numpy as np
from pdum.tl import Tensor, defmarker, defreducer, ops_count, pointwise, red, reduce, scan
from pdum.tl.lifting import lift_step
from pdum.tl.mdsl import exp as sym_exp


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


M_, K_, N_ = 3, 4, 5


def _matmul_step(x, w):
    xr = x.repeat("n", (0, N_))
    wr = w.repeat("m", (0, M_))
    p = xr * wr
    y = reduce(red.sum, p, ("k",))
    return y


def _matmul():
    x = T(np.zeros((M_, K_)), ("m", "k"))
    w = T(np.zeros((K_, N_)), ("k", "n"))
    return lift_step(_matmul_step, x=x.layout, w=w.layout)


def test_matmul_counts_muls_and_adds_separately():
    ls = _matmul()
    ops = ops_count(ls.region, names=ls.names)
    assert ops.total["mul"] == M_ * K_ * N_
    assert ops.total["add"] == M_ * N_ * (K_ - 1)


def test_mac_fusion_recognizes_the_contraction_pattern():
    ls = _matmul()
    ops = ops_count(ls.region, fuse_mac=True, names=ls.names)
    assert ops.total["mac"] == M_ * K_ * N_
    assert ops.total["mul"] == 0 and ops.total["add"] == 0


def test_mac_fusion_refuses_when_the_product_is_observed_elsewhere():
    def step(x, w):
        xr = x.repeat("n", (0, N_))
        wr = w.repeat("m", (0, M_))
        p = xr * wr
        y = reduce(red.sum, p, ("k",))
        p2 = p + p  # a second consumer observes the product
        return y, p2

    x = T(np.zeros((M_, K_)), ("m", "k"))
    w = T(np.zeros((K_, N_)), ("k", "n"))
    ls = lift_step(step, x=x.layout, w=w.layout)
    ops = ops_count(ls.region, fuse_mac=True, names=ls.names)
    assert ops.total["mac"] == 0  # p has two consumers; no fusion


def test_composite_pointwise_counts_by_tree():
    sig = defmarker("sig_oc", 1, lambda x: 1 / (1 + sym_exp(-x)))
    n = 4

    def step(x):
        s = pointwise(sig, x)
        return s

    ls = lift_step(step, x=T(np.zeros(n), ("i",)).layout)
    ops = ops_count(ls.region, names=ls.names)
    assert ops.per_var["s"] == {"div": n, "add": n, "exp": n, "neg": n}
    # exp's cost is the model's opinion, not the count's
    assert ops.weighted({"exp": 20.0}) == 3 * n + 20.0 * n


def test_scan_counts_folds_not_elements():
    def step(x):
        s = scan(red.sum, x, "t")
        return s

    ls = lift_step(step, x=T(np.zeros((3, 5)), ("b", "t")).layout)
    ops = ops_count(ls.region, names=ls.names)
    assert ops.total["add"] == 3 * (5 - 1)


def test_composite_scan_counts_lift_combine_project():
    lr = defreducer(
        "lr_oc",
        state=2,
        element=2,
        lift=lambda a, b: (a, b),
        combine=lambda left, right: (left[0] * right[0], right[0] * left[1] + right[1]),
        init=(1.0, 0.0),
        project=lambda A, B: B,
    )
    n = 6

    def step(a, b):
        h = scan(lr, (a, b), "t")
        return h

    ls = lift_step(step, a=T(np.zeros(n), ("t",)).layout, b=T(np.zeros(n), ("t",)).layout)
    ops = ops_count(ls.region, names=ls.names)
    # combine = 2 muls + 1 add per fold; identity lift/project cost nothing
    assert ops.per_var["h"] == {"mul": 2 * (n - 1), "add": n - 1}


def test_materialize_counts_copies_in_their_own_bucket():
    # materialize has no surface spelling in a step body — author the
    # region directly (the Builder door the naming law provides for)
    from pdum.dsl.ir import Builder, Region
    from pdum.dsl.ops import CORE_OPS
    from pdum.tl.dialect import TL_OPS, region_names, tensor_type_of_layout

    b = Builder({**CORE_OPS, **TL_OPS})
    xp = b.param(0, tensor_type_of_layout(T(np.zeros(7), ("i",)).layout))
    m = b.emit("tl.materialize", xp, order=("i",))
    region = Region(params=(xp,), body=(b.emit("core.yield", m),))
    ops = ops_count(region, names=region_names(region, ("x",), {id(m): "m"}))
    assert ops.total == {"copy": 7}
    assert ops.weighted({"copy": 0.0}) == 0.0


# --- the Region face (the excavation, LEVELS) -------------------------------


def test_region_face_counts_tiled_gemm_exactly():
    """The flagship count: the tiled-matmul zoo shape — tiling is layout,
    not semantics, so the totals are the plain matmul's, with the standard
    m*n*k MAC figure under fuse_mac."""
    from pdum.tl.zoo.gemm import GemmConfig, make_tiled_matmul

    cfg = GemmConfig()
    a = T(np.zeros((cfg.m, cfg.k)), ("m", "k"))
    b = T(np.zeros((cfg.k, cfg.n)), ("k", "n"))
    ls = lift_step(make_tiled_matmul(cfg), a=a.layout, b=b.layout)
    ro = ops_count(ls.region, names=ls.names)
    assert ro.total == {"mul": cfg.m * cfg.n * cfg.k, "add": cfg.m * cfg.n * (cfg.k - 1)}
    rf = ops_count(ls.region, fuse_mac=True, names=ls.names)
    assert rf.total == {"mac": cfg.m * cfg.n * cfg.k}


def test_region_face_yielded_reduce_still_fuses():
    """A yielded reduce consumes its product like any consumer — outputs are
    not consumption, but the reduce itself is."""

    def step(x, w):
        return reduce(red.sum, x * w, "k")

    x, w = T(np.zeros(4), ("k",)), T(np.zeros(4), ("k",))
    ls = lift_step(step, x=x.layout, w=w.layout)
    ro = ops_count(ls.region, fuse_mac=True, names=ls.names)
    assert ro.total["mac"] == 4 and "mul" not in ro.total and "add" not in ro.total
