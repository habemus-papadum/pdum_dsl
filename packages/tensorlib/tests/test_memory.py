"""L1 peak-memory simulator: aliases, closed forms, schedules."""

import numpy as np
import pytest
from pdum.tl import Tensor, iota, pointwise, red, reduce
from pdum.tl.autodiff import grad
from pdum.tl.compute import const_like
from pdum.tl.lifting import lift_step
from pdum.tl.markers import where
from pdum.tl.memory import peak_memory
from pdum.tl.zoo import gpt2, heat2d


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


X8 = {"x": T(np.zeros(8), ("i",))}  # 64 bytes


def test_views_are_free_and_keep_their_root_alive():
    def step(x):
        a = x * x  # 64
        s = a.slice(i=(0, 4))  # view: 0 bytes
        c = s * s  # 32; a must survive to here
        return c

    ls = lift_step(step, x=X8["x"].layout)
    r = peak_memory(ls.region, X8, names=ls.names)
    assert "s" not in r.alloc_bytes
    assert r.peak_bytes == 64 + 64 + 32 and r.peak_at == "c"
    rf = peak_memory(ls.region, X8, free_inputs=True, names=ls.names)
    assert rf.peak_bytes == 64 + 64  # x dies after a; peak is at a


def test_masks_and_positions_cost_nothing():
    def step(x):
        it = iota(x, "i")
        m = it < 4
        y = pointwise(where, m, x, const_like(x, 0.0))
        return y

    ls = lift_step(step, x=X8["x"].layout)
    r = peak_memory(ls.region, X8, names=ls.names)
    assert set(r.alloc_bytes) == {"x", "m", "y"}  # iota/consts absent entirely
    assert r.peak_bytes == 64 * 3


def test_the_schedule_moves_the_peak():
    def step(x):
        a = x * x  # 64
        ra = reduce(red.sum, a, "i")  # 8
        b = x + x  # 64
        rb = reduce(red.sum, b, "i")  # 8
        out = ra * rb  # 8
        return out

    ls = lift_step(step, x=X8["x"].layout)
    good = peak_memory(ls.region, X8, names=ls.names)
    bad = peak_memory(ls.region, X8, order=["x", "a", "b", "ra", "rb", "out"], names=ls.names)
    assert good.peak_bytes == 144  # x + ra + b + rb, at rb
    assert bad.peak_bytes == 200  # x + a + b + ra live together
    assert bad.peak_at == "ra"
    with pytest.raises(ValueError, match="topological"):
        peak_memory(ls.region, X8, order=["x", "ra", "a", "b", "rb", "out"], names=ls.names)


def test_fold_transient_counts_the_step():
    m = heat2d()
    r = peak_memory(m.region, m.inputs, names=m.names)
    u_bytes = 5 * 5 * 8
    assert r.peak_at == "uf"
    assert r.peak_bytes > 3 * u_bytes  # u0 + carry + step internals + out
    assert any(k == "(fold transient)" for k, _ in r.live_at_peak)


def test_gpt2_backward_needs_more_than_forward():
    m = gpt2()
    fwd = peak_memory(m.region, m.inputs, names=m.names)
    rg = grad(m.region, m.out, m.inputs, seed="dL", names=m.names)
    joint = peak_memory(rg.region, {**m.inputs, "dL": T(np.zeros((4, 5)), ("t", "v"))}, names=rg.names)
    assert fwd.peak_bytes > fwd.input_bytes  # activations dominate inputs? sanity
    assert joint.peak_bytes > fwd.peak_bytes
    assert joint.input_bytes == fwd.input_bytes + 4 * 5 * 8  # + the seed


# --- the Region face (the excavation, LEVELS) -------------------------------


def test_region_face_report_is_exact():
    """Const-free corpus: the full report, pinned — views are free, roots
    stay alive to the last alias use, and the default schedule is params
    then walk order."""

    def step(x, w):
        p = x * w
        s = p.slice(i=(0, 4))
        t = reduce(red.sum, s, "i")
        r = reduce(red.sum, p, "i")
        o = t + r
        return o

    x, w = T(np.zeros(8), ("i",)), T(np.zeros(8), ("i",))
    ls = lift_step(step, x=x.layout, w=w.layout)
    ro = peak_memory(ls.region, {"x": x, "w": w}, names=ls.names)
    assert (ro.peak_bytes, ro.peak_at) == (208, "r")  # x + w + p + t + r
    assert ro.live_at_peak == (("p", 64), ("r", 8), ("t", 8), ("w", 64), ("x", 64))
    assert ro.alloc_bytes == {"x": 64, "w": 64, "p": 64, "t": 8, "r": 8, "o": 8}
    assert ro.input_bytes == 128
    assert ro.timeline == (("x", 64), ("w", 128), ("p", 192), ("s", 192), ("t", 200), ("r", 144), ("o", 128))


def test_region_face_consts_never_move_the_needle():
    def step(x):
        y = x * 2.0
        z = y + 1.0
        return z

    x = T(np.zeros(8), ("i",))
    ls = lift_step(step, x=x.layout)
    ro = peak_memory(ls.region, {"x": x}, names=ls.names)
    # consts are deferred scalars: zero bytes, never scheduled
    assert [v for v, _ in ro.timeline] == ["x", "y", "z"]
    assert ro.alloc_bytes == {"x": 64, "y": 64, "z": 64}
    assert (ro.peak_bytes, ro.peak_at) == (192, "z")


def test_region_face_schedule_is_a_name_sequence():
    def step(x, w):
        a = x * x
        b = w * w
        c = a + b
        return c

    x, w = T(np.zeros(8), ("i",)), T(np.zeros(8), ("i",))
    ls = lift_step(step, x=x.layout, w=w.layout)
    base = peak_memory(ls.region, names=ls.names)
    vars_ = [v for v, _ in base.timeline]
    # a and b swapped is an equally valid schedule; the report stays coherent
    i, j = vars_.index("a"), vars_.index("b")
    vars_[i], vars_[j] = vars_[j], vars_[i]
    moved = peak_memory(ls.region, order=tuple(vars_), names=ls.names)
    assert moved.peak_bytes == base.peak_bytes  # symmetric here; validity is the point
    with pytest.raises(ValueError, match="not topological"):
        bad = ["c"] + [v for v in vars_ if v != "c"]
        peak_memory(ls.region, order=tuple(bad), names=ls.names)
