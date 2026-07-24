"""The scope + assemblage (200 §1.6/6.2/6.3): makers declare-at-use, units
compose with |, params become named inputs by capture identity, policies
are identity-bearing, taps select outputs, dropout is the mode-aware idiom."""

import numpy as np
import pytest
from pdum.dsl import events
from pdum.dsl.naming import NameCollision
from pdum.tl import Tensor
from pdum.tl.assemblage import assemblage, unit
from pdum.tl.ir import run
from pdum.tl.lifting import contract
from pdum.tl.scope import dropout, scope, tap


def T(arr, names):
    return Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)


def make_dense(s, cfg):
    w = s.param("w", d=cfg["d"], m=cfg["m"])

    @unit
    def dense(h):
        return contract(h, w)

    return dense


def make_scale(s, cfg):
    g = s.param("g", m=cfg["m"])

    @unit
    def scaled(h):
        y = h * g.repeat_like(h, but="m")
        tap(y, s / "y")
        return dropout(y, 0.5, s / "drop")

    return scaled


def _model(root, cfg, taps=()):
    m = make_dense(root / "dense", cfg) | make_scale(root / "out", cfg)
    return assemblage(m, scope=root, taps=taps, h=T(np.zeros((3, cfg["d"])), ("t", "d")).layout)


CFG = {"d": 4, "m": 2}


def test_makers_declare_level_first_names_and_the_program_runs():
    root = scope()
    a = _model(root, CFG)
    assert sorted(a.params) == ["dense.w", "out.g"]  # level-first contract names
    assert root.spec() == {"dense.w": {"d": 4, "m": 2}, "out.g": {"m": 2}}
    rng = np.random.default_rng(0)
    h = rng.standard_normal((3, 4))
    w = rng.standard_normal((4, 2))
    g = rng.standard_normal(2)
    env = run(a.program, {"h": T(h, ("t", "d")), "dense.w": T(w, ("d", "m")), "out.g": T(g, ("m",))})
    got = env[a.output].to_numpy(order=("t", "m"))
    mask = env[next(v for v in a.program.vars if v.startswith("mask"))].to_numpy(order=("t", "m"))
    want = np.where(mask, 0.0, (h @ w) * g / 0.5)
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_eval_mode_drops_the_dropout_and_identity_differs():
    """Policies are IDENTITY-BEARING: a train build and an eval build are
    different cache entries — the mode-as-loose-bool collision is
    unwritable (the P5 gate's policy-collision pin)."""
    root = scope()
    train = _model(root.with_(mode="train"), CFG)
    ev = _model(root.with_(mode="eval"), CFG)
    assert train.program is not ev.program
    assert not any(i.op == "random" for i in ev.program.instrs)  # identity under eval
    assert any(i.op == "random" for i in train.program.instrs)
    with events.forbid("assemblage.miss"):  # same policies: a warm hit, no rebuild
        again = _model(root.with_(mode="train"), CFG)
    assert again is train


def test_tap_sets_are_identity_bearing_and_select_outputs():
    """Different tap sets never share a derived Program (the tap-set
    identity pin); unrequested taps are pruned by DCE and cost nothing."""
    root = scope()
    bare = _model(root, CFG)
    tapped = _model(root, CFG, taps=("out.y",))
    assert tapped.program is not bare.program
    assert bare.taps == tapped.taps == {"out.y": "y"}  # the SITE exists either way
    assert "y" in tapped.outputs and "y" in tapped.program.vars
    got = run(tapped.program, {
        "h": T(np.ones((3, 4)), ("t", "d")),
        "dense.w": T(np.ones((4, 2)), ("d", "m")),
        "out.g": T(np.ones(2), ("m",)),
    })["y"].to_numpy()
    np.testing.assert_allclose(got, np.full((3, 2), 4.0))


def test_the_tie_capture_identity_makes_one_leaf():
    """Zoo gate 9's mechanism: ONE Param object captured by two units is
    ONE input leaf."""
    root = scope()
    shared = root.param("wte", v=3, d=4)

    @unit
    def first(h):
        return contract(h, shared)

    @unit
    def second(h):
        return contract(h, shared.rename(v="o"))

    a = assemblage(first | second, scope=root, h=T(np.zeros((2, 3)), ("t", "v")).layout)
    assert list(a.params) == ["wte"]  # one leaf, not two
    assert sum(1 for i in a.program.instrs if i.op == "input") == 2  # h + wte


def test_param_conflict_refuses_idempotent_redeclare_returns_the_object():
    root = scope()
    p1 = (root / "attn").param("wq", d=4)
    assert (root / "attn").param("wq", d=4) is p1  # idempotent: the same object
    with pytest.raises(NameCollision, match=r"leaf 'attn.wq' is already declared.*never auto-suffixed"):
        (root / "attn").param("wq", d=8)


def test_maker_level_pipe_refuses():
    root = scope()
    u = make_dense(root, CFG)
    with pytest.raises(TypeError, match=r"`\|` composes UNITS only"):
        u | make_scale  # a maker, not a unit


def test_streams_derive_from_site_paths():
    root = scope(root_key=7)
    a = (root / "h" / "0" / "drop").stream()
    b = (root / "h" / "1" / "drop").stream()
    assert a != b and a == (root / "h" / "0" / "drop").stream()
