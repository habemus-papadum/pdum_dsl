"""Step 5 — the rewrite engine: rules as data, one driver, stage legality."""

import pytest
from pdum.dsl import types as T
from pdum.dsl.ir import Builder, Region, VerifyError
from pdum.dsl.ops import CORE_OPS
from pdum.dsl.printer import print_program
from pdum.dsl.rewrite import MatchLog, Pat, Stage, check_legal, rewrite, run_stage

b = Builder(CORE_OPS)


def const(v, ty=T.f64):
    return b.emit("core.const", type=ty, value=v)


def is_const(m, name, value=None):
    n = m[name]
    return n.op == "core.const" and (value is None or dict(n.attrs)["value"] == value)


SIMPLIFY = [
    (Pat("core.add", ("x", "z"), guard=lambda m: is_const(m, "z", 0.0)), lambda bl, m: m["x"]),
    (Pat("core.mul", ("x", "z"), guard=lambda m: is_const(m, "z", 1.0)), lambda bl, m: m["x"]),
    (
        Pat("core.add", ("x", "y"), guard=lambda m: is_const(m, "x") and is_const(m, "y")),
        lambda bl, m: bl.emit(
            "core.const", type=m["root"].type, value=dict(m["x"].attrs)["value"] + dict(m["y"].attrs)["value"]
        ),
    ),
    (Pat("core.neg", (Pat("core.neg", ("x",)),)), lambda bl, m: m["x"]),
    (Pat("core.sub", ("x", "x")), lambda bl, m: bl.emit("core.const", type=m["root"].type, value=0.0)),
]


def env(slot=0, ty=T.f64):
    return b.emit("core.env", type=ty, slot=slot)


def prog(*roots):
    return Region(body=(b.emit("core.yield", *roots),))


def test_basic_rules_and_fixpoint_chaining():
    x = env()
    noisy = b.emit("core.add", b.emit("core.mul", b.emit("core.add", x, const(0.0)), const(1.0)), const(0.0))
    out = rewrite(prog(noisy), SIMPLIFY, CORE_OPS)
    assert out.body[-1].args[0] is x  # ((x+0)*1)+0 collapsed all the way to x


def test_const_folding_builds_new_nodes():
    out = rewrite(prog(b.emit("core.add", const(3.0), const(4.0))), SIMPLIFY, CORE_OPS)
    folded = out.body[-1].args[0]
    assert folded.op == "core.const" and dict(folded.attrs)["value"] == 7.0


def test_nonlinear_pattern_matches_structural_equality():
    # x - x -> 0 must fire for two structurally-equal but distinct env nodes.
    out = rewrite(prog(b.emit("core.sub", env(5), env(5))), SIMPLIFY, CORE_OPS)
    assert dict(out.body[-1].args[0].attrs)["value"] == 0.0
    # ...and must NOT fire for different slots:
    out2 = rewrite(prog(b.emit("core.sub", env(5), env(6))), SIMPLIFY, CORE_OPS)
    assert out2.body[-1].args[0].op == "core.sub"


def test_double_neg_nested_pattern():
    x = env()
    out = rewrite(prog(b.emit("core.neg", b.emit("core.neg", x))), SIMPLIFY, CORE_OPS)
    assert out.body[-1].args[0] is x


def test_dag_sharing_is_preserved_and_rewritten_once():
    x = env()
    shared = b.emit("core.add", x, const(0.0))  # rewrites to x
    root = b.emit("core.mul", shared, shared)  # SAME object twice
    log = MatchLog()
    out = rewrite(prog(root), SIMPLIFY, CORE_OPS, log=log)
    result = out.body[-1].args[0]
    assert result.args[0] is result.args[1]  # sharing survived the rewrite
    assert len(log.entries) == 1  # and the shared node was rewritten ONCE


def test_rules_reach_inside_regions():
    x = env()
    cond = b.emit("core.cmp", x, x, pred="lt")
    then = Region(body=(b.emit("core.yield", b.emit("core.add", x, const(0.0))),))
    other = Region(body=(b.emit("core.yield", x),))
    branch = b.emit("core.if", cond, regions=(then, other))
    out = rewrite(prog(branch), SIMPLIFY, CORE_OPS)
    assert out.body[-1].args[0].regions[0].body[-1].args[0] is x  # cleaned inside the branch


def test_rule_priority_is_order():
    x = env()
    first = [(Pat("core.add"), lambda bl, m: m["root"].args[0])]  # aggressive: add -> lhs
    both = first + SIMPLIFY
    out = rewrite(prog(b.emit("core.add", x, const(0.0))), both, CORE_OPS)
    assert out.body[-1].args[0] is x  # first rule fired (same result here, by priority)


def test_nontermination_guard_is_loud():
    ping = [(Pat("core.neg", ("x",)), lambda bl, m: bl.emit("core.neg", bl.emit("core.neg", m["root"])))]
    with pytest.raises(VerifyError, match="did not stabilize"):
        rewrite(prog(b.emit("core.neg", env())), ping, CORE_OPS, name="ping")


def test_params_are_never_rewritten():
    p = b.param(0, T.f64)
    anything = [(Pat(None), lambda bl, m: const(9.0))]  # rewrite literally everything...
    region = Region(params=(p,), body=(b.emit("core.yield", p),))
    out = rewrite(region, anything, CORE_OPS)
    assert out.params[0] is p  # ...except binders


def test_fresh_replacements_inherit_the_replaced_nodes_loc():
    from pdum.dsl.ir import Loc

    lit = b.emit("core.add", const(3.0), const(4.0), loc=Loc("art.py", 7))
    out = rewrite(prog(lit), SIMPLIFY, CORE_OPS)
    folded = out.body[-1].args[0]
    assert folded.op == "core.const" and folded.loc == Loc("art.py", 7)  # builder default


def test_survivor_nodes_keep_their_own_provenance():
    from pdum.dsl.ir import Loc

    x = env()  # loc=None: x's own story
    out = rewrite(prog(b.emit("core.add", x, const(0.0), loc=Loc("art.py", 9))), SIMPLIFY, CORE_OPS)
    assert out.body[-1].args[0] is x and x.loc is None  # x survives UNstamped, identity intact


def test_stage_legality_names_the_op_and_stage():
    toy_ops = dict(CORE_OPS)
    from pdum.dsl.ops import OpDef

    toy_ops["toy.blit"] = OpDef("toy.blit", lambda a, at, r: T.f64)
    tb = Builder(toy_ops)
    bad = Region(body=(tb.emit("core.yield", tb.emit("toy.blit")),))
    with pytest.raises(VerifyError, match=r"\[mid\] illegal op 'toy.blit'"):
        run_stage(bad, Stage("mid", [], legal=frozenset({"core"})), toy_ops)
    check_legal(bad, frozenset({"core", "toy"}), "abi")  # widening the target passes


def test_stage_runs_rules_then_checks():
    x = env()
    dirty = prog(b.emit("core.add", x, const(0.0)))
    log = MatchLog()
    out = run_stage(dirty, Stage("simplify", SIMPLIFY, legal=frozenset({"core"})), CORE_OPS, log=log)
    assert out.body[-1].args[0] is x and len(log.entries) == 1


def test_golden_print_after_stage():
    x, y = b.param(0, T.f64), b.param(1, T.f64)
    expr = b.emit("core.add", b.emit("core.mul", b.emit("core.add", x, const(0.0)), const(1.0)), y)
    region = Region(params=(x, y), body=(b.emit("core.yield", expr),))
    out = run_stage(region, Stage("simplify", SIMPLIFY, legal=frozenset({"core"})), CORE_OPS)
    assert print_program(out, name="clean") == (
        "clean(%p0: f64, %p1: f64) {\n  %0 = core.add %p0, %p1 : f64\n  core.yield %0\n}"
    )
