"""The anchor-and-absorb partitioner (330 §7.5): gpt2 carves into
recognized groups, the carved plan EXECUTES to the model's own output,
and repeated layers share kernels and analysis facts."""

from collections import Counter

import numpy as np

from pdum.tl.analysis import no_reanalysis
from pdum.tl.dialect import run_region, walk_region
from pdum.tl.partition import plan_model
from pdum.tl.zoo.gpt2 import gpt2


def _execute(plan, model):
    """The reference plan executor: worklist over carves, free views
    (renames) resolved at the edges — boundary wiring is metadata."""
    vals = {id(p): model.inputs[model.names[id(p)]] for p in model.region.params}
    pending = [c for c in plan.carves if c.root is not None]
    while pending:
        for c in pending:
            for b in c.bounds:  # resolve free views whose sources exist
                if id(b) not in vals and b.op == "tl.rename" and id(b.args[0]) in vals:
                    from pdum.dsl.ir import Builder, Region
                    from pdum.tl.fusion import OPS

                    bb = Builder(OPS)
                    pp = bb.param(0, b.args[0].type)
                    r = Region(params=(pp,), body=(bb.emit("core.yield", bb.emit(b.op, pp, **dict(b.attrs))),))
                    vals[id(b)] = run_region(r, [vals[id(b.args[0])]])
            if all(id(b) in vals for b in c.bounds):
                vals[id(c.root)] = run_region(c.kernel, [vals[id(b)] for b in c.bounds])
                pending.remove(c)
                break
        else:
            raise AssertionError("plan stalled: unresolvable boundary")
    return vals


def test_gpt2_carves_into_recognized_groups():
    plan = plan_model(gpt2().region)
    by = Counter(c.group.template for c in plan.carves)
    assert by["contraction-epilogue"] == 17  # q,k,v,sc,pv,w1,w2,wo per block + head:
    assert by["row-normalization"] == 2  # the 2-dim wo contract claims too (§7.6)
    assert by["row-statistics"] == 5  # every layernorm, two-pass, staged once
    assert by["map-chain"] >= 2  # the causal mask forests
    assert plan.coverage() > 0.9  # only the embedding gather remains red
    reasons = " ".join(c.group.reason for c in plan.carves if c.group.confidence == "red")
    assert "tl.take" in reasons  # the embedding gather, named — scatter_add's twin


def test_the_carved_plan_executes_to_the_models_output():
    """The partition is semantics-preserving BY CONSTRUCTION — group
    boundaries are just materialized tensors, so executing the carves
    in dependency order is bit-equal to running the model whole."""
    m = gpt2()
    plan = plan_model(m.region)
    vals = _execute(plan, m)
    root = m.region.body[-1].args[0]
    want = run_region(m.region, [m.inputs[m.names[id(p)]] for p in m.region.params])
    np.testing.assert_array_equal(
        vals[id(root)].to_numpy(order=m.order), want.to_numpy(order=m.order)
    )


def test_repeated_layers_share_kernels_and_analysis():
    """The 330 §4 law at model scale: layer 2's groups land on layer 1's
    content keys, and a warm re-plan performs ZERO new analysis."""
    m = gpt2()
    plan = plan_model(m.region)
    assert plan.distinct_kernels() < sum(1 for c in plan.carves if c.root is not None)
    keys = Counter(c.kernel.key for c in plan.carves if c.group.confidence != "red")
    assert max(keys.values()) >= 2  # at least one kernel serves two layers
    with no_reanalysis():  # the warmth pin: any recompute raises
        again = plan_model(m.region)
    assert again.coverage() == plan.coverage()


def test_the_backward_joint_partitions_and_reports_honestly():
    """The adjoint's joint region carves under the §7.6 rows: the
    computed-operand contraction claims the chart-wrapped adjoint
    products (upstream absorption, rowsum and broadcast shapes alike),
    and what remains red is named — mean chains, multi-dim
    contractions, the scatter_add embedding grad."""
    from pdum.tl.autodiff import grad

    m = gpt2()
    rg = grad(m.region, m.out, seed="dY", names=m.names)
    plan = plan_model(rg.region)
    by = Counter(c.group.template for c in plan.carves)
    assert by["contraction-epilogue"] >= 50  # the adjoint products claim (§7.6)
    assert by["map-chain"] > 20  # adjoint broadcast/mask plumbing claims as maps
    assert plan.coverage() > 0.8  # approaching forward parity; the rest is priced red
    with no_reanalysis():  # and the joint's facts cache like everything else
        plan_model(rg.region)


def test_the_rowstat_row_claims_every_layernorm_once():
    """§7.6 B: all five layernorms (two per block + the final) carve as
    row-statistics onto ONE canonical kernel, and the certificate is
    proved-exact — staging erases, nothing reassociates."""
    plan = plan_model(gpt2().region)
    rs = [c for c in plan.carves if c.group.template == "row-statistics"]
    assert len(rs) == 5
    assert len({c.kernel.key for c in rs}) == 1  # one kernel, paid once
    assert rs[0].group.certificate.verdict == "proved-exact"


def test_the_scatter_add_refusal_is_named():
    """§7.6 E: the embedding gradient is a genuine cross-program
    reduction and refuses RED with its op named — 340 §6's family,
    re-entry conditions recorded, never silent."""
    from pdum.tl.autodiff import grad

    m = gpt2()
    rg = grad(m.region, m.out, seed="dY", names=m.names)
    plan = plan_model(rg.region)
    reasons = " ".join(c.group.reason for c in plan.carves if c.group.confidence == "red")
    assert "tl.scatter_add" in reasons


def test_the_flash_joint_rematerializes_and_executes():
    """§7.8 end to end on the reference: the joint flash region claims
    WITH artifacts (m and den surfaced — the fold already carried both),
    the backward contractions re-derive P through traveled cones, and
    the carved plan executes to the naive joint's outputs within the
    flash license's tolerance (the saved statistics are the ONLINE
    ones — the already-priced deviation). Red groups run through the
    reference, the doctrine in action."""
    from pdum.tl.autodiff import grad
    from pdum.tl.tensor import Tensor
    from pdum.tl.zoo.tiles import flash_tile

    f = flash_tile(T=8, E=4, OD=4, SI=2)
    out = f.naive.body[-1].args[0]
    names = {id(p): f"p{i}" for i, p in enumerate(f.naive.params)}
    names[id(out)] = "out"
    rg = grad(f.naive, "out", seed="dO", names=names)
    plan = plan_model(rg.region)

    fl = next(c for c in plan.carves if c.group.template == "flash")
    assert fl.group.artifacts == ("m", "den") and len(fl.artifacts) == 2
    by = Counter(c.group.template for c in plan.carves)
    assert by["contraction-epilogue"] >= 5  # the adjoint claims through travel
    assert 0.8 < plan.coverage() <= 1.0  # copies count once; the scan stays red
    remat = [
        c
        for c in plan.carves
        if c.group.template == "contraction-epilogue"
        and sum(1 for n in walk_region(c.kernel) if n.op == "tl.reduce") >= 2
    ]
    assert remat  # at least one consumer re-derives P: sc travels as a prologue reduce

    rng = np.random.default_rng(5)
    ins = []
    for p in rg.region.params:
        dims = p.type.dims
        arr = rng.standard_normal(tuple(d.stop - d.start for d in dims))
        ins.append(Tensor.from_numpy(arr, tuple(d.name for d in dims)))
    want = run_region(rg.region, list(ins))

    vals = {id(p): v for p, v in zip(rg.region.params, ins)}
    def rv(b):  # bounds may be the bare node under a producer's chart root
        while id(b) not in vals and b.op in ("tl.with_charts", "tl.strip_charts", "tl.simplify"):
            b = b.args[0]
        return b

    pending = [c for c in plan.carves if c.root is not None]
    progressed = True
    while pending and progressed:
        progressed = False
        for c in list(pending):
            c_bounds = [rv(b) for b in c.bounds]
            if all(id(b) in vals for b in c_bounds):
                kernel = c.group.kernel if c.group.kernel is not None else c.kernel
                got = run_region(kernel, [vals[id(b)] for b in c_bounds])
                outs = got if isinstance(got, (tuple, list)) else (got,)
                for node, val in zip((c.root, *c.artifacts), outs):
                    vals[id(node)] = val
                    while node.op in ("tl.with_charts", "tl.strip_charts", "tl.simplify"):
                        node = node.args[0]  # chart roots alias their bare value
                        vals[id(node)] = val
                pending.remove(c)
                progressed = True
    assert not pending, "plan stalled: unresolvable boundary"
    for target, w in zip(rg.region.body[-1].args[0].args, want):
        x = target
        while id(x) not in vals and x.op in ("tl.with_charts", "tl.strip_charts", "tl.simplify"):
            x = x.args[0]
        arr = vals[id(x)].to_numpy(order=w.names)
        np.testing.assert_allclose(arr, w.to_numpy(order=w.names), rtol=1e-9, atol=1e-12)
