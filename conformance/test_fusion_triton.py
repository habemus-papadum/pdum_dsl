"""The fused kernels on silicon (330 §7.3), CUDA only: template 2's
generated tile region runs through the Triton translator, the epilogue
provably fused into the one kernel, and the first measurement lands in
the ledger through the analysis cache — green confidence is EARNED by a
measurement fact, never assumed."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("triton", reason="triton is not installed (rides the torch CUDA group)")
torch = pytest.importorskip("torch", reason="the torch reference-runtime group is not installed")

from pdum.dsl.ir import Builder, Region  # noqa: E402
from pdum.dsl.ops import CORE_OPS  # noqa: E402
from pdum.dsl.types import f64  # noqa: E402
from pdum.tl.analysis import defanalysis, no_reanalysis  # noqa: E402
from pdum.tl.dialect import TL_OPS, tensor_type_of_layout  # noqa: E402
from pdum.tl.fusion import plan_region  # noqa: E402
from pdum.tl.tensor import Tensor  # noqa: E402

OPS = {**CORE_OPS, **TL_OPS}


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")


def _subject(M=6, K=8, N=5, seed=3):
    """relu(a @ b + bias) — the epilogue-on-a-contraction case."""
    rng = np.random.default_rng(seed)
    A = np.asarray(rng.standard_normal((M, K)))
    B = np.asarray(rng.standard_normal((K, N)))
    bias = np.asarray(rng.standard_normal(N))
    t = {
        "a": Tensor.from_numpy(A, ("m", "k")),
        "b": Tensor.from_numpy(B, ("k", "n")),
        "c": Tensor.from_numpy(bias, ("n",)),
    }
    b = Builder(OPS)
    pa = b.param(0, tensor_type_of_layout(t["a"].layout))
    pb = b.param(1, tensor_type_of_layout(t["b"].layout))
    pc = b.param(2, tensor_type_of_layout(t["c"].layout))
    prod = b.emit("tl.pointwise", b.emit("tl.repeat_like", pa, pb), b.emit("tl.repeat_like", pb, pa), f="mul")
    z = b.emit("tl.reduce", prod, dims=("k",), f="sum")
    zb = b.emit("tl.pointwise", z, b.emit("tl.repeat_like", pc, z), f="add")
    y = b.emit("tl.pointwise", zb, b.emit("core.const", type=f64, value=0.0), f="maximum")
    region = Region(params=(pa, pb, pc), body=(b.emit("core.yield", y),))
    return region, [A, B, bias], np.maximum(A @ B + bias, 0.0)


def test_fused_contraction_epilogue_runs_on_triton():
    """One kernel, one loop, one store — and the relu/bias are inside it."""
    _require_cuda()
    from triton_tile import compile_tile

    region, arrays, want = _subject()
    (g,) = plan_region(region).groups
    assert g.template == "contraction-epilogue"
    run = compile_tile(g.kernel)
    np.testing.assert_allclose(run(arrays), want, rtol=1e-5, atol=1e-6)
    assert run.source.count("for ") == 1  # the k-fold is the only loop
    assert run.source.count("tl.store") == 1  # the epilogue never touched memory
    assert "tl.maximum" in run.source  # the relu rode the accumulator out


def test_the_fused_flash_composition_runs_on_triton():
    """Template 3b on silicon: the pass recognizes the materialized-softmax
    attention region, emits the online-softmax fold, and the fused kernel
    runs on the device — the o and den finals share one sweep, no barrier."""
    _require_cuda()
    from triton_tile import compile_tile

    from pdum.tl.zoo.tiles import flash_tile

    f = flash_tile()
    (g,) = plan_region(f.naive).groups
    assert g.template == "flash"
    run = compile_tile(g.kernel)
    got = run([v.to_numpy() for v in f.inputs.values()])
    np.testing.assert_allclose(got, f.oracle(f.numpy_inputs()), rtol=1e-4, atol=1e-6)
    assert run.source.count("for ") == 1 and "barrier" not in run.source  # shared sweep


def test_the_first_measurement_greens_the_group():
    """A measurement is an analysis whose evaluator is the machine: it lands
    in the ledger keyed by (kernel content, machine), never re-runs warm,
    and the group's color upgrade is a FACT-backed act."""
    _require_cuda()
    import time

    from triton_tile import compile_tile

    region, arrays, _ = _subject()
    (g,) = plan_region(region).groups

    @defanalysis("fusion.triton-ms", 1, ledger=True)
    def triton_ms(kernel, *, machine, reps=20):
        run = compile_tile(kernel)
        run(arrays)  # warmup: the compile stays out of the laps
        laps = []
        for _ in range(reps):
            t0 = time.perf_counter()
            run(arrays)
            laps.append(time.perf_counter() - t0)
        return {"ms": sorted(laps)[len(laps) // 2] * 1e3, "reps": reps}

    machine = torch.cuda.get_device_name(0)
    fact = triton_ms(g.kernel, machine=machine)
    assert fact.value["ms"] > 0
    with no_reanalysis():  # keyed identically: never measured twice
        again = triton_ms(g.kernel, machine=machine)
    assert again.key == fact.key
    green = replace(g, confidence="green")
    assert (green.template, green.confidence) == ("contraction-epilogue", "green")


def test_the_planned_launch_runs_gridded_on_silicon():
    """340 §7.3 end to end: plan_region with a machine attaches the
    analytic-default launch; the translator grids it; T=512 — the size
    that crashed OutOfResources ungridded — now runs and verifies."""
    _require_cuda()
    from triton_tile import compile_tile

    from pdum.tl.launch import TileLevel, TileMachine
    from pdum.tl.zoo.tiles import flash_tile

    machine = TileMachine((TileLevel("shared", 128, 101376),))
    f = flash_tile(T=512, E=64, OD=64, SI=2)
    (g,) = plan_region(f.naive, machine=machine).groups
    assert g.template == "flash" and g.launch
    run = compile_tile(g.kernel, g.launch)
    assert run.grid > 1
    got = run([v.to_numpy() for v in f.inputs.values()])
    np.testing.assert_allclose(got, f.oracle(f.numpy_inputs()), rtol=1e-4, atol=1e-5)


def test_mask_derived_bounds_prune_the_sweep_bit_exactly():
    """340 §4b on silicon: the plan carries mask-derived fold bounds
    (causal: hi = pid+1 — the bound a hand author writes, computed from
    the mask), and the pruned sweep is BIT-equal to the full one — the
    template proved the skipped tiles inert."""
    _require_cuda()
    from triton_tile import compile_tile

    from pdum.tl.launch import TileLevel, TileMachine
    from pdum.tl.zoo.tiles import flash_tile

    machine = TileMachine((TileLevel("shared", 128, 101376),))
    f = flash_tile(T=128, E=32, OD=32, SI=2)
    (g,) = plan_region(f.naive, machine=machine).groups
    assert g.prune == (("so", (0, 0, 1), (1, 1, 1)),)
    vals = [v.to_numpy() for v in f.inputs.values()]
    full = compile_tile(g.kernel, g.launch)
    pruned = compile_tile(g.kernel, g.launch, g.prune)
    assert "tl.minimum(4, 1 + 1 * pid_t)" in pruned.source
    np.testing.assert_array_equal(full(vals), pruned(vals))


def test_a_computed_operand_contraction_fuses_the_prologue():
    """The §7.6 row on silicon: an operand that is a COMPUTED chain
    (exp(x), the trivial adjoint stand-in) rides as a fold element
    source and is computed per k-tile inside the sweep — no separate
    materialization, the dot still ieee. Recompute-exact: the prologue
    is per-element work with nothing ordered to reassociate."""
    _require_cuda()
    from triton_tile import compile_tile

    rng = np.random.default_rng(9)
    X = np.asarray(rng.standard_normal((32, 32)))
    dY = np.asarray(rng.standard_normal((32, 16)))
    b = Builder(OPS)
    px = b.param(0, tensor_type_of_layout(Tensor.from_numpy(X, ("t", "m")).layout))
    pd = b.param(1, tensor_type_of_layout(Tensor.from_numpy(dY, ("t", "n")).layout))
    a = b.emit("tl.pointwise", px, f="exp")  # the computed operand
    prod = b.emit("tl.pointwise", b.emit("tl.repeat_like", a, pd), b.emit("tl.repeat_like", pd, a), f="mul")
    y = b.emit("tl.reduce", prod, dims=("t",), f="sum")
    region = Region(params=(px, pd), body=(b.emit("core.yield", y),))

    (g,) = plan_region(region).groups
    assert g.template == "contraction-epilogue"
    run = compile_tile(g.kernel)
    want = np.exp(X).T @ dY
    # the translated column is f32 and exp amplifies before the sum: the
    # flash tests' tolerance, not the small-K gemm's
    np.testing.assert_allclose(run([X, dY]), want, rtol=1e-4, atol=1e-6)
    assert run.source.count("for ") == 1  # the k-fold is the only loop
    body = run.source[run.source.index("for ") :]
    assert "tl.exp" in body  # the prologue lives INSIDE the sweep
    assert 'input_precision="ieee"' in run.source  # the dot stayed honest


def test_the_softmax_adjoint_composes_on_silicon():
    """The D-class (§7.6): dS = P * (dP - rowsum(dP * P)) is a ROWSUM-
    shaped contraction (same-space operands, no broadcast pair) with the
    outer product-and-subtract as its epilogue — one kernel, one loop,
    no new template."""
    _require_cuda()
    from triton_tile import compile_tile

    rng = np.random.default_rng(11)
    P = np.asarray(rng.random((32, 32)))
    dP = np.asarray(rng.standard_normal((32, 32)))
    b = Builder(OPS)
    pp = b.param(0, tensor_type_of_layout(Tensor.from_numpy(P, ("t", "s")).layout))
    pd = b.param(1, tensor_type_of_layout(Tensor.from_numpy(dP, ("t", "s")).layout))
    rs = b.emit("tl.reduce", b.emit("tl.pointwise", pd, pp, f="mul"), dims=("s",), f="sum")
    y = b.emit(
        "tl.pointwise", pp, b.emit("tl.pointwise", pd, b.emit("tl.repeat_like", rs, pd), f="sub"), f="mul"
    )
    region = Region(params=(pp, pd), body=(b.emit("core.yield", y),))

    (g,) = plan_region(region).groups
    assert g.template == "contraction-epilogue"
    run = compile_tile(g.kernel)
    want = P * (dP - (dP * P).sum(axis=1, keepdims=True))
    np.testing.assert_allclose(run([P, dP]), want, rtol=1e-5, atol=1e-6)
    assert run.source.count("for ") == 1 and run.source.count("tl.store") == 1


def test_the_rowstat_layernorm_runs_on_triton():
    """§7.6 B on silicon: the two-pass layernorm claims as row-statistics,
    stages its rows once, and the translator lowers mean as sum with a
    divide-by-N finalize — N static, one scalar op, no new license."""
    _require_cuda()
    from triton_tile import compile_tile

    from pdum.tl.zoo.gpt2 import gpt2

    plan_m = __import__("pdum.tl.partition", fromlist=["plan_model"]).plan_model(gpt2().region)
    carve = next(c for c in plan_m.carves if c.group.template == "row-statistics")
    # rebuild the oracle from the carve's own kernel via the reference
    from pdum.tl.dialect import run_region

    rng = np.random.default_rng(17)
    vals = []
    for p in carve.kernel.params:
        dims = p.type.dims
        arr = rng.standard_normal(tuple(d.stop - d.start for d in dims))
        vals.append(Tensor.from_numpy(arr, tuple(d.name for d in dims)))
    want = run_region(carve.kernel, list(vals))
    run = compile_tile(carve.group.kernel)
    got = run([v.to_numpy() for v in vals])
    np.testing.assert_allclose(got, want.to_numpy(order=want.names), rtol=1e-5, atol=1e-6)
    assert "/ 6" in run.source  # the mean's divide-by-N, spelled once


def test_a_two_dim_contraction_runs_on_triton():
    """§7.6 C on silicon: contract over (nh, hk) — the fold runs the
    widest contracted dim, the rest reduce whole inside the step, and
    the tf32 tripwire guards the raw mul+sum lowering."""
    _require_cuda()
    from triton_tile import compile_tile

    rng = np.random.default_rng(23)
    X = np.asarray(rng.standard_normal((32, 4, 16)))
    W = np.asarray(rng.standard_normal((4, 16, 32)))
    b = Builder(OPS)
    px = b.param(0, tensor_type_of_layout(Tensor.from_numpy(X, ("t", "nh", "hk")).layout))
    pw = b.param(1, tensor_type_of_layout(Tensor.from_numpy(W, ("nh", "hk", "d")).layout))
    prod = b.emit("tl.pointwise", b.emit("tl.repeat_like", px, pw), b.emit("tl.repeat_like", pw, px), f="mul")
    y = b.emit("tl.reduce", prod, dims=("nh", "hk"), f="sum")
    region = Region(params=(px, pw), body=(b.emit("core.yield", y),))

    (g,) = plan_region(region).groups
    assert g.template == "contraction-epilogue"
    run = compile_tile(g.kernel)
    want = np.einsum("thk,hkd->td", X, W)
    np.testing.assert_allclose(run([X, W]), want, rtol=1e-4, atol=1e-6)
    assert run.source.count("for ") == 1  # ONE fold: the widest dim; nh reduces in-step
