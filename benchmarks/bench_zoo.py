"""The zoo benchmark — the performance discipline's rig (310).

Per entry, per device, over identical inputs: the numpy reference column
(dialect.run_named — the deliberately inefficient denotational floor), then
a translated CHECK column and an idiomatic BASELINE column per framework
(torch, jax). OUTSIDE the default pytest gate on purpose: benchmarks run
deliberately, never as a side effect of the suite.

Two laws are load-bearing and stated once:

  * never benchmark a wrong program — every column is asserted against the
    entry's numpy denotation before its timed loop;
  * never benchmark a recompile — timed loops run under events.forbid on
    every cache-miss event (PR #8's warmth law), and framework compiles
    land in WARMUP: jax idiomatic columns are jitted, so their XLA compile
    happens on the first warmup call and the timed loop hits the jit cache
    (fixed shapes — no retrace). Async dispatch is closed with
    block_until_ready / cuda.synchronize inside the lap.

Profiles: "toy" is the zoo's own shapes (correctness-sized, timing noise);
"small" scales entries while keeping the REFERENCE column's materialized
broadcasts in memory (contract materializes the full (m, k, n) product —
compute.py's stated point). Pass --skip-reference to scale beyond it.

Usage:
  uv run --group torch --group jax python benchmarks/bench_zoo.py
  uv run --group jax python benchmarks/bench_zoo.py --frameworks jax --device cuda --profile small
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "conformance"))

from pdum.dsl import events  # noqa: E402
from pdum.tl.dialect import run_named  # noqa: E402
from pdum.tl.zoo import (  # noqa: E402
    GemmConfig,
    GPT2Config,
    LlamaConfig,
    MoEConfig,
    flash_attention,
    gpt2,
    heat2d,
    llama_block,
    moe,
    tiled_matmul,
)

# entry -> (ctor, model kwargs, baseline kwargs) per profile; "small" keeps
# the reference column's dense products under ~100 MB
PROFILES = {
    "toy": {
        "gpt2": (gpt2, {}, {}),
        "llama": (llama_block, {}, {}),
        "flash": (flash_attention, {}, {}),
        "heat2d": (heat2d, {}, {}),
        "moe": (moe, {}, {}),
        "gemm": (tiled_matmul, {}, {}),
    },
    "small": {
        "gpt2": (
            gpt2,
            {"cfg": GPT2Config(t=32, d=64, nh=4, hk=16, m=256, layers=2, v=128)},
            {"cfg": GPT2Config(t=32, d=64, nh=4, hk=16, m=256, layers=2, v=128)},
        ),
        "llama": (
            llama_block,
            {"cfg": LlamaConfig(t=32, d=64, g=4, r=2, c=8, kv=16, m=128)},
            {"cfg": LlamaConfig(t=32, d=64, g=4, r=2, c=8, kv=16, m=128)},
        ),
        "flash": (flash_attention, {"T": 64, "E": 32, "OD": 16}, {}),
        "heat2d": (heat2d, {"N": 64, "M": 64, "T": 8}, {"T": 8}),
        "moe": (
            moe,
            {"cfg": MoEConfig(t=64, d=32, e=8, k=2, cap=16, m=64)},
            {"cfg": MoEConfig(t=64, d=32, e=8, k=2, cap=16, m=64)},
        ),
        "gemm": (
            tiled_matmul,
            {"cfg": GemmConfig(m=64, n=64, k=64, mi=8, ni=8, ki=8)},
            {"cfg": GemmConfig(m=64, n=64, k=64, mi=8, ni=8, ki=8)},
        ),
    },
}

WARMTH = ("spec.miss", "artifact.miss", "kernel.miss", "assemblage.miss")
TOL = dict(rtol=1e-7, atol=1e-9)


def _param_inputs(m):
    out = {}
    for p in m.region.params:
        name = m.names[id(p)]
        order = tuple(d.name for d in p.type.dims)
        out[name] = m.inputs[name].to_numpy(order=order)
    return out


def _time(fn, *, repeat: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    with events.forbid(*WARMTH):  # the warmth law: a timed loop never compiles
        laps = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            laps.append(time.perf_counter() - t0)
    return statistics.median(laps) * 1e3


def _reference_column(m, want, opts):
    ref_inputs = dict(m.inputs)
    got = run_named(m.region, ref_inputs, m.names)[m.out].to_numpy(order=m.order)
    np.testing.assert_allclose(got, want, **TOL)

    def call():
        run_named(m.region, ref_inputs, m.names)

    return [("reference", _time(call, **opts))]


def _torch_columns(name, m, want, bkw, device, opts):
    import torch

    from torch_evaluator import run_named_torch
    from torch_zoo import BASELINES

    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    rows = []
    tin = _param_inputs(m)
    got = run_named_torch(m.region, tin, m.names, device=device)[m.out].numpy(order=m.order)
    np.testing.assert_allclose(got, want, **TOL)
    tin_dev = {k: torch.as_tensor(v).to(device) for k, v in tin.items()}

    def translated():
        run_named_torch(m.region, tin_dev, m.names, device=device)
        sync()

    rows.append(("torch/translated", _time(translated, **opts)))
    if name in BASELINES:
        binp = {k: torch.as_tensor(v).to(device=device) for k, v in m.numpy_inputs().items()}
        got = BASELINES[name](binp, **bkw).cpu().numpy()
        np.testing.assert_allclose(got, want, **TOL)

        def idiomatic():
            BASELINES[name](binp, **bkw)
            sync()

        rows.append(("torch/idiomatic", _time(idiomatic, **opts)))
    return rows


def _jax_columns(name, m, want, bkw, device, opts):
    import jax

    from jax_evaluator import run_named_jax
    from jax_zoo import BASELINES

    rows = []
    tin = _param_inputs(m)
    fields = run_named_jax(m.region, tin, m.names, device=device)
    np.testing.assert_allclose(fields[m.out].numpy(order=m.order), want, **TOL)
    dev = jax.devices("gpu" if device == "cuda" else "cpu")[0]
    tin_dev = {k: jax.device_put(v, dev) for k, v in tin.items()}

    def translated():
        out = run_named_jax(m.region, tin_dev, m.names, device=device)
        jax.block_until_ready([f.data for f in out.values()])

    rows.append(("jax/translated", _time(translated, **opts)))
    if name in BASELINES:
        binp = {k: jax.device_put(v, dev) for k, v in m.numpy_inputs().items()}
        jitted = jax.jit(lambda i: BASELINES[name](i, **bkw))
        np.testing.assert_allclose(np.asarray(jitted(binp)), want, **TOL)

        def idiomatic():
            jax.block_until_ready(jitted(binp))

        rows.append(("jax/idiomatic", _time(idiomatic, **opts)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", choices=sorted(PROFILES), default="toy")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--frameworks", default="torch,jax", help="comma-separated: torch, jax")
    ap.add_argument("--entries", default=None, help="comma-separated subset")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--skip-reference", action="store_true", help="drop the numpy column (scale past its memory)")
    args = ap.parse_args()

    frameworks = [f for f in args.frameworks.split(",") if f]
    columns = {"torch": _torch_columns, "jax": _jax_columns}
    unknown = [f for f in frameworks if f not in columns]
    if unknown:
        sys.exit(f"unknown framework(s) {unknown}; known: {sorted(columns)}")
    entries = PROFILES[args.profile]
    if args.entries:
        entries = {k: entries[k] for k in args.entries.split(",")}
    opts = dict(repeat=args.repeat, warmup=args.warmup)

    print(f"profile={args.profile} device={args.device} frameworks={frameworks} median of {args.repeat}")
    print(f"{'entry':10s} {'column':17s} {'ms/iter':>10s} {'vs reference':>13s}")
    for name, (ctor, mkw, bkw) in entries.items():
        m = ctor(**mkw)
        want = m.ref(m.numpy_inputs())
        rows = [] if args.skip_reference else _reference_column(m, want, opts)
        for fw in frameworks:
            rows += columns[fw](name, m, want, bkw, args.device, opts)
        base = rows[0][1] if rows and rows[0][0] == "reference" else None
        for col, ms in rows:
            speed = f"{base / ms:10.1f}x" if base is not None else f"{'—':>11s}"
            print(f"{name:10s} {col:17s} {ms:10.3f} {speed}")


if __name__ == "__main__":
    main()
