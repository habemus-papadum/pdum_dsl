"""The zoo benchmark — the performance discipline's first rig (310).

Three columns over identical inputs, per entry, per device:

  reference   the numpy interpreter (dialect.run_named) — the deliberately
              inefficient denotational floor
  translated  the Region -> torch evaluator (the CHECK column)
  idiomatic   hand-written torch (the BASELINE column — the bar)

OUTSIDE the default pytest gate on purpose (testpaths never collect this
directory): benchmarks are run deliberately, never as a side effect of the
21-second suite. Two disciplines are load-bearing and stated here once:

  * never benchmark a wrong program — every column is asserted against the
    entry's numpy denotation before its timed loop;
  * never benchmark a recompile — timed loops run under events.forbid on
    every cache-miss event (PR #8's warmth law). The interpreter columns
    compile nothing today, so the pin is trivially green; when these
    columns mount as rt Pairs with real artifact caches, the same pin
    starts doing real work.

Profiles: "toy" is the zoo's own shapes (correctness-sized, timing noise);
"small" scales entries while keeping the REFERENCE column's materialized
broadcasts in memory (contract materializes the full (m, k, n) product —
compute.py's stated point). Pass --skip-reference to scale beyond it later.

Usage:
  uv run --group torch python benchmarks/bench_zoo.py
  uv run --group torch python benchmarks/bench_zoo.py --profile small --device cuda
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "conformance"))

import torch  # noqa: E402
from torch_evaluator import run_named_torch  # noqa: E402
from torch_zoo import BASELINES  # noqa: E402

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

# entry -> (model ctor kwargs, baseline kwargs) per profile; "small" keeps the
# reference column's dense products under ~100 MB
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


def _param_inputs(m):
    out = {}
    for p in m.region.params:
        name = m.names[id(p)]
        order = tuple(d.name for d in p.type.dims)
        out[name] = m.inputs[name].to_numpy(order=order)
    return out


def _time(fn, *, repeat: int, warmup: int, sync) -> float:
    for _ in range(warmup):
        fn()
    sync()
    with events.forbid(*WARMTH):  # the warmth law: a timed loop never compiles
        laps = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            sync()
            laps.append(time.perf_counter() - t0)
    return statistics.median(laps) * 1e3


def bench_entry(name, ctor, mkw, bkw, *, device, repeat, warmup, skip_reference):
    m = ctor(**mkw)
    want = m.ref(m.numpy_inputs())
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    rows = []

    if not skip_reference:
        ref_inputs = dict(m.inputs)
        got = run_named(m.region, ref_inputs, m.names)[m.out].to_numpy(order=m.order)
        np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)
        def ref_call():
            run_named(m.region, ref_inputs, m.names)

        rows.append(("reference", _time(ref_call, repeat=repeat, warmup=warmup, sync=lambda: None)))

    tin = _param_inputs(m)
    got = run_named_torch(m.region, tin, m.names, device=device)[m.out].numpy(order=m.order)
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)
    tin_dev = {k: torch.as_tensor(v).to(device) for k, v in tin.items()}
    def trans_call():
        run_named_torch(m.region, tin_dev, m.names, device=device)

    rows.append(("translated", _time(trans_call, repeat=repeat, warmup=warmup, sync=sync)))

    if name in BASELINES:
        binp = {k: torch.as_tensor(v).to(device=device) for k, v in m.numpy_inputs().items()}
        got = BASELINES[name](binp, **bkw).cpu().numpy()
        np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)
        rows.append(("idiomatic", _time(lambda: BASELINES[name](binp, **bkw), repeat=repeat, warmup=warmup, sync=sync)))

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", choices=sorted(PROFILES), default="toy")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--entries", default=None, help="comma-separated subset")
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--skip-reference", action="store_true", help="drop the numpy column (scale past its memory)")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("no CUDA device available")
    entries = PROFILES[args.profile]
    if args.entries:
        entries = {k: entries[k] for k in args.entries.split(",")}

    print(f"profile={args.profile} device={args.device} torch={torch.__version__} median of {args.repeat}")
    print(f"{'entry':10s} {'column':11s} {'ms/iter':>10s} {'vs reference':>13s}")
    for name, (ctor, mkw, bkw) in entries.items():
        rows = bench_entry(
            name, ctor, mkw, bkw,
            device=args.device, repeat=args.repeat, warmup=args.warmup,
            skip_reference=args.skip_reference,
        )
        base = rows[0][1] if rows and rows[0][0] == "reference" else None
        for col, ms in rows:
            speed = f"{base / ms:10.1f}x" if base is not None else f"{'—':>11s}"
            print(f"{name:10s} {col:11s} {ms:10.3f} {speed}")


if __name__ == "__main__":
    main()
