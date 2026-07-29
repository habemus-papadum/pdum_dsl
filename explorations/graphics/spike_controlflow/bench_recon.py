"""Does reconstruction PAY? The flat and reconstructed translations of the
SAME artifact, both timed on device with timestamp queries.

This closes the loop between Part 1 and Part 2: Part 1 measured
handwritten select vs `if`; this measures the two shaders our own backend
emits for one kernel, over a real lattice, under two condition regimes:

  coherent  -- the argument is a smooth ramp, so the condition is a
               half-plane and whole workgroups agree. Divergence
               "recombines quickly" here in the strongest sense: it never
               diverges within a wave at all except on one boundary line.
  divergent -- the argument is per-pixel noise, so neighbouring lanes take
               opposite arms and no wave can skip either one.

Both shaders must produce bitwise-identical images; the script checks that
before reporting.
"""

from __future__ import annotations

import struct
from dataclasses import replace

import _paths  # noqa: F401
import numpy as np
from gpubench import Program, has_timestamps, warm_gpu
from ifrecon import translate
from kernels import T

from pdum.dsl import jit
from pdum.tl import compute, f32, global_idx
from pdum.tl.kernel import _compile
from pdum.tl.markers import exp, sin, sqrt, tanh
from wgsl_executor import _translate as flat_translate

RES = 2048


def heavy(lo, ka, kb):
    """Expensive arms (~20 ops, 6 transcendentals each) over a shared `v`.
    The captures ka/kb are abi slots, so each arm also owns an exclusive
    uniform load -- a leaf the flat form must issue unconditionally."""

    @jit()
    def go(v):
        if v > lo:
            y = (
                exp(v * ka) * sqrt(v * v + ka)
                + tanh(v * ka - 1.0) * ka
                + exp(v * 0.3) * tanh(v * 0.7)
                + sqrt(tanh(v * ka) * tanh(v * ka) + ka)
            )
        else:
            y = (
                exp(-v * kb) * sqrt(v * v + kb)
                - tanh(v * kb + 1.0) * kb
                - exp(-v * 0.4) * tanh(v * 0.9)
                - sqrt(tanh(v * kb) * tanh(v * kb) + kb)
            )
        return y

    return go


def veryheavy(lo, ka, kb):
    """The ALU-BOUND end: ~10 chained statements per arm (each a
    transcendental plus arithmetic), straight-line so `_liftable` still
    splices it. Written out rather than looped -- a `for` in a @jit body
    becomes a region and `_liftable` drops the whole function to the
    oracle path."""

    @jit()
    def go(v):
        if v > lo:
            a = tanh(v * ka)
            a = a + exp(a * ka) * ka
            a = a + sqrt(a * a + ka)
            a = a + tanh(a * ka - 1.0)
            a = a + exp(a * 0.3) * tanh(a * 0.7)
            a = a + sqrt(a * a + ka) * ka
            a = a + tanh(a * 0.9) * exp(a * 0.2)
            a = a + sqrt(tanh(a * ka) * tanh(a * ka) + ka)
            a = a + exp(a * 0.15) - tanh(a * ka)
            y = a + sqrt(a * a + 1.0)
        else:
            b = tanh(v * kb)
            b = b - exp(-b * kb) * kb
            b = b - sqrt(b * b + kb)
            b = b - tanh(b * kb + 1.0)
            b = b - exp(-b * 0.4) * tanh(b * 0.9)
            b = b - sqrt(b * b + kb) * kb
            b = b - tanh(b * 0.6) * exp(-b * 0.25)
            b = b - sqrt(tanh(b * kb) * tanh(b * kb) + kb)
            b = b - exp(-b * 0.12) + tanh(b * kb)
            y = b - sqrt(b * b + 1.0)
        return y

    return go


@compute
def coherent(f, img):
    i, j = global_idx("y", "x")
    img[i, j] = f(f32(i) * 0.002 + f32(j) * 0.002 - 2.0)  # a smooth half-plane


@compute
def divergent(f, img):
    i, j = global_idx("y", "x")
    img[i, j] = f(sin(f32(i) * 12.9898 + f32(j) * 78.233) * 4.0)  # per-pixel noise


def _uniform_values(art, f, img):
    """The staging bytes the launcher would pack, captured through the
    executor seam rather than re-implemented (210: the plan IS the ABI --
    neither side invents layout)."""
    grabbed = {}
    replace(art, executor=lambda values, staging: grabbed.setdefault("s", staging)).launch(
        (f, img), {}
    )
    _, meta = flat_translate(art)
    return np.asarray(
        [struct.unpack_from(fmt, grabbed["s"], off)[0] for _, off, fmt in meta["slots"]],
        dtype=np.float32,
    )


def one(name, kernel, f, tag=""):
    img = T(np.zeros((RES, RES)))
    art = _compile(kernel.fn, (f, img))
    uni = _uniform_values(art, f, img)
    src_flat, _ = flat_translate(art)
    src_recon, meta = translate(art)

    bufs = [np.zeros((RES, RES), np.float32), uni]
    grid = ((RES + 7) // 8, (RES + 7) // 8, 1)
    progs = {"flat": Program(src_flat, bufs), "recon": Program(src_recon, bufs)}
    t = {k: p.time_ms(grid, reps=25, warmup=8)["min"] for k, p in progs.items()}
    imgs = {k: p.read(0) for k, p in progs.items()}
    if not np.array_equal(imgs["flat"], imgs["recon"]):
        raise AssertionError(
            f"{name}: flat and recon images differ, max |d| "
            f"{np.nanmax(np.abs(imgs['flat'] - imgs['recon']))}"
        )
    st = meta["stats"]
    return (f"{tag}/{name}", t["flat"], t["recon"], st,
            len(src_flat.splitlines()), len(src_recon.splitlines()))


_TRIVIAL = """
@group(0) @binding(0) var<storage, read_write> buf0: array<f32>;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.y >= RESu || gid.x >= RESu) { return; }
  buf0[i32(gid.y) * RES + i32(gid.x)] = f32(gid.x) + f32(gid.y);
}
"""


def store_floor() -> float:
    """The same lattice, same store, NO arithmetic. Everything above this
    is what the arms actually cost -- without it a ratio of 1.0 cannot be
    told apart from 'the branch is free' and 'the kernel is not compute
    bound in the first place'."""
    src = _TRIVIAL.replace("RESu", f"{RES}u").replace("* RES", f"* {RES}")
    p = Program(src, [np.zeros((RES, RES), np.float32)])
    return p.time_ms(((RES + 7) // 8, (RES + 7) // 8, 1), reps=25, warmup=8)["min"]


def main():
    warm_gpu()
    floor = store_floor()
    rows = []
    for tag, f in (("~20-op arms", heavy(0.0, 0.37, 0.61)),
                   ("~90-op arms", veryheavy(0.0, 0.37, 0.61))):
        for name, kernel in (("coherent", coherent), ("divergent", divergent)):
            rows.append(one(name, kernel, f, tag))
    print(f"\n=== Part 2 on device — {RES}² lattice, one @compute kernel, "
          f"two translations of the SAME artifact ===")
    print(f"timer: {'timestamp-query' if has_timestamps() else 'wall'}, minimum of 25; "
          "flat and recon images verified BITWISE identical")
    print(f"store floor (same lattice, no arithmetic): {floor:.3f} ms — "
          f"everything below is arms ON TOP of that\n")
    hdr = (f"{'arms / condition':<24} {'flat ms':>9} {'recon ms':>9} {'recon/flat':>11}  "
           f"{'ALU>floor':>10}  {'sunk':>9}  {'lines':>10}")
    print(hdr)
    print("-" * len(hdr))
    for name, tf, tr, st, lf, lr in rows:
        print(f"{name:<24} {tf:9.3f} {tr:9.3f} {tr / tf:11.3f}  "
              f"{tf - floor:9.3f}m  {st['sunk']:>3}/{st['nodes']:<4}  {lf:>4}->{lr:<4}")


if __name__ == "__main__":
    main()
