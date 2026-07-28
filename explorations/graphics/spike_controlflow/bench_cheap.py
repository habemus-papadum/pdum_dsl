"""Part 1(a) — CHEAP ARMS: select vs a real `if`, 1-3 ops per arm.

The claim under test: when both arms are a couple of ALU ops, a real
`if` buys nothing, because the hardware shader compiler if-converts it
back into a select anyway. If that holds, our IR's select-normal-form
costs nothing here and a backend has no reason to reconstruct.

Divergence is CONTROLLED, not incidental: each thread carries a 32-bit
pattern and consumes one bit per iteration (rotate), so the same shader
runs a fully divergent regime (random bits per lane) and a wave-coherent
regime (one pattern shared by every lane in a workgroup) with identical
instruction counts. The per-iteration bit extraction (shift/and/cmp) is
byte-identical in both variants, so it cancels out of the A/B ratio.

The two variants must agree BITWISE — same arithmetic, same order — and
the script asserts that before reporting any time.
"""

from __future__ import annotations

import numpy as np
from gpubench import Program, calibrate, device, has_timestamps

N = 1 << 22  # 4M threads
K = 256  # inner iterations
WG = 128  # 65535 workgroups is the per-dimension dispatch cap; 4M/128 fits

_PRELUDE = """
@group(0) @binding(0) var<storage, read> seed: array<f32>;
@group(0) @binding(1) var<storage, read> pattern: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn armA(x0: f32) -> f32 {  // the CONTROL's arms: ~24 ops, 8 transcendentals
  var y = x0;
  for (var j: i32 = 0; j < 8; j = j + 1) { y = sin(y * 1.13 + 0.31) * 0.97; }
  return y;
}
fn armB(x0: f32) -> f32 {
  var y = x0;
  for (var j: i32 = 0; j < 8; j = j + 1) { y = cos(y * 1.07 - 0.19) * 0.93; }
  return y;
}

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  var x: f32 = seed[i];
  var bits: u32 = pattern[i];
  for (var k: i32 = 0; k < K; k = k + 1) {
    let c: bool = (bits & 1u) != 0u;
    bits = (bits >> 1u) | (bits << 31u);
BODY
  }
  out[i] = x;
}
"""

# --- (a1) a two-armed cheap update: 2 ops per arm -----------------------------
BODY_SELECT = """    x = select(x * 0.9991 - 0.29, x * 1.0009 + 0.31, c);"""
BODY_IF = """    if (c) { x = x * 1.0009 + 0.31; } else { x = x * 0.9991 - 0.29; }"""

# --- (a2) a clamp shape: min/max written three ways ---------------------------
CLAMP_STEP = """    x = x * 1.03 + select(-0.37, 0.41, c);"""
BODY_CLAMP_SELECT = CLAMP_STEP + """
    x = select(x, -2.0, x < -2.0);
    x = select(x,  2.0, x >  2.0);"""
BODY_CLAMP_IF = CLAMP_STEP + """
    if (x < -2.0) { x = -2.0; }
    if (x >  2.0) { x =  2.0; }"""
BODY_CLAMP_BUILTIN = CLAMP_STEP + """
    x = clamp(x, -2.0, 2.0);"""

# --- (a3) the METHODOLOGY CONTROL: expensive arms ----------------------------
# Not a claim about our IR — a check that this rig can SEE a difference when
# one exists. With ~24-op arms, `if` must beat `select` on a wave-coherent
# condition (half the work skipped) and tie it when divergent (both arms run).
BODY_HEAVY_SELECT = """    x = select(armA(x), armB(x), c);"""
BODY_HEAVY_IF = """    if (c) { x = armB(x); } else { x = armA(x); }"""

_K = {"cheap-2op": 256, "clamp": 256, "heavy-CONTROL": 16}


def _src(body: str, k: int) -> str:
    return _PRELUDE.replace("BODY", body).replace("WG", str(WG)).replace("K", str(k))


def _inputs(regime: str, rng):
    seed = rng.uniform(-1.0, 1.0, N).astype(np.float32)
    if regime == "divergent":  # every lane its own pattern: full wave divergence
        pat = rng.integers(0, 2**32, N, dtype=np.uint64).astype(np.uint32)
    elif regime == "coherent":  # one pattern per workgroup: every wave uniform
        per_wg = rng.integers(0, 2**32, (N + WG - 1) // WG, dtype=np.uint64).astype(np.uint32)
        pat = np.repeat(per_wg, WG)[:N]
    else:
        raise ValueError(regime)
    return seed, pat


def _prog(body, k, seed, pat):
    return Program(_src(body, k), [seed, pat, np.zeros(N, np.float32)])


def run():
    rng = np.random.default_rng(7)
    grid = (N // WG, 1, 1)
    rows = []
    for regime in ("divergent", "coherent"):
        seed, pat = _inputs(regime, rng)
        for case, variants in (
            ("cheap-2op", [("select", BODY_SELECT), ("if", BODY_IF)]),
            (
                "clamp",
                [
                    ("select", BODY_CLAMP_SELECT),
                    ("if", BODY_CLAMP_IF),
                    ("builtin", BODY_CLAMP_BUILTIN),
                ],
            ),
            ("heavy-CONTROL", [("select", BODY_HEAVY_SELECT), ("if", BODY_HEAVY_IF)]),
        ):
            outs, times = {}, {}
            for name, body in variants:
                p = _prog(body, _K[case], seed, pat)
                times[name] = p.time_ms(grid)
                outs[name] = p.read(2)
            ref = outs["select"]
            for name, o in outs.items():
                same = np.array_equal(o, ref)
                maxdiff = float(np.nanmax(np.abs(o - ref))) if not same else 0.0
                assert same or maxdiff < 1e-4, f"{case}/{regime}/{name} diverged: {maxdiff}"
                if not same:
                    print(f"  note: {case}/{regime}/{name} not bitwise-equal (max {maxdiff:.2e})")
            for name, _ in variants:
                rows.append((case, regime, name, times[name]["min"], times[name]["med"]))
    return rows


def report(rows):
    print(f"\n=== Part 1(a) cheap arms — N={N} threads, K={K} inner iterations ===")
    print(f"timer: {'timestamp-query' if has_timestamps() else 'submit-wait wall clock'}, minimum of 25\n")
    print(f"{'case':<10} {'regime':<10} {'variant':<9} {'min ms':>9} {'med ms':>9} {'ratio vs select':>16}")
    base = {}
    for case, regime, name, mn, md in rows:
        if name == "select":
            base[(case, regime)] = mn
    for case, regime, name, mn, md in rows:
        r = mn / base[(case, regime)]
        print(f"{case:<14} {regime:<10} {name:<9} {mn:9.3f} {md:9.3f} {r:16.3f}")


if __name__ == "__main__":
    print("adapter:", device()._spike_adapter_info["device"])
    print(calibrate())
    report(run())
