"""Part 1(b) — EXPENSIVE ARMS: a sphere-traced SDF raymarcher, 64 steps.

Variant A (straight-line): every thread runs all 64 steps; termination is
a `done` mask and the ray parameter freezes via select. This is what our
IR's select-normal-form lowers to today.
Variant B (early-exit): `break` on hit or escape.

The two are NUMERICALLY IDENTICAL by construction, not by luck: once A's
`done` is set, `t` stops advancing, so map() recomputes the same distance
forever and every masked update is a no-op. The script asserts bitwise
agreement on both outputs before reporting a single time.

Both variants carry the same per-iteration step counter (A's is a masked
select, B's a plain add) so the loop bodies differ ONLY in the control
construct -- and A's extra select is exactly the cost of straight-line
form, so leaving it in is the honest comparison.

The scene is ONE map() across all runs; only the CAMERA changes, so the
instruction mix is fixed and the convergence profile is the variable.
The profile is not asserted -- it is measured (mean/p90/saturated share
of the step counts, printed with every row).
"""

from __future__ import annotations

import numpy as np
from gpubench import Program, device, has_timestamps, warm_gpu

STEPS = 64
WG = 8

_SCENE = """
@group(0) @binding(0) var<storage, read_write> out_t: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_s: array<f32>;

fn sdSphere(p: vec3<f32>, c: vec3<f32>, r: f32) -> f32 { return length(p - c) - r; }
fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3<f32>(0.0, 0.0, 0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}
fn sdTorus(p: vec3<f32>, t: vec2<f32>) -> f32 {
  let q = vec2<f32>(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}
fn map(p: vec3<f32>) -> f32 {
  var d = p.y + 1.0;                                             // ground plane
  d = min(d, sdSphere(p, vec3<f32>(0.0, 0.0, 0.0), 1.0));
  d = min(d, sdTorus(p - vec3<f32>(2.5, 0.0, 0.0), vec2<f32>(0.8, 0.25)));
  d = min(d, sdBox(p - vec3<f32>(-2.5, 0.0, 0.0), vec3<f32>(0.7, 0.7, 0.7)));
  return d;
}

@compute @workgroup_size(WG, WG, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= RESu || gid.y >= RESu) { return; }
  let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / RES.0 * 2.0 - 1.0;
  let ro = vec3<f32>(ROX, ROY, ROZ);
  let ta = vec3<f32>(TAX, TAY, TAZ);
  let fwd = normalize(ta - ro);
  let rgt = normalize(cross(fwd, vec3<f32>(0.0, 1.0, 0.0)));
  let upv = cross(rgt, fwd);
  let rd = normalize(fwd * 1.6 + rgt * uv.x + upv * uv.y);

  var t: f32 = 0.0;
  var res: f32 = -1.0;
  var steps: f32 = 0.0;
BODY
  let i = gid.y * RESu + gid.x;
  out_t[i] = res;
  out_s[i] = steps;
}
"""

# A: no break -- 64 steps always, termination is a mask and a frozen `t`.
BODY_STRAIGHT = """
  var done: bool = false;
  for (var k: i32 = 0; k < STEPS; k = k + 1) {
    let d = map(ro + rd * t);
    steps = select(steps + 1.0, steps, done);
    let hit = (!done) && (d < EPS);
    let esc = (!done) && (d >= EPS) && (t > TMAX);
    res = select(res, t, hit);
    done = done || hit || esc;
    t = select(t + d, t, done);
  }"""

# B: the real early exit.
BODY_BREAK = """
  for (var k: i32 = 0; k < STEPS; k = k + 1) {
    let d = map(ro + rd * t);
    steps = steps + 1.0;
    if (d < EPS) { res = t; break; }
    if (t > TMAX) { break; }
    t = t + d;
  }"""

# name -> (ro, ta, eps, tmax). The convergence profile each produces is
# MEASURED (printed per row), never asserted.
SCENES = {
    "close": ((0.0, 0.6, 3.2), (0.0, 0.0, 0.0), 1e-3, 40.0),
    "grazing": ((0.0, -0.85, 7.0), (0.0, -0.9, 0.0), 1e-3, 40.0),
    "wide": ((4.0, 3.0, 6.0), (0.0, 0.0, 0.0), 1e-3, 40.0),
    # The CONTROL: a tolerance no ray reaches and a horizon none escapes, so
    # essentially every thread burns all 64 steps. Early exit has nothing to
    # exit early FROM here, so A/B must be ~1.0 -- that pins the ratios above
    # to the work skipped, not to any fixed overhead of the `break` form.
    "deep-CONTROL": ((4.0, 3.0, 6.0), (0.0, 0.0, 0.0), 1e-9, 1e9),
}


def _src(body: str, res: int, scene) -> str:
    ro, ta, eps, tmax = scene
    s = _SCENE.replace("BODY", body)
    for k, v in (
        ("RESu", f"{res}u"), ("RES.0", f"{float(res)}"),
        ("STEPS", str(STEPS)), ("EPS", repr(eps)), ("TMAX", repr(tmax)),
        ("WG", str(WG)),
        ("ROX", repr(ro[0])), ("ROY", repr(ro[1])), ("ROZ", repr(ro[2])),
        ("TAX", repr(ta[0])), ("TAY", repr(ta[1])), ("TAZ", repr(ta[2])),
    ):
        s = s.replace(k, v)
    return s


def _profile(steps: np.ndarray) -> str:
    f = steps.ravel()
    return (f"mean {f.mean():5.1f}  p90 {np.percentile(f, 90):5.1f}  "
            f"{100 * (f >= STEPS).mean():5.1f}% saturated")


def run(sizes=(512, 1024, 2048, 4096), sweeps=2):
    """The full matrix, swept `sweeps` times, minimum taken across sweeps.
    One sweep is not enough: program-creation order leaves a clock-ramp
    residue on whichever row goes first."""
    warm_gpu()
    best, prof, order = {}, {}, []
    for _ in range(sweeps):
        for name_s, scene in SCENES.items():
            for res in sizes:
                key = (name_s, res)
                if key not in best:
                    order.append(key)
                grid = ((res + WG - 1) // WG, (res + WG - 1) // WG, 1)
                zeros = [np.zeros((res, res), np.float32)] * 2
                out, times = {}, {}
                for name, body in (("straight", BODY_STRAIGHT), ("break", BODY_BREAK)):
                    p = Program(_src(body, res, scene), zeros)
                    times[name] = p.time_ms(grid, reps=25, warmup=8)["min"]
                    out[name] = (p.read(0), p.read(1))
                for buf, label in ((0, "res"), (1, "steps")):
                    a, b = out["straight"][buf], out["break"][buf]
                    if not np.array_equal(a, b):
                        bad = int((a != b).sum())
                        raise AssertionError(
                            f"{name_s}/{res} {label}: {bad}/{a.size} differ, "
                            f"max |d| = {np.nanmax(np.abs(a - b))}"
                        )
                prev = best.get(key)
                best[key] = (
                    (min(prev[0], times["straight"]), min(prev[1], times["break"]))
                    if prev else (times["straight"], times["break"])
                )
                prof[key] = _profile(out["break"][1])
    return [(n, r, *best[(n, r)], prof[(n, r)]) for n, r in order]


def report(rows):
    print(f"\n=== Part 1(b) SDF raymarcher, {STEPS} fixed steps ===")
    print(f"timer: {'timestamp-query' if has_timestamps() else 'submit-wait wall clock'}, "
          "minimum of 25; straight and break verified BITWISE identical\n")
    hdr = f"{"scene":<13} {"res":>6} {'A straight':>11} {'B break':>9} {'A/B':>6}   step profile"
    print(hdr)
    print("-" * len(hdr))
    for cam, res, ta, tb, prof in rows:
        print(f"{cam:<13} {res:>5}² {ta:10.3f}m {tb:8.3f}m {ta / tb:6.2f}   {prof}")


if __name__ == "__main__":
    print("adapter:", device()._spike_adapter_info["device"])
    report(run())
