# spike_controlflow — findings

(Written by the session lead from the spike agent's report; harness
blocks subagents writing report files. All code rerunnable:
`/Users/nehal/src/pdum_dsl/.venv/bin/python -B <script>.py` from this
dir; `_paths.py` puts the worktree packages on sys.path.)

## Verdict

The select-normal-form position **holds for branches and fails for
loops**. Select costs exactly nothing when arms are cheap (1.000×
measured, divergent AND coherent). It costs **2.2–4.0×** when used to
flatten a data-dependent TRIP COUNT — precisely the case the splicer
refuses today. If-reconstruction is cheap to build (~135 LOC),
provably correct (bitwise-identical), and worth nothing on today's
kernel shapes because pointwise kernels are store-bandwidth bound.
The exposure to fix is loops, not branches.

## Method

Apple M3 Ultra / Metal / wgpu-py 0.31.1, offscreen. Timestamp-query
feature requested at device creation — every number is a GPU
begin/end-of-pass duration, never wall clock through a sync readback
(210). Estimator: minimum of 25 reps, each rep its own submit + query
pair. Timer validated: doubling kernel work doubles duration (1.94,
2.00); long workloads approach amortized wall clock from below with a
~0.17 ms/submit encode gap.

## Part 1(a): cheap arms — parity is exact

`bench_cheap.py`: 4M threads, 256 inner iterations, 2–3 ALU ops per
arm; divergence CONTROLLED by per-thread bit patterns (same shader,
fully divergent vs wave-coherent, identical instruction counts). All
variants bitwise-identical before timing.

| case | regime | select (ms) | if (ms) | if/select |
|---|---|---|---|---|
| cheap-2op | divergent | 1.483 | 1.483 | 1.000 |
| cheap-2op | coherent | 1.483 | 1.483 | 1.000 |
| clamp | divergent | 1.639 | 1.638 | 1.000 |
| clamp | coherent | 1.639 | 1.639 | 1.000 |
| heavy-CONTROL | divergent | 2.474 | 2.517 | 1.017 |
| heavy-CONTROL | coherent | 2.474 | 1.318 | **0.533** |

The hardware compiler if-converts cheap branches back into selects.
heavy-CONTROL (~24-op arms) is a methodology control: the rig sees a
difference when one exists (0.533× coherent), so the 1.000s are a
measurement, not a blind spot. Incidental: hand-written select pairs
beat the `clamp()` builtin by 1.2%.

## Part 1(b): SDF raymarcher — where flattening costs

`bench_sdf.py`: sphere-traced SDF (plane+sphere+torus+box), 64 fixed
steps. A = all 64 steps, done-mask + frozen ray parameter (what
select-normal-form lowers to). B = real `break`. Numerically identical
BY CONSTRUCTION (asserted bitwise before timing); only the camera
varies, so convergence profile is the single variable.

A/B ratios:

| scene | 512² | 1024² | 2048² | 4096² | step profile |
|---|---|---|---|---|---|
| close | 2.92 | 3.52 | 3.83 | 3.97 | mean 16.6, p90 30, 3.2% saturated |
| grazing | 2.25 | 2.52 | 2.65 | 2.69 | mean 25.2, p90 61, 9.5% saturated |
| wide | 2.83 | 3.27 | 3.57 | 3.67 | mean 18.0, p90 41, 2.7% saturated |
| deep-CONTROL | 1.08 | 1.08 | 1.09 | 1.09 | mean 57.3, p90 64, 85.2% saturated |

Trustworthiness: (1) A is scene-independent to three digits at every
size (fixed work predicts exactly that); (2) the ratio tracks
64/mean-steps almost exactly (close: 3.86 predicted vs 3.97 measured)
— early exit retires near-whole waves; spatial coherence in a raymarch
is strong and the flat form throws all of it away; (3) deep-CONTROL
pins the interpretation: 85% of threads burn all 64 steps → ratio
1.09, so `break`'s fixed overhead is ~8–9% and the 2.2–4.0× is
skipped work, not artifact.

"Divergence recombines quickly" is TRUE for a branch (one arm, one
join) and FALSE for a data-dependent loop: lanes never recombine, they
run out of iterations at different times, and the flat form pays for
the slowest lane in every lane.

## Part 2: if-reconstruction

The IR really flattens: three `@compute` kernels whose fn-argument is
a `@jit` fn with real Python `if` statements compile to regions with
**zero** `core.if` (arms became `tl.pointwise f="where"`).

The pass (`ifrecon.py`): a node sinks into arm `slot` of select S
exactly when every use is S-at-that-slot or a node already sunk into
the same arm — a monotone fixpoint over the region DAG; each
(select, arm) is a scope; nested selects nest scopes; every node gets
its deepest exclusive scope. Sound because the value dialect is pure
and floats are non-trapping (210): the not-taken arm's value was
always discarded. **~135 LOC of actual pass** (91 analysis + 44
scoped emission) + 102 driver boilerplate adapted from
`wgsl_executor._translate`; expression rendering reuses
`wgsl_executor._Gen` UNCHANGED so marker tables cannot drift.

Correctness (`check_ifrecon.py`, three ways on identical inputs —
numpy reference / today's flat WGSL / reconstructed WGSL):

| subject | nodes | selects | sunk | scope depth | recon vs flat |
|---|---|---|---|---|---|
| band | 15 | 1 | 3 (20%) | 1 | BITWISE EQUAL |
| fat | 47 | 1 | 27 (57%) | 1 | BITWISE EQUAL |
| nested | 33 | 2 | 18 (55%) | 2 | BITWISE EQUAL |

`nested` gets every structural case right, including a uniform read
by BOTH inner arms hoisted to the outer-else scope (not the root, not
either arm); `fat` shares a subexpression between condition and both
arms and it stays hoisted.

Does it pay? **No, on today's shapes** (`bench_recon.py`, 2048²):
store floor (same store, zero arithmetic) is 0.098 ms; flat vs recon
with ~20-op and ~90-op arms all land 0.099–0.103 ms — even 90-op arms
add 4 µs over the floor. Pointwise kernels are store-bandwidth bound;
sign of the effect matches Part 1 physics, magnitude is noise.

## Structural obstacles (design-relevant)

1. **`args` is not dataflow.** `tl.iota` carries the lattice tensor as
   args[0] but its WGSL row never touches it — a naive `.args` walk
   drags the output buffer into the DAG (the spike's one silent bug:
   numerically correct, wrong code). Any pass over this IR must let
   the op table, not the arg list, define dataflow.
2. **`walk_region` doesn't descend into `n.regions`** — moot today
   (the splicer flattens `core.if` before artifacts exist), bites the
   moment anything with bounded control flow becomes translatable.
3. **Reconstruction is recovery, not inversion** — the pass will
   equally branch on a select the user never wrote as `if` (a direct
   `where`, or a marker lowering). Policy question for 280.
4. **The scope invariant needs one exemption** — operands emit in an
   ancestor-or-self scope of their consumer EXCEPT across a select's
   arm edges (sinking below is the point); the assertion caught a bad
   first version of the analysis.
5. **Uniformity is a live hazard on the render path** — sinking work
   under a real branch makes control flow non-uniform; compute is safe
   (explicit-LOD sampling, no barriers), but a fragment-stage version
   needs a uniformity check before sinking anything with implicit
   derivatives. Nothing enforces that today.
6. **No cost model** — the spike sinks everything legal; right for a
   spike, wrong for a backend.

## Rig sharp edges (carry to the next spike)

- The LAST timestamp pair in a batch resolves to zero on this driver —
  `gpubench.py` encodes one discard rep and drops samples below 0.75×
  median (without this, one run reported 4.987 ms for a 7.24 ms
  workload).
- The GPU idles low and ramps slowly: the first program timed in a
  process reads ~1.7× high; `warm_gpu()` burns sustained load and the
  matrix is swept twice taking the min.
- `max_compute_workgroups_per_dimension` = 65535 → a 1-D dispatch of
  4M threads needs workgroup size ≥ 128.

## What this means for 280

- **Keep select-normal-form for branches.** 1.000× measured, both
  regimes. The simplicity is free.
- **Don't build if-reconstruction yet.** Correct, ~135 lines, worth
  0–1% on every shape the splicer can currently produce. Ship it when
  an ALU-bound kernel with expensive arms and a coherent condition
  exists; it'll still be ~135 lines then.
- **The real exposure is loops, at 2.2–4.0×.** A raymarcher is the
  canonical graphics workload and the flat form costs it 3–4× at
  2048²+. Our IR doesn't even reach that case: `_liftable`
  (kernel.py:597) rejects any nested region except `core.if`, so a
  `for` in a `@jit` body drops the whole function to the per-element
  oracle path, which no device can translate. The design question is
  not "should the backend reconstruct if" — it is **"what does a
  bounded loop with an early exit look like in our IR, and can it
  lower to a real `break`."** That's where the 3× lives.

## Files

`gpubench.py` (timing harness) · `bench_cheap.py` · `bench_sdf.py` ·
`kernels.py` (Part 2 subjects; dumps regions) · `ifrecon.py` (the
pass) · `check_ifrecon.py` (three-way differential; `--show` prints
WGSL) · `bench_recon.py` (device flat-vs-recon + store floor) ·
`_paths.py`.
