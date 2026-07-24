# 210 — Backend notes: distilled knowledge for the L4-era builders

**Status: distillation (200 §3.3).** The device backends (C, WebGPU/WGSL)
were deleted at migration P1 — deliberately, with their learnings recorded
here. Whoever builds the fresh L4-era backends (CUDA, Metal, WebGPU) starts
from this page and the two reference executors, not from old code.

## Numeric policy (enforced on BOTH sides of every differential test)

- **Integer division/modulo are TRUNCATING** (C semantics), never Python's
  floored `//`/`%`. The reference twin computes them with exact integer
  helpers — routing through float division loses exactness past 2^53 (a
  review-caught silent-wrongness bug):
  `_tdiv(a,b) = q+1 if (q:=a//b) < 0 and q*b != a else q`;
  `_tmod(a,b) = a - _tdiv(a,b)*b`. These live in the reference evaluator's
  preamble today; every device backend must match them bit-for-bit.
- **Float modulo is `fmod`** (sign of the dividend) on both sides — C's `%`
  does not compile for doubles, Python's `%` is floored; both were caught by
  tests, neither is the policy.
- **u64 constants refuse** on targets whose literal range cannot carry them;
  inf/nan constants refuse at rendering (`repr(inf)` is not a literal — the
  reference spells `float('inf')` explicitly where a type allows it at all).
- f32 computes as f64 on the reference (Python has one float); narrowing
  becomes real per-device via a declared type map — never silently.

## The artifact carries its own contract

The grid-family bug class (150 F-series: dtype mismatch writing 8N bytes
into 4N; non-contiguous adoption clobbering neighbors; rank mismatch reading
past `dom[]`) had ONE root fix: **the compiled artifact carries its
input/output contract** (element kind, rank — as a header/metadata the
launcher parses), and **the launcher enforces it** (dtype, contiguity, rank
refusals) before any pointer crosses the ABI. Rebuild this pattern in every
device backend: contracts on artifacts, enforcement at launch, refusals that
quote the fix. The 200-era version generalizes it: boundary descriptors
(Buffer + Layout + Encoding) ARE the contract.

## WebGPU runtime learnings (measured on M3, step 9–10 era)

- **Synchronous readback is a fixed-latency protocol act, not bandwidth**:
  ~1.6 ms from 64² to 1024² — the submit→wait→map round-trip dominates and
  does not scale with size. Async/persistent-surface paths are where that
  cost dies; never benchmark a compute path through a sync readback and
  attribute the time to the kernel.
- **Timestamp queries** (begin/end-of-pass) are the honest GPU timer;
  request the feature at device creation when available; clamp tick deltas
  at ≥0 (drivers may report non-monotonic pass timestamps); cache the query
  set/buffer on the program object.
- **Encode and submit are separate acts** — one `_encode_frame` shared
  between the timed and untimed paths so they cannot drift, and the
  *encodable* is the API surface (the host owns passes and submits — the
  200 graphics tier's deliverable is a render bundle / draw-into-pass).
- **Uniform-buffer plan**: staging members are slot-format-typed (f32/i32/
  u32; bool reads `!= 0u` — bool is not host-shareable), members FROM the
  plan (hole-free), reserved words prefixed. The plan IS the ABI; both
  renderer and launcher read it, neither invents layout.
- Workgroup size is pipeline-creation-time (value-specialized bracket);
  dispatch dimensions are launcher data.

## Instrumentation methodology (bench, deleted with its demo consumers)

BenchmarkTools-style adaptive micro-benchmarking: warmup, tune
evals-per-sample above a minimum-resolution floor, sample to a time budget,
**minimum as the headline estimator** (noise is one-sided). Phase
decomposition by SEAM-WRAPPING (FastRecord.extract/.launch are plain fields
— instruments are temporary shims restored in `finally`), never by editing
the dispatch path. Wall-clock CI gates are retry-once shaped: a real
regression fails twice; a scheduler blip does not.

## The aliasing lesson (carried as a day-one test at P7)

A writable output overlapping a readable capture/argument is **silent
corruption** (verified by execution, twice, in the 150 review). The store
seam refuses overlap at dispatch (`shares_memory` over the leaves) with the
ping-pong message; in-place returns only ever as an L2-certified rewrite.
This is a test to write the day the store path exists — not a memory.

## The refusal voice (seeded as the joint battery at P3)

One shape: **what happened, the principle violated, the quoted fix, the
source location.** Refusal messages are frozen behavior (the refusal
contract battery pins them by wording); a drifted message is an API break.
The oracle rule rides with it: per-element host dispatch is
debug/oracle-grade; reference execution is always spelled
(`reference(f)(...)`); a plain call on an unrouted kind refuses — it never
silently interprets.
