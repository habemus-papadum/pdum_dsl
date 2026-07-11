# J2 — Judgment: the three proposals through the PERFORMANCE lens

*Judge report for the pdum.dsl redesign bake-off. Lens: hot-loop per-call cost in
pure Python (actual operations on a cache hit), compile-time latency, cache-key
construction cost, marshaling overhead, dispatch scaling. Inputs: P1–P3 (full
proposals), V1–V5 (verdicts), `docs/desiderata.md`,
`design/dsl_caching_layer.md`. July 2026.*

The three proposals converge on ~90% of the machinery (five surfaces, one
Registry, Node/Region micro-IR, two-tier cache, FastRecord, three region ops),
so this judgment lives in the remaining 10%: what *actually executes* on a hit,
what each proposal quietly leaves out of its per-call count, and what a miss
costs when live-coding forces recompiles.

---

## 1. Scoring table

Criteria: (a) prime-directive extensibility, (b) caching-thesis fidelity,
(c) hot-loop cost, (d) multi-backend fit, (e) transformation readiness,
(f) smallness/honesty of the line budget, (g) marshaling story,
(h) implementability/risk. 1–10 each.

| Criterion | P1 jax-school | P2 mlir-school | P3 nanopass-school |
|---|---:|---:|---:|
| (a) extensibility | 8 | 8 | **9** |
| (b) thesis fidelity | **9** | 7 | 8 |
| (c) hot-loop cost | 8 | 7 | **9** |
| (d) multi-backend fit | 8 | 8 | 8 |
| (e) transformation readiness | **9** | 8 | 8 |
| (f) LOC smallness/honesty | 5 | **9** | 7 |
| (g) marshaling story | 7 | **9** | 7 |
| (h) implementability/risk | 7 | 8 | **9** |
| **Total** | **61** | **64** | **66** |

**Winner through the performance lens: P3 (nanopass-school)** — narrowly, and
only after grafting two mechanisms it hand-waves (dependency guards from P1, the
leaves-to-launch channel from P2; §4).

---

## 2. The hit path, operation by operation

The lens question: what does one hot-loop iteration (rebuild closure + call)
actually execute? For a kernel with C captures, A arguments, L physical leaves,
G dependency-closure entries.

### P1 (jax-school, §4 of P1)

Phase A: 2 attr reads + C `safe_cell` + C fingerprints + `FnT` alloc + `Handle`
alloc. Hit path:

1. **Key build**: 6-tuple `(fn_type.template, env_fp, fingerprint_tuple(args),
   BE.token, BE.params, registry.generation)` — A fingerprint calls + tuple
   alloc.
2. One `THESIS.get` probe.
3. `_guards_ok(rec.guards)` — G pointer (`is`) compares.
4. `rec.extract(env, args, rec.staging)` — **one fused closure**: L leaf reads +
   L pack writes, no intermediate list.
5. `rec.launch(rec.staging)`.

Two performance findings, one good, one bad:

- **Good — the only complete count.** P1 is the only proposal that prices
  call-time dependency-drift checking (V1 implication 8: "check at call time and
  refuse/recompile on drift") into the hit path, as a precomputed
  `FastRecord.guards` tuple of `is`-compares. P2 and P3 claim the same
  correctness property and pay for it nowhere (below). P1's quoted budget is the
  honest one; the other two under-quote.
- **Bad — the un-memoized hash in the key.** The dict probe hashes
  `handle.fn_type.template`, a frozen dataclass wrapping the live `CodeType`.
  Frozen dataclasses recompute `__hash__` per call, and CPython does not cache
  code-object hashes (the `co_consts` tuple hash is recomputed every time; only
  the `co_code` bytes hash is cached). Every other component of every proposal's
  key is a precomputed fingerprint; P1 alone puts a live code-object hash on the
  per-call path. Trivially fixable (intern `FnT`, or precompute a template fp on
  the Handle like P2/P3 do) — but as written, P1 has the heaviest key build of
  the three, in the exact place the proposal claims "C-speed attribute reads."

### P2 (mlir-school, §4 of P2)

Phase A: cells + fingerprints + **`dict(zip(co_freevars, vals))` allocation per
rebuilt closure** (P1/P3 use tuples) + Handle. Hit path:

1. **Key build**: 3-tuple `(fp_head, fingerprint_tuple(args), ACTIVE.target_fp)`
   — `fp_head` is memoized on the Handle at phase A. Cheapest key build of the
   three as written.
2. One `_tier1.get` probe.
3. `rec.extract(env, args)` → leaves list (L-element allocation).
4. `rec.plan.pack_into(staging, leaves)` — L `struct.pack_into` calls.
5. `rec.launch(rec.staging, leaves)` — **leaves passed to launch** (correct;
   see §3, marshaling).

**The load-bearing hand-wave: the hot key is missing two components the
proposal's own text requires.** The hit-path key contains no `generation` and no
drift guards — both appear only on the miss path (§4 step 1: "dependency-closure
hash … checked here"). Yet P2's desiderata mapping (§5) claims
"dependency-closure hash checked at call time." These cannot both be true. As
specified: a redefinition (`generation` bump) or a drifted folded global leaves
stale FastRecords servable forever from the fp-keyed tier-1 dict. Either P2 adds
generation + guards to the hit path (cost parity with P1, its lean-key advantage
evaporates), or it specifies a bump-flush protocol (clear/swap tier-1 on
generation bump — actually the *fastest* possible design, paying for the rare
event at the event, but it is nowhere stated, and it does not cover per-call
drift of folded globals at all). P2's hit path looks lean partly because it is
incomplete. Scored accordingly on (b) and (c).

### P3 (nanopass-school, §4.3 of P3)

Phase A (~1–2 µs claimed, plausible): snapshot memo probe + cells + C
`registry.fingerprint` probes + `_FNTYPES` memo probe + Handle alloc. Hit path
(`Handle.__call__`, ~15 lines):

1. **Key build**: 5-tuple `(self._tid_fp, self.env_fp, _fp_tuple(args),
   _ACTIVE_BACKEND_FP, _GENERATION)` — every component either a precomputed
   Handle attribute or a module-level read. **The best key hygiene of the
   three**: complete (generation and backend fp included, unlike P2) and fully
   memoized (no live code-object hash, unlike P1).
2. One `_RECORDS.get` probe.
3. `rec.extract(env, args)` → leaves.
4. `rec.plan.pack_into(staging, leaves)`.
5. `rec.launch(rec.staging)`.

Two hand-waves:

- **Drift is claimed, not checked.** §4.2 records dependency-closure tags at
  miss time and §5 asserts "dependency-closure drift check in the key" — but no
  mechanism in the hit path reads those tags, and nothing bumps `_GENERATION`
  when a folded global's cell drifts. Same gap as P2, stated more quietly. The
  fix is exactly P1's guards tuple (graft §4.1).
- **`_tid_fp` and `env_fp` construction is unshown** — `make_handle` (§4.1)
  builds `env_fp` but never `_tid_fp`; presumably an interned per-code-object
  token from the `_FNTYPES` memo. Minor, but it is the component the whole key
  strategy leans on.

### Verdict on (c)

P3 9 (leanest *complete-able* key; adding guards is a tuple field), P1 8 (fused
extract is the best marshal step; live template hash and guards make it the
slowest key build, honestly), P2 7 (leanest key as written, but two required
components are absent and their addition erases the lead).

---

## 3. Marshaling overhead — and the buffer-leaf hole in P1 and P3

All three implement V4's ValueKind/PackPlan/FastRecord triple. Differences:

- **P1 fuses extract+pack** into one closure-compiled callable that writes
  staging directly — eliminates the per-call L-element leaves list that P2/P3
  allocate. Best scalar-uniform marshal step of the three.
- **P2 alone preserves the leaves channel to the launcher**:
  `rec.launch(rec.staging, leaves)`, matching V4 §3.6 exactly. This is not
  pedantry: **a `BufferLeaf` (captured ndarray / GPU buffer) cannot go through a
  `bytearray` staging buffer.** A CUDA `RawKernel` launch needs the live cupy
  arrays as kernel args *this call*; a WGSL bind group needs the current buffer
  handle. P1 (`rec.launch(rec.staging)`, §4) and P3 (`rec.launch(rec.staging)`,
  §4.3) both drop `leaves` at the launch boundary. Both work for the day-1
  scalar-uniform disk demo and **break structurally the week arrays land** —
  fresh array pointers per iteration have no path to the artifact. Since "one
  logical value → pointer + shape" is a founding requirement (desiderata §1),
  this is a real spec defect in the two leaner proposals, cheap to fix now,
  expensive to discover in M2. Hence (g): P2 9, P1 7, P3 7 (P1's fusion earns
  back what the launch signature loses only partially — a fused extract/pack
  *can* keep a side list for buffer leaves, but P1 doesn't say so).

All three pre-shape the same two escalations (exec-generated per-template
binder, then a native fastpath) behind an unchanged FastRecord contract — good;
no differentiation.

---

## 4. Compile-time latency (the miss path) and dispatch scaling

Misses matter for this project more than for an AOT compiler: live-coding bumps
`generation` and — in all three proposals — the sledgehammer invalidates
*everything*, so edit-latency ≈ (number of live kernels) × (per-miss cost).

- **P3 is the leanest ladder**: fused typing+lowering → (transform) → **one**
  `rewrite()` call combining backend extra_rules + decompositions +
  legalize_params → direct renderer walk (`render_c ≈ 120 lines walking
  regions into C text`). Fewest fixpoint traversals, fewest body rebuilds.
- **P1 is close**: lower → transform → rewrite passes → `Backend.render`
  callable (freeform).
- **P2 is the heaviest by construction**: four mandated stages
  (surface→mid→abi→render), each a RuleSet fixpoint over immutable ordered
  bodies (rebuild-on-change churn), each followed by an always-on legality scan
  (cheap, O(ops) prefix tests, but per-stage), and **rendering itself is a
  RuleSet-driven emitter** rather than a direct recursive walk. At pdum scale
  (tens–hundreds of nodes) this is milliseconds, not seconds — but P2's own risk
  #2 names the body-rebuild churn, and under a generation bump with dozens of
  live kernels the 3–4× stage multiplier is the difference between an
  imperceptible and a perceptible editor hiccup. Acceptable, but P2 buys
  auditability with the slowest recompiles of the three.

Dispatch scaling: all three are one dict probe, O(1) in cache entries — the
torch-guard failure mode (O(entries × guards)) is structurally avoided
everywhere. P1's guards add a per-entry constant (G pointer compares), which is
the correct price for drift safety, not a scaling term.

---

## 5. Line-budget honesty (f) — where the bodies are buried

- **P1: the 165-line lowerer is the proposal's load-bearing fantasy.**
  `kernel/lower.py` (165 LOC) claims parse + coherence check + name
  classification + fused typing+lowering + loc side channel. V1's own
  calibration for exactly this component is **800–1,500 lines** for the M1
  subset (cupy.jit's whole frontend: 1.8k). P1 is 5–10× under, *inside* the
  kernel budget, with no satellite home to grow into — so either the 1010-line
  kernel is fiction, or the lowerer's growth lands as kernel creep (P1's own
  risk R5). Also `marshal.py` at 105 vs V4's ~650-line calibration for the same
  scope. Score 5.
- **P2 is the honest one**: the lowerer is a ~900-line satellite, stated in the
  module map, kept out of the 1050-line kernel by an explicit architectural
  seam (kernel never imports `front/`). The kernel number means what it says.
  Score 9.
- **P3 is in between**: the 135-line `lower.py` is defensible *only because*
  lowering handlers are `("lower_ast")` rules — but the rule *content* for the
  Python subset appears in no LOC row (day-1 slice item 2 references "core
  `lower_ast` rules" with no home), and `registry.py` at 60 lines vs V3's
  750–900-line hook-kernel estimate is a 10× compression, explicitly flagged
  (deviation #3, capped at 150 before revisit) — flagged optimism is better
  than silent optimism, but it is still optimism. Score 7.

---

## 6. Per-proposal critique summary

### P1 — jax-school (61)

The most *complete* hot-path spec (guards, generation, fused extract) and the
deepest transformation readiness (per-aspect rule signatures, the pre-M2
vmap-over-control-flow spike, the ~180-line price tag on any fourth region op).
Its performance sins: the only live (un-memoized) hash in any proposal's
per-call key; the buffer-leaf launch hole shared with P3; and a kernel LOC story
that collapses on contact with V1's own numbers — the 165-line lowerer
guarantees either budget blowout or the exact kernel creep its risk R5 predicts.
Two mechanisms where one would do on day 1 (eval-rules interpreter backend *and*
renderers) also adds miss-path surface without hit-path benefit.

### P2 — mlir-school (64)

The most honest accounting, the only correct launch signature, the cheapest key
build — and an internally contradictory hit path: §5 promises call-time drift
checking that §4's five steps never perform, and `generation` is absent from the
tier-1 probe key entirely. The four-stage ladder with always-on legality and
RuleSet rendering is the slowest miss path of the three; under the shared
generation-sledgehammer invalidation model, that cost multiplies across every
live kernel on every edit. Per-phase-A `dict(zip(...))` env allocation is a
minor but real regression vs. tuple envs. P2's structure is the most auditable;
its per-call numbers are the least trustworthy as written.

### P3 — nanopass-school (66) — winner

The best complete key hygiene (everything precomputed, generation included), the
fewest mechanisms end to end (one rewrite driver, one renderer mechanism, one
combined legalization pass), the fastest miss path, and the strongest mechanical
enforcement of the properties this lens cares about (hit-path microbenchmark
gate, `flatten` allocation budget, `no_compile` mode, extension-locality test —
all CI-armed from the day-1 slice). Its wins are wins of *omission discipline*:
fewer moving parts on both the hit and miss paths. Its defects are omissions
too: drift safety asserted with no checking mechanism, `leaves` dropped at the
launch boundary (breaks buffer marshaling when arrays land), `_tid_fp`
construction unshown, and a flagged-but-real 10× compression of the hook kernel.
Every one of these has a ready graft from a non-winner.

---

## 7. Grafts into the synthesis (from the non-winners)

1. **P1's `FastRecord.guards`** — the precomputed `(cell, expected)` `is`-compare
   tuple checked on every hit. P3 asserts drift safety without paying for it;
   this is the mechanism, and it must be inside the microbenchmark gate so its
   cost stays counted.
2. **P2's `launch(staging, leaves)` signature** — the leaves channel to the
   launcher is the only path by which fresh buffer pointers/shape words reach a
   CUDA/WGSL/Metal launch each call. Graft into P3's FastRecord contract *before*
   arrays land, not after.
3. **P1's fused extract+pack closure** (extract writes staging directly,
   eliding the leaves list for scalar slots) — combined with graft 2: fused path
   for byte-packed slots, side tuple for buffer leaves. Best of both marshal
   steps.
4. **P2's memoized `fp_head`** (template fp + env fp collapsed once at phase A
   into a single Handle attribute) — shortens P3's 5-tuple key further; and make
   P2's implicit protocol explicit: **generation bump ⇒ swap/clear tier-1**, so
   the common-case key need not carry `generation` at all (pay for the rare
   event at the event). If adopted, drift guards (graft 1) remain per-call — they
   cover what bump-flush cannot.
5. **P2's differential key test** — perturb every declared key-relevant ambient
   dimension one at a time and assert the *named* tier of miss. This is the
   key-completeness property (P3 risk 5) turned into a regression test.
6. **P2's always-on stage legality (~30 lines)** — miss-path-only cost, O(ops)
   namespace checks; cheap insurance that never touches the hit path.
7. **P1's fourth-region-op price list** (~180 lines × live transforms, priced
   before acceptance) and the **pre-M2 vmap-over-`if`/`for` spike** — the
   cheapest early retirement of the one risk that could force IR churn later.
8. **P2's honest lowerer accounting** — give P3's `lower_ast` rule content a
   named LOC home (stdlib or a `front/` satellite) sized against V1's 800–1,500
   calibration, so the kernel cap stays credible.

---

## Design lessons for pdum.dsl

1. **The hit-path spec must be complete before it is fast.** Two of three
   proposals quoted lean per-call counts by omitting required work (generation,
   drift guards, the buffer-leaf channel). Adopt the rule: the five-step hit
   path in the synthesis document must enumerate *every* correctness obligation
   from V1/V4 and the hazard doc, with its per-call cost, before any
   microbenchmark number is quoted. An incomplete count is not a budget.
2. **No live hashes in the per-call key.** Every key component must be a
   precomputed fingerprint or an interned token stored on the Handle at phase A.
   Hashing a code object (or any frozen dataclass wrapping one) per call is a
   silent per-iteration tax — P1's as-written key demonstrates how easily it
   slips in. Test: assert the key build performs zero `__hash__` calls on
   non-interned objects (or simply benchmark key-build in isolation in CI).
3. **Pay for rare events at the event, not per call.** Generation-in-key
   (P1/P3) charges every hit for redefinitions that happen at human timescale;
   bump-flush (implicit in P2) charges the redefinition instead. Prefer
   bump-flush, made explicit — and keep per-call guards only for what has no
   publisher (drifted folded globals).
4. **The launch boundary needs two channels**: packed bytes (staging) for
   scalar/uniform slots, live objects (leaves) for buffer/pointer slots. Any
   FastRecord contract with a `launch(staging)`-only signature is a scalar-demo
   artifact that cannot marshal an array capture. Fix the signature before the
   first array lands.
5. **Miss-path latency is a product metric under sledgehammer invalidation.**
   Until the dependency-graph invalidator exists, every edit recompiles every
   live kernel; per-miss cost (number of fixpoint stages, rendering mechanism)
   multiplies by kernel count on every save. Keep the ladder short (P3's single
   combined legalization pass), and benchmark "N live kernels, one edit,
   time-to-first-frame" alongside the hit-path microbench.
6. **LOC honesty is a performance concern.** An under-budgeted kernel component
   (P1's 165-line lowerer) doesn't stay small — it grows *inside the kernel*,
   where every added line is a candidate for the dispatch path. Components
   calibrated by the verdicts at 800–1,500 lines must have satellite homes from
   day 1, so growth lands outside the code that owns the hot loop.
7. **CI gates are the only durable performance spec.** The winner's real edge is
   that its performance claims are armed as tests from the vertical slice:
   hit-path p50 gate, `flatten` allocation budget, `no_compile` mode,
   `compiles==1` over N frames, per-tier miss counters that name the differing
   key component. Adopt all five, plus graft 5's differential key test; a
   performance property without a gate is a press release.
