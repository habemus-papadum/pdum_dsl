# J3 — Pragmatist judgment of the three architecture proposals

*Judge role: PRAGMATIST. Question set: can the day-1 slice actually be built
quickly; where is the hidden complexity; dependency risk; is the ~1000-line
budget honest or does it hide load-bearing magic; failure modes under real
notebook/live-coding usage; fidelity to the types-not-values thesis under the
edge cases catalogued in `design/dsl_caching_layer.md`. Inputs: P1–P3 (full
docs), V1–V5 verdicts, `docs/desiderata.md`, the hazard doc. July 2026.*

**Preliminary observation the scores must be read against:** the three
proposals are ~85% the same design, because all three implement V1–V5
faithfully — same Node/Region micro-IR with three region ops, same `core.env`
value-free capture invariant, same five registration surfaces over one explicit
Registry, same ValueKind/PackPlan/FastRecord marshaling triple, same two-tier
cache with DerivedId transform identities, same five-step hit path, same
day-1 slice (orbiting disk on WGSL + Python with a differential oracle). The
pragmatist's job is therefore to score the 15% that differs, and that 15% is
almost entirely: (1) where the AST lowerer lives and who pays for its true
size, (2) what actually executes on the hit path, and (3) how mechanically the
discipline is enforced.

---

## 1. Scoring table

Criteria: (a) prime-directive extensibility, (b) caching-thesis fidelity,
(c) hot-loop cost, (d) multi-backend fit, (e) transformation readiness,
(f) smallness/honesty of the line budget, (g) marshaling story,
(h) implementability/risk. 1–10 each.

| Criterion | P1 jax-school | P2 mlir-school | P3 nanopass-school |
|---|---:|---:|---:|
| (a) extensibility | 7 | 7 | **9** |
| (b) caching fidelity | **9** | 7 | 8 |
| (c) hot-loop cost | 8 | 8 | 8 |
| (d) multi-backend fit | 8 | **9** | 8 |
| (e) transformation readiness | **9** | 8 | 8 |
| (f) budget honesty | 6 | **9** | 7 |
| (g) marshaling | 8 | 8 | 8 |
| (h) implementability/risk | 7 | 7 | **9** |
| **Total** | **62** | **63** | **65** |

**Winner through the pragmatist lens: P3 (nanopass-school)** — with two
mandatory grafts from P1 (per-call drift guards; fingerprint-soundness fuzz)
and one from P2 (honest satellite accounting for the lowerer) that repair its
two real weaknesses.

---

## 2. Per-proposal critique

### P1 — jax-school (62)

**The load-bearing hand-wave: a 165-line in-kernel lowerer.** P1 puts the
fused typing+lowering pass *inside* the kernel (`kernel/lower.py`, 165 LOC:
parse, coherence check, name classification, fused typing+lowering, loc
channel). V1 — the verdict P1 claims to adopt verbatim — calibrates this exact
component at **800–1,500 lines for the M1 subset** against cupy.jit's 1.8k
full frontend. 165 lines is defensible only for the day-1 expression-only
subset (P1's slice explicitly excludes assignments and loops). The moment the
subset widens to what M1 needs — assignments, `for`, arrays, call resolution
across overloads — the lowerer grows 5–8×, it lives in the kernel, and the
1,100-line CI cap detonates with **no provisioned overflow location**. P1's
risk register (R5, "a 1300-line lowerer") names the symptom but offers only
"the missing abstraction is named and moved into shared rules" — but the
lowerer's growth is not rule-shaped in P1's design; there is no `lower_ast`
aspect column, so widening syntax means editing kernel code. This is
simultaneously P1's budget-honesty failure (f=6) and its extensibility failure
(a=7): the prime directive's first axis, *new syntax*, has no registration
surface in P1.

**What P1 uniquely gets right — and it matters: the hit path is the only one
of the three that is correct under the hazard doc's verified edge cases.**
`FastRecord.guards` — precomputed `(cell, expected)` identity pairs checked as
pointer compares on every hit — is the only concrete per-call answer in any
proposal to two verified hazards:

1. *Frozen-global drift*: a template folds `math.pi`-class globals at lower
   time; the user rebinds one mid-session; P2 and P3 both serve the stale
   artifact on the hit path (both check dependency drift only at miss time or
   via an unspecified generation bump). P1 refuses/recompiles, never stale.
2. *Globals collision under freeze* (the hazard doc's verified `G=1` vs `G=2`
   case): two value-equal code objects closing over different global
   namespaces build the same thesis key in all three proposals — none puts a
   globals tag in the key — but only P1's guards catch the mismatch at call
   time. (Cost of P1's approach in the pathological both-live case: guard
   ping-pong recompiles — correct, hot-loop-hostile, and rare; acceptable.)

P1 also names the one silent-wrong-pixels failure mode the other two miss:
fingerprint **collision** (not miss). All three use fingerprints *as* the
probe key; the hazard doc's "fall back to full typeof on fingerprint miss"
does not save you when two different types produce equal fingerprints — that
is a wrong *hit*. P1's R4 property fuzz (`fingerprint(a) == fingerprint(b) ⟹
typeof(a) == typeof(b)`) is the right test and must survive into the
synthesis. Hence b=9.

**Transformation readiness is the best-de-risked (e=9):** the pre-M2 two-day
spike (vmap over `core.if`+`core.for` on the Python backend, batched-predicate
→ both-branch+select) with a **quantified** fallback trigger (>350 lines ⟹
re-hear tinygrad's direct-VJP per V5) is exactly how a pragmatist retires the
region-op × aspect tax before it compounds. The ~180-lines-per-region-op
pricing rule is the right constitutional mechanism.

**Verdict:** the best cache-correctness thinking in the field, attached to the
worst frontend accounting. Its correctness machinery must be grafted; its
kernel structure should not be adopted.

### P2 — mlir-school (63)

**The most honest budget in the field (f=9), and that honesty is a design
decision, not bookkeeping.** P2 is the only proposal that prices the lowerer
at V1's calibrated size (~900 lines) and gives it a home (`front/lower.py`,
satellite, kernel-may-not-import-it, CI-checked), stating outright that this
"is what keeps the kernel at ~1000 lines honestly rather than by accounting
tricks." Correct. Its risk register even names the lowerer's gravitational
pull toward the kernel as risk 3 with a 2–3 kloc reopen-xDSL trigger. A
pragmatist trusts numbers like these.

**But the hit path as written has two staleness bugs — sketch-level, yet the
sketch is the contract.** The fast key is
`(handle.fp_head, fingerprint_tuple(args), ACTIVE.target_fp)`:

1. **`generation` is absent from the fast key** (it appears only in the
   full-key miss path). `fp_head` is memoized on the Handle at phase A.
   Consequence in the primary notebook workflow: redefine a helper `@jit`
   function, re-run the render-loop cell — callers holding warm cache entries
   keep serving the old inlined artifact. This is precisely the
   live-coding-invalidation failure the project exists to avoid. P1 and P3
   both key the probe on generation per call.
2. **Dependency-closure drift is "checked here" — in the miss path.** On a
   hit, a drifted folded global is served stale. V1 §8 requires the check "at
   call time … never silently stale."

Both are one-line fixes, but a proposal whose §4 is titled "the entire
per-call path" and omits them earns b=7. The pragmatist rule: on the hot key,
sketches *are* the spec.

**What P2 uniquely gets right: always-on stage legality (d=9).** The
ConversionTarget-lite check — every lowering stage declares the op namespaces
legal at its output, illegal survivors error with op + `loc` — costs ~30 lines
and converts "progressive lowering" from convention into a machine-checked
invariant. For the Nth backend author it answers, mechanically, "exactly which
op set arrives at my renderer" — the cheapest possible contract for the
backend seam. The explicit `abi.slot` dialect stage also makes the marshaling
legalization auditable in printed IR. One contained risk: rendering structured
region code via RuleSet emitters is less proven than P3's plain
walk-the-regions render function (tinygrad's pattern-matcher rendering runs
over *linearized* code); the fallback is named and the risk is confined to
backend directories.

**Minor concept tax:** the `Dialect` bundle is genuinely just a tuple of
surface entries (checked: it adds no resolution semantics), and it will be
pleasant for shipping `units`/`ein` — keep it as sugar, watch that it never
grows install-order semantics. Day-1 slice is the largest of the three
(~2.5 kloc before first pixel), which with the honest lowerer costing is the
price of the accounting: h=7.

**Verdict:** the proposal you would hand to a second engineer — everything is
priced and checkable — but slowest to first pixel and, as written, the weakest
on the exact staleness edge cases the hazard doc verified.

### P3 — nanopass-school (65) — WINNER

**Why it wins the pragmatist lens: it is the only proposal whose discipline is
mechanical rather than aspirational, and its day-1 slice is the smallest real
system.** Concretely:

- **Fastest to first pixel:** ~1,075 budgeted lines *including* the first
  backend, plus ~350 WGSL + ~120 stdlib ≈ 1.5 kloc total for the slice, vs
  P1's ~1.9k and P2's ~2.5k — and P3's slice has a *richer* subset than P1's
  (local assignment, attribute swizzle, backend-registered `FragCoord`
  intrinsic, proving surface-D dialect ops on day 1).
- **One mechanism where the others have two.** Deviation 1 (first backend is a
  110-line Python *source renderer*, not an eval-rules interpreter) is the
  single best pragmatist call in any proposal: renderer+`exec` gives reference
  semantics and the differential oracle through the *same* mechanism every
  other backend uses, halving what must work on day 1. The deviation is
  flagged, the eval column stays reserved, and the cost-of-being-wrong is
  priced (~150 lines in `tools/`). This is what honest deviation looks like.
- **The prime directive is CI-enforced, not narrated (a=9).** The
  extension-locality test — adding `sinh`, a record method, **and a new
  statement form** must produce zero kernel diffs, run in M1 not M5 — is the
  only mechanism in any proposal that makes *syntax widening* a registration
  (`(ast.NodeType, "lower_ast")` rule entries) rather than an edit to a
  lowerer monolith. P1 widens syntax by editing kernel code; P2 by editing a
  900-line satellite. P3 alone satisfies the desiderata's first extensibility
  axis structurally. Per-file caps, the no_compile mode, the microbench gate,
  and milestones defined as *detectors going green* complete the picture.
- **Risk sequencing is the best (h=9):** M2's ray-march spike is the earliest
  probe in any proposal at the design's known weakest joint (three region ops
  vs `break`/early-exit), with the decision deferred until evidence exists.

**P3's load-bearing hand-wave #1: the uncounted `lower_ast` rule packs.** The
headline "1,075 CI-capped lines" rests on `lower.py` being a 135-line *driver*
that dispatches per-AST-node logic to `lower_ast` rules — and the rules' line
cost appears in **no table anywhere**. V1's 800–1,500-line estimate for the
fused typing+lowering pass does not evaporate because the dispatch is
externalized: the per-node typing logic, overload resolution order, assignment
functionalization, and env/scope threading must live in those rule packs
(likely 400–900 lines at M1). Two saving graces keep this at f=7 rather than
P1's 6: the driver-vs-rules split is at least *architecturally* coherent (the
driver owns env/scoping, rules own per-node emission — numba's dispatch-table
shape, and it is what makes the extension-locality test possible at all), and
the per-file caps guarantee the bust is loud and early rather than silent. The
synthesis must simply do what P2 did: put the rule packs on the books as a
satellite budget line.

**P3's load-bearing hand-wave #2: drift detection by unspecified mechanism.**
§4.4 asserts "generation bumped by redefinition **and dependency-closure
drift** (refuse-or-recompile, never silently stale)" — but nothing in P3
checks for drift on the hit path, and nothing else can bump the generation
when a folded global is rebound between calls. As written this is the same
staleness hole as P2's. It is exactly repaired by grafting P1's
`FastRecord.guards` (which P3's FastRecord and key-build step absorb without
interface change — the guards are pointer compares folded into key build,
which P1 already showed preserves the "key build + pack + launch, nothing
else" contract). With that graft, b would rise from 8 to 9.

**Also verify before trusting:** the WGSL backend at ~350 total lines is the
most optimistic of the three estimates for the same component (P2: 550, P1:
650). M0's `layout.py` and runtime exist as a quarry, so it is not fantasy,
but plan for ~500. And the heterogeneous rules-dict keying (op-name strings
and `ast` node types in one `(key, aspect)` matrix) wants a type-level cleanup
in the synthesis, not two parallel matrices.

---

## 3. Cross-cutting adversarial findings (apply to all three)

1. **Fingerprints-as-keys is a shared correctness cliff.** All three probe the
   thesis cache on `(template_fp, env_fp, arg_fp, …)` — the fingerprint *is*
   the key, so a fingerprint collision between different types is a silent
   wrong-artifact hit, not a recoverable miss. Only P1 tests for it. The
   synthesis needs P1's soundness fuzz as a permanent CI property test, and a
   stated rule for what a fingerprint may omit (nothing that `typeof`
   distinguishes).
2. **Phase-A cost is under-measured relative to its billing.** The driving
   pattern runs `make_handle` every frame; all three quote ~1–2 µs on scalar
   captures, but the first array capture reads `arr.flags` per fingerprint per
   frame. All three defer this to "the week arrays land" (P1 R1 says it
   explicitly). Acceptable, but the array-capture microbench must be a named
   M2 gate, not a hope.
3. **Nobody budgets eviction.** The hazard doc's L-cache-leak row (dead
   templates accumulating across notebook edit-rerun cycles, each holding a
   strong code-object ref) gets one clause in P2 ("LRU + per-key future") and
   silence in P1/P3. In the target workflow — hours of live editing — this is
   a real leak. Cheap fix; must be in the synthesis's cache.py.
4. **The ~1000-line kernel is honest in none of the three if "kernel" means
   "everything you must read to trust the cache."** The trust set is kernel +
   lowerer + fingerprint kinds ≈ 2.0–2.5 kloc in all three. That is still
   excellent (M0-comparable, numba-frontend-fraction) — but the synthesis
   should say it plainly rather than let the 1,000 headline imply otherwise.

---

## 4. Grafts from the non-winners into the synthesis

From **P1 (jax-school)**:

1. **`FastRecord.guards` on the hit path** — precomputed `(cell, expected)`
   identity pairs checked as pointer compares during key build; drift ⇒ treat
   as miss. The only concrete per-call answer to frozen-global drift and the
   verified globals-collision hazard. Non-negotiable graft.
2. **The fingerprint-soundness property fuzz** (`fp(a)==fp(b) ⟹
   typeof(a)==typeof(b)`) plus continuous randomized Python-vs-WGSL
   differential runs, as permanent CI.
3. **The priced constitutional-amendment rule for region ops** (~180 lines ×
   live transform columns per new region op) and the **pre-M2 vmap spike with
   its quantified fallback trigger** (>350 lines ⟹ tinygrad direct-VJP
   re-hear).
4. **`generation` read from the registry in every fast-key build** (P1's
   dispatch sketch is the correct one of the three).

From **P2 (mlir-school)**:

5. **Always-on stage legality** (`Stage(name, rules, legal)`, ~30-line
   namespace check, errors naming op + `loc`) — the cheapest machine-checked
   backend-seam contract; keep full grammar RuleSets debug-only per V2.
6. **Honest lowerer accounting**: budget the fused typing+lowering pass as a
   satellite at V1's 800–1,500 lines — while keeping P3's `lower_ast`
   rule-dispatch shape so the extension-locality test still holds. Count the
   rule packs.
7. **The named-miss differential key test**: perturb every declared key
   dimension one at a time (capture type int→float, target texture format,
   body edit, unit change) and assert the *named* tier and component of the
   miss ("env_types[0]", "codegen_opts", "template", plan-memo-only).
8. **The attr lint**: only `LiteralType`-originated constants may appear as
   `core.const` attrs, checked on printed IR — closes the "smuggle a value
   into attrs to make it work" hole in the one place values can enter.
9. **`Dialect` as pure bundling sugar** (`registry.install(dialect)`) for
   shipping units/einops as one artifact — with a test asserting it adds no
   resolution semantics.
10. **The two-threshold microbench** (alarm at 5 µs, fail at 10 µs) — early
    warning before the gate breaks.

---

## Design lessons for pdum.dsl

1. **Adopt P3's skeleton, P2's ledger, P1's paranoia.** The synthesis is:
   nanopass kernel + renderer-first backends + extension-locality CI (P3);
   the lowerer and its rule packs priced as an 800–1,500-line satellite with
   stage-legality checks (P2); guards, generation-in-fast-key, and the
   fingerprint fuzz on the hit path (P1). Each proposal's signature weakness
   is precisely covered by another's signature strength — this is a
   composition, not a compromise.
2. **Treat the hit-path sketch as normative spec and review it like one.**
   Two of three proposals shipped staleness bugs *in the five-line function
   the whole project exists to make correct* (missing generation; miss-only
   drift checks). Rule for the synthesis: the `__call__` sketch in the design
   doc is the contract; every hazard-doc row must point at the specific line
   of it (or of miss-time setup) that discharges it.
3. **The frontend is where all three budgets lie; make it a satellite of
   rules and count the rules.** V1's 800–1,500-line estimate is
   family-calibrated and no proposal beat it — they hid it (165 in-kernel;
   uncounted rule packs) or paid it (900 satellite). The synthesis keeps the
   135-line driver + `lower_ast` rule column (it is what makes new syntax a
   registration) and books the rule packs as `front/` satellite lines with
   their own cap.
4. **Fingerprint collisions, not fingerprint misses, are the real hazard of
   the fast key.** All three probe on fingerprints; a collision is a silent
   wrong hit. The soundness property test is as load-bearing as the cache
   itself — ship it in the vertical slice.
5. **Per-call guards are the missing hazard-doc mechanism — and they fit the
   hot-path contract.** Frozen-global drift and value-equal-code/different-
   globals cannot be keyed away cheaply; P1's pointer-compare guards folded
   into key build handle both at nanosecond cost. Any future "fold this
   global as a constant" feature must register a guard at the same moment —
   couple them in one API so they cannot drift apart.
6. **Prefer one mechanism at two-thirds fidelity over two mechanisms at
   full.** P3's render-to-Python-source-as-oracle beat both interpreter
   designs on total system risk. When the synthesis faces
   interpreter-vs-renderer, per-op-eval-vs-exec, plan-vs-generated-binder:
   the smaller mechanism count wins day 1, with the richer variant reserved
   behind an already-shaped contract.
7. **Discipline that isn't in CI is fiction.** The decisive gap between
   these proposals was never architecture (they converged); it was whether
   the prime directive, the line budget, and the hot-loop guarantee are
   *tests* (P3: extension-locality, per-file caps, no_compile, microbench
   gates as milestone exits) or *prose*. Every invariant in the synthesis
   design doc must name its CI test in the same paragraph, day 1.
8. **Budget eviction and long-session hygiene now.** Hours-long notebook
   sessions accumulate dead templates holding strong code-object refs (the
   hazard doc's L-cache leak); one LRU bound plus retirement of superseded
   templates in `cache.py` is ~20 lines nobody budgeted. Add it to the
   vertical slice's cache module before the first user hits it.
