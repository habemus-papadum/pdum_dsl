# J1 — Judge report: the maintainer lens

*Judgment over P1 (jax-school), P2 (mlir-school), P3 (nanopass-school), evaluated
against V1–V5, `docs/desiderata.md`, and `design/dsl_caching_layer.md`. Lens:
incremental extensibility (prime directive, double weight), smallness/readability,
cost of each future capability, coupling risk, contributor experience. July 2026.*

---

## 0. The judging problem: three proposals, one architecture

All three proposals implement the same verdict-driven machine: reflective phase A,
frozen `Node(op, type, args, attrs, regions)` with three region ops, value-free
`core.env`, one rewrite driver, one explicit Registry with five surfaces, ValueKind
→ LeafPath → PackPlan → FastRecord marshaling, two-tier cache, `DerivedId`
transform identities, dual-backend day-1 slice. Convergence this strong means the
verdicts did their job; it also means the judgment turns on the **residual
differences**, which are real and consequential:

1. **Where lowering logic lives** — kernel monolith (P1), satellite monolith (P2),
   or rule column (P3). This is the decisive axis, because "new syntax" is the
   *first-listed* extension axis of desiderata §6 and the prime directive is
   weighted double.
2. **Line-budget honesty** — who is counting the lowerer, and at what size.
3. **Hit-path completeness** — who actually implements V1's call-time
   dependency-drift check.
4. **Auditability machinery** — stage legality, extension-locality CI, attr lints.

---

## 1. Scoring table

Scale 1–10. (a) is the prime directive and is **counted twice** in the total
(max = 90).

| Criterion | P1 jax-school | P2 mlir-school | P3 nanopass-school |
|---|---:|---:|---:|
| (a) prime-directive extensibility (×2) | 7 | 7 | **9** |
| (b) caching-thesis fidelity | **9** | 8 | 8 |
| (c) hot-loop cost | **9** | 8 | **9** |
| (d) multi-backend fit | 8 | **9** | **9** |
| (e) transformation readiness | **9** | 8 | 8 |
| (f) smallness / honesty of line budget | 5 | **9** | 7 |
| (g) marshaling story | **9** | **9** | 8 |
| (h) implementability / risk | 7 | **8** | **8** |
| **Total (a doubled)** | **70** | **73** | **75** |

**Winner through the maintainer lens: P3 (nanopass-school)**, narrowly over P2,
with P1 third. The margin is small because the proposals share ~85% of their
architecture; the ordering is nonetheless stable under the lens, because the two
things the maintainer lens weighs most — does the #1 extension axis (syntax) go
through a registration surface, and is the budget enforced rather than asserted —
are exactly where P3 differs from the other two.

---

## 2. Per-proposal critique

### P1 — jax-school (70)

**Strengths.** The most rigorous caching-thesis treatment of the three: it is the
only proposal that actually puts V1's call-time dependency-drift check on the hit
path (`FastRecord.guards` as precomputed `(cell, expected)` pointer compares,
honestly folded into the key-build accounting) rather than claiming it in prose
and omitting it from the dispatch code. The fingerprint-soundness property fuzz
(`fingerprint(a) == fingerprint(b) ⟹ typeof(a) == typeof(b)`) is the sharpest
single test proposed anywhere. Its transformation section is the most concrete
(explicit `Transform("grad", passes=(jvp, partial_eval, transpose))` pipeline;
R2's pre-M2 vmap-over-if/for spike with a 350-line threshold and a named fallback
is a model risk-retirement plan). Its `extract(env, args, buf)` contract fills
staging directly — one fewer allocation than P2's leaves-tuple hop. Design lesson
9 ("if a proposed kernel component is neither a primitive nor one of the two
engines, it is a registration in disguise") is the best sentence in any proposal.

**The load-bearing hand-wave: `kernel/lower.py` at 165 lines.** V1 — a verdict P1
claims to adopt "essentially as written" — estimates the fused typing+lowering
layer at **800–1,500 lines for the M1 subset** (cupy.jit's complete kernel
language is 1.8k lines). P1 budgets 165 lines for parse + coherence check + the
closed fate taxonomy + fused typing+lowering + overload/method resolution + the
loc side channel, inside a 1,010-line kernel headline. There is no mechanism in
P1 that moves lowering *content* out of that file (unlike P3's rule column): the
handlers are in the pass. Either the kernel is really ~1,700–2,300 lines, or the
day-1-slice subset is being quoted as the steady-state budget. Since P1's whole
pitch is "a ~1010-line kernel," and since the lowerer is *in* the kernel, this
one number contaminates both the smallness claim and the extensibility claim.

**The extensibility consequence, which P1's own pitch quietly concedes.** P1's
zero-kernel-edit list is "the stdlib, all backends, records, units, autodiff/vmap,
and t-string mini-languages" — *syntax widening is absent*. In P1, widening the
accepted Python subset (a new statement form, `while`, augmented assignment) is
an edit to `kernel/lower.py` — a kernel edit, on the most frequently exercised
extension axis, in the file with the least honest budget. For a proposal whose
thesis is "types-not-values as a property of the data structures," it is striking
that "new syntax without kernel edits" is a property of nothing. Under a
double-weighted prime directive this caps (a) at 7 and (f) at 5.

**Minor.** The `registry.bindings: dict[int, str]` (id(global) → op name) for
name classification is an `id()`-keyed table in a design whose founding
anti-pattern is `id()`-keying; it is compile-time-only and paired with guards, so
it is defensible, but it deserves the same GC-reuse scrutiny V4 applies to Taichi.

### P2 — mlir-school (73)

**Strengths.** The most honest proposal. It is the only one that budgets the
lowerer at V1's measured size (~900 lines) and says out loud that keeping it
outside the kernel "is what keeps the kernel at ~1000 lines honestly rather than
by accounting tricks." Its two sharpenings are both genuinely good and cheap:
**always-on stage legality** (`Stage(name, rules, legal)`, ~30 lines of namespace
checks) turns progressive lowering from convention into a machine-checked
invariant with loc-bearing errors — this is the single best auditability idea in
the three proposals. The **`abi.slot` dialect stage** makes marshaling
legalization *visible in printed IR* — "which physical slots does this capture
become" is answerable by printing, the same way `attrs`-vs-`args` makes key
membership printable. Its risk-5 test (perturb every ambient key dimension one at
a time, assert the *named* tier of miss) and the attr lint (only
`LiteralType`-originated constants may appear as `core.const` attrs) are the best
key-hygiene tests proposed. The dialect ladder is the clearest one-picture
account of the whole compile pipeline.

**The load-bearing hand-wave: "the AST lowerer is a satellite."** The word
"satellite" is doing accounting work, not architecture work. Every other
satellite in P2's table attaches through one of the five surfaces; `front/lower.py`
attaches through none — it is a privileged 900-line component that implements
fate classification, overload resolution, the Literal lift, and typing, and
*every* syntax extension flows through editing it. P2 has no `lower_ast` aspect;
its aspect list is `type, eval, jvp, transpose, batch, unit`. So in P2, the
prime directive's first axis is served by a monolith — better isolated than P1's
(not in the kernel, kernel never imports it), but still numba's `typeinfer.py`
failure mode waiting at the end of the growth curve, and P2's own risk 3 admits
the file has "the strongest gravitational pull toward the kernel." A maintainer
evaluating "how cheaply does the next contributor add a statement form" gets the
answer: read and modify a 900-line fused pass. That caps (a) at 7.

**Minor.** The shown fast key `(fp_head, fingerprint_tuple(args),
ACTIVE.target_fp)` omits `generation` (present only in the full miss-path key) —
a stale-hit window after a world-age bump that the full-key text contradicts;
and the desiderata-mapping row claims the dependency-closure hash is "checked at
call time" while the five-step hit path contains no such check. Both are
resolvable inconsistencies, but in the layer where inconsistency is the product
risk. `Handle.env` as a dict (vs tuple in P1/P3) is a small phase-A tax. The
`Dialect` bundle is fine as pure aggregation but needs a one-line law ("a Dialect
may contain nothing that could not be registered surface-by-surface") to keep it
from drifting into a sixth surface.

### P3 — nanopass-school (75) — winner

**Strengths.** P3 is the only proposal in which the prime directive's first axis
is a registration: lowering handlers are `(ast.NodeType, "lower_ast")` entries in
the same op×aspect matrix as everything else, and the kernel's `lower.py` is a
135-line *driver*. Widening the Python subset = adding a rule; the kernel diff is
zero by construction, not by discipline. P3 then does the maintainer-decisive
thing: it makes the claim falsifiable, with an **extension-locality CI test**
("adding `sinh`, a record method, and a new statement form must produce zero
kernel diffs — run in M1, not M5"), per-file line caps, a hard kernel cap
(≤1,150) in CI, the `no_compile` assertion mode, and the hit-path microbenchmark
gate, all installed by the day-1 slice. "Budgets are architecture" is the prime
directive made mechanical; no other proposal enforces its own headline number.
Its milestone plan is sequenced by risk retirement (each milestone's exit
criterion is a §7 detector going green), and its first backend choice — a
110-line Python *renderer* rather than an eval interpreter — means every backend
goes through one mechanism (render source, compile, launch), so the day-1 slice
exercises the same seam shape that CUDA/Metal/C will use. Its deviations from the
verdicts are all flagged, argued, and priced ("cost of being wrong: ~150 lines in
tools/"), with tripwires (`registry.py ≤ 150 before we revisit`).

**The load-bearing hand-waves (it has three, all smaller than the others').**

1. *The `lower_ast` rule content is unaccounted and the mechanism is unproven for
   context-heavy statements.* The 135-line driver claim assumes lowering handlers
   decompose cleanly per AST node. Expressions and calls do; assignments, scoping,
   type-stable loop variables, and tuple unpacking need a shared typing
   environment (cupy's `Environment`) threaded through every rule — the driver
   must own that context object, and the rules' signature becomes
   `(ctx, ast_node) -> Node`. That is workable (nanopass and DaCe both prove
   visitor-granular extension), but the lines V1 measured (800–1,500 total) do
   not vanish: they move into rule packs whose home (`stdlib/`? `ops.py`?) and
   size P3 never states. P3's budget honesty is better than P1's — the kernel cap
   genuinely excludes the content — but the *system* count is undercounted by
   roughly the same amount P2 states openly.
2. *The dependency-drift check has no mechanism.* §4.4 says generation is "bumped
   by redefinition and dependency-closure drift," but nothing in the hit path
   checks drift and nothing is named as the bumper. V1 §8 requires a call-time
   check. P1 solved this (guards); P3 must graft it.
3. *The reserved-but-empty `eval` column* means AD rules, when they ship, will be
   finite-difference-tested against rendered-and-exec'd programs rather than
   per-op eval rules. Acceptable (single-op programs through the Python renderer
   are an adequate oracle) but slightly weaker than V5's day-1 `eval_rules`
   intent, and P3 should say so in its M3 plan.

**Why P3 wins the lens anyway.** Doubling the prime directive is the assignment,
and P3 is the only proposal whose architecture *and enforcement* both point at
it: all five verdict surfaces, plus syntax-as-rules, plus the CI machinery that
makes "new capability ⇒ zero kernel diff" a failing test instead of a design-doc
sentence. Its hand-waves are patchable by grafts (below) without moving any seam;
P1's and P2's hand-waves are structural (where does syntax growth go), and fixing
them means *becoming* P3 on that axis.

---

## 3. Adversarial cross-checks (things all three got right, verified)

- **`core.env` is value-free in all three** — the anti-numba invariant is
  structural everywhere; no proposal reintroduced `FreeVar(value)` under any
  alias. P3's grep-level assertion (no kernel type reachable from `Node` has an
  `object`-typed field except `attrs`) is the cheapest enforcement and should be
  adopted regardless of winner.
- **No proposal smuggles values into the fast key.** All fast keys are
  fingerprint-based with full-`typeof` fallback, per the hazard doc.
- **All three keep the five-surfaces law** (P2's `Dialect` and P1's
  `bind_global` are the closest approaches to a sixth; both survive scrutiny as
  aggregation/classification, not new semantics).
- **All three carry the same pre-shaped hot-path escalations** (exec'd binder →
  native fastpath) behind an unchanged FastRecord contract — the right way to
  hold a performance risk.
- **All three day-1 slices are dual-backend with a differential oracle and
  `compiles==1` gates.** No proposal defers the backend-seam proof.

---

## 4. Grafts into the final synthesis (P3 base)

From **P1 (jax-school)**:

1. **`FastRecord.guards`** — precomputed identity-compare tuples for
   dependency-closure drift, checked on the hit path, counted as key build. This
   directly fixes P3's hand-wave #2 and is the only faithful implementation of
   V1 §8 among the three.
2. **The fingerprint-soundness property fuzz**: `fingerprint(a) ==
   fingerprint(b) ⟹ typeof(a) == typeof(b)`, run continuously in CI.
3. **The pre-M2 transform spike**: vmap over `core.if` + `core.for` on the
   Python backend only, with the explicit threshold (>~350 lines ⇒ re-hear
   tinygrad's direct-VJP per V5's fallback) — retire the region-op × aspect tax
   before arrays land.
4. **`extract(env, args, buf)` fills staging directly** — drop the intermediate
   leaves tuple from the packer contract (P3 currently returns leaves then
   packs).
5. **The fourth-region-op pricing rule** stated as a budget line item: any
   proposed region op is priced at ~180 lines × live transform columns before
   acceptance.
6. **Design-lesson-9 discipline** as a PR review question: "is this kernel
   component a primitive or one of the two engines? If neither, it is a
   registration in disguise."

From **P2 (mlir-school)**:

7. **Always-on stage legality** (`Stage(name, rules, legal)` with ~30-line
   namespace checks, loc-bearing errors) — bolt onto P3's single rewrite driver;
   it costs nothing at pdum scale and makes the lowering ladder auditable.
8. **The `abi.slot` stage**: represent post-`legalize_params` physical slots as
   IR ops so the marshaling plan is visible in printed IR and golden-testable,
   rather than existing only as the PackPlan data structure.
9. **The perturbation key test**: for every declared ambient key dimension
   (backend param, generation, unit tier, capture type), mutate it one at a time
   and assert the *named* tier of miss. Plus the **attr lint** (only
   `LiteralType`-originated constants may appear as `core.const` attrs, checked
   on printed IR).
10. **Honest lowering accounting**: adopt P2's ~900-line figure as the planning
    number for driver + `lower_ast` rule packs combined, wherever the lines
    live; report it as its own CI-counted bucket next to the kernel cap so P3's
    rule column cannot become an accounting hole.
11. **`Dialect` as an aggregation value** for shipping units/einops bundles —
    with the explicit law that it may contain nothing unregistrable
    surface-by-surface.
12. **Generation in the fast key** — make P2's own inconsistency a test: the
    fast key must contain every component of the full key or a fingerprint
    thereof; assert with the perturbation test from graft 9.

---

## 5. Verdict

**Winner: P3 (nanopass-school), 75/90**, over P2 (73) and P1 (70). Adopt P3 as
the base architecture; apply grafts 1–12. The synthesis to hand the design team
is: *P3's kernel and enforcement culture, P1's hit-path completeness and
transform de-risking, P2's auditability machinery and honest accounting.*

The one open question the synthesis must resolve in M1, because it is P3's only
structural bet the others didn't make: **prove the `lower_ast` rule column on the
hardest day-1 statements** (assignment, type-stable locals, tuple unpacking) with
the typing-context object designed first. If per-node rules cannot cleanly share
the typing environment, the fallback is P2's satellite monolith — a contained
retreat (the kernel driver shrinks to a dispatcher either way), which is exactly
why P3 remains the safer base: its failure mode degrades into P2, while P2's
failure mode (lowerer bloat) has no cheap retreat.

---

## Design lessons for pdum.dsl

1. **Syntax is the extension axis most designs forget to make extensible.** All
   three proposals made backends, types, batteries, and transforms
   registrations; only one made *the accepted Python subset* a registration.
   Since desiderata §6 lists new syntax first, the synthesis must keep lowering
   handlers in the rule matrix (`lower_ast` column) — and must design the shared
   typing-context object those rules receive before writing the first rule.
2. **An unenforced budget is a pitch, not an architecture.** P1's 165-line
   lowerer inside a "1,010-line kernel" shows how the headline number fails
   silently. Adopt P3's mechanism: CI kernel cap, per-file caps, and — from P2 —
   a separately counted lowering bucket so rule columns can't hide lines. When a
   cap breaks, the design conversation happens then, with the overflow as
   evidence.
3. **The extension-locality test is worth more than any module map.** "Adding
   `sinh`, a record method, and a new statement form produces zero kernel diffs"
   as a failing-able M1 test is the prime directive made mechanical. Run it
   before the architecture has time to rot, not after.
4. **The hit path must carry the drift check, not the prose.** Two of three
   proposals claimed V1's call-time dependency-drift check and omitted it from
   their dispatch code. `FastRecord.guards` (precomputed pointer compares,
   counted as key build) is the pattern; any key component that appears only in
   the miss-path text is a stale-hit bug in waiting — assert fast-key/full-key
   consistency with a perturbation test.
5. **Make every invariant printable, then lint the printout.** The strongest
   auditability ideas in the pool all reduce to "the claim appears in printed
   IR": stage legality (which dialects may exist here), `abi.slot` (which
   physical slots a capture becomes), the attr lint (which values entered the
   key). A structural invariant you can print is one you can golden-test and
   grep.
6. **Prefer one mechanism over two, even for the reference backend.** P3's
   Python-source *renderer* means every backend — including the oracle —
   exercises the same render/compile/launch seam. Interpreter-style eval columns
   remain available as a later `tools/` addition; they should not be a second
   day-1 execution mechanism.
7. **Judge proposals by their failure modes, not only their happy paths.** P3
   won partly because its riskiest bet (lowering-as-rules) degrades gracefully
   into P2's shape, while P1/P2's monolithic lowerers have no cheap retreat from
   bloat. In a convergent design space, pick the base whose fallback is a
   contained retreat rather than a rewrite.
