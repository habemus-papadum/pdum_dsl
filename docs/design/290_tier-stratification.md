# 290 — Tier stratification: which syntax lives where

**Status: RATIFIED AND IMPLEMENTED (owner, 2026-07-28; landed the same
day).** The gate is live at every capture seam; the tier battery
(`test_stratification.py`), the ledger (`pdum/tl/ledger.toml` +
`conformance/test_ledger.py`), and the bundled repin are in the tree.
Implementation notes that amend the design as proposed: (a) the
adjudication mechanism is `check_tier` AT THE SEAMS rather than
per-tier registries — the node's own `loc` gives the offending source
line (better than rule-level errors), one door serves captured and
hand-authored regions alike, and the unknown-op door stays at the
registry untouched; (b) `tl.const` moved from COMPUTE to STRUCTURAL —
constants are every tier's vocabulary (the census showed kernels and
vertex stages emitting it); (c) §6.5's obligation is discharged — the
unit tier is empirically clean of `abi.slot` under live gates; the
stray census hit was transform/extent attribution noise as suspected.
Commissioned by the owner 2026-07-28; claimed by the dsl side (PR #7
thread). The graphics campaign's handoff (211, merged with PR #7)
delivered the owner rulings of `282_owner-questions.md` (campaign
branch): the kernel rulebook, the control-flow amendment, the
coverage ledger, the bounded-loop handoff. The §6 residue was then
ratified in one pass per the ruling sheet (all five). Two same-day
companion rulings ride with the ratification: refusal wording is
relaxed to tripwire standing (recorded in AGENTS.md + 210 — the §6.4
bundle is now tidiness, not ceremony), and `tl.sample` + the texture
objects are to move OUT of tl (§7 — destination owned by the
graphics/runtimes team). (Numbering note: 280–282 are the
campaign's; this doc took 290.)

## 1. The finding: tiers constrain SPELLING, not VOCABULARY

The hierarchy exists — as Python-syntax rule packs, each extending the
last (`LOWER_RULES` → `TL_RULES` → `KERNEL_RULES`/`UNIT_RULES` →
`VERTEX_RULES`/`FRAGMENT_RULES`), plus the per-kind yield protocol
(`lower_body`: a step yields tensors, a kernel yields its token, a
fragment yields one value). **A kind declares its yield. It does not
declare its vocabulary.** Every capture seam hands the Lowerer the same
full op registry `{**CORE_OPS, **TL_OPS, **ABI_OPS}` (kernel.py:1126,
dialect.py:1004, graphics.py:418/500, assemblage.py:207), and the call
rules fall through toward the widest tier (`_f_call → _k_call →
_tl_call`), so the ops that land in a region are an emergent property
of reachability, not a declared set. `tl.kind` — the tier's name — is
consulted exactly once in the dialect (dialect.py:748), and the
graphics stages lower under `tl.kind="compute"`: the label cannot even
tell a fragment from a kernel today.

Probed on the current tree (three `@compute` bodies; independently
replicated by the campaign's `probe_kernel_bleed.py`, 211 §2d):

- `img.flip("x")` inside a `@compute` body **lowers** (`tl.flip` lands
  in the kernel region), **runs** on the numpy interpreter, and dies
  only at the device as `Untranslatable: tl.flip` — a backend-gap
  message, at backend-compile time, for what is a language violation.
- `reduce(...)` inside a kernel body lowers too (`tl.reduce` +
  `tl.repeat` in the region). The alignment law blocks the *misaligned
  use* incidentally — and its refusal **quotes the next smuggle as the
  fix** (`repeat('x', (0, 4))`), pulling the author deeper into
  tensor-tier vocabulary inside the kernel. (The campaign's probe adds:
  `shift`/`pad`/`select` fail on similarly incidental grounds —
  containment/alignment, not tier — while `red.sum` refuses properly,
  in the voice the rulebook now adopts for the whole class.)
- `ghost = img + img` (whole-tensor math in a kernel) lowers, runs,
  and even translates to WGSL — *coincidentally*, because pointwise
  commutes with per-thread indexing. Any non-elementwise op in the
  same position diverges between the reference and the device.
- A `for` in a `@jit` body **silently drops to the per-element oracle
  path** (`_liftable`, kernel.py:597) — reference-green, no device
  will ever run it, and nothing says so (211 §1.4).

The consequence is structural: the reference interpreter happily runs
kernel regions no device tier will ever admit, so **reference-green is
not kernel-tier-legal** — exactly the class of silent wrongness the
differential methodology exists to kill.

## 2. Legality, translatability, and the ledger (RULING, 211 §1.3)

**Tier legality is a language fact** — one table, owner-ratified,
stable. **Backend translatability is a capability fact** — per-backend
rows that grow over time. The owner's coverage-ledger ruling gives the
distinction teeth: never-legal constructs REFUSE at the tier;
legal-but-not-yet-translatable constructs stay allowed on the
reference but live in a **visible, versioned, machine-checked ledger**
whose drift from reality fails a test. Adopting the campaign's
suggestion: the ledger is this stratification's own artifact — one
machine-readable table `tier × construct → {admitted, refused,
ledgered}`, produced by the admission mechanism, consumed by the
conformance batteries to decide skip-vs-fail. 211 §2a/2b is its seed
content (the device-admissible row set; the `Untranslatable`
inventory, identical across WGSL/MSL — including the `tl.split`/
`tl.merge` cliff that makes every multi-block launch untranslatable,
and `derive_vjp` output untranslatable *by construction* since
`_VJP_SUPPORTED` and the device rows are nearly disjoint). After
stratification, `Untranslatable` speaks only for ledger rows — and a
device suite finally asserts coverage: kernel vocabulary ⊆ translated
∪ ledgered.

## 3. The scattered classifications (the demand for one table)

Hand-rolled op subsets exist all over the tree, each an ad hoc partial
answer to "which ops belong to what":

- `LAYOUT_OP_NAMES` (dialect) — the layout family;
- `_VIEW = LAYOUT_OP_NAMES | {"with_value_units"}` (transforms);
- `_FREE_OPS` (opcount) — the zero-cost rows;
- `_LIFT` + `_STRUCTURAL_OPS` (kernel.py:566) — the spliceable-scalar
  subset for fn-arguments, itself a mini tier check already;
- `_HOST_OPS` (dialect) — host-foldable scalar math;
- `_VJP_SUPPORTED` / `_FOLD_STEP_SUPPORTED` (dialect) — the adjoint
  walker's own admission sets;
- the WGSL row tables (capability — becomes the ledger's device
  column; currently doubles as the only de-facto adjudicator, in the
  wrong voice).

This is the same disease the excavation cured for typing/eval rows
(definition-site consolidation) and 210/211 diagnose for emitter
tables ("they exist three times").

## 4. The design

### 4.1 Families once, tiers as unions

Op **families** are declared once, in `dialect.py`, next to the op
definitions (the house locality rule). The enumerable universe today:
23 `core.*`, 38 `tl.*` (18 of them layout adapters), `abi.slot`.
Composites like `contract` never mint ops (repeat_like + pointwise +
reduce), so the table stays small. But the universe is **open** — the
census (§4.4) observed `pw.*` and `math.*` marker-derived ops and a
registered `toy.blit` — so families cannot be a closed enumeration:
**an op declares its families where it is registered** (the open
registry's row gains a column), and the built-in declarations live in
`dialect.py`. An unregistered-family op admits nowhere by default.

- `STRUCTURAL` — core.param, core.env, core.const, core.yield,
  core.tuple, core.extract
- `SCALAR` — core.add/sub/mul/div/neg/mod/pow, core.cmp, core.select,
  core.cast, and the open `pw.*`/`math.*` families
- `DSL_HOST` — core.call/for/if/load/store/vec/field: the dsl's own
  program ops, legal in no tl region as *author vocabulary*
  (statement-`if` lowers to select-joins, never to `core.if` nodes —
  §4.2; quoted oracle programs carry them as data — §4.5)
- `ABI` — abi.slot
- `LATTICE` — tl.iota, tl.coord
- `POINTWISE` — tl.pointwise
- `COMPUTE` — tl.reduce, tl.scan, tl.fold, tl.take, tl.scatter_add,
  tl.argtopk, tl.argsort, tl.random, tl.materialize, tl.round_to,
  tl.repeat_like, tl.const, tl.with_value_units
- `LAYOUT` — the 18 adapters (slice, select, shift, rename, repeat,
  flip, split, merge, diagonal, window, decimate, pad, stencil,
  strip_charts, with_charts, with_labels, bind, simplify)
- `EFFECT` — tl.store, tl.read, tl.token
- `GRAPHICS` — tl.sample

**Author vocabulary vs machinery emissions.** The table adjudicates
what an AUTHOR's body may spell. The kernel machinery itself emits
`tl.split` (kernel.py:396, the grid bracket) and `tl.merge`
(kernel.py:899, global-index stores) — census-confirmed, and 211 §2b
names them the single biggest coverage cliff. These are not author
vocabulary and never refuse; they are part of the compute tier's
*definition*, recorded in the ledger as machinery rows so the device
coverage assertion still sees them.

Everything that today hand-rolls a subset derives from or asserts
against the families: `LAYOUT_OP_NAMES := LAYOUT`, `_VIEW := LAYOUT ∪
{with_value_units}`, `_LIFT` stays its own narrowing but asserts
`⊆ SCALAR` in tests.

### 4.2 The kernel rulebook and the control-flow amendment (RULINGS, 211 §1.1–1.2)

A kernel body admits exactly: reading a tensor at an index, writing a
tensor at an index, scalar math on those values, captured scalars as
uniforms, and records. Everything tensor-shaped — the 18 layout ops,
reductions, fold, the indexing family — is a **host act applied at
the call site**, with the view passed in. All accidental cases get
ONE refusal class, modeled on the existing reduction refusal's voice
("a host citizen here — kernels use it at a call site, not as a
value").

Control flow, as ruled: the mask law stands (spike-measured free);
expression `if` and predicate `and`/`or` are legal in ALL kernel
bodies, as the vertex tier already spells them. **Statement `if` is
admitted when its arms are store-free** — values may branch, effects
may not — statically adjudicated (a store is syntactically an
assignment to a buffer subscript), lowered by select-joining each
variable the arms assign (the value tier's existing join shape; a
variable assigned in only one arm with no prior definition refuses,
as there). An `if` containing a store refuses. Loops remain refused
at this tier — and the current SILENT drop of `for`-bodies to the
oracle path becomes a LOUD ledger event in the same change (211
§1.4's requirement: no tier degradation without a voice).

### 4.3 Adjudication: the registry IS the gate

The Lowerer already refuses ops missing from the registry it is handed
— the frozen `unknown op` refusal pins that door. The mechanism is
therefore already built; today every seam simply passes the full
union. The fix: each capture seam passes **its tier's registry**,
derived from the table. Helper inlining (dialect.py:829) inherits
`ctx.ops`, so helpers are checked under their *caller's* tier for
free; the same holds for every fall-through call rule — `_k_call` may
still fall through to `_tl_call`, but `_tl_call` can no longer emit
what the kernel registry doesn't carry.

Two refusals, distinct and both in the house voice:

- **unknown op** (exists nowhere) — wording unchanged, battery-pinned;
- **wrong tier** (exists, not here) — the rulebook's one class, e.g.:
  *"tl.flip is a host citizen here — kernels use layout at the call
  site, not as a value: apply the view OUTSIDE the body (stage it on
  the parameter) and pass the result in."*

A `check_tier(region, tier)` verifier covers regions that arrive
without capture (`fold_region`, hand-authored Builder regions,
transform outputs — dce/grad/checkpoint must PRESERVE tier, a cheap
invariant test). It recurses into sub-regions under the tier **the
carrying op declares** (§4.5), and its output IS the ledger row
source. `pdum.dsl` stays untouched — tiers are a tl concept, the dsl
remains tier-ignorant, and the loc budget is unaffected.

### 4.4 The census (RUN with this proposal, 2026-07-28)

An op census over every region the full suite builds — Region
construction instrumented, attributed to the innermost active capture
seam (`_compile` → compute, `lower_body(kind)` → step,
`_lower_vertex`/`_lower_fragment`, assemblage `_lower` → unit,
`lower_handle` → dsl; everything else → tensor). Suite green under the
instrument (802 passed, 2 skipped). Findings, beyond confirming the
§5 table for unit/step/vertex/fragment almost exactly:

1. **Kernel regions really contain `tl.split`/`tl.merge`** — the
   machinery emissions of §4.1, now classified as such (initially
   proposed as an author-facing GEOMETRY family; 211 §2b's file:line
   attribution settled that they are compiler-internal).
2. **Cross-tier quoting is real**: `core.for`/`core.if` observed
   inside the compute seam's extent — the oracle fn-argument channel
   carries dsl-tier programs as data (§4.5).
3. **Transform-rebuilt regions attribute to no tier** — i.e.
   transforms must preserve tier and the implementation records tier
   where regions are built, so the §4.3 invariant is checkable.

Census caveat, courtesy of the campaign's spike scars (211 §2e):
**`args` is not dataflow** — `tl.iota` carries the lattice tensor as
`args[0]` and never reads it, so the census walker (and any admission
walker) over-collects through args; the op table, not the arg list,
defines dataflow. Same lesson for `walk_region`: it does not descend
into `n.regions`, which is moot today and bites the day the bounded
loop lands. Both facts are implementation constraints on `check_tier`.

### 4.5 Quoted programs

An op may carry a foreign-tier region as DATA: `tl.fold` carries a
step-tier region inside a tensor-tier program; the oracle fn-argument
channel carries dsl-tier programs for identity and per-element
dispatch; the bounded loop will carry its body the same way.
`check_tier` recurses with the carried tier, declared on the carrying
op's row — never inferred from context.

## 5. The table (ratified where marked; residue in §6)

| tier (`tl.kind`) | families | excluded, deliberately |
|---|---|---|
| **tensor** (`unit`, `step`, producer, zoo) | STRUCTURAL, SCALAR, POINTWISE, COMPUTE, LAYOUT, tl.iota | EFFECT (no stores/reads outside kernels), GRAPHICS, ABI, tl.coord (thread position is a kernel/graphics fact) |
| **compute** (kernel) — **RATIFIED** (the rulebook, §4.2) | STRUCTURAL, SCALAR, ABI, LATTICE, POINTWISE, EFFECT; records; store-free statement `if` (select-join), expression `if` | LAYOUT/COMPUTE/indexing as author vocabulary (host acts at the call site; `tl.split`/`tl.merge` remain as machinery emissions, ledgered), GRAPHICS, loops (bounded loop is scheduled work, §7) |
| **vertex** | STRUCTURAL, SCALAR, ABI, tl.iota, POINTWISE, plus tl.select *on buffer params only* (the pulled-read spelling) | EFFECT (vertex is pure — it yields), COMPUTE, LAYOUT otherwise, GRAPHICS |
| **fragment** | STRUCTURAL, SCALAR, ABI, POINTWISE, GRAPHICS | EFFECT, COMPUTE, LAYOUT, LATTICE (position arrives as varyings/params) |
| **spliced fn-arg** (already enforced, kernel.py:610) | STRUCTURAL, `_LIFT` ⊂ SCALAR | everything else → the oracle path (a ledger row, not a silent drop) |

`step` sits in the tensor tier today; a narrower device-step tier is
scheduled for the device-fold era, not this proposal (211 §2b notes
`_FOLD_STEP_SUPPORTED` already narrows the adjoint's view of steps).
The vertex/fragment rows also fix the `tl.kind` mislabeling (both say
`"compute"` today — graphics.py:418/500): kinds become honest names
first, then gates.

Runtime-side constraints folded in from 211 §3: admission demands an
**affine layout, never contiguity** for store/read targets
(contiguity is today's executor limitation — a ledger row, not a
law); no op's semantics may depend on residency; OOB safety never
leans on a device-side net (the keying-ladder discipline is
preserved by construction: the reference refuses OOB, devices run
certified cases); i64 captures become a **declared narrowing or a
refusal** outside f32's exact range — the three silent narrowing
sites in 211 §2b are ledger rows until fixed.

## 6. The residue — RATIFIED in one pass (owner, 2026-07-28)

1. **The tensor-tier row** — ratified as proposed (exclusions:
   EFFECT, ABI, tl.coord, GRAPHICS).
2. **The vertex/fragment rows** — ratified, including the
   tl.select-on-buffer-params side condition and the honest kind
   names (vertex/fragment stop lowering as "compute").
3. **The ledger's shape** — ratified: TOML in the tensorlib package,
   bidirectional drift tests (a ledgered row that now translates
   fails too), facet rows each name the test that guards them (CI
   asserts it exists), version integer, conformance skips quote
   their row.
4. **The refusal-battery bundling** — ratified; under the same-day
   refusal relaxation (tripwire standing, AGENTS.md) the bundle is
   tidiness rather than an API-break ceremony: geometry supersession
   rewording + the rulebook's new pins land in one battery repin.
5. **`abi.slot` at the unit tier** — excluded, ratified
   conditionally: implementation attributes the stray
   "tensor"-bucket census hit and reports back before the table
   freezes; a legitimate unit emission comes back to the owner
   rather than silently widening the table.

Resolved since the first draft: window/stencil-in-kernel (the
rulebook: host acts at the call site — no in-body views), `tl.read`
at tensor tier (kernel-only; `take` is the tensor gather), residency
implications (211 §3: no admission changes; the aliasing refusal
gains load), GEOMETRY family (reclassified as machinery emissions),
and the quoted-program rule (§4.5, forced by the bounded loop).

## 7. Scheduled work this proposal creates or absorbs

- **Implementation** (after §6 adjudication): families table +
  per-tier registries, the rulebook refusal class, statement-`if`
  select-join lowering, `check_tier`, the ledger + its drift test,
  the loud oracle-drop, honest `tl.kind` labels, tier battery.
  Sequencing per 211 §4: this merges FIRST; the campaign rebases on
  it (they touch none of kernel.py/dialect.py/rule packs meanwhile).
- **The bounded loop (mine, whole — 211 §1.4):** fixed-max-count loop
  with declared early-exit; reference semantics = run to the bound
  and mask after exit (straight-line for AD and analyses);
  backend-licensed to lower to a real `break`. Design doc of its own;
  the flat form's 2.2–4.0× cost and the 1.09× saturated case
  (`spike_controlflow/FINDINGS.md`) are its evidence base.
- **`Tensor.to_numpy` fast path (mine — answering 211 §4):** the
  per-element materializer is 88–100% of every device launch; a
  strided-view fast path differential-tested against the naive loop
  (which stays as the oracle). Highest-leverage small fix in the
  tree.
- **Battery subjects (mine — answering 211 §2e):** a `where`/select
  conformance subject + wide-range/adversarial input families, riding
  the same battery change as the ledger wiring. Per-target math rows
  and their freeness proofs (the tanh clamp) stay with the campaign's
  runtimes package.
- **`tl.sample`'s home (handed to the graphics/runtimes team — owner
  direction, 2026-07-28):** the sample op and the texture/sampler
  objects belong together, and that place is not tl — sample
  parameters may prove backend-specific (WebGPU and Metal are the two
  graphics backends in view), so the natural destination is the
  runtimes package they are already designing. The tier table is
  unaffected: the GRAPHICS family row tracks the op wherever it
  lives, and the fragment tier's admission is by family, not by
  module path.

## 8. What does not change

No new ops, no semantic changes, no behavior change for any legal
program except where a ruling says so (statement-`if` becomes legal;
smuggles start refusing; the oracle drop gets a voice). The
unknown-op refusal keeps its wording. Backend `Untranslatable` keeps
its role, scoped to the ledger. Regions the suite builds today keep
building — the census is the proof obligation, and any in-repo
smuggle it finds is a bug fixed in the same change, differentially
tested.

## Appendix: the census, observed op sets (2026-07-28, suite @ 802 green)

Attribution is by innermost dynamic capture seam — see §4.4 for the
three noise sources (cross-tier quoting; transform-rebuilt regions;
args-overcollection).

- **compute**: abi.slot; core.{add, cmp, const, div, env, extract,
  for*, if*, mul, param, sub, tuple, yield}; pw.sqrt; tl.{const, iota,
  merge†, pointwise, read, split†, store, token}  (* = quoted oracle
  programs, §4.5; † = machinery emissions, §4.1)
- **vertex**: abi.slot; core.{const, div, param, tuple, yield};
  tl.{const, iota, pointwise, select}
- **fragment**: abi.slot; core.{add, cmp, const, env, if*, mul, param,
  sub, tuple, yield}; pw.{cos, sin, sqrt}; tl.{pointwise, sample}
- **unit**: core.{const, param, tuple, yield}; tl.{const, iota,
  pointwise, random, reduce, rename, repeat, repeat_like, select,
  take}
- **step**: core.{const, param, tuple, yield}; tl.{argsort, argtopk,
  bind, const, decimate, diagonal, flip, iota, merge, pad, pointwise,
  reduce, rename, repeat, repeat_like, round_to, scan, scatter_add,
  select, shift, slice, split, stencil, take, window, with_charts}
- **dsl**: core.{add, cast, cmp, const, env, extract, for, if, mul,
  param, sub, yield}
- **tensor** (untiered remainder; includes transform outputs and
  eager-producer regions): the full tensor surface plus abi.slot,
  core.{cast, field, for, if, mod, neg, select}, math.{cbrt, sinh,
  tanh}, pw.{abs, exp, log, maximum, minimum, sin, sqrt, tanh},
  toy.blit (an open-registry registration)
