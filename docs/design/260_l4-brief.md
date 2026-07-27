# 260 — The L4 brief: onboarding the next machine

Written at P9's close for the agent (and human) starting L4 on the
Linux box, off `main`. This is the context bootstrap: what to read, what
is RULED and closed, what to build first, and the traps already sprung
once. The owner's own guidance rides on top of this and wins.

## Read, in this order

1. **200_the-spec.md** — the ratified spec. §8 is YOUR brief (the
   runway); §7's P9 entry records what just landed; §4 the precision
   doctrine; §1.9 the indexing family.
2. **250_coordinate-algebra.md** — normative for the indexing surface
   at every tier. §10 is the value-door law (owner-ratified aligned
   law + `over=`).
3. **packages/tensorlib/docs/LEVELS.md** — K-A…K-G (the L4 design
   questions), the Program-excavation plan, the L2 blocker list, the
   open registry, and the P9 rulings.
4. **packages/tensorlib/docs/BOUNDARIES.md** — every deliberate limit,
   written next to its code. **210_backend-notes.md** — the distilled
   backend knowledge you inherit. **230_syntax_tour.ipynb** — executable;
   run it, don't skim it.
5. The zoo (`pdum/tl/zoo/`) — gemm.py is your first benchmark;
   trainer.py is the end-to-end stress; test_zoo pins every entry.

## Ruled and CLOSED — do not re-litigate

These are owner decisions with frozen artifacts. Changing one is a spec
amendment the owner makes, never a refactor:

- **No magic.** Nothing auto-promotes, auto-coerces, or acts implicitly.
  Coercions are visible doors (`f32`/`i32`/`snap`); ints never silently
  join float math; a Python literal's own type is a carrier declaration.
- **The call-boundary veto.** Coordinates cross calls AS Coordinates;
  the language never casts FOR you — the caller or the callee spells it.
- **Kernel stores never scatter.** Assignment at a value index refuses
  permanently (the refusal is the theorem, pinned live). Device scatter
  is reduce-by-index with a DECLARED monoid — that is K-G, yours to
  design, but the assignment pun stays banned.
- **Refusal messages are frozen API** pinned by literal wording; tests
  match them. Rewording one is an owner decision.
- **`extent` = the width, a host int, ONE rule at every tier.** The
  full domain is the Frame's job.
- **Named, order-free subscripts; strict frame identity + containment
  extent; indexing never promotes to scalar** (`.item()` is the exit).
- **take/scatter_add**: the aligned law (same name = same lattice) and
  `over=` (declared consumption) are ratified; indices are DATA
  (integer-carrier, unitless), never Coordinates; reference refuses OOB
  at run time — device OOB behavior is a descent-license matter, never
  silent.
- **Declarations over recognition; delete-don't-archive; canon speaks
  present tense; every limit is a conscious recorded decision**
  (BOUNDARIES). Oracle execution is spelled `reference(f)(...)`.
- **NEVER trigger the release workflow.**

## Your first act: the excavation (before ANY backend work)

The Program/Instr IR dies; the dialect (Region/OpDef, one Lowerer) is
the one engine. The plan is in LEVELS.md ("The Program excavation"):
retarget shape-reading analyses first (signatures, opcount, memory,
placement), then transforms, then autodiff LAST and alone
(`derive_vjp` on regions is the seed), then delete
`export_program`/`ir.run`. One consumer per step, suite green between.
Everything P9 added was built single-copy with thin adapter rows so
each op is a REPOINT, never a rewrite. Do not write new L4 machinery
against Programs — it would double the debt.

## Then: L4 proper

K-A…K-G in LEVELS.md are the queued design conversation. Flagships:
tiled GEMM (zoo/gemm.py — tiling already IS layout there), flash
attention, the fused stencil chain. Legality = certified rewrite chains
+ per-level WF certificates; objective = parent-memory traffic under
child capacity. The license schema stub (`pdum/tl/licenses.py`) is the
declaration shape your descent registry consumes — the worked GEMM
declaration shows f16 tiles + f32 accumulators. Adversarial input
families gate the flagships (−inf masks, cancellation, non-divisible
tails) — never random draws alone. Classic vertex attributes and
shared memory (the ONE remaining spec skip) are backend lowerings here.

## Traps already sprung once — don't rediscover

- **The dialect canonically SORTS attr dicts** (node identity). Any
  order-sensitive param must ride as a tuple of pairs (split's parts
  was silently mis-nested until P9 caught it).
- **Fold steps support only `{tl.pointwise, core.const, core.param,
  core.yield}`** — materialized consts inside fold steps refuse; the
  deferred-scalar const is load-bearing there.
- **Lowered bodies**: ONE def (no nested defs — ambiguous source
  recovery), no `*args` splat (no Starred rule), straight-line at the
  step/unit tier (the `for` unroll is kernel-only; unit-tier unroll is
  a registry door). Captured helpers inline, tuple returns included.
- **Frames are never tuple-indexed** (`Frame.__getitem__` is the point
  factory); there is no out-of-bounds Coordinate — end-of-domain
  slices are spelled by omission.
- **Pipes mask exit codes** in gate commands (`… | tail`) — check
  exit codes unpiped; it has eaten a green-looking commit twice.
- **FD cannot validate declared estimators** (straight-through): FD the
  declared path's own target, not the objective.
- The suite gate: `uv run pytest packages/`, `uv run ruff check .`,
  formatting only on files you touched (repo-wide `ruff format` sweeps
  pre-existing drift — don't).
