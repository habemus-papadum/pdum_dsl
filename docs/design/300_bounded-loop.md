# 300 — The bounded loop: declared early-exit, run to the bound, mask after

**Status: ruled (211 §1.4, owner option "a") and LANDED at reference
tier (2026-07-28).** The backend real-`break` license and kernel-body
loops are the scheduled remainder, deferred with the tiling era by
owner direction — this doc records what exists and the license shape.

## The construct

A fixed-max-count loop with a **declared early-exit**, spelled as the
Python it is:

```python
for _ in range(64):          # the bound is the law: bounded loops only
    h = sdf(o + t * d)
    if h < eps:              # THE declared exit: `if <cond>: break`
        break
    t = t + h * step
```

The spelling law: exactly one `if <cond>: break` (single-statement
body, no `else`) at the top level of the loop body, any position; the
condition is strict bool. A second exit clause refuses; a bare
`break`, or one nested deeper, keeps the base pack's refusal ("guard
with `if` instead"). Everything else about `for` is unchanged
(`range` bounds, strict i64, strict carry types, loop-var death).

## Semantics: values freeze at the exit

**Reference semantics = run to the bound and mask after exit.** The
iteration where the condition fires contributes its carries AS OF the
break point (updates later in that iteration are masked away);
subsequent iterations contribute nothing. For VALUES this is
identical to a real `break` — the masked form and the broken form
differ only in COST (skipped-work evaluation; the campaign measured
the flat form at 2.2–4.0× on a 64-step raymarcher, 1.09× when
saturated — `spike_controlflow/FINDINGS.md`). Both sides of every
mask evaluate, exactly like expression-`if` under the mask law — safe
under the IEEE non-trapping policy.

## The encoding, tier by tier

- **IR**: `core.for` gains an `exit` attr; its region yields
  `(carry, done)` — where `carry` is already the break-point select
  (`select(done, carries_at_break, carries_at_end)`), built at
  lowering (`_for_stmt`). No new op; the `_for` type rule is
  untouched (carries flow through).
- **Host (the reference renderer)**: the loop takes the break license
  trivially — `res, _done = res; if _done: break` (reference.py's
  `loop_join`). Identical values, honest early stop.
- **Kernel splice (`_flatten_fors`, kernel.py)**: static-bound
  `core.for` unrolls into the FLAT MASKED FORM before `_liftable` —
  first iteration peeled, then per iteration
  `cur' = select(stopped, cur, stepped)` and
  `stopped' = select(stopped, stopped, done)` — pure select chains
  over the loop's own predicates, no boolean constants (the lift
  rows carry `core.select` as `where`). Consequence, verified by
  differential and translation: **a loop-bodied fn-arg now SPLICES
  instead of dropping to the oracle, and the flat form translates to
  WGSL today** — the raymarcher shape reaches the device column with
  zero new device rows. Non-static bounds stay structured, and the
  oracle serves them LOUDLY (`kernel.oracle_fallback`; the ledger's
  `compute/oracle-fn-arg` row, updated at version 2).
- **AD**: the flat form is straight-line where/select chains — the
  one derivative table already serves it (where's at-kink rule
  applies at the exit boundary). No loop-structured adjoint exists or
  is needed at this tier.

## The backend license (future, with tiling)

A backend may lower the loop to a real `break` — the license is
exactly "values identical, cost lower", and the host renderer is the
existing proof that the license is sound. Two routes are on record:
keep the structured `core.for` visible to a device backend that wants
real control flow, or reconstruct from the flat form (the campaign's
if-reconstruction peephole: built, proven, shelved —
`spike_controlflow/ifrecon.py`, ~135 LOC). Choosing between them is
tiling-era work; nothing in the landed encoding forecloses either
(`walk_region`'s no-descent-into-`n.regions` caveat, 290 §4.4, bites
exactly here — the reconstruction route must walk regions).

## Deliberately out of scope

Kernel-BODY `for`+`break` (statement loops in `@compute` bodies stay
refused; the fn-arg path is the ruled consumer); `continue`; multiple
exits; `while` (refused by the base pack's bounded-only law);
non-static bounds on the splice path (oracle, loud).
