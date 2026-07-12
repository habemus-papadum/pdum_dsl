# Combinator & pipeline design notes

**Status:** design notes, recorded 2026-07-11 from the ch03-era discussion.
Companion to `docs/desiderata.md` §4.5/§5 and `design/deep-learning-notes.md`
§6.5 (the `>>` sugar question — this document answers it). Source semantics:
[`pdum_plumbum`](https://github.com/habemus-papadum/pdum_plumbum). Implies no
new kernel mechanisms; the one kernel-adjacent evolution (kind → Role) was
latent in the architecture.

---

## 1. The idea

One **blessed combinator library** — the way NumPy is blessed in numba —
providing plumbum-style pipeline syntax over kernels:

- **Definition is not application.** `|` composes stages into an inert,
  first-class pipeline value; `>` threads a value through it, exactly once.
  (`op = add(1) | multiply(2)` … later … `6 > op`.)
- Stage constructors are curried: `add(1)` *configures* a stage, it doesn't
  run anything.
- When the stages are jitted kernels, `|` must not build a Python-level chain
  of opaque calls — it mints a **new jitted composition** whose identity,
  caching, and compilation follow the same rules as everything else.

The deep reason this fits: **plumbum's define/apply split is the thesis's
phase split.** "Compose first, execute later" is phase A; `>` is phase-B
dispatch. A pipeline is a Handle built cheaply and repeatedly; ch02 already
measured the composition half (structural identity through nested Handles,
O(1) child incorporation, rebuild-hits).

## 2. Mechanism

### 2.1 Combinator identities are `Derived` templates

The blessed library mints compositions **without source**: a `pipe(f, g)`
Handle carries `Derived("pipe", …)` identity over the operands' `FnType`s,
and a registered **build rule** emits the composed body directly as IR at
lowering time (call f, feed g) — no `inspect.getsource`, no synthetic code
objects. This is the same seam t-string mini-languages use (a program-building
frontend), and it doubles as the dress rehearsal for transforms: `grad(f)`
uses exactly this Derived-template + build-pipeline path in M3/M4.

(A closure-factory `compose(f, g)` works today and is what ch02 measured; the
Derived form is the principled version for the blessed library because the
combinator's "body" is structure, not source.)

### 2.2 Roles: `kind` grows up

`Handle.kind` is today an uninterpreted string, deliberately. Combinators are
the feature that graduates it into a registered **Role** value: what kind of
kernel this is, optionally carrying a declared interface signature, and
naming the dialect that gives its bodies meaning.

**Settled at the ch04 walkthrough (2026-07-11, sharpened twice):** the
library pre-enumerates nothing. It ships machinery plus the single concept it
owns (`materializer`, backing its own `collect`); terminality is *structural*
(`Role.terminal` — nothing follows a terminal, a terminal may end anything),
not a pair rule. Even `device` — the base language's neutral composable and
`@jit`'s default label (name affirmed) — is owned by the stdlib/core-dialect
package and registered there once lowering exists; until then chapters/tests
register it as a labeled stand-in. Domain vocabularies (`fragment`,
`compute`, `audio_node`, …) are registered by their dialect/backend packages
at import time, the same discipline as backend-owned intrinsics.

### 2.3 The compatibility primitive: a rule, returning semantics

**"Compatible" is not a boolean — it is a lookup that returns the
composition's operational semantics:**

```
compose_rule(op, role_a, role_b) -> CompositionBuilder | IncompatibleRoles(explanation)
```

| `a \| b` where roles are… | The rule returns |
|---|---|
| `device \| device` | **fusion**: inline into one kernel body (monomorphic inlining; one artifact, zero call overhead) |
| `fragment_pass \| fragment_pass` | **orchestration**: a render graph — pass A to a texture, pass B samples it; entry points cannot inline |
| `audio_node \| audio_node` | a buffer-chain contract at block rate |
| `fragment \| device` | `IncompatibleRoles` with a remediation-bearing message |

Registered per `(op, role_a, role_b)` — the rule matrix again. The blessed
library ships roles + combinator ops + composition rules; the kernel doesn't
change. Precedent in plumbum itself: "one `@apb` stage upgrades the whole
pipeline to async" is a composition-time semantics decision driven by stage
properties. Fusion-vs-orchestration generalizes that into an
**execution-mode lattice**: any host-level stage demotes a pipeline from
fused to orchestrated, visibly.

### 2.4 When compatibility is checked (three tiers)

1. **Role check at pipe time** (phase A, always): cheap, catches category
   errors immediately, loud with an explanation — `MissingRule` discipline
   applied to composition.
2. **Type check at specialization** (phase B, always): `FnType` deliberately
   carries no arg/result types (device fns are Julia-style generic until
   specialized), so precise checking is inference's job at first application,
   failing with a `loc`-bearing error. This default keeps `pipe` polymorphic
   for free.
3. **Opt-in early precision**: roles may declare interfaces
   (`ImageFilter: vec2 → vec4`) for pipe-time type checks; and because
   lowering/inference are pure, `dry_check(pipeline, arg_types)` — inference
   without codegen — is a nearly-free probe for library authors and tests.

### 2.5 Caching payoffs

- Pipeline definitions rebuilt per frame hit the specialization cache (ch02,
  measured; Derived identities are value-compared the same way).
- **Associativity is unified by the artifact tier**: `(f|g)|h` and `f|(g|h)`
  have different template trees (different specialization keys) but flatten to
  identical IR — the content-addressed artifact tier gives them one compiled
  artifact. No canonicalization pass needed.

### 2.6 Mixed pipelines

A pipeline mixing plain Python callables and Handles has three named
outcomes, chosen by the role rules: **lift** the plain function if jittable;
**orchestrate** at host level (kernel A launches, Python stage runs, kernel B
launches — plumbum's normal semantics with compiled stages); or **refuse**
loudly. Never silently absorb a host stage into what claims to be fused.

## 3. Decision: internalize, don't (yet) modify plumbum

**Internalize.** The blessed library is a small in-repo satellite
(`combinators/`, later; ~100–200 lines including roles) implementing the
plumbum syntax discipline natively over Handles, crediting plumbum as the
source of the semantics. Rationale:

- What we need from plumbum is the **syntax discipline** (`|` inert
  composition, `>` single-shot application, curried stage constructors) —
  small. What we don't need in kernel pipelines: its host-execution
  machinery (async upgrade, iterator laziness, materializers).
- A dependency (either direction) couples kernel-pipeline semantics to an
  external release cycle for no mechanical gain.
- The future interop is the right kind of upstream change: when host
  orchestration exists, plumbum grows a tiny **protocol hook** ("if a stage
  defines `__pdum_compose__`, delegate composition to it"), so real plumbum
  host pipelines can contain kernel stages. ~20 lines upstream, no fork, and
  the two libraries stay independently evolvable. Revisit then, on the dev
  branch.

## 3b. Outputs, launch geometry, and materialization (added after the DPS discussion)

**Kernels don't return values** — physically they write into destinations
they were handed (a compute kernel's out-buffer, a fragment shader's render
target). The functional `y = f(x)` is the language-level truth; the ABI-level
truth is **destination-passing style (DPS)**, and the bridge is a standard
legalization (MLIR bufferization, XLA buffer assignment, C's `sret` are the
precedents; ours is simpler because pipelines are linear/tree-shaped).

- **Marshaling is bidirectional.** The input half exists in the design
  (flatten → `PackPlan` slots). The output half is its mirror: a result type
  legalizes into `abi.slot(out)` destinations, and a **`ResultPlan`** (built
  once per cache entry) says how to *allocate* destinations (from types, or
  from cheap arithmetic over runtime shape leaves via a "shape" rule column)
  and how to *unflatten* device buffers back into a logical value.
  Destinations are reused across calls while shapes hold (the FastRecord
  staging discipline). **Step 7's contract is therefore bidirectional from
  the start** — inputs and results — not outputs bolted on later.
- **Intermediates never touch Python.** Fused pipelines have no
  intermediates (one kernel). Orchestrated pipelines wire stage k's
  out-slots to stage k+1's in-slots — device-to-device buffer forwarding is
  *what orchestration composition rules produce*.
- **Materialization is a boundary act.** Between stages: opaque
  `DeviceValue`s (buffer + Type, no copy). To Python: only at the terminal —
  by role convention at `>`, or explicitly via a **materializer terminal**
  (plumbum's `list`/`alist` materializers, transplanted).
- **Launch geometry splits by the two-tier law** (Triton precedent):
  block/tile sizes change generated code → codegen params, in the artifact
  key; grid size is runtime data, usually auto-derived from output shape by
  a role rule. Syntax: `f[grid, block]` (Triton-familiar `__getitem__`)
  returns a *configured stage*. Until execution exists, the definition layer
  records config **conservatively as static** (changing it recompiles —
  never wrong, sometimes wasteful) and the static/runtime split lands with
  the backends.
- **In-place/donation** (`a >> f` overwriting `a`'s buffer — JAX
  `donate_argnums`) is parked as a later explicit opt-in.

## 4. Decision: sequencing (approved 2026-07-11)

**Do it next, as a step, not in parallel** — split in two so the syntax
question is answered before any GPU backend exists:

- **Step 3b — pipelines as values (definition layer), immediately after
  ch03.** Internalized syntax core: stage constructors, `|`, `>`-as-dispatch,
  Roles v1, the composition-rule registry, `IncompatibleRoles`. Provable
  *today* with the ch03 dummy-artifact technique: definition ≠ application,
  identity stability across rebuilds, role errors, associativity keys,
  pipeline-of-pipelines. Chapter: `ch04-pipelines-are-values`. This answers
  "will the syntax actually work" with running code, before the IR exists.
- **Execution follows the existing steps.** The lowering step grows the
  Derived/combinator **build rule** (forcing the Derived-lowering seam early
  — otherwise untested until transforms), and the Python-backend chapter
  executes `6 > pipeline` for real. From there on, **combinator style is the
  house style for every example** — the WGSL chapter's disk demo becomes a
  composed pipeline, which is the compelling-examples goal.
- Not in parallel because the step gate is the point: this surface is
  taste-heavy (the user's own library's feel), and it should calcify only
  after a walkthrough — same reason tuple→Vec was worth catching early.

Plan deltas when approved: insert step 3b (chapter numbering shifts by one
from the IR chapter on; ch00's map updates when the chapter lands); add the
build-rule line to the lowering step; restate the house-style rule in the
plan's working agreements.

## 5. Open questions (parked with the feature)

- **What `>` means for entry-point kernels**: application = *draw*, which
  drags in targets/drawers — likely the runtime's business (`target < pipeline`
  or a `render(pipeline, target)` terminal), not the combinator's.
- **Placeholders and fan-in**: plumbum's `>` fills the first argument;
  fan-in/fan-out combinators (`parallel`, `branch`) need their own role rules
  and their own identity shapes.
- **Effect ordering for orchestrated pipelines** (render-graph scheduling) —
  belongs to the runtime layer that owns passes, far later.
- **Does Role enter the specialization key?** Today `kind` doesn't; composition rules
  run at build time, so it likely never needs to. Revisit only if a role ever
  changes *generated code* for the same template.
- **Naming**: "Role" vs "kind" vs "interface" — settle at the ch04
  walkthrough, in the glossary.
