# V5 — Verdict: where program transformations live

Consolidation verdict for the pdum.dsl redesign. Question: where do `grad`, `vmap`,
and later transformations live so they are first-class but do not bloat day 1?
Inputs: R4 (JAX), R6 (torch.compile), R7 (tinygrad/nanopass/QBE), R2 (DaCe), the
desiderata, and `design/dsl_caching_layer.md`.

---

## Recommendation

**Transformations are IR-to-IR passes over the single flat typed IR, whose entire
per-op content is per-primitive rules registered in the same op/rule registry that
holds type rules and backend lowerings — JAX's rule matrix as the content model,
tinygrad's rewrite-pass mechanism as the execution model, run at phase B before
backend lowering.** A transformed function (`grad(f)`, `vmap(f)`) is an ordinary
template with a *structural* derived identity `("grad", base_code_identity,
static_params)` that flows into the same type-keyed cache as everything else.
Backend-native AD (MLX, JAX-on-backend) is **rejected as a semantic owner** and
admitted only through the `custom_vjp`-shaped escape hatch: an opaque backend
kernel may *register* a hand-written derivative rule, but `grad` itself is always
the frontend pass. Day 1 ships zero transformation code — only the registry
columns (empty), five IR invariants, and the derived-identity slot in the cache
key, totaling well under 100 lines of obligation.

The apparent three-way choice in the question is a false trichotomy at pdum's
scale. In JAX, "transformation as interpreter over primitives" means a `Trace`
subclass intercepting live Python execution; pdum has no tracing — its programs
already exist as IR (AST-derived). Walking a flat eqn list and consulting
`jvp_rules[op]` per eqn *is* an IR-to-IR pass. So options (1) and (2) collapse
into one design: **the pass is generic and dumb; the ops are smart.** What must be
rejected is the strawman version of (2) — a monolithic AD pass with a giant
`match` over ops — and all load-bearing versions of (3).

---

## Rationale

**1. Per-primitive rules are the only shape that satisfies the prime directive.**
Every extensibility axis in the desiderata reduces to "adding X = adding rows or
columns in the (op × aspect) matrix": new op = one column of rules; new
transformation = one aspect row (a registry dict + a ~100-line pass driver); new
backend = one `lower[backend]` row. A monolithic AD pass would instead couple
every future op to the AD module — each new intrinsic would require editing the
transformation, exactly the coupling the M0 review flagged for the core→WGSL
tables. R4's matrix table (§3) is the picture to keep on the wall.

**2. The mechanism is proven cheap in both lineages.** Autodidax: interpreter core
+ jvp + vmap ≈ 292 code lines; reverse mode (partial-eval + transpose) ≈ 439;
one structured-control-flow op ≈ 182 (R4 §5). tinygrad: reverse-mode autodiff is
`pm_gradient`, **132 lines of per-op VJP rules** applied by the same ~150-line
`graph_rewrite` engine that does simplification and rendering (R7 §1.1, lesson 7).
Two independent codebases converge on "AD = a rule table + a generic driver, a few
hundred lines." There is no scale argument for anything heavier.

**3. Backend delegation fails on pdum's own primary backend and on its thesis.**
WebGPU/WGSL — the one backend that exists and the one the domains (generative art,
interactive sim) need first — has no native AD; neither do the C and Python
backends. Delegation would make `grad(f)` a partial function of the backend,
i.e. not first-class. Worse, delegating to MLX/JAX hands the derivative artifact
to a framework whose cache keys on function-object identity and bakes captures as
constants (R4 §2: ~240× retrace penalty for rebuilt closures; JAX ships
`JAX_EXPLAIN_CACHE_MISSES` solely to diagnose it) — re-importing the exact
anti-pattern pdum.dsl exists to fix, in the hot loop, behind an interface we don't
control. And torch's AOTAutograd (R6 §5) is the cautionary tale for "reuse an
engine": it works only because a mature eager autograd engine predated the
compiler; pdum has no such engine, so that path is not even available.

**4. Composition and cache heat fall out structurally.** Because a transform maps
template → template, `jit(vmap(grad(f)))` is pass composition over IR — no trace
levels, no `Trace` stack. And because the derived identity is a *value*
(`("grad", code_identity, argnums)`), a user who writes `grad(loss)` freshly every
loop iteration still hits the cache — the derived key is value-equal each time.
This is a concrete improvement over JAX, where `jit(grad(f))` rebuilt per
iteration retraces perpetually. The transformation story and the caching thesis
reinforce each other instead of fighting.

**5. Transformation at phase B, after type inference, is forced by the thesis.**
Differentiation is type-dependent (integer/bool values carry no tangent; the
tangent signature is derived from the typed signature), and pdum types are only
known once `(env_types, arg_types)` arrive. So the pipeline per derived cache key
is: base IR → type inference → transform pass(es) → backend lowering. The
transformed IR is cached with the artifact; the hot path never re-runs the pass.

---

## Considered and rejected

| Alternative | Why it lost |
|---|---|
| **JAX-style tracing interpreters** (Trace/Tracer over `bind`) as the literal mechanism | Requires trace-based program acquisition, which R4 lesson 2 already rejects for pdum (shader-style `if`/`for` would need a `cond`/`scan` shadow dialect, violating "user code looks like Python"). The *rule matrix* survives; the *tracing machinery* does not — pdum's programs are already IR, so the "interpreter" degenerates into a pass over eqns. |
| **Monolithic IR-to-IR passes** (AD pass owning a closed `match` over all ops) | Violates the prime directive: every new op edits the AD pass, every new transformation re-enumerates all ops. Also loses graceful partiality — with registries, an op missing a `jvp` rule fails precisely and only when differentiated, with an error naming `(op, aspect)`. |
| **Backend-native AD as semantic owner** (delegate `grad` to MLX/JAX where available) | (a) Unavailable on WebGPU/C/Python — the primary targets — so `grad` becomes backend-conditional, i.e. not first-class. (b) Foreign caches are identity-keyed with baked consts (R4 §2), reintroducing retrace-per-iteration inside the hot loop. (c) `vmap(grad(f))` cannot compose across the delegation boundary. (d) Per-backend numeric semantics divergence. Admitted only as: per-op custom-derivative registrations that may *call* backend kernels, and as a test-time oracle (check pdum's `grad` against `mlx.grad` numerically). |
| **AOTAutograd shape** (trace through an existing AD engine post-capture) | No engine exists to trace through; R6 lesson 8 states this outright. Building the eager engine first would be strictly more work than the rule table. |
| **Source-level (AST→AST) AD** (Zygote/Tapenade style) | Runs before type inference, so it cannot use type information (which vars are differentiable) and produces a second copy of every frontend problem (name resolution, control flow) inside the transform. IR level is where binders are typed and control flow is structured — that is where AD is a local rule application. |
| **Deferring all AD decisions to post-day-1** ("add it when we get there") | The cheap obligations below (typed binders, pure eqns, structured control flow with explicit carries, params-not-consts for captures, registry columns) are nearly free now and near-impossible to retrofit — JAX's const-baking and FX's control-flow absence are both examples of early representation choices that permanently constrained the transformation story. |

---

## The architecture, concretely

### Registry: per-aspect dicts, open set of aspects

Per-aspect module-level dicts (JAX's literal shape), not fields on a closed OpDef
dataclass — so a future aspect (`unit_rule`, `cost_rule`) is a new dict + pass,
touching nothing existing:

```python
# pdum/dsl/rules.py — the whole mechanism (~40 lines with errors/introspection)
type_rules:  dict[Op, TypeRule]                 = {}   # day 1, populated
eval_rules:  dict[Op, EvalRule]                 = {}   # day 1 (Python backend / reference semantics)
lower_rules: dict[tuple[Op, str], LowerRule]    = {}   # day 1, per backend
jvp_rules:   dict[Op, JvpRule]                  = {}   # column exists day 1, EMPTY
transpose_rules: dict[Op, TransposeRule]        = {}   # column exists day 1, EMPTY
batch_rules: dict[Op, BatchRule]                = {}   # column exists day 1, EMPTY

linear_args: dict[Op, tuple[int, ...]]          = {}   # transposability metadata (declared, cheap)

class MissingRule(DslError):
    """raised as: op 'foo' has no 'jvp' rule — grad(f) touched it at loss.py:12"""
```

Rule signatures (builder-passing style — a rule *emits eqns*, it does not compute):

```python
JvpRule       = Callable[[Builder, list[Atom], list[Atom | Zero], Params],
                         tuple[list[Atom], list[Atom | Zero]]]   # (primals, tangents) -> (outs, tangent_outs)
TransposeRule = Callable[[Builder, list[Atom | UndefPrimal], Atom, Params],
                         list[Atom | None]]                      # cotangent-out -> cotangents-in
BatchRule     = Callable[[Builder, list[Atom], list[int | None], Params],
                         tuple[list[Atom], list[int | None]]]    # (args, bdims) -> (outs, out_bdims)
```

A "battery" op package registers into the same tables — one op = `type_rule` +
portable decomposition (or `lower` per backend) + optional `jvp`/`batch`. A user
device function with a hand-written derivative uses the same registration shape
(the `custom_jvp`/`custom_vjp` escape hatch is just a call-op whose rules are
user-supplied — R4 §3). **This is also the sanctioned home for backend-native
gradients:** an MLX-backed intrinsic may register a `jvp`/`transpose` rule whose
lowering calls an MLX kernel; semantic ownership stays in the table.

### Pass driver: generic, ~100 lines per transformation

```python
def jvp_pass(prog: Program) -> Program:
    b = Builder(tangent_signature(prog))          # each diff-typed param gets a tangent param
    env: dict[Var, tuple[Atom, Atom | Zero]] = bind_params(b, prog)
    for eqn in prog.eqns:                          # flat, ordered, typed
        rule = jvp_rules.get(eqn.op) or _missing(eqn, "jvp")
        primals  = [env[a][0] if isinstance(a, Var) else a for a in eqn.args]
        tangents = [env[a][1] if isinstance(a, Var) else Zero for a in eqn.args]
        outs, touts = rule(b, primals, tangents, eqn.params)   # recurses into sub-programs for if/loop/call
        for v, o, t in zip(eqn.out_binders, outs, touts): env[v] = (o, t)
    return b.finish([env[o] for o in prog.outs])
```

`vmap_pass` is the same skeleton with `(atom, batch_dim)` in `env`. Reverse mode
is composed, not monolithic (R4 lesson 6): `grad = jvp_pass ∘ partial_eval ∘
transpose_pass`, so the jvp rules are written once and reverse mode reuses them;
only `transpose_rules` for the linear ops are additional work.

### Cache integration: transforms are templates

```python
# template identity today:            (code_object [value-compared], globals_tag)
# derived template identity:          ("grad", base_identity, argnums, has_aux)
#                                     ("vmap", base_identity, in_axes_treedef)
# native cache key (unchanged shape): (template_identity, env_types, arg_types, backend, backend_params, generation)
```

Phase A on `grad(f)` is compile-free exactly like phase A on `f` — it wraps the
same captured `Env`. Phase B, on first call per type signature: infer types on
the *base* IR, run the transform pass(es), lower, cache under the derived key.
`grad(f)` rebuilt every loop iteration is value-equal → cache hit → the loop
stays hot. (torch precedent, R6 lesson 8: transformed artifacts flow through the
identical cache machinery.)

### User-facing syntax

```python
from pdum.dsl import jit, grad, value_and_grad, vmap

@jit
def loss(params, x):
    e = model(params, x) - target      # `target` captured: typed env slot, differentiable
    return (e * e).sum()

g   = grad(loss)                       # d loss / d params (argnums=0 default)
gs  = grad(loss, argnums=(0, 1))
v,gp = value_and_grad(loss)(params, x)
bf  = vmap(f, in_axes=(0, None))       # axis spec per arg; None = broadcast
h   = jit(vmap(grad(loss)))            # composition = pass composition; jit is idempotent

@jit
def render(p):
    dI = grad(intensity)(p)            # transforms usable *inside* DSL code later:
    ...                                # lowered as a call-op to the derived template
```

`grad`/`vmap` accept both decorated templates and plain functions (they apply
`jit`'s capture machinery first). Differentiation w.r.t. *captures* is expressible
(`grad(f, wrt=capture_name)` later) precisely because captures are typed
parameters, not baked consts.

---

## Day-1 obligations (the entire cost of "AD-ready without AD")

These are IR/registry invariants, not features. Estimated total: < 100 lines plus
discipline. Everything else (the passes, the rules) ships in later milestones.

1. **Flat, fully typed, pure eqn list.** Every binder carries a type; eqns may
   have multiple results; no hidden state; no aliasing/mutation *inside* the IR.
   Assignments functionalize during AST lowering (SSA-style renaming); memory
   effects (output stores) are confined to the program boundary (params in, one
   sink of stores at the end), never interleaved with the pure region. This is
   what makes every transformation a local rule application. (R4 lesson 4/6;
   tinygrad's single pure UOp graph, R7 §1.2.)
2. **Structured control flow as a minimal higher-order op set — `call`, `if`,
   `loop` — carrying sub-programs in `params`.** No unstructured jumps in the IR
   (frontend lowers or rejects `break`/early-`return` initially). `if`: two pure
   sub-programs with identical result types (differentiable by recursion;
   vmappable by both-branch + select, which *requires* branch purity — decide that
   now). `loop`: counted form with **explicit loop-carried values** of fixed type
   (scan shape) — the only loop form reverse mode can handle without heroics.
   Budget consciously: each higher-order op costs a rule from every transformation
   (~180 lines for `cond` in autodidax), so the set stays at three. (R4 §5, R6
   lesson 5 — FX's missing control flow is the anti-model.)
3. **Captures are typed `envvars`/params with marshaling annotations — never
   consts.** Already the thesis; restated here because it is *also* the AD
   enabler: JAX's `constvars + consts` is exactly what makes its captures
   non-differentiable-without-retrace. One substitution, two payoffs. (R4 §2.)
4. **Registry columns declared on day 1, unpopulated**: `jvp_rules`,
   `transpose_rules`, `batch_rules`, `linear_args`, plus the `MissingRule` error
   naming `(op, aspect, source location)`. Cost: ~10 lines. Ops written for M1
   simply don't register those aspects; nothing breaks until a transform touches
   them.
5. **Derived-identity slot in the cache key.** Template identity must be a
   sum type (`base | derived(tag, base, static_params)`) from day 1 so `grad(f)`
   and `f` never collide and derived templates are value-keyed (hot-loop safe).
   Cost: ~15 lines in the key structure.
6. **Reference eval semantics per op** (`eval_rules` — the Python backend). This
   is day-1 work anyway, and it is what lets transformation rules be tested
   op-by-op against finite differences before any GPU backend learns about them.

Sequencing after day 1 (autodidax-measured cost ordering, R4 lesson 6):
`vmap` (~100-line pass + 1 batch rule/op) → `jvp` (~100-line pass + 1 rule/op) →
reverse mode (`partial_eval` + `transpose_pass`, ~400–500 lines, ship last).
tinygrad's 132-line `pm_gradient` suggests the direct-VJP-rewrite shortcut is
viable if the jvp+transpose decomposition ever feels heavy for the op set we
actually have — the registry shape admits either implementation without user-visible
change.

---

## Confidence and what would change my mind

**Confidence: high** on the architecture (per-primitive rules in the shared
registry, executed as phase-B IR-to-IR passes, unified type-keyed cache, no
backend delegation). Four independent lines of evidence converge: JAX's matrix is
the extensibility proof, tinygrad's 132-line gradient is the mechanism-cost proof,
torch's AOTAutograd is the counterfactual (post-hoc AD needs a pre-existing
engine), and the caching thesis independently rules out delegation.

**Moderate confidence** on two details:

- *The loop-AD form.* The counted-loop-with-explicit-carries commitment is right
  for reverse mode, but if pdum's shader domain turns out to need data-dependent
  `while` (ray marching) *differentiated*, the story becomes bounded-trip-count +
  checkpointing — more design work than budgeted. Evidence that would change the
  verdict's shape (not its location): profiling real generative-art kernels
  showing differentiated unbounded loops are a day-100 need rather than day-1000.
- *jvp+transpose vs direct VJP rules.* If the op set stays small and pointwise-heavy,
  tinygrad-style direct VJP rules may beat the JAX decomposition on simplicity.
  The registry admits both; decide when reverse mode is actually built.

**What would overturn the recommendation itself:** (a) the IR redesign ends up
effect-heavy — arbitrary interleaved stores/scatters in the body that cannot be
confined to the boundary — making pure-eqn transformation rules unsound; the
fallback would be a runtime tape, and backend delegation would deserve a re-hear.
(b) The project narrows to MLX-only as a backend, in which case delegating to
`mlx.grad` and treating pdum as a frontend becomes defensible. (c) Measured
evidence that phase-B transform passes blow the interactive compile budget
(nothing in the inputs suggests this: passes are linear in eqn count and run once
per type signature).

---

## Design lessons for pdum.dsl

1. **One registry, many aspects, is the whole transformation architecture.**
   Declare `jvp_rules`/`transpose_rules`/`batch_rules`/`linear_args` as empty
   per-aspect dicts on day 1 (~10 lines) with a `MissingRule` error naming
   `(op, aspect, source location)`. Adding a transformation later = one dict +
   one ~100-line pass driver; adding an op never touches transformation code.
2. **Collapse "interpreter vs IR pass": the pass is generic, the ops are smart.**
   Write every transformation as the same skeleton — walk the flat typed eqn
   list, consult `rules[op]`, emit via a `Builder` — recursing into sub-programs
   of `call`/`if`/`loop`. Do not build tracing machinery; do not build monolithic
   per-transform visitors.
3. **Five day-1 IR invariants buy AD for free later**: typed binders everywhere;
   pure multi-result eqns with effects confined to the program boundary; exactly
   three higher-order ops (`call`, `if` with pure identically-typed branches,
   counted `loop` with explicit carries); captures as typed params (never
   consts); reference `eval_rules` per op for finite-difference testing.
4. **Transforms are templates; put the transform tag in the identity.** Template
   identity is `base | ("grad"/"vmap", base, static_params)` — value-compared, so
   `grad(f)` rebuilt per loop iteration hits the cache. Run transforms at
   phase B after type inference; cache transformed IR with the artifact; the hot
   path never re-transforms.
5. **Never delegate `grad` semantics to a backend.** Backend-native AD enters
   only through the custom-derivative escape hatch (an op whose registered
   jvp/vjp rule calls a backend kernel) and as a numerical test oracle. `grad`
   must produce IR that lowers to WGSL/C/Python like any other program — that is
   what "first-class" means here.
6. **Ship in the measured cost order**: type rules (M1, needed anyway) → vmap →
   jvp → reverse mode (jvp + partial-eval + transpose, the single most expensive
   component at ~400–500 lines — budget it as its own milestone). Keep the
   higher-order op set frozen at three until every existing transformation
   handles them, because each new one taxes all transformations (~180 lines
   each).
