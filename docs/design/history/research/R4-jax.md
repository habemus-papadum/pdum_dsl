# R4 — JAX internals: the model for first-class transformations

Research report for the pdum.dsl redesign. Verified against live sources July 2026;
JAX/jaxlib is at ~0.10.x (jaxlib 0.10.2 released 2026-06-17, Python >= 3.11)
([releases](https://github.com/jax-ml/jax/releases), [changelog](https://docs.jax.dev/en/latest/changelog.html)).
Primary sources cited inline; autodidax line counts measured directly from
`docs/autodidax.py` at `jax-ml/jax@main` (July 2026).

JAX is on our influence list for exactly one thing: **transformations (`grad`,
`vmap`, `jit`) that compose as first-class citizens**. This report explains the
machinery that makes that work — Tracer/Trace interpreters, the jaxpr IR, the
per-primitive rule registries, and the pytree marshaling layer — and documents,
with mechanics, the one place JAX is the anti-model: its `jit` cache keys on
**function object identity**, so a rebuilt closure retraces. That is the exact
weakness pdum.dsl exists to fix.

---

## 1. The tracing frontend: Tracers, abstract values, and what tracing cannot see

### Mechanics

JAX has no bytecode or AST frontend. `jax.jit(f)` **runs** `f` in the ordinary
CPython interpreter, but substitutes each array argument with a `Tracer` object —
a value wrapper that carries an **abstract value** (`aval`) and overloads every
operator. The central abstraction levels:

- `ConcreteArray(value)` — a specific array (used by eager `grad`);
- `ShapedArray(shape, dtype)` — "all arrays of this shape/dtype" — **the level
  `jit` traces at**, which is why one trace serves all values;
- (`UnshapedArray` exists but is essentially unused.)

Every operation inside `f` bottoms out in `bind(primitive, *args, **params)` —
the single interception point. `bind` finds the innermost active `Trace` (JAX
keeps a stack of `MainTrace` entries with levels; nested transformations = a
deeper stack), boxes arguments up to that trace's `Tracer` type, and calls
`trace.process_primitive(primitive, tracers, params)`. Every transformation is a
`Trace` subclass overriding `process_primitive`; tracing to an IR is just one
more interpreter (`JaxprTrace` records an equation instead of computing).
([autodidax Part 1](https://docs.jax.dev/en/latest/autodidax.html))

### What tracing cannot see

Because the real Python interpreter drives execution, everything that is not a
traced operation is **invisible and burned into the trace**:

- **Python control flow on traced values fails.** `if x < 3:` on a
  `ShapedArray((), bool)` tracer raises `TracerBoolConversionError` — the tracer
  "represents the set `{True, False}`" and cannot pick a branch
  ([control-flow docs](https://docs.jax.dev/en/latest/control-flow.html)).
- **Python control flow on *static* values is silently unrolled/specialized.**
  Loops over Python ints unroll; branches on Python bools bake in one arm. This
  is the feature ("Python-level control flow executes normally… the resulting
  jaxpr does not have to contain control flow" —
  [jaxpr docs](https://docs.jax.dev/en/latest/jaxpr.html)) and the trap
  (a 1000-iteration Python loop produces a 1000-equation jaxpr).
- **Data-dependent control flow needs structured primitives**: `lax.cond(pred,
  true_fun, false_fun, *operands)`, `lax.while_loop(cond_fun, body_fun,
  init_val)`, `lax.scan(f, init, xs)`, `lax.fori_loop(lo, hi, body, init)`. Each
  is a **higher-order primitive whose params contain sub-jaxprs** (traced
  branches/bodies), with hard structural constraints: both `cond` branches must
  return identical shapes/dtypes; the `scan`/`while` carry type is fixed across
  iterations. `while_loop` is forward-mode-differentiable only (no reverse
  without bounded trip count). Every transformation must then handle these
  primitives specially (a batched predicate turns `cond` into a `select`, etc.).
- **Side effects, prints, mutation of Python state** happen once, at trace time.

### Implication for shader/kernel-style code

Tracing is a natural fit for the *math* of a kernel, but a poor fit for the
*body style* of shader code, which is statement-heavy: `for` over neighbors,
early-exit branches, in-place accumulation. In JAX these must be rewritten into
`cond`/`scan` combinator form — precisely the "builder API / shadow language"
feel the desiderata reject ("user code looks like Python"). An AST/bytecode
frontend (numba-style) sees `if`/`for` directly and can lower them to real
structured IR without asking the user to change dialect. The lesson is **not**
"trace like JAX"; it is "steal JAX's interpreter/rule architecture and drive it
from an AST-derived IR instead of from operator overloading." Note that inside
a GPU kernel, `cond`-style predication vs. real branching is a backend decision
anyway — but the *user* should write `if`.

---

## 2. jaxpr: the IR, and precisely how closures are handled

### Shape of the IR

A jaxpr is flat, functional, first-order ANF — a let-chain of primitive
applications with explicitly typed binders
([jaxpr docs](https://docs.jax.dev/en/latest/jaxpr.html)):

```
jaxpr ::= { lambda Var* ; Var+ .        # constvars ; invars
            let Eqn*                    # eqns: Var+ = prim[params] Atom+
            in ( Expr+ ) }              # outvars
```

Concretely (autodidax's version; the real one adds effects, source info,
sharding):

```python
class Jaxpr(NamedTuple):
    in_binders: list[Var]        # each Var has an aval (ShapedArray)
    eqns:       list[JaxprEqn]   # JaxprEqn(primitive, inputs, params, out_binders)
    outs:       list[Atom]       # Var or Lit

class ClosedJaxpr(NamedTuple):
    jaxpr:  Jaxpr
    consts: list[Any]            # values for jaxpr.constvars
```

Properties worth copying: every binder is typed (`aval`), so **typechecking a
jaxpr is a ~50-line environment pass**; there is no scoping subtlety (flat
let-list, no shadowing); higher-order constructs (`cond`, `scan`, `jit` itself
as `pjit`/`jit_p`) carry sub-jaxprs in their **params**, so "the IR contains
functions" without the IR having lambda. Multiple transformations become
jaxpr→jaxpr functions (`jvp_jaxpr`, `vmap_jaxpr`, `transpose_jaxpr`).

### Constvars: how captured values are handled — baked, not parameterized

When tracing encounters a value that is not an argument — a Python-closure
capture, a global, a numpy array created at trace time — it does **not** become
a parameter. It becomes a **constvar**: hoisted to the front of the jaxpr, with
its concrete value stored in `ClosedJaxpr.consts` ("variables… introduced to
stand for constants that have been hoisted out"; scalars stay inline as `Lit`).
The compiled executable then either embeds the constant into the XLA binary or
holds it as a baked-in buffer. Either way:

> **The captured value is part of the compiled artifact, not an input to it.**

So JAX answers "how do I pass new capture values to old compiled code?" with:
*you don't — you retrace.* There is no env/uniform slot, no re-marshal path.
The pjit machinery even carries `const_args` through
`MeshExecutableFastpathData` — constants ride with the executable, not the call
([pjit.py](https://github.com/jax-ml/jax/blob/main/jax/_src/pjit.py)).

### How `jit`'s cache is keyed, and the retracing weakness in detail

The dispatch path (current `jax/_src/pjit.py`) has two cache tiers:

1. **C++ jit cache** (the fast path): `_cpp_pjit` wraps the user function with
   `_jax.pjit(...)` (C++), passing a `pxla.JitGlobalCppCacheKeys` (donate
   argnums, device/backend, shardings/layouts treedefs, compiler options). The
   C++ cache maps **(function object, static-arg values, arg treedefs, arg
   avals, cache keys)** → `MeshExecutableFastpathData` (the raw
   `xla_executable`, out treedef, shardings, `kept_var_bitvec`, …). Two global
   C++ caches of 8192 entries each (`_cpp_pjit_cache_fun_only`,
   `_cpp_pjit_cache_explicit_attributes`).
2. **Python tracing cache**: on C++ miss, `cache_miss` → `_infer_params` →
   `_infer_params_cached(fun, jit_info, arg_signature, avals, ctx_mesh)`, a
   `weakref_lru_cache` — the **function object is held by weak reference** and
   is the leading key component.

The documented failure mode
([slow-tracing debugging guide](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html),
[jit-compilation docs](https://docs.jax.dev/en/latest/jit-compilation.html)):

- "JAX indexes its in-memory tracing cache using the Python `id()` of the
  function object… functions allocated freshly on every call have changing ids,
  and JAX treats them as brand new functions and **retraces them perpetually**."
- "Avoid calling `jax.jit()` on temporary functions defined inside loops…
  because the cache relies on the hash of the function… each time the partial
  returns a function with a different hash"; "lambda will also return a
  function with a different hash." Their own benchmark: 379 ms (fresh
  lambda/partial per iteration) vs 1.59 ms (stable function) — **~240×**.
- The weakref means a dead lambda's cache entry is also *collected*, so the
  cache can't even accidentally hit later.
- Tooling exists purely to diagnose this: `jax.log_compiles(True)` and
  `JAX_EXPLAIN_CACHE_MISSES=1` ("TRACING CACHE MISS at my_script.py:65:8
  because id changed"), plus `config.no_tracing` which hard-errors on retrace.

**Contrast with the pdum.dsl thesis.** JAX's closure identity is
`(id(function object))` and captures are *values baked as constants* — numba's
mistake in different clothes. pdum.dsl's closure identity is `(code object
[value-compared], env_types)` and captures are *typed runtime data* marshaled
per call. In the driving usage pattern — a tight loop rebuilding a closure with
fresh values each iteration — JAX retraces every iteration by design (fresh
`id()`, and even with a stable id, changed consts would demand it); pdum.dsl
re-marshals a uniform buffer. JAX's escape hatches all shift the burden to the
user: hoist captures into arguments, keep the function object alive and stable,
or `functools.partial` + `static_argnums` (which recompiles per *value*). None
of them make "closure with fresh values" cheap. This is not an oversight JAX
can patch — retrace-on-new-identity is load-bearing because consts are baked —
it is an architectural commitment pdum.dsl declines.

---

## 3. THE HOOK MODEL: primitives × interpreters (the part to steal)

The deepest design fact about JAX: **a Primitive is just a name**. All meaning
lives in per-transformation registries. Verified current APIs
([extending guide](https://docs.jax.dev/en/latest/jax-primitives.html); the
public home of `Primitive` is now `jax.extend.core` — `jax.core.Primitive` was
deprecated in 0.4.x):

```python
from jax.extend import core
multiply_add_p = core.Primitive("multiply_add")     # a name + two flags
                                                    # (multiple_results, call_primitive)

# --- rule registrations, one per (primitive, transformation) pair ---

multiply_add_p.def_impl(ma_numpy_impl)              # eager eval: concrete values -> value

multiply_add_p.def_abstract_eval(ma_abstract)       # ShapedArray* -> ShapedArray
                                                    # (shape/dtype inference; used by every
                                                    #  staging transformation)

from jax.interpreters import mlir
mlir.register_lowering(multiply_add_p, ma_lowering, platform='cpu')
                                                    # per-platform codegen: emit StableHLO ops
                                                    # ('cpu' | 'cuda' | 'tpu' | None = default)

from jax.interpreters import ad
ad.primitive_jvps[multiply_add_p] = ma_value_and_jvp # (primals, tangents) ->
                                                     #   (primal_out, tangent_out)
ad.primitive_transposes[multiply_add_p] = ma_transpose
                                                    # cotangent-out -> cotangents-in
                                                    # (only for primitives linear in some args;
                                                    #  reverse mode = jvp + partial-eval + transpose)

from jax.interpreters import batching
batching.primitive_batchers[multiply_add_p] = ma_batch
                                                    # (batched_args, batch_dims) ->
                                                    #   (batched_result, out_batch_dim)
```

The registries are literally module-level dicts keyed by the primitive object
(`ad.primitive_jvps`, `ad.primitive_transposes`, `batching.primitive_batchers`,
`mlir._lowerings[platform]`), plus two methods on the primitive itself
(`def_impl`, `def_abstract_eval`). The matrix picture:

|                | `add_p` | `sin_p` | `cond_p` | *new op* |
|----------------|---------|---------|----------|----------|
| impl (eval)    | rule    | rule    | rule     | +1 rule  |
| abstract_eval  | rule    | rule    | rule     | +1 rule  |
| jvp            | rule    | rule    | rule     | +1 rule  |
| transpose      | rule    | —       | rule     | optional |
| batching       | rule    | rule    | rule     | +1 rule  |
| lowering (per platform) | rule | rule | rule  | +1/platform |
| ***new transformation*** | *one interpreter + one rule per primitive it meets* |

- **Adding an op** = one `Primitive` + a column of rules; ops missing a rule
  fail *only* when that transformation touches them (graceful partiality).
- **Adding a transformation** = one `Trace`/`Tracer` subclass (an interpreter
  over `bind`) + its own registry dict. `jvp`, `vmap`, `make_jaxpr`,
  partial-eval are all ~100–300-line instances of the same pattern.
- **Adding a backend** = one more `platform` in the lowering registry. Nothing
  else changes.
- Higher-order primitives are where the cost concentrates: `cond_p` needs each
  rule to recurse into its sub-jaxprs (`jvp_jaxpr` both branches and unify
  consts, batch the predicate into a `select`, …) — ~150 lines in autodidax vs
  ~5 per pointwise op.

### The escape hatches: `custom_jvp` / `custom_vjp`

For *user functions* (not new primitives) whose autodiff should be overridden —
numerical stability, implicit diff, calling out to opaque kernels
([custom-derivatives docs](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)):

```python
@jax.custom_jvp
def f(x): return jnp.sin(x)

@f.defjvp
def f_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    return f(x), jnp.cos(x) * t          # tangent_out must be LINEAR in t;
                                         # reverse mode is then derived by
                                         # transposing the linear part automatically

@jax.custom_vjp
def g(x): return jnp.sin(x)
def g_fwd(x): return g(x), jnp.cos(x)    # (primal_out, residuals) — user picks residuals
def g_bwd(res, ct): return (res * ct,)   # cotangents per input
g.defvjp(g_fwd, g_bwd)                   # but: forward-mode on g now raises TypeError
```

`nondiff_argnums=(0,)` handles callable/static params (they're passed first to
the rule). Mechanically these work by wrapping the function in a *call
primitive* (`custom_jvp_call_p`) whose jvp rule invokes the user rule — i.e.
the escape hatch is itself just the hook model applied once more. This matters
for pdum.dsl: a user-defined `Color.to_oklab()` device function with a
hand-written derivative should be *the same registration shape* as a built-in.

---

## 4. Pytrees: the logical-value → N-leaves marshaling layer

Every transformation boundary in JAX is monomorphized to flat lists by the
pytree registry ([pytree docs](https://docs.jax.dev/en/latest/pytrees.html)):

```python
leaves, treedef = jax.tree.flatten({'params': {'w': w, 'b': b}, 'lr': 0.1})
# leaves: [b, w, 0.1]   treedef: static structure, hashable, comparable
out = jax.tree.unflatten(treedef, new_leaves)
```

`jit`'s wrapper flattens args, passes `treedef` as part of the **cache key**
(structure is static, leaves are dynamic), runs the flat function, unflattens
outputs. `grad`/`vmap` do the same, which is why `in_axes` etc. accept
tree-prefixes. Custom types plug in:

```python
jax.tree_util.register_pytree_node(
    Color,
    lambda c: ((c.r, c.g, c.b), None),        # flatten -> (children, aux_data)
    lambda aux, kids: Color(*kids))           # unflatten(aux_data, children)

@jax.tree_util.register_pytree_node_class      # class carries tree_flatten/tree_unflatten
class Special: ...

jax.tree_util.register_dataclass(MyStruct, data_fields=[...], meta_fields=[...])
```

Two rules with teeth: `aux_data` must be hashable (it enters cache keys —
static/meta fields are *type-level*, data fields are *value-level*); and
flatten/unflatten must be cheap because they run **on every call**, not per
compile. Key paths (`SequenceKey`/`DictKey`/`GetAttrKey`,
`tree_flatten_with_path`) give leaf-addressed error messages.

**Mapping to pdum.dsl:** this is exactly the "one logical value → N physical
parameters" seam from the desiderata, with one refinement to adopt: JAX splits
a value into *(static structure → cache key)* + *(dynamic leaves → runtime
data)*. For pdum.dsl: `typeof(capture)` plays treedef (into the `FnType` key),
and a per-type `pack(value) -> physical params` plays leaves — an ndarray
yields (pointer, shape, strides); a future `Quantity` yields (magnitude,) with
the unit in the static half, which is precisely where compiler-level unit
conversion wants it. Pytrees prove one registry can serve every transformation
and every backend boundary; pdum.dsl's version needs one extra per-backend
stage pytrees don't have (leaves → uniform-buffer offsets / kernel args / C
ABI), since JAX delegates that to XLA.

---

## 5. Autodidax: what a minimal primitives+interpreters core costs

[Autodidax](https://docs.jax.dev/en/latest/autodidax.html) ("JAX core from
scratch") is the maintained pedagogical reimplementation. Measured from
`docs/autodidax.py` @ main, July 2026: **3048 total lines, 1671 lines of
actual code** (comments/prose stripped). Breakdown:

| Part | Contents | Code lines |
|---|---|---|
| 1. Interpreter core | `Primitive`, `bind`, `MainTrace` stack, `Trace`/`Tracer`, `ShapedArray`; **three interpreters**: `EvalTrace` (impl rules), `JVPTrace` (forward AD), `BatchTrace` (vmap) | **292** |
| — Pytrees + flattening | mini pytree registry, `flatten_fun`, generalizing jvp/vmap to trees | **193** |
| 2. Jaxprs | `Var`/`Lit`/`JaxprEqn`/`Jaxpr`, `JaxprTrace` + builder, `typecheck_jaxpr`, `eval_jaxpr`, `make_jaxpr` | **288** |
| 3. `jit` | `xla_call_p` as higher-order primitive, lowering jaxpr→MLIR/HLO, compile cache, jvp/vmap rules *for jit itself* (transformations commute with staging) | **276** |
| 4. `linearize`/`vjp`/`grad` | `PartialVal`, `PartialEvalTrace` (recipe DAG → jaxpr), `linearize = jvp ∘ partial-eval`, transpose rules, `eval_jaxpr_transposed`, `grad` | **439** |
| 5. Control flow | `cond_p`: staging both branches, const-unification, eval/jvp/vmap/abstract/lowering/transpose/partial-eval rules | **182** |

Readings of the numbers:

- The **entire hook architecture** — primitives, trace stack, two real
  transformations — is ~300 lines. This is the cheapest part, and the part
  with the highest leverage. A pdum.dsl kernel adopting the pattern spends
  well under its 1000-line budget on it.
- A **typed flat IR with builder, typechecker, and evaluator is ~290 lines.**
  Owning the IR is cheap at this scale; adopting xDSL should be justified by
  something other than "IRs are expensive."
- **Reverse-mode is the single most expensive component (~440 lines)** because
  it is not one thing: it is partial evaluation + linearization + transposition.
  Budget accordingly; forward-mode (jvp) is ~80 lines and should come first.
- **One structured-control-flow primitive costs ~180 lines** because every
  existing transformation must learn it. This is the recurring tax of
  higher-order primitives — the reason to keep their number tiny (JAX proper
  gets by on roughly: call/closed-call, cond, while, scan, custom-derivative
  calls).
- Missing from autodidax (so real-world costs are higher): kwargs, real error
  messages, `custom_jvp/vjp`, `while/scan`, donation, caching subtleties.

---

## 6. Dispatch cost: what keeps the hot loop hot

JAX's steady-state call path is instructive because it is the same problem as
pdum.dsl's phase-A/phase-B hot loop ([pjit.py](https://github.com/jax-ml/jax/blob/main/jax/_src/pjit.py)):

- The wrapper returned by `jax.jit` is a **C++ callable** (`_jax.pjit(...)`).
  On a hit it never enters author-level Python: C++ code flattens arguments
  (pytree treedefs are C++ objects — `PyTreeDef` lives in jaxlib), computes the
  signature (avals + treedef + static-arg hashes + `JitGlobalCppCacheKeys`),
  probes an 8192-entry C++ LRU, and invokes the cached
  `MeshExecutableFastpathData.xla_executable` directly, using precomputed
  `kept_var_bitvec`/layouts to marshal buffers.
- Only on a miss does it fall back to the Python `cache_miss` →
  `_infer_params_cached` (weakref-LRU tracing cache) → trace → lower
  (`_pjit_lower`, also memoized) → compile, then it installs fastpath data back
  into the C++ cache.
- Even so, JAX's per-call overhead is on the order of microseconds to tens of
  microseconds, and their guidance is architectural, not micro: **amortize
  dispatch by making the compiled region bigger** (jit the whole step function,
  not each op), and rely on async dispatch to hide the rest.

What this says for pdum.dsl, which wants to stay in pure Python:

1. JAX needs C++ because its *per-call key computation* is heavy: full pytree
   flatten + aval computation over arbitrarily nested arguments. pdum.dsl's
   phase-A key is deliberately lighter — `(code object, env fingerprint)` over
   a handful of captures — and the caching-layer doc's structural fingerprint
   memo is the pure-Python analogue of the C++ signature probe.
2. The winning structure is identical and worth copying: **one flat cache
   probe on the hot path, with everything precomputed at miss time** — the
   layout, the packer, the bound backend call — stored as a single "fastpath
   record" (pdum.dsl: `(compiled artifact, env layout, uniform packer)`). The
   hit path should be: fingerprint → dict get → pack values → launch; no
   re-derivation of layouts, no `typeof` lattice walk (M0's per-frame
   `flatten` re-lowering is exactly the anti-pattern this kills).
3. Measure the hit path in ns from day one; the floor for "a parameter write"
   through Python is a few dict ops + a `struct.pack_into`/numpy copy into a
   staging buffer. If that's ever insufficient, the JAX precedent says the fix
   is a narrow native fastpath *behind the same cache contract*, not a
   redesign.

---

## Design lessons for pdum.dsl

1. **Adopt the primitive/rule-registry matrix as the kernel's organizing
   principle.** An op is a name; every analysis/transformation/backend meaning
   is a rule in a registry keyed by `(op, aspect)` — `type_rule`, `eval_rule`,
   `jvp_rule`, `transpose_rule`, `batch_rule`, `lower[backend]`,
   `unit_rule`, …. This single structure simultaneously answers the desiderata's
   extensibility axes: new syntax = new ops + rules; new analysis = new aspect
   column; new backend = new `lower[·]` column. It also solves the M0 fault
   line "core imports the WGSL dialect tables": the dialect becomes rule
   registrations *into* the frontend, not an import *by* it. Autodidax proves
   the whole pattern costs ~300 lines.

2. **Do not adopt trace-based program acquisition.** Keep the AST (or bytecode)
   frontend so users write real `if`/`for` on runtime values — JAX's
   `cond`/`scan` combinator dialect is the shadow-language the aesthetics
   section forbids, and shader-style code is the worst case for it. The
   interpreter architecture is separable from tracing: pdum.dsl's "interpreters"
   walk its IR instead of intercepting Python execution, keeping the same
   rule-matrix extensibility with none of the trace blindness.

3. **JAX validates the thesis by negative example — and shows the diagnostics
   to build.** Its cache keys on function-object `id()` (held by weakref) and
   bakes captures into `ClosedJaxpr.consts`/the executable, so a rebuilt
   closure retraces perpetually (their docs: ~240× slowdown; dedicated tooling
   — `jax.log_compiles`, `JAX_EXPLAIN_CACHE_MISSES`, `no_tracing` — exists just
   to catch it). Two takeaways beyond vindication: (a) ship the equivalent
   observability from day one (`explain_cache_misses` naming the key component
   that differed — code edit vs env-type change vs backend param — plus a
   `no_compile` assertion mode for the render loop); (b) mirror the weakref
   discipline so the cache never pins dead templates (M0's L-cache leak).

4. **Make the IR flat, typed, and let higher-order ops carry sub-programs in
   params.** Jaxpr shows the sweet spot for a transformable IR: flat eqn list,
   every binder typed, ~290 lines including typechecker and evaluator —
   and `cond`/`scan`/device-fn-call as ops whose params hold sub-IRs, so
   transformations recurse structurally. But diverge on binding: where jaxpr
   has `constvars + consts` (captures as baked values), pdum.dsl's IR needs
   `envvars` typed by `env_types` — captures as *parameters with a marshaling
   annotation*, never values. That one substitution is the whole thesis at the
   IR level. Budget higher-order ops carefully: each one costs a rule from
   every existing transformation (~180 lines for `cond` in autodidax), so keep
   the set minimal (call, if, loop).

5. **Generalize pytrees into the typed marshaling seam.** Split every logical
   value into a static half (structure/type/unit → part of the cache key,
   hashable) and a dynamic half (ordered physical leaves → runtime data), with
   a user-facing registry (`register_pytree_node`-shaped: `flatten -> (leaves,
   static)`, `unflatten`) so user structs (`Color`), records, and future
   `Quantity` types plug into capture, caching, and transformations uniformly.
   Then add the stage JAX doesn't have: a per-backend layout planner mapping
   ordered leaves → uniform-buffer offsets / kernel args / C ABI slots,
   computed once per `FnType` and cached with the artifact. Units slot in as
   static-half data with a conversion inserted at pack time — exactly the
   desiderata's §4.4 hook.

6. **Sequence transformations the autodidax way, and design the IR for them
   now.** Cost ordering measured: abstract-eval/type rules (needed anyway) →
   vmap (~80 lines + a rule per op) → jvp (~80 lines) → reverse mode as
   jvp + partial-eval + transpose (~440 lines; by far the biggest, ship last).
   Requirements this imposes on day-one IR design even though AD ships later:
   multiple results per op, no hidden state, types on every binder, and
   transposability metadata (which args an op is linear in). Provide
   `custom_jvp`/`custom_vjp`-shaped escape hatches so a user device function
   with a hand-written derivative registers rules exactly like a built-in op —
   that's also the batteries answer: a "battery" is an op package (type rule +
   portable lowering in terms of core ops + optional per-backend fast path +
   optional derivative), addable without touching the kernel.

7. **Copy the two-tier dispatch shape, in Python.** Hot path = structural
   fingerprint → one dict probe → prebuilt fastpath record `(artifact, env
   layout, packer, launcher)` → pack + launch; everything else happens only at
   miss time and is stored precomputed. JAX needed a C++ fast path because its
   per-call signature is heavy (full pytree flatten + avals); pdum.dsl's key is
   a code object + a few capture fingerprints, so pure Python can hit the
   "parameter write" floor — but only if nothing on the hit path re-derives
   structure (M0's per-frame `flatten` violates this today). If a native
   fastpath is ever needed, JAX shows it can be added behind the same cache
   contract without redesign.

---

### Sources

- Autodidax (measured @ main, July 2026): <https://docs.jax.dev/en/latest/autodidax.html>, <https://github.com/jax-ml/jax/blob/main/docs/autodidax.py>
- jaxpr IR: <https://docs.jax.dev/en/latest/jaxpr.html>
- Defining primitives / rule registries: <https://docs.jax.dev/en/latest/jax-primitives.html>
- jit caching & retracing: <https://docs.jax.dev/en/latest/jit-compilation.html>, <https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html>
- Dispatch internals: <https://github.com/jax-ml/jax/blob/main/jax/_src/pjit.py>
- Control flow: <https://docs.jax.dev/en/latest/control-flow.html>
- Pytrees: <https://docs.jax.dev/en/latest/pytrees.html>
- Custom derivatives: <https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>
- Version status: <https://github.com/jax-ml/jax/releases>, <https://docs.jax.dev/en/latest/changelog.html>
