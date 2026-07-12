# Glossary

The project's working vocabulary, one entry per term, accumulated chapter by
chapter. Walkthrough feedback edits this file directly — if a name here feels
wrong, changing it *now* is cheap and changing it after five more chapters is
not. Terms marked *(forthcoming)* are defined by the architecture but not yet
implemented; their entries firm up when their chapter lands.

| Term | Meaning | Introduced |
|---|---|---|
| **the thesis** | Compiled artifacts are cached on the **types** of a closure's captures and arguments, never their values — so a hot loop that rebuilds closures with fresh values never recompiles. | ch00 |
| **phase A** | Capture, at decoration time: read the code object and closure cells, summarize capture types. Compile-free, cannot fail. | ch00 |
| **phase B** | Call time: on a cache miss, lower/compile once per type signature; on a hit, marshal values and launch. | ch00 |
| **Type** | A frozen, hashable, **structural summary** of a value — not its Python class. The sole vocabulary of cache identity. | ch01 |
| **summary function** | The view that `typeof` *chooses* what a type records (rank-only vs shape-in-type vs value buckets). Int range-bucketing is the built-in example. | ch01 |
| **typeof** | Value → `Type`, per registered kind. Loud on unregistered types. | ch01 |
| **Tuple vs Vec** | `typeof` summarizes Python tuples as `Tuple` (honest, element-wise, arity in the identity). `Vec` is IR-level only, produced by dialect lowering rules; tuple-as-vec packing is the backend's PackPlan decision. | ch01 |
| **fingerprint** | A cheap structural tag for the hot path, bound by the **soundness law**: equal fingerprints ⇒ equal `typeof` outcome. Enforced by a CI fuzz. | ch01 |
| **ValueKind** | One registration per Python type yielding the views of a value: `typeof` + `fingerprint` now; `leaf_types` + `flatten` in ch08. | ch01 |
| **TemplateId** | Code identity as a sum type: `Base` (a code object, value-compared) or `Derived` (transform-minted, e.g. `grad(f)`). | ch01 |
| **FnType** | The structural type of a DSL closure: `(TemplateId, env_types)`. The thesis in one value. | ch01 |
| **Literal lift** | The one explicit value-in-type opt-in (`LiteralType`): the value enters the cache key and compiles as a constant. | ch01 |
| **Handle** | The phase-A product: `FnType` + captured values + env fingerprints + source snapshot. A capture that is itself a Handle contributes its `FnType` to the parent — composition is structural. | ch02 |
| **SourceSnapshot** | Decoration-time source text (+ location), memoized per code object; the eager half of the stale-source defense. | ch02 |
| **specialization cache / artifact cache** | Tier 1: `(fp_head, arg_fp, backend_fp, generation) → FastRecord`, with LRU + retirement — the standard compiler concept (Julia's MethodInstance table, numba's overloads) done structurally; it is the thesis made mechanical. Tier 2: `(content_key, backend, flags) → artifact` — a content-addressed compilation cache (the ccache/tinygrad shape), generation-free. | ch03 |
| **generation** | A counter in every specialization key; `bump_generation()` clears tier 1 wholesale (tier 2 survives — content-addressed). | ch03 |
| **guard** | A precomputed `(holder, name, expected)` identity check catching dependency drift (rebound globals); refuse-or-recompile, never stale. Synthetic until classify_names (ch07) supplies real tags. | ch03 |
| **Node / Region** | The entire IR: one frozen node type (op, type, args, regions, attrs, loc); regions for structured control flow (`if`/`for`/`call` only, constitutionally). No field can hold a capture value — `attrs` is the const/Literal carve-out; `loc` is excluded from identity. | ch05 |
| **content key** | `Node.key`/`Region.key`: memoized sha256 over structure — the artifact-tier cache key. In-process; the disk cache re-keys structurally later. | ch05 |
| **rule / aspect / rule matrix** *(forthcoming)* | Per-op meaning lives in `(op, aspect)` registrations (`lower_ast`, `jvp`, …); passes are drivers over the matrix. | ch06 |
| **leaf / slot / PackPlan** *(forthcoming)* | Logical value → ordered leaves → physical slots, planned once per cache entry from types alone. | ch08 |
| **FastRecord** | The tier-1 entry: artifact + guards now; extractor/plan (ch08) and staging/launcher (ch09) complete the precompiled hit path. | ch03/ch09 |
| **Role** | What a stage *is* for composition: `Handle.kind`, grown into a registered value. Vocabularies ship with their owning packages — even `device` (the base language's neutral composable, `@jit`'s default) belongs to the stdlib/core-dialect package; the combinator library owns only `materializer`. Terminality is structural (`Role.terminal`), not a pair rule. | ch04 |
| **composition rule** | `(op, role, role) → semantics` — *how* two stages compose (`fuse`, `terminal`, later `orchestrate`), or a loud `IncompatibleRoles`. | ch04 |
| **Stage / Pipeline** | A configured kernel (`@op` constructor, `stage[config]`) and the inert, flattened, `Derived("pipe")`-identified composition of stages. Definition ≠ application (`>`). | ch04 |
| **materializer** | A terminal stage marking the device→host boundary (plumbum's materializers, transplanted); executes at ch09. | ch04 |
| **surface** *(forthcoming)* | One of the five registration doors (ops/rules, overloads, type extensions, backends, transforms). "Zero kernel diffs" is CI-enforced. | ch11 |
