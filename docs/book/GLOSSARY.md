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
| **ValueKind** | One registration per Python type yielding the views of a value: `typeof` + `fingerprint` now; `leaf_types` + `flatten` in ch07. | ch01 |
| **TemplateId** | Code identity as a sum type: `Base` (a code object, value-compared) or `Derived` (transform-minted, e.g. `grad(f)`). | ch01 |
| **FnType** | The structural type of a DSL closure: `(TemplateId, env_types)`. The thesis in one value. | ch01 |
| **Literal lift** | The one explicit value-in-type opt-in (`LiteralType`): the value enters the cache key and compiles as a constant. | ch01 |
| **Handle** | The phase-A product: `FnType` + captured values + env fingerprints + source snapshot. A capture that is itself a Handle contributes its `FnType` to the parent — composition is structural. | ch02 |
| **SourceSnapshot** | Decoration-time source text (+ location), memoized per code object; the eager half of the stale-source defense. | ch02 |
| **thesis cache / artifact cache** *(forthcoming)* | Tier 1: `(template, env types, arg types, backend, generation) → FastRecord`. Tier 2: `(IR content hash, backend, flags) → artifact`. | ch03 |
| **generation** *(forthcoming)* | A registry counter folded into every thesis key; the coarse invalidation knob. | ch03 |
| **guard** *(forthcoming)* | A precomputed identity check on the hit path catching dependency drift (rebound globals); refuse-or-recompile, never stale. | ch03/ch08 |
| **Node / Region** *(forthcoming)* | The entire IR: one frozen node type; structured control flow as regions (`if`/`for`/`call` only). No field can hold a capture value. | ch04 |
| **rule / aspect / rule matrix** *(forthcoming)* | Per-op meaning lives in `(op, aspect)` registrations (`lower_ast`, `jvp`, …); passes are drivers over the matrix. | ch05 |
| **leaf / slot / PackPlan** *(forthcoming)* | Logical value → ordered leaves → physical slots, planned once per cache entry from types alone. | ch07 |
| **FastRecord** *(forthcoming)* | The compiled cache entry: artifact + guards + extractor + plan + launcher. The hot path touches nothing else. | ch08 |
| **surface** *(forthcoming)* | One of the five registration doors (ops/rules, overloads, type extensions, backends, transforms). "Zero kernel diffs" is CI-enforced. | ch10 |
