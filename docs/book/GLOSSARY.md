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
| **strict core** | Core arith/cmp demand same-type operands; every conversion is an explicit `core.cast` in the IR. Promotion is a dialect's lowering policy (or absent), never kernel law — the Julia/MLIR model. | ch05 |
| **provenance** | `Node.loc`: `Loc`, `CallLoc` (inlining chains), or `FusedLoc` (merges) — never in identity or content keys. Fresh rewrite replacements inherit the replaced node's loc; survivors keep their own. Contract: a starting region, not DWARF (050). | ch06 |
| **rule / RuleSet / Stage** | A rule is `(Pat, fn)` data; the one greedy driver runs rule sets bottom-up — deterministic, DAG-preserving, budget-guarded against non-termination. A `Stage` adds conversion-target legality (`legal` = which op *namespaces* may remain, checked after fixpoint) and `forbid` (which op *names* must be gone — a namespace target can't say "`core.env` is eliminated", since `core.env` **is** `core`). | ch06/ch08 |
| **name fates** | The closed taxonomy lowering assigns every name: param, local, capture (value → `core.env` path / Handle → inlined callee). Anything else is a loud `NameFateError` — globals have no sanctioned fate yet. | ch07 |
| **lower_ast rule / build rule** | The language as data: `ast` node type → lowering fn, in satellite packs (base pack is STRICT — no auto-casts). `Derived` templates lower via build rules (pipe today; grad/vmap later) — structure, not source. | ch07 |
| **aspect / rule matrix** *(forthcoming)* | Per-op meaning as `(op, aspect)` registrations (`lower_ast`, `jvp`, …); passes are drivers over the matrix. | ch07+ |
| **leaf** | The closed marshaling vocabulary backends must be total over: `ScalarLeaf` (byte-packable), `BufferLeaf`/`ShapeLeaf`/`StrideLeaf` (with ndarrays). The architecture's `EnvLeaf` is the `FnType` walker: a captured kernel's leaves are its env's leaves, paths prefixed — the same paths lowering stamps on `core.env`. | ch08 |
| **alignment law** | `flatten(v)` yields exactly one value per entry of `leaf_entries(typeof(v))`, in order — fuzz-enforced like soundness; drift would corrupt packed bytes silently. | ch08 |
| **SlotSpec / PackPlan** | Leaf → `SlotSpec(source, convert, dest)`; a `PackPlan` is the ordered slots + staging size, built once per cache entry **from types alone**. `dest` vocabulary is backend-owned (`PackedDest` is the reference dense layout); `convert` is the units seat (step 15). | ch08 |
| **staging / leaves channel** | The two roads at call time: byte slots pack into a reused staging buffer; buffer-class leaves travel untouched to the launcher (`launch(staging, leaves)`). | ch08 |
| **abi.slot / legalize_params** | The marshaling decision as IR: `NORMALIZE_ENV` folds extract/field-of-env into env *paths*, then every `core.env` legalizes to a physical `abi.slot {src, offset, fmt}` under legality set `{core, abi}` — a surviving logical capture is a `VerifyError`, so per-frame flatten is structurally impossible. | ch08 |
| **ResultPlan** | The output mirror (DPS, 040 §3b): kernels write destinations, not returns; destinations allocate from the result type and result bytes unflatten back into the logical value. Bidirectional from the start. | ch08 |
| **FastRecord** | The tier-1 entry, complete: artifact (the exec'd/compiled function, carrying its listing) + guards (identity triples per captured cell) + compiled extract + plan + reused staging + launch. The hit path runs these and nothing else. | ch03/ch09 |
| **Registry** | Surface E, v1: ONE explicit object owning the kind table, lowering rule packs, Derived build rules, backends, and both cache tiers. Satellites register into `DEFAULT` at import; the kernel registers nothing. Layering/overloads complete it at step 10. | ch09 |
| **Backend** | A capability record: `name`, `render(region, plan)`, `compile(source)`, `fp` (in every specialization key). The §2.10 columns (`type_map`, `code_for_op` gating decompositions, `params_key`) arrive with the WGSL backend. | ch09 |
| **dispatch / batteries** | `Handle.__call__` → `DEFAULT.dispatch`: fingerprint args → key → tier-1 probe (guards inline) → extract → pack → launch. `import pdum.dsl` wires the base dialect + Python backend (batteries included); bare kernel imports stay registration-free and loud. | ch09 |
| **Role** | What a stage *is* for composition: `Handle.kind`, grown into a registered value. Vocabularies ship with their owning packages — even `device` (the base language's neutral composable, `@jit`'s default) belongs to the stdlib/core-dialect package; the combinator library owns only `materializer`. Terminality is structural (`Role.terminal`), not a pair rule. | ch04 |
| **composition rule** | `(op, role, role) → semantics` — *how* two stages compose (`fuse`, `terminal`, later `orchestrate`), or a loud `IncompatibleRoles`. | ch04 |
| **Stage / Pipeline** | A configured kernel (`@op` constructor, `stage[config]`) and the inert, flattened, `Derived("pipe")`-identified composition of stages. Definition ≠ application (`>`). | ch04 |
| **materializer** | A terminal stage marking the device→host boundary (plumbum's materializers, transplanted); executes at ch09. | ch04 |
| **config (bracket contract)** | `unit[...]`: an opaque hashable payload. Its own specialization regime (NOT the capture thesis): per component, strip → value-specialize (default) → type-specialize (rare); modes declared by the role/backend schema, overridable per kernel. Today: no schemas, so everything value-specializes. | ch04 |
| **surface** *(forthcoming)* | One of the five registration doors (ops/rules, overloads, type extensions, backends, transforms). "Zero kernel diffs" is CI-enforced. | ch11 |
