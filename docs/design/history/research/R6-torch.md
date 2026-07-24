# R6 — torch.compile as prior art: the industrial bytecode frontend and guard-based specialization

*Research memo for the pdum.dsl redesign. Verified against live sources July 2026.
Current stable: **PyTorch 2.13.0** (releases: 2.10.0 → Jan 2026, 2.11, 2.12.x, 2.13.0 per
[github.com/pytorch/pytorch/releases](https://github.com/pytorch/pytorch/releases)). Python 3.14
support for `torch.compile` landed in 2.10 (Jan 2026).*

torch.compile is a four-layer stack:

| Layer | Job | Where |
|---|---|---|
| **TorchDynamo** | capture: CPython bytecode → FX graph + guards + rewritten bytecode | `torch/_dynamo/`, `torch/csrc/dynamo/` |
| **AOTAutograd** | trace joint forward+backward, functionalize, decompose ATen | `torch/_functorch/aot_autograd.py` |
| **FX** | the graph IR everything above produces/consumes | `torch/fx/` |
| **TorchInductor** | FX(ATen) → define-by-run loop IR → Triton / C++-OpenMP | `torch/_inductor/` |

We care about it because it is the **largest deployed system that caches compiled artifacts
against live Python closures/frames** — the same problem our type-keyed cache solves — but it
solves it with *value/identity guards* instead of *type keys*. Sections 1–2 are the meat; 3–5
are briefer.

---

## 1. TorchDynamo: the bytecode frontend

### 1.1 Frame interception (PEP 523)

`@torch.compile` installs a custom frame-evaluation function via CPython's PEP 523 API
(`_PyInterpreterState_SetEvalFrameFunc`, wrapped in
[`torch/csrc/dynamo/eval_frame.c`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/dynamo/eval_frame.c),
866 lines of C). Every Python frame executed under the compile region is intercepted; the C shim
packs the frame (bytecode, f_locals, f_globals, builtins) and hands it to Python-side
`torch/_dynamo/convert_frame.py::_convert_frame_assert` (~102 KB module). Compiled results are
stashed **on the code object itself** via `_PyCode_SetExtra` (the PEP 523 scratch slot) as an
`ExtraState` — see §2.3.

Filename exclusion lists skip stdlib/numpy frames; frames that failed analysis are marked
skip-forever; frames over the recompile limit fall back to eager (ASPLOS'24 paper §3.2,
[pytorch2-2.pdf](https://docs.pytorch.org/assets/pytorch2-2.pdf)).

### 1.2 Symbolic bytecode interpretation

Dynamo **reimplements the CPython stack machine as an abstract interpreter**:
`InstructionTranslatorBase` in
[`torch/_dynamo/symbolic_convert.py`](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/symbolic_convert.py)
(267 KB — one handler method per bytecode opcode, ~200 handlers). State tracked: symbolic stack,
symbolic locals, exception contexts, the accumulating FX graph (`OutputGraph`, 211 KB), the
accumulating guard set, and a side-effect log.

Every runtime value is wrapped in a `VariableTracker` subclass
(`torch/_dynamo/variables/`, with `VariableBuilder._wrap` in `builder.py`, 227 KB, doing recursive
type dispatch): `TensorVariable` (holds an `fx.Proxy` into the graph + a FakeTensor for metadata —
never the real data), `ConstantVariable`, `ListVariable`/`ConstDictVariable`,
`UserFunctionVariable` (inlinable), `UserDefinedObjectVariable` (catch-all, lazily specialized
attribute-by-attribute). Each `VariableTracker` carries (a) the guards its existence depends on,
propagated by union through every operation, and (b) its `Source` — how to re-load it from
f_locals/globals at runtime (§2.1).

Key mechanics worth stealing context from:

- **Inlining with checkpoint/rollback.** On a call, Dynamo snapshots symbolic state, recursively
  traces the callee; if the callee would graph-break, it rolls back and breaks at the call site
  instead.
- **Control flow is specialized away.** Loops over Python lists are unrolled under a
  "list didn't change" guard; branches on tensor *metadata* are resolved and guarded; branches on
  tensor *data* force a graph break.
- **Closures/cells get four distinct treatments** (paper §3.6): cells created+destroyed inside the
  traced region are optimized away; pre-existing cells are read via generated
  `fn.__closure__[0].cell_contents` accesses (and guarded — `CLOSURE_MATCH`); escaping cells are
  reconstructed in output bytecode so callers can't tell the difference.
- **Side effects are deferred**: mutations are logged, applied by generated epilogue bytecode
  after the compiled graph runs; non-escaping mutations are dead-code-eliminated.

### 1.3 Graph breaks and continuation functions

On an unsupported construct, Dynamo compiles the partial graph, emits bytecode that (1) calls the
compiled artifact, (2) rebuilds the live Python stack via each `VariableTracker.reconstruct()`,
(3) runs the unsupported code natively, (4) calls a generated **resume function**
`resume_at_X(...livevars...)` whose prologue restores stack/try-block state and jumps to offset X
(`torch/_dynamo/resume_execution.py`, 37 KB). Resume functions are themselves intercepted and
compiled — so one Python function becomes a chain of compiled fragments stitched by rewritten
bytecode. ASPLOS'24 measured how well this works: on TorchBench, 70% of models capture with 0
graph breaks, 8% with 1–9, 22% with 10+; mean 21.1 graphs/model, 252.8 ops/graph.

### 1.4 The CPython-version treadmill (churn evidence)

This is the documented cost of a bytecode-level frontend:

- `torch/_dynamo/bytecode_transformation.py` alone (2,088 lines) contains **71
  `sys.version_info` branches** — separate code paths for 3.11/3.12/3.13/3.14 instruction
  encodings, `EXTENDED_ARG`, exception tables, `CACHE` slots, jump semantics.
- **Python 3.11** was the hardest port ("major changes to frame evaluation and bytecode semantics
  as part of the Faster CPython effort"); `torch.compile` was unsupported on 3.11 until PyTorch 2.1
  (~a year after 3.11 shipped).
- **Python 3.12** ([dev-discuss writeup](https://dev-discuss.pytorch.org/t/supporting-dynamo-in-python-3-12/2320)):
  frame ownership moved caller→callee, forcing Dynamo to *copy-paste CPython's shadow-frame
  allocation logic* and switch to `_PyObject_VirtualAlloc/Free`; bytecode reordering broke their
  block-stack heuristic; a `LOAD_SUPER_ATTR` flag-bit bug. The author's own assessment: CPython's
  requirements "are not documented well", bytecode layout "even more poorly documented than frame
  evaluation", and their with-block detection "is brittle and could break in future Python
  version updates". Support shipped in 2.4, ~9 months after 3.12.
- **Python 3.13** shipped in 2.6 (~3.5 months lag); **3.14**
  ([completion post](https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-14-completed/3276),
  Dec 2025) needed "moderate changes to the C/C++ parts of Dynamo" for CPython's new
  `PyStackRef`s, plus `__trunc__`-removal fallout; shipped in 2.10, again ~3.5 months lag.
  Free-threaded builds (3.13t/3.14t) still only have "basic support".

The steady state is: a dedicated engineer (williamwen42) lands a multi-month PR series per CPython
release, and no new Python version is usable with `torch.compile` for a quarter to a year. The
frontend surface that must track CPython (symbolic_convert + bytecode_transformation +
eval_frame.c + resume_execution) is roughly **350 KB of source**. This is the price of tracing
*arbitrary* Python including other libraries' frames. A DSL that owns its (small) input language
and reads **source/AST** instead pays none of this: the AST of the supported subset is far more
stable across CPython versions than bytecode (3.9→3.14 changed `ast` for our purposes almost not
at all, while every single release changed bytecode).

---

## 2. GUARDS — value-checked caching (the direct counterpart to our type keys)

### 2.1 What a guard is

A `Guard` (dataclass in `torch/_dynamo/guards.py`, 5,662 lines) is essentially
`(originating_source: Source, create_fn: GuardBuilder method)`. A **`Source`** says how to reach
the value from the frame at check time — `LocalSource("x")`, `GlobalSource("W")`,
`GetItemSource(...)`, `AttrSource(...)`, `ClosureSource`, chained arbitrarily deep
([`torch/_dynamo/source.py`](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/source.py):
1,419 lines, **58 Source classes**). Guards accumulate during tracing: every attribute read,
every dict lookup, every `isinstance` the interpreter resolved becomes a guard on the thing it
looked at.

`GuardBuilder` (same file) has ~48 guard kinds. Representative set:

| Guard | Checks |
|---|---|
| `TYPE_MATCH` | `type(x) is T` via cached `check_type_id` (C) |
| `ID_MATCH` | `x is <the exact object>` (used for functions, nn.Modules by default) |
| `EQUALS_MATCH` / `CONSTANT_MATCH` | value equality for specialized constants (ints, strings, ...) |
| `TENSOR_MATCH` | dtype, device, requires_grad, dispatch keys, ndim, and (static mode) sizes+strides |
| `SHAPE_ENV` | the accumulated symbolic-shape predicates, e.g. `2 <= L['a'].size()[0]`, `L['b'].size()[0] == L['a'].size()[0]` |
| `SEQUENCE_LENGTH`, `DICT_KEYS_MATCH`, `DICT_VERSION`, `DICT_CONTAINS` | container structure |
| `CLOSURE_MATCH`, `FUNCTION_MATCH`, `BUILTIN_MATCH`, `CLASS_MATCH`, `MODULE_MATCH` | code/function identity |
| `NN_MODULE`, `EMPTY_NN_MODULE_HOOKS_DICT` | module identity + "no hooks appeared" |
| `GRAD_MODE`, `DETERMINISTIC_ALGORITHMS`, `TORCH_FUNCTION_STATE`, `DEFAULT_DEVICE`, `GLOBAL_STATE` | ambient interpreter state |
| `DUPLICATE_INPUT`, `install_no_tensor_aliasing_guard` | relational: aliasing between inputs |
| `WEAKREF_ALIVE`, `HASATTR`, `NOT_PRESENT_IN_GENERIC_DICT` | liveness / shape of object dicts |

The ASPLOS paper (written at ~30 guard types) notes guards are emitted by *every* layer — Dynamo,
AOTAutograd, Inductor can all "introduce guards to protect specializations". Guards are
independent predicates; the only cross-guard structure is deduplication and the relational
aliasing guards.

### 2.2 How guards are evaluated per call: the C++ GuardManager tree

`CheckFunctionManager` compiles the guard list into a **C++ tree** (`RootGuardManager` /
`GuardManager` / `LeafGuard` / `GuardAccessor`, in
[`torch/csrc/dynamo/guards.cpp`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/dynamo/guards.cpp),
301 KB). Design (from the in-source comment): one `GuardManager` node per *object*; edges are
**accessors** (`GetAttrGuardAccessor`, `DictGetItemGuardAccessor`, `ClosureGuardAccessor`,
`FuncDefaultsGuardAccessor`, `TupleGetItemGuardAccessor`, ...); each node holds its leaf guards
(`check_type_id`, equality, tensor property checks — all in C++, no Python frames on the hot
path). So checking `L['self'].layer.weight.dtype` walks the accessor chain once, and all guards
on one object share the fetch. Two runtime optimizations:

- **fail-fast reordering**: children are re-sorted by observed failure count, so historically
  discriminating guards run first;
- a **`diff_guard_root`** subset tree (only guards that have ever differed between cache entries)
  used under `skip_guard_eval_unsafe` stance.

### 2.3 The cache proper

Per code object (`extra_state.h/.cpp`): `ExtraState` holds a `std::list<CacheEntry>`;
each `CacheEntry` = `{guard_manager, root_mgr (C++ tree), diff_guard_root_mgr, code (the rewritten
bytecode), backend, trace_annotation}`. Lookup (`extra_state.cpp::lookup_in_list`) is a **linear
scan**:

```cpp
for (CacheEntry& e : entries) {
    if (backend_match(e.backend, backend)
        && run_root_guard_manager(e.root_mgr, f_locals))   // full guard tree eval
        return &e;                                          // + LRU move-to-front
}
return nullptr;   // -> retrace & compile a new entry
```

So the per-call cost is **O(entries × guards-per-entry)** tree evaluation — *not* a hash lookup.
Limits: `torch._dynamo.config.recompile_limit = 8` entries per code object,
`accumulated_recompile_limit = 256` per program; on overflow the frame is permanently skipped
(runs eager). The escape hatches PyTorch had to add are themselves evidence of the cost:
`torch.compiler.set_stance("skip_guard_eval_unsafe")` ("These guards, though efficient, add some
overhead" — [set_stance docs](https://docs.pytorch.org/docs/2.13/generated/torch.compiler.set_stance.html)),
`eager_on_recompile`, `fail_on_recompile`, `error_on_graph_break` (new in 2.9), plus a
`profile_guard_manager` API and a "Dynamo cache lookup" bucket in the profiler because guard time
on parameter-heavy models (thousands of tensor guards) is a real production line item — the whole
C++ GuardManager rewrite (default since ~2.4) existed to claw back what had been Python-level
`eval` of a giant conjunctive expression.

### 2.4 Dynamic shapes: guards doing the type system's job

Shapes start **static**: first compile specializes on exact sizes (a `TENSOR_MATCH` on
sizes/strides). On the first size mismatch, "automatic dynamic" recompiles with that dimension as
a `SymInt`; 0/1-sized dims always stay specialized; equal runtime sizes are unified by "duck
shaping" plus an equality guard. Every branch the traced code takes on a symbolic size adds a
`SHAPE_ENV` predicate (e.g. `2*s0 >= 16`). This is a *discovered, per-trace refinement type*,
learned by running and guarded by predicates — the exact information our design puts in the
declared type key up front (dtype + rank in the key; sizes as runtime data).

### 2.5 The trade: value guards vs. type keys

| | Dynamo guards | pdum.dsl type keys |
|---|---|---|
| Key derivation | *discovered during tracing*: everything the trace touched gets a predicate | *computed a priori*: `typeof` over captures+args, fixed arity |
| Lookup | linear scan of entries, each = full predicate-tree eval over live objects | one hash of a small key tuple → dict hit |
| Per-call cost | O(entries × guards); thousands of checks on big models; needed a C++ rewrite, fail-count sorting, LRU, and unsafe-skip knobs | O(#captures + #args) `typeof` + hash; constant, small, no knobs needed |
| Can specialize on values? | yes, automatically (constants, shapes, dict keys, module identity) — and *over*-specializes, hence recompile limits and automatic-dynamic heuristics | only by explicit opt-in (`Literal`/`Val`-style lifting) |
| Soundness domain | arbitrary Python: mutable globals, monkey-patching, nn.Module attribute churn, aliasing — guards re-verify the world every call | a closed DSL subset: soundness = correctness of `typeof` + code-object identity + explicit invalidation (generation counter) |
| Failure mode | silent recompile storms (mitigated by `TORCH_LOGS=recompiles`, `fail_on_recompile`) | silent *reuse* if `typeof` is too coarse — our hazard analysis (dsl_caching_layer.md) already owns this |

The deep reason Dynamo needs guards: **it never sees a type signature**. Its "types" are whatever
facts tracing happened to consult, so the cache key is an open-ended conjunction of observations,
and the only way to check membership is to re-observe. Our design inverts this: because the DSL
declares its value universe, the key is a total function of the inputs, checkable without looking
at the compiled artifact at all. Dynamo pays per-call to buy generality over all of Python; we
pay a language-subset restriction to buy an O(1) hot path. Both systems agree on the essentials
we already believe: key on *function identity* (they guard `__code__`/`ID_MATCH`; we use
code-object value equality), guard/represent *closure contents* explicitly (`CLOSURE_MATCH`,
`FuncClosureSource` ↔ our env_types), and treat *ambient state* as part of the key
(`GRAD_MODE`/`DEFAULT_DEVICE` ↔ our backend-parameters-in-key lesson from M0).

Two Dynamo ideas worth importing even into a type-keyed world: (1) **relational guards**
(aliasing between two inputs) — a pure type key cannot express "these two array captures share
storage"; if a backend ever cares (e.g. WebGPU binding aliasing rules), we need either a
normalization step at marshal time or an explicit relational component in the key. (2) The
**diff-guard** idea: when several cache entries exist for one code object, only check the
components that actually differ — for us, if multiple type keys share a template, comparing keys
already does this for free; it's an argument for keeping the key a flat tuple of small
components.

---

## 3. FX: the graph IR

([docs](https://docs.pytorch.org/docs/2.13/fx.html); design paper arXiv:2112.08429.)

- **`Graph`** = ordered doubly-linked list of **`Node`**s. Node fields: `op` (one of exactly six:
  `placeholder`, `get_attr`, `call_function`, `call_method`, `call_module`, `output`), `target`,
  `args`, `kwargs`, `name`, plus a free-form `node.meta` dict (where shape-prop/FakeTensor
  metadata lives).
- **`GraphModule`** = an `nn.Module` holding a `Graph`; `recompile()` **generates Python source**
  for `forward()` from the graph (`traced.code` is readable, debuggable with pdb). The IR
  round-trips through Python — that's the whole serialization story.
- **`Proxy`** = the tracing value: operator overloads record nodes. `symbolic_trace` feeds Proxies
  through `forward()`. Dynamo does *not* use `symbolic_trace` — it drives `fx.Proxy` from the
  bytecode interpreter — but produces the same Graph type.
- **Transforms**: iterate `graph.nodes` and mutate; or `fx.Interpreter` (re-execute node-by-node,
  override per-op methods — this is how ShapeProp and many analyses are written); or
  `fx.Transformer` (Interpreter that emits a new graph); or `replace_pattern` (subgraph
  find/replace). `graph.lint()` validates.
- **Limits** (documented): no data-dependent control flow (that's Dynamo's job — FX itself is a
  single straight-line dataflow graph); `len()`/builtins need `fx.wrap`; specialization via
  `concrete_args`; customization via `Tracer.is_leaf_module`. The flat six-opcode design is why
  every consumer of FX is easy to write — and why control flow had to live *outside* the IR (as
  multiple graphs + resume functions, or as `torch.cond` higher-order ops bolted on later).

The FX lesson is about **IR sociology**: a tiny node vocabulary + Python-source codegen + a meta
dict made third-party passes trivially writable, at the price of pushing control flow and
polymorphism out of the IR entirely. Our IR needs `if`/`for` (M0's known fault line), so we can't
copy FX wholesale — but "node kinds you can count on one hand, everything else is `target` +
`meta`" is a proven extensibility recipe.

## 4. TorchInductor (brief): one define-by-run IR, two+ codegen targets

([design post](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747).)
Pipeline: FX(ATen) → **decompositions** (big-op → small-op, written *in Python as PyTorch code*,
supplied as a dict to AOTAutograd) → **lowerings** → define-by-run IR: a buffer's computation is a
Python closure `inner_fn(index: List[sympy.Expr])` that calls a virtualized ops interface —
`ops.load("x", i1 + i0*size1)`, `ops.add(...)`, `ops.store(...)`. Sizes/strides are sympy
expressions throughout (specializing on 0/1, guarding the rest). Codegen = **swap the
implementation of `ops.*`**: a Triton handler pretty-prints Triton kernels, a C++/OpenMP handler
prints C++; the scheduler decides fusion via `can_fuse()`/`score_fusion()`. `TensorBox`/
`StorageBox` mirror Tensor/Storage so views and mutation are representable.

For pdum.dsl the fusion/scheduling machinery is out of scope, but two seams are directly
relevant: (a) **decompositions-in-Python** is the "batteries economics" answer — define `mean`
once as DSL-level code over primitive ops, and every backend gets it for the cost of the
primitives; (b) the **`ops.*` handler swap** is a concrete, proven shape for our backend seam: the
IR calls an abstract emitter interface; a backend is "an object implementing ~50 small methods",
not a tree-walker that must know the whole IR. (This is tagless-final style; it composes — Inductor
stacks wrapper handlers for masking, indirect indexing, etc.)

## 5. AOTAutograd (brief): where AD sits

AOTAutograd takes Dynamo's forward FX graph, runs the *real eager autograd engine* over
**FakeTensors** (metadata-only tensors) to record a **joint forward+backward graph**, then:
functionalizes (mutation → pure ops), applies decompositions, and splits the joint graph with a
**min-cut partitioner** that chooses which activations to save vs. recompute in backward
(rematerialization for memory). Data-dependent ops can't run on FakeTensors, so they graph-break
upstream in Dynamo.

Contrast with JAX: JAX's `grad` is a per-primitive interpreter transformation (every primitive
carries a JVP/transpose rule; transformations compose because they're all interpreters over the
same jaxpr). AOTAutograd instead **reuses the existing runtime's AD by tracing through it** — AD
correctness comes from the eager engine, and the compiler only ever sees the already-differentiated
graph. That's the pragmatic choice when you have a mature AD engine and an IR (FX) with no
control flow; it also means AD in PyTorch is *not* user-extensible per-op at the IR level the way
JAX rules are. pdum.dsl has no legacy engine to reuse, and wants `grad`/`vmap` as first-class
composable transforms — which argues for the JAX shape (small primitive set + per-primitive
transformation rules over our IR), not the AOTAutograd shape. The reusable AOTAutograd ideas are
narrower: trace AD **after** capture on metadata-only values, keep the joint graph around so a
scheduling pass can decide recompute-vs-store, and let transformations add entries to the same
cache key space as ordinary compilation (in torch, transformed artifacts are guarded/cached
through the identical machinery).

---

## Design lessons for pdum.dsl

1. **Bytecode frontends buy generality with a permanent CPython tax — don't pay it.** 71
   version branches in one Dynamo file, ~350 KB of version-sensitive frontend, 3–12 month support
   lag per Python release, and self-described brittle heuristics over undocumented internals.
   Dynamo needs bytecode because it must trace *arbitrary* Python including foreign libraries
   mid-frame. Our DSL owns a small subset with source available (decorated defs); the AST
   frontend (desiderata §7.1) is the right call. Adopt numba's posture only if we ever need to
   trace code we didn't decorate — we don't.

2. **Type keys are the O(1) degenerate case of guards; keep them degenerate.** Dynamo's per-call
   cost (linear scan × C++ predicate-tree eval, recompile_limit=8, unsafe-skip escape hatches,
   a profiler bucket just for cache lookup) is what "cache key = observations made during
   tracing" costs at scale. Our hot path must remain: `typeof` each capture/arg → build tuple →
   one dict hit. Any feature that would require *re-inspecting object graphs per call* (guard
   creep) should instead either enter the type key explicitly or be rejected.

3. **But adopt the guard system's *inventory* as our checklist of what must live in the key.**
   Dynamo guards ambient state (`GRAD_MODE`, `DEFAULT_DEVICE`, global torch state), backend
   identity (`backend_match` in lookup), function identity, closure contents, and container
   structure. Translation: our cache key must include backend + backend parameters (M0 gap),
   generation/world-age, code identity, env_types — and nothing else may vary behavior. Their
   relational guards (input aliasing) name a real hole in pure type keys: decide explicitly
   whether aliasing between array captures is (a) irrelevant per backend, (b) normalized at
   marshal time, or (c) an explicit key component.

4. **Value specialization must be opt-in with a budget.** Dynamo specializes on values by default
   (static shapes, constant ints) and then needs automatic-dynamic, 0/1 special cases, and
   recompile limits to un-dig itself. Confirms our `Val{}`/`Literal` stance: values enter the key
   only by explicit lifting; consider a per-template compile-count warning (their
   `recompile_limit` idea repurposed as a lint, not a fallback).

5. **Steal FX's IR sociology, not its IR.** A handful of node kinds + open `meta` dict + readable
   codegen made an ecosystem of third-party passes possible. Our IR needs real control flow
   (FX's absence of it is why Dynamo must shatter functions into graph fragments + resume
   functions — a machinery we should never need), but keep the node vocabulary minimal and put
   analysis results (types, units, shapes) in a `meta`-like side channel so new analyses don't
   change the IR schema. An `Interpreter`-with-overridable-per-op-methods base class is the
   cheapest extension seam for analyses and transforms.

6. **The backend seam should be an emitter interface, not a tree-walker.** Inductor's virtualized
   `ops.*` handler — backend = object implementing a small method-per-primitive interface,
   codegen = running the IR's closures against a printing handler — is a proven shape for
   "core doesn't know what WGSL is". Fits M0's fault line (core currently imports WGSL intrinsic
   tables) and makes backend #4 a file, not a fork.

7. **Batteries via decompositions.** Inductor implements the long tail of ATen by *decomposing in
   Python* to a small primitive set, once, backend-independently. Our `mean`/`clip`/records
   batteries should be DSL-level definitions lowered through the same pipeline as user code;
   only the primitive set is per-backend work.

8. **Put AD at the JAX layer, not the AOTAutograd layer.** AOTAutograd shows AD *can* be a
   post-capture graph-to-graph transform, but only because a full eager AD engine already
   existed to trace through. For us, per-primitive rules (JVP/transpose, batching) over a small
   IR are less total work, user-extensible per primitive (same table a new backend or a units
   rule plugs into), and compose with vmap. Keep AOTAutograd's one structural trick: transforms
   produce ordinary templates that flow into the *same* type-keyed cache (`grad(f)` is just
   another (code identity, env_types) key).

### Sources

- Dynamo deep dive: <https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamo_deepdive.html>
- PyTorch 2 paper (ASPLOS'24): <https://docs.pytorch.org/assets/pytorch2-2.pdf>
- Guards/caching source: <https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/guards.py>,
  <https://github.com/pytorch/pytorch/blob/main/torch/csrc/dynamo/guards.cpp>,
  <https://github.com/pytorch/pytorch/blob/main/torch/csrc/dynamo/extra_state.cpp>,
  <https://github.com/pytorch/pytorch/blob/main/torch/csrc/dynamo/eval_frame.c>
- CPython churn: <https://dev-discuss.pytorch.org/t/supporting-dynamo-in-python-3-12/2320>,
  <https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-13-completed/2738>,
  <https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-14-completed/3276>
- FX: <https://docs.pytorch.org/docs/2.13/fx.html>
- Inductor: <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>
- set_stance / guard-overhead knobs: <https://docs.pytorch.org/docs/2.13/generated/torch.compiler.set_stance.html>
- Releases/versions: <https://github.com/pytorch/pytorch/releases>
