# R2 — DaCe: frontend technique, SDFG IR, batteries, codegen, and what is separable

Research report for the pdum.dsl redesign. Sources: the DaCe source tree cloned at
commit `9775f40f` (2026-07-03, version **2.0.0a4**), the official docs, and the
release history. All file paths below are relative to the `dace/` package in
<https://github.com/spcl/dace>.

**Current status (verified July 2026).** DaCe is actively maintained by SPCL/ETH.
Stable line is v1.0.x (v1.0.0 Nov 2025 — "faithful to the original 2019 paper",
introduced experimental control-flow regions; v1.0.2 Mar 2026). The main branch is
the **2.0 alpha** series (a4, June 2026): explicit control-flow regions are now
default (`use_explicit_cf=True`), autodiff/ONNX/PyTorch integration landed, and
**FPGA codegen was extracted to a separate repository** (there is no
`codegen/targets/fpga.py` in the 2.0 tree; remaining targets: `cpu`, `cuda`,
`mpi`, `snitch`, `sve`, `mlir`, plus the frame code generator). Python-frontend
docs still describe support from 3.7; the parser has explicit Python **3.14**
handling (`__annotate__` closures in `frontend/python/parser.py`).
Releases: <https://github.com/spcl/dace/releases>. IR docs:
<https://spcldace.readthedocs.io/en/latest/sdfg/ir.html>.

**Scale (line counts, 2.0.0a4):** whole package ~160k lines of Python.
`frontend/python` 18.3k, `sdfg` (IR) 26.9k, `codegen` 18.4k, `libraries`
(batteries) 18.5k, `transformation` 39.1k. This is the cost of the full
dataflow-analysis approach; keep it in mind as each mechanism below is sized.

---

## 1. Frontend: AST-based, source-required — the open question settled

**DaCe's Python frontend is AST-based, not bytecode-based.** The pipeline is:

1. `astutils.function_to_ast(f)` (`frontend/python/astutils.py:27`) calls
   `inspect.getsource(f)` + `ast.parse`, falling back to `dill.source.getsource`
   for interpreter-defined functions. If source is unavailable it raises:
   *"Cannot obtain source code for dace program… use IPython/Jupyter or place the
   source code in a file."* This is the standing cost of the AST route: **no
   source, no compile** (numba's bytecode route has no such restriction).
2. `preprocessing.preprocess_dace_program` (`preprocessing.py:1569`, file is
   1,709 lines) runs a stack of `ast.NodeTransformer`s **to fixpoint** (config
   `frontend.preprocessing_passes`, default: iterate until `ast.dump` stops
   changing): `StructTransformer`, `ModuleResolver` (rewrites `np.` aliases to
   canonical module names), `GlobalResolver`, `LoopUnroller`,
   `ExpressionInliner`, `ContextManagerInliner`, `ConditionalCodeResolver`
   (constant-folds `if` on compile-time values), `DeadCodeEliminator`,
   then `CallTreeResolver` (recursively preprocesses nested calls) and
   `ArrayClosureResolver`.
3. `newast.parse_dace_program` → `ProgramVisitor` (`newast.py`, **5,876 lines**)
   walks the preprocessed AST and *directly emits SDFG* — no intermediate typed
   AST. Statements become states/control-flow regions; expressions become
   tasklets, maps, and memlets; every call/operator/method/attribute is looked up
   in the `Replacements` registry (§3).

### Types and symbols

- **Declared** via type hints: `def f(A: dace.float64[M, N], x: dace.int32)`.
  `M`, `N` are `dace.symbol` objects — sympy symbols living in the function's
  globals. Annotations are the AOT path.
- **Inferred (JIT mode)**: with no hints, the *actual argument values* at first
  call are converted by `data.create_datadescriptor(obj)`
  (`data/creation.py:24`) into data descriptors. This one ~110-line function is
  the entire value→type story: dispatches on `Data` passthrough, a
  `__descriptor__`/`descriptor` **protocol for user types**, torch tensors,
  anything with `__array_interface__`/`__cuda_array_interface__` (structs
  handled via dtype fields), cupy arrays, sympy expressions, numpy scalars,
  builtins, `None` (→ `void*`), strings, callables (→ opaque `callback` type).
- A descriptor is `Array(dtype, shape, strides, storage, …)` or `Scalar(dtype)`.
  Crucially, in JIT mode the shape entries are **concrete integers** — so a JIT
  DaCe program specializes (re-parses *and* recompiles) **per concrete shape**.
  Only annotated symbolic shapes generalize across sizes. (Contrast numba and
  pdum: `ndim` in the type, shape as runtime data.)

### Closures, globals, free symbols

DaCe **merges locals and globals into one namespace**:
`_get_locals_and_globals(f)` (`parser.py:48`) builds a dict from `f.__globals__`
updated with `zip(f.__code__.co_freevars, cells)` — closure variables simply
shadow globals. This dict is **re-read on every `__call__`**
(`parser.py:432`), so rebinding a global/free variable between calls is seen.
`GlobalResolver` (`preprocessing.py:429`) then walks every `Name`/`Attribute` in
Load context and splits referenced values into four fates:

| Referenced value | Fate | Mechanism |
|---|---|---|
| scalar constant, string, dtype | **frozen as compile-time constant** into the AST | `closure.closure_constants[qualname] = value`; AST node replaced by a literal |
| `dace.symbol` | stays symbolic | replaced by `ast.Name(symbol.name)` |
| array (`dtypes.is_array`) | **lifted to a hidden argument** | renamed `__g_<sanitized_qualname>`, descriptor via `create_datadescriptor`, plus a **re-evaluation thunk** `lambda: eval(qualname, globals)` stored in `closure.closure_arrays[arrname] = (qualname, desc, eval_thunk, …)`; dedup by `id(value)` in `closure.array_mapping` |
| callable | try to parse as nested `DaceProgram`; on *any* failure, register a **callback** into the interpreter (`closure.callbacks`), passed as a function-pointer argument (`dtypes.callback`); object methods handled by binding `__self__` as `methodobj` |
| SDFG / `__sdfg__`-convertible | inlined as nested SDFG | `closure.closure_sdfgs` |

The result object is `SDFGClosure` (`frontend/python/common.py`):
`closure_arrays`, `closure_constants`, `callbacks`, `closure_sdfgs`,
`array_mapping`, `nested_closures` — with `combine_nested_closures()` hoisting
nested functions' captured arrays into the top-level closure (flat namespace,
dedup by object identity).

**Key mechanism for pdum:** captured arrays are not compile-time constants and
not snapshots — they are *promoted to arguments*, and the stored thunk
re-evaluates `eval('obj.attr.arr', globals)` **at every call** to fetch the
current value (`__sdfg_closure__`, `parser.py:337`). That is exactly a
"logical capture → physical argument" marshaling seam, implemented as
name-mangled extra parameters plus per-capture value thunks.

### Program cache — what is the key?

`DaceProgramCache` (`frontend/python/cached_program.py`, 153 lines) is an LRU
(`LimitedSizeDict`, default size from config `frontend.cache_size`) per
`DaceProgram` instance mapping `ProgramCacheKey → (sdfg, compiled_sdfg)`.
The key (`cached_program.py:54`) is a frozen tuple of:

1. `arg_types`: `{name: str(descriptor.to_json())}` — dtype, **concrete shape**
   in JIT mode, strides, storage, alignment…
2. `closure_types`: descriptors of closure arrays, **re-evaluated via the thunks
   at call time** (so a captured array swapped for a differently-shaped one
   changes the key);
3. `closure_constants`: **values** of frozen scalar constants (re-`eval`ed each
   call — a changed global scalar is a cache miss → full re-parse + recompile);
4. `specified_args` (which defaults were overridden);
5. `id()` of registered SDFG call hooks.

So DaCe is *value-keyed for scalar captures* and *descriptor-keyed for array
captures/arguments*. There is no code-object identity in the key at all — the
cache lives on the `DaceProgram` object (numba-style identity anchoring), and
redefinition simply creates a new `DaceProgram`. There is no invalidation story
beyond LRU eviction; a mutated global scalar is caught only because constants
are re-evaluated per call (each `__call__` also re-runs `eval` for every
constant and `create_datadescriptor` for every closure array — the **hot path
is not hot**: on the order of dicts, `eval`s, and a sympy `solve` in
`infer_symbols_from_datadescriptor` (`parser.py:81`) to recover symbol values
(N, strides) from actual arrays at every call).

On-disk: the *binary* cache is a build folder keyed by config `cache ∈ {single,
hash, unique, name}`; with `hash`, the folder name embeds
`md5(sdfg.to_json())` (`sdfg/sdfg.py:1206`), and `SDFG.hash_sdfg()` computes a
SHA-256 of the JSON with nondeterministic fields (names, guids, transformation
history, instrumentation) stripped. `sdfg.compile()` skips codegen if the
library file already exists and `recompile` is off (`sdfg/sdfg.py:2531`).
`regenerate_code=False` / `load_precompiled_sdfg` let users pin a hand-tuned
binary to a program.

---

## 2. The SDFG IR: what it buys, what it costs

The Stateful DataFlow multiGraph (docs:
<https://spcldace.readthedocs.io/en/latest/sdfg/ir.html>) is a two-level IR:

- **Control level:** a hierarchy of `ControlFlowRegion`s (2.0: explicit
  `LoopRegion`, `ConditionalBlock` classes, `sdfg/state.py:3202,3215,3815`)
  whose leaves are **states**; interstate edges carry a symbolic condition plus
  a dict of **symbol assignments** — the *only* place a symbol can change.
- **Dataflow level:** each state is a DAG of access nodes (data containers),
  tasklets (opaque compute, "may not access memory except via edges"), map
  entry/exit pairs (parametric parallelism over symbolic ranges), and nested
  SDFGs. Edges carry **memlets**.

A `Memlet` (`memlet.py`, 759 lines) = `(data, subset, other_subset, volume,
dynamic, wcr, …)` where `subset` is a symbolic multidimensional range
(`begin:end:step` per dimension, sympy expressions over symbols) and `wcr` is a
write-conflict-resolution lambda (`lambda a, b: a + b`) making reductions
first-class data movement. `MemletTree` propagates subsets through map scopes.
Shapes, strides, map ranges, and interstate conditions are all
**sympy** expressions (`symbolic.py`, 2,698 lines of sympy wrapping — pickling,
C++ printing, equality semantics, assumptions).

**What it buys them.**
- Every byte moved is explicit ⇒ transformations (tiling, fusion, GPU offload,
  double-buffering) are graph rewrites with checkable legality; ~39k lines of
  `transformation/` operate on the IR without re-analyzing Python.
- Symbolic shapes ⇒ one compile serves all sizes; symbols become plain scalar
  kernel arguments.
- The same IR is the exchange format for tools (VS Code visual optimizer,
  serialization to JSON `.sdfg` files with strict round-trip tests in codegen).
- WCR + maps give portable parallel semantics that each backend schedules its
  own way.

**What it costs (why their analysis style is heavier than pdum wants).**
- The frontend must *prove* dataflow: every subscript is parsed into a symbolic
  subset (`memlet_parser.py`), every statement's reads/writes are computed, and
  memlet propagation must be correct through arbitrary scope nesting. That is
  why `newast.py` alone is 5.9k lines — converting *statements* is easy;
  converting *array accesses into provably-correct symbolic subsets* is not.
- sympy on the critical path: parsing, simplification, propagation, and even
  the per-call symbol inference (`sympy.solve`) are heavyweight; DaCe's answer
  to compile latency is caching folders, not fast compiles. Interactive
  first-compile latency is seconds, not milliseconds.
- The dataflow model rejects or awkwardly encodes anything without static data
  access (recursion unsupported; lists/dicts/sets unsupported as data; classes
  JIT-only; `try/except`, `yield`, dynamic access → interpreter **callback**
  fallback with a warning).
- Correct-by-construction graph surgery is verbose: even library-node expansion
  code (§3) spends most of its lines wiring connectors and memlets.

The supported subset is honestly documented per Python-reference section in
`frontend/python/python_supported_features.md` (keywords supported: `if/elif/
else, for, while, break, continue, def, return, lambda, with, and/or/not`;
rejected: `class, try, global, import, yield, async, del, pass, raise`) — a
documentation discipline worth copying.

---

## 3. Batteries: replacements (frontend) + library nodes (IR) + expansions (backend)

DaCe's numpy story has **two layers with a clean seam**, and it is the single
most transferable design in the project.

### Layer 1: frontend replacements — syntax → IR, once

`frontend/common/op_repository.py` (165 lines) is a registry of five dicts with
decorator-based registration:

```python
@oprepo.replaces('numpy.matmul')            # by pydoc name
@oprepo.replaces_operator('Array', 'Add', otherclass='Scalar')
@oprepo.replaces_method('Array', 'astype')  # method-call syntax on a type
@oprepo.replaces_attribute('Array', 'T')    # attribute syntax
@oprepo.replaces_ufunc('ufunc')             # generic ufunc protocol
def _op(visitor, sdfg, state, ...) -> Tuple[str]: ...
```

Each replacement receives the visitor + SDFG + state + operand array names and
returns output array names; it *builds a small subgraph* (or inserts one
library node). `ProgramVisitor.visit_Call` resolution order: `__sdfg__`
convertibles → `Replacements.get(full_qualified_name)` → method/attribute
tables (with **MRO walk**: `_get_all_bases` tries subclass names before base
class names, so `View` falls back to `Array` rules) → nested-program parsing →
interpreter callback. Operator dispatch is `(lhs_type_name, rhs_type_name,
op_name)` with MRO product search (`op_repository.py:45`).

The numpy surface lives in `frontend/python/replacements/` (~7.4k lines total:
`operators.py` 928 implements *all* Python operators over Array/Scalar/symbol
with broadcasting; `ufunc.py` 1,884 implements the full ufunc protocol —
`reduce`/`accumulate`/`outer` included — **once, generically**, driven by small
per-ufunc tables of `{name, operator, expression}`; `reduction.py`,
`array_creation*.py`, `linalg.py`, `fft.py`, …). Method-call syntax on user
types is exactly `replaces_method(classname, method)` — pdum's `Color.to_hsv()`
desideratum is this table.

### Layer 2: library nodes — IR-level intrinsics with per-target expansions

A `LibraryNode` (`sdfg/nodes.py:1333`) is an opaque graph node with typed
connectors, properties (e.g. `alpha`, `beta`, `transA`), a dict of
**implementations**, and a `default_implementation`. Registration is
declarative (`library.py`, 224 lines — the whole mechanism):

```python
@dace.library.node
class MatMul(LibraryNode):
    implementations = {"specialize": SpecializeMatMul}
    default_implementation = "specialize"

@dace.library.expansion
class ExpandGemmPure(ExpandTransformation):
    environments = []                      # no external deps
    @staticmethod
    def expansion(node, state, sdfg): ...  # returns SDFG subgraph / another node

class ExpandGemmMKL(...):   environments = [environments.intel_mkl.IntelMKL]
class ExpandGemmCuBLAS(...): environments = [environments.cublas.cuBLAS]
```

Mechanics worth stealing:

- **`numpy.matmul` is registered once** (`replacements/linalg.py` inserts a
  `MatMul` node). `MatMul` is a *meta-node*: its `specialize` expansion
  (`libraries/blas/nodes/matmul.py:189`) inspects operand ranks at expansion
  time and **delegates to Gemm / BatchedMatMul / Gemv / Dot** — shape-driven
  overload resolution *inside the IR*, after types are known.
- Every node has a **`pure` expansion**: a portable definition as a plain SDFG
  subgraph (maps + tasklets + WCR), which every backend can already compile.
  Fast paths (OpenBLAS/MKL/cuBLAS/rocBLAS/PBLAS — six expansions in `gemm.py`,
  628 lines) are *optional accelerations selected per node instance or via
  `dace.library.change_default(blas, 'cuBLAS')`*. **This is the answer to the
  batteries-economics question: one portable definition, O(1) per new op;
  backend-specialized versions added incrementally where they pay.**
- **Environments** (`@dace.library.environment`) carry the build-system
  half of an intrinsic: cmake packages/flags, headers, init/finalize code,
  dependencies (topologically sorted at codegen). An expansion *declares* what
  it needs; codegen collects `used_environments` and emits the build config.
- Expansion happens in `generate_code`: `sdfg.expand_library_nodes()` runs
  recursively before target dispatch (`codegen/codegen.py:206`), followed by a
  re-run of connector type inference. Libraries are plain Python modules
  self-registering via `dace.library.register_library(__name__, 'blas')` —
  third-party batteries (`libraries/torch`, `onnx`, `stencil`, …) plug in
  without touching core.

---

## 4. Codegen: dispatcher-of-targets, one C ABI, ctypes call path

### Target structure

`codegen/codegen.py:generate_code` (296 lines) drives everything: validate →
control-flow raising → connector type inference → library expansion → frame
generation. Targets are classes extending `TargetCodeGenerator`
(`codegen/target.py`), self-registered via `@registry.autoregister_params
(name='cuda')`. The **frame code generator** (`targets/framecode.py`, 1.1k
lines) emits the program skeleton (state machine, allocations, init/exit) in
C++, and a `TargetDispatcher` (`codegen/dispatcher.py`) routes each IR element
to a target by **predicate registration** — a target's `__init__` claims work:

```python
dispatcher.register_map_dispatcher(dtypes.GPU_SCHEDULES, self)      # by schedule enum
dispatcher.register_array_dispatcher(GPU_STORAGES, self)            # by storage enum
dispatcher.register_node_dispatcher(self, self.node_dispatch_predicate)
dispatcher.register_copy_dispatcher(src_storage, dst_storage, schedule, self)
```

(`targets/cuda.py:109-140`; illegal storage pairs register an `illegal_copy`
sentinel that raises at codegen.) So "backend" decomposes into **five
orthogonal capabilities — states, scopes/maps, nodes, array
allocation-per-storage-class, and copies-per-storage-pair** — and a new target
implements only the slices it claims; CPU is the default for everything
unclaimed. CUDA codegen is a *subclass user* of CPU codegen for tasklet bodies
(with an acknowledged TODO that CPU codegen should be further factored). All
targets emit C++/CUDA source compiled via **CMake** (`codegen/compiler.py`);
`snitch`/`sve`/`mlir` show the seam genuinely admits exotic targets.

### Marshaling / ABI

The generated library exports exactly three symbols per SDFG
(`codegen/codegen.py:generate_headers`):

```c
Handle_t __dace_init_<name>(symbol args...);     // allocates persistent state struct
void __program_<name>(Handle_t, data args...);   // the program
int  __dace_exit_<name>(Handle_t);
```

- The **argument list is derived from the IR**: `sdfg.arglist()` = sorted
  non-transient data descriptors + free symbols. Each argument is one C
  parameter: arrays decay to a **raw typed pointer** (`double * __restrict__`),
  scalars by value, callbacks as function pointers. **Shapes/strides are not
  passed with the array** — they were either baked in as constants (JIT mode)
  or are **symbols passed as separate scalar arguments** and threaded through
  generated index arithmetic. This is DaCe's answer to "one logical value → N
  physical parameters": the split is done *in the IR* (descriptor + symbols)
  rather than in a marshaling layer; the runtime side recovers symbol values
  from the numpy arrays by solving `desc.shape/strides == arg.shape/strides`
  with sympy (`infer_symbols_from_datadescriptor`).
- A generated **state struct** holds persistent allocations (transients with
  SDFG lifetime, GPU streams, library handles like cublasHandle_t, environment
  `state_fields`); `__dace_init` builds it once, calls take it as the handle —
  cheap per-call re-entry, expensive stuff done once. (pdum's uniform-buffer /
  pipeline-object story is the same shape.)
- Python side: `CompiledSDFG` (`codegen/compiled_sdfg.py`, 811 lines) loads the
  library through a **stub DLL** (`ReloadableDLL`) that can force-unload for
  recompile-and-reload in one process. `__call__` = `construct_arguments()`
  (kwargs → ordered arglist → `make_ctypes_argument` per value, with view/type
  checks) → `fast_call()` (raw `self._cfunc(handle, *cargs)`), plus return-array
  allocation (`__return*` arrays allocated numpy/cupy-side from symbolic shapes
  evaluated against current symbols). For hot loops DaCe explicitly exposes
  the split: users cache `construct_arguments()` output and invoke
  `fast_call()` directly — an acknowledgment that the friendly path is slow.

### Recompilation behavior, summarized

| Change | Effect |
|---|---|
| same types/shapes, new array values | cache hit; values re-marshaled per call (thunks re-`eval` captured arrays) |
| new concrete shape, JIT mode (no symbolic annotation) | **new key → re-parse + full recompile** |
| new shape, annotated symbolic dims | hit; new symbol values passed as args |
| changed captured scalar / global scalar | new key (constants are values in the key) → re-parse + recompile |
| captured array replaced by same-descriptor array | hit (descriptor equality); thunk fetches new pointer |
| edited function source | *not detected* on the same `DaceProgram`; new decoration = new object = new cache |
| SDFG graph edited | new content hash → new build folder |

---

## 5. Separable vs. inherently tied to the dataflow model

**Conceptually separable / reusable for pdum.dsl:**

1. **The replacements registry** (`op_repository.py`, 165 lines): five tables
   (function-by-qualname, operator-by-type-pair, method, attribute, ufunc) with
   decorator registration and MRO fallback. Nothing about it is
   dataflow-specific — the registered callables could just as well emit pdum IR
   nodes. This *is* the "dialect as input to the frontend" seam pdum needs.
2. **The two-layer battery design**: high-level op → IR meta-node (shape-driven
   delegation at expansion time) → per-target expansions with a mandatory
   portable `pure` fallback + declarative build environments. Directly answers
   desiderata §7.4.
3. **`create_datadescriptor` as a single value→type protocol** including the
   `__descriptor__` hook for user types — a clean pattern for pdum's `typeof`,
   though pdum must add int-range bucketing and layout canonicalization DaCe
   skips (DaCe can skip them because shapes/strides are in the descriptor).
4. **Preprocessing as composable AST passes run to fixpoint** — closure/global
   resolution, constant folding of `if`s, loop unrolling, context-manager
   inlining as independent `NodeTransformer`s. The `GlobalResolver` fate-split
   (constant / symbol / lifted-argument / callback / nested-program) is a
   complete taxonomy pdum should mirror, with one deliberate inversion: DaCe
   freezes scalar captures by value; pdum types them.
5. **Predicate-based codegen dispatcher**: backends claim (states, scopes,
   nodes, storage classes, copy pairs) instead of owning whole programs; a
   default target covers the rest. Scales to partial backends — exactly pdum's
   incremental-backend story.
6. **The three-symbol C ABI + persistent state struct + stub-DLL reload**, and
   the `__call__` vs `construct_arguments`/`fast_call` split (though pdum
   should make the fast path the only path).
7. **Honest subset documentation** (`python_supported_features.md`) and the
   serialization round-trip test in `generate_code`.

**Inherently tied to the dataflow model (do not adopt):**

- Memlets, subset algebra, memlet propagation, WCR — the entire explicit
  data-movement layer, and the ~5.9k-line AST→SDFG converter whose bulk exists
  to compute them.
- sympy as the type/shape substrate, including per-call `sympy.solve` for
  symbol recovery. pdum's (dtype, ndim, layout) types + runtime shape arguments
  make this unnecessary.
- The state-machine/control-region hierarchy as *the* program representation
  (pdum wants structured control flow in a small IR, not a CFG-of-dataflow-DAGs).
- Interstate-edge symbol assignment semantics, transformation framework sized
  to graph surgery (39k lines), and the graph-first tooling ecosystem.
- Value-keyed scalar constants + eval-thunks-per-call caching. It works for
  DaCe because their unit of reuse is "one program, many runs"; it is the
  anti-pattern for pdum's tight-loop closure rebuilding (a changed captured
  scalar means full re-parse + C++ recompile).

---

## Design lessons for pdum.dsl

1. **AST frontend is proven at scale, with one hard precondition: source
   availability.** DaCe ships production AST-based compilation (with 3.14
   support) and never touches bytecode; version churn is handled by targeting
   `ast` (stable, structured) rather than opcodes. Adopt AST, but decide the
   no-source policy up front (fail loudly like DaCe, and keep pdum's
   code-object identity for caching — DaCe has no template identity at all and
   therefore no redefinition story).
2. **Split "name resolution + capture classification" from "IR construction"
   as separate passes.** DaCe's `GlobalResolver` produces an explicit closure
   object (`closure_arrays` with per-capture re-evaluation thunks,
   `closure_constants`, `callbacks`) *before* any lowering. pdum's phase A can
   emit the same artifact, but keyed on **types** where DaCe keys on values —
   and DaCe's per-call `eval` thunks show what to avoid on the hot path:
   resolve capture *slots* once, re-read only cell contents per call.
3. **Steal the replacements registry nearly verbatim (~150-200 lines).**
   Tables for functions/operators/methods/attributes with MRO fallback give
   pdum: numpy-call batteries, operator overloading on DSL types, and the
   `Color.to_hsv()` method syntax — all as registrations, no core changes.
   Make the registry an explicit *dialect object passed to the frontend*, not
   module-level singletons (DaCe's one real modularity sin here).
4. **Answer batteries economics with meta-node + mandatory portable expansion +
   optional per-backend fast paths.** One registration per op; a `pure`
   definition in terms of pdum IR primitives that *every* backend can lower;
   `cuBLAS`-style specializations added independently later, selected by
   node/type, each declaring its build environment. New backend cost = lower
   the primitive set, inherit the whole library.
5. **Do marshaling as "descriptor + symbols", but bind symbols structurally,
   not by solving.** DaCe's ABI (array → bare pointer; shape/stride symbols →
   separate scalar params; persistent state struct built once by
   `__dace_init`) is a good concrete model for the logical-value→
   N-physical-parameters layer, including WebGPU uniform packing. But recover
   dims by *position* in the descriptor (`arr.shape[0] → param n0`), never by
   per-call symbolic equation solving.
6. **Precompute the call path at compile time.** DaCe rebuilds kwargs dicts,
   re-evals constants, and re-runs descriptor creation on every call, then
   offers `fast_call` as the escape hatch. pdum should compile a *marshaling
   plan* (ordered slot-extraction program) per cache entry so the default call
   is the fast call — this is the "loop stays hot" requirement made concrete.
7. **Backend seam = capability claims against a default backend, not a
   monolithic per-target emitter.** Predicate/enum-based dispatch (schedules,
   storage classes, copy pairs) lets a new backend start by claiming only
   kernels-and-copies while the Python/C backend handles the rest — matching
   pdum's incremental-backend directive. Also copy the discipline that adding
   a target never edits the frontend (DaCe targets self-register).
8. **Treat DaCe's shape handling as a warning for the type lattice.** JIT mode
   keys on concrete shapes (recompile per size); the alternative costs sympy.
   pdum's honest types should keep `ndim`/layout in the type and shapes in the
   marshaled values by default, with `Literal`-style opt-in for baking
   dimensions — exactly the middle path DaCe doesn't have.
