# R1 — Numba internals: the ergonomic north star and the documented anti-pattern

*Research report for the pdum.dsl redesign. Evidence gathered July 11, 2026, from a
fresh clone of `numba/numba` (commit `faf66df`, July 10, 2026 — post-0.66.0 mainline)
and `NVIDIA/numba-cuda` (July 6, 2026 — v0.30.x line), plus live release notes and
issue tracker. All file paths below are relative to the numba repo root; all line
counts are from these clones, not estimates.*

Current state, verified live:

- numba **0.66.0** released July 1, 2026; supports CPython 3.10–3.14
  ([PyPI](https://pypi.org/project/numba/), [releases](https://github.com/numba/numba/releases)).
- **numba-cuda** (NVIDIA-maintained CUDA target) latest v0.30.4, July 3, 2026 — and its
  README now carries a **maintenance-mode notice**: security/critical fixes only, new
  development moved to [numba-cuda-mlir](https://github.com/NVIDIA/numba-cuda-mlir),
  a **from-scratch MLIR rebuild** shipped with CUDA 13.3 (May 2026)
  ([numba-cuda README](https://github.com/NVIDIA/numba-cuda),
  [releases](https://github.com/NVIDIA/numba-cuda/releases)).

---

## 1. Frontend: bytecode → Numba IR

### 1.1 The pipeline

Numba never sees source or AST. The untyped frontend is a three-stage
reconstruction of structure that CPython's compiler already had and threw away:

1. **`numba/core/bytecode.py`** — wraps the function object: `FunctionIdentity`
   (module, qualname, code object) and `ByteCode` (a scan of `dis` instructions with
   argument decoding per Python version).
2. **`numba/core/byteflow.py` (2,458 lines)** — `Flow`: *"Simulate execution to
   recover dataflow and controlflow information."* An abstract interpreter over the
   CPython **value stack**: a `State` per basic block carries a virtual stack; there
   are **130 `op_*` handlers** (one per opcode numba supports), each popping/pushing
   symbolic values and recording an "instruction info" dict. Block kinds
   (`LOOP/TRY/EXCEPT/FINALLY/WITH`) are tracked to recover structured regions;
   3.11+ exception tables get their own path. Output: a CFG plus, for every
   instruction, which stack temporaries it consumed/produced.
3. **`numba/core/interpreter.py` (3,558 lines)** — `Interpreter`: a *second* set of
   **157 `op_*` handlers** replays the flow and emits register-based **Numba IR**
   (`numba/core/ir.py`, 1,702 lines: `Assign`, `Expr`, `Branch`, `Jump`, `Global`,
   `FreeVar`, `Const`, …). SSA construction is a separate later pass
   (`numba/core/ssa.py`).

Then come **peephole reconstruction passes** — needed *only because the AST-level
structure was destroyed by the bytecode compiler*, ~1,300 lines of interpreter.py:
`peep_hole_call_function_ex_to_call_function_kw` (rebuild kwargs calls from
`CALL_FUNCTION_EX` dict-building sequences, ~600 lines with its helpers),
`peep_hole_list_to_tuple` (undo `LIST_TO_TUPLE` build patterns),
`peep_hole_fuse_dict_add_updates`, `peep_hole_delete_with_exit`,
`peep_hole_split_at_pop_block`.

### 1.2 Why bytecode at all

The genuine advantages, worth stating fairly:

- Works on **any live function object** — no source file needed (REPL, exec'd code,
  functools-wrapped code), decorators already applied, `__closure__` cells and
  `__globals__` directly available.
- Exact CPython semantics for evaluation order and scoping come for free — the
  bytecode *is* the semantics.
- No need for a name-resolution layer: `LOAD_GLOBAL`/`LOAD_DEREF` tell you exactly
  where each name comes from.

### 1.3 The real, measured cost: per-release churn

- **268 `PYVERSION` conditionals in `numba/core/*.py` alone**; **225 of them in just
  `byteflow.py` + `interpreter.py`**. Entire opcode handlers exist in 2–4 variants
  (`if PYVERSION in ((3, 12), (3, 13), (3, 14)): ... elif PYVERSION in ((3, 10),
  (3, 11)): ... else: raise NotImplementedError`). Example: `op_LOAD_DEREF` has
  separate 3.11+ and 3.10 bodies because cell/free var indexing changed.
- Support lag per CPython release (verified against release notes):
  - 3.12 (Oct 2023) → numba 0.59.0, **Jan 31, 2024** (~4 months)
    ([notes](https://numba.readthedocs.io/en/stable/release/0.59.0-notes.html) — and
    profiling support had to be *disabled* for 3.12 at release).
  - 3.13 (Oct 2024) → numba 0.61.0, **Jan 16, 2025** (~3.5 months)
    ([notes](https://numba.readthedocs.io/en/stable/release/0.61.0-notes.html));
    the port ([issue #9413](https://github.com/numba/numba/issues/9413)) was blocked
    for a month by a CPython alpha metaclass bug and churned on try/except handling.
  - 3.14 (Oct 2025) → numba 0.63.0, **Dec 8, 2025**, with a dedicated
    **0.63.0beta whose sole purpose was 3.14 support**
    ([notes](https://numba.readthedocs.io/en/latest/release/0.63.0-notes.html),
    [umbrella issue #9957](https://github.com/numba/numba/issues/9957)).
- **Patch releases break it too**: CPython 3.13.4 changed comprehension bytecode
  (extra `GET_ITER`) and numba compilation of list comprehensions failed on a
  micro-release upgrade —
  [issue #10101](https://github.com/numba/numba/issues/10101). The bytecode is
  explicitly *not* a stable interface, and numba pays for that fact continuously.

### 1.4 The closure anti-pattern, located precisely

In `interpreter.py::op_LOAD_DEREF` (3.11+ variant), a captured variable becomes:

```python
value = self.get_closure_value(idx)          # the *runtime value* in the cell
gl = ir.FreeVar(idx, name, value, loc=self.loc)
```

`ir.FreeVar` carries the **value** into the IR; lowering freezes it as a constant.
Consequently every closure instance with a different captured value is a different
program — this is the exact behavior `design/closure_specialization.md` documents,
and the single most important thing pdum.dsl exists to do differently. Note the
mechanism: the frontend consumes values at *IR construction* time, so no later
layer can recover value-independence. The capture/compile phase separation must be
designed in at the frontend.

---

## 2. Typing

### 2.1 Inference: CPA over Numba IR

`numba/core/typeinfer.py` (1,797 lines) opens with its own summary:

> *Type inference base on CPA. The algorithm guarantees monotonic growth of
> type-sets for each variable. Steps: 1. seed initial types 2. build constraints
> 3. propagate constraints 4. unify types. Constraint propagation is precise and
> does not regret (no backtracing).*

One `TypeVar` per IR variable (with optional `literal_value` qualifier); constraint
objects (`ArgConstraint`, `CallConstraint`, `BuildTupleConstraint`, …) push types
forward along dataflow; unification via a conversion lattice
(`numba/core/typeconv/`, with a castgraph and a C++ fast path). Literal types
(`types.IntegerLiteral(3)`) are numba's opt-in value-dependent specialization —
the `Val{}`-style mechanism, triggered by `prefer_literal=True` templates or
`ForceLiteralArg` exceptions during typing.

### 2.2 Typing contexts and templates

`numba/core/typing/context.py` (829 lines): `BaseContext.resolve_function_type(fn_type,
args, kws)` walks the templates attached to the function type. Templates
(`numba/core/typing/templates.py`, 1,337 lines):

- **`ConcreteTemplate`** — a static list `cases: [Signature]`, matched against arg
  types (used for operators, math builtins).
- **`AbstractTemplate`** — `generic(self, args, kws) -> Signature | None`; arbitrary
  Python logic computes a signature (the workhorse).
- **`CallableTemplate`** — `generic(self)` returns a typer callable mirroring the
  Python signature.
- **`AttributeTemplate`** for getattr resolution.

Registration is via module-level `Registry` objects (`infer`, `infer_global`,
`infer_getattr`) that contexts consume **lazily** through `RegistryLoader` streams —
new registrations made after context creation are picked up on next refresh. This
lazy-global-registry pattern is load-bearing and fragile (ordering, refresh timing,
target selection all interact).

### 2.3 `typeof`

`numba/core/typing/typeof.py`: `typeof(val, purpose)` dispatches through a
`functools.singledispatch` function `typeof_impl`, with two extension hooks:
`typeof_impl.register(MyClass)` and the duck-typed `val._numba_type_` property.
Buffer-protocol objects and cffi functions are special-cased. Critically, there is
a **C fast path** — `numba/_typeof.cpp` (1,157 lines) — used by the dispatcher on
every call, with the comment *"the behaviour for Purpose.argument must match
_typeof.c"*: the Python and C implementations must be kept in sync by hand. The
call-site dispatcher itself is also C (`numba/_dispatcher.cpp`, 1,691 lines),
resolving typecodes → compiled entry points. Numba pays ~2,850 lines of C to make
per-call type-keyed dispatch fast — a direct warning about pdum.dsl's hot loop:
`typeof`-per-call is the tax the architecture levies on every invocation, and numba
found pure Python too slow for it.

Dispatcher caching (`numba/core/dispatcher.py`, 1,329 lines): compiled results live
in `self.overloads: OrderedDict[arg_types_tuple -> CompileResult]`; the key is the
tuple of `typeof_pyval(a)` for each argument. Types only — but captures never get
this treatment because they were frozen in §1.4.

---

## 3. Extension APIs — the batteries mechanism

### 3.1 `@overload` is the whole game

`numba/core/extending.py` (597 lines — genuinely small) defines the public surface.
The core move of `@overload(func)`:

```python
def decorate(overload_func):
    template = make_overload_template(func, overload_func, opts, strict,
                                      inline, prefer_literal, **kwargs)
    infer(template)                                   # register for typing
    if callable(func):
        infer_global(func, types.Function(template))  # bind to the global
    return overload_func
```

One decorated function provides **both typing and implementation**: it is called at
*typing time* with the numba **types** of the arguments, does `isinstance` dispatch
on them, and returns a **pure-Python implementation function** specialized to those
types (or `None` to decline). The returned impl is then compiled *by the current
target's own `@jit`* — `_OverloadFunctionTemplate._get_jit_decorator()` looks up
`jit_registry[target_hw]` where `target_hw` comes from the template's
`metadata['target']` (default `'generic'`) resolved against the active target via
MRO (§5). Its inferred signature becomes the call's type; lowering calls the
compiled dispatcher (or inlines it, per `inline='never'|'always'|cost_fn`).

`@overload_method(typ, name)` / `@overload_attribute(typ, name)` are the same
mechanism registered through `infer_getattr` — **this is numba's method-call syntax
on user types**, and it's cheap: one decorator, no separate typing class.
`register_jitable` is literally a trivial `@overload(fn)` of a function over itself.

### 3.2 How `np.mean` is actually provided

`numba/np/arraymath.py` lines 434–497 — the real thing, abbreviated:

```python
@overload(np.mean)
@overload_method(types.Array, "mean")      # same impl doubles as a.mean()
def array_mean(a):
    if isinstance(a, (types.Integer, types.Boolean)):
        def _scalar_mean(a): return np.float64(a) + 0.0
        return _scalar_mean
    elif isinstance(a, types.Array):
        ...
        acc_init = get_accumulator(dtype, 0)          # closed-over constants,
        nan_value = dtype.type(np.nan)                # computed at TYPING time
        def array_mean_impl(a):
            if a.size == 0: return nan_value
            c = acc_init
            for v in np.nditer(a): c += v.item()
            return dtype.type(c / a.size)
        return array_mean_impl
```

Type-level decisions (accumulator dtype, NaN policy, datetime handling) happen in
ordinary Python at typing time; the returned impl is plain nopython Python closed
over the resulting *constants*. This is a two-stage macro system — and note that
here numba freezing closure values as constants is exactly what you *want*: the
generator/impl split makes "frozen" captures the deliberate staging mechanism.

### 3.3 How much of the battery set is portable this way — measured

Counts from the clone (non-test code):

| mechanism | count | nature |
|---|---|---|
| `@overload(...)` in `numba/np`+`cpython`+`misc`+`typed` | **491** | pure-Python, target-portable |
| `@overload_method(...)` repo-wide | **230** | pure-Python, target-portable |
| `@intrinsic` repo-wide | **166** | llvmlite codegen, LLVM-bound |
| `@lower_builtin` repo-wide | **272** | hand-written llvmlite lowering |

The structure is a **pyramid**: a hand-lowered base — `numba/np/arrayobj.py`
(7,545 lines: array construction, getitem/setitem, views, iterators, `nditer`) plus
`numba/core/lowering.py`/`imputils.py` and the NRT (reference-counted memory
runtime) — on top of which the big battery layer (`arraymath.py` 5,298 lines,
121 `@overload`s; most of `numba/cpython/`'s 15,629 lines) is **pure nopython
Python**.

Decisive real-world evidence of portability: **numba-cuda took the battery layer by
copy**. `numba_cuda/numba/cuda/np/arraymath.py` is 5,214 lines vs mainline's 5,298
(117 vs 121 `@overload`s) — near-identical, imports rewritten from `numba.np...` to
`numba.cuda.np...`, NVIDIA 2025 copyright header on top. The batteries traveled to
a GPU target essentially unchanged, *because they are written against the type
system and a small primitive base, not against LLVM*. What did NOT travel free is
the base of the pyramid (arrayobj-level lowering, the datamodel, the NRT), which
numba-cuda also had to carry (its `np/arrayobj.py` is 7,700 lines).

`@intrinsic` is the escape hatch below `@overload`: the decorated function receives
`(typing_context, *arg_types)` and returns `(signature, codegen)` where
`codegen(context, builder, sig, args)` emits llvmlite IR directly. It is registered
as both a template and a `lower_builtin` in one shot (`_IntrinsicTemplate` calls
`self._get_target_registry('intrinsic').lower(...)`). Everything written with
`@intrinsic`/`@lower_builtin` is LLVM-coupled — NVIDIA's numba-cuda-mlir migration
notes say exactly this: plain-kernel code ports by changing an import; **"code
using extension APIs will require modifications as Numba-CUDA-MLIR uses MLIR
instead of LLVM IR"**.

### 3.4 Structured data: Record and structref

- **`types.Record`** (`numba/core/types/npytypes.py`): fields as
  `(name, {type, offset})` + `size` + `aligned` — an explicit byte-layout type
  mapping 1:1 to NumPy structured dtypes; `Record.make_c_struct([...])` computes a
  C layout (by asking the CPU target context for ABI sizes — a core type
  constructor that reaches into a target, noted as a coupling smell).
- **`structref`** (`numba/experimental/structref.py`, only **400 lines**): a
  mutable, pass-by-reference struct = a `StructRef` type + two models
  (`StructRefModel` = pointer to payload, `StructRefPayload` = the fields) +
  generic getattr/setattr lowering + `define_proxy` to generate a Python-side proxy
  class with box/unbox. 400 lines buys user-defined records with attribute access
  and methods (methods then come free via `@overload_method`). This is the
  capability §4.3 of the desiderata wants, and it is small *given* the datamodel
  layer beneath it.

---

## 4. The datamodel layer — numba's marshaling story (closest prior art)

`numba/core/datamodel/` is remarkably small and is the part most directly relevant
to pdum.dsl's logical-value → physical-parameters problem:

| file | lines | role |
|---|---|---|
| `models.py` | 1,375 | the model classes |
| `packer.py` | 213 | `ArgPacker` / `DataPacker` |
| `manager.py` | 68 | type→model registry |
| `registry.py` | 18 | `@register_model` decorator |

### 4.1 One logical type, FOUR physical representations

`DataModel` (base class, `models.py`) gives every frontend type four LLVM-level
representations plus conversions between them:

```python
class DataModel:
    def get_value_type(self):    ...   # SSA-register form (inside a function)
    def get_data_type(self):     ...   # in-memory form (array element, struct field)
    def get_argument_type(self): ...   # ABI form for calls — may be a NESTED TUPLE
    def get_return_type(self):   ...   # ABI form for returns
    def as_data/from_data(self, builder, value): ...
    def as_argument/from_argument(self, builder, value): ...
    def as_return/from_return(self, builder, value): ...
    def load_from_data_pointer(self, builder, ptr): ...
    def traverse(self, builder): ...   # for NRT refcount visiting
```

Key design point: **`get_argument_type()` returns a *tree*** — `StructModel`
returns `tuple(m.get_argument_type() for m in members)`, recursively. An array
(`ArrayModel`, a `StructModel` with members `meminfo, parent, nitems, itemsize,
data*, shape: UniTuple(intp, ndim), strides: UniTuple(intp, ndim)`) therefore
flattens at the ABI to **5 + 2·ndim scalar parameters**. That is precisely
"one logical value → N physical parameters", solved by structural recursion over
per-type models.

### 4.2 ArgPacker — flatten/unflatten at the call boundary

`packer.py::ArgPacker` (~90 lines of logic):

- constructor: look up each frontend arg's model, collect `get_argument_type()`
  trees, build `_be_args = flatten(trees)` and an `_Unflattener` that remembers the
  tree shape;
- `as_arguments(builder, values)` — caller side: `as_argument` each value, flatten;
- `from_arguments(builder, args)` — callee side: unflatten, `from_argument` each.

Rationale in the docstring: nested structs have architecture-specific ABI rules
(alignment, address spaces on OpenCL/CUDA), so numba side-steps *all* of that by
only ever passing primitive leaves. This is the correct trick for pdum.dsl's
multi-backend ABI problem and it costs ~200 lines.

`DataPacker` is the sibling for packing a set of typed values into one memory blob
(`as_data` into an anonymous struct + `load_from_data_pointer`) — the shape of a
WebGPU uniform-buffer writer, minus the std140-style layout rules.

### 4.3 Registration and per-target override

`DataModelManager` (68 lines): `{type class → model factory}` with a
`WeakKeyDictionary` cache of `{type instance → model instance}` (models are
per-type-instance because e.g. `ArrayModel` depends on ndim). Two composition
operations matter: **`copy()`** and **`chain(other)`** (a `ChainMap`) — documented
purpose: *"inherit from the default data model and specialize it for a custom
target."* numba-cuda maintains its own datamodel dir this way (e.g. different
pointer address spaces). So per-backend physical representation of the same
logical type is a solved, tiny mechanism: a chained dict.

### 4.4 Boxing/unboxing

The Python↔native boundary is separate from the datamodel:
`numba/core/pythonapi.py` (1,742 lines) + `numba/core/boxing.py` (1,292 lines)
implement `@box`/`@unbox` per type (`NativeValue` wraps the unboxed LLVM value +
error bit). It is big because it speaks CPython C-API via llvmlite. For pdum.dsl
the equivalent is "how does a NumPy array / Python float become bytes in a GPU
buffer" — per-backend, and the lesson is that numba kept it out of the datamodel
proper: representation ≠ transport.

### 4.5 The flaw to avoid

Every `get_*_type()` returns an **llvmlite `ir.Type`**. The datamodel's *interface*
is target-generic but its *vocabulary* is LLVM. That single decision is why
`@intrinsic` code breaks on the MLIR rewrite and why a WGSL backend could never
reuse numba models. pdum.dsl's equivalent layer should let each backend define the
leaf vocabulary (WGSL scalar + uniform slot, CUDA arg, ctypes type, Python object)
under a shared structural recursion.

---

## 5. Target extension and the numba-cuda reality check

### 5.1 The mechanism (small and good)

`numba/core/target_extension.py` — **169 lines total**:

- A class hierarchy of target tokens: `Generic` → `CPU`/`GPU` → `CUDA`, plus
  `NPyUfunc`.
- Three registries: `target_registry: str → Target class`,
  `jit_registry: Target → jit decorator`, `dispatcher_registry: Target →
  Dispatcher class`.
- `target_override(name)` — a thread-local context manager setting the active
  target during typing.
- Resolution rule: an `@overload(..., target='cuda')` template is applicable iff
  the active target class `inherits_from` (issubclass) the declared one — so
  `target='generic'` batteries serve every backend, `target='gpu'` serves CUDA,
  and the *most derived* registration wins by registry install order per target
  context (`base.py::install_registry`).

This is a genuinely good design at a genuinely small size: overload selection by
target **MRO**, with `'generic'` as the portable default.

### 5.2 What actually happened (the two-strike verdict)

Strike 1 — **the fork**. The out-of-tree CUDA target did not stay a plugin.
Verified in the numba-cuda clone: `numba_cuda/numba/cuda/core/` contains
**vendored copies of essentially all of `numba.core`** — `byteflow.py`,
`interpreter.py`, `ir.py`, `typeinfer.py`, `lowering` support, `typed_passes.py`,
`untyped_passes.py`, `ssa.py`, `compiler_machinery.py`, … each under a 2025 NVIDIA
copyright header; `numba_cuda/numba/cuda/np/` and `cpython/` vendor the battery
layer; `compiler.py` imports `from numba.cuda.core import ir as numba_ir`, not
`numba.core`. Total non-test Python in numba-cuda: **~155,000 lines**. The
target-extension API was not enough to let the flagship external target track a
moving core; NVIDIA chose wholesale duplication over the dependency.

Strike 2 — **the rewrite**. numba-cuda is now in maintenance mode, replaced by
**numba-cuda-mlir**: *"built from scratch using MLIR and the latest NVVM
toolchain"*, ~1.4× faster JIT compile (geomean), same `@cuda.jit` surface,
kernels port by changing an import — but extension-API users must rewrite because
the codegen substrate changed from LLVM IR to MLIR
([numba-cuda-mlir](https://github.com/NVIDIA/numba-cuda-mlir),
[numba-cuda README](https://github.com/NVIDIA/numba-cuda)).

Reading: the *user-facing* contracts (`@cuda.jit`, `@overload`-style typed-Python
batteries) survived both events; the *internal* contracts (IR, datamodel-as-LLVM,
lowering registries, pass manager) survived neither. The parts of numba worth
copying are exactly the parts that survived.

## 6. Why numba is heavy — measured

Line counts from the July 2026 clone (`.py` unless noted):

| subsystem | lines | notes |
|---|---|---|
| `numba/` total Python | 304,677 | of which tests 121,280 → **~183k non-test** |
| `numba/core/` | 58,434 | the compiler proper |
| `numba/np/` | 32,422 | NumPy batteries (arrayobj 7,545; arraymath 5,298) |
| `numba/cpython/` | 15,629 | Python-semantics batteries |
| `numba/parfors/` | 10,953 | auto-parallelization (deeply invasive pass) |
| C/C++ (`.c/.h/.cpp`) | 29,869 | NRT, `_dispatcher.cpp` 1,691, `_typeof.cpp` 1,157 |
| frontend (`byteflow`+`interpreter`+`ir`) | 7,718 | plus 268 PYVERSION forks in core |
| typing (`typeinfer`+`typing/templates`+`typing/context`) | 3,963 | |
| lowering (`lowering`+`base`+`imputils`) | 3,437 | |
| datamodel | 1,674 | the small gem |
| boxing (`pythonapi`+`boxing`) | 3,034 | CPython boundary |
| pipeline (`compiler`+`typed_passes`+`untyped_passes`+`compiler_machinery`) | ~4,700 | |
| `target_extension.py` | 169 | the other small gem |
| `extending.py` | 597 | public extension surface |
| `experimental/structref.py` | 400 | |

Plus the **llvmlite** dependency (a separate project, version-locked to numba) and
LLVM itself dominating compile latency (NVIDIA's headline for the MLIR rewrite was
compile time).

Coupling points that show up in the code itself:

- **Global mutable registries everywhere** (`default_manager`, `builtin_registry`,
  typing `Registry` objects, `target_registry`) consumed lazily via
  `RegistryLoader`; import order is semantics. `extending.py` carries the comment
  *"TODO: abort now if the kwarg 'target' relates to an unregistered target, this
  requires sorting out the circular imports first"* — plus 5 more explicit
  circular-import workarounds in `numba/core`.
- **Typing and lowering are two parallel registration namespaces** — every feature
  not expressible via `@overload` must be written twice (`ConcreteTemplate` +
  `@lower_builtin`) and kept in sync by convention.
- **Python/C duplication on the hot path** — `typeof` and dispatch each exist twice
  (Python + C) with "must match" comments.
- **Core types reach into targets** — e.g. `types.Record.make_c_struct` imports
  `numba.core.registry.cpu_target` to compute layouts; flags/`targetconfig` thread
  through everything.
- The **pass manager** (`compiler_machinery.py`) makes every pass a class with
  registration ceremony; contributors routinely fork the whole pipeline
  (numba-cuda has its own `compiler.py` defining a parallel pass list).

---

## Design lessons for pdum.dsl

1. **Front on AST, not bytecode.** The measured bytecode tax: 268 version forks in
   core, 287 op-handlers in two parallel interpreters, ~1,300 lines of peephole
   passes to *re-derive the AST you could have started from*, a 2–4 month lag per
   CPython minor release, and breakage from a **patch** release (3.13.4,
   issue #10101). Bytecode's real advantages (decorators applied, closures and
   globals resolvable, no source needed) are recoverable on the AST path via
   `inspect.getsource` + `__closure__`/`__globals__` inspection — accept "source
   must be available" as a documented constraint. pdum.dsl targets one modern
   Python (3.14, t-strings) and needs new *syntax* to be pluggable, which is an
   AST-shaped requirement anyway.

2. **Never let captured values into the IR.** Numba's anti-pattern is one line:
   `ir.FreeVar(idx, name, value, loc)` — the frontend consumes cell *values* at IR
   build time, and no downstream layer can undo it. In pdum.dsl the frontend must
   emit a symbolic slot (name + capture index); values arrive only through the
   marshaling layer at call time. Make this a structural impossibility, not a
   convention.

3. **Steal `@overload` wholesale — it is the answer to batteries economics.** One
   pure-Python function, called with *types*, returning a type-specialized
   pure-Python *impl* compiled by the *active backend's own jit*: this single
   mechanism provides typing + implementation + method syntax
   (`@overload_method`) + staging (type-time constants closed over by the impl) in
   ~600 lines of framework. Measured payoff: 491 `@overload` batteries vs 272
   hand-lowered builtins in numba, and the `@overload` layer moved to a GPU target
   verbatim (numba-cuda's arraymath is a 98% identical copy). Budget: define the
   ~20-primitive hand-lowered base per backend (indexing, iteration, allocation,
   math intrinsics); write everything above it once in the DSL subset itself.

4. **Copy the datamodel *shape*, replace its vocabulary.** The
   value/data/argument/return four-representation scheme, tree-shaped
   `get_argument_type()` + `ArgPacker` flatten/unflatten, and
   `DataModelManager.copy()/chain()` for per-backend overrides directly solve
   "one logical value → N physical parameters" in ~1,700 lines (packer is ~200).
   But numba's models speak llvmlite types, which is why the MLIR rewrite breaks
   every extension. In pdum.dsl the structural recursion should be core; the leaf
   vocabulary (WGSL uniform slot, CUDA kernel arg, ctypes field, Python object)
   must be supplied by the backend. Units auto-conversion (desiderata §4.4) slots
   naturally into `as_argument` — it is an argument-side conversion, exactly where
   numba already does representation changes.

5. **The hot loop needs a fast `typeof`, by design not by C.** Numba ships 2,850
   lines of C (`_typeof.cpp`, `_dispatcher.cpp`) kept in sync with Python "by
   comment" because per-call type computation was too slow. pdum.dsl's equivalent:
   cache the (already-planned) env-layout memo per code object and make the
   per-call path a dict hit on precomputed type keys + a buffer write; measure it
   from day one; keep one implementation.

6. **Target tokens + MRO overload resolution is 169 lines — take it; global
   registries are the poison — don't.** The `Generic→CPU/GPU→CUDA` class lattice
   with `target='generic'` defaults is a clean, tiny answer to "portable batteries
   with per-backend overrides." But bind registrations to explicit context objects
   passed at construction (dialect-as-input, per the M0 review), not to
   import-order-dependent module-level registries with lazy loaders and
   circular-import TODOs.

7. **Design the extension API as the *only* stable contract, and keep codegen out
   of it.** History's verdict: numba's user-level contracts (`@jit`, `@overload`
   typed-Python batteries) survived both the numba-cuda fork and the MLIR rewrite;
   everything that touched LLVM (`@intrinsic`, `@lower_builtin`, datamodel
   leaves) died twice — first forcing NVIDIA to vendor ~all of `numba.core`
   (~155k-line copy), then breaking users in the numba-cuda-mlir migration. For
   pdum.dsl: extensions should compose typed DSL-level definitions; dropping to
   backend source (WGSL/CUDA snippets) must be possible but explicitly marked
   per-backend, never the default extension path.

8. **Two parallel registration namespaces (typing vs lowering) are avoidable
   duplication.** Numba needs `ConcreteTemplate` + `@lower_builtin` pairs kept in
   sync by hand wherever `@overload` doesn't reach. A from-scratch design should
   make the single-definition path (lesson 3) primary and make even backend
   primitives one object carrying both a type rule and per-backend emission,
   keyed by the same target tokens.

---

### Sources

- numba source @ `faf66df` (2026-07-10): <https://github.com/numba/numba> — files cited inline.
- numba-cuda source (2026-07-06): <https://github.com/NVIDIA/numba-cuda> (maintenance notice in README; vendored core under `numba_cuda/numba/cuda/core/`).
- numba-cuda-mlir: <https://github.com/NVIDIA/numba-cuda-mlir>.
- numba releases: <https://github.com/numba/numba/releases>, <https://pypi.org/project/numba/>.
- Release notes: [0.59.0 / Py3.12](https://numba.readthedocs.io/en/stable/release/0.59.0-notes.html), [0.61.0 / Py3.13](https://numba.readthedocs.io/en/stable/release/0.61.0-notes.html), [0.63.0 / Py3.14](https://numba.readthedocs.io/en/latest/release/0.63.0-notes.html).
- Version-support issues: [#9413 Py3.13](https://github.com/numba/numba/issues/9413), [#9957 Py3.14](https://github.com/numba/numba/issues/9957), [#10217 Py3.14 CI](https://github.com/numba/numba/issues/10217), [#10101 3.13.4 comprehension breakage](https://github.com/numba/numba/issues/10101), [#9883 Py>3.13 support discussion](https://github.com/numba/numba/issues/9883).
- numba-cuda releases (v0.30.4, 2026-07-03): <https://github.com/NVIDIA/numba-cuda/releases>.
- CUDA 13.3 / numba-cuda-mlir coverage: <https://www.phoronix.com/news/NVIDIA-CUDA-13.3-Released>.
