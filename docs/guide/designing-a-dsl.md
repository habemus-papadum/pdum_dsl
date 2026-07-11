# Designing a DSL

This guide is for using `pdum.dsl` as a **framework** — to build a DSL that is *not*
WebGPU shaders. Maybe you want to target the CPU, generate C or CUDA, drive real-time
audio, or just experiment with compiler ideas. The reusable core gives you closure
capture, a structural type system, a type-keyed cache with live-coding generations, a
typed IR, and monomorphic inlining. You supply a frontend dialect (intrinsics + a type
discipline) and a backend (an emitter + a runtime).

!!! info
    The WebGPU backend is the worked reference implementation; read it alongside this
    guide ([`src/pdum/dsl_reference/backends/wgsl/`](https://github.com/habemus-papadum/pdum_dsl/tree/main/src/pdum/dsl_reference/backends/wgsl)
    and [`webgpu/runtime.py`](https://github.com/habemus-papadum/pdum_dsl/blob/main/src/pdum/dsl_reference/webgpu/runtime.py)).
    The alternate-backend code in this guide is an **illustrative sketch** — it is not in
    the repo — included to show the contracts you implement.

## The split

```
            reusable core (backend-independent)         you provide
  ┌───────────────────────────────────────────────┐  ┌──────────────────────┐
  jit.py        capture: @jit, Handle, Program        │  intrinsics / dialect │
  types.py      Type lattice + typeof                  │  emitter: IR -> target│
  cache.py      SpecCache + generation                 │  runtime: run artifact│
  ir.py         typed IR nodes                          └──────────────────────┘
  frontend/     ast_lower: source -> IR
  passes/       infer (types), inline (monomorphize)
```

What you reuse unchanged: **capture**, the **cache**, **inlining**, and the **IR**. What
you adapt: the **type rules** and **intrinsics** (the dialect), and you write a new
**emitter** and **runtime**.

## The reusable pieces and their contracts

### Capture — `jit.py`

`@jit(kind=...)` is backend-agnostic; `kind` is just a string your backend interprets
(WGSL uses `"fragment"`/`"device"`). [`make_handle`](../reference/core.md) reads the
code object and closure cells and returns a `Handle` carrying:

- `fntype` — `FnType(code_object, env_types)`, the structural identity used as the cache
  key. Code objects compare *by value*, so an unchanged re-run hits and an edit misses.
- `env` — the captured *values* (never in the key).
- `source`, `pyfunc` — for lowering.

Calling a `Handle` builds a `Program` (a deferred application), so higher-order DSLs work
out of the box. You generally don't touch this layer.

### The type system — `types.py`

Types are frozen, hashable, structural values (so they drop straight into a cache key).
`typeof(value)` maps a runtime Python value to a `Type`. To extend the lattice:

1. Add a frozen `Type` subclass (e.g. a `FixedType(bits, frac)` for fixed-point audio).
2. Add a `typeof` case mapping the Python value to it. **Order matters** (e.g. `bool`
   before `int`); be honest about ranges (the int case buckets `i64`/`u64`/error).
3. Teach your backend how to lay it out / emit it.

`typeof` is on the hot path *and* defines correctness: too coarse a type silently reuses
the wrong specialization. See [Type System](../theory/type-system.md).

### The cache — `cache.py`

[`SpecCache`](../reference/core.md) is **generic over the artifact type**. It never
inspects what it stores; on a miss it calls your `compile_fn`:

```python
artifact = cache.get_or_compile(fntype, arg_types, compile_fn)
```

The key is `(fntype, arg_types, current_generation())`. The artifact is whatever your
`compile_fn` returns — a WGSL pipeline, a CPU function pointer, a compiled C module.
`compile_count`/`hit_count` are exposed for assertions, and a per-key future makes
concurrent misses compile once.

`generation` is a global counter folded into every key. `bump_generation()` invalidates
everything (the "sledgehammer"; precise dependency-graph invalidation is planned). Most
live-coding invalidation is automatic: editing a body yields a value-unequal code object
→ a new `FnType` → a natural miss.

### The IR — `ir.py`

A small typed expression/statement tree: `Lit`, `Name` (scope `uniform`/`local`/`arg`),
`Intrinsic`, `Swizzle`, `Unary`, `BinOp`, `Compare`, `Select`, `MakeVec`, `Call`; and
statements `Let`, `Return`, wrapped in a `Function`. Nodes carry a `type` filled in by
inference. It is mostly neutral; `Swizzle`/`MakeVec` lean vector-ish but are harmless for
scalar DSLs.

### The frontend — `frontend/ast_lower.py`

The `Lowerer` turns source into IR and **is your syntax seam**. It classifies each name
as `uniform` (a free variable), `arg` (a parameter), or `local` (assigned), and maps
`builtins.<X>` attribute chains to `Intrinsic` nodes. Restrict or widen the accepted
Python subset here.

!!! warning "Coupling to factor out"
    Today the lowerer imports `INTRINSIC_NAMES` and the inference pass imports
    `BUILTIN_CALLS`/`INTRINSIC_WGSL` from `backends.wgsl.intrinsics`. That is a temporary
    core→backend dependency. When you add a second backend, move the intrinsic/builtin
    tables into a backend-supplied "dialect" object passed into the lowerer/inference.
    This is a known refactor (see [WGSL Backend](../theory/wgsl-backend.md)).

### Inference — `passes/infer.py`

A bottom-up pass assigning a `Type` to every IR node from the uniform (env) types and any
argument types. It encodes promotion (the WGSL pass promotes mixed int/float to float).
Add type rules for your dialect's operations and builtins here.

### Inlining — `passes/inline.py`

`flatten(program)` inlines every device call into one `Function`, merges all captured
uniforms into one namespace (with stable, deterministic names), and resolves higher-order
arguments. Because the design is **fully monomorphizing**, each call site has exactly one
target, so this is pure substitution and there is no dynamic dispatch. It returns a
`Flattened(fn, names, types, values)`:

- `fn` — the inlined IR (use it at compile time);
- `names`/`types` — the merged uniform layout *structure* (stable across frames);
- `values` — the *current* captured values (re-collected each call).

You reuse this unchanged.

## Anatomy of a backend

A backend is two things: an **emitter** and a **runtime**.

### 1. The emitter: typed IR → target

Walk the inlined, typed `Function` and produce your target representation. The WGSL
emitter is a pure `IR -> str` function ([`emit.py`](../reference/wgsl.md)) plus a memory
layout ([`layout.py`](../reference/wgsl.md)). Keep the emitter pure (no device/runtime
calls) so you can golden-test it without hardware.

### 2. The runtime: compile + execute

A small driver that, per invocation:

1. `flat = flatten(program)` — inline + collect current captured values.
2. `artifact = cache.get_or_compile(entry.fntype, program.arg_types, lambda: build(flat))`
   — `build` emits the target and turns it into something executable (your expensive
   step, cached).
3. Feed `flat.values` into the artifact's input mechanism (a uniform buffer for WGSL;
   function arguments or a memory block for a CPU/native backend).
4. Execute.

The WGSL runtime is `Drawer.update` + `Drawer.show` in
[`webgpu/runtime.py`](../reference/runtime.md).

### Illustrative sketch: a CPU "evaluator" backend

This is **not in the repo** — it shows the contract end to end. The emitter compiles the
IR to a Python closure; the runtime caches it and calls it with the captured values.

```python
# illustrative only — sketch of a non-WGSL backend
from pdum.dsl_reference import SpecCache
from pdum.dsl_reference.passes.inline import flatten
from pdum.dsl_reference.passes.infer import infer_function
from pdum.dsl_reference import ir

def emit_python(flat):
    """typed IR -> a Python function (uniforms passed as a dict `u`)."""
    def ev(node, env):
        match node:
            case ir.Lit(): return node.value
            case ir.Name(scope="uniform"): return env[node.name]
            case ir.Name(scope="local"):  return env[node.name]
            case ir.BinOp():
                a, b = ev(node.left, env), ev(node.right, env)
                return {"+": a+b, "-": a-b, "*": a*b, "/": a/b}[node.op]
            case ir.Select():
                return ev(node.if_true, env) if ev(node.cond, env) else ev(node.if_false, env)
            # ... other node kinds ...
    def run(uniforms):
        env = dict(uniforms)
        for stmt in flat.fn.body:
            if isinstance(stmt, ir.Let):    env[stmt.name] = ev(stmt.value, env)
            if isinstance(stmt, ir.Return): return ev(stmt.value, env)
    return run

class CpuRuntime:
    def __init__(self):
        self.cache = SpecCache()
    def call(self, program):
        flat = flatten(program)
        infer_function(flat.fn, {n: flat.types[n] for n in flat.names})  # if your emit needs types
        fn = self.cache.get_or_compile(
            program.entry.fntype, program.arg_types, lambda: emit_python(flat)
        )
        return fn(flat.values)          # values change per call; fn is reused
```

The shape is identical to WGSL: `flatten` → `get_or_compile` → feed values → execute.
Only the artifact (a Python closure vs. a GPU pipeline) and the input mechanism (a dict
vs. a uniform buffer) differ.

## What to key the cache on

`(FnType, arg_types, generation)` is the baseline. **Add any backend parameter that
changes the emitted artifact** to the key. For WGSL the target texture format is such a
parameter; today it is *not* in the key (a known limitation — one `Drawer` per format).
For a native backend, the CPU target (`march`) and toolchain version belong in the key,
and in an on-disk cache as well (planned).

## Extension points

- **Value-dependent specialization** (`Literal`/`Val`, *planned*): let a selected capture
  be lifted *into* the type so its value participates in the key and gets constant-folded.
  Default stays "captures are runtime data"; opt in per capture. For WGSL this is the
  "bake as a `const`" vs. "uniform" switch.
- **User-defined structs** (`@gpu_struct`, *planned*): register a dataclass as an
  *isbits* compound type with a known field layout — modeled on numba's `Record` /
  `structref`. This is what makes compound captures and varyings ergonomic.
- **Fingerprints** (*planned*): a structural fast-path key to avoid a full `typeof` per
  call on the hottest paths.
- **A neutral dialect object**: factor intrinsics/builtins out of the WGSL backend so the
  frontend/inference take them as input (removes the coupling noted above).
- **Precise invalidation** (*planned*): replace the global `generation` sledgehammer with
  a dependency graph (Julia-style world age).

## Where to look in the code

| Concern | Module |
|---|---|
| Capture, handles, programs | `src/pdum/dsl_reference/jit.py` |
| Types + `typeof` | `src/pdum/dsl_reference/types.py` |
| Cache + generation | `src/pdum/dsl_reference/cache.py` |
| IR | `src/pdum/dsl_reference/ir.py` |
| Source → IR | `src/pdum/dsl_reference/frontend/ast_lower.py` |
| Inference | `src/pdum/dsl_reference/passes/infer.py` |
| Inlining / monomorphization | `src/pdum/dsl_reference/passes/inline.py` |
| Reference backend (emit/layout) | `src/pdum/dsl_reference/backends/wgsl/` |
| Reference runtime | `src/pdum/dsl_reference/webgpu/runtime.py` |

Then read [Theory & Internals](../theory/overview.md) for the timing model and the design
rationale behind each contract.
