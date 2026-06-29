# Type System

The type system is the backend-independent core that the cache keys on. Types are
**frozen, hashable, structural values** so they go straight into a cache key. This page
describes [`types.py`](../reference/core.md) (the lattice + `typeof`) and
[`passes/infer.py`](../reference/frontend.md) (inference), and is explicit about the gap
between *types that exist in the lattice* and *types `typeof` can actually produce today*.

## The lattice (implemented)

```python
Type                       # base
IntType(bits, signed)      # repr: i32, u64, ...
FloatType(bits)            # repr: f32, f64
BoolType()                 # bool
NoneType()                 # none
VecType(elem, n)           # vec2<f32>, ...   (n in 2..4)
TupleType(elems)           # (i64, f64) — arity is part of the type
FnType(template, env_types)# Fn<qualname>(...)  — a DSL closure's structural type
```

Singletons for the common scalars are provided: `i32, u32, i64, u64, f32, f64, boolean,
none`. Equality and hashing are structural (frozen dataclasses), so `IntType(64, True) ==
IntType(64, True)` and the value is usable as a dict key.

## `typeof`: value → type

`typeof(value)` runs over every capture (phase A) and every argument (phase B). It is on
the hot path **and** it defines correctness — too coarse and the wrong specialization is
silently reused.

| Python value | `typeof` result | Note |
|---|---|---|
| `True` / `False` | `BoolType()` | checked **before** `int` (`bool` ⊂ `int`) |
| `int` in signed 64-bit range | `IntType(64, True)` | |
| `int` in `[2**63, 2**64)` | `IntType(64, False)` | overflows `i64` → `u64` |
| larger `int` | raises `BigIntError` | a captured `5` and `2**70` must not share a type |
| `float` | `FloatType(64)` | Python floats are doubles |
| `None` | `NoneType()` | |
| `tuple` | `TupleType(...)` | recursive, element-wise; arity is part of the type |
| a DSL `Handle` | its `FnType` | nested closure |

The lattice is **honest about Python**: a Python `int` is 64-bit and a `float` is 64-bit.
Narrowing to a backend's concrete widths (WGSL has only 32-bit scalars) is the backend's
job, not the type system's — see *Two type levels* below.

!!! warning "What `typeof` does *not* produce yet"
    `VecType` and `FnType` exist in the lattice and are fully handled by inference,
    layout, and emission — but `typeof` never **produces** a `VecType` (a Python tuple maps
    to `TupleType`, not `VecType`), and `narrow_type` (uniform layout) currently **rejects**
    `TupleType`. The practical consequence: **captured uniforms must be scalars today**.
    Vector/tuple/struct uniforms need a `typeof`/layout path that doesn't exist yet — this
    is the single most useful near-term extension (it unlocks `center=(cx, cy)` captures).

## Two type levels (a subtlety worth knowing)

There are two related but distinct type "worlds", and conflating them causes bugs:

1. **Capture types (honest, cache key).** `typeof` yields `i64`/`f64` for Python
   ints/floats. These form the `FnType` and the cache key. They are about *identity*: does
   this closure share a specialization with that one?
2. **WGSL types (narrowed, emission).** The uniform layout narrows `i64 → i32`, `f64 →
   f32`, `bool → u32` ([`narrow_type` in `layout.py`](../reference/wgsl.md)), and the
   inference pass works in this narrowed world — IR literals infer as `i32`/`f32`, uniform
   `Name`s take their *narrowed* type from `layout.uniform_types()`.

So the cache distinguishes `i64` from `u64` (correct identity), while the emitter sees
`i32` (what WGSL can store). Keep this split in mind when reading
[`compile.py`](../reference/wgsl.md): it builds the layout first (narrowing), then runs
inference against the narrowed uniform types.

## Inference

`infer_function(fn, uniform_types, arg_types)` walks the (already-inlined) IR bottom-up and
sets every node's `type`. The M0 rules:

- **`Lit`**: `bool → boolean`, `int → i32`, `float → f32`.
- **`Name`**: by scope — `uniform` from `uniform_types`, `arg` from `arg_types`, `local`
  from the types accumulated as `Let`s are processed in order.
- **`Intrinsic`**: from the dialect table (`frag_coord → vec4<f32>`).
- **`Swizzle`**: 1 component → the element scalar; N → `VecType(elem, N)`.
- **`BinOp`**: numeric promotion — if either side is a `VecType`, that vector; else if
  either is float, `f32`; else `i32`.
- **`Compare`**: always `boolean`.
- **`Select`**: the type of the true branch (branches are assumed to match).
- **`MakeVec`**: `VecType(f32, len)` — vectors are float in M0.
- **`Call`**: the result rule from the builtin table (`sqrt → f32`, `length → f32`,
  `min/max → first arg`, …).

!!! note "Planned"
    A real inference pass with: union/`Optional` types once any path yields `None`;
    integer vectors and matrices; proper coercion (not just int→`f32` in a float context);
    and inference *through* `arg` types for non-inlined polymorphic call sites (not needed
    while everything monomorphizes).

## User-defined structs (planned)

The motivating examples use compound values — `RGB`, `Point` — the way numba exposes
`isbits` records. The plan: a `@gpu_struct` registration over a dataclass that yields a
compound `Type` with a known field layout, modeled on numba's `types.Record` (from a
structured dtype) and the `structref` extension. This is what makes compound captures and
(later) varyings ergonomic. Numba is **not** a runtime dependency; interop with numba
types is a separate, future backend concern.

## Value-dependent specialization (planned)

"Key on types, not values" is only universally safe with no value-dependent types. Real
kernels sometimes want a captured value as a compile-time constant — an array length that
enables a static loop bound, a flag that selects a branch. The plan is an explicit opt-in
marker (`Literal[v]` / `Val{v}`, as in numba/Julia) that lifts a *selected* capture into
the type, so its value participates in the key and gets constant-folded. Everything
unmarked stays runtime uniform data. For WGSL this is precisely the "bake as a `const`
(recompile per value)" vs. "it's a uniform (free to change)" switch.

## Where to look

| Concern | Symbol |
|---|---|
| Lattice + `typeof` | `pdum.dsl.types` |
| Narrowing to WGSL widths | `pdum.dsl.backends.wgsl.layout.narrow_type` |
| Inference | `pdum.dsl.passes.infer.infer_function` |
| Builtin/intrinsic type rules | `pdum.dsl.backends.wgsl.intrinsics` |

The full target lattice (arrays with layout/byteorder, `Optional`/union, `str`/`bytes`,
opaque callables) is specified in
[`design/dsl_caching_layer.md`](https://github.com/habemus-papadum/pdum_dsl/blob/main/design/dsl_caching_layer.md);
this implementation covers the subset above.
