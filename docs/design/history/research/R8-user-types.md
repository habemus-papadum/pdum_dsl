# R8 — User-defined types in embedded DSLs: records, methods, and physical units

*Research memo for the pdum.dsl redesign. July 2026. All project states verified
against live sources this month; citations inline.*

Scope: (1) how records/structs with methods are done in numba and what the
GPU-side ABI constraints are (WGSL/CUDA/Metal); (2) units-of-measure prior art
across compile-time (F#, mp-units, uom, Unitful.jl) and runtime (Pint, unyt,
astropy) systems, plus the JAX attempts (jpu, unxt); (3) where units should live
relative to the type-keyed cache; (4) how method-call syntax can be a
registration protocol rather than a compiler feature.

Baseline assumed from the repo: types are frozen dataclasses used directly as
cache-key components (`src/pdum/dsl_reference/types.py`), specialization keys on
`(FnType, arg_types, generation)` and never on values
(`design/dsl_caching_layer.md`), and marshaling ("one logical value → N physical
parameters") is a first-class seam (`docs/desiderata.md` §1, §7.6).

---

## 1. Records/structs on accelerators

### 1.1 numba Record dtypes: a structural type built from NumPy dtypes

numba's structured-array support (current release: numba 0.66.0, 2026-07-01,
Python 3.10–3.14 — <https://pypi.org/project/numba/>) is built on
`numba.core.types.npytypes.Record`. The mechanics that matter for us
(<https://github.com/numba/numba/blob/main/numba/core/types/npytypes.py>):

- **Identity is structural, encoded as a string.** The type's `name` is built as
  `Record({fields};{size};{aligned})` where each field contributes
  `(type, offset[, alignment][, title])`, and `key` (numba's equality/hash
  basis) is just that name. Two records with the same fields, offsets, itemsize
  and alignedness are *the same type* regardless of where the dtype object came
  from — deliberately sidestepping NumPy's dtype-equality warts (numpy #5715).
  This is exactly the property a type-keyed cache needs: the byte layout **is**
  the identity.
- **Offsets are explicit, not recomputed.** Fields are stored as
  `{name: _RecordField(type, offset, alignment, title)}`; `Record.offset(name)`
  reads it back. The layout is decided once, at type construction, from the
  NumPy dtype (including `align=True` padding) or via `Record.make_c_struct`,
  which walks `(name, type)` pairs applying C ABI alignment
  (`offset += align - misaligned`) — i.e. numba contains a small, reusable
  "layout planner" per ABI.
- **Field access lowering** is a getattr on the record type resolved by the
  typing phase to the field's type, then lowered to a
  load/store at `base_ptr + offset`. In user code both `rec.field` and
  `rec['field']` work; field names must be compile-time constants
  (<https://numba.readthedocs.io/en/stable/reference/numpysupported.html>).
- **Limits worth knowing:** structured scalars may not contain other structured
  scalars (nested *arrays* are allowed via `NestedArray`, which carries its
  shape in the type); attribute access on record scalars is flagged for
  eventual deprecation; there is experimental **width subtyping**
  (`[('a','f8'),('b','i8')]` usable where `[('a','f8')]` is expected) — a
  cache-sharing trick: one compiled artifact serves any record whose prefix
  matches.

### 1.2 numba structref: nominal user types with methods, in 400 lines

`numba.experimental.structref` is the modern way to give users a mutable,
pass-by-reference struct with methods. The whole module is **400 lines**
(verified against `main`, 2026-07:
<https://github.com/numba/numba/blob/main/numba/experimental/structref.py>), and
its API surface is five functions:

```python
@structref.register                 # (1) registers the data model for the type class
class ColorType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(t)) for name, t in fields)

class Color(structref.StructRefProxy):   # (2) Python-side proxy object
    def __new__(cls, r, g, b):
        return structref.StructRefProxy.__new__(cls, r, g, b)

structref.define_proxy(Color, ColorType, ["r", "g", "b"])  # (3) ctor + boxing + attrs
```

Internally `define_proxy` = `define_constructor` + `define_attributes` +
`define_boxing`; `register` installs a `StructModel`-style data model so the
payload lowers to an LLVM struct. Methods are then attached **from outside the
compiler**, per type, with `@overload_method` (§4). Notable caveat: structrefs
currently defeat numba's on-disk cache
(<https://github.com/numba/numba/issues/10021>) — their types don't serialize —
a reminder that user-type identity must be designed for the disk key, not just
the in-process key (`dsl_caching_layer.md` already flags this for `FnType`).

The **low-level path** (the "Interval" tutorial,
<https://numba.readthedocs.io/en/stable/extending/interval-example.html>) shows
what a from-scratch type registration costs — ten distinct registrations:
`types.Type` subclass → `typeof_impl.register` (value→type) →
`as_numba_type.register` (annotation→type) → `type_callable` (constructor
typing) → `register_model` (`models.StructModel` with `(name, type)` members) →
`make_attribute_wrapper` per field → `overload_attribute` for computed
properties → `lower_builtin` for the constructor → `unbox` → `box`. It's ~150
lines per type by hand, which is precisely why structref (which automates 8 of
the 10) exists. **Lesson: design the automated path first;** numba grew it
second and the seams show.

### 1.3 GPU struct ABI: three backends, three layouts for the same struct

The same logical `Color { r: f32, g: f32, b: f32 }` (or anything with a
3-vector) has *different* sizes/alignments per backend — the single strongest
argument that layout planning belongs in the backend-owned marshaling layer:

| Type | WGSL (storage) | WGSL (uniform) | CUDA C++ | Metal MSL |
|---|---|---|---|---|
| `f32` | align 4, size 4 | align 4 | align 4, size 4 | align 4, size 4 |
| 3-vector of f32 | **align 16, size 12** | align 16, size 12 | `float3`: **align 4, size 12** (built-in; user `__align__(16)` optional) | `float3`: **align 16, size 16**; `packed_float3`: align 4, size 12 |
| 4-vector of f32 | align 16, size 16 | align 16, size 16 | `float4`: align 16, size 16 | align 16, size 16 |
| struct | align = max member align; size rounded up to align | member align additionally **rounded up to 16** for nested structs; array stride multiple of 16 | C++ rules (+ `__align__`) | C++ rules |

Sources: WGSL memory-layout rules §14.4 incl. uniform-address-space extra
constraints (<https://www.w3.org/TR/WGSL/#memory-layouts>, corroborated by
<https://webgpufundamentals.org/webgpu/lessons/webgpu-memory-layout.html> —
offset formula `roundUp(AlignOf(member), offsetPrev + sizePrev)`); CUDA
alignment (<https://leimao.github.io/blog/CUDA-Data-Alignment/>,
<https://forums.developer.nvidia.com/t/own-float3-and-float4/30236>); Metal
(<https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>,
<https://developer.apple.com/forums/thread/98020>).

Mechanically, per backend:

- **WebGPU/WGSL**: a captured struct becomes a (slice of a) uniform buffer. The
  emitter must compute WGSL offsets (with the uniform 16-byte rounding) and the
  Python-side pack function must write bytes at *those* offsets. This is the
  generalization of the reference asset's scalar-only uniform packing
  (`webgpu/runtime.py`); the missing piece flagged in desiderata §7 ("capture→
  layout path for vectors/tuples/structs doesn't exist") is exactly a
  `LayoutPlan: [(name, offset, size, encode_fn)]` computed from the struct
  type. WGSL has **no methods and no operator overloading on user structs** —
  every method must be lowered to a free function (§1.4).
- **CUDA via CuPy RawKernel**: kernel arguments are passed by value; "the CUDA
  driver copies exactly sizeof(param_type) bytes from the NumPy object's data
  pointer — you must match size, alignment, padding by defining a corresponding
  NumPy dtype" (<https://docs.cupy.dev/en/stable/user_guide/kernel.html>,
  struct-by-value discussion in
  <https://github.com/cupy/cupy/issues/4828>). So the CUDA marshaling plan is
  literally "emit a C struct decl + build the matching `np.dtype` with explicit
  offsets" — numba's `Record.make_c_struct` is a working model of the
  offset computation. CUDA C++ *does* allow `__device__` member functions, but
  targeting the free-function lowering everywhere keeps one strategy.
- **Metal via MLX custom kernels**: MSL is C++-based, same struct rules except
  `float3` is 16/16 (unlike CUDA); prefer `packed_float3` for tight buffers.
  Again free functions suffice.

### 1.4 Worked lowering: `Color.to_oklab()`

How a method-bearing struct flows through a numba-like pipeline, adapted to the
pdum.dsl reference architecture:

```python
@dsl.struct                    # frontend-only registration, no backend knowledge
class Color:
    r: f32; g: f32; b: f32

@dsl.method(Color, "to_oklab")     # body is ordinary DSL code, compiled like any fn
def to_oklab(self) -> Vec3:
    l = 0.4122*self.r + 0.5363*self.g + 0.0514*self.b
    ...
    return vec3(cbrt(l), ...)
```

1. **typeof**: a captured `Color(0.1, 0.2, 0.3)` maps to
   `StructType(tag='Color', fields=(('r',f32),('g',f32),('b',f32)))` — a frozen
   dataclass, hashable, straight into `env_types` → the cache key. Values never
   enter the key (same rule as today).
2. **typing**: `c.to_oklab()` hits the generic rule "method call on a type with
   a method table entry" → look up `(StructType tag/shape, 'to_oklab')` in the
   registry → the registered DSL function; type its body with `self:
   StructType(...)` bound (monomorphic, so this is just the existing inference
   over one more function).
3. **inlining**: the reference asset already monomorphically inlines closures
   (`passes/inline.py`); a method call is the same transform with `self`
   substituted — no new IR concept.
4. **WGSL emission**: struct decl + free function; the emitter's only new job
   is the layout table:

```wgsl
struct Color { r: f32, g: f32, b: f32 };            // align 4 here; but as a
// uniform member inside the packed Uniforms struct: offset rounded to 16
fn Color_to_oklab(self_: Color) -> vec3<f32> { ... } // methods erased to free fns
```

5. **marshaling**: capture of a `Color` value contributes 3 slots to the
   uniform `LayoutPlan` (or 1 slot of 12 bytes at a 16-aligned offset).
   Per-frame update = `memcpy` of 12 bytes at a known offset — hot-loop cost is
   a value write, as required.

---

## 2. Units of measure: prior art

### 2.1 F# units of measure — erased, checked, **no auto-conversion**

(<https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/units-of-measure>)

- `[<Measure>] type cm`, values written `55.0<miles/hour>`; the compiler
  normalizes unit *formulas* to a canonical form (single numerator/denominator,
  alphabetized: `kg m s^-2` ≡ `m /s s * kg` → `kg m/s^2`) — canonicalization of
  the **dimension expression**, not of values.
- **Units are fully erased at runtime** ("compiled … the units of measure are
  eliminated, so the units are lost at run time") — zero cost, but no runtime
  reflection (`ToString` of a unit is impossible).
- **Crucially: F# does *not* auto-convert.** `float<cm> + float<m>` is a type
  error; users define conversion *constants* (`let cmPerMeter : float<cm/m> =
  100.0<cm/m>`) and multiply explicitly. Derived units (`type N = kg m / s^2`)
  are definitional equalities, not scaled conversions. F# answers "auto-convert
  or reject?" with **reject** — safe, but exactly the ergonomics pdum.dsl's
  desiderata §4.4 says it wants to improve on ("auto-converted by the
  compiler").

### 2.2 mp-units (C++) — unit stays in the type; conversion at operations

(<https://github.com/mpusz/mp-units>, C++29 standardization candidate;
<https://mpusz.github.io/mp-units/HEAD/users_guide/framework_basics/value_conversions/>,
<https://mpusz.github.io/mp-units/HEAD/users_guide/framework_basics/quantity_arithmetics/>)

- Values are stored **in the declared unit**, *not* normalized to base — the
  type is `quantity<Reference, Rep>` where the reference carries the unit.
- Same-kind, different-unit arithmetic **auto-converts to the common unit** at
  the operation: "the result … will be the common type of the arguments";
  `1 * km + 1.5 * m == 1001.5 * m` (result in the finer unit, so it's exact).
- Conversions are implicit only when **value-preserving** (floating rep, or
  integral rep going to a finer unit); lossy directions require explicit
  `value_cast<U>()` / `.force_in(U)`; potentially-overflowing scalings are
  compile errors even for zero values (`int8_t(1)*km` → m rejected).
  Truncation, when forced, is toward zero, with fixed-point intermediate math
  to stay within 1 ulp.
- Distinguishes `quantity` vs `quantity_point` (affine: temperatures,
  timestamps) — the offset-unit problem is solved by a *separate type*, not by
  smarter conversion.

### 2.3 Rust uom — canonicalize at construction, erase everywhere else

(<https://docs.rs/uom/latest/uom/>)

- `Quantity<D, U, V>` where `U` is the *system's base-unit set*; **every value
  is normalized to the base unit at construction**:
  `Length::new::<millimeter>(3.0)` stores `0.003` (meters). After the boundary,
  all arithmetic is raw `f32`/`f64` ops — "zero runtime cost over the raw
  storage type."
- The cost of this choice: representation error for non-float reps (their
  `autoconvert` feature exists because integral reps aren't zero-cost), and
  values printed/debugged are in base units unless re-converted out.
- uom is the clean industrial example of **"canonicalize at the boundary,
  dimension-only inside"** — the strategy that maps onto a marshaling layer.

### 2.4 Unitful.jl — units in the type, under a type-keyed JIT (the direct analog)

(<https://github.com/PainterQubits/Unitful.jl/blob/master/docs/src/types.md>,
<https://github.com/PainterQubits/Unitful.jl/blob/master/docs/src/conversion.md>)

Julia is pdum.dsl's specialization model, and Unitful is what "units in the
type" does to it:

- `Quantity{T<:Number, D, U}` — **dimension `D` and units `U` are type
  parameters**. Every distinct unit is a distinct type; Julia specializes
  (recompiles) per unit type. "By putting units in the type signature …
  staged functions can offload as much of the unit computation to compile time
  as possible" — unit factors constant-fold to nothing in hot code.
- Mixed-unit `+`/`-` **auto-promotes via `promote_unit`, defaulting to SI base
  units** (`𝐌·𝐋²/(𝐓²·𝚯)` → `kg·m²/(s²·K)`); users can override the preferred
  unit per dimension. `uconvert` is the explicit form.
- The documented failure mode is **type instability**: arrays mixing units
  promote or fall back to abstract element types "which cannot be stored
  efficiently and will incur a performance penalty" — i.e. units-in-type
  multiplies the number of concrete types flowing through a type-keyed
  compiler, and every new unit combination is a new specialization.

### 2.5 Python runtime libraries — values carry units, incompatible with JIT

- **Pint / unyt / astropy.units** all attach a unit object to a value/array at
  *runtime* (wrapper or ndarray subclass), convert on demand
  (`q.to(u)`, `to_base_units()`), and are unusable inside numba nopython code —
  ndarray subclasses/wrappers don't type
  (<https://github.com/numba/numba/issues/5827>); the standard workaround is to
  strip to `.magnitude` at the jit boundary and re-attach after. That
  "strip/convert at the boundary" folk practice is, in effect, a manual version
  of the marshaling-layer placement in §3.
- **jpu** (JAX + Pint, <https://github.com/dfm/jpu>): unit propagation happens
  **at trace time**, so "jitted functions should see no runtime cost"; but it
  requires `jpu.numpy` wrappers (JAX has no ufunc-dispatch hook for foreign
  arrays), implements only a subset, and is explicitly experimental — v0.0.5,
  2025-04-26, 58 commits. Status: proof-of-concept, low maintenance.
- **unxt** (<https://github.com/GalacticDynamics/unxt>; JOSS/arXiv paper
  2026-03, <https://arxiv.org/abs/2603.08770>): the current serious attempt.
  Built on quax (array-ish objects inside JAX transforms) with astropy.units as
  the unit backend; works through `jit`/`grad`/`vmap`. The load-bearing detail
  (verified in source,
  <https://github.com/GalacticDynamics/unxt/blob/main/src/unxt/_src/quantity/quantity.py>):

  ```python
  class Quantity(...):
      value: Shaped[Array, "*shape"] = eqx.field(converter=...)
      unit:  AbstractUnit            = eqx.field(static=True, converter=parse_unit)
  ```

  The unit is **static pytree aux-data** — it is part of jit's cache key, while
  the value is traced. Changing a value: no retrace. Changing a unit: retrace +
  recompile. unxt also has first-class **dimensions** distinct from units, and
  **unit systems** (a chosen set of base units per domain) — the exact
  vocabulary pdum.dsl needs.

**Convergent finding:** every system that compiles (F#, mp-units, uom, Unitful,
jpu, unxt) keeps units on the **static/type side and erases them from runtime
data**; none pays a runtime tag. They differ only in *where values get
re-scaled*: at construction (uom), at each mixed operation (mp-units, Unitful
promotion), or never/explicit-only (F#).

---

## 3. Placement: units in the Type, at the marshaling boundary, or both

The question from desiderata §7.7, decided against two hard constraints: the
cache keys on types-not-values, and the hot loop costs a value write.

### 3.1 The three candidate placements

**(A) Full unit in the Type** (Unitful/unxt model).
`typeof(3.0*mm) = f64<mm>`; kernels compile with the capture's concrete unit;
conversion factors constant-fold into the artifact.
- ✅ zero marshaling arithmetic; exact for oddball reps.
- ❌ *the artifact is keyed on the unit*: a user who retypes a slider from mm to
  inch triggers a *shader recompile* mid-interaction — a value-adjacent change
  producing a compile, which violates the spirit (if not the letter) of
  types-not-values. Unit combinatorics multiply specializations (Unitful's
  documented instability problem). Cross-closure sharing shrinks: `closure(3*mm)`
  and `closure(3*inch)` stop sharing an `FnType`.

**(B) Dimension in the Type, unit at the marshaling boundary** (uom model,
lifted to the JIT boundary).
`typeof(3.0*mm) = f64<L>` (dimension only — a rational-exponent vector over the
base dimensions). The kernel is compiled against a **canonical basis** (SI base
units, or a per-kernel/user-declared unit system, unxt-style). Marshaling
converts on pack: the pack plan for a capture of declared unit `mm` bakes in
`×1e-3`.
- ✅ cache key = dimension: `3.0*mm` and `3.0*inch` share one artifact;
  unit changes cost a pack-plan rebuild (one Python-level multiply baked into a
  closure), never a compile. Hot loop: multiply-then-write ≈ a value write.
- ✅ unit *checking* still happens at compile time — dimensions are in the type,
  so `length + time` is a compile error inside the kernel; unit *scaling* is a
  boundary concern, exactly where desiderata §4.4 asks for it ("at the level
  where it arranges the arguments passed to compiled functions").
- ❌ integral reps: canonicalizing `3*mm` into an `i32`-meters slot is
  catastrophic (0). Rule: value-preserving conversions only (float reps
  convert freely; integer reps require exact factors or an explicit cast) —
  mp-units' implicit/explicit split, applied at pack time.
- ❌ debugging shows canonical values, not the user's units (uom's known wart) —
  mitigated because the *handle* still knows the declared unit; only the device
  sees canonical.

**(C) Both** — dimension in the artifact key, unit in a second, cheaper key.

### 3.2 Recommendation: (B) with an explicit escape hatch — i.e. a disciplined (C)

Concretely:

- **Types carry dimension, not unit.** `Quantity(f64, dim=Dim(L=1))` is the
  frozen-dataclass type; `Dim` is a hashable vector of `Fraction` exponents
  over the 7 SI base dimensions (+ angle as an 8th, pragmatic for
  graphics/audio). This drops into `env_types`/`arg_types` unchanged, so the
  **native-artifact cache key** contains dimensions only.
- **The pack plan carries the unit.** The existing "env layout" memo
  (`dsl_caching_layer.md` — a side table keyed by `FnType`, not a field of it)
  gains a per-slot converter. Key the pack-plan memo on
  `(FnType, arg_types, concrete_units)`: a *second-tier* key that includes
  units is fine because a pack-plan miss costs microseconds of Python (build a
  new list of `(offset, factor, encode)`), not a backend compile. This is the
  "both" in (C), with the expensive cache and the cheap cache keyed
  differently.
- **Literals inside kernel bodies fold at compile time.** `3*mm` written in DSL
  source is a compile-time constant; the frontend folds it to canonical
  `0.003` during lowering. Only *captured/argument* quantities go through the
  pack-time multiply.
- **Affine units** (°C, °F, dB re …): pack-time `scale*x + offset` handles
  marshaling, but in-kernel arithmetic needs the point/delta distinction —
  adopt mp-units' `quantity` vs `quantity_point` split as two dimension-carrying
  types when units ship; do not special-case temperature.
- **Escape hatch = the existing `Literal`/`Val` lift.** If a user wants the
  factor constant-folded into the artifact (e.g., an exactness argument, or an
  integer-rep kernel), they mark the capture, and the *unit* (or the value)
  enters the type explicitly — reusing the value-dependent-specialization
  door the caching design already has, instead of a units-specific mechanism.

### 3.3 Worked example: capturing `3.0*mm` where the kernel works in meters

```python
wavelength = 3.0 * mm                     # Quantity(value=3.0, unit=mm) — plain Python object

@jit
def shade(uv):                            # captures `wavelength`
    return sin(uv.x / wavelength)         # typing: uv.x [L] / wavelength [L] → dimensionless ✔

# Phase A (capture): typeof(3.0*mm) → Quantity(f64, dim=L). env_types = (Quantity(f64,L),).
#   FnType = (code, (Quantity(f64,L),)). No compile. No unit in the key.
# Phase B (first call): compile once for this FnType. Kernel is emitted in the
#   canonical basis: the uniform slot is "wavelength, f64→f32, meters".
#   Pack plan slot: (offset=16, factor=1e-3, cast=f32).
# Frame loop:
wavelength = 3.1 * mm    # → pack: write f32(3.1 * 1e-3) at offset 16. No recompile.
wavelength = 0.12 * inch # same dim → SAME artifact; pack-plan memo misses on the
                         # unit tier only → rebuild slot factor to 0.0254. µs, no compile.
wavelength = 3.0 * ms    # dim L→T: env_types differ → different FnType → honest
                         # compile-or-type-error, exactly like any type change today.
```

When it happens: dimension errors at phase-B typing (compile time); unit
scaling at pack time (per unit *change*, factor baked; per frame, one fused
multiply-write); nothing at kernel runtime.

---

## 4. Method-call syntax as a registration protocol

### 4.1 What numba actually does

`@overload_method(typ_class, "name")`
(<https://numba.readthedocs.io/en/stable/extending/high-level.html>) registers a
function that runs **at typing time**: it receives *types*, may inspect them,
and returns an *implementation function* — ordinary Python that numba then
compiles as if it were `@jit`ted (or returns `None` to let other overloads
try, giving ad-hoc polymorphism by chaining):

```python
@overload_method(ColorType, "to_oklab")
def ov_to_oklab(self):
    if not isinstance(self, ColorType): return None
    def impl(self):
        return _to_oklab_math(self.r, self.g, self.b)   # plain (DSL-compilable) code
    return impl
```

The deep idea: **the compiler core contains one generic rule** — "method call
on type T: consult the registry at (T-class, name); type and compile whatever
comes back, in the target language itself." All batteries (`@overload` for
free functions like `len`/`np.mean`, `@overload_attribute` for properties,
`@overload_classmethod`, `@intrinsic` only for the rare LLVM-level case) are
data in registries, not compiler code. This is also numba's answer to the
desiderata "batteries economics" question: batteries are DSL-level definitions
lowered through the same pipeline, so they cost one portable definition, not
one per backend.

### 4.2 The pdum.dsl shape (small)

```python
# core (one-time, ~80–120 lines): the protocol
MethodTable: dict[tuple[type[Type] | Type, str], DslFn] = {}

def method(typ, name):                       # public decorator, dialect-scoped table
    def reg(fn): MethodTable[(typ, name)] = to_dsl_fn(fn); return fn
    return reg

# typing pass: on Call(Attribute(obj, name)):
#   sig = MethodTable[(typeclass_of(typeof(obj)), name)]  — miss → nice error
#   type body with self=typeof(obj); monomorphic
# lowering: rewrite to a plain call fn(self, *args) → existing inline pass
# backend: nothing new; methods have been erased to free functions already
```

- **Everything stays type-keyed:** the method body is compiled per
  `(method_fn_code, self_type, arg_types)` through the *same* specialization
  cache as closures — a method is just a function whose first capture-like
  parameter is `self`. No new cache, no new key shape.
- **Structural vs nominal:** numba `Record` is structural (layout-string
  identity), structref is nominal (type-class identity), and methods hang off
  the *nominal* handle. Recommendation: `StructType(tag, fields)` with the
  `fields` tuple in `__eq__`/`__hash__` (cache correctness = layout) **and** a
  `tag` for method lookup and diagnostics; two tags with identical fields may
  share layout plans but not methods.
- **Extensibility check (the prime directive):** adding `Color.to_oklab()`
  touches: user/library file only (`@dsl.struct` + `@dsl.method`). Adding the
  `StructType` kind itself touches: types.py (one dataclass), the typing rule
  (once), each backend's layout planner (per-backend, unavoidable — §1.3). No
  core edits per new struct or per new method — matching how structref +
  overload_method live entirely outside numba's core.
- **Units connect here too:** `(3.0*mm).to(inch)` in DSL source, or
  `q.magnitude`, are just method-table entries on `Quantity` types — the units
  feature adds registry *rows*, not compiler *rules*.

---

## Design lessons for pdum.dsl

1. **Struct layout is a per-backend pack plan, not a property of the type.**
   The same 3-float struct is 12/16-layout in WGSL-uniform, 12/4 in CUDA, 16/16
   in Metal. Define `StructType(tag, fields)` dimension-of-truth in the core,
   and make each backend expose a `plan_layout(StructType) -> LayoutPlan`
   (`[(field, offset, size, encode)]`, plus total size/align). The reference
   asset's scalar uniform packer generalizes to exactly this; numba's
   `Record.make_c_struct` (~40 lines) is a copyable model of the offset
   computation.

2. **Keep byte layout inside the type's equality only where layout *is*
   identity.** numba keys `Record` on fields+offsets+size+aligned and that is
   what makes its cache sound. For pdum.dsl: `StructType.fields` (names+types,
   ordered) belongs in `__eq__`/`__hash__`; backend-computed offsets do NOT
   (they'd drag WGSL-vs-CUDA into the frontend key) — they live in the
   backend's layout memo, keyed by `(backend, StructType)`.

3. **Methods = one generic typing rule + a registry; bodies are DSL code.**
   Adopt the `@overload_method` architecture: `@dsl.method(Type, name)`
   registers a DSL-compilable function; the core's only knowledge is
   "getattr-call ⇒ registry lookup ⇒ inline as a free function." This is ~100
   core lines, gives library authors first-class methods with zero compiler
   edits, erases cleanly to WGSL (which has no methods), and doubles as the
   batteries mechanism (`@dsl.overload(np.mean)` later) — one answer to two
   desiderata questions (§4.3, §7.4).

4. **Units: dimension in the Type, unit in the pack plan, `Literal`-lift as
   the escape hatch.** Cache artifacts on `Quantity(rep, Dim)` (rational
   exponent vector, frozen, hashable); compile kernels in a canonical basis
   (SI base by default, unxt-style declarable unit systems later); bake the
   scale factor into the marshaling slot so a unit change rebuilds a pack plan
   (µs) and never recompiles a shader; a *dimension* change is an honest type
   change. This is the only placement consistent with both "types, not values"
   and "the loop stays hot" — full-unit-in-type (Unitful/unxt) provably causes
   recompiles on unit change and specialization blowup; runtime units
   (Pint et al.) provably don't survive a JIT boundary at all.

5. **Two-tier keying is legitimate: expensive cache coarse, cheap cache fine.**
   The artifact cache keys on dimensions; the pack-plan memo keys additionally
   on concrete units. Generalize this pattern: any datum that changes what
   *bytes get written where* (units, endianness, target texture format) but not
   *what code runs* belongs in the marshaling key, not the artifact key. This
   also cleanly files the desiderata's "two type levels" fault line: honest
   frontend types in the artifact key, backend-narrowed/laid-out forms in
   backend-owned side tables.

6. **Enforce value-preserving conversion rules at the pack boundary
   (mp-units' split).** Float reps: convert implicitly. Integer reps:
   implicit only when the factor is exactly representable; otherwise a
   compile-time error naming `value_cast`-style explicit syntax. Affine units
   are a separate point-type, not a smarter converter.

7. **Design the automated registration path first.** numba's 10-step manual
   type registration predates its 400-line structref automation, and the
   result is two APIs and folklore about which to use. pdum.dsl should ship
   `@dsl.struct` (dataclass-syntax → StructType + constructor + typeof + field
   access, in one decorator) as *the* path, keeping the underlying registries
   (typeof rule, method table, layout planner) public for the 1% who need to
   bypass it.

8. **Plan user-type identity for the disk cache now.** numba structrefs still
   defeat the on-disk cache (numba #10021). `StructType` must serialize
   structurally (tag + field names + field types), and method-table
   entries must contribute their source hash to the dependency-closure hash
   that `dsl_caching_layer.md` already specifies for the disk key —
   otherwise editing `to_oklab`'s body silently reuses stale kernels.

### Source index

- numba high-level extending API — <https://numba.readthedocs.io/en/stable/extending/high-level.html>
- numba Interval (low-level registration) — <https://numba.readthedocs.io/en/stable/extending/interval-example.html>
- numba structref source (400 lines) — <https://github.com/numba/numba/blob/main/numba/experimental/structref.py>
- numba Record/NestedArray source — <https://github.com/numba/numba/blob/main/numba/core/types/npytypes.py>
- numba structured-array support — <https://numba.readthedocs.io/en/stable/reference/numpysupported.html>
- numba 0.66.0 (2026-07-01) — <https://pypi.org/project/numba/>
- structref vs disk cache — <https://github.com/numba/numba/issues/10021>; ndarray subclasses unsupported — <https://github.com/numba/numba/issues/5827>
- WGSL memory layout — <https://www.w3.org/TR/WGSL/#memory-layouts>; <https://webgpufundamentals.org/webgpu/lessons/webgpu-memory-layout.html>
- CUDA alignment — <https://leimao.github.io/blog/CUDA-Data-Alignment/>; <https://forums.developer.nvidia.com/t/own-float3-and-float4/30236>
- CuPy RawKernel structs — <https://docs.cupy.dev/en/stable/user_guide/kernel.html>; <https://github.com/cupy/cupy/issues/4828>
- Metal Shading Language spec — <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>
- F# units of measure — <https://learn.microsoft.com/en-us/dotnet/fsharp/language-reference/units-of-measure>
- mp-units — <https://github.com/mpusz/mp-units>; value conversions — <https://mpusz.github.io/mp-units/HEAD/users_guide/framework_basics/value_conversions/>
- uom — <https://docs.rs/uom/latest/uom/>
- Unitful.jl types & conversion — <https://github.com/PainterQubits/Unitful.jl/blob/master/docs/src/types.md>; <https://github.com/PainterQubits/Unitful.jl/blob/master/docs/src/conversion.md>
- jpu — <https://github.com/dfm/jpu>
- unxt — <https://github.com/GalacticDynamics/unxt>; paper — <https://arxiv.org/abs/2603.08770>; Quantity source (unit is `eqx.field(static=True)`) — <https://github.com/GalacticDynamics/unxt/blob/main/src/unxt/_src/quantity/quantity.py>
