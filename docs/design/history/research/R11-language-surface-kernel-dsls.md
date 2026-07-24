# R11 — Python language subsets in GPU kernel DSLs (Triton / Taichi / cupy.jit)

*Research agent survey, 2026-07-12, verified against current official
docs/source. Construct-level companion to R5 (system-level analysis).
Consumed by the book's ch07a lay-of-the-land interlude.*

| Construct | Triton `@triton.jit` | Taichi `@ti.kernel`/`@ti.func` | cupy `@cupyx.jit.rawkernel` |
|---|---|---|---|
| `if`/`elif`/`else` | yes (constexpr cond → compile-time fold; dynamic → scf.if) | yes (`if ti.static(c)` → compile-time; dead branch discarded) | yes (constant cond folded at transpile time) |
| `for` | partial (only `range`/`tl.static_range`; error: "Only `range` and `static_range` iterators are currently supported"; hints: `num_stages`, `loop_unroll_factor`) | yes (range, `ti.ndrange`, struct-for over fields, `ti.grouped`, static-for `for i in ti.static(range(n))` unrolled; outermost range/ndrange/struct-for auto-parallelized; no `for…else`) | partial (`range()` only; `cupyx.jit.range()` for `#pragma unroll`; no `for…else`) |
| `while` | yes (no `else`; loop-carried vars must be triton values) | yes (serial only, never parallelized) | yes (no `else`) |
| ternary `x if c else y` | yes | yes | yes (branch types must unify) |
| `and`/`or` | partial (scalars only, short-circuit; non-scalar tensor → "Boolean value of Tensor with more than one value is ambiguous", use `&`/`\|`) | partial (default lowered to bitwise AND/OR, no short-circuit unless `short_circuit_operators=True`) | yes (left-to-right; chained compare `a<b<c` rejected: "Comparison of 3 or more values is not implemented" — Taichi allows chaining, Triton allows it on scalars) |
| tuples | yes (values, kernel args — flattened in ABI, returns; namedtuples too) | partial (multi-value return in `ti.func` single statement; vectors/matrices preferred) | yes (lowered to C++ tuple) |
| unpack assignment `a,b = …` | yes | yes (grammar: target may be `(target_list)`/`[target_list]`) | yes (single target only; `a = b = c` rejected) |
| aug-assign `+=` | yes (desugared to BinOp+Assign; cannot reassign constexpr in loop) | yes (on fields, `x[i] += v` is atomic) | yes (scalars in place) |
| early `return` | partial (allowed at top level / inside top-level `if`; error inside `for`/`while`/`with`) | no ("return … may only occur once … and it must be at the bottom of the function body") | yes (return types unified across paths) |
| comprehensions | partial ("only tuple comprehensions are supported", single generator, no nesting) | partial (list/dict comprehensions, compile-time evaluated) | no |
| `lambda` | no | no (not in language reference) | no as expression (but a lambda may itself be the decorated kernel — source recovered by line-number hunt) |
| nested `def` | no ("nested function definitions are not allowed … Move the helper function to module level") | no (undocumented/unsupported; helpers must be module-level `@ti.func`) | no ("Nested functions are not supported currently") |
| classes / structs | partial (no `ClassDef`; no user struct decl in stable API — tuples/namedtuples are the aggregates; attribute assignment rejected) | yes via Python-scope `@ti.dataclass` / `ti.types.struct` (see below); no `ClassDef` inside kernel | no ("class is not supported currently") |
| recursion | no (calls are inlined; only other `@triton.jit` fns callable) | no at runtime ("Runtime recursion is not allowed"); yes at compile time via `ti.static` guard + `ti.template()` arg | no (re-entry detected via placeholder, rejected) |
| globals | error unless constexpr ("Triton kernels can only access global variables that are instantiated as constexpr"; env override `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1`; value drift at launch raises) | frozen ("treats global variables as compile-time constants … does not track changes afterwards") | frozen silently (snapshotted via `getclosurevars` per signature; no invalidation) |
| kwargs | yes in calls (`visit_keyword`, `Starred` unpacking); kernel-def defaults baked in; no `*args`/`**kwargs` params | partial (struct constructors use kwargs; kernel/func params: no defaults, no `*args`/`**kwargs`) | no (`*args`/`**kwargs`/defaults/kw-only rejected in def; in calls only `dtype=` for ufuncs) |
| slicing / subscript | partial (load only incl. `None`/newaxis; "`__setitem__` is not supported in triton") | partial (field/matrix subscript read+write; matrix slicing needs static indices) | partial (element indexing on arrays/tuples; advanced indexing rejected) |
| other rejected (calls whitelist etc.) | callables must be JITFunction/triton builtins/constexpr fns; no `break`/`continue` (falls to "unsupported AST node type: {}"); no Try/ClassDef/Lambda | no `try`, `with`, `import`, `del` (not in language grammar); `break` illegal when nearest enclosing loop is the parallelized loop | callables: builtins, `cupyx.jit` device fns, ufuncs, casts; rejected with messages: `del`, `with` ("Switching contexts are not allowed"), `raise`/`try` ("throw/catch are not allowed"), `import`, `global`/`nonlocal`, `async` |

## Constexpr / template specialization syntax

- **Triton**: param annotation `BLOCK: tl.constexpr` (value becomes part of
  cache key, folds in `if`/arith); local `x: tl.constexpr = expr`;
  `tl.static_range`/plain `range` over constexpr bounds unrolls;
  `tl.static_assert`; opt-out of implicit value specialization via
  `do_not_specialize`.
- **Taichi**: `x: ti.template()` param (inlined at compile time, keyed by
  object; non-Taichi-object templates are read-only in kernel);
  `ti.static(...)` marks compile-time evaluation for `if`, `for … in
  ti.static(range(n))`, and recursion guards.
- **cupy.jit**: none — no constexpr annotation; specialization is implicit
  per argument-type signature (plus `index_32_bits` value leak);
  constant-operand expressions folded in Python at transpile time.

## Struct declaration syntax

- **Taichi** (two equivalent forms; inheritance unsupported; methods:
  `@ti.func` = Taichi scope, undecorated = Python scope):

  ```python
  @ti.dataclass
  class Sphere:
      center: vec3
      radius: ti.f32
      @ti.func
      def area(self): return 4 * math.pi * self.radius * self.radius

  Sphere = ti.types.struct(center=vec3, radius=ti.f32)
  s = Sphere(center=ti.math.vec3(0.0), radius=4.0)   # kwargs ctor, works inside kernels
  ```
- **Triton**: no user-facing struct declaration; aggregate args are Python
  tuples/namedtuples, flattened into the physical signature at launch.
- **cupy.jit**: none.

Sources: [triton.jit API](https://triton-lang.org/main/python-api/generated/triton.jit.html),
[triton code_generator.py](https://github.com/triton-lang/triton/blob/main/python/triton/compiler/code_generator.py),
[cupy user guide: kernels](https://docs.cupy.dev/en/stable/user_guide/kernel.html),
[cupyx/jit/_compile.py](https://github.com/cupy/cupy/blob/main/cupyx/jit/_compile.py),
[Taichi language reference](https://docs.taichi-lang.org/docs/language_reference),
[Taichi kernels/functions](https://docs.taichi-lang.org/docs/kernel_function),
[Taichi metaprogramming](https://docs.taichi-lang.org/docs/meta),
[Taichi dataclass](https://docs.taichi-lang.org/docs/dataclass).
Local grounding: R5 (system-level; consistent with the above, adds
cache-key/marshaling context).
