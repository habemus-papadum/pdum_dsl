# R9 — PEP 750 t-strings and hosted mini-languages

Research report for the pdum.dsl redesign. Topic: PEP 750 template strings as the
hosting mechanism for embedded mini-languages (einops-style notation and beyond),
and how they plug into the type-keyed compilation cache.

**Verification status.** All Python semantics below were **verified empirically on
CPython 3.14.0** (the interpreter in this repo's `.venv`), not taken from the PEP
text. Probe scripts and raw output are reproduced inline. einops mechanics were
read from the current `main` source (einops 0.8.2 on PyPI as of July 2026).
Ecosystem survey checked against live GitHub/PyPI in July 2026.

Sources:
- PEP 750 (final): <https://peps.python.org/pep-0750/>
- `string.templatelib` docs: <https://docs.python.org/3.14/library/string.templatelib.html>
- einops source: <https://github.com/arogozhnikov/einops> (`einops/einops.py`, `einops/parsing.py`)
- Ecosystem index: <https://github.com/t-strings/awesome-t-strings>
- sql-tstring: <https://github.com/pgjones/sql-tstring>
- tdom: <https://github.com/t-strings/tdom>
- psycopg 3 t-string queries: <https://www.psycopg.org/psycopg3/docs/basic/tstrings.html>

---

## 1. PEP 750 final semantics on Python 3.14 (shipped)

### 1.1 The types

Both types live in `string.templatelib` (module exports exactly `Template`,
`Interpolation`, `convert`). A literal `t"a {x} b {y:>{w}} c"` evaluates eagerly
(left-to-right, verified) to a `Template`:

```python
t = t"a {x} b {y:>{w}} c"       # x=5, y=6.0, w=5
t.strings          # ('a ', ' b ', ' c')          tuple[str, ...], len = N+1
t.values           # (5, 6.0)                     tuple of interpolation values
t.interpolations   # (Interpolation(5, 'x', None, ''),
                   #  Interpolation(6.0, 'y', None, '>5'))
list(t)            # ['a ', Interpolation(...), ' b ', Interpolation(...), ' c']
                   # __iter__ OMITS empty strings — use .strings when position matters
```

`Interpolation` is shallowly immutable, has
`__match_args__ = ("value", "expression", "conversion", "format_spec")`:

- `value: object` — the evaluated expression result.
- `expression: str` — the *source text* inside the braces (`'x'`, `'y'`). Stable
  per call site; empty string if constructed manually.
- `conversion: Literal['a','r','s'] | None` — the `!r`-style flag. Syntactically
  fixed at the call site.
- `format_spec: str` — **already-rendered** spec; nested interpolations inside the
  spec are eagerly evaluated to text (`{y:>{w}}` with `w=5` yields `'>5'`).

`templatelib.convert(obj, conversion)` applies f-string conversion semantics
(`'s'`→`str`, `'r'`→`repr`, `'a'`→`ascii`, `None`→identity).

Other structural facts (verified):
- `Template + Template` works (strings tuples fuse at the seam:
  `('kernel ', ' endkernel ', ' end')`); `Template + str` raises `TypeError`.
- `Template` is **not subclassable** (`not an acceptable base type`).
- A no-interpolation `t"plain"` has `strings=('plain',)`, `interpolations=()`.
- Adjacent interpolations get `''` entries in `.strings` — `t"{a} -> {b}"` has
  `strings == ('', ' -> ', '')`.

### 1.2 Template identity — the load-bearing facts for caching

**Fact 1: `Template` equality and hash are by object identity.** Verified: two
templates from the same call site with equal contents compare unequal
(`t1 == t2` → `False`) and hash differently. `Template` is hashable, but the hash
is the default identity hash. **A `Template` instance is therefore useless as a
cache key** — every evaluation of the literal makes a fresh, distinct-hashing
object.

**Fact 2: `.strings` is a compile-time constant.** The CPython compiler stores the
static-strings tuple in `co_consts` and the runtime builds the template with
dedicated opcodes. Disassembly of `def make(v): return t"kernel {v} end"`:

```
LOAD_CONST     2 (('kernel ', ' end'))     ← the .strings tuple is a constant
LOAD_FAST_BORROW  0 (v)
LOAD_CONST     1 ('v')                     ← the expression text is a constant
BUILD_INTERPOLATION  2
BUILD_TUPLE    1
BUILD_TEMPLATE
```

Consequences, all verified on 3.14.0:
- `make(1).strings is make(2).strings` → **True**. The identical tuple object is
  reused on every evaluation of that call site. Zero per-call allocation for the
  static part.
- Two *different* call sites in the same module with identical literal text also
  shared the tuple (`co_consts` deduplication within a compilation unit) — but this
  is an implementation nicety, **not guaranteed across modules or across
  re-`exec`**. Value equality (`==`/`hash`) always holds for equal text; identity
  (`is`) only sometimes.
- `.strings` is a `tuple[str, ...]`: **hashable, orderable, picklable — a perfect
  dict key.** A dict keyed on one call site's `.strings` is hit by an equal-text
  template from another call site (verified).

**Fact 3: notebook/re-run behavior mirrors code objects.** Because `.strings`
compares by value, re-running an *unchanged* cell produces a new-but-equal tuple →
cache **hit**; editing the static text produces an unequal tuple → cache **miss**
and recompile. This is exactly the invalidation semantics
`design/dsl_caching_layer.md` establishes for `__code__`-keyed caching (code
objects hash by value too). T-strings and `@jit` functions can share one
invalidation story.

### 1.3 Costs (measured, CPython 3.14.0, Apple Silicon)

| Operation | Cost |
|---|---|
| Evaluate a 2-interpolation t-string literal | **108 ns** |
| Equivalent f-string (for scale) | 166 ns |
| `.values` access | 42 ns |
| Build template + compute `(strings, value-types)` key + dict hit | **393 ns** |
| Same with `id(strings)` two-level fast path | 265 ns |
| Bare Python closure rebuild (the thesis baseline) | 100 ns |

A t-string cache hit costs ~4× a closure rebuild — a few hundred nanoseconds,
comfortably inside "the loop stays hot" territory, and small against any GPU
dispatch. The `id()` fast path exists but only saves ~130 ns and carries an
id-reuse hazard (must retain the tuple to pin its id); at these magnitudes it is
not worth the complexity — key on the tuple by value.

---

## 2. The structural mirror: t-strings are closures

The correspondence with the project thesis is exact:

| Closure concept (dsl_caching_layer.md) | t-string concept |
|---|---|
| code identity: `func.__code__`, value-compared | `.strings` tuple (+ conversions), value-compared, `co_consts`-interned |
| typed environment: `env_types` | `tuple(typeof(v) for v in t.values)` |
| environment values: `Env` instance | `.values` |
| Phase A (capture, compile-free) | evaluating the literal — 108 ns, no parsing |
| Phase B (compile at first call) | first cache miss on `(strings, value-types)` |
| marshal values per call | re-read `.values`, write to uniform buffer / kernel args |

Evaluating a t-string in a tight loop *is* rebuilding a closure with fresh values:
CPython has already split code identity from captured values for us, and interned
the code-identity part as a constant. The cache design drops straight in:

```python
# ~40 lines is a realistic budget for this whole shim.
from string.templatelib import Template

class MiniLang:
    def __init__(self, compile_fn):          # compile_fn: (strings, convs, env_types) -> Artifact
        self.compile_fn = compile_fn
        self.cache: dict[tuple, Artifact] = {}

    def __call__(self, t: Template, /):
        convs = tuple(i.conversion for i in t.interpolations)
        env_types = tuple(typeof(v) for v in t.values)     # the project's honest typeof
        key = (t.strings, convs, env_types)                # value-hashed, all static+types
        art = self.cache.get(key)
        if art is None:
            art = self.cache[key] = self.compile_fn(t.strings, convs, env_types)
        return art.bind(t.values)             # marshal values only — the hot path
```

`compile_fn` parses the static strings into the mini-language's AST, lowers to the
project IR, and runs the ordinary type-keyed pipeline. `art.bind(values)` is the
same logical-value → physical-parameters marshaling the closure path uses (one
interpolation may become pointer + shape, exactly like an array capture).

### 2.1 Gotchas (each verified)

1. **Never key on the `Template` object.** Identity hash; every literal
   evaluation is a fresh key; the cache would grow monotonically and never hit.
   This is the single most likely implementation error.

2. **`.strings` under-determines the syntax when the DSL reads more than values.**
   Verified collisions:
   - `t"{a} -> {b}"` and `t"{b} -> {a}"` — **identical** `.strings`
     `('', ' -> ', '')`, different `expression` texts. Harmless if the DSL treats
     holes as opaque value slots (values are marshaled per call anyway); fatal if
     the DSL assigns meaning to the *expression text* (e.g. using `{h}` as a named
     axis). If expression text is semantic, add
     `tuple(i.expression for i in t.interpolations)` to the key — it is a
     `co_consts` constant, so it is static and free.
   - `t"{a}"` vs `t"{a!r}"` — identical `.strings`, different `conversion`. If the
     DSL gives conversions meaning, include the conversions tuple (static per call
     site, as above). Including it unconditionally is cheap and safe.

3. **`format_spec` is NOT static.** `t"{a:>{w}}"` re-renders the spec from the
   current `w` on every evaluation — same `.strings`, different `format_spec`
   per call (verified `'>5'` vs `'>9'`). A mini-language must treat `format_spec`
   as a *runtime value channel* (marshal it like a value, key on its `typeof`, or
   value-lift it explicitly) or reject non-empty specs. Silently folding it into
   the compile is a cache-correctness bug of exactly the class the caching doc
   warns about.

4. **`__iter__` omits empty strings.** Any parser that walks `iter(template)` to
   recover hole positions is wrong for adjacent interpolations; parse
   `t.strings`/`t.interpolations` positionally (strings has exactly N+1 entries).

5. **Don't rely on `is` for `.strings` across modules.** Constant interning is
   per-compilation-unit. Value equality is the contract; identity is an
   opportunistic bonus.

6. **`Template` is final.** No subclass-based API (`class Einops(Template)` is
   impossible); mini-languages are functions/callables *over* templates. This
   pushes the ecosystem — correctly — toward `lang(t"...")` call syntax.

7. **Static type checking of the sub-language doesn't exist yet.** A t-string is
   just a `Template` to mypy/pyright; pattern errors surface at first call (same
   as the rest of the JIT). PEP 750's authors discuss editor tooling (tagging
   functions so IDEs highlight the embedded grammar) but as of July 2026 there is
   no standardized `Annotated`-style marker; sql-tstring et al. simply accept
   `Template`.

### 2.2 Value lifting fits naturally

Some interpolations should be compile-time constants (an axis size that determines
loop bounds, a filter width that should unroll). The project already has the
`Val{}`-style explicit lift for closures; for t-strings it is the same move at the
API level: `rearrange(t"... ({Val(4)} h) ...")` or a per-language rule that
integer-typed interpolations in *structure positions* are lifted (then
`typeof(Val(4))` carries the value into the key by design, not by accident).
einops (below) shows why this matters: axis sizes change the emitted plan.

---

## 3. einops as the exemplar notation

einops 0.8.2; mechanics below read directly from `einops/einops.py` and
`einops/parsing.py` on `main` (July 2026).

### 3.1 The grammar

Pattern = `LHS -> RHS`, whitespace-separated tokens per side (`parsing.py`,
~200 lines, char-by-char scanner, no parser generator):

- **axis name** — Python identifier, no leading/trailing `_`, keywords warned;
- **`1`** — unit axis (insert/squeeze);
- **integer > 1** — anonymous axis of fixed length (`AnonymousAxis`);
- **`(a b)`** — composition: one physical dim ⇄ product of elementary axes; no
  nesting; on LHS it *splits* (requires all-but-one length known), on RHS it
  *merges*;
- **`...`** (normalized to a single `…` char) — batch dims, at most one per side;
- **`_`** — allowed only where `allow_underscore=True`.

Validation: identifier rules, duplicate detection, one-ellipsis rule, balanced
non-nested parens. `ParsedExpression` exposes `composition` (list of lists),
`identifiers` (set), `has_ellipsis`, `has_non_unitary_anonymous_axes`.
Operations: `rearrange` (pure data movement), `reduce` with
`min|max|sum|mean|prod|any|all` or a *hashable* callable, `repeat`, plus `einsum`
and `pack`/`unpack` with separate grammars.

### 3.2 einops already runs a two-level type-keyed cache

This is the striking find: einops independently converged on the project's cache
architecture, with *rank* and *shape* playing the role of types.

**Level 1 — structure, keyed on (pattern text, op, axis-arg names, rank):**

```python
@functools.lru_cache(256)
def _prepare_transformation_recipe(
    pattern: str, operation: Reduction,
    axes_names: tuple[str, ...],   # names of explicitly passed axes_lengths — not their values
    ndim: int,
) -> TransformRecipe: ...
```

`TransformRecipe` (a plain class, immutable by convention — dataclasses avoided
only for `torch.jit.script` compatibility) is rank-polymorphic and value-free:
`elementary_axes_lengths` (with unknowns as placeholders),
`axis_name2elementary_axis`, `input_composition_known_unknown`,
`axes_permutation`, `first_reduced_axis`, `added_axes`, `output_composite_axes`.

**Level 2 — concrete plan, keyed on (recipe, actual shape, axis-length values):**

```python
_reconstruct_from_shape = functools.lru_cache(1024)(_reconstruct_from_shape_uncached)
# (recipe: TransformRecipe, shape: tuple[int,...], axes_dims: tuple[tuple[str,int],...])
#   -> CookedRecipe = (init_shapes, axes_reordering, reduced_axes,
#                      added_axes, final_shapes, n_axes_w_added)
```

The recipe participates in the L2 key via its default identity hash — sound only
because L1 guarantees one recipe object per (pattern, op, names, ndim). Level 2
does axis-length inference here: solve each dimension's known/unknown split,
check divisibility, error on mismatch. On a `TypeError` (symbolic shapes —
tf/torch tracing), it falls through to the uncached function: **graceful
degradation when the key isn't hashable**, a pattern worth copying.

**Execution** (`_apply_recipe`) is a fixed 5-step pipeline over a ~5-method
backend protocol:

```python
tensor = backend.reshape(tensor, init_shapes)        # if init_shapes
tensor = backend.transpose(tensor, axes_reordering)  # if permutation nontrivial
tensor = _reduce_axes(tensor, reduction_type, reduced_axes, backend)  # if any
tensor = backend.add_axes(tensor, n_axes, pos2len)   # repeat/broadcast, if any
tensor = backend.reshape(tensor, final_shapes)       # if final_shapes
```

Mapping to project vocabulary: L1 ≈ compile keyed on code identity + rank-types;
L2 ≈ specialization keyed on value-lifted shapes (`Val{shape}`); `_apply_recipe`
≈ marshal + dispatch. einops keys L2 on shape *values* because reshape plans are
shape-dependent — the honest version of what the project would express as an
explicit value lift, not a violation of types-not-values.

### 3.3 What an einops t-string DSL needs from OUR IR

Surface sketch:

```python
y = rearrange(t"b (h {ph}) (w {pw}) c -> b h w ({ph} {pw} c)")   # patchify
z = reduce(t"b h w c -> b c", "mean")
```

Static strings carry the pattern (code identity); interpolations carry axis
lengths (values — lifted, since they shape the plan) or, later, tensors
themselves (`t"{img}: b h w c -> b c"`-style binding).

Frontend cost is small: a ~200-line parser (einops proves the budget) plus the
~40-line cache shim from §2. The IR is where the real requirements land. The
reference IR (`src/pdum/dsl_reference/ir.py`, 127 lines) is a scalar expression
tree — `Lit/Name/BinOp/Call/MakeVec/...` — with **no array type, no shape, no
loops**. An einops lowering needs, in order of necessity:

1. **A ranked array type** in the type lattice: `Array(elem, rank)` at minimum;
   shapes appear as value-lifted parameters, not in the structural type (matching
   einops L1/L2 split: rank in the cache key, shape in the specialization key).
2. **Shape-metadata ops that are marshaling, not computation:** `reshape` and
   view-`transpose` are pure descriptor rewrites (strides/shape). If the IR's
   logical-value → physical-parameters mapping carries (pointer, shape, strides),
   `rearrange` without reduction compiles to *zero kernel code* — just a
   different marshaling of the same buffer. This falls out of the ABI design the
   desiderata already demands (§1: one logical value → N physical params).
3. **A reduction primitive** over designated axes: either a first-class
   `Reduce(op, axes, operand)` node or a loop nest the backend can pattern-match.
   This is the only part that emits real kernel code (and on WebGPU means
   workgroup reductions — a backend battery, not a frontend concern).
4. **Broadcast/`add_axes`** for `repeat` (stride-0 views where the backend
   allows, else an emit-time loop).
5. **Symbolic dim arithmetic** at the shape level: products and exact division
   with residual-unknown solving (`h = H / ph`, error if indivisible) — ~50 lines,
   evaluated at specialization time when shapes become known.

Notable non-requirement: einops needs no control flow in the *pattern* language —
the entire mini-language lowers to 5 structured ops. A t-string sub-DSL can ship
before the Python-subset frontend grows statements, provided the IR has arrays,
reduce, and descriptor-rewriting marshal.

---

## 4. Survey: t-string DSLs since 3.14

Index: <https://github.com/t-strings/awesome-t-strings>. The ecosystem is young
(3.14 shipped Oct 2025) but the API conventions have already converged.

| Project | Domain | Pattern of note |
|---|---|---|
| **sql-tstring** (pgjones) | SQL | `sql(t"SELECT ... WHERE a = {a}") -> (query, params)` — pure function Template→(str, list); sentinel values (`Absent`, `IsNull`) *rewrite structure* — a value's type changes the emitted program; `sql_context(columns=..., tables=...)` distinguishes identifier holes from value holes; dialect via context (`qmark`/`$`/asyncpg). No caching of parsed SQL (parse is cheap; they don't have our hot-loop constraint). |
| **psycopg 3** | SQL | `cur.execute(t"... {x}")` — a *major* driver accepting `Template` natively; values → server-side parameters. The strongest adoption signal. |
| **t-sql** | SQL | Template → parameterized query, multiple paramstyles. |
| **tdom** (t-strings org) | HTML | `html(t"<p>{x}</p>") -> Node`; nested templates compose (interpolation values that are themselves Templates/Nodes get spliced, not escaped); components via `<{Component} .../>` — an interpolation in *tag position* is a callable; `conversion`/`format_spec` honored via `format_interpolation`/`convert`. |
| **tstring-html** (koxudaxi) | HTML | Rust-powered parser of the static parts; auto-escape; JSX-style components; "parser-first" framing — parse `.strings` once, insert values per render. |
| **tstring-structured-data** (koxudaxi) | JSON/TOML/YAML | Same parser-first architecture for data languages. |
| **ludic**, **pyhtml-enhanced** | HTML/web | Existing frameworks retrofitting Template acceptance. |
| **regex-template** (treyhunner) | regex | Auto-`re.escape` interpolations — hole-position-aware escaping. |
| **tstringlogger** | logging | Deferred formatting: Template stored, rendered only if the record is emitted — exploits laziness of *rendering* (evaluation of values is still eager). |
| **tstr** | compat | Backport shim for <3.14 — relevant only if pdum.dsl ever loosens its 3.14 floor (it shouldn't; the `co_consts` interning is 3.14+). |

API patterns worth copying:

1. **A mini-language is a plain function `Template -> Artifact`.** Universal
   convention (`sql(t"...")`, `html(t"...")`). No registration, no subclassing
   (impossible anyway), trivially pluggable — matches the prime directive.
2. **Accept `Template`, reject `str`, in the signature.** `def sql(q: Template)`
   makes passing an f-string a type error — injection-safety in SQL, cache-safety
   for us (an f-string would smuggle values into code identity).
3. **Parser-first: parse `.strings`, then splice values.** The koxudaxi libraries
   name the architecture; ours adds the missing third step (cache the parse on the
   strings key — none of them need it because none re-parse in a hot loop; we do).
4. **Nested templates compose.** tdom splices Template-valued interpolations as
   sub-trees. For pdum.dsl: an interpolation whose value is a jitted kernel or
   another mini-language artifact should splice as a call/inline — the composition
   story for sub-DSLs, and the `+`-concat strings-fusion semantics (§1.1) even
   gives template assembly a sane cache identity.
5. **Typed sentinels that legally change code identity.** sql-tstring's `Absent`
   removes clauses — sound under our thesis because `typeof(Absent)` differs, so
   it lands in a different cache entry. Value-dependent structure via *types* is
   the sanctioned mechanism; document it as such.
6. **Interpolation position determines meaning.** tag-position = component
   (tdom), identifier-position needs allow-listing (sql-tstring). For einops:
   hole inside `(...)` = axis length; hole before `:` = tensor binding.

---

## Design lessons for pdum.dsl

1. **Adopt `(t.strings, conversions, env_types)` as the canonical mini-language
   cache key** — value-hashed, mirroring `__code__` value-hashing for `@jit`
   functions. Include `tuple(i.expression ...)` iff the language reads expression
   text. Never key on the `Template` object (identity hash — verified) and never
   fold `format_spec` into the static key (it is runtime data — verified).

2. **Give t-string languages the same two-phase lifecycle as closures.** Phase A
   = literal evaluation (108 ns, compile-free); phase B = compile on first
   `(strings, types)` miss; steady state = 300–400 ns key+lookup plus value
   marshaling. One cache infrastructure, two frontends (code objects,
   templates) — build the cache API to accept any value-hashable "code identity"
   token so both share eviction, generations, and disk-cache policy.

3. **Spec the mini-language plug-in seam as `Template -> IR fragment`, exposed to
   users as a plain callable.** Ecosystem-standard (`sql()`, `html()`), zero
   registration machinery, and the ~40-line cache shim in §2 is the entire
   per-language infrastructure cost. Type in the signature as `Template` so str
   is rejected statically.

4. **Copy einops' two-level split into the array story:** structural key carries
   *rank* (L1, `Array(elem, rank)` in the type), concrete *shapes* enter only as
   explicit value-lifts at specialization (L2). einops independently converged on
   exactly this architecture (`lru_cache` on pattern+ndim, then on shape values)
   — treat it as independent confirmation of the thesis, and copy its graceful
   fallback to uncached compilation when a key component isn't hashable.

5. **The IR needs exactly four things before it can host einops:** a ranked array
   type; `reshape`/`transpose` as *marshaling-descriptor rewrites* (zero kernel
   code — they must live in the logical-value→physical-parameters layer the
   desiderata already mandates); a `Reduce(op, axes)` primitive; and ~50 lines of
   shape arithmetic (product/exact-division solving) run at specialization time.
   No control flow in the IR is required — a t-string einops can ship before the
   statement-level frontend.

6. **Design interpolation-role rules per language, positionally.** Survey
   precedent: tag-position callables (tdom), allow-listed identifiers
   (sql-tstring). For einops-style: holes in composition position are lifted axis
   lengths; consider tensor-binding holes later. Where a hole's *type* changes
   program structure (sql-tstring's `Absent`), that is sound under the thesis and
   should be the documented idiom for value-dependent structure.

7. **Nested-template composition is the sub-DSL composition story.** Template
   `+` fuses strings tuples deterministically (verified) and tdom-style splicing
   of Template-valued interpolations gives assembled programs a well-defined
   cache identity — design the IR-fragment interface so a spliced artifact
   inlines as a call.

8. **Pin the 3.14 floor and document the interning bonus, not the guarantee:**
   `.strings` is a `co_consts` constant (same object per call site — verified via
   `dis`: `LOAD_CONST` + `BUILD_INTERPOLATION`/`BUILD_TEMPLATE`), but cross-module
   sharing is opportunistic; the contract is value equality. Skip `id()`-based
   fast paths — measured saving is ~130 ns against an id-reuse hazard.
