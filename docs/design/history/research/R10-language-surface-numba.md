# R10 — numba's language surface (constructs + @jitclass)

*Research agent survey, 2026-07-12. Sources: numba.readthedocs.io stable
(`reference/pysupported.html`, `user/jitclass.html`, structref extending
docs). Rows marked "verified" were empirically tested against numba 0.66.0 /
CPython 3.12 in nopython mode; the docs' construct list is stale in two
places (marked ⚠). Consumed by the book's ch07a lay-of-the-land interlude.*

## Language construct support matrix

| Construct | Status | Limitation (if partial) |
|---|---|---|
| `if / elif / else` | supported | |
| ternary `a if c else b` | supported (verified) | |
| `for x in ...` + `range` | supported | |
| `while` | supported | |
| `break` / `continue` | supported | |
| early `return` / multiple returns | supported | all returns must unify to one type |
| `and` / `or` / `not` | supported | |
| chained comparisons `0 < x < 10` | supported (verified) | |
| augmented assignment `+=` etc. | supported (verified) | |
| tuple literals, indexing, unpacking, swap | supported (verified) | heterogeneous tuples: const-index only; iteration needs `literal_unroll()` (verified fail on plain hetero iter) |
| starred unpacking `a, *b = t` | unsupported (verified) | `UNPACK_EX` opcode rejected |
| list literals / `list()` | partial | reflected lists deprecated, must be type-homogeneous; prefer `numba.typed.List`; plain Python list args still accepted as reflected (verified) |
| list comprehension (incl. nested) | supported (verified) | listed as "partial": result is numba list, not array; supports parallel array comprehension under `parallel=True` |
| dict literals / `dict()` | partial | only `numba.typed.Dict` semantics — fixed key/value types; passing a plain Python `dict` argument fails (verified) |
| dict comprehension | supported (verified) ⚠ | works on 0.66, returns `numba.typed.Dict`; docs' construct list still says unsupported |
| set literals / `set()` | partial | must be homogeneous; refcounted element types (e.g. strings) unsupported |
| set comprehension | unsupported (verified) | |
| generator expressions | unsupported (verified) | fails in "inline closures" pass, both in `sum(...)` and `for` loops |
| `lambda` (defined + called locally) | supported (verified) | same rules as inner functions |
| nested functions / closures | partial (verified) | must be non-recursive, called locally; may be passed as argument to jit fns but **cannot be returned** (verified fail) |
| generators / `yield` | supported (verified) | no `.send()` / `.throw()` / `.close()` |
| `yield from` | unsupported (verified) | |
| `raise E` / `raise E(args)` | partial | only those two forms; exception class + args must be compile-time constants |
| `try / except` | partial (verified) | only bare `except` or `except Exception` — error verbatim: "Exception matching is limited to `<class 'Exception'>`"; `except ValueError` rejected; `except ... as e` rejected ("Exception object cannot be stored into variable"); bare re-raise rejected ("re-raising of an exception is not yet supported") |
| `try / finally`, `try/except/else` | supported (verified) | |
| `with` | partial | only `numba.objmode()` context manager |
| `match` | partial (verified) ⚠ | undocumented; literal cases and capture-with-guard compile fine (plain compares in bytecode); sequence/class/mapping patterns rejected (`MATCH_SEQUENCE` etc. unsupported opcodes) |
| `assert` (with message) | supported (verified) | |
| `print` | partial (verified) | numbers and strings only; no `file=` or `sep=` |
| f-strings / str ops | partial (verified) | f-strings without format specs, `{}` contents must have `str()` overload (str/int); `f"{x:.2f}"` fails ("format spec in f-strings not supported yet"); 40+ str methods, concat, `*`, `.upper()` etc. work but can be slower than CPython |
| recursion (self) | supported (verified) | callee must have a control-flow path that returns without recursing; type must be inferable |
| global variable read | supported (verified) | frozen as compile-time constant at first compile |
| global variable write | unsupported (verified) | `STORE_GLOBAL` opcode rejected |
| default args / keyword calls / `*args` | supported (verified) | `*args` received as a tuple |
| `**kwargs` in function signature | unsupported (verified) | |
| decorator syntax inside a jit function | supported (verified) | works iff the decorator is itself compilable (it's just a call); function values can be passed but not returned or re-assigned to a different function |
| classes (`class`) | unsupported except `@jitclass` | see below |
| walrus `:=` | unsupported (verified) | type-inference failure on named expressions ("Type of variable ... cannot be determined") — even `(z := x+1) + z` |
| slicing | supported (verified) | lists/arrays incl. negative step; tuples: constant slices only |
| `del` | unsupported (verified, documented) | |
| `async def/with/for` | unsupported | by design (no event loop / object model) |
| `isinstance` | supported (verified) | against numba-known types |

## `@jitclass` (numba.experimental)

- **Field spec**: list of `(name, numba_type)` 2-tuples or `OrderedDict`
  passed to `@jitclass(spec)`; alternatively inferred from **class-level type
  annotations** via `as_numba_type` — but NumPy array fields must appear in
  the explicit spec (dtype/ndim not expressible as plain annotations). Typed
  containers declared with `types.ListType(...)` / `types.DictType(...)` or
  `typeof(instance)`.
- **Works**: methods (compiled nopython), `__init__` (must set all spec'd
  fields), properties (**getter/setter only, no deleter**), `staticmethod`,
  ~60 dunders (`__add__`, `__getitem__`, `__len__`, `__hash__`, comparisons,
  in-place ops).
- **Limitations (docs verbatim-ish)**: class object acts as *just the
  constructor* inside jit code; `isinstance()` only works in the interpreter;
  interpreter-side manipulation of instances is not optimized; CPU only; no
  inheritance; no pickling.
- **structref** (`numba.experimental.structref`): lower-level mutable
  pass-by-reference struct; requires manual boilerplate —
  `@structref.register` on a `types.StructRef` subclass, a `StructRefProxy`
  Python-side proxy, and `define_proxy()`/`define_boxing()` to link them;
  methods added via `@overload_method`.

## Rejected by design, with stated reasons

- **Exception objects are never materialized** in compiled code → hence no
  `except ... as e`, no storing exceptions in variables, no re-raise, and
  matching limited to `Exception` itself.
- **Functions are not first-class objects**: "Numba does not handle function
  objects as real objects"; once a variable is bound to a function it cannot
  be re-bound to a different function; functions can be passed in but not
  returned.
- **Reflected lists/sets rejected if heterogeneous** "even if the types are
  compatible" — whole-container static typing is the compilation model; dict
  reflection dropped entirely in favor of `typed.Dict` (typing +
  thread-safety: typed dict is safe for concurrent reads, not writes).
- **Globals are compile-time constants** — mutation is meaningless
  post-compile, so `STORE_GLOBAL` is rejected.
- **`KeyboardInterrupt` / `SystemExit` are masked** (signals ignored) during
  compiled execution.
- **Coroutine features** (`send/throw/close`, `async *`) excluded — no
  runtime object protocol in nopython mode.

Caveat for machine consumption: numba's support is defined at the
**bytecode/type-inference level**, not the AST level — e.g. `match` partially
works only because simple patterns lower to ordinary compares, and failures
surface as `UnsupportedBytecodeError` (opcode-level) or `TypingError`
(inference-level), not syntax errors.
