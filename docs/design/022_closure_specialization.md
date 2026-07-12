# Closure specialization & caching in numba

Companion notes for `closure.ipynb`. Numba version verified against: **0.65.1**
(`.venv/lib/python3.14/site-packages/numba`).

## The question

Given a closure factory and a higher-order kernel:

```python
import numba, numpy as np

def closure(x):
    @numba.njit
    def inner(y):
        return x + y
    return inner

@numba.njit
def m(a, f):
    for i in range(len(a)):
        a[i] = f(a[i])

a = np.arange(10)
m(a, closure(5))   # call 1
m(a, closure(6))   # call 2
```

When call 2 runs, does `m(a, closure(6))` trigger a **second compilation of the
inner** function returned by `closure`, or is numba smart enough to **reuse the
specialization** it already built for the inner during call 1 (same code, same
argument type `int64 -> int64`)?

## The answer

**It recompiles.** Numba does not reuse the prior specialization — and in fact
*both* `m` and the inner are compiled a second time. Numba keys its caches on
**object identity**, never on source/bytecode structure, and each `closure(...)`
call manufactures a fresh dispatcher object. There is a deeper reason too: the
captured value `x` is **frozen into the IR as a compile-time constant**, so the
two inners do not even share machine code (`return 5 + y` vs `return 6 + y`).

| Cache | Keyed on | Consequence for call 2 |
|-------|----------|------------------------|
| Inner's specialization cache (`Dispatcher.overloads`) | the inner **instance** + arg types | fresh instance → empty cache → recompiles |
| Numba type of a passed njit fn (`types.Dispatcher`) | the dispatcher **object identity** (weakref) | different object → different type → `m` recompiles |
| Captured free var `x` | read at trace time, baked in as a **constant** | the two inners are different code, not just different cache slots |

## How it was verified (source chain)

Every `closure(...)` call re-runs `def inner` and `@numba.njit` wraps it in a
**new** `Dispatcher`, whose specialization cache is initialized empty per object:

| Step | File:line | What it shows |
|------|-----------|---------------|
| Per-instance cache is fresh per object | `core/dispatcher.py:204` | `self.overloads = collections.OrderedDict()` runs in each dispatcher's `__init__` |
| Passed njit fn → numba type | `core/dispatcher.py:807` | `_numba_type_` returns `types.Dispatcher(self)` (the specific instance) |
| That type is keyed on identity | `core/types/functions.py:519, 490` | `Dispatcher` is a `WeakType`; `__eq__` is true only when both wrap the **same object** (`obj is other._wr()`) |
| Typing `m` forces the inner to compile | `core/types/functions.py:534` → `core/dispatcher.py:304,319` | `get_call_type` → `get_call_template` calls `self.compile((int64,))` to "ensure an overload is available" |
| The recompile guard reads the **instance** cache | `core/dispatcher.py:884-887` | `existing = self.overloads.get(tuple(args))` — a fresh inner's dict is empty → miss → compile |
| No disk-cache rescue | `core/dispatcher.py:889` | `cache=True` is not set, so `self._cache.load_overload` is a no-op |
| Free var `x` captured at trace time | `core/interpreter.py:2447-2448` | `op_LOAD_DEREF` does `value = self.get_closure_value(idx)` and stores `ir.FreeVar(idx, name, value)` |
| Free var typed as a constant | `core/typeinfer.py:1472-1473` | `ir.FreeVar` goes through `typeof_global` — the same path as a frozen global constant |

The chain for call 2: `closure(6)` builds a new dispatcher → its
`types.Dispatcher` type is unequal to call 1's → `m` builds a 2nd specialization
→ typing that specialization calls `inner.compile((int64,))` → the new inner's
`overloads` is empty → cache miss → fresh compile, with `x = 6` baked in as a
constant.

## How it was verified (empirically)

```python
f5, f6 = closure(5), closure(6)
f5 is f6                         # False  -- distinct dispatcher objects
f5._numba_type_ == f6._numba_type_  # False  -- distinct numba types

m(a, f5)
len(m.overloads)   # 1
f5.signatures      # [(int64,)]   f5._cache_misses == {(int64,): 1}
f6.signatures      # []           -- untouched

m(a, f6)
len(m.overloads)   # 2            -- m recompiled (Dispatcher type differs)
f6.signatures      # [(int64,)]   f6._cache_misses == {(int64,): 1}  -- fresh compile
```

The miss counters are the proof: `f6` records a genuine cache miss / compile for
`(int64,)`, and `m` grows from 1 to 2 specializations.

## Implications for the alternate DSL

The goal is the opposite policy: **compile the inner once for a given parameter
type signature, then reuse it across capture values** (so `closure(5)` and
`closure(6)` share one compiled `inner` for `int64 -> int64`, and `m` stays at a
single specialization). Numba's design blocks this in three independent places —
each must change.

| # | Numba behavior | What blocks reuse | DSL change required |
|---|----------------|-------------------|---------------------|
| 1 | Specialization cache keyed on dispatcher **instance identity** | a new closure object always misses | Key the cache on a **structural identity**: `(code_object, captured-var types, arg types)`. Two inners from the same `def` with the same capture/arg types resolve to the same cache slot. |
| 2 | A passed function's type is `Dispatcher(instance)`, identity-based | the kernel `m` re-specializes per closure instance | Give closures a **structural function type** — same `(code, freevar types, signature)` → equal type. Then `m` compiles once and accepts any matching closure. |
| 3 | Free vars **frozen as constants** (`x` baked into IR at trace time) | the two inners are literally different machine code, so even a shared cache slot would be wrong | Do **closure conversion**: lower the inner to `(code_ptr, env)` where captures live in a runtime environment passed as a hidden argument, instead of constant-folding the cell value. |

Item 3 is the load-bearing one and easy to miss. Specialization caching keyed on
types is meaningless unless the capture is a **runtime value** rather than a
compile-time constant. The order of operations for the DSL:

1. **Closure-convert** so a function value is `(code, environment)`; captures become
   runtime data typed by their numba-equivalent types, not frozen literals.
2. **Type closures structurally** so the type carries `(code identity, freevar
   types, param types)` and ignores both object identity and capture values.
3. **Key the specialization cache** on that structural signature, so the first
   `int64 -> int64` compile of the inner is reused for every later capture value of
   the same types.

The tradeoff this surrenders is numba's constant-folding of captures: numba can
specialize on the *value* `x = 5` (enabling constant propagation / dead-code
elimination), at the cost of a recompile whenever the value changes. The DSL's
policy trades that per-value optimization for one compile per type signature —
the right call when capture values vary at runtime but their types do not.
