"""Arrays as captures: the ndarray kinds, the indexing surface, and axes.

Design: ``100_arrays-and-axes.md``. The shape of the thing:

- **Types** — kernel ``Array`` is the rank-generic summary (dtype, rank,
  device — never shape). ``ShapedArray`` turns the §13 dial (shape in the
  type: one specialization per shape, strides const-fold).  ``NamedArray``
  is the xarray exercise: axis *names* in the type. Names ride
  fingerprints and lower-time checks and are GONE by codegen — pedantry
  at zero machine-code cost, which is the whole thesis applied to axes.
- **Marshaling** — an array capture's leaves are one ``BufferLeaf`` (the
  payload, travels the leaves channel) plus i64 shape and stride slots in
  staging (rank-generic; ``ShapedArray`` ships the buffer alone). Shape
  and stride reads lower as ordinary ``core.env`` sub-path nodes, so
  ``legalize_params`` turns them into ``abi.slot`` staging reads for free;
  only the buffer needs a dialect op (``array.buffer``) because the kernel
  deliberately refuses to legalize leaves-channel captures itself.
- **Indexing** — anonymous arrays: positional ``a[i, j]``, every axis
  exactly once, strict i64. Named arrays: ``a.isel(y=i, x=j)`` keywords
  are MANDATORY (positional on a named array is refused — transposition
  is the bug names exist to kill). ``sel`` (label-based) is host-side
  work, deferred with reasons in 100 §2.
- **v1 scope** — arrays are CAPTURES, read-only, C-contiguous, indexed to
  scalars. Argument arrays wait for the arg-side normalize (step-7 note);
  views, writes, and array results are recorded cuts (100 §6).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..kernel.ir import VerifyError
from ..kernel.lower import MissingRule, fmt
from ..kernel.ops import OpDef
from ..kernel.pack import BufferLeaf, ScalarLeaf
from ..kernel.types import Array, Type, f32, f64, i32, i64
from ..kernel.valuekind import BUILTINS, KindTable


@dataclass(frozen=True, slots=True)
class ShapedArray(Array):
    """The §13 dial, one notch: shape in the type. Strides become
    compile-time constants; staging loses the shape/stride slots; the
    cache specializes per shape. Opt in per value with ``Shaped(x)``."""

    shape: tuple = ()

    def __repr__(self) -> str:
        return f"array<{self.dtype!r},{'×'.join(map(str, self.shape))}>"


@dataclass(frozen=True, slots=True)
class NamedArray(Array):
    """The xarray exercise: axis names in the type — the pedantic pick.
    ``isel`` keywords are checked against ``dims`` at lower time; the
    rendered code is IDENTICAL to positional indexing."""

    dims: tuple = ()

    def __repr__(self) -> str:
        return f"array<{self.dtype!r},{','.join(self.dims)}>"


@dataclass(frozen=True)
class Shaped:
    """Wrap a capture to put its SHAPE in the type (specialize per shape)."""

    array: object

    @property
    def shape(self):
        return self.array.shape


@dataclass(frozen=True)
class Named:
    """Wrap a capture to NAME its axes (xarray does this implicitly)."""

    array: object
    dims: tuple

    @property
    def shape(self):
        return self.array.shape


_DTYPES = {"float64": f64, "float32": f32, "int64": i64, "int32": i32}


def _summarize(a, cls=Array, **extra) -> Array:
    import numpy as np

    if not isinstance(a, np.ndarray):
        raise TypeError(f"expected a numpy array, got {type(a).__name__}")
    dt = _DTYPES.get(a.dtype.name)
    if dt is None:
        raise TypeError(f"dtype {a.dtype.name!r} has no DSL summary (v1: {sorted(_DTYPES)})")
    if not a.flags["C_CONTIGUOUS"]:
        raise TypeError("v1 adopts C-contiguous arrays only — np.ascontiguousarray(x) it, or wait for views")
    return cls(dt, a.ndim, "C", "=", a.flags.writeable, "cpu", **extra)


class _NdKind:
    """numpy.ndarray: rank-generic summary; payload + shape + strides leaves."""

    def typeof(self, v, table: KindTable) -> Type:
        return _summarize(v)

    def fingerprint(self, v, table: KindTable):
        return self.typeof(v, table)

    def flatten(self, v, table: KindTable) -> tuple:
        s = v.itemsize
        return (v.reshape(-1), *v.shape, *(st // s for st in v.strides))


class _ShapedKind:
    def typeof(self, v: Shaped, table: KindTable) -> Type:
        return _summarize(v.array, ShapedArray, shape=tuple(v.array.shape))

    def fingerprint(self, v: Shaped, table: KindTable):
        return self.typeof(v, table)

    def flatten(self, v: Shaped, table: KindTable) -> tuple:
        return (v.array.reshape(-1),)


class _NamedKind:
    def typeof(self, v: Named, table: KindTable) -> Type:
        dims = tuple(map(str, v.dims))
        if len(dims) != v.array.ndim or len(set(dims)) != len(dims):
            raise TypeError(f"need {v.array.ndim} distinct axis names, got {dims!r}")
        return _summarize(v.array, NamedArray, dims=dims)

    def fingerprint(self, v: Named, table: KindTable):
        return self.typeof(v, table)

    def flatten(self, v: Named, table: KindTable) -> tuple:
        s = v.array.itemsize
        return (v.array.reshape(-1), *v.array.shape, *(st // s for st in v.array.strides))


class _XArrayKind:
    """xarray.DataArray: dims come from the value itself — ``typeof`` IS the
    adapter. Any DLPack-ish payload xarray wraps reduces to its numpy data
    in v1; the 090 interop contract widens this at step 14."""

    def typeof(self, v, table: KindTable) -> Type:
        import numpy as np

        if not isinstance(v.data, np.ndarray):
            raise TypeError(f"v1 adopts numpy-backed DataArrays; .data is {type(v.data).__name__}")
        return _summarize(v.data, NamedArray, dims=tuple(map(str, v.dims)))

    def fingerprint(self, v, table: KindTable):
        return self.typeof(v, table)

    def flatten(self, v, table: KindTable) -> tuple:
        a = v.data
        return (a.reshape(-1), *a.shape, *(st // a.itemsize for st in a.strides))


def _rank_generic_leaves(t: Array, table: KindTable) -> tuple:
    shp = tuple(((1 + a,), ScalarLeaf("i64")) for a in range(t.ndim))
    std = tuple(((1 + t.ndim + a,), ScalarLeaf("i64")) for a in range(t.ndim))
    return (((0,), BufferLeaf()), *shp, *std)


def _payload(v):
    """The numpy array under any capturable array value. Order matters:
    ``np.ndarray.data`` is a memoryview, so the raw case must come first."""
    import numpy as np

    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, (Shaped, Named)):
        return v.array
    return v.data  # xarray.DataArray


def _flat_child(t: Array, i: int):
    """Compiled-extractor descent: step 0 = the flat payload; then shape
    axes; then strides in elements — the same order ``flatten`` declares."""
    if i == 0:
        if isinstance(t, ShapedArray):
            # Shape lives in the TYPE here, and guards are identity-only — an
            # in-place `x.shape = ...` mutation would silently serve strides
            # baked for the old shape. Drift police, satellite edition:
            def checked(v, t=t):
                p = _payload(v)
                if tuple(p.shape) != t.shape:
                    raise RuntimeError(
                        f"Shaped capture drifted: compiled for shape {t.shape}, the array is now "
                        f"{tuple(p.shape)} — rebuild the closure (Shaped puts shape in the TYPE)"
                    )
                return p.reshape(-1)

            return t, checked
        return t, lambda v: _payload(v).reshape(-1)
    if i <= t.ndim:
        return i64, lambda v, a=i - 1: _payload(v).shape[a]
    return i64, lambda v, a=i - 1 - t.ndim: (lambda p: p.strides[a] // p.itemsize)(_payload(v))


# Aspects: NamedArray inherits Array's (MRO); ShapedArray ships the buffer alone.
BUILTINS.register_aspect("leaves", Array, _rank_generic_leaves)
BUILTINS.register_aspect("leaves", ShapedArray, lambda t, table: (((0,), BufferLeaf()),))
BUILTINS.register_aspect("child", Array, _flat_child)


def _rebuild_refused(t, it, rec):
    raise VerifyError("array RESULTS are not in v1 — kernels return scalars; DPS out-arrays arrive with chaining")


BUILTINS.register_aspect("rebuild", Array, _rebuild_refused)


def _load_rule(args, attrs, regions) -> Type:
    base, idx = args
    if not isinstance(base, Array) or idx != i64:
        raise TypeError(f"array.load wants (array, i64 linear index), got {args!r}")
    return base.dtype


ARRAY_OPS = {
    "array.load": OpDef("array.load", _load_rule, frozenset({"Pure"})),
    "array.buffer": OpDef("array.buffer"),  # attrs: src=("env", *path) | ("arg", i); type = the erased Array
    "array.dim": OpDef("array.dim", lambda a, at, r: i64),  # attrs: src, sub — a staging read the
    #   renderer resolves from THE PLAN (arg-side twin of the env abi.slot route)
}


def _linear_index(ctx, node, base, indices):
    """Explicit stride arithmetic — machine code identical for named and
    positional, and for captures vs ARGUMENTS. Rank-generic strides:
    captures read staging via core.env sub-paths (legalized to abi.slot for
    free); arguments read staging via `array.dim` (renderer-resolved from
    the plan, the same pattern as `array.buffer`). ShapedArray strides are
    core.const either way (const-folded)."""
    t = base.type
    if base.op == "core.env":
        src = ("env", *dict(base.attrs)["slot"])
    elif base.op == "core.param":
        src = ("arg", dict(base.attrs)["index"])
    else:
        raise MissingRule(
            f"array indexing needs a captured or argument array, got a computed value "
            f"(array-valued expressions arrive with the tensor dialect) [{fmt(ctx.loc(node))}]"
        )
    if t.ndim == 0:
        raise MissingRule(f"a 0-d array has no axes to index — capture the scalar instead [{fmt(ctx.loc(node))}]")
    if len(indices) != t.ndim:
        raise MissingRule(
            f"every axis exactly once: {t.ndim} indices needed, got {len(indices)} "
            f"(views are a recorded cut) [{fmt(ctx.loc(node))}]"
        )
    for ix in indices:
        if ix.type != i64:
            raise TypeError(f"indices are strict i64, got {ix.type!r} [{fmt(ctx.loc(node))}]")
    if isinstance(t, ShapedArray):  # strides from the type: constants
        strides = []
        acc = 1
        for extent in reversed(t.shape):
            strides.insert(0, ctx.emit("core.const", node=node, type=i64, value=acc))
            acc *= extent
    elif src[0] == "env":  # capture strides: core.env sub-paths, legalized to abi.slot for free
        strides = [ctx.emit("core.env", node=node, type=i64, slot=(*src[1:], 1 + t.ndim + a)) for a in range(t.ndim)]
    else:  # argument strides: array.dim, renderer-resolved from the plan
        strides = [ctx.emit("array.dim", node=node, src=src, sub=1 + t.ndim + a) for a in range(t.ndim)]
    linear = None
    for ix, st in zip(indices, strides):
        term = ctx.emit("core.mul", ix, st, node=node)
        linear = term if linear is None else ctx.emit("core.add", linear, term, node=node)
    # ERASE the refinement on the emitted node: names (and shape) have done
    # their lowering-time job; the IR carries only what codegen needs, so a
    # named kernel and its positional twin produce IDENTICAL content keys —
    # tier 2 proves the pedantry is free.
    erased = Array(t.dtype, t.ndim, t.layout, t.byteorder, t.writeable, t.device)
    buf = ctx.emit("array.buffer", node=node, type=erased, src=src)
    return ctx.emit("array.load", buf, linear, node=node)


def make_subscript_rule(prev):
    def _subscript(ctx, node):
        base = ctx.lower(node.value)
        if isinstance(base.type, NamedArray):
            raise MissingRule(
                f"this array's axes have NAMES {base.type.dims!r} — positional indexing is refused; "
                f"use .isel({', '.join(d + '=...' for d in base.type.dims)}) [{fmt(ctx.loc(node))}]"
            )
        if isinstance(base.type, Array):
            raw = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            return _linear_index(ctx, node, base, [ctx.lower(e) for e in raw])
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            return ctx.emit("core.extract", base, node=node, index=node.slice.value)
        return prev(ctx, node)  # the base pack's refusal text, unchanged

    return _subscript


def make_call_rule(prev):
    def _call(ctx, node):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "isel":
            base = ctx.lower(f.value)
            if not isinstance(base.type, NamedArray):
                raise MissingRule(f"isel() needs a NAMED array, got {base.type!r} [{fmt(ctx.loc(node))}]")
            dims = base.type.dims
            given = {kw.arg: kw.value for kw in node.keywords}
            woven = ctx.context.get("woven") or {}
            active = {d: woven[d] for d in dims if d in woven}  # over owns these axes here
            if set(given) & set(active):
                raise MissingRule(
                    f"axis {sorted(set(given) & set(active))!r} is mapped away here (over owns it) "
                    f"[{fmt(ctx.loc(node))}]"
                )
            if node.args or set(given) != set(dims) - set(active):
                raise MissingRule(
                    f"isel is pedantic on purpose: name every axis exactly once as a keyword — "
                    f"expected {set(dims) - set(active) or '{}'!r}, got {set(given) or '{}'!r} "
                    f"[{fmt(ctx.loc(node))}]"
                )
            if active:
                hits = ctx.context.get("woven_hits")
                if hits is not None:
                    hits.append(tuple(active))
            indices = [active[d] if d in active else ctx.lower(given[d]) for d in dims]
            return _linear_index(ctx, node, base, indices)
        return prev(ctx, node)

    return _call


def install(registry) -> None:
    for op, defn in ARRAY_OPS.items():
        registry.ops[op] = defn
    registry.lower_rules[ast.Subscript] = make_subscript_rule(registry.lower_rules[ast.Subscript])
    registry.lower_rules[ast.Call] = make_call_rule(registry.lower_rules[ast.Call])
    # array.load spellings ship STATICALLY with each backend's own CODE_FOR_OP
    # (python + C both) — registration order cannot strand a renderer.


def register_kinds() -> None:
    """numpy (and xarray, if present) become capturable. Guarded imports:
    the core install must not demand either library (090 §2)."""
    try:
        import numpy as np
    except ImportError:
        return
    BUILTINS.register(np.ndarray, _NdKind())
    BUILTINS.register(Shaped, _ShapedKind())
    BUILTINS.register(Named, _NamedKind())
    try:
        import xarray as xr
    except ImportError:
        return
    BUILTINS.register(xr.DataArray, _XArrayKind())


register_kinds()
