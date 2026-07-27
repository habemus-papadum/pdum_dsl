"""The tl dialect over the dsl core (240 C4.1) — promoted from the C3 spike.

ONE IR, many dialects (the pivot's objective): tensor-typed SSA lives on
the dsl ``Node``/``Region`` infrastructure as the ``tl.*`` op family, with
the layout lattice AS the type — so tl's alignment law is an ordinary
type rule that refuses at emission with source locations, and content
keys (``region.key``) come from the core for free.

Design items resolved here (the spike's frictions):

- **The per-kind yield protocol.** Every region ends in ``core.yield``
  (Julia's rule, at the IR level); WHAT a body yields is the KIND's
  declaration, never user ceremony: a ``step`` yields its returned
  value, a ``compute`` kernel yields the final ordering token (its
  effect) — and a kernel ``return`` refuses with the pinned voice.
- **Object-based intrinsic recognition.** ``thread_idx`` and the S.1
  vocabulary (``pointwise``, ``const_like``) are recognized by OBJECT
  IDENTITY from the function's own globals — never by name string.
- **The op-selection pattern, written once.** Operator/comparison rules
  are built by ONE factory (``_typed_rule``): lower the children, and
  TENSOR-TYPEDNESS selects the dialect path (a table row), else the
  base value pack serves the scalar subtree unchanged. A dialect
  extends by rows, not by wrapping rules.
- **The two-pass mechanism (owner-ruled).** Pass 1
  (``check_fold_step_supported``) walks a region and refuses anything
  the machinery cannot handle, WITH the reason; pass 2 does the work.

The labeling frame (charts/labels/levels) is IN type identity — the
degenerate frame stays implicit, per the chart doctrine — so frames
misalign at emission and content keys distinguish them; dtype stays at
the compute layer (tensor.py's doctrine) until a backend column asks
(C5+). Schedules (store-all vs revolve) are EVALUATION STRATEGIES over
the same regions — never IR (the C3b finding); an L4 certified descent
may bake one later.
"""

from __future__ import annotations

import ast as pyast
import struct
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from pdum.dsl.derivative import TABLE, Const, Prim
from pdum.dsl.ir import Builder, Region
from pdum.dsl.lower import Lowerer, check_coherence
from pdum.dsl.ops import CORE_OPS, PURE, OpDef
from pdum.dsl.pack import ABI_OPS
from pdum.dsl.registry import DEFAULT
from pdum.dsl.types import Type, f64
from pdum.dsl.value import LOWER_RULES, _assign, _binop, _call, _compare

from .compute import _tensor_like, const_like
from .compute import contract as _eager_contract
from .compute import extent as _eager_extent
from .compute import iota as _eager_iota
from .compute import pointwise as _eager_pw
from .compute import reduce as _eager_reduce
from .compute import repeat_like as _eager_repeat_like
from .compute import scan as _eager_scan
from .ir import _LAYOUT_OPS, Instr, Token, _dense_like, _store, eval_instr, infer_instr, pw_marker
from .lifting import _HOST_BIN, _HOST_CMP, _METHODS, _STRUCTURAL_SLOT, _Intrinsic
from .markers import Marker
from .tensor import Tensor, alignment

# --- types -------------------------------------------------------------------


def _dim_key(d):
    """One dim's identity entry: (name, start, stop), growing the labeling
    frame (chart, labels, level) only when one exists — plain integer
    indexing is the degenerate frame left implicit (chart.py's doctrine,
    mirrored in the type)."""
    if d.chart is None and d.labels is None and d.level is None:
        return (d.name, d.start, d.stop)
    return (d.name, d.start, d.stop, d.chart, d.labels, d.level)


@dataclass(frozen=True)
class TensorType(Type):
    """A tensor-typed SSA value, constructed FROM a Layout. IDENTITY (what
    alignment, fold binders, and content keys compare) is the layout's
    observable frame: per dim its name, domain, and labeling frame. The
    full Layout is the REQUIRED non-identity shadow — strides/offset are
    representation, read only by inference through the incumbent
    authority. dtype stays at the compute layer (tensor.py's doctrine)
    until a backend column asks (C5+)."""

    layout: object = field(compare=False, repr=False)  # the shadow — required
    dims: tuple = field(init=False)  # identity, derived from the layout

    def __post_init__(self):
        object.__setattr__(self, "dims", tuple(_dim_key(d) for d in self.layout.dims))


@dataclass(frozen=True)
class TokenType(Type):
    """The ordering token — stores consume and produce it (dataflow)."""


def tensor_type(t: Tensor) -> TensorType:
    return TensorType(t.layout)


def tensor_type_of_layout(lay) -> TensorType:
    return TensorType(lay)


_of_layout = tensor_type_of_layout


def _minus_dim(tt: TensorType, name: str) -> TensorType:
    # the incumbent fold-element shadow: dense over the surviving dims
    return TensorType(_dense_like(tuple(d for d in tt.layout.dims if d.name != name)))


# --- the ops, typed by rules (tl's alignment law AS type rules) --------------


def _r_pointwise(args, attrs, regions):
    ts = [a for a in args if isinstance(a, TensorType)]
    if not ts:
        raise TypeError("tl.pointwise wants at least one tensor operand")
    # tl's alignment law, by ITS OWN diagnosis engine (name-based, order-free,
    # frames compared) — the fixes are the incumbent recipes; scalars broadcast
    issues = alignment(*(SimpleNamespace(layout=t.layout) for t in ts))
    if issues:
        details = "\n".join(f"  {m!r}" for m in issues)
        raise TypeError(f"tl.pointwise wants ALIGNED operands:\n{details}")
    return TensorType(infer_instr(Instr("out", "pointwise", ("a0",), {}), {"a0": ts[0].layout}))


def _r_iota(args, attrs, regions):
    (t,) = args
    if not isinstance(t, TensorType):
        raise TypeError("tl.iota wants the lattice source tensor")
    if attrs["name"] not in [d[0] for d in t.dims]:
        raise TypeError(f"tl.iota: the lattice has no dim {attrs['name']!r}")
    return TensorType(infer_instr(Instr("out", "iota", ("a0",), {"name": attrs["name"]}), {"a0": t.layout}))


def _r_read(args, attrs, regions):
    """A buffer READ at computed integer indices (P8, S.3's third
    extension): one index FIELD per dim of the read tensor; the result
    lives on the index fields' lattice. Gradient-free through the
    indices; the adjoint through the read tensor is scatter (P9) — the
    VJP pass refuses until then."""
    tex, *idx = args
    if not isinstance(tex, TensorType):
        raise TypeError("tl.read wants a tensor to read")
    ts = [i for i in idx if isinstance(i, TensorType)]
    if len(idx) != len(tex.dims) or not ts:
        raise TypeError(
            f"tl.read wants one integer index field per dim of the read tensor "
            f"({len(tex.dims)} dims, {len(idx)} indices given)"
        )
    issues = alignment(*(SimpleNamespace(layout=t.layout) for t in ts))
    if issues:
        details = "\n".join(f"  {m!r}" for m in issues)
        raise TypeError(f"tl.read wants ALIGNED index fields:\n{details}")
    return TensorType(infer_instr(Instr("out", "pointwise", ("a0",), {}), {"a0": ts[0].layout}))


def _r_store(args, attrs, regions):
    tok, dst, val = args
    if not isinstance(tok, TokenType):
        raise TypeError("tl.store threads the ordering token first")
    if isinstance(val, TensorType):
        if frozenset(dst.dims) != frozenset(val.dims):
            raise TypeError(f"tl.store wants the value aligned to the target: {dst!r} vs {val!r}")
    # a scalar-typed value broadcasts over the target lattice (pointwise's law)
    return TokenType()


def _r_fold(args, attrs, regions):
    init, src = args
    (step,) = regions
    if not (isinstance(init, TensorType) and isinstance(src, TensorType)):
        raise TypeError("tl.fold wants (init state, element source) tensors")
    elem = _minus_dim(src, attrs["dim"])
    s_t, m_t = (p.type for p in step.params)
    if s_t != init:
        raise TypeError(f"tl.fold: the state binder is {s_t!r} but the init state is {init!r}")
    if m_t != elem:
        raise TypeError(f"tl.fold: the element binder is {m_t!r} but slicing {attrs['dim']!r} gives {elem!r}")
    if step.body[-1].args[0].type != init:
        raise TypeError("tl.fold: the step must yield the state type")
    return init


# dict-valued instr params (the layout-method family's kwargs). Node attrs
# are IDENTITY and must be hashable, so the dialect FREEZES these to sorted
# item tuples at emission and THAWS them at every incumbent boundary.
_DICT_PARAMS = frozenset({"parts", "ranges", "coords", "deltas", "mapping", "extents", "charts", "labels", "levels"})


def _freeze_params(params: dict) -> dict:
    return {k: tuple(sorted(v.items())) if k in _DICT_PARAMS and isinstance(v, dict) else v for k, v in params.items()}


def _thaw_params(params: dict) -> dict:
    return {
        k: dict(v)
        if k in _DICT_PARAMS and isinstance(v, tuple) and v and all(isinstance(e, tuple) and len(e) == 2 for e in v)
        else v
        for k, v in params.items()
    }


def _r_bridge(base):
    """The migration bridge (240 C4.3): the type rule IS the incumbent
    shadow inference — build the instruction, ask ``infer_instr``. One
    source of truth for layout semantics while both worlds coexist."""

    def rule(args, attrs, regions):
        names = tuple(f"a{i}" for i in range(len(args)))
        shadows = {}
        for n, t in zip(names, args):
            if not isinstance(t, TensorType):
                raise TypeError(f"tl.{base} wants tensor operands, got {t!r}")
            shadows[n] = t.layout
        return _of_layout(infer_instr(Instr("out", base, names, _thaw_params(dict(attrs))), shadows))

    return rule


_BRIDGED = ("reduce", "scan", "materialize", "round_to", "repeat_like", "random", "with_value_units", "const") + tuple(
    _LAYOUT_OPS
)

TL_OPS = {
    # per-launch scalar slots are abi.slot — the dsl's marshaling dialect,
    # ONE concept both tiers (240 C5; the uniform channel is its kernel face)
    "tl.token": OpDef("tl.token", lambda a, at, r: TokenType(), PURE),
    "tl.iota": OpDef("tl.iota", _r_iota, PURE),
    "tl.pointwise": OpDef("tl.pointwise", _r_pointwise, PURE),
    "tl.read": OpDef("tl.read", _r_read, PURE),
    "tl.store": OpDef("tl.store", _r_store, PURE),  # the effect rides the token
    "tl.fold": OpDef("tl.fold", _r_fold, PURE, nregions=1),
    **{f"tl.{base}": OpDef(f"tl.{base}", _r_bridge(base), PURE) for base in _BRIDGED},
}


# --- the rule pack: ONE typed-rule factory; dialects extend by rows ----------

_BIN_MARKER = {pyast.Add: "add", pyast.Sub: "sub", pyast.Mult: "mul", pyast.Div: "div"}
_CMP_MARKER = {pyast.Lt: "lt", pyast.Gt: "gt", pyast.LtE: "le", pyast.GtE: "ge"}


def _typed_rule(table, base_rule, pick):
    """The op-selection pattern, once: lower the children; a tensor operand
    selects the dialect row from ``table``; otherwise the base value pack
    serves the node unchanged."""

    def rule(ctx, node):
        operands = [ctx.lower(a) for a in _operands_of(node)]
        tensorish = any(hasattr(o, "type") and isinstance(o.type, TensorType) for o in operands)
        f = table.get(type(pick(node)))
        if tensorish:
            if f is None:
                raise ValueError(f"operator {type(pick(node)).__name__} has no pointwise primitive")
            lifted = [
                o if hasattr(o, "type") else ctx.emit("core.const", node=node, type=f64, value=float(o))
                for o in operands
            ]
            return ctx.emit("tl.pointwise", *lifted, node=node, f=f)
        if not any(hasattr(o, "type") for o in operands):  # pure host math folds on the host
            host = _HOST_BIN if isinstance(node, pyast.BinOp) else _HOST_CMP
            return host[type(pick(node))](*operands)
        return base_rule(ctx, node)

    return rule


def _operands_of(node):
    if isinstance(node, pyast.BinOp):
        return (node.left, node.right)
    return (node.left, node.comparators[0])


def _pick_binop(node):
    return node.op


def _pick_cmp(node):
    return node.ops[0] if len(node.ops) == 1 else None


def _globals_of(ctx):
    return ctx.handle.pyfunc.__globals__


def capture_shim(fn):
    """The kernel/step capture SHIM: a Handle's snapshot/coherence surface
    with closure values kept RAW — bodies close over helpers, markers, and
    staged transforms (compile-time CITIZENS), and host scalars (DATA);
    neither is a typed env slot."""
    from pdum.dsl import capture as _cap

    snap = _cap._SNAPSHOTS.get(fn.__code__)
    if snap is None:
        snap = _cap._SNAPSHOTS[fn.__code__] = _cap._take_snapshot(fn)
    env = {}
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            env[name] = cell.cell_contents
        except ValueError:
            pass
    return SimpleNamespace(
        snapshot=snap,
        pyfunc=fn,
        env=env,
        freevars=fn.__code__.co_freevars,
        fntype=SimpleNamespace(template=SimpleNamespace(label=fn.__qualname__)),
        table=None,
    )


def _lookup(ctx, name):
    """One resolution order everywhere: locals -> closure freevars (raw) ->
    the body's globals."""
    v = ctx.locals.get(name)
    if v is None and isinstance(ctx.handle.env, dict):
        v = ctx.handle.env.get(name)
    if v is None:
        v = _globals_of(ctx).get(name)
    return v


class NotHost(Exception):
    """An AST node is not a host (structural) value."""


def _host(ctx, node):
    """Structural extraction: constants, tuples, host-bound names, and
    attribute chains — never lowered IR (a tensor reaching a structural
    slot refuses with the annotation fix, per the lifting doctrine)."""
    if isinstance(node, pyast.Constant):
        return node.value
    if isinstance(node, pyast.Tuple):
        return tuple(_host(ctx, e) for e in node.elts)
    if isinstance(node, pyast.UnaryOp) and isinstance(node.op, pyast.USub):
        return -_host(ctx, node.operand)
    if isinstance(node, pyast.BinOp) and type(node.op) in _HOST_BIN:  # host math (n - 1 in an extent)
        return _HOST_BIN[type(node.op)](_host(ctx, node.left), _host(ctx, node.right))
    if isinstance(node, pyast.Name):
        v = _lookup(ctx, node.id)
        if v is None or hasattr(v, "type") or isinstance(v, tuple) and any(hasattr(x, "type") for x in v):
            raise NotHost()
        if type(v).__name__ == "Param":  # a future-tensor citizen, never host data
            raise NotHost()
        return v
    if isinstance(node, pyast.Attribute):
        return getattr(_host(ctx, node.value), node.attr)
    if isinstance(node, pyast.Dict):
        return {_host(ctx, k): _host(ctx, v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, pyast.Call) and isinstance(node.func, pyast.Name):
        import builtins

        fn = _lookup(ctx, node.func.id)
        if fn is None:
            fn = getattr(builtins, node.func.id, None)
        if fn is _eager_extent:  # a structural READ of a lowered tensor's type
            t = ctx.lower(node.args[0])
            want = _host(ctx, node.args[1])
            if hasattr(t, "type") and isinstance(t.type, TensorType):
                return next(d[2] - d[1] for d in t.type.dims if d[0] == want)
        if callable(fn) and not hasattr(fn, "fp"):
            args = [_host(ctx, a) for a in node.args]
            kwargs = _host_kwargs(ctx, node.keywords)
            return fn(*args, **kwargs)  # host evaluation in a structural position
    raise NotHost()


def _host_kwargs(ctx, keywords):
    out = {}
    for kw in keywords:
        if kw.arg is None:  # a **splat of a host dict
            out.update(_host(ctx, kw.value))
        else:
            out[kw.arg] = _host(ctx, kw.value)
    return out


def _scalar_lift(ctx, node, v):
    return v if hasattr(v, "type") else ctx.emit("core.const", node=node, type=f64, value=float(v))


def _tl_call(ctx, node):
    """Call resolution, OBJECT-IDENTITY-FIRST: the tl intrinsics and the S.1
    vocabulary are recognized as the objects the body's globals actually
    hold (direct-name imports — the package attr ``compute`` is the
    DECORATOR, the P5 shadowing lesson); markers via the registry;
    everything else is the base pack."""
    if isinstance(node.func, pyast.Name):
        obj = _lookup(ctx, node.func.id)
        if isinstance(obj, _Intrinsic) and obj.name == "thread_idx":
            lattice = ctx.root.params[-1]  # the writable target (S.3 convention)
            names = [c.value for c in node.args]
            return tuple(ctx.emit("tl.iota", lattice, node=node, name=n) for n in names)  # ALWAYS a tuple
        if obj is _eager_pw:  # the S.1 STEP-tier spelling
            marker = _lookup(ctx, node.args[0].id) or ctx.context["registry"].overloads.get(node.args[0].id)
            if marker is None or not hasattr(marker, "name"):
                raise ValueError(f"pointwise wants a marker first, got {node.args[0].id!r}")
            rest = [_scalar_lift(ctx, node, ctx.lower(a)) for a in node.args[1:]]
            return ctx.emit("tl.pointwise", *rest, node=node, f=marker.name)
        if obj is const_like:  # scalar broadcast: the const IS the operand
            return ctx.lower(node.args[1])
        if obj is _eager_reduce or obj is _eager_scan:  # the S.1 spellings
            f = _host(ctx, node.args[0])
            fname = f if isinstance(f, str) else f.name
            arg1 = node.args[1]
            if isinstance(arg1, pyast.Tuple):
                operands = tuple(ctx.lower(e) for e in arg1.elts)
            else:
                operand = ctx.lower(arg1)
                operands = operand if isinstance(operand, tuple) else (operand,)
            dims = _host(ctx, node.args[2]) if len(node.args) > 2 else _host(ctx, node.keywords[0].value)
            key = "dims" if obj is _eager_reduce else "dim"
            op = "tl.reduce" if obj is _eager_reduce else "tl.scan"
            return ctx.emit(op, *operands, node=node, f=fname, **{key: dims})
        if obj is _eager_repeat_like:  # THE alignment primitive (S.1)
            x, like = (ctx.lower(a) for a in node.args)
            return ctx.emit("tl.repeat_like", x, like, node=node)
        if obj is _eager_iota:
            src = ctx.lower(node.args[0])
            extra = {"unit": _host(ctx, node.args[2])} if len(node.args) > 2 else {}
            return ctx.emit("tl.iota", src, node=node, name=_host(ctx, node.args[1]), **extra)
        if obj is _eager_extent:  # a structural READ: host data from the TYPE
            t = ctx.lower(node.args[0])
            want = _host(ctx, node.args[1])
            return next(d[2] - d[1] for d in t.type.dims if d[0] == want)
        if obj is _eager_contract:  # ONE visible line over the primitives (S.1)
            a, bb = (ctx.lower(x) for x in node.args)
            axis = _host(ctx, node.keywords[0].value) if node.keywords else _host(ctx, node.args[2])
            names = (axis,) if isinstance(axis, str) else tuple(axis)
            ra = ctx.emit("tl.repeat_like", a, bb, node=node)
            rb = ctx.emit("tl.repeat_like", bb, a, node=node)
            prod = ctx.emit("tl.pointwise", ra, rb, node=node, f="mul")
            return ctx.emit("tl.reduce", prod, node=node, f="sum", dims=names)
        obj2 = obj if obj is not None else ctx.context["registry"].overloads.get(node.func.id)
        if isinstance(obj2, Marker) or type(obj2).__name__ == "CompositeMarker":
            args = [_scalar_lift(ctx, node, ctx.lower(a)) for a in node.args]
            if any(hasattr(a, "type") and isinstance(a.type, TensorType) for a in args):
                if ctx.context.get("tl.kind") != "compute":  # the STEP tier spells pointwise
                    raise ValueError(
                        f"{obj2.name} is a marker — tensor-tier application is spelled "
                        f"pointwise({obj2.name}, ...) (bare names lower only inside "
                        f"scalar marker bodies)"
                    )
                return ctx.emit("tl.pointwise", *args, node=node, f=obj2.name)
        if callable(obj) and not isinstance(obj, (Marker, _Intrinsic)):
            try:
                h_args = [_host(ctx, a) for a in node.args]
                h_kwargs = _host_kwargs(ctx, node.keywords)
            except NotHost:
                h_args = None
            if h_args is not None:
                if getattr(obj, "__staged__", False):  # the declared staging door (C1/C2)
                    result = obj(*h_args, **h_kwargs)
                    if getattr(result, "fp", None) is None:
                        raise ValueError(
                            f"staged transform {node.func.id!r} returned "
                            f"{type(result).__name__!r}, not a function citizen — staged "
                            f"transforms produce fp-carrying values"
                        )
                    ctx.context.setdefault("tl.recipes", {})[id(result)] = (obj, tuple(h_args), dict(h_kwargs))
                    return result
                result = obj(*h_args, **h_kwargs)  # structural host evaluation (implicit)
                if getattr(result, "fp", None) is not None:
                    raise ValueError(
                        f"{node.func.id!r} returned a function value at lower time without "
                        f"being a declared staged transform — decorate it with @staged "
                        f"(pdum.dsl.staged), or build the value outside the body"
                    )
                return result
            if hasattr(obj, "__code__"):  # a captured helper over lowered values: INLINE
                if getattr(obj, "__module__", "") in ("pdum.tl.compute", "pdum.tl.ir"):
                    raise ValueError(
                        f"{node.func.id!r} is tensor-library machinery — it lowers by "
                        f"recognition, not inlining; the recognized set is the primitive set"
                    )
                args_l = tuple(_arg_value(ctx, a) for a in node.args)
                kwargs_l = {}
                for kw in node.keywords:
                    if kw.arg is None:
                        kwargs_l.update(_host(ctx, kw.value))
                    else:
                        kwargs_l[kw.arg] = _arg_value(ctx, kw.value)
                return _inline_plain(ctx, obj, args_l, node, kwargs_l)
    if isinstance(node.func, pyast.Attribute):  # the frozen layout-method family
        base = ctx.lower(node.func.value)
        name = node.func.attr
        if hasattr(base, "type") and isinstance(base.type, TensorType):
            if name not in _METHODS:
                raise ValueError(f"tensors have no method {name!r} in step bodies")
            try:
                args = [_host(ctx, a) for a in node.args]
                kwargs = _host_kwargs(ctx, node.keywords)
            except NotHost:
                raise ValueError(_STRUCTURAL_SLOT.format(what=f".{name}(...)")) from None
            op, pack = _METHODS[name]
            return ctx.emit(f"tl.{op}", base, node=node, **_freeze_params(pack(args, kwargs)))
    return _call(ctx, node)


def _arg_value(ctx, a):
    """An argument for helper inlining: host values (strings, tuples with
    host math, captured objects) stay host; everything else lowers.
    Tuples of lowered values ride as tuples."""
    try:
        return _host(ctx, a)
    except NotHost:
        if isinstance(a, pyast.Tuple):
            return tuple(_arg_value(ctx, e) for e in a.elts)
        return ctx.lower(a)


def _inline_plain(ctx, fn, args, node, kwargs=None):
    """Capture-and-call: a plain helper inlines through a CHILD lowerer over
    the same rules and the same build context — its bindings claim, and its
    claims collide honestly (the naming law reaches inlined bodies).
    Keyword-only parameters bind from the call's kwargs, then defaults."""
    handle = capture_shim(fn)
    check_coherence(handle)
    child = Lowerer(handle, ctx.rules, ctx.ops, ctx.derived, wrap=ctx.loc(node), context=ctx.context, root=ctx.root)
    code = fn.__code__
    pos = list(code.co_varnames[: code.co_argcount])
    kwonly = list(code.co_varnames[code.co_argcount : code.co_argcount + code.co_kwonlyargcount])
    kwargs = dict(kwargs or {})
    if len(args) > len(pos):
        raise TypeError(f"{fn.__qualname__} takes {len(pos)} positional arguments, got {len(args)}")
    child.locals.update(zip(pos, args))
    for name in pos[len(args) :]:
        if name in kwargs:
            child.locals[name] = kwargs.pop(name)
        elif fn.__defaults__ and name in pos[len(pos) - len(fn.__defaults__) :]:
            child.locals[name] = fn.__defaults__[pos.index(name) - (len(pos) - len(fn.__defaults__))]
        else:
            raise TypeError(f"{fn.__qualname__} missing argument {name!r}")
    for name in kwonly:
        if name in kwargs:
            child.locals[name] = kwargs.pop(name)
        elif fn.__kwdefaults__ and name in fn.__kwdefaults__:
            child.locals[name] = fn.__kwdefaults__[name]
        else:
            raise TypeError(f"{fn.__qualname__} missing keyword-only argument {name!r}")
    if kwargs:
        raise TypeError(f"{fn.__qualname__} got unexpected keyword arguments {sorted(kwargs)}")
    return child.run_body()


def _tl_assign(ctx, node):
    tgt = node.targets[0]
    if isinstance(tgt, pyast.Subscript):  # the store (full aligned lattice; indices join C4.2)
        target, value = ctx.lower(tgt.value), ctx.lower(node.value)
        tok = ctx.context.get("tl.token") or ctx.emit("tl.token", node=node)
        ctx.context["tl.token"] = ctx.emit("tl.store", tok, target, value, node=node)
        return None
    value = ctx.lower(node.value)  # dispatches through the ACTIVE pack's rules
    if isinstance(tgt, pyast.Tuple) and isinstance(value, tuple):  # y, x = thread_idx(...)
        for e, v in zip(tgt.elts, value):
            ctx.locals[e.id] = v
            _record_name(ctx, e.id, v)
        return None
    if isinstance(tgt, pyast.Name):
        ctx.locals[tgt.id] = value
        _record_name(ctx, tgt.id, value)
        return None
    return _assign(ctx, node)


def _record_name(ctx, name, value):
    """The naming law: binding names become SSA names — recorded here,
    consumed by the exporter (first binding wins)."""
    if hasattr(value, "op"):
        ctx.context.setdefault("tl.names", {}).setdefault(id(value), name)


def _tl_name(ctx, node):
    """Locals first; captured host values return RAW (the incumbent step
    semantics — scalars bake; the literal-doctrine uniform channel reaches
    the step tier with its own slice, recorded in 240); unknown names speak
    the step voice."""
    if node.id in ctx.locals:
        return ctx.locals[node.id]
    v = _lookup(ctx, node.id)
    if v is not None:
        return v
    raise ValueError(f"unknown name {node.id!r} in a step body")


def _refuse_straightline(ctx, node):
    raise ValueError(
        f"step bodies are straight-line: a {type(node).__name__} statement cannot be "
        f"lowered — bounded control flow exists only in the value language (S.2)"
    )


def _refuse_branch_expr(ctx, node):
    raise ValueError(
        "step bodies are straight-line: if/and/or cannot be lowered — "
        "use where(cond, a, b); the branch is data flow here"
    )


def _tl_subscript(ctx, node):
    base = ctx.lower(node.value)
    if isinstance(base, tuple):  # host-aggregate destructure
        return base[_host(ctx, node.slice)]
    if hasattr(base, "type") and isinstance(base.type, TensorType):
        raise ValueError("tensor subscripts do not exist here — use .slice()/.select()")
    from pdum.dsl.value import _subscript as _base_subscript

    return _base_subscript(ctx, node)


def _tl_attribute(ctx, node):
    base = ctx.lower(node.value)
    if hasattr(base, "type") and isinstance(base.type, TensorType):
        raise ValueError(f"tensors have no attribute access in step bodies (.{node.attr})")
    if not hasattr(base, "type"):
        return getattr(base, node.attr)  # host attribute (red.mean, cfg.d)
    from pdum.dsl.value import _attribute as _base_attribute

    return _base_attribute(ctx, node)


def _tl_expr_stmt(ctx, node):
    """An effectful statement call (taps); docstrings vanish."""
    if isinstance(node.value, pyast.Constant):
        return None
    if isinstance(node.value, pyast.Call):
        ctx.lower(node.value)
        return None
    from pdum.dsl.value import _expr_stmt as _base_expr_stmt

    return _base_expr_stmt(ctx, node)


def _tl_tuple(ctx, node):
    """tl bodies: tuples are HOST aggregates of values (the incumbent
    semantics) — destructured structurally, never core.tuple interiors;
    a region's yield materializes the terminal tuple."""
    return tuple(ctx.lower(e) for e in node.elts)


def _tl_constant(ctx, node):
    """tl bodies: constants are HOST values (the incumbent semantics) —
    they lift to IR only on meeting tensors (scalar broadcast)."""
    return node.value


def _tl_unary(ctx, node):
    if isinstance(node.op, pyast.USub):
        v = ctx.lower(node.operand)
        if hasattr(v, "type") and isinstance(v.type, TensorType):
            return ctx.emit("tl.pointwise", v, node=node, f="neg")
        if not hasattr(v, "type"):
            return -v
    from pdum.dsl.value import _unary as _base_unary

    return _base_unary(ctx, node)


TL_RULES = {
    **LOWER_RULES,
    pyast.Expr: _tl_expr_stmt,
    pyast.Tuple: _tl_tuple,
    pyast.Constant: _tl_constant,
    pyast.UnaryOp: _tl_unary,
    pyast.Assign: _tl_assign,
    pyast.BinOp: _typed_rule(_BIN_MARKER, _binop, _pick_binop),
    pyast.Compare: _typed_rule(_CMP_MARKER, _compare, _pick_cmp),
    pyast.Call: _tl_call,
    pyast.Name: _tl_name,
    pyast.Subscript: _tl_subscript,
    pyast.Attribute: _tl_attribute,
    pyast.If: _refuse_straightline,
    pyast.For: _refuse_straightline,
    pyast.While: _refuse_straightline,
    pyast.IfExp: _refuse_branch_expr,
    pyast.BoolOp: _refuse_branch_expr,
}


# --- the per-kind yield protocol + the body driver ---------------------------


def lower_body(
    fn, arg_types: tuple, *, kind: str, registry=None, host: dict | None = None, out_names: dict | None = None
) -> Region:
    """Lower a body through the DSL Lowerer with the tl pack. WHAT the
    region yields is the KIND's declaration (the per-kind yield protocol):
    a ``step`` yields its returned value; a ``compute`` kernel yields the
    final ordering token — and a kernel ``return`` refuses."""
    handle = capture_shim(fn)
    check_coherence(handle)
    ctx = Lowerer(handle, TL_RULES, {**CORE_OPS, **TL_OPS, **ABI_OPS}, {}, context={"registry": registry or DEFAULT})
    ctx.context["tl.kind"] = kind
    if out_names is not None:
        ctx.context["tl.names"] = out_names  # the naming law's ledger, for the exporter
    params = tuple(ctx.builder.param(i, t) for i, t in enumerate(arg_types))
    ctx.params = params
    names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
    ctx.locals.update(zip(names, params))
    ctx.locals.update(host or {})  # structural/kw bindings (dim names, scalars)
    from .producer import _fn_ast

    tree = _fn_ast(fn)  # a def OR a lambda (the incumbent extractor)
    if kind == "step":
        if isinstance(tree, pyast.Lambda):
            result = ctx.lower(tree.body)
        else:
            for stmt in tree.body[:-1]:
                ctx.lower(stmt)
            last = tree.body[-1]
            if not isinstance(last, pyast.Return) or last.value is None:
                raise ValueError("a step body must end in `return <tensor(s)>`")
            result = ctx.lower(last.value)
        flat = result if isinstance(result, tuple) else (result,)
        if not all(hasattr(v, "type") and isinstance(v.type, TensorType) for v in flat):
            raise ValueError(f"a step must return tensors, got {flat!r}")
        yielded = result if not isinstance(result, tuple) else ctx.builder.emit("core.tuple", *result)
        return Region(params=params, body=(ctx.builder.emit("core.yield", yielded),))
    if kind != "compute":
        raise ValueError(f"unknown body kind {kind!r} — kinds declare their yields here")
    for stmt in tree.body:
        if isinstance(stmt, pyast.Return):
            raise ValueError("kernels return nothing — stores into writable arguments are the effect")
        ctx.lower(stmt)
    tok = ctx.context.get("tl.token")
    if tok is None:
        raise ValueError(f"{fn.__qualname__} stores nothing — a kernel's effect is its stores")
    return Region(params=params, body=(ctx.builder.emit("core.yield", tok),))


# --- pass 1 of the two-pass mechanism ----------------------------------------

_FOLD_STEP_SUPPORTED = {"tl.pointwise", "core.const", "core.param", "core.yield"}


def walk_region(region: Region):
    seen: set[int] = set()

    def go(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for a in n.args:
            yield from go(a)
        yield n

    for node in region.body:
        yield from go(node)


def check_fold_step_supported(step: Region) -> None:
    """PASS 1 (owner-ruled): is this step a shape the fold machinery
    handles? Anything else refuses NOW, with the reason — never
    mid-derivation."""
    for n in walk_region(step):
        if n.op not in _FOLD_STEP_SUPPORTED:
            raise TypeError(
                f"tl.fold step contains {n.op!r}, which the fold adjoint does not support yet "
                f"(supported: {', '.join(sorted(_FOLD_STEP_SUPPORTED))}) — the adjoint derives "
                f"per-op through the one derivative table, and {n.op!r} has no region rule here"
            )
        if n.op == "tl.pointwise":
            f = dict(n.attrs)["f"]
            if f not in TABLE:
                raise TypeError(
                    f"tl.fold step applies marker {f!r}, which has no row in the derivative "
                    f"table — the table grows only when a primitive joins the core"
                )


# --- the VJP: region-in, region-out ------------------------------------------


def _splice_tl(b, tree):
    """A table slope tree -> dialect IR: Prim -> tl.pointwise, schema Const
    -> a scalar core.const; a dsl Node leaf (a primal operand) rides as-is."""
    if isinstance(tree, Prim):
        return b.emit("tl.pointwise", *(_splice_tl(b, a) for a in tree.args), f=tree.op)
    if isinstance(tree, Const):
        return b.emit("core.const", type=f64, value=float(tree.value))
    return tree


def _substitute(b, region: Region, param_map: dict):
    """Rebuild the region's DAG over new params; return (out, order)."""
    sub_env = dict(param_map)
    order: list = []

    def sub(n):
        if id(n) in sub_env:
            return sub_env[id(n)]
        if n.op == "core.const":
            made = b.emit("core.const", type=n.type, value=dict(n.attrs)["value"])
        else:
            made = b.emit(n.op, *(sub(a) for a in n.args), **dict(n.attrs))
        sub_env[id(n)] = made
        order.append(made)
        return made

    return sub(region.body[-1].args[0]), order


def _pullback(b, order: list, out, seed) -> dict:
    """Reverse accumulation over a substituted DAG — the ONE region adjoint
    walker (240 C4.3b). Per-op rules: pointwise slopes spliced from THE
    table; reduce sum/mean (repeat back, mean divides by the static reduced
    numel); repeat_like (reduce-sum over the added dims — the like operand
    is layout-reference only, per doctrine). Returns id(node) -> adjoint;
    absence is an exact zero."""
    adj: dict[int, object] = {id(out): seed}

    def acc(operand, term):
        prev = adj.get(id(operand))
        adj[id(operand)] = term if prev is None else b.emit("tl.pointwise", prev, term, f="add")

    for node in reversed(order):
        a = adj.get(id(node))
        if a is None:
            continue
        if node.op == "tl.pointwise":
            rules = TABLE[dict(node.attrs)["f"]]
            for rule, operand in zip(rules, node.args):
                if rule is None or operand.op == "core.const":
                    continue
                acc(operand, b.emit("tl.pointwise", _splice_tl(b, rule(*node.args)), a, f="mul"))
        elif node.op == "tl.reduce":
            f = dict(node.attrs)["f"]
            (operand,) = node.args
            rep = b.emit("tl.repeat_like", a, operand)
            if f == "mean":
                red = {d[0] for d in operand.type.dims} - {d[0] for d in node.type.dims}
                n_red = 1
                for d in operand.type.dims:
                    if d[0] in red:
                        n_red *= d[2] - d[1]
                rep = b.emit("tl.pointwise", rep, b.emit("core.const", type=f64, value=float(n_red)), f="div")
            acc(operand, rep)
        elif node.op == "tl.repeat_like":
            x = node.args[0]
            added = tuple(d[0] for d in node.type.dims if d[0] not in {q[0] for q in x.type.dims})
            acc(x, b.emit("tl.reduce", a, f="sum", dims=added))
    return adj


_VJP_SUPPORTED = {"tl.pointwise", "tl.reduce", "tl.repeat_like", "core.const", "core.param", "core.yield"}
_VJP_REDUCERS = {"sum", "mean"}


def check_vjp_supported(region: Region) -> None:
    """PASS 1 for the general VJP: refuse — with the reason — anything the
    adjoint walker cannot derive through yet. The engine grows per-op, by
    declaration, never by silent guessing."""
    for n in walk_region(region):
        if n.op not in _VJP_SUPPORTED:
            raise TypeError(
                f"derive_vjp: {n.op!r} has no region-adjoint rule yet (supported: {', '.join(sorted(_VJP_SUPPORTED))})"
            )
        if n.op == "tl.pointwise" and dict(n.attrs)["f"] not in TABLE:
            raise TypeError(f"derive_vjp: marker {dict(n.attrs)['f']!r} has no row in the derivative table")
        if n.op == "tl.reduce" and dict(n.attrs)["f"] not in _VJP_REDUCERS:
            raise TypeError(
                f"derive_vjp: reducer {dict(n.attrs)['f']!r} adjoint arrives with the "
                f"first-occurrence-mask slice — sum/mean today"
            )


def derive_vjp(region: Region, ops=None) -> Region:
    """The GENERAL straight-line region VJP (240 C4.3b), region-in/
    region-out: params are the originals plus the upstream adjoint of the
    yield; yields a core.tuple of adjoints, one per original param (an
    exact-zero adjoint materializes as a zero field aligned to its param).
    Two-pass: check first, derive second."""
    check_vjp_supported(region)
    b = Builder(ops or {**CORE_OPS, **TL_OPS})
    new_params = tuple(b.param(("v", i), p.type) for i, p in enumerate(region.params))
    p_seed = b.param(("v", len(new_params)), region.body[-1].args[0].type)
    out, order = _substitute(b, region, {id(p): q for p, q in zip(region.params, new_params)})
    adj = _pullback(b, order, out, p_seed)
    outs = []
    for p in new_params:
        a = adj.get(id(p))
        if a is None:  # exact zero, aligned to the param
            a = b.emit("tl.pointwise", p, b.emit("core.const", type=f64, value=0.0), f="mul")
        outs.append(a)
    result = outs[0] if len(outs) == 1 else b.emit("core.tuple", *outs)
    return Region(params=(*new_params, p_seed), body=(b.emit("core.yield", result),))


def derive_step_vjp(step: Region, ops=None) -> Region:
    """The fold step's adjoint, AS A REGION: params (state, element,
    upstream state-adjoint); yields d(state). The element's adjoint must
    be gradient-free (the dropout-mask discipline) — checked. A thin role
    assignment over the ONE walker."""
    check_fold_step_supported(step)  # pass 1, always
    b = Builder(ops or {**CORE_OPS, **TL_OPS})
    s_t, m_t = (p.type for p in step.params)
    p_s, p_m, p_ds = b.param(("v", 0), s_t), b.param(("v", 1), m_t), b.param(("v", 2), s_t)
    out, order = _substitute(b, step, {id(step.params[0]): p_s, id(step.params[1]): p_m})
    adj = _pullback(b, order, out, p_ds)
    if adj.get(id(p_m)) is not None:
        raise TypeError("derive_step_vjp: the element adjoint must be gradient-free in this slice")
    ds = adj.get(id(p_s))
    if ds is None:
        raise TypeError("derive_step_vjp: the state never reaches the output")
    return Region(params=(p_s, p_m, p_ds), body=(b.emit("core.yield", ds),))


# --- the evaluation column (ir.run's successor for this dialect) -------------

# scalar value-dialect ops that can ride inside a kernel region (a spliced
# fn-argument's all-scalar subtrees, 240 C5.3): evaluated by THE markers,
# which compute on host scalars by law
_HOST_OPS = {"core.add": "add", "core.sub": "sub", "core.mul": "mul", "core.div": "div", "core.neg": "neg"}
_HOST_OPS["core.select"] = "where"


def run_region(region: Region, values: list, uniforms: bytes | None = None):
    """Evaluate a dialect region over tl Tensors — fields slice at ABSOLUTE
    coordinates, so closed-form random fields regenerate exactly. Stores
    write through the target's buffer (the ONE effect, token-ordered);
    ``uniforms`` is the launch's packed staging bytes, read by ``abi.slot``
    at its offset/fmt (unmarked captures are DATA — the literal doctrine,
    240 C4.2b — riding the dsl's marshaling discipline since C5)."""
    memo: dict[int, object] = {}
    by_param = {id(p): v for p, v in zip(region.params, values)}

    def ev(n):
        if id(n) not in memo:
            memo[id(n)] = _ev(n)
        return memo[id(n)]

    def _ev(n):
        attrs = dict(n.attrs)
        if id(n) in by_param:
            return by_param[id(n)]
        if n.op == "core.const":
            return attrs["value"]
        if n.op == "core.yield":
            return ev(n.args[0])
        if n.op == "core.tuple":
            return tuple(ev(a) for a in n.args)
        if n.op == "core.extract":
            return ev(n.args[0])[attrs["index"]]
        if n.op == "tl.token":
            return Token()
        if n.op == "abi.slot":
            return struct.unpack_from(attrs["fmt"], uniforms, attrs["offset"])[0]
        if n.op == "tl.iota":
            return _eager_iota(ev(n.args[0]), attrs["name"])
        if n.op == "tl.pointwise":
            ops_v = [ev(a) for a in n.args]
            ref = next(v for v in ops_v if isinstance(v, Tensor))
            ops_v = [v if isinstance(v, Tensor) else const_like(ref, float(v)) for v in ops_v]
            return _eager_pw(pw_marker(attrs["f"]), *ops_v)
        if n.op == "tl.read":
            tex, *idxs = (ev(a) for a in n.args)
            ref = next(v for v in idxs if isinstance(v, Tensor))
            names = tuple(d.name for d in ref.layout.dims)
            shape = [d.size for d in ref.layout.dims]
            arrs = []
            for v in idxs:
                a = v.to_numpy(order=names) if isinstance(v, Tensor) else np.full(shape, float(v))
                if not np.all(np.floor(a) == a):
                    raise ValueError("tl.read: read indices must be exact integers (the carrier discipline)")
                arrs.append(a.astype(np.int64))
            for a, d in zip(arrs, tex.layout.dims):
                if np.any((a < d.start) | (a >= d.stop)):
                    raise ValueError(
                        f"tl.read: index out of bounds on dim {d.name!r} (domain [{d.start}, {d.stop})) — "
                        f"the reference REFUSES out-of-bounds (an oracle has no undefined behavior); "
                        f"device backends need not check"
                    )
            src = tex.to_numpy(order=tuple(d.name for d in tex.layout.dims))
            out = src[tuple(a - d.start for a, d in zip(arrs, tex.layout.dims))]
            return _tensor_like(np.asarray(out, dtype=np.float64), ref.layout.dims)
        if n.op == "tl.store":
            tok, dst, val = (ev(a) for a in n.args)
            if not isinstance(val, Tensor):  # scalars broadcast (pointwise's law)
                val = const_like(dst, float(val))
            return _store(tok, dst, val)
        if n.op == "core.cmp":  # scalar value-dialect rows: spliced fn-arg
            return pw_marker(attrs["pred"])(*(ev(a) for a in n.args))  # subtrees (240 C5.3)
        if n.op in _HOST_OPS or n.op.startswith("pw."):
            return pw_marker(_HOST_OPS.get(n.op, n.op[3:]))(*(ev(a) for a in n.args))
        if n.op.startswith("tl.") and n.op[3:] in _BRIDGED:
            ops_v = [ev(a) for a in n.args]
            names = tuple(f"a{i}" for i in range(len(ops_v)))
            return eval_instr(Instr("out", n.op[3:], names, _thaw_params(attrs)), dict(zip(names, ops_v)))
        if n.op == "tl.fold":
            init, src = ev(n.args[0]), ev(n.args[1])
            dim = attrs["dim"]
            lo, hi = next((d[1], d[2]) for d in n.args[1].type.dims if d[0] == dim)
            s = init
            for q in range(lo, hi):
                s = run_region(n.regions[0], [s, src.select(**{dim: q})])
            return s
        raise AssertionError(f"run_region: unexpected op {n.op!r}")

    return ev(region.body[-1])


# --- schedules: EVALUATION STRATEGIES over the same regions (never IR) -------


def fold_grad(step: Region, vjp: Region, init, src, dim: str, *, slots: int | None = None):
    """d(sum(final state))/d(init) — store-all when ``slots`` is None, else
    segment-checkpoint revolve: recompute each segment from its checkpoint
    during the backward, RE-SELECTING elements at absolute coordinates
    (the recompute theorem's mechanism; bit-identical by construction)."""
    src_dims = [(d.name, d.start, d.stop) for d in src.layout.dims]
    lo, hi = next((s, e) for (name, s, e) in src_dims if name == dim)
    stride = hi - lo if slots is None else -(-(hi - lo) // slots)
    checkpoints = {lo: init}
    s = init
    for q in range(lo, hi):
        if q != lo and (q - lo) % stride == 0:
            checkpoints[q] = s
        s = run_region(step, [s, src.select(**{dim: q})])
    ds = const_like(s, 1.0)
    starts = sorted(checkpoints)
    for seg in reversed(range(len(starts))):
        base = starts[seg]
        end = starts[seg + 1] if seg + 1 < len(starts) else hi
        states = [checkpoints[base]]
        for q in range(base, end):  # the recomputation the theorem is about
            states.append(run_region(step, [states[-1], src.select(**{dim: q})]))
        for q in reversed(range(base, end)):
            ds = run_region(vjp, [states[q - base], src.select(**{dim: q}), ds])
    return ds


# --- the migration view: regions rendered as incumbent Programs --------------


def export_program(region: Region, param_names: tuple, names_of: dict | None = None):
    """The MIGRATION VIEW (240 C4.3c): render a dialect region as an
    incumbent ``Program``, so every Program consumer — autodiff (max/scan/
    layout adjoints included), signatures, opcount, memory — serves regions
    UNCHANGED while lowering moves onto the one engine. Adjoint knowledge
    stays single-copy (the incumbent's, battle-tested); this view dies when
    the last consumer retargets. Returns ``(program, output_vars)``."""
    from pdum.dsl.naming import Namer

    from .ir import Program

    names = Namer()
    names_of = names_of or {}
    instrs: list = []
    var_of: dict[int, str] = {}
    for pname, p in zip(param_names, region.params):
        names.claim(pname)
        instrs.append(Instr(pname, "input", (), {}))
        var_of[id(p)] = pname

    def emit(op, operands, hint, **params):
        var = names.derive(hint)
        instrs.append(Instr(var, op, tuple(operands), params))
        return var

    def const_for(value, ref):
        """A scalar const materialized over the REFERENCE operand's layout,
        charts/labels/placement restamped so eager alignment holds (the
        incumbent const_like discipline, ported)."""
        lay = ref.type.layout
        dims = tuple((d.name, (d.start, d.stop)) for d in lay.dims)
        v = emit("const", (), "c", value=float(value), dims=dims)
        charts = {d.name: d.chart for d in lay.dims if d.labels is None}
        labels = {d.name: d.labels for d in lay.dims if d.labels is not None}
        if any(c is not None for c in charts.values()):
            v = emit("with_charts", (v,), "c", charts=charts)
        if labels:
            v = emit("with_labels", (v,), "c", labels=labels)
        levels = {d.name: d.level for d in lay.dims}
        if any(lv is not None for lv in levels.values()):
            v = emit("bind", (v,), "c", levels=levels)
        return v

    yielded = region.body[-1].args[0]
    for n in walk_region(region):
        if id(n) in var_of or n.op in ("core.yield", "core.const"):
            continue  # consts materialize at their use sites, with a reference layout
        if n.op == "tl.fold":
            raise TypeError("export_program: fold regions arrive with the assemblage slice")
        if n.op == "core.tuple":
            if n is not yielded:
                raise TypeError("export_program: interior tuples have no Program spelling")
            continue
        if not n.op.startswith("tl."):
            raise TypeError(f"export_program: {n.op!r} has no Program spelling")
        base = n.op[3:]
        ref = next((a for a in n.args if isinstance(a.type, TensorType)), None)
        operands = []
        for a in n.args:
            if a.op == "core.const":
                operands.append(const_for(dict(a.attrs)["value"], ref))
            else:
                operands.append(var_of[id(a)])
        hint = names_of.get(id(n), base.rsplit(".", 1)[-1])  # binding names become SSA names
        var_of[id(n)] = emit(base, operands, hint, **_thaw_params(dict(n.attrs)))
    outs = tuple(var_of[id(a)] for a in yielded.args) if yielded.op == "core.tuple" else (var_of[id(yielded)],)
    return Program(tuple(instrs)), outs
