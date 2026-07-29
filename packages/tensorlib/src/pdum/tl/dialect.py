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
from .coords import Frame
from .indexing import argsort as _eager_argsort
from .indexing import argtopk as _eager_argtopk
from .indexing import scatter_add as _eager_scatter_add
from .indexing import take as _eager_take
from .layout import Dim, Layout, _dense_like
from .lifting import _HOST_BIN, _HOST_CMP, _METHODS, _STRUCTURAL_SLOT, _Intrinsic
from .markers import Marker, pw_marker, reducer
from .tensor import Tensor, Token, _store, alignment

# --- types -------------------------------------------------------------------


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
    dims: tuple[Frame, ...] = field(init=False)  # identity, derived from the layout

    def __post_init__(self):
        object.__setattr__(self, "dims", tuple(d.frame for d in self.layout.dims))


@dataclass(frozen=True)
class TokenType(Type):
    """The ordering token — stores consume and produce it (dataflow)."""


@dataclass(frozen=True)
class CoordType(Type):
    """A Coordinate at the kernel tier (design 250): a typed handle whose
    frame is the dim identity and whose backing (the ``tl.coord`` op's one
    operand) is the value field realizing it — an iota, a zero field, or
    an affine chain. Value dataflow never flows THROUGH a coordinate:
    subscripts check the type and consume the backing, coercion (f32/i32)
    returns the backing, and arithmetic on the handle refuses. Only the
    ambient doors mint it — CoordType IS the ambient-derived credential."""

    frame: Frame


def tensor_type(t: Tensor) -> TensorType:
    return TensorType(t.layout)


def tensor_type_of_layout(lay) -> TensorType:
    return TensorType(lay)


_of_layout = tensor_type_of_layout


def _minus_dim(tt: TensorType, name: str) -> TensorType:
    # the incumbent fold-element shadow: dense over the surviving dims
    return TensorType(_dense_like(tuple(d for d in tt.layout.dims if d.name != name)))


# --- the ops, typed by rules (tl's alignment law AS type rules) --------------


def _iota_layout(base, name):
    """The iota shadow: a closed-form field addressed only along its dim
    (stride on the named dim, 0 elsewhere — a FunctionalBuffer citizen)."""
    from dataclasses import replace as _replace

    d = base.dim(name)
    new = tuple(_replace(x, stride=(8 if x.name == d.name else 0)) for x in base.dims)
    return Layout(new, offset=-8 * d.start)


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
    return TensorType(_dense_like(ts[0].layout.dims))


def _r_coord(args, attrs, regions):
    (backing,) = args
    if not isinstance(backing, TensorType):
        raise TypeError("tl.coord wants a lattice-valued backing")
    return CoordType(attrs["frame"])


def _r_iota(args, attrs, regions):
    (t,) = args
    if not isinstance(t, TensorType):
        raise TypeError("tl.iota wants the lattice source tensor")
    if attrs["name"] not in [d.name for d in t.dims]:
        raise TypeError(f"tl.iota: the lattice has no dim {attrs['name']!r}")
    return TensorType(_iota_layout(t.layout, attrs["name"]))


def _r_read(args, attrs, regions):
    """A buffer READ at computed integer indices (P8, S.3's third
    extension): one index FIELD per dim of the read tensor; the result
    lives on the index fields' lattice. Gradient-free through the
    indices; the adjoint through the read tensor is a scatter — the
    tensor tier owns it (scatter_add, 200 §1.9); the kernel VJP pass
    refuses until its consumer arrives (an L4-era wiring)."""
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
    return TensorType(_dense_like(ts[0].layout.dims))


def _r_sample(args, attrs, regions):
    """Texture sampling (S.4 textures v1): two aligned normalized-coord
    FIELDS (row, col); the texture/sampler are RUNTIME objects riding the
    artifact's texture ledger (attrs carry the index and the sampler's
    descriptor — compile constants). v1 limits, recorded: 2D, lod=0, one
    format, R channel; mips/arrays/formats are future work."""
    cy, cx = args
    if not (isinstance(cy, TensorType) and isinstance(cx, TensorType)):
        raise TypeError("tl.sample wants two coordinate fields (row, col), normalized to [0, 1]")
    issues = alignment(SimpleNamespace(layout=cy.layout), SimpleNamespace(layout=cx.layout))
    if issues:
        details = "\n".join(f"  {m!r}" for m in issues)
        raise TypeError(f"tl.sample wants ALIGNED coordinate fields:\n{details}")
    return TensorType(_dense_like(cy.layout.dims))


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
    """The widened fold contract (the excavation, LEVELS — the carry-dict
    debt burned): args = k state inits ++ m element sources; the step's
    binders match positionally; the step YIELDS the k next states in state
    order (+ the emitted value iff out=("emit",)); out=("final", i) returns
    state i's final value. The name tuples are contract (reporting keys);
    the carry is POSITIONAL — no name-based carry map exists."""
    at = dict(attrs)
    state, element = tuple(at["state"]), tuple(at["element"])
    k, m = len(state), len(element)
    if k < 1:
        raise TypeError("tl.fold needs at least one state tensor")
    if len(args) != k + m:
        raise TypeError(f"tl.fold takes {k} state init(s) + {m} element source(s), got {len(args)} operands")
    if not all(isinstance(a, TensorType) for a in args):
        raise TypeError("tl.fold wants tensor-typed state inits and element sources")
    (step,) = regions
    dim = at["dim"]
    if m == 0 and at.get("extent") is None:
        raise TypeError("tl.fold without element sources needs extent=(start, stop)")
    inits, srcs = args[:k], args[k:]
    for s in srcs:
        d = next((dd for dd in s.dims if dd.name == dim), None)
        if d is None:
            raise TypeError(f"tl.fold: element source has no scan dim {dim!r}")
        if d.chart is not None or d.labels is not None:
            raise TypeError(f"fold scan dim {dim!r} must be chartless/unlabeled (strip_charts first)")
    elems = tuple(_minus_dim(s, dim) for s in srcs)
    binders = tuple(p.type for p in step.params)
    want = tuple(inits) + elems

    def _frames(tt):
        # order-insensitive frame agreement: presentation order is never
        # semantics (220 §11); binders bind by position, frames by name
        return sorted(tt.dims, key=lambda d: d.name)

    if len(binders) != len(want) or any(_frames(bb) != _frames(w) for bb, w in zip(binders, want)):
        raise TypeError(f"tl.fold: step binders {binders!r} do not match (state inits + element slices) {want!r}")
    yielded = step.body[-1].args[0]
    ytypes = tuple(a.type for a in yielded.args) if yielded.op == "core.tuple" else (yielded.type,)
    out = tuple(at["out"])
    # the step declares k next-states (state order) + at most ONE emit slot;
    # out=("final", i) may run a step that also declares the slot (the
    # adjoint machinery re-outs one step region under several fold nodes)
    if len(ytypes) not in (k, k + 1):
        raise TypeError(
            f"tl.fold: the step must yield the {k} next state(s) in state order "
            f"(+ at most one emitted value) — got {len(ytypes)}"
        )
    if out[0] == "emit" and len(ytypes) != k + 1:
        raise TypeError("tl.fold: out=('emit',) needs the step to yield the emitted value after the states")
    for i in range(k):
        if _frames(ytypes[i]) != _frames(inits[i]):
            raise TypeError(f"tl.fold: carry {state[i]!r} changes the state type: {inits[i]!r} -> {ytypes[i]!r}")
    if out[0] == "final":
        return inits[out[1]]
    start, stop = _src_domain(srcs[0], dim) if m else tuple(at["extent"])
    return _of_layout(_dense_like((Dim(dim, 0, start, stop),) + tuple(ytypes[k].layout.dims)))


def _src_domain(tt: TensorType, dim: str) -> tuple[int, int]:
    d = next(dd for dd in tt.dims if dd.name == dim)
    return d.start, d.stop


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
        return _of_layout(_infer_base(base, [shadows[n] for n in names], _thaw_params(dict(attrs))))

    return rule


# --- the semantic rows, at the ops' definition site (the excavation's last
# fold: the retired linear IR's eval/infer dispatch absorbed here) ------------

# Layout/GuardedLayout share these ops' names and signatures with Tensor,
# so evaluation and inference dispatch through ONE table — a new layout op
# is added in exactly one place. (pad/stencil/simplify are the three
# genuine special cases: their shadows ride the guard machinery.)
LAYOUT_ADAPTERS = {
    "slice": lambda t, p: t.slice(**p["ranges"]),
    "select": lambda t, p: t.select(**p["coords"]),
    "shift": lambda t, p: t.shift(**p["deltas"]),
    "rename": lambda t, p: t.rename(**p["mapping"]),
    "repeat": lambda t, p: t.repeat(p["name"], p["extent"], p.get("chart"), p.get("labels")),
    "flip": lambda t, p: t.flip(p["name"]),
    "split": lambda t, p: t.split(p["name"], **p["parts"]),
    "merge": lambda t, p: t.merge(tuple(p["parts"]), p["name"], p.get("start", 0)),
    "diagonal": lambda t, p: t.diagonal(tuple(p["parts"]), p["name"], p.get("chart")),
    "window": lambda t, p: t.window(p["name"], p["k_name"], p["k"], p.get("dilation", 1)),
    "decimate": lambda t, p: t.decimate(p["name"], p["factor"], p.get("phase", 0)),
    "pad": lambda t, p: t.pad(p["fill"], **p["extents"]),
    "stencil": lambda t, p: t.stencil(p["name"], p["k"], p.get("k_name"), p.get("fill", 0), p.get("dilation", 1)),
    "strip_charts": lambda t, p: t.strip_charts(),
    "with_charts": lambda t, p: t.with_charts(**p["charts"]),
    "with_labels": lambda t, p: t.with_labels(**p["labels"]),
    "bind": lambda t, p: t.bind(**p["levels"]),
    "simplify": lambda t, p: t.simplify(),
}


def _infer_base(base: str, lays, p):
    """One op's shadow from its operands' layouts (the retired infer_instr's
    rows). Layouts only — no data, no names."""
    from .guarded import GuardedLayout, pad_layout, stencil_layout

    if base == "const":
        dims = tuple(
            Dim(name, 0, *(extent if isinstance(extent, tuple) else (0, extent))) for name, extent in p.get("dims", ())
        )
        return Layout(dims)  # stride-0 broadcast, exactly like the eval row
    if base == "random":
        return _dense_like(lays[0].dims)
    if base == "reduce":
        dims = p["dims"]
        dim_names = (dims,) if isinstance(dims, str) else tuple(dims)
        return _dense_like(tuple(d for d in lays[0].dims if d.name not in dim_names))
    if base == "scan":
        return _dense_like(lays[0].dims)
    if base == "materialize":
        from dataclasses import replace as _replace

        dims = tuple(_replace(lays[0].dim(n), chart=None, labels=None) for n in tuple(p["order"]))
        return _dense_like(dims)
    if base == "stage":
        # 320 §4: the ONE copying op with a residence — a dense copy in the
        # declared order at the named level. Charts RIDE (same physics,
        # different residence); chart-stripping stays materialize's separate
        # contract. `level` is a name resolved by cost analysis, never here.
        order = p.get("order")
        dims = tuple(lays[0].dim(n) for n in tuple(order)) if order else lays[0].dims
        return _dense_like(dims)
    if base == "round_to":
        return _dense_like(lays[0].dims)
    if base == "repeat_like":
        # the batching-unawareness mechanism (220): added dims are LAYOUT-
        # DERIVED from the like operand — referenced for its layout only
        x, like = lays[0], lays[1]
        have = {d.name for d in x.dims}
        for d in like.dims:
            if d.name not in have:
                x = x.repeat(d.name, (d.start, d.stop), d.chart, d.labels)
                if d.level is not None:
                    x = x.bind(**{d.name: d.level})
        return x
    if base == "take":
        from .indexing import infer_take

        return infer_take(lays, p)
    if base == "scatter_add":
        from .indexing import infer_scatter

        return infer_scatter(lays, p)
    if base in ("argtopk", "argsort"):
        from .indexing import infer_producer

        return infer_producer(base, lays, p)
    if base == "pad":
        return pad_layout(lays[0], p["extents"])
    if base == "stencil":
        return stencil_layout(lays[0], p["name"], p["k"], p.get("k_name"), p.get("dilation", 1))
    if base == "simplify":
        s = lays[0]
        return s.simplify() if isinstance(s, GuardedLayout) else s
    if base == "with_value_units":
        return lays[0]
    return LAYOUT_ADAPTERS[base](lays[0], p)


def _eval_base(base: str, vals, p):
    """One op's reference evaluation over Tensors (the retired eval_instr's
    rows) — the eager compute layer applied by name."""
    from .compute import _const_tensor, _materialize, _round_to

    if base == "const":
        return _const_tensor(p)
    if base == "random":
        from .random import _field

        return _field(p["dist"], p["key"], vals[0])
    if base == "reduce":
        return _eager_reduce(reducer(p["f"]), vals[0] if len(vals) == 1 else tuple(vals), p["dims"], p.get("zero"))
    if base == "scan":
        return _eager_scan(reducer(p["f"]), vals[0] if len(vals) == 1 else tuple(vals), p["dim"], p.get("zero"))
    if base == "materialize":
        return _materialize(vals[0], p)
    if base == "stage":
        from .compute import _tensor_like

        t = vals[0]
        order = tuple(p["order"]) if p.get("order") else t.names
        arr = t.to_numpy(order=order) if order else t.to_numpy()
        dims = tuple(t.layout.dim(n) for n in order)
        return _tensor_like(arr, dims, value_units=t.value_units)
    if base == "round_to":
        return _round_to(vals[0], p["encoding"])
    if base == "repeat_like":
        return _eager_repeat_like(vals[0], vals[1])
    if base == "take":
        return _eager_take(vals[0], vals[1], dim=p["dim"])
    if base == "scatter_add":
        return _eager_scatter_add(vals[0], vals[1], dim=p["dim"], extent=p["extent"], over=p.get("over"))
    if base == "argtopk":
        return _eager_argtopk(vals[0], dim=p["dim"], k=p["k"], k_name=p["k_name"])
    if base == "argsort":
        return _eager_argsort(vals[0], dim=p["dim"])
    if base == "with_value_units":
        return vals[0].with_value_units(p["value_units"])
    return LAYOUT_ADAPTERS[base](vals[0], p)


_BRIDGED = (
    "reduce",
    "scan",
    "materialize",
    "stage",
    "round_to",
    "repeat_like",
    "random",
    "with_value_units",
    "const",
    "take",
    "scatter_add",
    "argtopk",
    "argsort",
) + tuple(LAYOUT_ADAPTERS)

TL_OPS = {
    # per-launch scalar slots are abi.slot — the dsl's marshaling dialect,
    # ONE concept both tiers (240 C5; the uniform channel is its kernel face)
    "tl.token": OpDef("tl.token", lambda a, at, r: TokenType(), PURE),
    "tl.coord": OpDef("tl.coord", _r_coord, PURE),
    "tl.iota": OpDef("tl.iota", _r_iota, PURE),
    "tl.pointwise": OpDef("tl.pointwise", _r_pointwise, PURE),
    "tl.read": OpDef("tl.read", _r_read, PURE),
    "tl.sample": OpDef("tl.sample", _r_sample, PURE),
    "tl.store": OpDef("tl.store", _r_store, PURE),  # the effect rides the token
    "tl.fold": OpDef("tl.fold", _r_fold, PURE, nregions=1),
    **{f"tl.{base}": OpDef(f"tl.{base}", _r_bridge(base), PURE) for base in _BRIDGED},
}


# --- the rule pack: ONE typed-rule factory; dialects extend by rows ----------

_BIN_MARKER = {pyast.Add: "add", pyast.Sub: "sub", pyast.Mult: "mul", pyast.Div: "div"}
_CMP_MARKER = {pyast.Lt: "lt", pyast.Gt: "gt", pyast.LtE: "le", pyast.GtE: "ge", pyast.Eq: "eq", pyast.NotEq: "ne"}

_COORD_MATH = (
    "no arithmetic on a Coordinate — a point is not a number in the kernel either: "
    "coerce it first (f32(c) for value math, i32(c) for index math); coordinate "
    "arithmetic (250 §9) arrives with its first consumer"
)


def _coerce_scalar(name: str, v):
    """The explicit coercion doors (250 §3): a Coordinate yields its backing
    field (f64 interior per 210; the name records declared intent); a host
    int PROMOTES explicitly (ints never silently join float math — the rule
    `extent(c)` exists to exercise: f32(extent(c))). Everything else is
    already a value; the coercion refuses rather than launder it."""
    if hasattr(v, "type") and isinstance(v.type, CoordType):
        return v.args[0]
    if isinstance(v, bool):
        raise TypeError(f"{name}() takes a Coordinate or an int; bool is not a number here")
    if isinstance(v, int):
        return float(v) if name == "f32" else v
    raise TypeError(f"{name}() coerces a Coordinate (or promotes a host int); already a value — drop the coercion")


def _typed_rule(table, base_rule, pick):
    """The op-selection pattern, once: lower the children; a tensor operand
    selects the dialect row from ``table``; otherwise the base value pack
    serves the node unchanged."""

    def rule(ctx, node):
        operands = [ctx.lower(a) for a in _operands_of(node)]
        if any(hasattr(o, "type") and isinstance(o.type, CoordType) for o in operands):
            raise TypeError(_COORD_MATH)
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
        if any(not hasattr(o, "type") for o in operands):  # a scalar NODE (a uniform slot)
            # mixed with host scalars: the all-scalar subtree stays in the
            # VALUE dialect (the C5.3 law), hosts lifted to consts
            lifted = [
                o if hasattr(o, "type") else ctx.emit("core.const", node=node, type=f64, value=float(o))
                for o in operands
            ]
            if isinstance(node, pyast.BinOp):
                return ctx.emit(_CORE_BIN[type(node.op)], *lifted, node=node)
            return ctx.emit("core.cmp", *lifted, node=node, pred=f)
        return base_rule(ctx, node)

    return rule


_CORE_BIN = {pyast.Add: "core.add", pyast.Sub: "core.sub", pyast.Mult: "core.mul", pyast.Div: "core.div"}


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
                return next(d.size for d in t.type.dims if d.name == want)
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
            out = []
            for n in names:  # Coordinates (250): the iota backing, frame-typed
                backing = ctx.emit("tl.iota", lattice, node=node, name=n)
                frame = next(f for f in lattice.type.dims if f.name == n)
                out.append(ctx.emit("tl.coord", backing, node=node, frame=frame))
            return tuple(out)  # ALWAYS a tuple
        if isinstance(obj, _Intrinsic) and obj.name in ("f32", "i32"):
            return _coerce_scalar(obj.name, ctx.lower(node.args[0]))
        if obj is _eager_pw:  # the S.1 STEP-tier spelling
            marker = _lookup(ctx, node.args[0].id) or ctx.context["registry"].overloads.get(node.args[0].id)
            if marker is None or not hasattr(marker, "name"):
                raise ValueError(f"pointwise wants a marker first, got {node.args[0].id!r}")
            rest = [_scalar_lift(ctx, node, ctx.lower(a)) for a in node.args[1:]]
            return ctx.emit("tl.pointwise", *rest, node=node, f=marker.name)
        if obj is const_like:  # scalar broadcast: the const IS the operand
            try:
                v = _host(ctx, node.args[1])
            except Exception:
                v = None
            if isinstance(v, int) and not isinstance(v, bool):
                # the literal's own type is the carrier declaration (the
                # eager face's law): an int MATERIALIZES at integer carrier
                # over the ref's lattice — index arithmetic (§1.9
                # linearizations) never rides f64. Floats stay the DEFERRED
                # scalar (pointwise's broadcast law; fold steps require it).
                ref = ctx.lower(node.args[0])
                dims = tuple((d.name, (d.start, d.stop)) for d in ref.type.layout.dims)
                return ctx.emit("tl.const", node=node, value=v, dims=dims, dtype="int64")
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
            if hasattr(t, "type") and isinstance(t.type, CoordType):
                return t.type.frame.size  # the coordinate face: the frame's width, a host INT
            want = _host(ctx, node.args[1])
            return next(d.size for d in t.type.dims if d.name == want)
        if obj in (_eager_take, _eager_scatter_add, _eager_argtopk, _eager_argsort):  # the §1.9 family
            name = obj.__name__
            arity = 2 if obj in (_eager_take, _eager_scatter_add) else 1
            if len(node.args) != arity:
                raise TypeError(f"{name} takes {arity} tensor operand(s) — dim/extent/k/k_name are keywords")
            ops = [ctx.lower(a) for a in node.args]
            kw = {k.arg: _host(ctx, k.value) for k in node.keywords}
            if name == "scatter_add" and not hasattr(ops[0], "type"):
                # a scalar values operand (a deferred const_like float)
                # materializes over the IDX lattice: one contribution per
                # index element — the count-scatter
                idx_lay = ops[1].type.layout
                dims = tuple((d.name, (d.start, d.stop)) for d in idx_lay.dims)
                ops[0] = ctx.emit("tl.const", node=node, value=float(ops[0]), dims=dims)
            return ctx.emit(f"tl.{name}", *ops, node=node, **kw)
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
            if any(hasattr(a, "type") and isinstance(a.type, CoordType) for a in args):
                raise TypeError(_COORD_MATH)
            if any(hasattr(a, "type") and isinstance(a.type, TensorType) for a in args):
                if ctx.context.get("tl.kind") not in KERNEL_KINDS:  # the tensor tiers spell pointwise
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
                if getattr(obj, "__module__", "") == "pdum.tl.compute":
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
        if hasattr(v, "type") and isinstance(v.type, CoordType):
            raise TypeError(_COORD_MATH)
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
        return check_tier(Region(params=params, body=(ctx.builder.emit("core.yield", yielded),)), kind)
    if kind != "compute":
        raise ValueError(f"unknown body kind {kind!r} — kinds declare their yields here")
    for stmt in tree.body:
        if isinstance(stmt, pyast.Return):
            raise ValueError("kernels return nothing — stores into writable arguments are the effect")
        ctx.lower(stmt)
    tok = ctx.context.get("tl.token")
    if tok is None:
        raise ValueError(f"{fn.__qualname__} stores nothing — a kernel's effect is its stores")
    return check_tier(Region(params=params, body=(ctx.builder.emit("core.yield", tok),)), kind)


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


# --- 290: the stratification — families once, tiers as unions ----------------
# The one table (owner-ratified 2026-07-28). Families are declared HERE, at
# the ops' home; a tier is a union of families plus named extras. The open
# marker families (pw.*, math.*) admit by prefix; an op in no family admits
# nowhere (290 §4.1 — the open-registry default).

_FAMILIES = {
    "STRUCTURAL": frozenset(
        {"core.param", "core.env", "core.const", "core.yield", "core.tuple", "core.extract", "tl.const"}
    ),
    "SCALAR": frozenset(
        {"core.add", "core.sub", "core.mul", "core.div", "core.neg", "core.mod", "core.pow"}
        | {"core.cmp", "core.select", "core.cast"}
    ),
    "ABI": frozenset({"abi.slot"}),
    "LATTICE": frozenset({"tl.iota", "tl.coord"}),
    "POINTWISE": frozenset({"tl.pointwise"}),
    "COMPUTE": frozenset(
        {"tl.reduce", "tl.scan", "tl.fold", "tl.take", "tl.scatter_add", "tl.argtopk", "tl.argsort"}
        | {"tl.random", "tl.materialize", "tl.stage", "tl.round_to", "tl.repeat_like", "tl.with_value_units"}
    ),
    "LAYOUT": frozenset(f"tl.{n}" for n in LAYOUT_ADAPTERS),
    "EFFECT": frozenset({"tl.store", "tl.read", "tl.token"}),
    "GRAPHICS": frozenset({"tl.sample"}),
}
_FAMILY_OF = {op: fam for fam, ops in _FAMILIES.items() for op in ops}

KERNEL_KINDS = frozenset({"compute", "vertex", "fragment"})  # bare markers spell here

TIERS = {
    "tensor": frozenset({"STRUCTURAL", "SCALAR", "POINTWISE", "COMPUTE", "LAYOUT"}),
    "unit": frozenset({"STRUCTURAL", "SCALAR", "POINTWISE", "COMPUTE", "LAYOUT"}),
    "step": frozenset({"STRUCTURAL", "SCALAR", "POINTWISE", "COMPUTE", "LAYOUT"}),
    "compute": frozenset({"STRUCTURAL", "SCALAR", "ABI", "LATTICE", "POINTWISE", "EFFECT"}),
    "vertex": frozenset({"STRUCTURAL", "SCALAR", "ABI", "POINTWISE"}),
    "fragment": frozenset({"STRUCTURAL", "SCALAR", "ABI", "POINTWISE", "GRAPHICS"}),
    # 320: the tile tier — tensor ops on pre-selected tiles. COMPUTE is NOT
    # admitted as a family: the extras below name its tile-legal subset, and
    # the K-G ops (take/scatter/argtopk/argsort/random) refuse at the tier.
    "tile": frozenset({"STRUCTURAL", "SCALAR", "POINTWISE", "LAYOUT", "EFFECT"}),
}
_TIER_EXTRA = {
    "tensor": frozenset({"tl.iota"}),
    "unit": frozenset({"tl.iota"}),
    "step": frozenset({"tl.iota"}),
    "compute": frozenset({"tl.split", "tl.merge"}),  # machinery emissions (grid bracket,
    # global-index stores) — never author-spelled, ledgered for device coverage (290 §4.1)
    "vertex": frozenset({"tl.iota", "tl.select"}),  # select = the pulled read, params only
    "tile": frozenset({"tl.iota", "tl.reduce", "tl.scan", "tl.fold", "tl.repeat_like", "tl.stage", "tl.materialize"}),
}
_QUOTED_TIER = {"tl.fold": "step"}  # ops carrying a foreign-tier region as DATA (290 §4.5)

_TIER_FIX = {  # the rulebook's one refusal class: a family-appropriate quoted fix
    "LAYOUT": "apply the view outside the body (stage it on the parameter) and pass the result in",
    "COMPUTE": "compute it at the call site and pass the result in",
    "EFFECT": "stores/reads are kernel effects — write a @compute kernel and launch it",
    "GRAPHICS": "textures sample in fragment stages today",
    "ABI": "captured scalars are build-time values at this tier, not launch uniforms",
    "LATTICE": "thread position is a kernel/graphics fact — spell positions with iota here",
}


def tier_admits(op: str, tier: str) -> bool:
    fam = _FAMILY_OF.get(op)
    if fam is None and op.startswith(("pw.", "math.")):
        fam = "SCALAR"  # the open marker families
    return (fam is not None and fam in TIERS[tier]) or op in _TIER_EXTRA.get(tier, ())


def check_tier(region: Region, tier: str) -> Region:
    """The stratification gate (290 §4.3): every op a region carries must be
    in its tier's vocabulary; sub-regions check under the tier the CARRYING
    op declares (§4.5), never the ambient one. The first violation refuses
    in the rulebook voice, quoting the node's own source."""
    from pdum.dsl.ir import format_loc

    for n in walk_region(region):
        if not tier_admits(n.op, tier):
            fam = _FAMILY_OF.get(n.op)
            fix = _TIER_FIX.get(fam, f"it has no {tier}-tier meaning (290)")
            where = f" [{format_loc(n.loc)}]" if n.loc is not None else ""
            raise ValueError(
                f"{n.op} is a host citizen here — the {tier} tier uses it at a call site, "
                f"not as a value: {fix}{where}"
            )
        if tier == "vertex" and n.op == "tl.select" and n.args and n.args[0].op != "core.param":
            where = f" [{format_loc(n.loc)}]" if n.loc is not None else ""
            raise ValueError(
                f"tl.select in a vertex stage reads a BUFFER PARAMETER at a coordinate (the "
                f"pulled-read spelling) — layout algebra on derived values stages outside{where}"
            )
        for sub in n.regions:
            quoted = _QUOTED_TIER.get(n.op, tier)
            if tier == "tile" and n.op == "tl.fold":
                # 320 §3: the tile-fold's step widens INSIDE the tile
                # discipline — the quoted "step" tier would readmit the K-G
                # ops the tile tier refuses
                quoted = "tile"
            check_tier(sub, quoted)
    return region


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
                red = {d.name for d in operand.type.dims} - {d.name for d in node.type.dims}
                n_red = 1
                for d in operand.type.dims:
                    if d.name in red:
                        n_red *= d.size
                rep = b.emit("tl.pointwise", rep, b.emit("core.const", type=f64, value=float(n_red)), f="div")
            acc(operand, rep)
        elif node.op == "tl.repeat_like":
            x = node.args[0]
            added = tuple(d.name for d in node.type.dims if d.name not in {q.name for q in x.type.dims})
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


def run_region(region: Region, values: list, uniforms: bytes | None = None, textures: list | None = None):
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
        if n.op == "tl.coord":  # a typed handle: its value IS its backing
            return ev(n.args[0])
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
        if n.op == "tl.sample":
            cy, cx = (ev(a) for a in n.args)
            mirror = (textures or [])[attrs["idx"]]
            names_r = tuple(d.name for d in cy.layout.dims)
            a_y = cy.to_numpy(order=names_r)
            a_x = cx.to_numpy(order=names_r)
            Hm, Wm = mirror.shape
            wrap = attrs["address"] == "repeat"

            def resolve(i, n_):
                return np.mod(i, n_) if wrap else np.clip(i, 0, n_ - 1)

            if attrs["filter"] == "nearest":
                iy = resolve(np.floor(a_y * Hm).astype(np.int64), Hm)
                ix = resolve(np.floor(a_x * Wm).astype(np.int64), Wm)
                out = mirror[iy, ix]
            else:  # bilinear, texel centers at (i + 0.5)/N (the WebGPU convention)
                ty, tx = a_y * Hm - 0.5, a_x * Wm - 0.5
                y0, x0 = np.floor(ty).astype(np.int64), np.floor(tx).astype(np.int64)
                fy, fx = ty - y0, tx - x0
                m = mirror
                v00 = m[resolve(y0, Hm), resolve(x0, Wm)]
                v01 = m[resolve(y0, Hm), resolve(x0 + 1, Wm)]
                v10 = m[resolve(y0 + 1, Hm), resolve(x0, Wm)]
                v11 = m[resolve(y0 + 1, Hm), resolve(x0 + 1, Wm)]
                out = (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy
            return _tensor_like(np.asarray(out, dtype=np.float64), cy.layout.dims)
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
            return _eval_base(n.op[3:], [ev(a) for a in n.args], _thaw_params(attrs))
        if n.op == "tl.fold":
            state, element = tuple(attrs["state"]), tuple(attrs["element"])
            k, m = len(state), len(element)
            dim, out = attrs["dim"], tuple(attrs["out"])
            vals = [ev(a) for a in n.args]
            carried, srcs = list(vals[:k]), vals[k:]
            if m:
                lo, hi = next((d.start, d.stop) for d in n.args[k].type.dims if d.name == dim)
            else:
                lo, hi = attrs["extent"]
            emitted = []
            for q in range(lo, hi):
                res = run_region(n.regions[0], carried + [s.select(**{dim: q}) for s in srcs])
                res = res if isinstance(res, tuple) else (res,)
                carried = list(res[:k])
                if out[0] == "emit":
                    emitted.append(res[k])
            if out[0] == "final":
                return carried[out[1]]
            shadow = _dense_like((Dim(dim, 0, lo, hi),) + tuple(emitted[0].layout.dims))
            onames = emitted[0].names
            arr = np.stack([e.to_numpy(order=onames) if onames else e.to_numpy() for e in emitted], axis=0)
            return _tensor_like(arr, shadow.dims)
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


def fold_region(
    step: Region,
    *,
    dim: str,
    state: tuple,
    element: tuple,
    out: tuple,
    init_types: tuple,
    src_types: tuple = (),
    extent: tuple | None = None,
):
    """Author the minimal fold-bearing region: one param per state init and
    element source (in that order), one ``tl.fold`` over ``step``, yield.
    Returns ``(region, fold_node)`` — the node so callers can name it
    (``names_of={id(fold_node): ...}``)."""
    b = Builder({**CORE_OPS, **TL_OPS})
    params = tuple(b.param(i, t) for i, t in enumerate(tuple(init_types) + tuple(src_types)))
    kwargs = dict(dim=dim, state=tuple(state), element=tuple(element), out=tuple(out))
    if extent is not None:
        kwargs["extent"] = tuple(extent)
    fold = b.emit("tl.fold", *params, regions=(step,), **kwargs)
    return check_tier(Region(params=params, body=(b.emit("core.yield", fold),)), "tensor"), fold


# --- the naming law over regions ---------------------------------------------


# layout-op membership as NAMES — what the analyses consult ("layout ops move
# coordinates, never values")
LAYOUT_OP_NAMES = frozenset(LAYOUT_ADAPTERS)


def region_names(region: Region, param_names: tuple, names_of: dict | None = None) -> dict[int, str]:
    """The naming law over a region: params CLAIM their declared names; every
    other node DERIVES from its binding name (``names_of``, first-binding-wins)
    or its op's base name. This is the ONE assignment — ``export_program``
    renders it, and the region-native analyses key their reports by it, so
    name-keyed reports agree across the migration view by construction.
    Returns ``id(node) -> name`` (consts, yields, and the yielded tuple are
    plumbing: unnamed)."""
    from pdum.dsl.naming import Namer

    names = Namer()
    names_of = names_of or {}
    out: dict[int, str] = {}
    for pname, p in zip(param_names, region.params):
        out[id(p)] = names.claim(pname)
    yielded = region.body[-1].args[0] if region.body else None
    for n in walk_region(region):
        if id(n) in out or n.op in ("core.yield", "core.const") or (n.op == "core.tuple" and n is yielded):
            continue
        out[id(n)] = names.derive(names_of.get(id(n), n.op.rsplit(".", 1)[-1]))
    return out


def run_named(region: Region, inputs: dict, names) -> dict:
    """Run a region with name-keyed inputs and get name-keyed outputs — the
    reference-execution door the naming law implies: params bind by their
    claimed names, and the yield's slots report under theirs."""
    order = [names[id(p)] for p in region.params]
    missing = [k for k in order if k not in inputs]
    if missing:
        raise KeyError(
            f"missing input {missing[0]!r} — virtual leaves analyze for free but "
            f"execute only once provisioned: provision(root, source=init(...)"
            f"|safetensors(...)) (200 §1.7)"
        )
    res = run_region(region, [inputs[k] for k in order])
    yielded = region.body[-1].args[0]
    outs = tuple(yielded.args) if yielded.op == "core.tuple" else (yielded,)
    return dict(zip(tuple(names[id(x)] for x in outs), res if isinstance(res, tuple) else (res,)))
