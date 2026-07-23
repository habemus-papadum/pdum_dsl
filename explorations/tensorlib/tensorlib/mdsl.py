"""The marker DSL: composite markers as an owned expression-tree IR.

The stability boundary of this module is the Node schema — three tiny frozen
dataclasses (Arg / Const / Prim) — and it deliberately imports NOTHING from
the main pdum.dsl package (whose syntax tooling is still in flux) and nothing
from the rest of tensorlib. Everything else is layered around that schema:

- consumers (in compute.py / autodiff.py): numpy evaluation, symbolic
  partial derivatives, later carrier/unit signature propagation — all walk
  Nodes and never care where they came from;
- producers: the ~40-line TRACER here (operator-overloaded `Sym`s executing
  a plain lambda), and, once the main repo's frontend stabilizes, an adapter
  mapping its lowered AST onto the same Nodes. Swapping producers can never
  force a rewrite of consumers — that is the no-rewrite guarantee, held by
  the schema rather than by promise.

Composite POINTWISE markers (`defmarker`) are scalar expression trees over
the primitive marker names ("add", "mul", "exp", ...). Their partial
derivatives are DERIVED by tree rewriting (`CompositeMarker.partial(i)`
returns another registered composite), so new activation functions
differentiate automatically — the hand-maintained gradient table stops
growing, and the silent-zero-gradient class of bug dies structurally.

Composite REDUCERS (`defreducer`) carry structured (tuple) state: an
element-to-state `lift`, an associative `combine` over (left ++ right)
state, an `init` state, and a `project` back to a scalar. This is the shape
SSM / linear-recurrence scans need (h_t = a_t·h_{t-1} + b_t via the pair
combine (A1,B1)⊕(A2,B2) = (A1·A2, A2·B1 + B2)). Associativity is DECLARED,
not verified — property-test it now; it becomes a typeclass instance
obligation in the Lean model.

No control flow in marker bodies: `where` is the branch (matching the
program IR's no-branching rule), and tracing a Python `if` on a Sym raises.
Constants are VALUE-space, so floats are honest here (0.5, sqrt(2)/2);
Fraction stays available for exact rationals — the coordinates-exact /
values-inexact doctrine, applied.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from fractions import Fraction

# ----------------------------------------------------------------------
# the IR — the stability boundary
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Arg:
    """A formal parameter of the marker (by position)."""

    index: int


@dataclass(frozen=True)
class Const:
    """A literal: int/Fraction (exact) or float (value-space)."""

    value: object


@dataclass(frozen=True)
class Prim:
    """Application of a PRIMITIVE marker, referenced by name."""

    op: str
    args: tuple


Node = Arg | Const | Prim


def _is_const(n, v) -> bool:
    return isinstance(n, Const) and n.value == v


# ----------------------------------------------------------------------
# the tracer — one producer of Nodes (frontends are pluggable)
# ----------------------------------------------------------------------


class Sym:
    """A symbolic scalar; executing a plain lambda over Syms yields the tree."""

    __slots__ = ("node",)

    def __init__(self, node):
        self.node = node

    def __bool__(self):
        raise TypeError(
            "Python control flow cannot be traced into a marker body; "
            "use where(cond, a, b) — the branch is data flow here"
        )

    def __add__(self, o):
        return Sym(Prim("add", (self.node, _lift(o))))

    __radd__ = __add__

    def __mul__(self, o):
        return Sym(Prim("mul", (self.node, _lift(o))))

    __rmul__ = __mul__

    def __sub__(self, o):
        return Sym(Prim("sub", (self.node, _lift(o))))

    def __rsub__(self, o):
        return Sym(Prim("sub", (_lift(o), self.node)))

    def __truediv__(self, o):
        return Sym(Prim("div", (self.node, _lift(o))))

    def __rtruediv__(self, o):
        return Sym(Prim("div", (_lift(o), self.node)))

    def __neg__(self):
        return Sym(Prim("neg", (self.node,)))


def _lift(x) -> Node:
    if isinstance(x, Sym):
        return x.node
    if isinstance(x, bool) or not isinstance(x, (int, float, Fraction)):
        raise TypeError(f"cannot lift {x!r} into a marker body")
    return Const(x)


def _fn(op, arity):
    def apply(*args):
        if len(args) != arity:
            raise TypeError(f"{op} takes {arity} arguments")
        return Sym(Prim(op, tuple(_lift(a) for a in args)))

    return apply


exp = _fn("exp", 1)
log = _fn("log", 1)
tanh = _fn("tanh", 1)
sqrt = _fn("sqrt", 1)
sin = _fn("sin", 1)
cos = _fn("cos", 1)
maximum = _fn("maximum", 2)
minimum = _fn("minimum", 2)
where = _fn("where", 3)
eq = _fn("eq", 2)
ne = _fn("ne", 2)
le = _fn("le", 2)
lt = _fn("lt", 2)
ge = _fn("ge", 2)
gt = _fn("gt", 2)


def trace(fn, arity: int) -> Node:
    return _lift(fn(*(Sym(Arg(i)) for i in range(arity))))


def _trace_tuple(fn, arity: int) -> tuple:
    out = fn(*(Sym(Arg(i)) for i in range(arity)))
    if not isinstance(out, tuple):
        out = (out,)
    return tuple(_lift(o) for o in out)


# ----------------------------------------------------------------------
# symbolic differentiation — partial derivatives by tree rewriting
# ----------------------------------------------------------------------

# per-primitive local-slope builders: arg nodes -> Node; None = gradient-free
# position (the carrier discipline at the tree level)
_D = {
    "add": (lambda a, b: Const(1), lambda a, b: Const(1)),
    "sub": (lambda a, b: Const(1), lambda a, b: Const(-1)),
    "neg": (lambda a: Const(-1),),
    "mul": (lambda a, b: b, lambda a, b: a),
    "div": (
        lambda a, b: Prim("div", (Const(1), b)),
        lambda a, b: Prim("neg", (Prim("div", (a, Prim("mul", (b, b)))),)),
    ),
    "exp": (lambda a: Prim("exp", (a,)),),
    "log": (lambda a: Prim("div", (Const(1), a)),),
    "sqrt": (lambda a: Prim("div", (Const(1), Prim("mul", (Const(2), Prim("sqrt", (a,)))))),),
    "sin": (lambda a: Prim("cos", (a,)),),
    "cos": (lambda a: Prim("neg", (Prim("sin", (a,)),)),),
    "tanh": (lambda a: Prim("sub", (Const(1), Prim("mul", (Prim("tanh", (a,)), Prim("tanh", (a,)))))),),
    "maximum": (lambda a, b: Prim("ge", (a, b)), lambda a, b: Prim("gt", (b, a))),
    "minimum": (lambda a, b: Prim("le", (a, b)), lambda a, b: Prim("lt", (b, a))),
    "where": (
        None,  # the condition is gradient-free
        lambda c, x, y: Prim("where", (c, Const(1), Const(0))),
        lambda c, x, y: Prim("where", (c, Const(0), Const(1))),
    ),
    "eq": (None, None),
    "ne": (None, None),
    "le": (None, None),
    "lt": (None, None),
    "ge": (None, None),
    "gt": (None, None),
}


def diff(node: Node, i: int) -> Node:
    """d(node)/d(Arg(i)), with light zero/one folding to keep trees small."""
    if isinstance(node, Arg):
        return Const(1) if node.index == i else Const(0)
    if isinstance(node, Const):
        return Const(0)
    if node.op not in _D:
        raise NotImplementedError(f"primitive {node.op!r} has no derivative builder in mdsl._D")
    rules = _D[node.op]
    total: Node = Const(0)
    for j, arg in enumerate(node.args):
        rule = rules[j]
        if rule is None:
            continue
        inner = diff(arg, i)
        if _is_const(inner, 0):
            continue
        local = rule(*node.args)
        if _is_const(local, 0):
            continue
        if _is_const(inner, 1):
            term = local
        elif _is_const(local, 1):
            term = inner
        else:
            term = Prim("mul", (local, inner))
        total = term if _is_const(total, 0) else Prim("add", (total, term))
    return total


# ----------------------------------------------------------------------
# composite markers and reducers (+ registries: programs stay data)
# ----------------------------------------------------------------------

COMPOSITE_MARKERS: dict[str, "CompositeMarker"] = {}
COMPOSITE_REDUCERS: dict[str, "CompositeReducer"] = {}

_PRIMITIVE_NAMES = frozenset(_D)


def node_digest(node: Node) -> str:
    """Content address of a tree (dataclass reprs are deterministic).
    Digest-derived names make registration automatic and idempotent: the
    same lambda traced in a loop lands on the same registry entry, so the
    registry behaves as a cache rather than a namespace — the main repo's
    build-in-a-loop-against-a-cache philosophy, applied here."""
    return hashlib.sha1(repr(node).encode()).hexdigest()[:10]


def _register_marker(name: str, arity: int, body: Node) -> "CompositeMarker":
    existing = COMPOSITE_MARKERS.get(name)
    if existing is not None and existing.body == body and existing.arity == arity:
        return existing  # content-equal re-registration is a no-op
    m = CompositeMarker(name, arity, body)
    COMPOSITE_MARKERS[name] = m
    return m


@dataclass(frozen=True, eq=False)
class CompositeMarker:
    name: str
    arity: int
    body: Node

    def partial(self, i: int) -> "CompositeMarker":
        """The i-th partial derivative, as another registered composite —
        derived once by tree rewriting, then reused by name."""
        if not 0 <= i < self.arity:
            raise IndexError(f"{self.name} has arity {self.arity}")
        return _register_marker(f"{self.name}.d{i}", self.arity, diff(self.body, i))

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, eq=False)
class CompositeReducer:
    """Structured-state reducer: lift each element (m scalars) to a state
    (k scalars), combine states associatively, project the answer."""

    name: str
    state: int  # k
    element: int  # m
    lift: tuple  # k Nodes over m Args
    combine: tuple  # k Nodes over 2k Args (left state ++ right state)
    init: tuple  # k numbers: the identity state (empty reduction)
    project: Node  # 1 Node over k Args
    associative: bool = True  # declared, not verified — property-test it

    def __repr__(self) -> str:
        return self.name

    # ---- derived machinery for the reverse pass (registered on demand) ----

    def component_markers(self):
        """(C_i over 2k args, L_j over m args, P over k args) as composite
        markers, so their partials come from the same derived-`diff`
        machinery as every other composite."""
        k, m = self.state, self.element
        cs = tuple(_register_marker(f"{self.name}.C{i}", 2 * k, self.combine[i]) for i in range(k))
        ls = tuple(_register_marker(f"{self.name}.L{j}", m, self.lift[j]) for j in range(k))
        p = _register_marker(f"{self.name}.P", k, self.project)
        return cs, ls, p

    def state_scanner(self, j: int) -> "CompositeReducer":
        """The same fold projecting state component j — how the reverse pass
        materializes the forward state trajectory as ordinary tensors."""
        name = f"{self.name}.s{j}"
        if name in COMPOSITE_REDUCERS:
            return COMPOSITE_REDUCERS[name]
        r = CompositeReducer(
            name, self.state, self.element, self.lift, self.combine, self.init, Arg(j), self.associative
        )
        COMPOSITE_REDUCERS[name] = r
        return r

    def adjoint_scanner(self, i: int) -> "CompositeReducer":
        """The reversed-time backward recurrence r_u = M_u·r_{u-1} + g_u as a
        matrix linear-recurrence reducer over state (M: k×k, v: k),
        projecting v_i. The M part of the state is never projected, so the
        first element's M slot (which has no recurrence edge to carry) may
        hold boundary garbage harmlessly."""
        name = f"{self.name}.adj{i}"
        if name in COMPOSITE_REDUCERS:
            return COMPOSITE_REDUCERS[name]
        k = self.state
        s = k * k + k
        r = CompositeReducer(
            name=name,
            state=s,
            element=s,
            lift=tuple(Arg(j) for j in range(s)),
            combine=_matrix_pair_nodes(k),
            init=tuple(1.0 if (j < k * k and j // k == j % k) else 0.0 for j in range(s)),
            project=Arg(k * k + i),
        )
        COMPOSITE_REDUCERS[name] = r
        return r


def _sum_nodes(terms):
    if not terms:
        return Const(0)
    total = terms[0]
    for t in terms[1:]:
        total = Prim("add", (total, t))
    return total


def _matrix_pair_nodes(k: int) -> tuple:
    """Combine for the matrix linear recurrence: (Ml,vl) ⊕ (Mr,vr) =
    (Mr·Ml, Mr·vl + vr). State layout: M row-major (k²), then v (k).
    Associative by matrix algebra — the generic backward-scan carrier."""
    s = k * k + k

    def ml(r, c):
        return Arg(r * k + c)

    def vl(r):
        return Arg(k * k + r)

    def mr(r, c):
        return Arg(s + r * k + c)

    def vr(r):
        return Arg(s + k * k + r)

    out = []
    for r in range(k):
        for c in range(k):
            out.append(_sum_nodes([Prim("mul", (mr(r, x), ml(x, c))) for x in range(k)]))
    for r in range(k):
        out.append(Prim("add", (_sum_nodes([Prim("mul", (mr(r, x), vl(x))) for x in range(k)]), vr(r))))
    return tuple(out)


def defmarker(name: str | None, arity: int, fn) -> CompositeMarker:
    """Trace a plain lambda over symbolic scalars into a composite marker.

    sigmoid = defmarker("sigmoid", 1, lambda x: 1 / (1 + exp(-x)))

    `name=None` derives a content-addressed name (`m_<digest>`) from the
    traced tree — naming becomes optional, and re-tracing the same body in
    a loop dedupes onto the same registry entry (the registry as cache,
    not namespace).
    """
    if name in _PRIMITIVE_NAMES:
        raise ValueError(f"{name!r} is a primitive marker name")
    body = trace(fn, arity)
    if name is None:
        name = f"m_{node_digest(body)}"
    return _register_marker(name, arity, body)


def defreducer(
    name: str, *, state: int, element: int, lift, combine, init, project=None, associative: bool = True
) -> CompositeReducer:
    """Define a structured-state reducer. `lift(e1..em) -> k-tuple`,
    `combine(left, right) -> k-tuple` where left/right are k-tuples of
    symbolic scalars, `init` is the k identity values, `project(s1..sk) ->
    scalar` (default: the first state component).

        linrec = defreducer("linrec", state=2, element=2,
            lift=lambda a, b: (a, b),
            combine=lambda l, r: (l[0] * r[0], r[0] * l[1] + r[1]),
            init=(1.0, 0.0),
            project=lambda A, B: B)
    """
    lift_nodes = _trace_tuple(lift, element)
    if len(lift_nodes) != state:
        raise ValueError(f"lift must produce {state} components, got {len(lift_nodes)}")
    combine_nodes = _trace_tuple(lambda *args: combine(tuple(args[:state]), tuple(args[state:])), 2 * state)
    if len(combine_nodes) != state:
        raise ValueError(f"combine must produce {state} components")
    if len(init) != state:
        raise ValueError(f"init must have {state} components")
    project_node = trace(project, state) if project is not None else Arg(0)
    r = CompositeReducer(name, state, element, lift_nodes, combine_nodes, tuple(init), project_node, associative)
    COMPOSITE_REDUCERS[name] = r
    return r
