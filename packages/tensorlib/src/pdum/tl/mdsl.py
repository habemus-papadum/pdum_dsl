"""The marker DSL: composite markers as an owned expression-tree IR.

The Node schema (Arg / Const / Prim) lives in nodes.py — the declared
stability boundary. This module is one PRODUCER of Nodes (the ~40-line
operator-overloading tracer; the shared-syntax AST producer replaces it at
P4) plus the symbolic differentiation machinery and the composite
marker/reducer declarations, registered into the cache-backed registries
(registry.py — idempotent, derivation-under-cache, conflict-refusing).

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

from . import producer
from .derivative import TABLE, diff  # noqa: F401 — diff re-exported for consumers
from .nodes import Arg, Const, Node, Prim  # noqa: F401 — re-exported for consumers
from .registry import MARKERS, REDUCERS

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

    apply.op, apply.arity = op, arity  # the AST producer resolves captured calls by these
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
# composite markers and reducers (registered: programs stay data)
# ----------------------------------------------------------------------

_PRIMITIVE_NAMES = frozenset(TABLE)


def node_digest(node: Node) -> str:
    """Content address of a tree (dataclass reprs are deterministic).
    Digest-derived names make registration automatic and idempotent: the
    same lambda traced in a loop lands on the same registry entry, so the
    registry behaves as a cache rather than a namespace — the main repo's
    build-in-a-loop-against-a-cache philosophy, applied here."""
    return hashlib.sha1(repr(node).encode()).hexdigest()[:10]


def _marker_content(m: "CompositeMarker") -> tuple:
    return (m.arity, m.body)


def _register_marker(name: str, arity: int, body: Node) -> "CompositeMarker":
    return MARKERS.register(name, CompositeMarker(name, arity, body), _marker_content)


@dataclass(frozen=True, eq=False)
class CompositeMarker:
    name: str
    arity: int
    body: Node

    def partial(self, i: int) -> "CompositeMarker":
        """The i-th partial derivative — a cache entry computed on demand
        from a cache entry: the tree rewrite runs once per name, ever."""
        if not 0 <= i < self.arity:
            raise IndexError(f"{self.name} has arity {self.arity}")
        name = f"{self.name}.d{i}"
        return MARKERS.derive(name, lambda: CompositeMarker(name, self.arity, diff(self.body, i)))

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
        return REDUCERS.derive(
            name,
            lambda: CompositeReducer(
                name, self.state, self.element, self.lift, self.combine, self.init, Arg(j), self.associative
            ),
        )

    def adjoint_scanner(self, i: int) -> "CompositeReducer":
        """The reversed-time backward recurrence r_u = M_u·r_{u-1} + g_u as a
        matrix linear-recurrence reducer over state (M: k×k, v: k),
        projecting v_i. The M part of the state is never projected, so the
        first element's M slot (which has no recurrence edge to carry) may
        hold boundary garbage harmlessly."""
        name = f"{self.name}.adj{i}"
        k = self.state
        s = k * k + k
        return REDUCERS.derive(
            name,
            lambda: CompositeReducer(
                name=name,
                state=s,
                element=s,
                lift=tuple(Arg(j) for j in range(s)),
                combine=_matrix_pair_nodes(k),
                init=tuple(1.0 if (j < k * k and j // k == j % k) else 0.0 for j in range(s)),
                project=Arg(k * k + i),
            ),
        )


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
    """Lower a plain function (a def or a lambda) into a composite marker
    via the AST producer — one named, inspectable body tree over primitives
    (the marker-granularity gate, S.2). Nothing is executed.

    sigmoid = defmarker("sigmoid", 1, lambda x: 1 / (1 + exp(-x)))

    `name=None` derives a content-addressed name (`m_<digest>`) from the
    lowered tree — naming becomes optional, and re-lowering the same body
    in a loop dedupes onto the same registry entry (the registry as cache,
    not namespace).
    """
    if name in _PRIMITIVE_NAMES:
        raise ValueError(f"{name!r} is a primitive marker name")
    (body,) = producer.lower(fn, producer.scalars(arity))
    if name is None:
        name = f"m_{node_digest(body)}"
    return _register_marker(name, arity, body)


def defreducer(
    name: str, *, state, element: int, lift, combine, init, project=None, associative: bool = True
) -> CompositeReducer:
    """Define a structured-state reducer; bodies lower via the AST producer.

    `state` is a count OR a record class (a frozen dataclass / NamedTuple):
    with `state=k`, `lift(e1..em) -> k-tuple`, `combine(left, right)` takes
    k-tuples (subscripted by literal index) and `project(s1..sk) -> scalar`
    (default: the first component). With `state=State`, lift returns
    `State(...)`, combine takes two State-typed arguments (fields by
    attribute) and returns one, project takes ONE State argument, and
    `init` may be a `State(...)` instance — record-typed reducer state
    (S.2); the state layout is the record's field order.

        linrec = defreducer("linrec", state=2, element=2,
            lift=lambda a, b: (a, b),
            combine=lambda l, r: (l[0] * r[0], r[0] * l[1] + r[1]),
            init=(1.0, 0.0),
            project=lambda A, B: B)
    """
    record = state if producer.is_record(state) else None
    k = len(producer.record_fields(record)) if record else state
    lift_nodes = producer.lower(lift, producer.scalars(element))
    if len(lift_nodes) != k:
        raise ValueError(f"lift must produce {k} components, got {len(lift_nodes)}")
    if record:
        combine_nodes = producer.lower(
            combine, producer.record_binding(record, k) + producer.record_binding(record, k, k)
        )
        project_bindings = producer.record_binding(record, k)
    else:
        combine_nodes = producer.lower(combine, producer.tuple_binding(k) + producer.tuple_binding(k, k))
        project_bindings = producer.scalars(k)
    if len(combine_nodes) != k:
        raise ValueError(f"combine must produce {k} components")
    if producer.is_record(type(init)):
        init = tuple(getattr(init, f) for f in producer.record_fields(type(init)))
    if len(init) != k:
        raise ValueError(f"init must have {k} components")
    project_node = producer.lower(project, project_bindings)[0] if project is not None else Arg(0)
    r = CompositeReducer(name, k, element, lift_nodes, combine_nodes, tuple(init), project_node, associative)
    return REDUCERS.register(
        name, r, lambda v: (v.state, v.element, v.lift, v.combine, v.init, v.project, v.associative)
    )
