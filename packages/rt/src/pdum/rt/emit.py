"""ONE expression generator — the rows, once, for every C-family target.

The measured claim this module makes structural (210, the graphics
campaign): 91/91 emitted statements convert between WGSL and MSL under
FOUR LEXICAL RULES — cast spelling, float-literal suffix, declaration
form, and vector type spelling. No operator or builtin row differs,
``select`` argument order included. So the marker tables and the walker
live here ONCE (they lived in three byte-identical copies before this
package), and a target is a ``Dialect``: the four lexical fields plus
the three leaf spellings a shell names its resources with.

The walk is over DATAFLOW, and two facts about that are load-bearing:

- **Walk args, not regions.** ``Node.regions`` carries a foreign-tier
  body as DATA, checked under the tier its carrying op declares (290
  §4.5) — it is not this tier's dataflow. Compute regions arrive flat
  (the kernel tier unrolls ``for``), and an op that does carry a region
  has no row here and refuses by name.
- **Args are not dataflow by themselves — the op table says what each
  one MEANS.** ``tl.store(token, dst, value)`` holds an ordering token
  in arg 0 and a DESTINATION in arg 1; neither is an expression to
  evaluate. The walk supplies ORDER (post-order, so a store's
  predecessors emit first); ``Gen.expr``'s table supplies meaning, and
  dispatches on the op before it touches an arg.

Regions are walked STRUCTURALLY — ops are strings, attrs are data — so
nothing here imports ``pdum.tl``. Artifacts arrive duck-typed.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from . import mathrows
from .contract import Binding


class Untranslatable(Exception):
    """This region has no row for some op — the reason names it."""


# Rule 2's lexer: what counts as a float literal in emitted text.
_FLOAT_LIT = re.compile(r"\d*\.\d+(?:[eE][+-]?\d+)?|\d+\.\d*(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+")


# --- the leaf spellings: hooks because the SHELL names the resources ---------
# Both compute shells agree today (buffers are ``buf{i}``, the slot buffer is
# ``U``, the thread coordinate is one builtin). They are hooks because CUDA's
# ambient row is COMPOSED — ``blockIdx.x * blockDim.x + threadIdx.x`` — rather
# than a single builtin (283 §5), and that is a leaf-row difference, not a
# fifth lexical rule.


def _buffer_row(d: Dialect, index: int, elem: str) -> str:
    return f"buf{index}[{elem}]"


def _iota_row(d: Dialect, comp: str) -> str:
    return f"{d.fty}(gid.{comp})"  # THE one-line ambient row (the alignment law's gift)


def _slot_row(d: Dialect, index: int) -> str:
    return f"U[{index}]"


@dataclass(frozen=True)
class Dialect:
    """One target's lexical rules, and nothing else about the target.

    ``fty``/``ity`` carry rule 1 (cast spelling — the type NAME is the
    cast in both languages), ``suffix`` rule 2, ``c_decl`` rule 3,
    ``vec_fmt`` rule 4. Compute is scalar-only today so rule 4 fires
    zero times here; it fires exactly once per render program, on the
    clip-space position, and it is the row the vectorized-load idiom
    will meet in the CUDA era (210).
    """

    name: str
    fty: str  # the float type NAME — declarations and casts both
    ity: str  # the int cast spelling
    bty: str  # the bool type name
    suffix: str  # the float literal suffix
    c_decl: bool  # True: `float v = e;`   False: `let v: f32 = e;`
    vec_fmt: str  # rule 4: "vec{n}<{t}>" | "{t}{n}"
    buffer_row: Callable[..., str] = field(default=_buffer_row)
    iota_row: Callable[..., str] = field(default=_iota_row)
    slot_row: Callable[..., str] = field(default=_slot_row)

    def lit(self, v) -> str:
        s = repr(float(v))
        if "inf" in s or "nan" in s:
            raise Untranslatable(f"a non-finite literal {s!r} (210: refuses at rendering)")
        return (s if ("." in s or "e" in s) else s + ".0") + self.suffix

    def decl(self, ty: str, var: str, expr: str) -> str:
        return f"  {ty} {var} = {expr};" if self.c_decl else f"  let {var}: {ty} = {expr};"

    def vec(self, n: int, ty: str | None = None) -> str:
        return self.vec_fmt.format(n=n, t=ty or self.fty)

    def rowtext(self, text: str) -> str:
        """A math row's spelling (mathrows.py) is target-NEUTRAL text;
        rule 2 applies to its literals like it applies to any other
        emitted literal. Applied BEFORE the operand substitutes, so an
        operand's own already-spelled literals are never re-suffixed."""
        return text if not self.suffix else _FLOAT_LIT.sub(lambda m: m.group(0) + self.suffix, text)


WGSL = Dialect(name="wgsl", fty="f32", ity="i32", bty="bool", suffix="", c_decl=False, vec_fmt="vec{n}<{t}>")
MSL = Dialect(name="msl", fty="float", ity="int", bty="bool", suffix="f", c_decl=True, vec_fmt="{t}{n}")


# --- the tables: ONE copy in the tree from here on ---------------------------
_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_CMP = {"lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "==", "ne": "!="}
_FNS = {f: f for f in ("sqrt", "exp", "log", "tanh", "abs", "floor", "sin", "cos")}
_MINMAX = {"maximum": "max", "minimum": "min"}
_CORE_INFIX = {"core.add": "+", "core.sub": "-", "core.mul": "*", "core.div": "/"}


class Gen:
    """The expression walker: CSE by node id, one statement per node.

    A node emits at most once — ``names`` keys on ``id(node)``, which is
    the sharing the builder already established (the same Node object IS
    the same value). Bools stay bool-typed and widen only where a float
    operand is wanted, because both languages spell the widening the
    same way and neither has an implicit bool→float coercion.

    ``leaf`` is the stage's escape hatch: ``(node, gen) -> (expr, bool)``
    or ``None`` to fall through to the shared table. Leaves are where a
    stage's resources live (a compute param is a buffer read; a fragment
    param is a varying), so they cannot live in the table.
    """

    def __init__(self, d: Dialect, leaf):
        self.d = d
        self.leaf = leaf
        self.lines: list[str] = []
        self.names: dict[int, str] = {}
        self.bools: set[int] = set()
        self.math: list[str] = []  # applied mathrows, in first-application order
        self.n = 0

    def go(self, node) -> str:
        if id(node) in self.names:
            return self.names[id(node)]
        expr, is_bool = self.expr(node)
        var = f"e{self.n}"
        self.n += 1
        self.lines.append(self.d.decl(self.d.bty if is_bool else self.d.fty, var, expr))
        self.names[id(node)] = var
        if is_bool:
            self.bools.add(id(node))
        return var

    def operand(self, node) -> str:
        v = self.go(node)
        if id(node) not in self.bools:
            return v
        return f"select({self.d.lit(0.0)}, {self.d.lit(1.0)}, {v})"

    def cond(self, node) -> str:
        v = self.go(node)
        return v if id(node) in self.bools else f"({v} != {self.d.lit(0.0)})"

    def call(self, marker: str, spelling: str, ops) -> str:
        """A marker-function site — and the ONE place the target numeric
        contract applies. A row substitutes its spelling for every
        dialect (the tanh clamp is free where the library is already
        correct, and its freeness is proven, mathrows.py); the applied
        names ride out on ``math`` so the artifact can say what it
        substituted (LaunchContract.math)."""
        row = mathrows.row(marker)
        if row is None:
            return f"{spelling}({', '.join(ops)})"
        if marker not in self.math:
            self.math.append(marker)
        return self.d.rowtext(row.spelling).format(*ops)

    def expr(self, node) -> tuple[str, bool]:
        got = self.leaf(node, self)
        if got is not None:
            return got
        if node.regions:  # dataflow is args; a carried region is another tier's body
            raise Untranslatable(f"{node.op} carries a region (structured control flow reaches the device at L4)")
        attrs = dict(node.attrs)
        op = node.op
        if op in ("core.const", "tl.const"):
            return self.d.lit(attrs["value"]), False
        if op == "tl.pointwise":
            f = attrs["f"]
            ops = [self.operand(a) for a in node.args]
            if f in _INFIX:
                return f"({ops[0]} {_INFIX[f]} {ops[1]})", False
            if f == "neg":
                return f"(-{ops[0]})", False
            if f in _CMP:
                return f"({ops[0]} {_CMP[f]} {ops[1]})", True
            if f == "where":
                return f"select({ops[2]}, {ops[1]}, {self.cond(node.args[0])})", False
            if f in _MINMAX:
                return self.call(f, _MINMAX[f], ops), False
            if f in _FNS:
                return self.call(f, _FNS[f], ops[:1]), False
            raise Untranslatable(f"marker {f!r}")
        if op in _CORE_INFIX:
            a, b = (self.operand(x) for x in node.args)
            return f"({a} {_CORE_INFIX[op]} {b})", False
        if op == "core.neg":
            return f"(-{self.operand(node.args[0])})", False
        if op == "core.cmp":
            a, b = (self.operand(x) for x in node.args)
            return f"({a} {_CMP[attrs['pred']]} {b})", True
        if op == "core.select":
            t_, e_ = self.operand(node.args[1]), self.operand(node.args[2])
            return f"select({e_}, {t_}, {self.cond(node.args[0])})", False
        if op.startswith(("pw.", "math.")):  # the open marker families (290 §4.1)
            f = op.split(".", 1)[1]
            ops = [self.operand(a) for a in node.args]
            if f in _FNS:
                return self.call(f, _FNS[f], ops[:1]), False
            if f in _MINMAX:
                return self.call(f, _MINMAX[f], ops), False
            raise Untranslatable(f"scalar op {op}")
        raise Untranslatable(op)


# --- the structural region walk ----------------------------------------------


def walk(region):
    """Post-order over dataflow: a node's args, then the node, each once.

    Regions are NOT descended (see the module docstring): they carry
    another tier's body as data. Params are reached as args like any
    other node — the leaf row decides what a param means."""
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


@dataclass(frozen=True)
class ComputeRows:
    """A compute region's statements plus the facts a shell needs. The
    shell (how buffers declare, what the entry point is called, where
    the thread size goes) is the column's; everything here is shared."""

    lines: tuple[str, ...]
    stores: tuple[str, ...]
    axes: tuple[str, ...]  # lattice dim names, outer to inner
    comp: dict  # dim name -> the gid component that carries it
    extents: dict  # dim name -> (start, stop)
    slots: tuple  # (name, offset, fmt) HOST staging rows, offset-ordered
    bound: tuple[int, ...]  # region-param index per binding slot (see below)
    writable: tuple[bool, ...]  # parallel to `bound`
    math: tuple[str, ...]  # applied mathrows, by name

    def threads(self) -> tuple[int, int, int]:
        """The lattice extent IN THREADS — target-neutral lattice
        arithmetic. How many groups that is, is the runtime's affair
        (and on Metal it is nobody's: ``dispatchThreads:`` takes this)."""
        ext = [self.extents[a][1] - self.extents[a][0] for a in self.axes]
        return (ext[1], ext[0], 1) if len(self.axes) == 2 else (ext[0], 1, 1)

    def guard_expr(self) -> str:
        """The overhang predicate, for a runtime that launches whole
        groups. Dead where the runtime launches exact grids — which is
        why it is a contract clause and not universal source (210)."""
        return " || ".join(f"gid.{self.comp[a]} >= {self.extents[a][1] - self.extents[a][0]}u" for a in self.axes)


def default_thread_size(rows: ComputeRows) -> tuple[int, int, int]:
    """The naive one-thread-per-point policy both compute columns start
    from. Thread sizing SPECIALIZES (owner-ruled) wherever it lands, so
    this is a default the caller overrides, never a constant."""
    return (8, 8, 1) if len(rows.axes) == 2 else (64, 1, 1)


def compute_bindings(rows: ComputeRows) -> tuple[Binding, ...]:
    """The compute stage's binding table, as DATA. One stage, so one
    table; the render era re-indexes per stage (Metal's vertex and
    fragment tables are separate, 210). Row k binds the launch value at
    ``rows.bound[k]`` — the two lists are the interface between the
    generated text and the launcher."""
    out = [Binding(f"buf{k}", k, "storage", w) for k, w in enumerate(rows.writable)]
    if rows.slots:
        out.append(Binding("U", len(rows.writable), "storage", False))
    return tuple(out)


def _row_major(dims) -> list[int]:
    """Element strides over a type's own dims, innermost contiguous."""
    strides, acc = [], 1
    for d in reversed(dims):
        strides.append(acc)
        acc *= d.size
    return list(reversed(strides))


def compute_rows(art, d: Dialect) -> ComputeRows:
    """A kernel artifact's region -> statements, in any dialect.

    Buffers index ROW-MAJOR over their own type dims, each coordinate
    the gid component of that dim NAME — the alignment law's gift, and
    the reason the launcher's contiguous repack is the matching upload.
    (The strided-view read the demo proved needs the launch VALUES,
    which a generator does not have; it arrives with residency.)

    A parameter the body never touches gets NO binding slot: WebGPU's
    ``layout="auto"`` prunes an unreferenced resource from the pipeline
    layout, so declaring one makes the bind group unbuildable — a
    runtime rule the backend must anticipate in source (210). Pruning is
    harmless everywhere else (Metal ignores unused arguments; CUDA's
    table IS the parameter list), so it happens once, here.
    """
    region = art.region
    params = list(region.params)
    if not art.tensor_params:
        raise Untranslatable("a kernel with no tensor parameters")
    lattice = params[len(art.tensor_params) - 1]  # the writable target (S.3 convention)
    axes = tuple(dim.name for dim in lattice.type.dims)
    if len(axes) > 2:
        raise Untranslatable("rank-3+ lattices (the 2D/1D subset translates today)")
    comp = dict(zip(axes, ("y", "x") if len(axes) == 2 else ("x",)))
    extents = {dim.name: (dim.start, dim.stop) for dim in lattice.type.dims}

    slots = list(art.uniforms)  # (name, offset, fmt) — the kernel's own captures
    for _pname, _fixed, _wrap, base, plan, _extract in art.arg_slots:  # spliced fn-arg blocks
        slots += [(f"arg{base}", base + s.dest.offset, s.dest.fmt) for s in plan.slots]
    slots.sort(key=lambda s: s[1])
    slot_index = {off: i for i, (_n, off, _f) in enumerate(slots)}

    # Binding slots, assigned in parameter order over the params the body
    # actually reaches: the walk only yields nodes reachable from the body,
    # so a param absent from it is a param no row names.
    reached = {id(n) for n in walk(region) if n.op == "core.param"}
    bound = tuple(i for i, p in enumerate(params) if id(p) in reached)
    slot_of = {i: k for k, i in enumerate(bound)}

    def elem_index(node) -> str:
        parts = []
        for dim, stride in zip(node.type.dims, _row_major(node.type.dims)):
            if dim.name not in comp:
                raise Untranslatable(f"a buffer dim {dim.name!r} outside the launch lattice")
            parts.append(f"{d.ity}(gid.{comp[dim.name]}) * {stride}")
        return " + ".join(parts) or "0"

    def leaf(n, g):
        if n.op == "core.param":
            return d.buffer_row(d, slot_of[params.index(n)], elem_index(n)), False
        if n.op == "tl.iota":
            name = dict(n.attrs)["name"]
            if name not in comp:
                raise Untranslatable(f"an iota over dim {name!r} outside the launch lattice")
            return d.iota_row(d, comp[name]), False
        if n.op == "abi.slot":
            return d.slot_row(d, slot_index[dict(n.attrs)["offset"]]), False
        if n.op == "tl.read":
            tex, *idx = n.args
            if tex.op != "core.param":
                raise Untranslatable("a read of a non-parameter tensor")
            strides = _row_major(tex.type.dims)
            # No bounds check: the reference refuses OOB, so any case that ran
            # there is certified UB-free here (the keying-ladder ruling, 210).
            parts = [f"{d.ity}({g.operand(ix)}) * {st}" for ix, st in zip(idx, strides)]
            return d.buffer_row(d, slot_of[params.index(tex)], " + ".join(parts)), False
        return None

    gen = Gen(d, leaf)
    stores = []
    for n in walk(region):
        if n.op == "tl.store":  # the op table, not the arg positions: dst is a place
            _tok, dst, val = n.args
            if dst.op != "core.param":
                raise Untranslatable("a store into a non-parameter tensor")
            stores.append(f"  {d.buffer_row(d, slot_of[params.index(dst)], elem_index(dst))} = {gen.operand(val)};")

    n_tensors = len(art.tensor_params)
    writable = tuple(
        (art.tensor_params[i] in art.writable) if i < n_tensors else True  # tap buffers write
        for i in bound
    )
    return ComputeRows(
        lines=tuple(gen.lines),
        stores=tuple(stores),
        axes=axes,
        comp=comp,
        extents=extents,
        slots=tuple(slots),
        bound=bound,
        writable=writable,
        math=tuple(gen.math),
    )
