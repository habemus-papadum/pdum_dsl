"""The target-NEUTRAL half of the demo: program -> artifacts.

Everything in this file is shared verbatim by both backends. It is the
answer to "how much of a graphics backend is actually per-target?", and
the answer is: the shells, not the rows.

Four sections:

  1. RESIDENCY -- one device buffer per host ``Buffer``, and the element
     index expression for a strided view into it. Lifted from
     spike_runner/residency.py with the one device call (buffer creation)
     replaced by an injected ``alloc`` callback, which is all it took to
     make it target-neutral. ``index_expr`` was ALREADY target-neutral:
     it emits arithmetic over coordinate strings the caller supplies.

  2. DIALECT -- the three lexical rules spike_metal measured (cast
     spelling, literal suffix, C declaration form) reified as four
     fields. This is the experiment: if the rows really are portable
     modulo three rules, one generator parameterized by this dataclass
     should emit both languages with no per-target branches in the
     operator table. It does. See ``_Gen`` -- there is no ``if
     dialect.name == ...`` anywhere below.

  3. LOWERING, ONCE -- ``lower_program`` runs every pdum-side lowering
     (the compute artifact, ``_lower_vertex``, ``_lower_fragment``) and
     returns a ``Lowered`` both backends consume. ``LOWERINGS`` counts
     the calls; the demo asserts it stays at 1 across a whole animated
     run on both backends.

  4. THE SLOT CHANNEL -- staging bytes for the three stages. This is the
     mechanism the whole demo rests on: a mouse-driven scalar is a
     CAPTURED SCALAR that rides ``abi.slot``, so moving the mouse
     rewrites a handful of bytes and never touches the IR, the shader
     text, or the pipeline.

Known duplication, deliberate and on the hurt-list already (spike_runner
H3 / spike_metal FAIL-4): the staging pack loop appears here for the
FOURTH and FIFTH time in the repo, because it is fused into
``_Artifact.launch`` and into ``graphics.render`` and cannot be reached
without a launch. And the device REPRESENTATION of the staging plan (a
flat ``array<f32>``, one slot per element) is invented here for both
targets -- the plan describes host bytes and stops.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from pdum.tl.dialect import _thaw_params, walk_region

# ===========================================================================
# 1. residency
# ===========================================================================

HOST_ITEM = 8  # the reference interior is f64
DEV_ITEM = 4  # both device languages are f32


class Untranslatable(Exception):
    """This region has no translation yet -- the reason names the op."""


@dataclass
class Resident:
    """One host Buffer's device twin. ``gpu`` is whatever the injected
    allocator returned -- a wgpu.GPUBuffer or an MTLBuffer; nothing in
    this file looks inside it."""

    gpu: object
    n_f32: int
    label: str


class Residency:
    """host Buffer identity -> ONE device buffer, for either target.

    The payoff is the same on both: the cylinder mesh is one host buffer
    with two field views (theta at offset 0, h at offset 8, both stride
    16). The compute kernel reads/writes the field views; the vertex
    shader pulls the record. That is one upload and one binding, because
    every leaf indexes through the view's own layout instead of assuming
    a contiguous repack.
    """

    def __init__(self, alloc):
        self._alloc = alloc  # (np.ndarray f32) -> device buffer handle
        self._by_id: dict[int, Resident] = {}
        self._keep: list = []

    def resident(self, tensor, label: str = "") -> Resident:
        buf = tensor.buffer
        key = id(buf)
        got = self._by_id.get(key)
        if got is not None:
            return got  # the second view of the same buffer: no second upload
        if buf.data is None:
            raise RuntimeError(f"{buf!r} is not host-readable")
        data = np.frombuffer(buf.data, dtype=np.float64).astype(np.float32)
        got = Resident(gpu=self._alloc(data), n_f32=data.size, label=label or f"buf@{key:x}")
        self._by_id[key] = got
        self._keep.append(buf)
        return got

    def residents(self) -> list[Resident]:
        return list(self._by_id.values())


def index_expr(view, coords: dict, ity: str) -> str:
    """The device ELEMENT index of ``view`` at ``coords``, as source text.

    ``coords`` maps dim name -> a Python int (a selected constant) or an
    integer expression string. ``ity`` is the dialect's int cast spelling,
    the ONLY target-dependent thing here and it is not even used unless a
    caller wants a cast in its coordinate.

    Host f64 byte strides divide by 8; device f32 element indices are the
    same numbers, because both sides scale by their own itemsize. That is
    why a stride-16 field view over a record buffer reads in place with no
    repack.
    """
    off = view.layout.offset
    if off % HOST_ITEM:
        raise ValueError(f"layout offset {off} is not f64-aligned")
    const = off // HOST_ITEM
    parts: list[str] = []
    for d in view.layout.dims:
        if d.stride % HOST_ITEM:
            raise ValueError(f"dim {d.name}: byte stride {d.stride} is not f64-aligned")
        step = d.stride // HOST_ITEM
        if d.name not in coords:
            raise KeyError(f"no coordinate for dim {d.name!r}")
        c = coords[d.name]
        if isinstance(c, int):
            const += (c - d.start) * step
            continue
        base = c if d.start == 0 else f"({c} - {d.start})"
        parts.append(f"({base})" if step == 1 else f"({base}) * {step}")
    if const or not parts:
        parts.append(str(const))
    return " + ".join(parts)


# ===========================================================================
# 2. the dialect -- the three lexical rules, and nothing else
# ===========================================================================


@dataclass(frozen=True)
class Dialect:
    """What separates WGSL from MSL at the ROW level. Four fields.

    spike_metal measured 123/123 emitted compute statements converting
    under exactly these rules. This demo extends the claim to the render
    stage by generating both languages' rows from one walker.
    """

    name: str
    fty: str  # the float type NAME, used for both declarations and casts
    ity: str  # the int cast spelling
    bty: str  # the bool type name
    suffix: str  # the float literal suffix
    c_decl: bool  # True: `float v = e;`   False: `let v: f32 = e;`

    def lit(self, v) -> str:
        s = repr(float(v))
        if "inf" in s or "nan" in s:
            raise Untranslatable(f"a non-finite literal {s!r}")
        return (s if ("." in s or "e" in s) else s + ".0") + self.suffix

    def decl(self, ty: str, var: str, expr: str) -> str:
        return f"  {ty} {var} = {expr};" if self.c_decl else f"  let {var}: {ty} = {expr};"


WGSL = Dialect(name="wgsl", fty="f32", ity="i32", bty="bool", suffix="", c_decl=False)
MSL = Dialect(name="msl", fty="float", ity="int", bty="bool", suffix="f", c_decl=True)


_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_CMP = {"lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "==", "ne": "!="}
_FNS = {f: f for f in ("sqrt", "exp", "log", "tanh", "abs", "floor", "sin", "cos")}
_CORE_INFIX = {"core.add": "+", "core.sub": "-", "core.mul": "*", "core.div": "/"}


class _Gen:
    """ONE expression generator, both languages.

    Every operator row below is shared: ``select(f, t, cond)`` has the
    same argument order and semantics in WGSL and MSL, ``max``/``min``
    and all eight math builtins spell the same, and the bool->float
    widening is the same. The dialect supplies the type names, the
    literal suffix and the declaration form; nothing else varies.
    """

    def __init__(self, d: Dialect, leaf):
        self.d = d
        self.leaf = leaf
        self.lines: list[str] = []
        self.names: dict[int, str] = {}
        self.bools: set[int] = set()
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

    def expr(self, node) -> tuple[str, bool]:
        got = self.leaf(node, self)
        if got is not None:
            return got
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
            if f in ("maximum", "minimum"):
                return f"{'max' if f == 'maximum' else 'min'}({ops[0]}, {ops[1]})", False
            if f in _FNS:
                return f"{_FNS[f]}({ops[0]})", False
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
        if op.startswith("pw."):
            f = op[3:]
            if f in _FNS:
                return f"{_FNS[f]}({self.operand(node.args[0])})", False
            if f in ("maximum", "minimum"):
                a, b = (self.operand(x) for x in node.args)
                return f"{'max' if f == 'maximum' else 'min'}({a}, {b})", False
        raise Untranslatable(op)


# ===========================================================================
# 3. lowering, ONCE -- and the binding plans, as DATA
# ===========================================================================

LOWERINGS: list[str] = []  # every lower_program call appends; the demo asserts len == 1
TRANSLATIONS: list[str] = []  # every rows generation appends; must not grow inside the loop


@dataclass
class BufferBinding:
    """One resident buffer at one binding slot. The binding table is
    returned as DATA, never left implicit in the shader text -- Metal has
    no bind-group object and WebGPU's ``layout="auto"`` prunes, so the
    two targets can only agree if the plan is explicit (spike_metal
    FAIL-3)."""

    binding: int
    resident: Resident
    label: str


class _Binder:
    """Assigns binding slots in first-use order, so a buffer touched by
    two params (a record's two field views) binds ONCE."""

    def __init__(self, res: Residency):
        self.res = res
        self.order: list[BufferBinding] = []
        self._by_id: dict[int, BufferBinding] = {}

    def slot(self, tensor, label: str) -> int:
        key = id(tensor.buffer)
        got = self._by_id.get(key)
        if got is None:
            got = BufferBinding(len(self.order), self.res.resident(tensor, label), label)
            self._by_id[key] = got
            self.order.append(got)
        return got.binding


@dataclass
class Lowered:
    """Everything the pdum tier produces for this program, produced ONCE
    and handed to every backend. Nothing in here mentions a target."""

    # the compute stage
    art: object
    values: tuple  # the kernel's tensor arguments, in region-param order
    labels: tuple
    # the vertex stage
    v_region: object
    v_ctx: dict = field(repr=False)
    v_slots: tuple = ()
    varyings: tuple = ()
    flats: frozenset = frozenset()
    draw_count: int = 0
    vs_args: tuple = ()
    # the fragment stage
    f_region: object = None
    f_ctx: dict = field(repr=False, default_factory=dict)
    f_slots: tuple = ()
    fparams: tuple = ()
    fs_args: tuple = ()
    vs_fn: object = None
    fs_fn: object = None


def artifact_of(kernel, *args):
    """The compiled compute artifact, without launching it.

    There is no public door (spike_runner H2): ``ComputeKernel.__call__``
    keys, compiles AND launches, and the artifact never escapes
    ``_invoke``. So we rebuild the cache key from four private helpers.
    If ``_invoke``'s key construction ever drifts we silently compile a
    second artifact into the same Memo.
    """
    from pdum.tl.kernel import KERNELS, _arg_fp, _code_fp, _compile, _env_fp

    key = (_code_fp(kernel.fn), _env_fp(kernel.fn), tuple(_arg_fp(a) for a in args), ())
    return KERNELS.get_or_compile(key, lambda: _compile(kernel.fn, args, (), None))


def lower_program(kernel, kernel_args, labels, pso, vs_args, fs_args) -> Lowered:
    """Every lowering the program needs, exactly once."""
    from pdum.tl.dialect import tensor_type
    from pdum.tl.graphics import _check_pairing, _lower_fragment, _lower_vertex
    from pdum.tl.tensor import Tensor

    LOWERINGS.append(getattr(kernel, "fn", kernel).__qualname__)

    art = artifact_of(kernel, *kernel_args)
    bound = dict(zip(art.params, kernel_args))
    values = tuple(bound[n] for n in art.tensor_params)

    v_region, vnames, flats, count, _lat, v_ctx = _lower_vertex(pso.vs, tuple(vs_args))
    _check_pairing(frozenset(vnames), pso.required)
    v_slots = tuple(sorted((dict(nd.attrs)["offset"], dict(nd.attrs)["fmt"]) for nd in v_ctx["k.uniforms"].values()))

    # The fragment is lowered against a DUMMY (2, 2) varying lattice.
    # ``_lower_fragment`` demands lattice-shaped varying types and does
    # not use their extents -- a false dependency on resolution, and the
    # only reason ``render_wgpu`` takes ``shape`` at all (spike_runner
    # FINDINGS). Because it is false, ONE compiled pipeline serves every
    # window size, which is what makes a resizable window free.
    ref_lat = Tensor.from_numpy(np.zeros((2, 2)), ("y", "x"))
    v_types = {n: tensor_type(ref_lat) for n in vnames}
    f_region, f_ctx, _touched, fparams = _lower_fragment(pso.fs, tuple(fs_args), v_types)
    if f_ctx["k.fn_markers"]:
        raise Untranslatable("an oracle-class fragment fn-argument on the device")

    f_slots = []
    for nd in f_ctx["k.uniforms"].values():
        at = dict(nd.attrs)
        f_slots.append((at["offset"], at["fmt"]))
    for blk in f_ctx["k.arg_plans"].values():
        for s in blk["plan"].slots:
            f_slots.append((blk["base"] + s.dest.offset, s.dest.fmt))

    return Lowered(
        art=art,
        values=values,
        labels=tuple(labels),
        v_region=v_region,
        v_ctx=v_ctx,
        v_slots=v_slots,
        varyings=tuple(vnames),
        flats=flats,
        draw_count=count,
        vs_args=tuple(vs_args),
        f_region=f_region,
        f_ctx=f_ctx,
        f_slots=tuple(sorted(f_slots)),
        fparams=tuple(fparams),
        fs_args=tuple(fs_args),
        vs_fn=pso.vs.fn,
        fs_fn=pso.fs.fn,
    )


# ===========================================================================
# 3b. the ROWS -- one generator, both languages
# ===========================================================================


@dataclass
class ComputeRows:
    buffers: tuple  # BufferBinding, in binding order
    slots: tuple  # (offset, fmt) sorted -- the staging read plan
    slot_binding: int | None
    lines: tuple  # the emitted statements
    stores: tuple
    axes: tuple
    extents: dict
    threads: tuple  # the launch geometry, in THREADS (target-neutral)


def compute_rows(low: Lowered, res: Residency, d: Dialect) -> ComputeRows:
    """The kernel region -> statements + a binding plan, in either language.

    The shell (how buffers are declared, what the entry point looks like,
    where the workgroup size goes) is the backend's; this is everything
    else.
    """
    TRANSLATIONS.append(f"{d.name}:compute")
    art, values = low.art, low.values
    params = list(art.region.params)
    if not art.tensor_params:
        raise Untranslatable("a kernel with no tensor parameters")
    lattice = params[len(art.tensor_params) - 1]
    axes = tuple(dim.name for dim in lattice.type.dims)
    if len(axes) > 2:
        raise Untranslatable("rank-3+ lattices (the 2D/1D subset translates today)")
    comp = dict(zip(axes, ("y", "x") if len(axes) == 2 else ("x",)))
    extents = {dim.name: (dim.start, dim.stop) for dim in lattice.type.dims}

    slots = [(o, f) for _n, o, f in art.uniforms]
    for _pname, _fixed, _wrap, base, plan, _extract in art.arg_slots:
        slots += [(base + s.dest.offset, s.dest.fmt) for s in plan.slots]
    slots.sort()
    slot_index = {off: i for i, (off, _) in enumerate(slots)}

    binder = _Binder(res)
    labels = low.labels or tuple(art.tensor_params)

    def read_at(p) -> str:
        i = params.index(p)
        view = values[i]
        coords = {}
        for dim in view.layout.dims:
            if dim.name not in comp:
                raise Untranslatable(f"a buffer dim {dim.name!r} outside the launch lattice")
            coords[dim.name] = f"{d.ity}(gid.{comp[dim.name]})"
        b = binder.slot(view, labels[i] if i < len(labels) else f"p{i}")
        return f"b{b}[{index_expr(view, coords, d.ity)}]"

    def leaf(n, g):
        if n.op == "core.param":
            return read_at(n), False
        if n.op == "tl.iota":
            name = dict(n.attrs)["name"]
            if name not in comp:
                raise Untranslatable(f"an iota over dim {name!r} outside the launch lattice")
            return f"{d.fty}(gid.{comp[name]})", False
        if n.op == "abi.slot":
            return f"U[{slot_index[dict(n.attrs)['offset']]}]", False
        return None

    gen = _Gen(d, leaf)
    stores = []
    for n in walk_region(art.region):
        if n.op == "tl.store":
            _tok, dst, val = n.args
            if dst.op != "core.param":
                raise Untranslatable("a store into a non-parameter tensor")
            stores.append(f"  {read_at(dst)} = {gen.operand(val)};")

    ext = [extents[a][1] - extents[a][0] for a in axes]
    threads = (ext[1], ext[0], 1) if len(axes) == 2 else (ext[0], 1, 1)
    return ComputeRows(
        buffers=tuple(binder.order),
        slots=tuple(slots),
        slot_binding=len(binder.order) if slots else None,
        lines=tuple(gen.lines),
        stores=tuple(stores),
        axes=axes,
        extents=extents,
        threads=threads,
    )


@dataclass
class RenderRows:
    buffers: tuple  # BufferBinding pulled by the VERTEX stage
    v_slot_binding: int | None
    f_slot_binding: int | None
    v_lines: tuple
    v_outs: tuple  # (px, py, *varyings) as expressions
    f_lines: tuple
    f_out: str  # the scalar color0 expression


def render_rows(low: Lowered, res: Residency, d: Dialect) -> RenderRows:
    """The PSO's two regions -> statements + a binding plan, either language.

    Vertex pulling (250/stage-5b): there are no vertex attributes and no
    vertex-buffer layout descriptor on either target. Each attribute is a
    read of a storage buffer at the vertex index -- and because the read
    goes through ``index_expr``, it reads the RECORD BUFFER in place at
    its real stride rather than a repacked contiguous copy.
    """
    TRANSLATIONS.append(f"{d.name}:render")
    plan = list(low.v_ctx.get("g.param_plan", ()))
    buf_params = list(low.v_region.params[: len(plan)])
    binder = _Binder(res)
    v_slot_index = {off: i for i, (off, _) in enumerate(low.v_slots)}

    def pulled(pi: int, extra: dict) -> str:
        i, fname = plan[pi]
        b = low.vs_args[i]
        view, label = (b.field(fname), f"vb{i}.{fname}") if fname is not None else (b, f"vb{i}")
        coords = dict(extra)
        for dim in view.layout.dims:
            if dim.name == "vertex_id":
                coords[dim.name] = f"{d.ity}(vid)"
            elif dim.name not in coords:
                raise Untranslatable(f"a vertex buffer dim {dim.name!r} with no selected coordinate")
        return f"b{binder.slot(view, label)}[{index_expr(view, coords, d.ity)}]"

    def v_leaf(n, g):
        if n.op == "tl.iota" and dict(n.attrs).get("name") == "vertex_id":
            return f"{d.fty}(vid)", False
        if n.op == "abi.slot":
            return f"VU[{v_slot_index[dict(n.attrs)['offset']]}]", False
        if n.op == "tl.select" and n.args and n.args[0] in buf_params:
            return pulled(buf_params.index(n.args[0]), _thaw_params(dict(n.attrs))["coords"]), False
        if n.op == "core.param":
            if n in buf_params:
                return pulled(buf_params.index(n), {}), False
            raise Untranslatable("a vertex-stage param beyond the pulled buffers")
        return None

    vg = _Gen(d, v_leaf)
    v_outs = tuple(vg.operand(a) for a in low.v_region.body[-1].args[0].args)

    f_params = list(low.f_region.params)
    f_slot_index = {off: i for i, (off, _) in enumerate(low.f_slots)}

    def f_leaf(n, g):
        if n.op == "core.param":
            return f"vin.f{f_params.index(n)}", False
        if n.op == "abi.slot":
            return f"U[{f_slot_index[dict(n.attrs)['offset']]}]", False
        return None

    fg = _Gen(d, f_leaf)
    f_out = fg.operand(low.f_region.body[-1].args[0])

    nxt = len(binder.order)
    v_slot_binding = None
    if low.v_slots:
        v_slot_binding, nxt = nxt, nxt + 1
    f_slot_binding = nxt if low.f_slots else None
    return RenderRows(
        buffers=tuple(binder.order),
        v_slot_binding=v_slot_binding,
        f_slot_binding=f_slot_binding,
        v_lines=tuple(vg.lines),
        v_outs=v_outs,
        f_lines=tuple(fg.lines),
        f_out=f_out,
    )


# ===========================================================================
# 4. the slot channel -- the per-frame host work, all of it
# ===========================================================================


def kernel_staging(art) -> bytes:
    """The compute stage's staging bytes for the CURRENT captured
    environment. A lift of the block fused inside ``_Artifact.launch``
    (kernel.py:1100-1111)."""
    from pdum.dsl.pack import pack_into
    from pdum.tl.producer import _captured

    if not art.staging_size:
        return b""
    staging = bytearray(art.staging_size)
    env = _captured(art.fn) if art.uniforms else {}
    for name, off, fmt in art.uniforms:
        struct.pack_into(fmt, staging, off, env[name])
    for pname, fixed, wrap, base, plan, extract in art.arg_slots:
        if pname is not None:
            raise NotImplementedError("fn-valued kernel params are not rebound per frame here")
        f = wrap(fixed) if wrap is not None else fixed
        pack_into(plan, memoryview(staging)[base : base + plan.staging_size], extract(f.captures, ()))
    return bytes(staging)


def vertex_staging(v_ctx, vs_fn) -> bytes:
    from pdum.tl.graphics import _env_staging

    return _env_staging(v_ctx, vs_fn)


def fragment_staging(low: Lowered) -> bytes:
    """The fragment's staging block -- the third copy of a loop that
    lives in graphics.py:608-621 and wgsl_executor.py:475-489."""
    from pdum.dsl.pack import pack_into
    from pdum.tl.producer import _captured

    f_ctx = low.f_ctx
    if not f_ctx["k.uniform_size"]:
        return b""
    staging = bytearray(f_ctx["k.uniform_size"])
    bound = dict(zip(low.fparams, low.fs_args))
    env = _captured(low.fs_fn) if f_ctx["k.uniforms"] else {}
    for name, nd in f_ctx["k.uniforms"].items():
        at = dict(nd.attrs)
        struct.pack_into(at["fmt"], staging, at["offset"], env[name])
    for blk in f_ctx["k.arg_plans"].values():
        f = bound[blk["pname"]] if blk["pname"] is not None else blk["fixed"]
        if blk["wrap"] is not None:
            f = blk["wrap"](f)
        view = memoryview(staging)[blk["base"] : blk["base"] + blk["plan"].staging_size]
        pack_into(blk["plan"], view, blk["extract"](f.captures, ()))
    return bytes(staging)


def slot_values(staging: bytes, slots) -> np.ndarray:
    """Staging bytes -> the flat f32 array both shaders' ``U`` buffers
    want. Note the silent narrowing: an i32/i64 slot would land here as
    f32 on BOTH targets, which 210 forbids. No slot in this demo is
    integral, so the demo does not exercise the bug it would otherwise
    reproduce."""
    return np.asarray([struct.unpack_from(f, staging, o)[0] for o, f in slots], dtype=np.float32)


def check_swap(compiled_fn, fn) -> None:
    """``spun(angle)`` mints a fresh closure per frame, so the frame loop
    hands the pipeline a different function object than the one it was
    compiled for. That is fine -- the angle rides the slot channel and
    never enters the IR -- but nothing in the library will say so, and
    ``_env_staging`` will happily pack a structurally DIFFERENT closure
    into our layout. The guard is ours to write."""
    from pdum.tl.kernel import _code_fp, _env_fp

    if fn is compiled_fn:
        return
    a = (_code_fp(compiled_fn), _env_fp(compiled_fn))
    b = (_code_fp(fn), _env_fp(fn))
    if a != b:
        raise ValueError(
            f"this pipeline was compiled for {compiled_fn.__qualname__} with environment {a[1]}; "
            f"the frame presents {b[1]} -- a different specialization needs a different pipeline"
        )
