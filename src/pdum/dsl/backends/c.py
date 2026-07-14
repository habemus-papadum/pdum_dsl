"""The C backend: the seam generalizes beyond GPUs (design 100 §5).

The first *citizen* of the contribution point — a real target package in
``backends/``, serving the same "device" family as the Python twin through
the identical ABI: ``launch(staging, leaves)``. Rendered C99, compiled with
the system ``cc`` to a shared library, called through ``ctypes``. The point
is not speed; it is N=3 source targets over ONE IR and a second *runtime
shape* (dlopen vs exec vs wgpu) under one calling convention — the 090 §4
evidence gathered before step 14 abstracts the runtime protocol.

Deliberate v1 shape:

- ``double kernel(const unsigned char* staging, void* const* bufs)`` —
  scalar results only (tuple results are refused loudly; they belong to
  fragment targets and future DPS out-arrays).
- Staging reads go through ``memcpy`` inlines (the dense reference layout
  does not align slots; the compiler turns these into plain loads).
- Tuple-typed joins/carries are SCALARIZED: a Tuple value is N lane
  variables (``v9_0, v9_1``), declared by the construct that produces
  them. Nested tuples are refused (flat joins only, like the base pack
  emits).
- Numeric policy (070): trunc div/mod is C's native behavior — the twin
  side compensates, never this side.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from ctypes import CDLL, c_bool, c_char_p, c_double, c_float, c_int32, c_int64, c_uint32, c_uint64, c_void_p
from pathlib import Path

from ..kernel.ir import Node, Region, VerifyError
from ..kernel.pack import PackPlan
from ..kernel.registry import Backend, Out
from ..kernel.types import Array, Scalar, i64
from ..kernel.types import Tuple as TupleType

_CTYPE = {
    "f64": "double", "f32": "float", "i64": "int64_t", "i32": "int32_t",
    "u64": "uint64_t", "u32": "uint32_t", "bool": "bool",
}  # fmt: skip
_LOADER = {
    "<d": "ld_f64", "<f": "ld_f32", "<q": "ld_i64", "<i": "ld_i32",
    "<Q": "ld_u64", "<I": "ld_u32", "<?": "ld_b",
}  # fmt: skip
_RESTYPE = {
    "f64": c_double,
    "f32": c_float,
    "i64": c_int64,
    "i32": c_int32,
    "u64": c_uint64,
    "u32": c_uint32,
    "bool": c_bool,
}
_BIN = {"core.add": "+", "core.sub": "-", "core.mul": "*", "core.div": "/", "core.mod": "%"}
_PREDS = {"lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "==", "ne": "!="}

_PRELUDE = """\
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#define LD(name, T, n) static inline T name(const unsigned char* p){ T v; memcpy(&v, p, n); return v; }
LD(ld_f64, double, 8) LD(ld_f32, float, 4) LD(ld_i64, int64_t, 8) LD(ld_i32, int32_t, 4)
LD(ld_u64, uint64_t, 8) LD(ld_u32, uint32_t, 4) LD(ld_b, bool, 1)
"""


def _ctype(t) -> str:
    if isinstance(t, Scalar):
        return _CTYPE[t.kind]
    if isinstance(t, Array):
        return f"const {_CTYPE[t.dtype.kind]}*"  # a buffer value IS its element pointer
    raise VerifyError(f"the C target has no spelling for values of type {t!r}")


def _lanes(t: TupleType) -> tuple:
    for e in t.elems:
        if not isinstance(e, Scalar):
            raise VerifyError(f"the C target scalarizes FLAT tuples only, got {t!r}")
    return t.elems


def render(region: Region, plan: PackPlan, backend=None, name: str = "kernel", *, grid: bool = False) -> str:
    args_by_index = {s.source.index: s for s in plan.slots if s.source.root == "arg" and not s.source.sub}
    channel = [s for s in plan.slots if s.dest is None]

    def names_of(node: Node, nm: dict):
        """A Tuple-typed value IS its lane variables."""
        if isinstance(node.type, TupleType):
            return [f"{nm[id(node)]}_{k}" for k in range(len(_lanes(node.type)))]
        return [nm[id(node)]]

    def expr_of(node: Node, nm: dict) -> str:
        attrs = dict(node.attrs)
        arg = [nm[id(a)] for a in node.args]
        if node.op == "core.param":
            if grid:  # the GRID family: params ARE domain coordinates (loop vars)
                return f"c{attrs['index']}"
            spec = args_by_index.get(attrs["index"])
            if spec is None:
                raise VerifyError(
                    f"argument {attrs['index']} has no scalar slot (composite arguments are a recorded cut)"
                )
            return f"{_LOADER[spec.dest.fmt]}(staging + {spec.dest.offset})"
        if node.op == "abi.slot":
            return f"{_LOADER[attrs['fmt']]}(staging + {attrs['offset']})"
        if node.op == "array.buffer":
            src = attrs["src"]
            for k, s in enumerate(channel):
                if s.source.root == src[0] and (s.source.index, *s.source.sub) == (*src[1:], 0):
                    return f"((const {_ctype(node.type.dtype)}*)bufs[{k}])"
            raise VerifyError(f"no buffer leaf for {src!r}")
        if node.op == "array.dim":
            src, sub = attrs["src"], attrs["sub"]
            for s in plan.slots:
                if s.source.root == src[0] and (s.source.index, *s.source.sub) == (*src[1:], sub):
                    return f"{_LOADER[s.dest.fmt]}(staging + {s.dest.offset})"
            raise VerifyError(f"no dim slot for {src!r}[{sub}]")
        if node.op == "array.load":
            return f"{arg[0]}[{arg[1]}]"
        if node.op == "core.const":
            v = attrs["value"]
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, float) and not (v == v and abs(v) != float("inf")):
                raise VerifyError("inf/nan constants are refused (numeric policy)")
            if isinstance(v, int) and node.type == Scalar("u64"):
                # A bare literal ≥ 2^63 has no C type without ULL games, and the twin
                # does not wrap at 64 bits yet — silent divergence beats nothing ONLY
                # if it is not silent. Refuse until the wrap policy is implemented.
                raise VerifyError("u64 constants have no C spelling in v1 (twin wrap semantics pending)")
            return f"{v!r}LL" if isinstance(v, int) and node.type == Scalar("i64") else repr(v)
        if node.op == "core.mod" and node.type.kind[0] == "f":
            return f"fmod({arg[0]}, {arg[1]})"  # C's % is integers-only; fmod IS the trunc policy
        if node.op in _BIN:
            return f"{arg[0]} {_BIN[node.op]} {arg[1]}"
        if node.op == "core.pow":
            if node.type != Scalar("f64") and node.type != Scalar("f32"):
                raise VerifyError("integer pow has no C spelling in v1 — float it")
            return f"pow({arg[0]}, {arg[1]})"
        if node.op == "core.neg":
            return f"-{arg[0]}"
        if node.op == "core.cmp":
            return f"{arg[0]} {_PREDS[attrs['pred']]} {arg[1]}"
        if node.op == "core.cast":
            return f"({_ctype(attrs['to'])})({arg[0]})"
        if node.op == "core.select":
            return f"({arg[0]} ? {arg[1]} : {arg[2]})"
        if node.op == "core.extract":
            (a,) = node.args
            if isinstance(a.type, TupleType):
                return f"{nm[id(a)]}_{attrs['index']}"
            raise VerifyError(f"C extract needs a scalarized tuple, got {a.type!r}")
        table = backend.code_for_op if backend is not None else CODE_FOR_OP
        template = table.get(node.op)
        if template:
            return template.format(*arg)
        raise VerifyError(f"the C target has no rendering for {node.op!r}")

    def statement(node: Node, nm: dict):
        if isinstance(node.type, TupleType):
            if node.op != "core.tuple":
                raise VerifyError(f"C tuple values must be literal core.tuple, got {node.op!r}")
            pieces = [f"{_ctype(a.type)} {lane} = {nm[id(a)]};" for lane, a in zip(names_of(node, nm), node.args)]
            return " ".join(pieces)
        return f"{_ctype(node.type)} {nm[id(node)]} = {expr_of(node, nm)};"

    def declare(node: Node, nm: dict) -> str:
        if isinstance(node.type, TupleType):
            return " ".join(f"{_ctype(e)} {lane};" for e, lane in zip(_lanes(node.type), names_of(node, nm)))
        return f"{_ctype(node.type)} {nm[id(node)]};"

    def assign(dst: Node, src: Node, nm: dict, ind: str) -> list:
        return [f"{ind}{d} = {s};" for d, s in zip(names_of(dst, nm), names_of(src, nm))]

    def branch_join(node, nm, result_of, emit_block, path, ind):
        out = [f"{ind}{declare(node, nm)}", f"{ind}if ({nm[id(node.args[0])]}) {{"]
        for i, label in ((0, "} else {"), (1, "}")):
            out += emit_block((*path, (id(node), i)), ind + "  ")
            out += assign(node, node.regions[i].body[-1].args[0], nm, ind + "  ")
            out.append(f"{ind}{label}")
        return out

    def loop_join(node, nm, result_of, emit_block, path, ind):
        lo, hi, init = node.args
        iv, carry = node.regions[0].params
        out = [f"{ind}{declare(node, nm)}"]
        out += assign(node, init, nm, ind)
        out.append(f"{ind}for (int64_t {nm[id(iv)]} = {nm[id(lo)]}; {nm[id(iv)]} < {nm[id(hi)]}; ++{nm[id(iv)]}) {{")
        out.append(f"{ind}  {declare(carry, nm)}")
        out += assign(carry, node, nm, ind + "  ")
        out += emit_block((*path, (id(node), 0)), ind + "  ")
        out += assign(node, node.regions[0].body[-1].args[0], nm, ind + "  ")
        out.append(f"{ind}}}")
        return out

    from ._emit import emit_dominated

    indent = "  " * (len(region.params) + 1) if grid else "  "
    lines, nm, result = emit_dominated(region, statement, branch_join, indent=indent, loop=loop_join)
    if not isinstance(result.type, Scalar):
        raise VerifyError(f"the C target returns scalars in v1, got {result.type!r} (fragment tuples are WGSL's)")
    body = "\n".join(lines)
    if grid:
        rank = len(region.params)
        heads, idx = [], "c0"
        for k in range(rank):
            heads.append(f"{'  ' * (k + 1)}for (int64_t c{k} = 0; c{k} < dom[{k}]; ++c{k}) {{")
            if k:
                idx = f"({idx}) * dom[{k}] + c{k}"
        tails = [f"{'  ' * (k + 1)}}}" for k in range(rank - 1, -1, -1)]
        write = f"{'  ' * (rank + 1)}out[{idx}] = {nm[id(result)]};"
        ot = _ctype(result.type)
        sig = f"void {name}(const unsigned char* staging, void* const* bufs, {ot}* out, const int64_t* dom)"
        return (
            f"/* pdum-restype: grid {result.type.kind} rank {rank} */\n{_PRELUDE}\n"
            f"#if defined(_WIN32)\n__declspec(dllexport)\n#endif\n"
            f"{sig} {{\n" + "\n".join(heads) + f"\n{body}\n{write}\n" + "\n".join(tails) + "\n}\n"
        )
    sig = f"{_ctype(result.type)} {name}(const unsigned char* staging, void* const* bufs)"
    return (
        f"/* pdum-restype: {result.type.kind} */\n{_PRELUDE}\n"
        f"#if defined(_WIN32)\n__declspec(dllexport)\n#endif\n"
        f"{sig} {{\n{body}\n  return {nm[id(result)]};\n}}\n"
    )


def is_available() -> bool:
    return shutil.which("cc") is not None or shutil.which("clang") is not None


def compile_source(source: str, name: str = "kernel"):
    cc = shutil.which("cc") or shutil.which("clang")
    if cc is None:
        raise RuntimeError("no C compiler on PATH (`cc`/`clang`) — the C target is probe-gated")
    d = Path(tempfile.mkdtemp(prefix="pdum-c-"))
    src, lib = d / f"{name}.c", d / f"{name}.so"
    src.write_text(source)
    cmd = [cc, "-O2", "-shared", "-fPIC", str(src), "-o", str(lib), "-lm"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VerifyError(f"cc failed:\n{proc.stderr}\n--- source ---\n{source}")
    dll = CDLL(str(lib))
    fn = getattr(dll, name)
    meta = source.split("pdum-restype: ", 1)[1].split(" */", 1)[0].split()
    kind = meta[0]
    if kind == "grid":  # domain kernel: void fn(staging, bufs, out, dom)
        fn.restype = None
        fn.argtypes = (c_char_p, c_void_p, c_void_p, c_void_p)
    else:
        fn.restype = _RESTYPE[kind]
        fn.argtypes = (c_char_p, c_void_p)

    class CProgram:
        __pdum_source__ = source
        _keepalive = (dll, d)
        # The grid CONTRACT rides the artifact (stage-2 review: the launcher
        # cannot validate what it cannot see — dtype/rank mismatches were
        # silent corruption): element kind + domain rank, from the header.
        grid_kind = meta[1] if kind == "grid" else None
        grid_rank = int(meta[3]) if kind == "grid" else None

        @staticmethod
        def call(staging, buffers):
            ptrs = (c_void_p * max(1, len(buffers)))(*(b.ctypes.data for b in buffers))
            return fn(bytes(staging), ptrs)

        @staticmethod
        def grid_call(staging, buffers, out, dom):
            import numpy as np

            ptrs = (c_void_p * max(1, len(buffers)))(*(b.ctypes.data for b in buffers))
            d = np.asarray(dom, dtype=np.int64)
            fn(bytes(staging), ptrs, out.ctypes.data, d.ctypes.data)
            return out

    return CProgram


# --- the GRID family: params are integer domain coordinates (130 §4.3) ---------
# The in-artifact domain loop — ONE dispatch fills the whole out array, which
# is what repays the ray-march verdict (per-lane dispatch drowned a 7x body
# win). `over`'d kernels compose unchanged: the lane is just one more
# coordinate, and the batch axis joins the domain exactly as 110 predicted.


def _grid_param_types(target) -> tuple:
    from ..stdlib.transforms import Over

    lanes, t = 0, target
    while not hasattr(t, "pyfunc"):  # unwrap over-chains: each adds one coordinate
        if isinstance(t, Over):
            lanes, t = lanes + 1, t.captures[0]
            continue
        raise VerifyError(f"the grid launches kernels and over-chains; {type(t).__name__} has no domain contract")
    return (i64,) * (t.pyfunc.__code__.co_argcount + lanes)


def _grid_plan(env_types, arg_types, table):
    from ..kernel.pack import plan_from_types

    return plan_from_types(env_types, (), table)  # coords are loop vars, never staging


def render_grid(region: Region, plan: PackPlan, backend=None, name: str = "kernel") -> str:
    return render(region, plan, backend, name, grid=True)


_NPKIND = {"f64": "float64", "f32": "float32", "i64": "int64", "i32": "int32", "u64": "uint64", "u32": "uint32"}


def make_grid_launcher(artifact, plan: PackPlan):
    def launch(staging, leaves):
        import numpy as np

        dom_spec = [x for x in leaves if isinstance(x, Out)]
        if not dom_spec:
            raise VerifyError("grid kernels need a domain: k(out=(N, ...)) or an out ARRAY to fill")
        spec = dom_spec[-1].value
        buffers = tuple(x for x in leaves if not isinstance(x, Out))
        want = np.dtype(_NPKIND[artifact.grid_kind])
        if isinstance(spec, np.ndarray):
            if spec.dtype != want:  # raw pointer writes: a mismatch is heap corruption, not coercion
                raise VerifyError(f"grid out dtype {spec.dtype} != kernel result {want} — pass a matching array")
            if not spec.flags["C_CONTIGUOUS"]:
                raise VerifyError("grid out must be C-contiguous (the artifact writes row-major from the base)")
            out = spec  # adopt: the array's shape IS the domain
        else:
            shape = (spec,) if isinstance(spec, int) else tuple(spec)
            out = np.empty(shape, dtype=want)
        if out.ndim != artifact.grid_rank:
            raise VerifyError(f"this kernel has {artifact.grid_rank} domain coordinate(s); out has rank {out.ndim}")
        return artifact.grid_call(staging, buffers, out, out.shape)

    return launch


def make_launcher(artifact, plan: PackPlan):
    def launch(staging, leaves):
        if any(isinstance(x, Out) for x in leaves):  # peel by TYPE, never tail (the Out contract)
            raise VerifyError("the C target takes no launch domain (out= is for compute families)")
        return artifact.call(staging, leaves)

    return launch


CODE_FOR_OP: dict = {
    "math.sqrt": "sqrt({0})",
    "math.exp": "exp({0})",
    "math.sin": "sin({0})",
    "math.cos": "cos({0})",
    "math.floor": "floor({0})",
    "math.abs": "fabs({0})",
    "math.min": "fmin({0}, {1})",
    "math.max": "fmax({0}, {1})",
    "array.load": None,  # native (rendered inline above); presence gates decompositions
    "core.tuple": None,  # scalarized natively — keeps the extract-of-tuple fold OFF
}

C = Backend(
    name="backends.c",
    render=render,
    compile=compile_source,
    fp=("backends.c", 1),
    make_launcher=make_launcher,
    code_for_op=CODE_FOR_OP,
)


C_GRID = Backend(
    name="backends.c.grid",
    render=render_grid,
    compile=compile_source,
    fp=("backends.c.grid", 1),
    plan=_grid_plan,
    param_types=_grid_param_types,
    make_launcher=make_grid_launcher,
    code_for_op=CODE_FOR_OP,
)


def install_grid(registry, *, default: bool = False, kinds: tuple = ()) -> None:
    """The GRID family record: explicit choice, like the scalar C target."""
    from dataclasses import replace

    registry.register_backend(replace(C_GRID, code_for_op=dict(C_GRID.code_for_op)), default=default, kinds=kinds)


def install(registry, *, default: bool = False) -> None:
    """Never claims the default slot unless asked: the Python twin stays the
    'device' family's routed target; the C target is an explicit choice
    (tier-2 override — a child registry with `default=True` — until the
    device axis brings tier-1 dispatch at step 14)."""
    from dataclasses import replace

    registry.register_backend(replace(C, code_for_op=dict(C.code_for_op)), default=default)
