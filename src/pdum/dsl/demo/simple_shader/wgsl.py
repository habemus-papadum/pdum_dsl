"""The WGSL backend: the seam proven on a GPU. SATELLITE — zero kernel edits.

Compute leads, fragment follows, one runtime module (070 §5, R12). The
compute-family contract v1: **a compute kernel's parameters ARE its thread
coordinates** — `f(i, j)` runs once per grid point with `i = f32(gid.x)`,
`j = f32(gid.y)` — and the call passes the launch DOMAIN, not per-thread
arguments: ``k(out=(W, H))`` dispatches a W×H grid and returns the result
read back as floats. The domain is launcher data (it rides the leaves
channel; changing resolution NEVER recompiles — grid strips, 070 §3). The
fragment variant is the same body called with pixel coordinates, its scalar
result broadcast to grayscale rgba (colors arrive with tuples, step 10).

Backend policies, stated once:

- **Narrowing lives here** (§2.10 ``type_map`` made real): WGSL has no f64
  and no i64 — this backend computes f64 as ``f32``, i64 as ``i32``, in both
  the rendered text and the uniform layout. Differential tests vs the Python
  backend therefore compare within f32 tolerance.
- **The uniform layout is the plan**: ``_uniform_dest`` is this backend's
  ``dest_for`` policy — scalar leaves become 4-byte slots in one uniform
  struct (members named by byte offset: ``env.m8`` — the prefix is ``m``,
  because ``f16``/``f32`` at those offsets are reserved WGSL words). Same ``plan_from_types``
  machinery as ch08, different bytes. Compute ARGS are excluded from the
  plan: they are thread coordinates, not packed values.
- **`@workgroup_size` is pipeline-creation-time** (R12), so the workgroup
  size is baked into the rendered text = the artifact key. v1 fixes it
  (64 linear / 8×8 planar); the value-specialized ``block`` bracket lands
  with the config-schema surface.
- **Per frame** (the M0 discipline, now behind FastRecord): pack staging →
  ``queue.write_buffer`` → encode → dispatch/draw → submit. Pipelines and
  buffers live in the artifact; readback only at the boundary.
"""

from __future__ import annotations

import struct

from ...backends._emit import emit_dominated
from ...combinators import register_role
from ...kernel.ir import Node, Region, VerifyError
from ...kernel.pack import PackedDest, PackPlan, ScalarLeaf, plan_from_types
from ...kernel.registry import DEFAULT, Backend
from ...kernel.types import Scalar, f64

_BIN = {"core.add": "+", "core.sub": "-", "core.mul": "*", "core.div": "/", "core.mod": "%"}
_PREDS = {"lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "==", "ne": "!="}
_WTYPE = {"f64": "f32", "f32": "f32", "i64": "i32", "i32": "i32", "u64": "u32", "u32": "u32", "bool": "bool"}
_WFMT = {"f32": "<f", "i32": "<i", "u32": "<I", "bool": "<I"}  # bool packs as u32 in uniform blocks
WORKGROUP = 64  # v1 fixed; becomes the value-specialized `block` bracket with the schema surface


def _uniform_dest(path, leaf, offset):
    if not isinstance(leaf, ScalarLeaf):
        return None, offset  # buffer leaves ride the leaves channel (ndarrays, ch12)
    fmt = _WFMT[_WTYPE[leaf.kind]]
    return PackedDest(offset, fmt), offset + struct.calcsize(fmt)


def _plan(env_types, arg_types, table) -> PackPlan:
    return plan_from_types(env_types, (), table, _uniform_dest)  # args = thread coords: never packed


def _param_types(target) -> tuple:
    """Compute-family contract v1: params are thread coordinates, f64 at the
    language level (the backend narrows). Launch rank must equal param count."""
    pyfunc = getattr(target, "pyfunc", None)
    if pyfunc is None:  # a single-stage Pipeline can reach here via `value > stage`
        raise VerifyError(
            "the wgsl backend launches kernels, not pipelines — fused compute "
            "pipelines arrive with the orchestrate encode plan (070 §3)"
        )
    return (f64,) * pyfunc.__code__.co_argcount


# --- the renderer ----------------------------------------------------------------


_MEMBER = {"<f": "f32", "<i": "i32", "<I": "u32"}  # slot fmt -> uniform member type


def render(region: Region, plan: PackPlan, backend=None, name: str = "kernel") -> str:
    """Legalized Region -> a WGSL `fn kernel_body(p0: f32, …) -> f32` plus the
    Env uniform struct (omitted when the kernel captures nothing). Member
    types come from the slot FORMAT — an i64 capture is an i32 member, and a
    bool capture is a u32 member (bool is not host-shareable in WGSL; the
    abi.slot read compares against 0u). Same dominator-placed lazy-branch
    emission as the Python backend."""
    # Members come from the PLAN, not from surviving abi.slots: WGSL lays
    # struct members out sequentially, so a capture that folded away would
    # otherwise leave a hole and silently shift every later member's bytes
    # (review-caught). The plan is dense by construction; unused members are
    # harmless and keep offsets literal.
    lines, names, result = _emit_wgsl(region, backend.code_for_op if backend is not None else CODE_FOR_OP)
    header = ""
    if plan.slots:
        members = ",\n".join(f"  m{s.dest.offset}: {_MEMBER[s.dest.fmt]}" for s in plan.slots)
        header = f"struct Env {{\n{members}\n}}\n@group(0) @binding(0) var<uniform> env: Env;\n"
    params = ", ".join(f"p{i}: f32" for i in range(len(region.params)))
    ret = names[id(result)]
    return header + f"fn kernel_body({params}) -> f32 {{\n" + "\n".join(lines) + f"\n  return {ret};\n}}\n"


def _emit_wgsl(region: Region, spelling_table=None) -> tuple[list[str], dict, object]:
    """WGSL spelling over the shared dominator-placed walker (``_emit``)."""

    def expr_of(node: Node, names: dict) -> str:
        attrs = dict(node.attrs)
        arg = [names[id(a)] for a in node.args]
        if node.op == "core.param":
            return f"p{attrs['index']}"
        if node.op == "abi.slot":
            if isinstance(node.type, Scalar) and node.type.kind == "bool":
                return f"(env.m{attrs['offset']} != 0u)"  # bool travels as u32 (not host-shareable)
            return f"env.m{attrs['offset']}"
        if node.op == "core.const":
            v = attrs["value"]
            if isinstance(v, bool):
                return "true" if v else "false"
            if isinstance(v, float):
                if v != v or v in (float("inf"), float("-inf")):  # WGSL has no non-finite
                    raise VerifyError("WGSL has no inf/nan literals (and may assume them absent — R15)")
                r = repr(v)
                return r if ("." in r or "e" in r) else r + ".0"
            if not -(2**31) <= v < 2**31:  # narrowing i64->i32 must not silently overflow
                raise VerifyError(f"int constant {v} does not fit i32 (WGSL has no i64)")
            return str(v)
        if node.op == "core.pow":
            return f"pow({arg[0]}, {arg[1]})"
        if node.op in _BIN:
            return f"{arg[0]} {_BIN[node.op]} {arg[1]}"
        if node.op == "core.neg":
            return f"-{arg[0]}"
        if node.op == "core.cmp":
            return f"{arg[0]} {_PREDS[attrs['pred']]} {arg[1]}"
        if node.op == "core.cast":
            to = attrs["to"]
            if not isinstance(to, Scalar):
                raise VerifyError(f"wgsl backend cannot cast to {to!r}")
            return f"{_WTYPE[to.kind]}({arg[0]})"
        if node.op == "core.select":
            return f"select({arg[2]}, {arg[1]}, {arg[0]})"  # select(false_val, true_val, cond)
        tbl = spelling_table if spelling_table is not None else CODE_FOR_OP
        template = tbl.get(node.op)
        if template:
            return template.format(*arg)
        if template is None and node.op in tbl:  # spell(None) claims native support this renderer lacks
            raise VerifyError(f"{node.op!r} was spelled None ('native') but this renderer has no native handling")
        raise VerifyError(f"wgsl backend has no rendering for {node.op!r}")

    def wtype(node: Node) -> str:
        return _WTYPE[node.type.kind] if isinstance(node.type, Scalar) else "f32"

    def statement(node, nm):
        return f"let {nm[id(node)]}: {wtype(node)} = {expr_of(node, nm)};"

    def branch_join(node, nm, result_of, emit_block, path, ind):
        res = nm[id(node)]
        out = [f"{ind}var {res}: {wtype(node)};", f"{ind}if ({nm[id(node.args[0])]}) {{"]
        out += emit_block((*path, (id(node), 0)), ind + "  ")
        out.append(f"{ind}  {res} = {result_of(0)};")
        out.append(f"{ind}}} else {{")
        out += emit_block((*path, (id(node), 1)), ind + "  ")
        out.append(f"{ind}  {res} = {result_of(1)};")
        out.append(f"{ind}}}")
        return out

    return emit_dominated(region, statement, branch_join, indent="  ")


# --- the runtime -------------------------------------------------------------------

_DEVICE = None


def device():
    global _DEVICE
    if _DEVICE is None:
        import wgpu

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        wanted = [f for f in ("timestamp-query",) if f in [str(x) for x in adapter.features]]
        _DEVICE = adapter.request_device_sync(required_features=wanted)  # bench's GPU clock (ch11b)
    return _DEVICE


_PROBED: bool | None = None


def is_available() -> bool:
    global _PROBED
    if _PROBED is None:
        try:
            _PROBED = device() is not None
        except Exception:
            _PROBED = False  # cached: repeated probes must not re-pay adapter requests
    return _PROBED


def _domain(leaves, expect_rank: int) -> tuple:
    from ...kernel.registry import Out

    tagged = [x for x in leaves if isinstance(x, Out)]  # by TYPE, never tail position
    if not tagged:
        raise VerifyError("this kernel needs a launch domain: k(out=(W, H)) or k(out=N)")
    dom = tagged[-1].value
    dom = (dom,) if isinstance(dom, int) else tuple(dom)
    if len(dom) != expect_rank:
        raise VerifyError(f"kernel has {expect_rank} thread-coordinate params; out has rank {len(dom)}")
    return dom


class ComputeProgram:
    """The compute artifact: shader module + pipeline + uniform buffers, built
    once at the miss. Per call: write_buffer(staging) → dispatch(grid) →
    read back f32s. Out buffer reused while the domain holds."""

    def __init__(self, body: str, nparams: int, staging_size: int):
        import wgpu

        if nparams not in (1, 2):
            raise VerifyError("compute v1 supports 1D and 2D launch domains")
        self.wgpu, self.nparams, self.has_env = wgpu, nparams, staging_size > 0
        self.wg = WORKGROUP  # captured HERE: baked source and dispatch math cannot desync
        gids = ["f32(gid.x)", "f32(gid.y)"][:nparams]
        wg = f"{self.wg}" if nparams == 1 else "8, 8"
        self.source = (
            body
            + "@group(0) @binding(1) var<storage, read_write> out: array<f32>;\n"
            + "@group(0) @binding(2) var<uniform> dims: vec4<u32>;\n"
            + f"@compute @workgroup_size({wg})\n"
            + "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
            + "  if (gid.x >= dims.x || gid.y >= dims.y) { return; }\n"
            + "  let idx = gid.y * dims.x + gid.x;\n"
            + f"  out[idx] = kernel_body({', '.join(gids)});\n"
            + "}\n"
        )
        dev = device()
        self.pipeline = dev.create_compute_pipeline(
            layout="auto",
            compute={"module": dev.create_shader_module(code=self.source), "entry_point": "main"},
        )
        self.ubuf = (
            dev.create_buffer(size=staging_size, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            if self.has_env
            else None
        )
        self.dbuf = dev.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._out = self._query = None

    def _encode_frame(self, staging, w, h, timestamp_writes=None):
        """The per-frame encode sequence, shared by ``__call__`` and
        ``timed_call`` so the timed frame can never drift from the real one."""
        dev = device()
        if self.has_env:
            dev.queue.write_buffer(self.ubuf, 0, bytes(staging))
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass(**({"timestamp_writes": timestamp_writes} if timestamp_writes else {}))
        cp.set_pipeline(self.pipeline)
        cp.set_bind_group(0, self._out[2])
        if self.nparams == 1:
            cp.dispatch_workgroups((w + self.wg - 1) // self.wg)
        else:
            cp.dispatch_workgroups((w + 7) // 8, (h + 7) // 8)
        cp.end()
        return enc

    def __call__(self, staging, leaves):
        dom = _domain(leaves, self.nparams)
        w, h = dom[0], (dom[1] if len(dom) > 1 else 1)
        count = w * h
        dev, wgpu = device(), self.wgpu
        if self._out is None or self._out[1] != (w, h):
            # Domain changed: out buffer, dims, AND the bind group rebuild here —
            # bind groups are (pipeline, buffer-set)-tier state, never per-frame
            # (R12's cost ranking; the per-frame tier is write_buffer + encode).
            buf = dev.create_buffer(size=count * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
            dev.queue.write_buffer(self.dbuf, 0, struct.pack("<4I", w, h, 1, 0))
            entries = [
                {"binding": 1, "resource": {"buffer": buf, "offset": 0, "size": count * 4}},
                {"binding": 2, "resource": {"buffer": self.dbuf, "offset": 0, "size": 16}},
            ]
            if self.has_env:
                entries.insert(
                    0, {"binding": 0, "resource": {"buffer": self.ubuf, "offset": 0, "size": self.ubuf.size}}
                )
            bg = dev.create_bind_group(layout=self.pipeline.get_bind_group_layout(0), entries=entries)
            self._out = (buf, (w, h), bg)
        dev.queue.submit([self._encode_frame(staging, w, h).finish()])
        data = dev.queue.read_buffer(self._out[0], 0, size=count * 4)
        return struct.unpack(f"<{count}f", data)

    def timed_call(self, staging, leaves):
        """One frame, decomposed (bench.gpu_timeline): encode+submit host time,
        GPU execution from begin/end-of-pass timestamps (nanoseconds, per the
        WebGPU spec), and the blocking readback. None without the feature."""
        import time

        dev, wgpu = device(), self.wgpu
        if "timestamp-query" not in [str(f) for f in dev.features]:
            return None
        dom = _domain(leaves, self.nparams)
        w, h = dom[0], (dom[1] if len(dom) > 1 else 1)
        self(staging, leaves)  # ensure buffers/bind group exist for this domain
        if self._query is None:
            qs = dev.create_query_set(type=wgpu.QueryType.timestamp, count=2)
            qbuf = dev.create_buffer(size=16, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
            self._query = (qs, qbuf)
        qs, qbuf = self._query
        writes = {"query_set": qs, "beginning_of_pass_write_index": 0, "end_of_pass_write_index": 1}
        t0 = time.perf_counter()
        enc = self._encode_frame(staging, w, h, timestamp_writes=writes)
        enc.resolve_query_set(qs, 0, 2, qbuf, 0)
        dev.queue.submit([enc.finish()])
        t1 = time.perf_counter()
        count = w * h
        data = dev.queue.read_buffer(self._out[0], 0, size=count * 4)
        assert len(data) == count * 4
        t2 = time.perf_counter()
        ticks = struct.unpack("<2Q", dev.queue.read_buffer(qbuf))
        # some drivers report non-monotonic begin/end pass timestamps; never go negative
        return {"encode+submit": t1 - t0, "gpu": max(0, ticks[1] - ticks[0]) / 1e9, "readback": t2 - t1}


class FragmentProgram:
    """The fragment artifact: fullscreen-triangle pipeline + offscreen target.
    The body runs per pixel with (x, y) = frag position; the scalar result
    broadcasts to grayscale rgba. Returns rows of floats in [0, 1]."""

    FORMAT = "rgba8unorm"

    def __init__(self, body: str, nparams: int, staging_size: int):
        import wgpu

        if nparams != 2:
            raise VerifyError("fragment kernels take exactly (x, y) pixel-coordinate params")
        self.wgpu, self.has_env = wgpu, staging_size > 0
        self.source = (
            body
            + "@vertex\nfn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {\n"
            + "  var p = array<vec2<f32>, 3>(vec2f(-1.0, -3.0), vec2f(3.0, 1.0), vec2f(-1.0, 1.0));\n"
            + "  return vec4f(p[i], 0.0, 1.0);\n}\n"
            + "@fragment\nfn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {\n"
            + "  let v = kernel_body(pos.x, pos.y);\n"
            + "  return vec4f(v, v, v, 1.0);\n}\n"
        )
        dev = device()
        module = dev.create_shader_module(code=self.source)
        self.pipeline = dev.create_render_pipeline(
            layout="auto",
            vertex={"module": module, "entry_point": "vs_main"},
            fragment={"module": module, "entry_point": "fs_main", "targets": [{"format": self.FORMAT}]},
            primitive={"topology": "triangle-list"},
        )
        self.ubuf = (
            dev.create_buffer(size=staging_size, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            if self.has_env
            else None
        )
        self._bg = (  # env-only bind group: fully static, artifact-tier (R12)
            dev.create_bind_group(
                layout=self.pipeline.get_bind_group_layout(0),
                entries=[{"binding": 0, "resource": {"buffer": self.ubuf, "offset": 0, "size": self.ubuf.size}}],
            )
            if self.has_env
            else None
        )
        self._target = None

    def __call__(self, staging, leaves):
        w, h = _domain(leaves, 2)
        dev, wgpu = device(), self.wgpu
        if self._target is None or self._target[2:] != (w, h):
            tex = dev.create_texture(
                size=(w, h, 1),
                format=self.FORMAT,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
            )
            self._target = (tex, tex.create_view(), w, h)
        if self.has_env:
            dev.queue.write_buffer(self.ubuf, 0, bytes(staging))
        enc = dev.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[
                {
                    "view": self._target[1],
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0, 0, 0, 1),
                }
            ]
        )
        rp.set_pipeline(self.pipeline)
        if self._bg is not None:
            rp.set_bind_group(0, self._bg)
        rp.draw(3)
        rp.end()
        dev.queue.submit([enc.finish()])
        row = (w * 4 + 255) // 256 * 256
        data = dev.queue.read_texture({"texture": self._target[0]}, {"bytes_per_row": row}, (w, h, 1))
        px = bytes(data)
        scale = 1.0 / 255.0  # rows-of-floats is the v1 boundary contract; per-row R-channel slices
        return [[b * scale for b in px[j * row : j * row + w * 4 : 4]] for j in range(h)]


# --- the Backend records -------------------------------------------------------------


def _render_with_meta(region, plan, backend=None, name="kernel"):
    # nparams + staging size ride a header comment: deterministic text, so the
    # content-addressed artifact tier stays honest, and compile() needs no side channel.
    return f"// nparams={len(region.params)} staging={plan.staging_size}\n" + render(region, plan, backend)


def _compile_factory(program_cls):
    def compile_source(source: str, name: str = "kernel"):
        head, body = source.split("\n", 1)
        meta = dict(kv.split("=") for kv in head.removeprefix("// ").split())
        program = program_cls(body, int(meta["nparams"]), int(meta["staging"]))
        program.__pdum_source__ = source
        return program

    return compile_source


CODE_FOR_OP: dict = {}  # ONE table for both wgsl cells (same spelling language)

COMPUTE = Backend(
    name="demo.simple_shader.wgsl.compute",
    render=_render_with_meta,
    compile=_compile_factory(ComputeProgram),
    fp=("demo.simple_shader.wgsl", "compute", 1, WORKGROUP),
    plan=_plan,
    param_types=_param_types,
    code_for_op=CODE_FOR_OP,
)
FRAGMENT = Backend(
    name="demo.simple_shader.wgsl.fragment",
    render=_render_with_meta,
    compile=_compile_factory(FragmentProgram),
    fp=("demo.simple_shader.wgsl", "fragment", 1, FragmentProgram.FORMAT),
    plan=_plan,
    param_types=_param_types,
    code_for_op=CODE_FOR_OP,
)


def install(registry) -> None:
    """Role vocabularies ship WITH their backends (the ch04 rule): 'compute'
    and 'fragment' arrive here, routed here; 'device' stays on python."""
    # Demo-scoped KIND names: "compute" and "fragment" are reserved for the
    # real families (step 14, 080 §3) — a demo must never squat on a name a
    # future package will use with richer semantics (ch10 walkthrough rule).
    from dataclasses import replace

    register_role("simple_shader.compute", hint="demo compute kernel: params are thread coordinates")
    register_role("simple_shader.fragment", hint="demo fragment kernel: params are pixel coordinates")
    # Per-registry record copies with fresh spelling tables (shared singletons
    # would let one registry's spell() rewrite another's — review-caught):
    registry.register_backend(replace(COMPUTE, code_for_op=dict(COMPUTE.code_for_op)), kinds=("simple_shader.compute",))
    registry.register_backend(
        replace(FRAGMENT, code_for_op=dict(FRAGMENT.code_for_op)), kinds=("simple_shader.fragment",)
    )


install(DEFAULT)
