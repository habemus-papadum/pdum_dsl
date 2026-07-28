"""The Metal half: the MSL shell around the shared rows, and PyObjC Metal.

spike_metal proved the COMPUTE rows port between WGSL and MSL under three
lexical rules. This file is the render-stage extension of that result, and
the honest summary is: the rows still port for free (``program.py``
generates both languages from one walker with no per-target branch), and
the SHELL needed four structural decisions that have no WGSL counterpart.

  S1. TWO BINDING INDEX SPACES. WGSL puts every resource in one bind
      group with one index space shared by both stages. Metal has no
      bind-group object at all: buffers are entry-point PARAMETERS, and
      ``setVertexBuffer:atIndex:`` and ``setFragmentBuffer:atIndex:``
      index into two separate tables. So the shared binding plan (which
      is correct for WGSL as-is) has to be RE-INDEXED per stage here --
      the geometry and the vertex slot buffer keep their numbers, the
      fragment slot buffer moves from binding 3 to buffer 0. This is
      spike_metal FAIL-3 (the binding table is co-owned) showing up in
      the render stage, where it is worse than in compute: the same
      logical resource has two different indices depending on who reads
      it.

  S2. THE VARYING STRUCT IS THE INTERFACE, and its attributes differ in
      kind. WGSL tags each member with ``@location(i)`` and interpolation
      with ``@interpolate(flat)``. MSL matches the vertex output struct
      to the fragment's ``[[stage_in]]`` parameter by DECLARATION ORDER
      (no location numbers), and spells flat as ``[[flat]]`` on the
      member. Same information, different carrier.

  S3. THE STAGE IS A RETURN-TYPE QUALIFIER, not an attribute. WGSL says
      ``@vertex fn vs_main(...) -> VOut``; MSL says ``vertex VOut
      vs_main(...)``. And the fragment's color attachment is implicit in
      MSL's return type for a single target where WGSL writes
      ``@location(0)``.

  S4. VERTEX PULLING NEEDS NO VERTEX DESCRIPTOR ON EITHER TARGET, which
      is the pleasant surprise. ``MTLRenderPipelineDescriptor.vertexDescriptor``
      stays nil, exactly as wgpu's ``vertex={"buffers": []}`` stays empty:
      an attribute is a storage-buffer read at ``[[vertex_id]]``, and
      because the read goes through ``index_expr`` it reads the record
      buffer in place at stride 2 f32 -- ``b0[i32(vid) * 2 + 1]`` in
      WGSL is ``b0[int(vid) * 2 + 1]`` in MSL and nothing else changes.

Everything else -- the whole expression body of both stages -- came out
of the shared generator unmodified. The three lexical rules held.

Runtime notes. ``newLibraryWithSource:`` is Metal's in-process compiler,
so no Xcode and no offline .metallib. Unified memory shows up in the
per-frame path: a slot buffer's bytes are written straight through
``contents()`` with no queue call and no staging -- the closest thing
this codebase has to "the uniform IS host memory". We
``waitUntilCompleted`` every frame, which makes that safe without a
fence; a real presenter would use an in-flight semaphore instead.
"""

from __future__ import annotations

import Metal
import numpy as np
import program as P

PIPELINES: list[str] = []

_PIXEL = {
    "r32float": Metal.MTLPixelFormatR32Float,
    "bgra8unorm": Metal.MTLPixelFormatBGRA8Unorm,
    "bgra8unorm-srgb": Metal.MTLPixelFormatBGRA8Unorm_sRGB,
}

# target format -> (MSL return type, the expression from the scalar color0)
_SURFACE = {
    "r32float": ("float", "{c}"),
    "bgra8unorm": ("float4", "float4({c}, {c}, {c}, 1.0f)"),
    "bgra8unorm-srgb": ("float4", "float4({c}, {c}, {c}, 1.0f)"),
}

_PRELUDE = ["#include <metal_stdlib>", "using namespace metal;"]

FRAGMENT_SLOT_INDEX = 0  # S1: the fragment stage's buffer table is its own


# --- the shells --------------------------------------------------------------


def compute_source(rows: P.ComputeRows) -> str:
    d = P.MSL
    params = [f"    device {d.fty}* b{b.binding} [[buffer({b.binding})]]," for b in rows.buffers]
    if rows.slot_binding is not None:
        params.append(f"    const device {d.fty}* U [[buffer({rows.slot_binding})]],")
    n = len(rows.axes)
    comp = dict(zip(rows.axes, ("y", "x") if n == 2 else ("x",)))
    # We keep the WebGPU-shaped guard and dispatch whole threadgroups even
    # though Metal's dispatchThreads: would make it dead code. That is a
    # deliberate choice for this demo: with the guard in, the two compute
    # sources are line-for-line the same program and the cross-backend
    # diff measures the LANGUAGES, not two different dispatch strategies.
    guard = " || ".join(f"gid.{comp[a]} >= {rows.extents[a][1] - rows.extents[a][0]}u" for a in rows.axes)
    return "\n".join(
        [
            *_PRELUDE,
            "kernel void main0(",
            *params,
            "    uint3 gid [[thread_position_in_grid]])",
            "{",
            f"  if ({guard}) {{ return; }}",
            *rows.lines,
            *rows.stores,
            "}",
        ]
    )


def render_source(low: P.Lowered, rows: P.RenderRows, target_format: str) -> str:
    """S1-S4 in one function. The rows come in already generated."""
    d = P.MSL
    if target_format not in _SURFACE:
        raise P.Untranslatable(f"no surface expansion for target format {target_format!r}")
    ret_ty, ret_expr = _SURFACE[target_format]

    # S2: order IS the location; flat is a member attribute.
    members = [
        f"  {d.fty} f{i}{' [[flat]]' if n in low.flats else ''};" for i, n in enumerate(low.varyings)
    ]
    lines = [*_PRELUDE, "struct VOut {", "  float4 pos [[position]];", *members, "};"]

    # S1: the vertex stage's own buffer table. Geometry and VU keep the
    # shared plan's indices.
    v_params = [f"    const device {d.fty}* b{b.binding} [[buffer({b.binding})]]," for b in rows.buffers]
    if rows.v_slot_binding is not None:
        v_params.append(f"    const device {d.fty}* VU [[buffer({rows.v_slot_binding})]],")
    lines += [
        "vertex VOut vs_main(",  # S3: the stage is a return-type qualifier
        *v_params,
        "    uint vid [[vertex_id]])",
        "{",
        *rows.v_lines,
        "  VOut o;",
        f"  o.pos = float4({rows.v_outs[0]}, {rows.v_outs[1]}, 0.0f, 1.0f);",
        *[f"  o.f{i} = {v};" for i, v in enumerate(rows.v_outs[2:])],
        "  return o;",
        "}",
    ]

    # S1 again: the FRAGMENT's table starts over at 0, so the same slot
    # buffer that is binding 3 in WGSL is buffer 0 here.
    f_params = [f"    const device {d.fty}* U [[buffer({FRAGMENT_SLOT_INDEX})]],"] if low.f_slots else []
    lines += [
        f"fragment {ret_ty} fs_main(",
        *f_params,
        "    VOut vin [[stage_in]])",
        "{",
        *rows.f_lines,
        f"  return {ret_expr.format(c=rows.f_out)};",
        "}",
    ]
    return "\n".join(lines)


# --- the runtime -------------------------------------------------------------


class MetalRuntime:
    """Device + queue + the library cache. Constructible with options and
    holding its own caches, which is what spike_runner's H5 said the wgpu
    singleton refuses to be."""

    def __init__(self):
        dev = Metal.MTLCreateSystemDefaultDevice()
        if dev is None:
            raise RuntimeError("no Metal device on this machine")
        self.device = dev
        self.queue = dev.newCommandQueue()

    def library(self, source: str):
        lib, err = self.device.newLibraryWithSource_options_error_(source, None, None)
        if lib is None:
            raise RuntimeError(f"MSL compile failed: {err}\n--- source ---\n{source}")
        return lib

    def compute_pipeline(self, source: str, entry: str = "main0"):
        fn = self.library(source).newFunctionWithName_(entry)
        pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if pso is None:
            raise RuntimeError(f"compute pipeline creation failed: {err}")
        return pso

    def render_pipeline(self, source: str, target_format: str):
        lib = self.library(source)
        desc = Metal.MTLRenderPipelineDescriptor.alloc().init()
        desc.setVertexFunction_(lib.newFunctionWithName_("vs_main"))
        desc.setFragmentFunction_(lib.newFunctionWithName_("fs_main"))
        # S4: vertexDescriptor stays nil -- vertex pulling needs no layout.
        desc.colorAttachments().objectAtIndexedSubscript_(0).setPixelFormat_(_PIXEL[target_format])
        pso, err = self.device.newRenderPipelineStateWithDescriptor_error_(desc, None)
        if pso is None:
            raise RuntimeError(f"render pipeline creation failed: {err}")
        return pso

    def buffer_from(self, arr: np.ndarray):
        a = np.ascontiguousarray(arr)
        return self.device.newBufferWithBytes_length_options_(
            a.tobytes(), a.nbytes, Metal.MTLResourceStorageModeShared
        )

    def buffer_empty(self, nbytes: int):
        return self.device.newBufferWithLength_options_(max(nbytes, 4), Metal.MTLResourceStorageModeShared)


_RT: list[MetalRuntime] = []


def runtime() -> MetalRuntime:
    if not _RT:
        _RT.append(MetalRuntime())
    return _RT[0]


class _SlotBuffer:
    """A uniform buffer on unified memory: the refresh is a memcpy into
    memory the GPU already sees. No queue call, no staging buffer, no
    upload -- contrast wgpu's ``queue.write_buffer``."""

    def __init__(self, rt: MetalRuntime, slots, index: int):
        self.slots, self.index = tuple(slots), index
        self.nbytes = max(4 * len(self.slots), 4)
        self.gpu = rt.buffer_empty(self.nbytes)
        self.host = np.frombuffer(self.gpu.contents().as_buffer(self.nbytes), dtype=np.float32)
        self.writes = 0

    def refresh(self, staging: bytes) -> int:
        vals = P.slot_values(staging, self.slots)
        self.host[: vals.size] = vals
        self.writes += 1
        return 4 * len(self.slots)


class Engine:
    """One program, compiled once for one target format. Same shape as the
    WebGPU Engine: build, then update/encode."""

    backend_name = "metal"

    def __init__(self, low: P.Lowered, target_format: str, rt: MetalRuntime | None = None):
        self.low = low
        self.rt = rt or runtime()
        self.target_format = target_format
        self.res = P.Residency(self.rt.buffer_from)

        # ---- compute ----
        self.c_rows = P.compute_rows(low, self.res, P.MSL)
        self.c_source = compute_source(self.c_rows)
        PIPELINES.append("msl:compute")
        self.c_pipeline = self.rt.compute_pipeline(self.c_source)
        self.c_slots = (
            _SlotBuffer(self.rt, self.c_rows.slots, self.c_rows.slot_binding)
            if self.c_rows.slot_binding is not None
            else None
        )
        n = len(self.c_rows.axes)
        self.c_tg = Metal.MTLSizeMake(*((8, 8, 1) if n == 2 else (64, 1, 1)))
        t = self.c_rows.threads
        tg = (8, 8, 1) if n == 2 else (64, 1, 1)
        self.c_groups = Metal.MTLSizeMake(*[(t[i] + tg[i] - 1) // tg[i] for i in range(3)])

        # ---- render ----
        self.r_rows = P.render_rows(low, self.res, P.MSL)
        self.r_source = render_source(low, self.r_rows, target_format)
        PIPELINES.append(f"msl:render:{target_format}")
        self.r_pipeline = self.rt.render_pipeline(self.r_source, target_format)
        self.v_slots = (
            _SlotBuffer(self.rt, low.v_slots, self.r_rows.v_slot_binding)
            if self.r_rows.v_slot_binding is not None
            else None
        )
        self.f_slots = _SlotBuffer(self.rt, low.f_slots, FRAGMENT_SLOT_INDEX) if low.f_slots else None
        self.slot_bytes = 0

    def offscreen(self, size):
        return Offscreen(self.rt, size)

    # -- per frame --------------------------------------------------------
    def update(self, vs_fn) -> int:
        P.check_swap(self.low.vs_fn, vs_fn)
        n = 0
        if self.c_slots is not None:
            n += self.c_slots.refresh(P.kernel_staging(self.low.art))
        if self.v_slots is not None:
            n += self.v_slots.refresh(P.vertex_staging(self.low.v_ctx, vs_fn))
        if self.f_slots is not None:
            n += self.f_slots.refresh(P.fragment_staging(self.low))
        self.slot_bytes = n
        return n

    def encode(self, cmd, color_texture, clear=(0.0, 0.0, 0.0, 1.0)) -> None:
        """THE shared encode path: offscreen hands in its own texture,
        the window hands in a drawable's."""
        ce = cmd.computeCommandEncoder()
        ce.setComputePipelineState_(self.c_pipeline)
        for b in self.c_rows.buffers:
            ce.setBuffer_offset_atIndex_(b.resident.gpu, 0, b.binding)
        if self.c_slots is not None:
            ce.setBuffer_offset_atIndex_(self.c_slots.gpu, 0, self.c_slots.index)
        ce.dispatchThreadgroups_threadsPerThreadgroup_(self.c_groups, self.c_tg)
        ce.endEncoding()

        rpd = Metal.MTLRenderPassDescriptor.renderPassDescriptor()
        ca = rpd.colorAttachments().objectAtIndexedSubscript_(0)
        ca.setTexture_(color_texture)
        ca.setLoadAction_(Metal.MTLLoadActionClear)
        ca.setStoreAction_(Metal.MTLStoreActionStore)
        ca.setClearColor_(Metal.MTLClearColorMake(*clear))
        re = cmd.renderCommandEncoderWithDescriptor_(rpd)
        re.setRenderPipelineState_(self.r_pipeline)
        for b in self.r_rows.buffers:  # S1: the VERTEX table
            re.setVertexBuffer_offset_atIndex_(b.resident.gpu, 0, b.binding)
        if self.v_slots is not None:
            re.setVertexBuffer_offset_atIndex_(self.v_slots.gpu, 0, self.v_slots.index)
        if self.f_slots is not None:  # S1: the FRAGMENT table, its own indices
            re.setFragmentBuffer_offset_atIndex_(self.f_slots.gpu, 0, self.f_slots.index)
        re.drawPrimitives_vertexStart_vertexCount_(Metal.MTLPrimitiveTypeTriangle, 0, self.low.draw_count)
        re.endEncoding()


class Offscreen:
    """An r32float texture + a readback buffer, created ONCE. The blit
    mirrors wgpu's ``copy_texture_to_buffer`` so the two backends'
    readback paths are like-for-like."""

    def __init__(self, rt: MetalRuntime, size):
        self.rt = rt
        self.H, self.W = size
        desc = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
            Metal.MTLPixelFormatR32Float, self.W, self.H, False
        )
        desc.setUsage_(Metal.MTLTextureUsageRenderTarget | Metal.MTLTextureUsageShaderRead)
        desc.setStorageMode_(Metal.MTLStorageModeShared)
        self.tex = rt.device.newTextureWithDescriptor_(desc)
        self.bpr = self.W * 4
        self.buf = rt.buffer_empty(self.bpr * self.H)

    def draw(self, engine: Engine, clear=(0.0, 0.0, 0.0, 1.0)) -> np.ndarray:
        cmd = self.rt.queue.commandBuffer()
        engine.encode(cmd, self.tex, clear=clear)
        bl = cmd.blitCommandEncoder()
        bl.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_(
            self.tex,
            0,
            0,
            Metal.MTLOriginMake(0, 0, 0),
            Metal.MTLSizeMake(self.W, self.H, 1),
            self.buf,
            0,
            self.bpr,
            self.bpr * self.H,
        )
        bl.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        raw = np.frombuffer(self.buf.contents().as_buffer(self.bpr * self.H), dtype=np.float32)
        return np.flipud(raw.reshape(self.H, self.W)).astype(np.float64)


def run_window(low: P.Lowered, mouse, on_frame, width=900, height=600, title="pdum"):
    from metal_window import run  # AppKit lives in its own file

    run(low, mouse, on_frame, width=width, height=height, title=title)
