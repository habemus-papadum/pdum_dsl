"""The WebGPU half: the WGSL shell around the shared rows, and wgpu.

Two things live here and nothing else does:

  THE SHELL -- how WGSL declares buffers (module-scope
  ``@group/@binding`` + ``var<storage, ...>``), what an entry point looks
  like (``@compute @workgroup_size(...)`` / ``@vertex`` / ``@fragment``),
  how a vertex-to-fragment struct is spelled (``@builtin(position)`` +
  ``@location(i)``), and how the fragment's scalar color0 expands to a
  surface's four channels.

  THE RUNTIME -- device, buffers, shader modules, pipelines, bind
  groups, encode, submit, readback.

The COMPILE-ONCE / BIND / ENCODE split is spike_runner's, unchanged and
for the same reason: lowering plus WGSL text measured 0.937 ms, which is
seven times the entire warm frame. ``Engine.__init__`` does all of it;
``update()`` writes only the slot bytes; ``encode()`` touches only a
command encoder.

Note ``TARGET_FORMAT keys the pipeline but RESOLUTION does not``. The
fragment's color0 is a scalar in the lowering, so the 1-to-4 channel
surface expansion is baked into the generated text -- an offscreen
r32float pipeline and a windowed bgra8unorm-srgb pipeline are genuinely
two compiled artifacts of one program. Resolution enters nowhere, which
is why the same Engine renders 64x96 and 900x600.
"""

from __future__ import annotations

import numpy as np
import program as P

PIPELINES: list[str] = []  # every pipeline creation appends; the demo asserts the count

# target format -> (WGSL return type, the expression from the scalar color0)
_SURFACE = {
    "r32float": ("f32", "{c}"),
    "bgra8unorm": ("vec4<f32>", "vec4<f32>({c}, {c}, {c}, 1.0)"),
    "rgba8unorm": ("vec4<f32>", "vec4<f32>({c}, {c}, {c}, 1.0)"),
    "bgra8unorm-srgb": ("vec4<f32>", "vec4<f32>({c}, {c}, {c}, 1.0)"),
    "rgba8unorm-srgb": ("vec4<f32>", "vec4<f32>({c}, {c}, {c}, 1.0)"),
}


def make_device():
    import wgpu

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    return adapter.request_device_sync()


# --- the shells --------------------------------------------------------------


def compute_source(rows: P.ComputeRows) -> str:
    d = P.WGSL
    lines = [
        f"@group(0) @binding({b.binding}) var<storage, read_write> b{b.binding}: array<{d.fty}>;"
        for b in rows.buffers
    ]
    if rows.slot_binding is not None:
        lines.append(f"@group(0) @binding({rows.slot_binding}) var<storage, read> U: array<{d.fty}>;")
    n = len(rows.axes)
    wg = (8, 8, 1) if n == 2 else (64, 1, 1)
    comp = dict(zip(rows.axes, ("y", "x") if n == 2 else ("x",)))
    # The bounds guard exists because WebGPU can only dispatch WHOLE
    # workgroups, so the last one overhangs. It is emitted SOURCE whose
    # necessity is a runtime property (spike_metal FAIL-2).
    guard = " || ".join(f"gid.{comp[a]} >= {rows.extents[a][1] - rows.extents[a][0]}u" for a in rows.axes)
    return "\n".join(
        [
            *lines,
            f"@compute @workgroup_size({wg[0]}, {wg[1]}, {wg[2]})",
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
            f"  if ({guard}) {{ return; }}",
            *rows.lines,
            *rows.stores,
            "}",
        ]
    )


def render_source(low: P.Lowered, rows: P.RenderRows, target_format: str) -> str:
    d = P.WGSL
    if target_format not in _SURFACE:
        raise P.Untranslatable(f"no surface expansion for target format {target_format!r}")
    ret_ty, ret_expr = _SURFACE[target_format]
    fields = [
        f"  @location({i}){' @interpolate(flat)' if n in low.flats else ''} f{i}: {d.fty},"
        for i, n in enumerate(low.varyings)
    ]
    lines = ["struct VOut {", f"  @builtin(position) pos: vec4<{d.fty}>,", *fields, "}"]
    # ONE bind group serves both stages: WGSL has a single binding index
    # space across the whole pipeline.
    for b in rows.buffers:
        lines.append(f"@group(0) @binding({b.binding}) var<storage, read> b{b.binding}: array<{d.fty}>;")
    if rows.v_slot_binding is not None:
        lines.append(f"@group(0) @binding({rows.v_slot_binding}) var<storage, read> VU: array<{d.fty}>;")
    if rows.f_slot_binding is not None:
        lines.append(f"@group(0) @binding({rows.f_slot_binding}) var<storage, read> U: array<{d.fty}>;")
    lines += [
        "@vertex",
        "fn vs_main(@builtin(vertex_index) vid: u32) -> VOut {",
        *rows.v_lines,
        "  var o: VOut;",
        f"  o.pos = vec4<{d.fty}>({rows.v_outs[0]}, {rows.v_outs[1]}, 0.0, 1.0);",
        *[f"  o.f{i} = {v};" for i, v in enumerate(rows.v_outs[2:])],
        "  return o;",
        "}",
        "@fragment",
        f"fn fs_main(vin: VOut) -> @location(0) {ret_ty} {{",
        *rows.f_lines,
        f"  return {ret_expr.format(c=rows.f_out)};",
        "}",
    ]
    return "\n".join(lines)


# --- the runtime -------------------------------------------------------------


class _SlotBuffer:
    def __init__(self, device, slots, binding):
        import wgpu

        self.device, self.slots, self.binding = device, tuple(slots), binding
        self.gpu = device.create_buffer(
            size=max(4 * len(slots), 4), usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        )
        self.writes = 0

    def refresh(self, staging: bytes) -> int:
        self.device.queue.write_buffer(self.gpu, 0, P.slot_values(staging, self.slots).tobytes())
        self.writes += 1
        return 4 * len(self.slots)

    def entry(self) -> dict:
        return {"binding": self.binding, "resource": {"buffer": self.gpu, "offset": 0, "size": self.gpu.size}}


class Engine:
    """One program, compiled once for one target format.

    Everything expensive happens in ``__init__``. After that a frame is:
    ``update()`` (a few dozen bytes of slot traffic) then ``encode()``
    into an encoder the HOST owns and submits.
    """

    backend_name = "webgpu"

    def __init__(self, low: P.Lowered, target_format: str, device=None):
        import wgpu

        self.low = low
        self.device = device or make_device()
        self.target_format = target_format
        self.res = P.Residency(
            lambda a: self.device.create_buffer_with_data(
                data=a.tobytes(),
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
            )
        )

        # ---- compute ----
        self.c_rows = P.compute_rows(low, self.res, P.WGSL)
        self.c_source = compute_source(self.c_rows)
        cmod = self.device.create_shader_module(code=self.c_source)
        PIPELINES.append("wgsl:compute")
        self.c_pipeline = self.device.create_compute_pipeline(
            layout="auto", compute={"module": cmod, "entry_point": "main"}
        )
        entries = [_buf_entry(b) for b in self.c_rows.buffers]
        self.c_slots = None
        if self.c_rows.slot_binding is not None:
            self.c_slots = _SlotBuffer(self.device, self.c_rows.slots, self.c_rows.slot_binding)
            entries.append(self.c_slots.entry())
        self.c_bind = self.device.create_bind_group(
            layout=self.c_pipeline.get_bind_group_layout(0), entries=entries
        )
        ext = [self.c_rows.extents[a][1] - self.c_rows.extents[a][0] for a in self.c_rows.axes]
        # threads -> workgroups: the rounding is a RUNTIME act (it exists
        # only because the API cannot dispatch a partial group).
        self.c_groups = ((ext[1] + 7) // 8, (ext[0] + 7) // 8, 1) if len(ext) == 2 else ((ext[0] + 63) // 64, 1, 1)

        # ---- render ----
        self.r_rows = P.render_rows(low, self.res, P.WGSL)
        self.r_source = render_source(low, self.r_rows, target_format)
        rmod = self.device.create_shader_module(code=self.r_source)
        PIPELINES.append(f"wgsl:render:{target_format}")
        self.r_pipeline = self.device.create_render_pipeline(
            layout="auto",
            vertex={"module": rmod, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            fragment={"module": rmod, "entry_point": "fs_main", "targets": [{"format": target_format}]},
        )
        entries = [_buf_entry(b) for b in self.r_rows.buffers]
        self.v_slots = self.f_slots = None
        if self.r_rows.v_slot_binding is not None:
            self.v_slots = _SlotBuffer(self.device, low.v_slots, self.r_rows.v_slot_binding)
            entries.append(self.v_slots.entry())
        if self.r_rows.f_slot_binding is not None:
            self.f_slots = _SlotBuffer(self.device, low.f_slots, self.r_rows.f_slot_binding)
            entries.append(self.f_slots.entry())
        self.r_bind = (
            self.device.create_bind_group(layout=self.r_pipeline.get_bind_group_layout(0), entries=entries)
            if entries
            else None
        )
        self.slot_bytes = 0

    def offscreen(self, size):
        return Offscreen(self.device, size)

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

    def encode(self, enc, color_view, clear=(0.0, 0.0, 0.0, 1.0)) -> None:
        """THE shared encode path: offscreen and windowed differ only in
        the view handed in and in who submits."""
        import wgpu

        cp = enc.begin_compute_pass()
        cp.set_pipeline(self.c_pipeline)
        cp.set_bind_group(0, self.c_bind)
        cp.dispatch_workgroups(*self.c_groups)
        cp.end()

        rp = enc.begin_render_pass(
            color_attachments=[
                {
                    "view": color_view,
                    "clear_value": clear,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        rp.set_pipeline(self.r_pipeline)
        if self.r_bind is not None:
            rp.set_bind_group(0, self.r_bind)
        rp.draw(self.low.draw_count)
        rp.end()


def _buf_entry(b: P.BufferBinding) -> dict:
    return {
        "binding": b.binding,
        "resource": {"buffer": b.resident.gpu, "offset": 0, "size": b.resident.gpu.size},
    }


class Offscreen:
    """An r32float attachment + its readback staging buffer, created ONCE."""

    def __init__(self, device, size):
        import wgpu

        self.device = device
        self.H, self.W = size
        self.tex = device.create_texture(
            size=(self.W, self.H, 1),
            format=wgpu.TextureFormat.r32float,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.view = self.tex.create_view()
        self.bpr = (self.W * 4 + 255) // 256 * 256
        self.staging = device.create_buffer(
            size=self.bpr * self.H, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        )

    def draw(self, engine: Engine, clear=(0.0, 0.0, 0.0, 1.0)) -> np.ndarray:
        enc = self.device.create_command_encoder()
        engine.encode(enc, self.view, clear=clear)
        enc.copy_texture_to_buffer(
            {"texture": self.tex},
            {"buffer": self.staging, "bytes_per_row": self.bpr, "rows_per_image": self.H},
            (self.W, self.H, 1),
        )
        self.device.queue.submit([enc.finish()])
        raw = np.frombuffer(self.device.queue.read_buffer(self.staging), dtype=np.float32)
        img = raw.reshape(self.H, self.bpr // 4)[:, : self.W]
        return np.flipud(img).astype(np.float64)  # device y-up -> the reference row convention


def run_window(low: P.Lowered, mouse, on_frame, width=900, height=600, title="pdum"):
    """rendercanvas + wgpu. The mouse arrives as a pointer event in
    CANVAS pixels, top-left origin; ``mouse`` is the demo's shared
    normalized-cursor object and every backend converts into it."""
    from rendercanvas.auto import RenderCanvas, loop

    device = make_device()
    canvas = RenderCanvas(size=(width, height), title=title, update_mode="continuous", max_fps=60)
    ctx = canvas.get_context("wgpu")
    fmt = ctx.get_preferred_format(device.adapter)
    ctx.configure(device=device, format=fmt)
    engine = Engine(low, fmt, device=device)
    print(f"[webgpu] surface format {fmt}; one pipeline, {low.draw_count} vertices, no per-frame lowering")

    def _pointer(event):
        w, h = canvas.get_logical_size()
        mouse.set(event["x"] / max(w, 1), event["y"] / max(h, 1))

    canvas.add_event_handler(_pointer, "pointer_move")

    @canvas.request_draw
    def draw():
        vs_fn = on_frame()
        engine.update(vs_fn)
        enc = device.create_command_encoder()
        engine.encode(enc, ctx.get_current_texture().create_view(), clear=(0.02, 0.02, 0.03, 1.0))
        device.queue.submit([enc.finish()])  # the host owns the submit

    loop.run()
