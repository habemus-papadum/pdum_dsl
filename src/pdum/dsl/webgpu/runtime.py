"""The WebGPU runtime that turns the type-keyed cache into a no-recompile render
loop.

The crux (the thesis): ``Drawer.update`` looks up ``(FnType, arg_types, generation)``
in its :class:`~pdum.dsl.cache.SpecCache`. On a **hit** (capture *values* changed
but *types* did not) it only repacks the uniform buffer and ``write_buffer``\\ s it —
no shader/pipeline rebuild. On a **miss** it compiles WGSL → pipeline once. Over a
render loop that moves a parameter around, ``cache.compile_count`` stays at 1 while
``uniform_writes`` grows with the frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import wgpu
from wgpu.backends.auto import gpu

from ..backends.wgsl.compile import WgslModule, emit_from_flat
from ..backends.wgsl.layout import Layout
from ..cache import SpecCache
from ..jit import Handle, Program
from ..passes.inline import flatten


def _round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


@dataclass
class GpuProgram:
    """A compiled, GPU-resident shader: the cache's artifact type for WebGPU."""

    pipeline: Any  # GPURenderPipeline (wgpu handles are dynamically typed)
    bind_group: Any  # GPUBindGroup
    uniform_buffer: Any  # GPUBuffer
    layout: Layout
    module: WgslModule


class OffscreenTarget:
    """Render to an owned texture; ``read_pixels`` for tests / image output."""

    def __init__(self, device, size=(256, 256), format="rgba8unorm"):
        self.device = device
        self.size = size
        self.format = format
        w, h = size
        self.texture = device.create_texture(
            size=(w, h, 1),
            format=format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self._view = self.texture.create_view()

    def get_view(self):
        return self._view

    def read_pixels(self) -> bytes:
        """Read back the texture as tightly-packed RGBA bytes (row stride removed)."""
        w, h = self.size
        bytes_per_row = _round_up(w * 4, 256)
        rbuf = self.device.create_buffer(
            size=bytes_per_row * h,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        enc = self.device.create_command_encoder()
        enc.copy_texture_to_buffer(
            {"texture": self.texture, "mip_level": 0, "origin": (0, 0, 0)},
            {"buffer": rbuf, "offset": 0, "bytes_per_row": bytes_per_row, "rows_per_image": h},
            (w, h, 1),
        )
        self.device.queue.submit([enc.finish()])
        rbuf.map_sync(wgpu.MapMode.READ)
        padded = bytes(rbuf.read_mapped())
        rbuf.unmap()
        if bytes_per_row == w * 4:
            return padded
        out = bytearray()
        for row in range(h):
            start = row * bytes_per_row
            out += padded[start : start + w * 4]
        return bytes(out)


class WindowTarget:
    """Render to a rendercanvas present surface (glfw window)."""

    def __init__(self, present_context, format):
        self.present_context = present_context
        self.format = format

    @property
    def size(self):
        return tuple(self.present_context.physical_size)

    def get_view(self):
        return self.present_context.get_current_texture().create_view()


class Drawer:
    """Holds the compiled pipeline + uniform buffer for one shader and updates
    uniforms each frame. Reuses the pipeline across capture-value changes."""

    def __init__(self, context: Context, target):
        self.context = context
        self.device = context.device
        self.target = target
        self.cache = SpecCache()
        self.uniform_writes = 0
        self._current: GpuProgram | None = None

    @property
    def compile_count(self) -> int:
        return self.cache.compile_count

    def update(self, program: Program | Handle) -> GpuProgram:
        """Phase B: resolve the specialization (compile once) and write uniforms.

        ``flatten`` runs every frame to collect the *current* merged uniform values
        from the freshly-rebuilt closures; the expensive shader/pipeline build runs
        only on a cache miss (types changed)."""
        if isinstance(program, Handle):
            program = Program(program, ())
        flat = flatten(program)
        compiled: GpuProgram = self.cache.get_or_compile(
            program.entry.fntype, program.arg_types, lambda: self._build(flat)
        )
        data = compiled.layout.pack(flat.values)
        self.device.queue.write_buffer(compiled.uniform_buffer, 0, data)
        self.uniform_writes += 1
        self._current = compiled
        return compiled

    def _build(self, flat) -> GpuProgram:
        mod = emit_from_flat(flat)
        dev = self.device
        sm = dev.create_shader_module(code=mod.wgsl)
        size = mod.layout.size
        ubuf = dev.create_buffer(size=size, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        bgl = dev.create_bind_group_layout(
            entries=[{"binding": 0, "visibility": wgpu.ShaderStage.FRAGMENT, "buffer": {"type": "uniform"}}]
        )
        bind_group = dev.create_bind_group(
            layout=bgl,
            entries=[{"binding": 0, "resource": {"buffer": ubuf, "offset": 0, "size": size}}],
        )
        pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
        pipeline = dev.create_render_pipeline(
            layout=pl,
            vertex={"module": sm, "entry_point": mod.vs},
            fragment={"module": sm, "entry_point": mod.fs, "targets": [{"format": self.target.format}]},
            primitive={"topology": "triangle-list"},
        )
        return GpuProgram(pipeline, bind_group, ubuf, mod.layout, mod)

    def show(self, clear=(0.0, 0.0, 0.0, 1.0)) -> None:
        """Encode + submit a render pass drawing the full-screen triangle."""
        if self._current is None:
            raise RuntimeError("call update(program) before show()")
        view = self.target.get_view()
        enc = self.device.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[{"view": view, "clear_value": clear, "load_op": "clear", "store_op": "store"}]
        )
        rp.set_pipeline(self._current.pipeline)
        rp.set_bind_group(0, self._current.bind_group)
        rp.draw(3, 1, 0, 0)
        rp.end()
        self.device.queue.submit([enc.finish()])


class Context:
    """Owns the wgpu adapter + device and constructs drawers."""

    def __init__(self, power_preference: str = "high-performance"):
        self.adapter = gpu.request_adapter_sync(power_preference=power_preference)
        self.device = self.adapter.request_device_sync()

    def offscreen_drawer(self, size=(256, 256), format="rgba8unorm") -> Drawer:
        return Drawer(self, OffscreenTarget(self.device, size, format))

    def window_drawer(self, size=(640, 480), title="pdum.dsl"):
        """Open a glfw window and return ``(canvas, drawer)``. Drive it with
        :meth:`run`. Importing rendercanvas.glfw here keeps tests display-free."""
        from rendercanvas.glfw import RenderCanvas

        canvas = RenderCanvas(size=size, title=title)
        present = canvas.get_context("wgpu")
        fmt = present.get_preferred_format(self.adapter)
        present.configure(device=self.device, format=fmt)
        return canvas, Drawer(self, WindowTarget(present, fmt))

    def run(self, canvas, frame_fn) -> None:
        """Wire a per-frame callback into the rendercanvas event loop (blocking)."""
        from rendercanvas.glfw import loop

        canvas.request_draw(frame_fn)
        loop.run()
