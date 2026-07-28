"""Counting copies by instrumenting the device, not by reading the source.

Wraps the four methods that move bytes across the PCIe/unified-memory
boundary — buffer creation with data, ``write_buffer``, ``write_texture``,
and the synchronous ``read_buffer`` map — at the CLASS level (wgpu's Python
objects reject instance attribute assignment).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Counts:
    create_with_data: int = 0
    create_with_data_bytes: int = 0
    write_buffer: int = 0
    write_buffer_bytes: int = 0
    write_texture: int = 0
    read_buffer: int = 0
    read_buffer_bytes: int = 0
    pipelines: int = 0
    shader_modules: int = 0
    detail: list = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"upload(create_buffer_with_data) x{self.create_with_data} = {self.create_with_data_bytes}B, "
            f"write_buffer x{self.write_buffer} = {self.write_buffer_bytes}B, "
            f"readback(read_buffer) x{self.read_buffer} = {self.read_buffer_bytes}B, "
            f"shader modules x{self.shader_modules}, pipelines x{self.pipelines}"
        )


@contextmanager
def counting(device):
    cls, qcls = type(device), type(device.queue)
    saved = {
        (cls, "create_buffer_with_data"): cls.create_buffer_with_data,
        (cls, "create_shader_module"): cls.create_shader_module,
        (cls, "create_compute_pipeline"): cls.create_compute_pipeline,
        (cls, "create_render_pipeline"): cls.create_render_pipeline,
        (qcls, "read_buffer"): qcls.read_buffer,
        (qcls, "write_buffer"): qcls.write_buffer,
        (qcls, "write_texture"): qcls.write_texture,
    }
    c = Counts()

    def cbd(self, *a, **kw):
        data = kw.get("data") if "data" in kw else (a[0] if a else b"")
        n = len(memoryview(data).cast("B")) if data is not None else 0
        c.create_with_data += 1
        c.create_with_data_bytes += n
        c.detail.append(("create_buffer_with_data", n))
        return saved[(cls, "create_buffer_with_data")](self, *a, **kw)

    def rb(self, *a, **kw):
        out = saved[(qcls, "read_buffer")](self, *a, **kw)
        c.read_buffer += 1
        c.read_buffer_bytes += len(memoryview(out).cast("B"))
        c.detail.append(("read_buffer", len(memoryview(out).cast("B"))))
        return out

    def wb(self, buffer, offset, data, *a, **kw):
        n = len(memoryview(data).cast("B"))
        c.write_buffer += 1
        c.write_buffer_bytes += n
        c.detail.append(("write_buffer", n))
        return saved[(qcls, "write_buffer")](self, buffer, offset, data, *a, **kw)

    def wt(self, *a, **kw):
        c.write_texture += 1
        return saved[(qcls, "write_texture")](self, *a, **kw)

    def sm(self, *a, **kw):
        c.shader_modules += 1
        return saved[(cls, "create_shader_module")](self, *a, **kw)

    def cp(self, *a, **kw):
        c.pipelines += 1
        return saved[(cls, "create_compute_pipeline")](self, *a, **kw)

    def rp(self, *a, **kw):
        c.pipelines += 1
        return saved[(cls, "create_render_pipeline")](self, *a, **kw)

    cls.create_buffer_with_data, cls.create_shader_module = cbd, sm
    cls.create_compute_pipeline, cls.create_render_pipeline = cp, rp
    qcls.read_buffer, qcls.write_buffer, qcls.write_texture = rb, wb, wt
    try:
        yield c
    finally:
        for (k, name), fn in saved.items():
            setattr(k, name, fn)
