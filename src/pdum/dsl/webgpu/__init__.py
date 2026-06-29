"""WebGPU runtime: ``Context`` (device) and ``Drawer`` (the per-shader object that
holds the compiled pipeline + uniform buffer and updates uniforms each frame).

Imports ``wgpu`` lazily-ish (at module import), so this subpackage is only loaded
when a caller actually wants to render — the core stays GPU-free.
"""

from .runtime import Context, Drawer, GpuProgram, OffscreenTarget, WindowTarget

__all__ = ["Context", "Drawer", "GpuProgram", "OffscreenTarget", "WindowTarget"]
