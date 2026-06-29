"""Compile a fragment entry (Program/Handle) to a WGSL module + uniform layout.

Pipeline: ``flatten`` (inline device fns, merge uniforms) → ``build_layout`` →
``infer`` → ``emit``. Pure (no GPU): the runtime turns a ``WgslModule`` into an
actual pipeline. ``compile_fragment`` is the structure path used at compile time;
the per-frame *values* come from re-running ``flatten`` (see the runtime)."""

from __future__ import annotations

from dataclasses import dataclass

from ...passes.infer import infer_function
from ...passes.inline import Flattened, flatten
from .emit import emit_module
from .layout import Layout, build_layout


@dataclass
class WgslModule:
    wgsl: str
    layout: Layout
    vs: str = "vs_main"
    fs: str = "fs_main"


def emit_from_flat(flat: Flattened) -> WgslModule:
    layout = build_layout(tuple(flat.names), tuple(flat.types[n] for n in flat.names))
    infer_function(flat.fn, layout.uniform_types())
    wgsl = emit_module(flat.fn, layout)
    return WgslModule(wgsl=wgsl, layout=layout)


def compile_fragment(program) -> WgslModule:
    """Flatten + emit. Accepts a fragment Handle or a Program (``shader(img)``)."""
    return emit_from_flat(flatten(program))
