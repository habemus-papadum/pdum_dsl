"""The MSL column — the compute half, structurally the WGSL column's twin.

That the two files are spelled alike, one dialect apart, IS the seam's
existence proof (the Metal spike's finding): the rows came out of the
one walker unmodified and every difference below is SHELL. Four of
them, and each is a contract clause rather than a style choice:

- buffers are ENTRY-POINT PARAMETERS with ``[[buffer(i)]]``, read-only
  spelled by the C qualifier ``const device`` — there is no bind-group
  object to declare, so the binding table can only be honored by
  reading it as DATA (210);
- the entry point cannot be called ``main`` (MSL is a C++ dialect);
- the thread size appears NOWHERE in the source — it is an argument to
  the dispatch call, which is why a backend returning only source text
  cannot express Metal;
- ``guard="exact"``: ``dispatchThreads:`` launches non-uniform
  threadgroups covering exactly the requested grid, so the overhang
  guard is dead code and is not emitted. Proven bitwise-equal to the
  guarded WebGPU-shaped dispatch.

PyObjC is the whole runtime — no Swift/ObjC shim, and
``newLibraryWithSource:`` is Metal's in-process compiler, so JIT MSL
needs no Xcode. ``Metal`` imports LAZILY inside the acquirer and the
compiler; importing ``pdum.rt`` off a Mac is safe.
"""

from __future__ import annotations

import numpy as np

from .. import emit, staging, transfer
from ..contract import LaunchContract
from ..registry import register_acquirer, register_compiler
from ..select import MetalRuntime, metal

ENTRY = "main0"  # `main` is reserved in MSL

# GPUStartTime/GPUEndTime are on every command buffer: no query set, no
# feature request, nothing to negotiate at creation (210). The name is
# accepted so a program can spell one feature set for both columns.
FEATURES = {"timestamps": None}


def _lower(art, thread_size):
    d = emit.MSL
    rows = emit.compute_rows(art, d)
    ts = tuple(thread_size) if thread_size else emit.default_thread_size(rows)
    bindings = emit.compute_bindings(rows)
    params = [
        f"    {'device' if b.writable else 'const device'} {d.fty}* {b.name} [[buffer({b.index})]]," for b in bindings
    ]
    source = "\n".join(
        [
            "#include <metal_stdlib>",
            "using namespace metal;",
            f"kernel void {ENTRY}(",
            *params,
            f"    {d.vec(3, 'uint')} gid [[thread_position_in_grid]])",
            "{",
            *rows.lines,  # no guard: the grid is exact
            *rows.stores,
            "}",
        ]
    )
    contract = LaunchContract(
        thread_size=ts,
        guard="exact",  # dispatchThreads: covers the grid exactly; the guard would be dead
        bindings=(bindings,),  # one stage, one table (the render era indexes two)
        slots=staging.device_slots(rows.slots),
        math=rows.math,
    )
    return source, contract, rows


def generate(art, thread_size: tuple[int, ...] | None = None) -> tuple[str, LaunchContract]:
    """A kernel artifact -> (MSL compute source, launch contract). The
    thread size is IN the contract and nowhere in the source, which is
    the sharpest reason a backend cannot return source alone (210)."""
    source, contract, _rows = _lower(art, thread_size)
    return source, contract


class Device:
    """Device + queue, CONSTRUCTIBLE with options — which is what the old
    WebGPU singleton refused to be. It holds no pipeline cache of its
    own: a pipeline is built once per compiled executor, and the content
    door is what keeps that once (registry.py)."""

    def __init__(self):
        import Metal

        dev = Metal.MTLCreateSystemDefaultDevice()
        if dev is None:
            raise RuntimeError("no Metal device on this machine")
        self.metal = Metal
        self.device = dev
        self.queue = dev.newCommandQueue()

    @property
    def unified(self) -> bool:
        """True on Apple silicon — host and device address one
        allocation, so residency is not a transfer question (210)."""
        return bool(self.device.hasUnifiedMemory())

    def pipeline(self, source: str, entry: str = ENTRY):
        lib, err = self.device.newLibraryWithSource_options_error_(source, None, None)
        if lib is None:
            raise RuntimeError(f"MSL compile failed: {err}\n--- source ---\n{source}")
        fn = lib.newFunctionWithName_(entry)
        if fn is None:
            raise RuntimeError(f"no entry point {entry!r} in the compiled library")
        pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if pso is None:
            raise RuntimeError(f"pipeline creation failed: {err}")
        return pso

    def buffer(self, arr: np.ndarray):
        return self.device.newBufferWithBytes_length_options_(
            arr.tobytes(), arr.nbytes, self.metal.MTLResourceStorageModeShared
        )

    @staticmethod
    def read(buf, count: int) -> np.ndarray:
        """'Readback' on unified memory is a view of the buffer's own
        bytes — the wait already happened at ``waitUntilCompleted``."""
        return np.frombuffer(buf.contents().as_buffer(count * 4), dtype=np.float32).copy()


def acquire(features: tuple[str, ...] = ()):
    """The device. Every feature this column knows is free, so the list
    only has to be RECOGNIZED — an unknown name still refuses, because a
    silently dropped request is worse than an absent one."""
    for name in features:
        if name not in FEATURES:
            raise ValueError(f"unknown feature {name!r} — this column knows {', '.join(sorted(FEATURES))}")
    return Device()


def compile_compute(art, device: Device, thread_size: tuple[int, ...] | None = None):
    """Pipeline once, launch many — the same ``(values, staging) ->
    effect`` shape the launcher already serves."""
    source, contract, rows = _lower(art, thread_size)
    pso = device.pipeline(source)
    names = transfer.value_names(art)
    bound, host_slots, grid = rows.bound, rows.slots, rows.threads()
    Metal = device.metal

    def executor(values, staging_bytes):
        bufs, shapes = [], []
        for i in bound:  # binding slot k carries values[bound[k]]
            arr = transfer.host_f32(values[i])
            shapes.append(arr.shape)
            bufs.append(device.buffer(arr))
        if host_slots:  # the DECLARED narrowing, one packer for every column
            bufs.append(device.buffer(staging.pack_device(host_slots, staging_bytes)))
        cmd = device.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        for i, b in enumerate(bufs):
            enc.setBuffer_offset_atIndex_(b, 0, i)
        enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSizeMake(*grid), Metal.MTLSizeMake(*contract.thread_size))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()  # synchronization, not transfer
        for i, buf, shape in zip(bound, bufs, shapes):
            if names[i] in art.writable:
                transfer.writeback(values[i], device.read(buf, int(np.prod(shape))).reshape(shape))
        return None

    executor.source = source  # what ran, and (via the contract) what it substituted
    executor.contract = contract
    return executor


register_acquirer(MetalRuntime, acquire)
register_compiler(metal, compile_compute)
