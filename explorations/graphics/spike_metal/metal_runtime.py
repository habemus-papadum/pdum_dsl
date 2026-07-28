"""The Metal RUNTIME -- device, buffers, transfer, launch, timing.

The other half of the 280 hypothesis. This module owns everything the
draft's definition assigns to a runtime (device management, host<->device
transfer, launch, events/profiling) and it deliberately knows NOTHING
about our IR: no `walk_region`, no marker tables, no artifact. Its input
is shader SOURCE TEXT plus numpy arrays. If the runtime/backend line is
real, this file should be writable by someone who has never seen
`pdum.tl` -- and it was.

Bindings: `pyobjc-framework-Metal` (installed into the repo venv by this
spike; additive -- pyobjc-core, pyobjc-framework-cocoa,
pyobjc-framework-metal, no existing package changed). PyObjC turns out to
be entirely adequate: the selector-mangled names are mechanical
(`newLibraryWithSource:options:error:` ->
`newLibraryWithSource_options_error_`), and the two operations this spike
actually needed from the bridge -- reading a buffer's contents and
ADOPTING host memory without a copy -- both work directly on Python
buffer objects. No Swift/ObjC shim was needed.

THE UNIFIED-MEMORY POINT, concretely. `MTLDevice.hasUnifiedMemory` is
True on Apple silicon: the CPU and GPU address the SAME physical DRAM.
Two consequences this module exposes and `unified.py` measures:

  * `newBufferWithBytesNoCopy:` ADOPTS a page-aligned host allocation.
    The MTLBuffer and the numpy array are then two views of one
    allocation. A kernel's stores are visible in the numpy array with no
    readback call at all -- `waitUntilCompleted` is a SYNCHRONIZATION
    act, not a transfer act. There is no `read_buffer` in the Metal path
    because there is nothing to read back.
  * Even the ordinary `newBufferWithBytes:` path is a plain memcpy into
    shared storage, not a staged upload across a bus.

Contrast the WebGPU protocol, which is portable precisely because it
assumes discrete memory: `create_buffer_with_data` stages an upload, and
`queue.read_buffer` is a submit->map->wait round trip that 210 measured
at a FIXED ~1.6 ms from 64^2 to 1024^2. That fixed cost is a protocol
artifact, and on unified memory it is pure ceremony.

Timing: `MTLCommandBuffer.GPUStartTime/GPUEndTime` give a real
device-side interval in seconds with no query-set plumbing and no
feature request at device creation -- notably easier than the WebGPU
timestamp-query path that `spike_runner` had to monkeypatch the device
singleton to reach (its H5). We keep spike_controlflow's estimator
discipline: warm the GPU first (it idles low and ramps slowly, so the
first program timed in a process reads high), and take the MINIMUM over
reps as the headline per 210.
"""

from __future__ import annotations

import mmap
import time

import Metal
import numpy as np

PAGE = 16384  # Apple silicon page size; newBufferWithBytesNoCopy: wants both
# the base address and the length aligned to it.


class _Held:
    """A page-aligned host allocation Metal can adopt with no copy.
    `mmap` is the portable way to get guaranteed page alignment from
    Python; numpy arrays reject attribute assignment, so the mapping's
    lifetime is held here alongside the array that views it."""

    __slots__ = ("arr", "mm", "length")

    def __init__(self, arr, mm, length):
        self.arr, self.mm, self.length = arr, mm, length


def aligned_alloc(shape, dtype=np.float32) -> _Held:
    n = int(np.prod(shape)) if shape else 1
    itemsize = np.dtype(dtype).itemsize
    length = ((n * itemsize + PAGE - 1) // PAGE) * PAGE
    mm = mmap.mmap(-1, length)
    arr = np.frombuffer(mm, dtype=dtype, count=n).reshape(shape)
    assert arr.ctypes.data % PAGE == 0, "mmap did not return a page-aligned base"
    return _Held(arr, mm, length)


class MetalRuntime:
    """Device + queue + the library/pipeline caches.

    One instance per process is the intended use, mirroring the wgpu
    `_device()` singleton -- but note this one is CONSTRUCTIBLE with
    options and holds its own caches, which is exactly what
    spike_runner's H5 said the wgpu singleton refuses to be.
    """

    def __init__(self):
        dev = Metal.MTLCreateSystemDefaultDevice()
        if dev is None:
            raise RuntimeError("no Metal device on this machine")
        self.device = dev
        self.queue = dev.newCommandQueue()
        self._pipelines: dict[str, object] = {}

    # --- device facts ------------------------------------------------------
    @property
    def name(self) -> str:
        return str(self.device.name())

    @property
    def unified(self) -> bool:
        return bool(self.device.hasUnifiedMemory())

    # --- the backend's product becomes a pipeline (RUNTIME act) ------------
    def pipeline(self, source: str, entry: str = "main0"):
        """Compile MSL source and build a compute pipeline state, cached
        on the source text. `newLibraryWithSource:` is the RUNTIME
        compiler inside Metal.framework -- no `xcrun metal` and no
        offline .metallib needed, which is what makes a JIT backend
        possible without Xcode installed."""
        key = f"{entry}\x00{source}"
        got = self._pipelines.get(key)
        if got is not None:
            return got
        lib, err = self.device.newLibraryWithSource_options_error_(source, None, None)
        if lib is None:
            raise RuntimeError(f"MSL compile failed: {err}\n--- source ---\n{source}")
        fn = lib.newFunctionWithName_(entry)
        if fn is None:
            raise RuntimeError(f"no entry point {entry!r} in the compiled library")
        pso, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if pso is None:
            raise RuntimeError(f"pipeline creation failed: {err}")
        self._pipelines[key] = pso
        return pso

    # --- transfer ----------------------------------------------------------
    def buffer_copy(self, arr: np.ndarray):
        """The COPYING path -- the honest analogue of wgpu's
        `create_buffer_with_data`. On unified memory this is a memcpy
        into shared storage, not a bus transfer."""
        a = np.ascontiguousarray(arr)
        return self.device.newBufferWithBytes_length_options_(
            a.tobytes(), a.nbytes, Metal.MTLResourceStorageModeShared
        )

    def buffer_adopt(self, held: _Held):
        """The ZERO-COPY path: adopt the host allocation itself. The
        returned MTLBuffer and `held.arr` are two views of ONE
        allocation. Requires page-aligned base AND page-multiple length,
        which `aligned_alloc` guarantees. The deallocator is None: we
        keep ownership, Metal must not free it."""
        buf = self.device.newBufferWithBytesNoCopy_length_options_deallocator_(
            held.mm, held.length, Metal.MTLResourceStorageModeShared, None
        )
        if buf is None:
            raise RuntimeError("newBufferWithBytesNoCopy: refused this allocation")
        return buf

    def buffer_empty(self, nbytes: int):
        return self.device.newBufferWithLength_options_(nbytes, Metal.MTLResourceStorageModeShared)

    @staticmethod
    def read(buf, count: int, dtype=np.float32) -> np.ndarray:
        """'Readback' on unified memory is just a view of the buffer's
        contents. There is no map, no submit, no wait -- the wait already
        happened at `waitUntilCompleted`. Kept as a named method only so
        the WGSL-path comparison in unified.py is like-for-like."""
        nbytes = count * np.dtype(dtype).itemsize
        return np.frombuffer(buf.contents().as_buffer(nbytes), dtype=dtype).copy()

    @staticmethod
    def view(buf, shape, dtype=np.float32) -> np.ndarray:
        """A live, NON-copying view of device-visible memory."""
        n = int(np.prod(shape)) if shape else 1
        nbytes = n * np.dtype(dtype).itemsize
        return np.frombuffer(buf.contents().as_buffer(nbytes), dtype=dtype).reshape(shape)

    # --- launch ------------------------------------------------------------
    def dispatch(self, pso, buffers, grid, threadgroup, *, exact_grid=False, wait=True):
        """Encode + submit one compute dispatch.

        `grid` is in THREADS. With `exact_grid` we call
        `dispatchThreads:` (non-uniform threadgroups) and the shader
        needs no bounds guard; otherwise we round up to whole
        threadgroups and call `dispatchThreadgroups:`, which is what
        WebGPU's `dispatch_workgroups` is limited to and why the WGSL
        path must emit a guard.

        Returns the command buffer so the caller can read GPU timings.
        """
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(pso)
        for i, b in enumerate(buffers):
            enc.setBuffer_offset_atIndex_(b, 0, i)
        tg = Metal.MTLSizeMake(*threadgroup)
        if exact_grid:
            enc.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSizeMake(*grid), tg)
        else:
            groups = Metal.MTLSizeMake(
                (grid[0] + threadgroup[0] - 1) // threadgroup[0],
                (grid[1] + threadgroup[1] - 1) // threadgroup[1],
                (grid[2] + threadgroup[2] - 1) // threadgroup[2],
            )
            enc.dispatchThreadgroups_threadsPerThreadgroup_(groups, tg)
        enc.endEncoding()
        cmd.commit()
        if wait:
            cmd.waitUntilCompleted()
        return cmd

    @staticmethod
    def gpu_seconds(cmd) -> float:
        """The honest device-side interval. No query set, no feature
        request at device creation -- contrast spike_runner's H5, where
        reaching WebGPU timestamp queries required monkeypatching the
        `graphics._GPU` singleton."""
        return float(cmd.GPUEndTime() - cmd.GPUStartTime())

    def warm(self, seconds: float = 0.35):
        """spike_controlflow's rig finding: the GPU idles low and ramps
        slowly, so the first program timed in a process reads ~1.7x
        high. Burn sustained load before measuring anything."""
        src = """
#include <metal_stdlib>
using namespace metal;
kernel void main0(device float* o [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {
  float a = float(gid.x) * 1e-6f;
  for (int i = 0; i < 512; ++i) { a = fma(a, 1.0000001f, 1e-7f); a = sqrt(a * a + 1.0f); }
  o[gid.x] = a;
}
"""
        pso = self.pipeline(src)
        buf = self.buffer_empty(1 << 22)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            self.dispatch(pso, [buf], (1 << 20, 1, 1), (256, 1, 1))


_RT: list[MetalRuntime] = []


def runtime() -> MetalRuntime:
    if not _RT:
        _RT.append(MetalRuntime())
    return _RT[0]
