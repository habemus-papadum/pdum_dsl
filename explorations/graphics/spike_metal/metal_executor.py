"""The GLUE: artifact -> (MSL backend + Metal runtime) -> executor column.

The twin of `wgsl_executor.compile_wgsl` / `wgpu_artifact`. It exists to
answer the lead's last question -- "what is the minimal analogous
`metal_artifact(art)`?" -- and the answer is: THIS FILE, and it is 40
lines of substance.

Every block below is tagged [BACKEND] / [RUNTIME] / [SHARED] as the seam
audit's primary evidence. The tags are not decoration: they were
assigned by diffing this file against `compile_wgsl` line by line, and
[SHARED] means the line is LOGICALLY IDENTICAL to its WGSL counterpart,
not merely similar.

The one structural improvement over `compile_wgsl` -- and it is a seam
finding, not a style preference: the WGSL executor computes launch
geometry as `(ext[1] + 7) // 8` inline at the dispatch call, FUSING two
different things. "How many threads does this lattice need" is
target-neutral artifact arithmetic; "how many workgroups is that" is a
runtime concern that exists only because the API can't dispatch a
partial group. Here the executor computes THREADS (shared) and
`MetalRuntime.dispatch` does the rounding (runtime). Splitting them is
what made it visible that Metal need not round at all
(`dispatchThreads:`), and therefore that the WGSL shader's bounds guard
is a consequence of a RUNTIME limitation leaking into BACKEND output.

WHAT THIS FILE PROVES ABOUT ZERO COPY, negatively and usefully.
`mode="adopt"` removes the readback -- results appear in host memory
with no transfer call. It CANNOT remove the upload, because
`_Artifact.launch` hands the executor `Tensor` objects and the only way
to get bytes out of one is `t.to_numpy(order=...)`, which manufactures a
fresh f64->f32 contiguous host array every launch. Both backends pay
that identically (`wgsl_executor.py:260` is the same line). So on
unified memory the remaining copy is not a device transfer at all --
it is our own host-side repack, and it is the whole cost. That is
spike_runner's H4 arriving from a completely different direction: the
Metal path makes the transfer free and the repack is what's left
standing. See `unified.py` for the numbers.
"""

from __future__ import annotations

import struct
from dataclasses import replace

import numpy as np
from metal_runtime import aligned_alloc, runtime
from msl_backend import METAL_FP, Untranslatable, _translate  # noqa: F401 -- re-exported
from pdum.tl.tensor import Tensor


def compile_msl(art, *, mode: str = "copy", exact_grid: bool = False):
    """The backend column's THIRD executor: (values, staging) -> effect,
    behind the same signature `run_region` serves.

    mode="copy"  -- mirrors the wgpu path exactly (upload, dispatch, read).
    mode="adopt" -- writable tensors ride an adopted page-aligned host
                    allocation, so the kernel's stores land directly in
                    the numpy array with no readback call.
    """
    source, meta = _translate(art, exact_grid=exact_grid)  # [BACKEND] the only backend line
    rt = runtime()  # [RUNTIME]
    pso = rt.pipeline(source, meta["entry"])  # [RUNTIME] consumes the backend's product

    def executor(values, staging):
        writable = set(art.writable)  # [SHARED] artifact reasoning, no target in it
        bufs, orders, held = [], [], []
        for i, t in enumerate(values):
            # [SHARED] verbatim from compile_wgsl: dim order, f32 narrowing,
            # contiguity. This is the layout destruction of spike_runner H4,
            # committed identically by both backends.
            order = tuple(d.name for d in t.layout.dims)
            arr = np.ascontiguousarray(t.to_numpy(order=order), dtype=np.float32)
            name = art.tensor_params[i] if i < len(art.tensor_params) else f"tap:{i}"
            orders.append((name, order, arr.shape))
            if mode == "adopt" and name in writable:  # [RUNTIME] zero-copy adoption
                h = aligned_alloc(arr.shape, np.float32)
                h.arr[...] = arr
                held.append(h)
                bufs.append(rt.buffer_adopt(h))
            else:  # [RUNTIME] the copying path, wgpu's create_buffer_with_data
                held.append(None)
                bufs.append(rt.buffer_copy(arr))
        if meta["slots"]:
            # [SHARED] byte-identical to compile_wgsl: the staging plan IS the
            # ABI (210), and neither backend invents its layout -- but note BOTH
            # separately invent the DEVICE representation (a flat f32 array),
            # which the plan does not describe. That is 210's contradiction,
            # reproduced rather than fixed, and it is a SHARED-tier gap.
            uvals = [struct.unpack_from(fmt, staging, off)[0] for _, off, fmt in meta["slots"]]
            bufs.append(rt.buffer_copy(np.asarray(uvals, dtype=np.float32)))

        # [SHARED] launch geometry in THREADS -- pure lattice arithmetic.
        axes = meta["axes"]
        ext = [meta["extents"][a][1] - meta["extents"][a][0] for a in axes]
        grid = (ext[1], ext[0], 1) if len(axes) == 2 else (ext[0], 1, 1)

        # [RUNTIME] encode, submit, synchronize.
        rt.dispatch(pso, bufs, grid, meta["threadgroup"], exact_grid=meta["exact_grid"])

        # [SHARED] the writeback discipline: which params are writable, and the
        # store seam. Only the middle line (how bytes come back) is runtime.
        from pdum.tl.ir import Token, _store

        for (name, order, shape), buf, h, t in zip(orders, bufs, held, values):
            if name not in writable:
                continue
            if h is not None:  # [RUNTIME] ZERO COPY: already in host memory
                raw = h.arr
            else:  # [RUNTIME] the copying readback
                raw = rt.read(buf, int(np.prod(shape))).reshape(shape)
            _store(Token(), t, Tensor.from_numpy(raw.astype(np.float64), order))
        return None

    return executor


def metal_artifact(art, *, mode: str = "copy", exact_grid: bool = False):
    """The drop-in, exactly parallel to `wgpu_artifact`: the same
    artifact with only the backend column swapped -- `launch()` then runs
    staging/rebind/overlap identically and lands on Metal.

    That this is spelled identically to the WebGPU version, one word
    apart, IS the seam's existence proof.
    """
    return replace(art, executor=compile_msl(art, mode=mode, exact_grid=exact_grid))
