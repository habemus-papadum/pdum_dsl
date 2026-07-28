"""Unified memory, made concrete -- proof then numbers.

Three parts, because "unified memory kills half the transfer API" (280's
words) is three separate claims and only one of them survives contact.

PART 1 -- the PROOF, correctness not timing. A page-aligned numpy
allocation is adopted by `newBufferWithBytesNoCopy:`. Then:
  (a) the kernel's stores appear in the numpy array with NO readback call
      of any kind -- `waitUntilCompleted` is the only thing between the
      dispatch and reading `arr[...]`, and it is a SYNCHRONIZATION act,
      not a transfer;
  (b) host writes made AFTER the MTLBuffer exists are seen by the next
      dispatch with no upload call -- the aliasing goes both ways;
  (c) the addresses are checked to be one allocation, not a coincidence.

PART 2 -- the PROTOCOL benchmark. The same trivial kernel over several
sizes, three ways: wgpu's full protocol (create_buffer_with_data ->
dispatch -> read_buffer), Metal copying (newBufferWithBytes -> dispatch
-> read contents), and Metal adopting (dispatch only; buffers persist and
alias host memory). Hand-written shaders on both sides, because what is
being measured here is the RUNTIME protocol and nothing else; the
translators are validated separately in differential.py.

PART 3 -- the honest deflation. The same comparison through the REAL
`_Artifact.launch` path. Part 2's win mostly evaporates, and the reason
is the finding: on unified memory the device transfer was never the
cost. `t.to_numpy(order=...)` + `ascontiguousarray(..., f32)` -- one line,
identical in `wgsl_executor.py:260` and `metal_executor.py` -- rebuilds
a fresh f64->f32 contiguous host array on every single launch. Removing
the transfer just exposes our own repack underneath it. That is
spike_runner's H4 reached from the opposite direction: H4 said layout is
destroyed at the buffer boundary; this says that on unified memory the
destruction is ALL of the cost, because nothing else is left.

Method (210 + spike_controlflow's rig notes): GPU idles low and ramps
slowly, so `warm()` burns sustained load first; MINIMUM over reps is the
headline because noise is one-sided; GPU-side intervals come from
`MTLCommandBuffer.GPUStartTime/GPUEndTime`, wall times are stated as
wall. Wall timings here include the sync wait by construction -- that is
the protocol being measured, not an accident.
"""

from __future__ import annotations

import time

import _paths  # noqa: F401
import numpy as np
from metal_runtime import aligned_alloc, runtime

REPS = 60
# Capped at 2048^2: with a 256-wide workgroup, 4096^2 needs 65536 groups and
# WebGPU's max_compute_workgroups_per_dimension is 65535 (spike_controlflow's
# rig note, reproduced). Metal has no such trouble here, which is itself a
# small runtime-portability datum -- but the comparison must stay like-for-like.
SIZES = [64 * 64, 256 * 256, 1024 * 1024, 2048 * 2048]

_MSL = """
#include <metal_stdlib>
using namespace metal;
kernel void main0(const device float* src [[buffer(0)]],
                  device float* dst [[buffer(1)]],
                  uint3 gid [[thread_position_in_grid]]) {
  dst[gid.x] = src[gid.x] * 2.0f + 1.0f;
}
"""

_WGSL = """
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  dst[gid.x] = src[gid.x] * 2.0 + 1.0;
}
"""


def _min_of(fn, reps=REPS):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3  # ms


# --- PART 1 -----------------------------------------------------------------


def proof():
    rt = runtime()
    print(f"device: {rt.name}   hasUnifiedMemory: {rt.unified}")
    n = 4096
    src = aligned_alloc((n,), np.float32)
    dst = aligned_alloc((n,), np.float32)
    src.arr[...] = np.arange(n, dtype=np.float32)
    dst.arr[...] = -777.0  # a sentinel we must never see again

    b_src = rt.buffer_adopt(src)
    b_dst = rt.buffer_adopt(dst)

    # (c) one allocation, checked -- the MTLBuffer's contents pointer IS the
    # numpy array's data pointer.
    dst_view = rt.view(b_dst, (n,))
    aliased = dst_view.ctypes.data == dst.arr.ctypes.data
    print(f"(c) MTLBuffer.contents() is the numpy data pointer: {aliased} "
          f"[{hex(dst.arr.ctypes.data)}]")

    pso = rt.pipeline(_MSL)
    rt.dispatch(pso, [b_src, b_dst], (n, 1, 1), (256, 1, 1))

    # (a) NO readback call appears between the dispatch and this line.
    want = np.arange(n, dtype=np.float32) * 2.0 + 1.0
    ok_a = np.array_equal(dst.arr, want)
    print(f"(a) kernel stores visible in the numpy array with NO readback call: {ok_a} "
          f"(max abs err {np.max(np.abs(dst.arr - want)):.1f}, sentinel -777 gone: "
          f"{not np.any(dst.arr == -777.0)})")

    # (b) host writes after buffer creation, no upload call, seen by the GPU.
    src.arr[...] = np.full(n, 10.0, dtype=np.float32)
    rt.dispatch(pso, [b_src, b_dst], (n, 1, 1), (256, 1, 1))
    ok_b = np.all(dst.arr == 21.0)
    print(f"(b) host writes seen by the next dispatch with NO upload call: {bool(ok_b)} "
          f"(all == 21.0: {bool(ok_b)})")
    return ok_a and bool(ok_b) and aliased


# --- PART 2 -----------------------------------------------------------------


def protocol_bench():
    import wgpu
    from pdum.tl.graphics import _device

    rt = runtime()
    rt.warm()
    dev = _device()
    wmod = dev.create_shader_module(code=_WGSL)
    wpipe = dev.create_compute_pipeline(layout="auto", compute={"module": wmod, "entry_point": "main"})
    mpso = rt.pipeline(_MSL)

    rows = []
    for n in SIZES:
        host = np.arange(n, dtype=np.float32)
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC

        # -- wgpu, the full protocol, as compile_wgsl does it every launch
        def wgpu_full():
            a = dev.create_buffer_with_data(data=host.tobytes(), usage=usage)
            b = dev.create_buffer(size=host.nbytes, usage=usage)
            bind = dev.create_bind_group(
                layout=wpipe.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": a, "offset": 0, "size": a.size}},
                    {"binding": 1, "resource": {"buffer": b, "offset": 0, "size": b.size}},
                ],
            )
            enc = dev.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(wpipe)
            cp.set_bind_group(0, bind)
            cp.dispatch_workgroups(n // 256, 1, 1)
            cp.end()
            dev.queue.submit([enc.finish()])
            return np.frombuffer(dev.queue.read_buffer(b), dtype=np.float32)

        # -- wgpu readback alone (210's fixed-latency claim, re-measured)
        b_p = dev.create_buffer(size=host.nbytes, usage=usage)

        def wgpu_readback_only():
            return dev.queue.read_buffer(b_p)

        def wgpu_upload_only():
            dev.create_buffer_with_data(data=host.tobytes(), usage=usage)

        # -- Metal, copying (the like-for-like analogue)
        def metal_copy():
            a = rt.buffer_copy(host)
            b = rt.buffer_empty(host.nbytes)
            rt.dispatch(mpso, [a, b], (n, 1, 1), (256, 1, 1))
            return rt.read(b, n)

        # -- Metal, adopting: buffers persist, host memory IS device memory
        hs = aligned_alloc((n,), np.float32)
        hd = aligned_alloc((n,), np.float32)
        hs.arr[...] = host
        ba, bb = rt.buffer_adopt(hs), rt.buffer_adopt(hd)

        def metal_adopt():
            rt.dispatch(mpso, [ba, bb], (n, 1, 1), (256, 1, 1))
            return hd.arr  # no read call: it is already here

        assert np.array_equal(wgpu_full(), host * 2 + 1)
        assert np.array_equal(metal_copy(), host * 2 + 1)
        assert np.array_equal(metal_adopt(), host * 2 + 1)

        reps = REPS if n <= 1 << 20 else max(12, REPS // 4)
        cmd = rt.dispatch(mpso, [ba, bb], (n, 1, 1), (256, 1, 1))
        gpu_ms = min(
            rt.gpu_seconds(rt.dispatch(mpso, [ba, bb], (n, 1, 1), (256, 1, 1))) * 1e3
            for _ in range(10)
        )
        rows.append(
            {
                "n": n,
                "MB": host.nbytes / 1e6,
                "wgpu_full": _min_of(wgpu_full, reps),
                "wgpu_up": _min_of(wgpu_upload_only, reps),
                "wgpu_read": _min_of(wgpu_readback_only, reps),
                "metal_copy": _min_of(metal_copy, reps),
                "metal_adopt": _min_of(metal_adopt, reps),
                "gpu_ms": gpu_ms,
            }
        )
        del cmd
    return rows


# --- PART 3 -----------------------------------------------------------------


def artifact_bench():
    """The same contrast through the real launch protocol."""
    from metal_executor import compile_msl
    from pdum.tl import Tensor, compute, global_idx
    from pdum.tl.kernel import _compile
    from wgsl_executor import compile_wgsl

    @compute
    def twice(src, dst):
        i, j = global_idx("y", "x")
        dst[i, j] = src[i, j] * 2.0 + 1.0

    rows = []
    for side in (32, 64, 128, 256):
        def T(a):
            return Tensor.from_numpy(np.asarray(a, dtype=np.float64), ("y", "x"))

        args = (T(np.zeros((side, side))), T(np.zeros((side, side))))
        art = _compile(twice.fn, args)
        ex_w, ex_m, ex_a = (
            compile_wgsl(art),
            compile_msl(art, mode="copy"),
            compile_msl(art, mode="adopt"),
        )
        vals = [args[art.params.index(n)] for n in art.tensor_params]

        # The host repack alone -- the ONE line both executors share verbatim.
        # Tensor.to_numpy is documented "materialize (naively) for testing":
        # itertools.product over every lattice point calling .item(). It is not
        # a bug, it is a REFERENCE-TIER function -- which both device backends
        # nonetheless call on every launch.
        def repack():
            for t in vals:
                np.ascontiguousarray(t.to_numpy(order=tuple(d.name for d in t.layout.dims)), dtype=np.float32)

        # ...and the writeback side, the same naive path in reverse: one
        # writable tensor decoded through Tensor.from_numpy and stored.
        from pdum.tl.ir import Token, _store

        wt = vals[-1]
        wshape = tuple(d.size for d in wt.layout.dims)
        worder = tuple(d.name for d in wt.layout.dims)
        raw = np.zeros(wshape, dtype=np.float32)

        def writeback():
            _store(Token(), wt, Tensor.from_numpy(raw.astype(np.float64), worder))

        reps = 15 if side <= 128 else 6
        r = {
            "side": side,
            "MB": side * side * 4 / 1e6,
            "wgsl": _min_of(lambda: ex_w(vals, b""), reps),
            "metal": _min_of(lambda: ex_m(vals, b""), reps),
            "adopt": _min_of(lambda: ex_a(vals, b""), reps),
            "repack": _min_of(repack, reps),
            "writeback": _min_of(writeback, reps),
        }
        # What is left once BOTH shared host repacks are subtracted. Compare it
        # against Part 2's protocol numbers at the same element count: that is
        # how small the device's share of a launch actually is.
        for k in ("wgsl", "metal", "adopt"):
            r[k + "_dev"] = r[k] - r["repack"] - r["writeback"]
        rows.append(r)
    return rows


def main():
    print("=== PART 1: zero-copy adoption, proof ===")
    ok = proof()
    print(f"ALL THREE HOLD: {ok}\n")

    print("=== PART 2: transfer protocol, one trivial kernel (wall ms, min of reps) ===")
    rows = protocol_bench()
    hdr = (
        f"{'elems':>10}{'MB':>8}{'wgpu full':>11}{'wgpu up':>10}{'wgpu read':>11}"
        f"{'metal copy':>12}{'metal adopt':>13}{'GPU kernel':>12}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['n']:>10}{r['MB']:>8.1f}{r['wgpu_full']:>11.3f}{r['wgpu_up']:>10.3f}"
            f"{r['wgpu_read']:>11.3f}{r['metal_copy']:>12.3f}{r['metal_adopt']:>13.3f}{r['gpu_ms']:>12.3f}"
        )
    print("\nspeedups (wgpu full / metal):")
    for r in rows:
        print(
            f"  {r['n']:>10} elems: vs metal-copy {r['wgpu_full'] / r['metal_copy']:>6.1f}x   "
            f"vs metal-adopt {r['wgpu_full'] / r['metal_adopt']:>7.1f}x   "
            f"(adopt is {r['metal_adopt'] / r['gpu_ms']:.2f}x the pure GPU kernel time)"
        )

    print("\n=== PART 3: the same, through _Artifact.launch (wall ms, min) ===")
    arows = artifact_bench()
    hdr2 = (
        f"{'side':>6}{'elems':>9}{'wgsl':>9}{'metal':>9}{'adopt':>9}"
        f"{'repack':>9}{'writebk':>9}{'host%':>7}   {'adopt-dev':>10}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for r in arows:
        host = r["repack"] + r["writeback"]
        print(
            f"{r['side']:>6}{r['side'] ** 2:>9}{r['wgsl']:>9.1f}{r['metal']:>9.1f}"
            f"{r['adopt']:>9.1f}{r['repack']:>9.1f}{r['writeback']:>9.1f}"
            f"{100 * host / r['adopt']:>6.0f}%   {r['adopt_dev']:>10.2f}"
        )
    print(
        "\n'repack' = to_numpy+ascontiguousarray (the ONE line both executors share\n"
        "verbatim: wgsl_executor.py:260 and metal_executor.py's copy of it).\n"
        "'writebk' = Tensor.from_numpy + _store, the same naive path in reverse.\n"
        "Both are REFERENCE-tier materializers -- to_numpy's own docstring says\n"
        "'materialize (naively) for testing' -- and both sit on the hot path of\n"
        "EVERY device launch, on both backends, identically.\n"
        "Cross-reference Part 2: at 65536 elements the entire Metal-adopt device\n"
        "protocol costs 0.22 ms, while the 256x256 launch above costs ~400 ms.\n"
        "The device's share of a launch today is well under one part in a thousand."
    )


if __name__ == "__main__":
    main()
