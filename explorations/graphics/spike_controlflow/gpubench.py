"""GPU timing harness for the control-flow spike (210's doctrine).

Timestamp queries are the honest timer: the adapter here (M3 Ultra,
Metal) exposes `timestamp-query`, so we request it at device creation
and read begin/end-of-pass ticks. NEVER time a compute path through a
sync readback (210: the submit->wait->map round-trip is ~1.6 ms of
fixed protocol latency and would swamp every number below).

Estimator: MINIMUM over reps (noise is one-sided).

Each rep is its OWN submit writing its OWN query pair, so passes cannot
pipeline into each other; a single resolve+readback at the end pays the
protocol cost once, outside every measured window.
"""

from __future__ import annotations

import time

import numpy as np
import wgpu

_DEV = None
_HAS_TS = None


def device():
    """The device, with timestamp-query requested when the adapter has it."""
    global _DEV, _HAS_TS
    if _DEV is None:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        feats = set(adapter.features)
        want = [f for f in ("timestamp-query",) if f in feats]
        _HAS_TS = bool(want)
        _DEV = adapter.request_device_sync(required_features=want)
        _DEV._spike_adapter_info = adapter.info
    return _DEV


def has_timestamps() -> bool:
    device()
    return _HAS_TS


class Program:
    """One compute pipeline over a fixed set of storage buffers."""

    def __init__(self, source: str, buffers: list[np.ndarray], label: str = ""):
        self.dev = device()
        self.source = source
        self.label = label
        self.module = self.dev.create_shader_module(code=source)
        self.pipeline = self.dev.create_compute_pipeline(
            layout="auto", compute={"module": self.module, "entry_point": "main"}
        )
        usage = (
            wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
        )
        def _raw(a):  # u32 buffers upload their bits; everything else is f32
            a = np.asarray(a)
            if a.dtype == np.uint32:
                return np.ascontiguousarray(a).tobytes()
            return np.ascontiguousarray(a, np.float32).tobytes()

        self.bufs = [self.dev.create_buffer_with_data(data=_raw(a), usage=usage) for a in buffers]
        self.shapes = [np.asarray(a).shape for a in buffers]
        entries = [
            {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
            for i, b in enumerate(self.bufs)
        ]
        self.bind = self.dev.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0), entries=entries
        )

    def _encode(self, grid, ts=None):
        """The one encode path, shared by timed and untimed runs (210:
        encode and submit are separate acts, and the timed/untimed paths
        must not drift)."""
        enc = self.dev.create_command_encoder()
        kw = {}
        if ts is not None:
            qs, i = ts
            kw["timestamp_writes"] = {
                "query_set": qs,
                "beginning_of_pass_write_index": 2 * i,
                "end_of_pass_write_index": 2 * i + 1,
            }
        cp = enc.begin_compute_pass(**kw)
        cp.set_pipeline(self.pipeline)
        cp.set_bind_group(0, self.bind)
        cp.dispatch_workgroups(*grid)
        cp.end()
        return enc

    def read(self, i: int) -> np.ndarray:
        raw = np.frombuffer(self.dev.queue.read_buffer(self.bufs[i]), dtype=np.float32)
        return raw.reshape(self.shapes[i]).copy()

    def time_ms(self, grid, reps: int = 25, warmup: int = 3) -> dict:
        """Minimum GPU pass duration in ms over `reps`, plus the sample."""
        for _ in range(warmup):
            self.dev.queue.submit([self._encode(grid).finish()])
        self.dev._poll_wait() if hasattr(self.dev, "_poll_wait") else None
        if has_timestamps():
            # MEASURED ARTIFACT (this driver): the LAST pass of a batch always
            # resolves to a zero delta -- its end-of-pass write is not visible
            # to the resolve that trails it. One extra rep is encoded and its
            # pair discarded, so no real sample is ever a truncated one.
            n = reps + 1
            qs = self.dev.create_query_set(type="timestamp", count=2 * n)
            resolve = self.dev.create_buffer(
                size=8 * 2 * n,
                usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC,
            )
            for i in range(n):
                self.dev.queue.submit([self._encode(grid, ts=(qs, i)).finish()])
            enc = self.dev.create_command_encoder()
            enc.resolve_query_set(qs, 0, 2 * n, resolve, 0)
            self.dev.queue.submit([enc.finish()])
            ticks = np.frombuffer(self.dev.queue.read_buffer(resolve), dtype=np.uint64)
            ns = np.maximum(  # drivers may report non-monotonic pass timestamps
                ticks[1::2].astype(np.int64) - ticks[0::2].astype(np.int64), 0
            )
            ms = (ns.astype(np.float64) / 1e6)[:-1]
            med = float(np.median(ms))
            keep = ms[ms > 0.75 * med]  # a sample far BELOW median is a truncated
            dropped = int(ms.size - keep.size)  # write, not a fast run -- never a min
            return {"method": "timestamp", "min": float(keep.min()), "med": med,
                    "n": int(keep.size), "dropped": dropped}
        # fallback: submit -> wait, many iterations, minimum
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            self.dev.queue.submit([self._encode(grid).finish()])
            self.dev.queue.read_buffer(self.bufs[-1], buffer_offset=0, size=4)
            samples.append((time.perf_counter() - t0) * 1e3)
        return {"method": "submit-wait", "min": float(min(samples)),
                "med": float(np.median(samples)), "n": reps}


def warm_gpu(ms: float = 500.0) -> None:
    """Bring the GPU to a steady clock before any measurement.

    MEASURED: without this the first program timed in a process reads
    ~1.7x high (an M3 Ultra idles at a low clock and ramps under load).
    A per-program warmup does not fix it -- the ramp is longer than a few
    dispatches -- so the sweep burns a sustained load first.
    """
    src = """
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  var x: f32 = f32(gid.x) * 1e-6;
  for (var i: i32 = 0; i < 4000; i = i + 1) { x = fma(x, 0.9999, 1e-7); }
  out[gid.x] = x;
}
"""
    p = Program(src, [np.zeros(1 << 18, np.float32)])
    grid = ((1 << 18) // 64, 1, 1)
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < ms:
        for _ in range(20):
            p.dev.queue.submit([p._encode(grid).finish()])
        p.read(0)


def calibrate() -> str:
    """Sanity on the tick->ns conversion, two ways:
    (1) doubling the kernel's work must double the reported duration
        (linear, ~zero intercept);
    (2) at a workload long enough to swamp per-submit host overhead the
        timestamp must approach amortized wall clock from below."""
    src = """
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  var x: f32 = f32(gid.x) * 1e-6;
  for (var i: i32 = 0; i < ITERS; i = i + 1) { x = fma(x, 0.9999, 1e-7); }
  out[gid.x] = x;
}
"""
    out = []
    mins = []
    for iters in (40000, 80000, 160000):
        p = Program(src.replace("ITERS", str(iters)), [np.zeros(1 << 16, np.float32)])
        grid = ((1 << 16) // 64, 1, 1)
        ts = p.time_ms(grid, reps=12)
        mins.append(ts["min"])
        out.append(f"{iters} iters -> {ts['min']:.3f} ms")
    scale = [mins[1] / mins[0], mins[2] / mins[1]]
    p = Program(src.replace("ITERS", "160000"), [np.zeros(1 << 16, np.float32)])
    grid = ((1 << 16) // 64, 1, 1)
    p.dev.queue.submit([p._encode(grid).finish()])
    p.read(0)
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        p.dev.queue.submit([p._encode(grid).finish()])
    p.read(0)
    wall = (time.perf_counter() - t0) * 1e3 / n
    return (
        "calibration: " + "; ".join(out)
        + f"  doubling ratios {scale[0]:.2f}, {scale[1]:.2f} (want ~2.0)"
        + f"; amortized wall {wall:.3f} ms vs timestamp {mins[2]:.3f} ms"
        + f" (host encode overhead = {wall - mins[2]:.3f} ms/submit)"
    )


if __name__ == "__main__":
    d = device()
    print("adapter:", d._spike_adapter_info)
    print("timestamp-query:", has_timestamps())
    print(calibrate())
