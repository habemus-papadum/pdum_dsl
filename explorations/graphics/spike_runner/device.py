"""The device, with features we are allowed to ask for.

``pdum.tl.graphics._device`` (graphics.py:718) is a lazy singleton that
creates its device with NO required features:

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    _GPU.append(adapter.request_device_sync())

There is no seam to pass ``required_features``, so 210's "timestamp queries
are the honest GPU timer" is unreachable through the library's own device —
the feature must be requested at creation and cannot be added later. The
spike gets one by creating the device itself and SEEDING the singleton list
before anything else touches it, so textures/samplers (which resolve through
``_device()``) still land on the same device. That seeding is a monkeypatch
into a private module global; it is on the hurt-list.
"""

from __future__ import annotations

TIMESTAMP = "timestamp-query"


def make_device(want_timestamps: bool = True):
    """Create the device (timestamps when the adapter offers them) and make
    it the one ``pdum.tl.graphics`` will hand out."""
    import wgpu

    from pdum.tl import graphics

    if graphics._GPU:  # somebody already forced the featureless singleton
        return graphics._GPU[0], TIMESTAMP in graphics._GPU[0].features

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    feats = [TIMESTAMP] if (want_timestamps and TIMESTAMP in adapter.features) else []
    device = adapter.request_device_sync(required_features=feats)
    graphics._register_wgpu_kinds(wgpu)  # what _device() would have done for us
    graphics._GPU.append(device)  # THE MONKEYPATCH: seed the private singleton
    return device, TIMESTAMP in device.features


class PassTimer:
    """Timestamp queries, cached on the runner (210: 'cache the query set /
    buffer on the program object'). Two writes per pass, resolved into one
    buffer, read once per timed frame. Deltas clamp at >= 0 — drivers may
    report non-monotonic pass timestamps."""

    def __init__(self, device, n_passes: int):
        import wgpu

        self.device = device
        self.n = n_passes
        self.period_ns = 1.0  # wgpu resolves timestamps in nanoseconds
        self.query_set = device.create_query_set(type=wgpu.QueryType.timestamp, count=2 * n_passes)
        self.resolve = device.create_buffer(
            size=8 * 2 * n_passes,
            usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )

    def writes(self, i: int) -> dict:
        return {
            "query_set": self.query_set,
            "beginning_of_pass_write_index": 2 * i,
            "end_of_pass_write_index": 2 * i + 1,
        }

    def resolve_into(self, enc) -> None:
        enc.resolve_query_set(self.query_set, 0, 2 * self.n, self.resolve, 0)

    def read_ms(self) -> list[float]:
        import numpy as np

        raw = np.frombuffer(self.device.queue.read_buffer(self.resolve), dtype=np.uint64)
        out = []
        for i in range(self.n):
            a, b = int(raw[2 * i]), int(raw[2 * i + 1])
            out.append(max(0.0, (b - a) * self.period_ns / 1e6))
        return out
