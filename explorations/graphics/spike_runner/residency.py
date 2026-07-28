"""Residency, faked: ONE device buffer per host ``Buffer``.

This is the layer L2 will own and that does not exist today. The facts it
needs are all present in the Tensor already — ``tensor.buffer`` is a shared
identity across views, ``layout.offset`` is a byte offset into it, and each
``Dim.stride`` is a byte stride — but nothing between the Tensor and the
device consumes them: every executor path goes through ``to_numpy()`` +
``ascontiguousarray``, which flattens a view into a fresh contiguous host
array and loses the buffer identity that made it a view.

The one real conversion this layer performs is the ENCODING step 210 calls
for: the host interior is f64, WGSL has no f64, so a device buffer is the
host buffer reinterpreted elementwise f64 -> f32. Element INDICES survive
that change unaltered (both sides scale by their own itemsize), so a host
byte stride of 16 over f64 is a device element stride of 2 over f32 — which
is why a strided field view can be read on the device without repacking.

Spike limit, stated: every buffer here is uniformly f64 (plain f64 tensors
and all-f64 record dtypes). A mixed-dtype record would need a real encoding
descriptor per field; there isn't one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOST_ITEM = 8  # f64 — the reference interior itemsize (210's numeric policy)
DEV_ITEM = 4  # f32 — WGSL's widest float


@dataclass
class Resident:
    """One host Buffer's device twin, plus the copy ledger for it."""

    gpu: object  # wgpu.GPUBuffer
    n_f32: int
    label: str
    uploads: int = 0  # host -> device copies since creation
    downloads: int = 0  # device -> host copies since creation


def dev_elems(host_buffer) -> int:
    if host_buffer.nbytes % HOST_ITEM:
        raise ValueError(f"{host_buffer!r}: not a whole number of f64 elements")
    return host_buffer.nbytes // HOST_ITEM


def host_f32(host_buffer) -> np.ndarray:
    """The whole buffer, reinterpreted f64 -> f32. Structured dtypes are
    irrelevant here: the buffer is featureless bytes and every field is f64,
    so the elementwise narrowing is well defined over the whole extent."""
    if host_buffer.data is None:
        raise RuntimeError(f"{host_buffer!r} is not host-readable")
    return np.frombuffer(host_buffer.data, dtype=np.float64).astype(np.float32)


def index_expr(view, coords: dict) -> str:
    """The device ELEMENT index of ``view`` at ``coords``, as WGSL.

    ``coords`` maps each of the view's dim names to either a Python int (a
    selected constant) or a WGSL i32 expression string. This is the whole
    residency payoff: the expression is built from the view's OWN layout, so
    a strided field view over a record buffer reads in place.
    """
    off = view.layout.offset
    if off % HOST_ITEM:
        raise ValueError(f"layout offset {off} is not f64-aligned")
    const = off // HOST_ITEM
    parts: list[str] = []
    for d in view.layout.dims:
        if d.stride % HOST_ITEM:
            raise ValueError(f"dim {d.name}: byte stride {d.stride} is not f64-aligned")
        step = d.stride // HOST_ITEM
        if d.name not in coords:
            raise KeyError(f"no coordinate for dim {d.name!r}")
        c = coords[d.name]
        if isinstance(c, int):
            const += (c - d.start) * step
            continue
        base = c if d.start == 0 else f"({c} - {d.start})"
        parts.append(f"({base})" if step == 1 else f"({base}) * {step}")
    if const or not parts:
        parts.append(str(const))
    return " + ".join(parts)


class Residency:
    """The registry: host Buffer identity -> one resident device buffer.

    Keyed by ``id(buffer)`` because ``Buffer`` is ``eq=False`` (identity IS
    the key, which is what we want) but is not hashable as a dataclass with
    a memoryview field.
    """

    def __init__(self, device):
        self.device = device
        self._by_id: dict[int, Resident] = {}
        self._keepalive: list = []

    def resident(self, tensor, label: str = "") -> Resident:
        """Get (creating and uploading once) the device twin of the buffer
        this tensor views. The SECOND view of the same buffer is a hit — no
        second upload, which is exactly what the current executors cannot do."""
        buf = tensor.buffer
        key = id(buf)
        got = self._by_id.get(key)
        if got is not None:
            return got
        import wgpu

        data = host_f32(buf)
        gpu = self.device.create_buffer_with_data(
            data=data.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )
        res = Resident(gpu=gpu, n_f32=data.size, label=label or f"buf@{key:x}")
        res.uploads += 1
        self._by_id[key] = res
        self._keepalive.append(buf)  # the registry must outlive nothing; keep ids stable
        return res

    def readback(self, tensor) -> np.ndarray:
        """Device -> host, whole buffer, as f32 elements. Counted."""
        res = self.resident(tensor)
        res.downloads += 1
        raw = self.device.queue.read_buffer(res.gpu)
        return np.frombuffer(raw, dtype=np.float32)

    def view_from_readback(self, tensor) -> np.ndarray:
        """Read one VIEW back out of its resident buffer, honoring strides —
        the host-side mirror of ``index_expr``."""
        flat = self.readback(tensor)
        idx = np.zeros(1, dtype=np.int64)
        idx[0] = tensor.layout.offset // HOST_ITEM
        for d in tensor.layout.dims:
            step = d.stride // HOST_ITEM
            axis = np.arange(d.size, dtype=np.int64) * step
            idx = (idx[..., None] + axis).reshape(-1) if idx.size else axis
        return flat[idx]

    def copy_ledger(self) -> dict:
        return {
            r.label: {"uploads": r.uploads, "downloads": r.downloads, "f32_elems": r.n_f32}
            for r in self._by_id.values()
        }

    def reset_ledger(self) -> None:
        for r in self._by_id.values():
            r.uploads = r.downloads = 0
