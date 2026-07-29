"""The WGSL column — the compute half (render/encodable is next).

Two halves that meet only at a string: ``generate`` is IR → source text
plus the launch contract and touches no device; the runtime acquires an
adapter, builds a pipeline, and launches. They are separable by
construction (the same WGSL would serve a Dawn harness), which is what
``Pair`` says and what this file's import graph enforces — ``wgpu``
imports LAZILY, inside the acquirer and the compiler, so importing
``pdum.rt`` on a machine with no adapter is safe.

The shell is all that is target-specific: bindings are module-scope
``@group/@binding`` declarations emitted FROM the contract's binding
table (never implicit in text, 210), the thread size lands IN the
source as ``@workgroup_size``, and the guard is live code because
WebGPU dispatches whole workgroups — ``guard="emitted"``. Every
statement between the braces comes from ``emit`` unchanged.
"""

from __future__ import annotations

import numpy as np

from .. import emit, staging, transfer
from ..contract import LaunchContract
from ..registry import register_acquirer, register_compiler
from ..select import WgpuRuntime, webgpu

# The feature vocabulary is OURS, mapped per column: "timestamps" must be
# requested at device creation here (unreachable afterwards — the lesson of
# the old graphics._device() singleton) and is free on Metal.
FEATURES = {"timestamps": "timestamp-query"}


def _lower(art, thread_size):
    d = emit.WGSL
    rows = emit.compute_rows(art, d)
    ts = tuple(thread_size) if thread_size else emit.default_thread_size(rows)
    bindings = emit.compute_bindings(rows)
    decls = [
        f"@group(0) @binding({b.index}) var<storage, {'read_write' if b.writable else 'read'}> "
        f"{b.name}: array<{d.fty}>;"
        for b in bindings
    ]
    source = "\n".join(
        [
            *decls,
            f"@compute @workgroup_size({', '.join(str(n) for n in ts)})",
            f"fn main(@builtin(global_invocation_id) gid: {d.vec(3, 'u32')}) {{",
            f"  if ({rows.guard_expr()}) {{ return; }}",
            *rows.lines,
            *rows.stores,
            "}",
        ]
    )
    contract = LaunchContract(
        thread_size=ts,
        guard="emitted",  # whole workgroups: the overhang guard above is live code
        bindings=(bindings,),  # one stage, one table
        slots=staging.device_slots(rows.slots),
        math=rows.math,
    )
    return source, contract, rows


def generate(art, thread_size: tuple[int, ...] | None = None) -> tuple[str, LaunchContract]:
    """A kernel artifact -> (WGSL compute source, launch contract). A
    backend returns both; source alone cannot express a target (210)."""
    source, contract, _rows = _lower(art, thread_size)
    return source, contract


def acquire(features: tuple[str, ...] = ()):
    """The device, with features requested AT CREATION."""
    import wgpu

    required = []
    for name in features:
        if name not in FEATURES:
            raise ValueError(f"unknown feature {name!r} — this column knows {', '.join(sorted(FEATURES))}")
        required.append(FEATURES[name])
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    missing = [f for f in required if f not in adapter.features]
    if missing:
        raise RuntimeError(
            f"this adapter does not offer {', '.join(missing)} — features are creation-time and cannot be added later"
        )
    return adapter.request_device_sync(required_features=required)


def compile_compute(art, device, thread_size: tuple[int, ...] | None = None):
    """Pipeline once, launch many. The executor carries the launch
    protocol's shape — ``(values, staging) -> effect`` — so the kernel
    tier's launcher runs staging/rebind/overlap identically and lands on
    the device."""
    import wgpu

    source, contract, rows = _lower(art, thread_size)
    module = device.create_shader_module(code=source)
    pipeline = device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "main"})
    names = transfer.value_names(art)
    bound, host_slots = rows.bound, rows.slots  # binding slot k carries values[bound[k]]
    groups = _groups(rows.threads(), contract.thread_size)
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC

    def executor(values, staging_bytes):
        bufs, shapes = [], []
        for i in bound:
            arr = transfer.host_f32(values[i])
            shapes.append(arr.shape)
            bufs.append(device.create_buffer_with_data(data=arr.tobytes(), usage=usage))
        entries = [{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)]
        if host_slots:
            # The DECLARED narrowing (staging.py): an integer slot outside f32's
            # exact range refuses here instead of arriving silently wrong.
            data = staging.pack_device(host_slots, staging_bytes).tobytes()
            ubuf = device.create_buffer_with_data(data=data, usage=wgpu.BufferUsage.STORAGE)
            entries.append({"binding": len(bufs), "resource": {"buffer": ubuf, "offset": 0, "size": ubuf.size}})
        bind = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)
        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bind)
        cp.dispatch_workgroups(*groups)
        cp.end()
        device.queue.submit([enc.finish()])
        for i, buf, shape in zip(bound, bufs, shapes):
            if names[i] in art.writable:
                raw = np.frombuffer(device.queue.read_buffer(buf), dtype=np.float32).reshape(shape)
                transfer.writeback(values[i], raw)
        return None

    executor.source = source  # what ran, and (via the contract) what it substituted
    executor.contract = contract
    return executor


def _groups(threads: tuple[int, ...], thread_size: tuple[int, ...]) -> tuple[int, ...]:
    """Whole-workgroup rounding — LAUNCHER data, and the reason the guard
    exists. Metal does not round at all (``dispatchThreads:``)."""
    return tuple((g + t - 1) // t for g, t in zip(threads, thread_size))


register_acquirer(WgpuRuntime, acquire)
register_compiler(webgpu, compile_compute)
