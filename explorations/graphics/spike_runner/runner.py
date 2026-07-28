"""The encodable factoring: compile ONCE, encode EVERY frame.

The split this spike is about:

  COMPILE-ONCE   lower the kernel region and the PSO, generate WGSL, create
                 shader modules, pipelines, the bind-group layout, the bind
                 group, and every persistent buffer (resident geometry + the
                 slot-channel staging buffers).

  PER-FRAME      refresh staging bytes (one ``queue.write_buffer`` per
                 changed slot buffer), encode passes into a caller-supplied
                 command encoder, and let the HOST submit.

``FrameRunner.encode`` is the single encode path 210 asks for: the offscreen
verifier and the windowed presenter both call it and differ only in the
texture view they hand it and in who submits.

Nothing here re-lowers, re-translates, re-creates a pipeline, or re-uploads
geometry after construction. The per-frame device traffic is exactly the
scalar slot bytes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
from residency import Residency
from wgsl_gen import gen_compute, gen_render


# --- reaching into the kernel tier ------------------------------------------
# There is NO public way to obtain a compiled artifact. ``ComputeKernel``
# exposes ``__call__`` (compile-if-needed AND launch) and ``taps()``; the
# artifact itself never escapes ``_invoke`` (kernel.py:177-198). To compile
# once and launch many times we rebuild the cache key by hand out of four
# private helpers and the private Memo. Every name below is an underscore.
def artifact_of(kernel, *args, tap_names=()):
    from pdum.tl.kernel import KERNELS, _arg_fp, _code_fp, _compile, _env_fp

    key = (_code_fp(kernel.fn), _env_fp(kernel.fn), tuple(_arg_fp(a) for a in args), tuple(tap_names))
    return KERNELS.get_or_compile(key, lambda: _compile(kernel.fn, args, tuple(tap_names), None))


def kernel_staging(art) -> bytes:
    """The staging bytes for the CURRENT captured environment.

    A verbatim lift of the staging block inside ``_Artifact.launch``
    (kernel.py:1100-1111). It is fused into launch, so a runner that wants
    the bytes without a launch has to duplicate it — including the ``wrap``
    replay and the fn-arg block packing."""
    from pdum.dsl.pack import pack_into
    from pdum.tl.producer import _captured

    if not art.staging_size:
        return b""
    staging = bytearray(art.staging_size)
    env = _captured(art.fn) if art.uniforms else {}
    for name, off, fmt in art.uniforms:
        struct.pack_into(fmt, staging, off, env[name])
    for pname, fixed, wrap, base, plan, extract in art.arg_slots:
        if pname is not None:
            raise NotImplementedError("spike: fn-valued kernel params are not rebound per frame")
        f = wrap(fixed) if wrap is not None else fixed
        pack_into(plan, memoryview(staging)[base : base + plan.staging_size], extract(f.captures, ()))
    return bytes(staging)


def slot_values(staging: bytes, slots) -> np.ndarray:
    """Staging bytes -> the f32 array the shader's ``U`` buffer wants.

    Note what this is: the slot channel is already physical (offsets + struct
    formats), but every consumer re-derives the same unpack loop, and the
    device-side representation (a flat ``array<f32>``, one slot per element)
    is invented independently by each backend rather than coming from the
    plan. 210 says the plan IS the ABI; today the plan stops at the host."""
    return np.asarray([struct.unpack_from(f, staging, o)[0] for o, f in slots], dtype=np.float32)


@dataclass
class _SlotBuffer:
    gpu: object
    slots: tuple
    binding: int
    writes: int = 0

    def refresh(self, device, staging: bytes) -> None:
        vals = slot_values(staging, self.slots)
        device.queue.write_buffer(self.gpu, 0, vals.tobytes())
        self.writes += 1


def _slot_buffer(device, slots, binding, n_min=1):
    import wgpu

    size = max(4 * len(slots), 4 * n_min)
    gpu = device.create_buffer(size=size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    return _SlotBuffer(gpu=gpu, slots=tuple(slots), binding=binding)


class CompiledCompute:
    """A @compute kernel, compiled once against resident buffers."""

    def __init__(self, device, res: Residency, kernel, args, labels=None):
        self.device, self.res = device, res
        self.art = artifact_of(kernel, *args)
        bound = dict(zip(self.art.params, args))
        self.values = [bound[n] for n in self.art.tensor_params]
        labels = labels or list(self.art.tensor_params)
        self.prog = gen_compute(self.art, self.values, res, labels)
        module = device.create_shader_module(code=self.prog.source)
        self.pipeline = device.create_compute_pipeline(
            layout="auto", compute={"module": module, "entry_point": "main"}
        )
        entries = [
            {"binding": b.binding, "resource": {"buffer": b.resident.gpu, "offset": 0, "size": b.resident.gpu.size}}
            for b in self.prog.buffers
        ]
        self.slot_buf = None
        if self.prog.slot_binding is not None:
            self.slot_buf = _slot_buffer(device, self.prog.slots, self.prog.slot_binding)
            entries.append(
                {
                    "binding": self.prog.slot_binding,
                    "resource": {"buffer": self.slot_buf.gpu, "offset": 0, "size": self.slot_buf.gpu.size},
                }
            )
        self.bind_group = device.create_bind_group(layout=self.pipeline.get_bind_group_layout(0), entries=entries)
        self.refresh()

    def refresh(self) -> int:
        """Per-frame host work: re-extract the captured scalars and write
        them. Returns bytes written."""
        if self.slot_buf is None:
            return 0
        staging = kernel_staging(self.art)
        self.slot_buf.refresh(self.device, staging)
        return 4 * len(self.slot_buf.slots)

    def encode(self, enc, timestamp_writes=None) -> None:
        kw = {"timestamp_writes": timestamp_writes} if timestamp_writes else {}
        cp = enc.begin_compute_pass(**kw)
        cp.set_pipeline(self.pipeline)
        cp.set_bind_group(0, self.bind_group)
        cp.dispatch_workgroups(*self.prog.workgroups)
        cp.end()


class CompiledRender:
    """A PSO, compiled once for one target format, drawing resident geometry."""

    def __init__(self, device, res: Residency, pso, vs_args, fs_args, target_format="r32float"):
        import wgpu

        self.device, self.res = device, res
        self.target_format = target_format
        self.prog = gen_render(pso, vs_args, fs_args, res, target=target_format)
        self._vs_fn, self._fs_fn = pso.vs.fn, pso.fs.fn
        module = device.create_shader_module(code=self.prog.source)
        self.pipeline = device.create_render_pipeline(
            layout="auto",
            vertex={"module": module, "entry_point": "vs_main", "buffers": []},
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            fragment={
                "module": module,
                "entry_point": "fs_main",
                "targets": [{"format": target_format}],
            },
        )
        entries = [
            {"binding": b.binding, "resource": {"buffer": b.resident.gpu, "offset": 0, "size": b.resident.gpu.size}}
            for b in self.prog.buffers
        ]
        self.v_slot_buf = self.f_slot_buf = None
        if self.prog.v_slot_binding is not None:
            self.v_slot_buf = _slot_buffer(device, self.prog.v_slots, self.prog.v_slot_binding)
            entries.append(
                {
                    "binding": self.prog.v_slot_binding,
                    "resource": {"buffer": self.v_slot_buf.gpu, "offset": 0, "size": self.v_slot_buf.gpu.size},
                }
            )
        if self.prog.f_slot_binding is not None:
            self.f_slot_buf = _slot_buffer(device, self.prog.f_slots, self.prog.f_slot_binding)
            entries.append(
                {
                    "binding": self.prog.f_slot_binding,
                    "resource": {"buffer": self.f_slot_buf.gpu, "offset": 0, "size": self.f_slot_buf.gpu.size},
                }
            )
        for i, (t, s) in enumerate(self.prog.tex_pairs):
            entries.append({"binding": self.prog.tex_base + 2 * i, "resource": t.create_view()})
            entries.append({"binding": self.prog.tex_base + 2 * i + 1, "resource": s})
        self.bind_group = (
            device.create_bind_group(layout=self.pipeline.get_bind_group_layout(0), entries=entries)
            if entries
            else None
        )
        self.refresh_vertex(pso.vs.fn)
        self.refresh_fragment(fs_args)

    # -- the per-frame uniform path -------------------------------------------
    def refresh_vertex(self, vs_fn) -> int:
        """Refresh the vertex stage's captured scalars from ``vs_fn``'s
        CURRENT closure.

        ``spun(angle)`` mints a new closure per angle, so the frame loop hands
        us a different function object than the one we compiled. That is fine
        — the angle rides the slot channel and never enters the IR — but
        NOTHING in the library will tell us so. ``_env_staging`` will happily
        pack a structurally different closure into our layout, so the guard
        below is ours to write: same code fingerprint, same environment
        fingerprint, same uniform layout."""
        from pdum.tl.graphics import _env_staging

        if self.v_slot_buf is None:
            return 0
        self._check_swap(vs_fn)
        staging = _env_staging(self.prog.v_ctx, vs_fn)
        self.v_slot_buf.refresh(self.device, staging)
        return 4 * len(self.v_slot_buf.slots)

    def _check_swap(self, vs_fn) -> None:
        from pdum.tl.kernel import _code_fp, _env_fp

        if vs_fn is self._vs_fn:
            return
        a = (_code_fp(self._vs_fn), _env_fp(self._vs_fn))
        b = (_code_fp(vs_fn), _env_fp(vs_fn))
        if a != b:
            raise ValueError(
                f"this pipeline was compiled for {self._vs_fn.__qualname__} with environment {a[1]}; "
                f"the frame presents {b[1]} — a different specialization needs a different pipeline"
            )

    def refresh_fragment(self, fs_args) -> int:
        """Repack the fragment's slot block. Same duplication as
        ``kernel_staging``: the packing loop lives inside ``render``
        (graphics.py:608-621) and inside ``render_wgpu``
        (wgsl_executor.py:475-489); this is its third copy."""
        from pdum.dsl.pack import pack_into
        from pdum.tl.producer import _captured

        if self.f_slot_buf is None:
            return 0
        f_ctx = self.prog.f_ctx
        staging = bytearray(f_ctx["k.uniform_size"])
        bound = dict(zip(self.prog.fparams, fs_args))
        env = _captured(self._fs_fn) if f_ctx["k.uniforms"] else {}
        for name, nd in f_ctx["k.uniforms"].items():
            at = dict(nd.attrs)
            struct.pack_into(at["fmt"], staging, at["offset"], env[name])
        for blk in f_ctx["k.arg_plans"].values():
            f = bound[blk["pname"]] if blk["pname"] is not None else blk["fixed"]
            if blk["wrap"] is not None:
                f = blk["wrap"](f)
            view = memoryview(staging)[blk["base"] : blk["base"] + blk["plan"].staging_size]
            pack_into(blk["plan"], view, blk["extract"](f.captures, ()))
        self.f_slot_buf.refresh(self.device, bytes(staging))
        return 4 * len(self.f_slot_buf.slots)

    def encode(self, enc, color_view, clear=(0.0, 0.0, 0.0, 1.0), timestamp_writes=None) -> None:
        import wgpu

        kw = {"timestamp_writes": timestamp_writes} if timestamp_writes else {}
        rp = enc.begin_render_pass(
            color_attachments=[
                {
                    "view": color_view,
                    "clear_value": clear,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            **kw,
        )
        rp.set_pipeline(self.pipeline)
        if self.bind_group is not None:
            rp.set_bind_group(0, self.bind_group)
        rp.draw(self.prog.draw_count)
        rp.end()


class FrameRunner:
    """compute-then-draw, compiled once. THE shared encode path."""

    def __init__(self, device, res, compute: CompiledCompute, render: CompiledRender):
        self.device, self.res = device, res
        self.compute, self.render = compute, render
        self.uniform_bytes_last = 0

    def update(self, *, vs_fn=None) -> int:
        """Per-frame HOST work: everything that must happen before encode.
        No device queue traffic except the slot writes."""
        n = self.compute.refresh()
        if vs_fn is not None:
            n += self.render.refresh_vertex(vs_fn)
        self.uniform_bytes_last = n
        return n

    def encode(self, enc, color_view, clear=(0.0, 0.0, 0.0, 1.0), timer=None) -> None:
        """ONE encode path. Offscreen and windowed callers differ only in the
        view they pass and in what they do with the encoder afterwards."""
        self.compute.encode(enc, timestamp_writes=timer.writes(0) if timer else None)
        self.render.encode(enc, color_view, clear=clear, timestamp_writes=timer.writes(1) if timer else None)
