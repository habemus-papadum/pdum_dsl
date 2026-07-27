"""The WebGPU CONFORMANCE executor (200 §7, the P8 gate) — translation-only.

This is NOT a performance backend (the Not-doing list): it renders a
kernel REGION to naive WGSL — one thread per lattice point, no tiling,
no fusion — compiles it through wgpu, and executes it over real device
buffers. It plugs into the executor COLUMN built at 240 C5.2:
``dataclasses.replace(artifact, executor=compile_wgsl(artifact))``
reuses the whole launch protocol (staging pack, rebind, the overlap
refusal) and swaps only the backend column behind the same content-key
discipline (`WGPU_FP` is `_EXECUTOR_FP`'s second value).

THE AMBIENT DECISION, resolved here (recorded in memory + the ledger):
``tl.iota`` stays the ambient's spelling. Its translation row IS a
one-line lookup — dim name → global-invocation-id component — because
the alignment law makes every kernel-region iota a coordinate function
of the thread. The unification thesis survives into the device era; no
dedicated ambient op is needed.

Numeric policy (210, both sides): the device computes f32 (WGSL has no
f64); differentials against the f64 reference state their tolerances.
The device does NOT bounds-check reads — the reference refuses OOB, so
any case in-bounds on reference is certified UB-free before it runs
here (the keying-ladder ruling).

Honest skips: a region containing an op with no translation row raises
``Untranslatable`` naming the op; the battery skips with that reason.
"""

from __future__ import annotations

import struct
from dataclasses import replace

import numpy as np
from pdum.tl.dialect import walk_region
from pdum.tl.tensor import Tensor

WGPU_FP = ("wgsl", "wgpu")  # the backend column's second value


class Untranslatable(Exception):
    """This region has no WGSL translation yet — the reason names the op."""


_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_CMP = {"lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "==", "ne": "!="}
_FNS = {"sqrt": "sqrt", "exp": "exp", "log": "log", "tanh": "tanh", "abs": "abs", "floor": "floor"}
_CORE_INFIX = {"core.add": "+", "core.sub": "-", "core.mul": "*", "core.div": "/"}


def _device():
    from pdum.tl.graphics import _device as dev

    return dev()


def _dims_of(node):
    return tuple(d[0] for d in node.type.dims)


def _translate(art) -> tuple[str, dict]:
    """Region -> WGSL compute shader. Returns (source, meta): buffer
    order, per-buffer dims, the writable set, the uniform slot list."""
    region = art.region
    params = list(region.params)
    lattice = params[len(art.tensor_params) - 1] if art.tensor_params else None
    if lattice is None:
        raise Untranslatable("a kernel with no tensor parameters")
    axes = _dims_of(lattice)  # dim name -> gid component, by the writable's order
    if len(axes) > 2:
        raise Untranslatable("rank-3+ lattices (the 2D/1D subset translates today)")
    comp = {name: c for name, c in zip(axes, ("y", "x") if len(axes) == 2 else ("x",))}
    extents = {d[0]: (d[1], d[2]) for d in lattice.type.dims}

    slots = list(art.uniforms)  # (name, offset, fmt) — the kernel's own captures
    for _, _, _, base, plan, _ in art.arg_slots:  # spliced fn-arg blocks
        for s in plan.slots:
            slots.append((f"arg{base}", base + s.dest.offset, s.dest.fmt))
    slot_index = {off: i for i, (_, off, _) in enumerate(sorted(slots, key=lambda s: s[1]))}

    names: dict[int, str] = {}
    lines: list[str] = []
    bools: set[int] = set()
    stores: list[str] = []
    counter = [0]

    def buf_index(p) -> str:
        """The linear index of a buffer at the thread's coordinates —
        row-major over ITS dims, each coordinate the gid component of
        that dim name (the alignment law's gift)."""
        dims = p.type.dims
        strides, acc = [], 1
        for d in reversed(dims):
            strides.append(acc)
            acc *= d[2] - d[1]
        parts = []
        for d, s in zip(dims, reversed(strides)):
            if d[0] not in comp:
                raise Untranslatable(f"a buffer dim {d[0]!r} outside the launch lattice")
            parts.append(f"i32(gid.{comp[d[0]]}) * {s}")
        return " + ".join(parts) or "0"

    def as_f32(nid, expr):
        return f"select(0.0, 1.0, {expr})" if nid in bools else expr

    def go(n):
        if id(n) in names:
            return names[id(n)]
        expr, is_bool = _expr(n)
        var = f"v{counter[0]}"
        counter[0] += 1
        ty = "bool" if is_bool else "f32"
        lines.append(f"  let {var}: {ty} = {expr};")
        names[id(n)] = var
        if is_bool:
            bools.add(id(n))
        return var

    def operand(n) -> str:
        v = go(n)
        return as_f32(id(n), v)

    def _expr(n) -> tuple[str, bool]:
        attrs = dict(n.attrs)
        if n.op == "core.param":
            i = params.index(n)
            return f"buf{i}[{buf_index(n)}]", False
        if n.op == "tl.iota":
            name = attrs["name"]
            if name not in comp:
                raise Untranslatable(f"an iota over dim {name!r} outside the launch lattice")
            return f"f32(gid.{comp[name]})", False  # THE one-line ambient row
        if n.op == "core.const":
            return _lit(attrs["value"]), False
        if n.op == "tl.const":
            return _lit(attrs["value"]), False
        if n.op == "abi.slot":
            return f"U[{slot_index[attrs['offset']]}]", False
        if n.op == "tl.pointwise":
            f = attrs["f"]
            ops = [operand(a) for a in n.args]
            if f in _INFIX:
                return f"({ops[0]} {_INFIX[f]} {ops[1]})", False
            if f == "neg":
                return f"(-{ops[0]})", False
            if f in _CMP:
                return f"({ops[0]} {_CMP[f]} {ops[1]})", True
            if f == "where":
                cond = go(n.args[0])
                c = cond if id(n.args[0]) in bools else f"({as_f32(id(n.args[0]), cond)} != 0.0)"
                return f"select({ops[2]}, {ops[1]}, {c})", False
            if f == "maximum":
                return f"max({ops[0]}, {ops[1]})", False
            if f == "minimum":
                return f"min({ops[0]}, {ops[1]})", False
            if f in _FNS:
                return f"{_FNS[f]}({ops[0]})", False
            raise Untranslatable(f"marker {f!r}")
        if n.op in _CORE_INFIX:
            a, b = (operand(x) for x in n.args)
            return f"({a} {_CORE_INFIX[n.op]} {b})", False
        if n.op == "core.neg":
            return f"(-{operand(n.args[0])})", False
        if n.op == "core.cmp":
            a, b = (operand(x) for x in n.args)
            return f"({a} {_CMP[attrs['pred']]} {b})", True
        if n.op == "core.select":
            c = go(n.args[0])
            cond = c if id(n.args[0]) in bools else f"({as_f32(id(n.args[0]), c)} != 0.0)"
            return f"select({operand(n.args[2])}, {operand(n.args[1])}, {cond})", False
        if n.op.startswith("pw."):
            f = n.op[3:]
            if f in _FNS:
                return f"{_FNS[f]}({operand(n.args[0])})", False
            if f in ("maximum", "minimum"):
                return f"{'max' if f == 'maximum' else 'min'}({operand(n.args[0])}, {operand(n.args[1])})", False
            raise Untranslatable(f"scalar op pw.{f}")
        if n.op == "tl.read":
            tex, *idx = n.args
            if tex.op != "core.param":
                raise Untranslatable("a read of a non-parameter tensor")
            i = params.index(tex)
            dims = tex.type.dims
            strides, acc = [], 1
            for d in reversed(dims):
                strides.append(acc)
                acc *= d[2] - d[1]
            parts = []
            for ix, d, s in zip(idx, dims, reversed(strides)):
                parts.append(f"i32({operand(ix)}) * {s}")  # no bounds check: reference-certified
            return f"buf{i}[{' + '.join(parts)}]", False
        raise Untranslatable(n.op)

    def _lit(v) -> str:
        s = repr(float(v))
        return s if ("." in s or "e" in s or "inf" in s or "nan" in s) else s + ".0"

    # walk once: emit stores in token order; everything else demand-driven
    for n in walk_region(region):
        if n.op == "tl.store":
            _, dst, val = n.args
            if dst.op != "core.param":
                raise Untranslatable("a store into a non-parameter tensor")
            i = params.index(dst)
            stores.append(f"  buf{i}[{buf_index(dst)}] = {operand(val)};")
        elif n.op in ("tl.token", "core.yield"):
            continue

    bindings = []
    for i, p in enumerate(params):
        writable = i < len(art.tensor_params) and art.tensor_params[i] in art.writable
        writable = writable or i >= len(art.tensor_params)  # tap buffers write
        mode = "read_write" if writable else "read"
        bindings.append(f"@group(0) @binding({i}) var<storage, {mode}> buf{i}: array<f32>;")
    if slots:  # layout="auto" prunes unused bindings: U exists only when slots do
        bindings.append(f"@group(0) @binding({len(params)}) var<storage, read> U: array<f32>;")

    wg = "@workgroup_size(8, 8, 1)" if len(axes) == 2 else "@workgroup_size(64, 1, 1)"
    guards = " || ".join(
        f"gid.{comp[a]} >= {extents[a][1] - extents[a][0]}u" for a in axes
    )
    src = "\n".join(
        [
            *bindings,
            f"@compute {wg}",
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
            f"  if ({guards}) {{ return; }}",
            *lines,
            *stores,
            "}",
        ]
    )
    meta = {
        "slots": sorted(slots, key=lambda s: s[1]),
        "axes": axes,
        "extents": extents,
        "n_params": len(params),
    }
    return src, meta


def compile_wgsl(art):
    """The backend column's second executor: (values, staging) -> effect,
    behind the same signature run_region serves. Buffers upload f32,
    writables read back into the tensors' own buffers."""
    source, meta = _translate(art)
    device = _device()
    import wgpu

    module = device.create_shader_module(code=source)
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={"module": module, "entry_point": "main"}
    )

    def executor(values, staging):
        writable = set(art.writable)
        bufs, orders = [], []
        for i, t in enumerate(values):
            order = tuple(d.name for d in t.layout.dims)
            arr = np.ascontiguousarray(t.to_numpy(order=order), dtype=np.float32)
            name = art.tensor_params[i] if i < len(art.tensor_params) else f"tap:{i}"
            usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
            bufs.append(device.create_buffer_with_data(data=arr.tobytes(), usage=usage))
            orders.append((name, order, arr.shape))
        entries = [{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bufs)]
        if meta["slots"]:
            uvals = [struct.unpack_from(fmt, staging, off)[0] for _, off, fmt in meta["slots"]]
            ubuf = device.create_buffer_with_data(
                data=np.asarray(uvals, dtype=np.float32).tobytes(), usage=wgpu.BufferUsage.STORAGE
            )
            entries.append({"binding": len(bufs), "resource": {"buffer": ubuf, "offset": 0, "size": ubuf.size}})
        bind = device.create_bind_group(layout=pipeline.get_bind_group_layout(0), entries=entries)
        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipeline)
        cp.set_bind_group(0, bind)
        axes = meta["axes"]
        ext = [meta["extents"][a][1] - meta["extents"][a][0] for a in axes]
        if len(axes) == 2:
            cp.dispatch_workgroups((ext[1] + 7) // 8, (ext[0] + 7) // 8, 1)
        else:
            cp.dispatch_workgroups((ext[0] + 63) // 64, 1, 1)
        cp.end()
        device.queue.submit([enc.finish()])
        from pdum.tl.ir import Token, _store

        for (name, order, shape), buf, t in zip(orders, bufs, values):
            if name in writable:
                raw = np.frombuffer(device.queue.read_buffer(buf), dtype=np.float32).reshape(shape)
                _store(Token(), t, Tensor.from_numpy(raw.astype(np.float64), order))
        return None

    return executor


def wgpu_artifact(art):
    """The drop-in: the same artifact with only the backend column
    swapped — launch() then runs staging/rebind/overlap identically and
    lands on the device."""
    return replace(art, executor=compile_wgsl(art))
