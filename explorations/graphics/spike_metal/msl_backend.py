"""The MSL BACKEND — the Metal twin of `wgsl_executor._translate`.

This module is IR -> Metal Shading Language source text, and nothing
else. It touches no device, allocates no buffer, and does not know that
Metal has a queue: that is `metal_runtime.py`'s job. The split is the
hypothesis 280 asked this spike to test, and keeping it physically
enforced (this file does not `import Metal`) is how the spike tests it
rather than asserting it.

WHY THIS IS A DELIBERATE THIRD COPY of the marker tables (after
`wgsl_executor._translate` and `wgsl_executor._Gen.expr`): the lead
asked for the duplication as EVIDENCE. So this file is a structural
line-for-line clone of `_translate` -- same function names, same local
names, same walk order, same comments where the logic is unchanged --
so that `rowdiff.py` can diff the two mechanically and report which
rows are genuinely target-specific. The answer (see rowdiff output) is
that the op table is ~93% identical and the differences collapse to
FIVE parameters. That number is the finding; the copy is how it was
measured, not how it should ship.

THE FIVE THINGS THAT ACTUALLY DIFFER (everything else is byte-identical
modulo those):

  1. float literals need an `f` suffix (`1.5f`) -- MSL has no `double`,
     and an unsuffixed literal is a portability trap, not an error.
  2. the numeric casts are spelled `float(x)`/`int(x)`, not
     `f32(x)`/`i32(x)`.
  3. `let v: f32 = e;` is `float v = e;` -- a C declaration, not a WGSL
     `let` binding.
  4. buffers are ENTRY-POINT PARAMETERS with `[[buffer(i)]]`, not
     module-scope `@group/@binding` declarations; read-only is spelled
     by the C qualifier `const device` rather than the address-space
     access mode `var<storage, read>`.
  5. the entry point cannot be called `main` (MSL is a C++ dialect and
     `main` is reserved) and carries no workgroup size -- see below.

EVERY OPERATOR ROW IS IDENTICAL. `select(f, t, cond)` has the same
argument order and the same semantics in both languages; `max`, `min`,
`sqrt`, `exp`, `log`, `tanh`, `abs`, `floor`, `sin`, `cos` are spelled
the same; the infix and comparison tables are the same; the bool->f32
widening (`select(0.0, 1.0, b)`) is the same. That is not luck --
both languages are C-family shading languages over the same IEEE f32
discipline that 210 imposes on both sides of every differential.

THE ONE ROW THAT MOVED ACROSS THE RUNTIME/BACKEND LINE -- the finding
280 wants. In WGSL the workgroup size is `@workgroup_size(8, 8, 1)`,
part of the SHADER TEXT, so it is backend output. In Metal the
threadgroup size is an argument to `dispatchThreadgroups:` -- pure
runtime data, absent from the source. 210 records the WebGPU-era rule
as "workgroup size is pipeline-creation-time; dispatch dimensions are
launcher data"; Metal splits that sentence differently. So this module
still CHOOSES the threadgroup size (it is a codegen-adjacent policy
decision that wants to see the loop structure) but must hand it to the
runtime through `meta`, rather than baking it into the text. A backend
that returns only `str` cannot express Metal.

AND THE GUARD, which is the same finding from the other side. The
`if (gid.x >= N) return;` prologue exists in the WGSL path because
WebGPU can only dispatch whole workgroups, so the last one overhangs.
Metal has `dispatchThreads:threadsPerThreadgroup:` (non-uniform
threadgroups), which launches EXACTLY the requested grid and makes the
guard dead code. So a line of emitted SOURCE is or is not needed
depending on which RUNTIME entry point the launcher will call.
`exact_grid=True` drops it; `differential.py` runs both and checks they
agree bitwise. A clean backend/runtime split cannot decide this on
either side alone -- it is a negotiated pair, and that is the sharpest
place our two definitions failed to carve.
"""

from __future__ import annotations

from pdum.tl.dialect import walk_region

METAL_FP = ("msl", "metal")  # this backend column's value, parallel to WGPU_FP


class Untranslatable(Exception):
    """This region has no MSL translation yet -- the reason names the op."""


# --- the marker tables: byte-identical to wgsl_executor's, deliberately ------
_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_CMP = {"lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "==", "ne": "!="}
_FNS = {f: f for f in ("sqrt", "exp", "log", "tanh", "abs", "floor", "sin", "cos")}
_CORE_INFIX = {"core.add": "+", "core.sub": "-", "core.mul": "*", "core.div": "/"}


def _dims_of(node):
    return tuple(d.name for d in node.type.dims)


def _translate(art, *, exact_grid: bool = False) -> tuple[str, dict]:
    """Region -> MSL compute shader. Returns (source, meta): buffer
    order, per-buffer dims, the writable set, the uniform slot list.

    `exact_grid` drops the bounds guard, which is sound only if the
    launcher dispatches with `dispatchThreads:` (exact grid) rather
    than `dispatchThreadgroups:` (whole workgroups).
    """
    region = art.region
    params = list(region.params)
    lattice = params[len(art.tensor_params) - 1] if art.tensor_params else None
    if lattice is None:
        raise Untranslatable("a kernel with no tensor parameters")
    axes = _dims_of(lattice)  # dim name -> gid component, by the writable's order
    if len(axes) > 2:
        raise Untranslatable("rank-3+ lattices (the 2D/1D subset translates today)")
    comp = {name: c for name, c in zip(axes, ("y", "x") if len(axes) == 2 else ("x",))}
    extents = {d.name: (d.start, d.stop) for d in lattice.type.dims}

    slots = list(art.uniforms)  # (name, offset, fmt) -- the kernel's own captures
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
        """The linear index of a buffer at the thread's coordinates --
        row-major over ITS dims, each coordinate the gid component of
        that dim name (the alignment law's gift)."""
        dims = p.type.dims
        strides, acc = [], 1
        for d in reversed(dims):
            strides.append(acc)
            acc *= d.size
        parts = []
        for d, s in zip(dims, reversed(strides)):
            if d.name not in comp:
                raise Untranslatable(f"a buffer dim {d.name!r} outside the launch lattice")
            parts.append(f"int(gid.{comp[d.name]}) * {s}")  # DIFF 2: i32() -> int()
        return " + ".join(parts) or "0"

    def as_f32(nid, expr):
        return f"select(0.0f, 1.0f, {expr})" if nid in bools else expr  # DIFF 1 only

    def go(n):
        if id(n) in names:
            return names[id(n)]
        expr, is_bool = _expr(n)
        var = f"v{counter[0]}"
        counter[0] += 1
        ty = "bool" if is_bool else "float"  # DIFF 3: f32 -> float
        lines.append(f"  {ty} {var} = {expr};")  # DIFF 3: C decl, not `let x: T =`
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
            return f"float(gid.{comp[name]})", False  # THE one-line ambient row (DIFF 2)
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
                c = cond if id(n.args[0]) in bools else f"({as_f32(id(n.args[0]), cond)} != 0.0f)"
                return f"select({ops[2]}, {ops[1]}, {c})", False  # SAME arg order as WGSL
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
            cond = c if id(n.args[0]) in bools else f"({as_f32(id(n.args[0]), c)} != 0.0f)"
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
                acc *= d.size
            parts = []
            for ix, d, s in zip(idx, dims, reversed(strides)):
                parts.append(f"int({operand(ix)}) * {s}")  # no bounds check: reference-certified
            return f"buf{i}[{' + '.join(parts)}]", False
        raise Untranslatable(n.op)

    def _lit(v) -> str:
        """DIFF 1. MSL has no `double`; an unsuffixed literal is a
        portability trap. inf/nan reach here only if the reference let
        them through -- 210 says those refuse at rendering, and they
        refuse identically on both targets (`repr(inf)` is not a
        literal in either language)."""
        s = repr(float(v))
        if "inf" in s or "nan" in s:
            raise Untranslatable(f"a non-finite literal {s!r} (210: refuses at rendering)")
        return (s if ("." in s or "e" in s) else s + ".0") + "f"

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

    # DIFF 4: buffers are ENTRY PARAMETERS, not module-scope declarations.
    # Note the writable computation itself is verbatim from the WGSL path --
    # it is artifact reasoning, not target reasoning.
    bindings = []
    for i, p in enumerate(params):
        writable = i < len(art.tensor_params) and art.tensor_params[i] in art.writable
        writable = writable or i >= len(art.tensor_params)  # tap buffers write
        qual = "device" if writable else "const device"
        bindings.append(f"    {qual} float* buf{i} [[buffer({i})]],")
    if slots:
        bindings.append(f"    const device float* U [[buffer({len(params)})]],")

    # DIFF 5 / THE SEAM FINDING: the threadgroup size is chosen HERE (it is a
    # codegen-adjacent policy) but does not appear in the source -- it rides
    # `meta` to the runtime, which passes it to dispatchThreadgroups:.
    tg = (8, 8, 1) if len(axes) == 2 else (64, 1, 1)
    guards = " || ".join(
        f"gid.{comp[a]} >= {extents[a][1] - extents[a][0]}u" for a in axes
    )
    body_guard = [] if exact_grid else [f"  if ({guards}) {{ return; }}"]
    src = "\n".join(
        [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "kernel void main0(",  # DIFF 5: `main` is reserved in MSL
            *bindings,
            "    uint3 gid [[thread_position_in_grid]])",
            "{",
            *body_guard,
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
        "threadgroup": tg,  # NOT in the source -- the runtime needs it
        "exact_grid": exact_grid,  # whether the guard was omitted
        "entry": "main0",
    }
    return src, meta
