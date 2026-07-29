"""Emission-time if-RECONSTRUCTION over our IR.

Our splicer flattens `core.if` to `core.select` / `tl.pointwise f="where"`
(kernel.py `_splice_fn`, the if-as-select row). This pass runs at WGSL
emission time and puts the `if` back where it pays -- WITHOUT the IR ever
growing a branch node, so typing, autodiff and the straight-line analyses
keep the flat form they were designed for.

THE ANALYSIS (exclusive use / arm dominance). A node may sink into arm
`slot` of select S exactly when every use of it is either S-at-that-slot
or another node already sunk into the same arm. That is dominance by the
(S -> arm) edge, computed as a monotone fixpoint over the region DAG.
Sinking is sound here because the value dialect is PURE and floats are
IEEE NON-TRAPPING (210): the not-taken arm's value was always discarded,
so declining to compute it changes nothing. It cannot even change NaN
behaviour, because a NaN in the dead arm never reached the result.

THE SCOPE TREE. Each (select, arm) that sinks anything becomes a scope
whose parent is the select's own scope; nested selects nest their scopes.
Every node is assigned its DEEPEST exclusive scope. The soundness
invariant -- an operand's scope is always an ancestor-or-self of its
consumer's scope -- is asserted, not assumed.

EMISSION reuses `wgsl_executor._Gen` unchanged for expressions, so the
marker tables cannot drift between the flat and reconstructed backends;
only line PLACEMENT differs. `recon_artifact` swaps this translator into
`wgsl_executor.compile_wgsl` so buffers, uniforms and readback are the
existing code path verbatim and the shader source is the only variable.
"""

from __future__ import annotations

from contextlib import contextmanager

import _paths  # noqa: F401

import wgsl_executor as flat
from pdum.tl.dialect import walk_region
from wgsl_executor import Untranslatable

ROOT = 0


def is_select(n) -> bool:
    return n.op == "core.select" or (
        n.op == "tl.pointwise" and dict(n.attrs).get("f") == "where"
    )


# --- the analysis ------------------------------------------------------------


def live_set(stores, leaf):
    """The nodes the EXISTING emitter actually materializes, discovered by
    running it as a probe.

    SHARP EDGE (measured): `args` is not the same as "operands the backend
    reads". `tl.iota` carries the lattice tensor as an arg but emits
    `f32(gid.x)` and never touches it, so a naive walk over `args` drags
    the output buffer into the DAG and the reconstructed shader grows a
    dead `buf0[...]` read the flat one does not have. Asking the real
    generator what it named keeps the two translations over the SAME graph
    -- the only way `recon == flat` bitwise is a meaningful claim.
    """
    probe = flat._Gen(leaf)
    for n in stores:
        probe.operand(n.args[2])
    return set(probe.names)


def _topo(roots, live=None):
    """Post-order over the value DAG: every operand precedes its consumer."""
    seen, order = set(), []

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for a in n.args:
            if live is None or id(a) in live:
                visit(a)
        order.append(n)

    for r in roots:
        visit(r)
    return order


def _users(order, roots, live):
    """id(node) -> list of (id(consumer), arg slot). Roots record a use by
    None, which no select can ever own -- so a stored value never sinks."""
    u = {id(n): [] for n in order}
    for n in order:
        for i, a in enumerate(n.args):
            if id(a) in live:
                u[id(a)].append((id(n), i))
    for r in roots:
        u[id(r)].append((None, None))
    return u


def _exclusive(sel, slot, users, live):
    """The nodes dominated by the (sel -> arm) edge: a monotone fixpoint,
    growing the sunk set until no further node has all its uses inside."""
    cone = {id(n) for n in _topo([sel.args[slot]], live)}
    sunk, changed = set(), True
    while changed:
        changed = False
        for nid in cone - sunk:
            ok = True
            for pid, ai in users[nid]:
                if pid == id(sel):
                    if ai != slot:  # also the condition, or the other arm
                        ok = False
                        break
                elif pid not in sunk:  # escapes the arm (or is a store root)
                    ok = False
                    break
            if ok:
                sunk.add(nid)
                changed = True
    return sunk


def assign_scopes(roots, live):
    """-> (order, home, scopes, arms). `home[id(n)]` is the scope a node is
    emitted in; `arms[id(select)]` is its (then_scope, else_scope)."""
    order = _topo(roots, live)
    users = _users(order, roots, live)
    home = {id(n): ROOT for n in order}
    scopes = [{"parent": None, "select": None, "slot": None, "depth": 0}]
    arms: dict[int, tuple[int, int]] = {}

    def assign(sel, parent):
        made = []
        for slot in (1, 2):
            sunk = _exclusive(sel, slot, users, live)
            sc = len(scopes)
            scopes.append(
                {"parent": parent, "select": sel, "slot": slot,
                 "depth": scopes[parent]["depth"] + 1}
            )
            made.append(sc)
            for nid in sunk:
                home[nid] = sc
            # Nested selects: outermost first (reverse topological). A select
            # re-homed deeper by an earlier recursion is skipped -- its scope
            # is created by the select that actually owns it.
            for n in reversed(order):
                if id(n) in sunk and is_select(n) and home[id(n)] == sc:
                    assign(n, sc)
        arms[id(sel)] = tuple(made)

    for n in reversed(order):
        if is_select(n) and home[id(n)] == ROOT:
            assign(n, ROOT)

    def ancestor_or_self(a, b):  # is scope `a` an ancestor-or-self of `b`?
        while b is not None:
            if b == a:
                return True
            b = scopes[b]["parent"]
        return False

    # THE soundness invariant, checked not assumed: an operand is emitted in
    # an ancestor-or-self scope of its consumer, so it is always in scope at
    # the use. A select is the ONE exemption -- sinking its arms below it is
    # the whole point, and the arm value is read at the assignment INSIDE the
    # arm block, never after it closes.
    for n in order:
        for i, a in enumerate(n.args):
            if id(a) not in home:  # a non-live arg the backend never reads
                continue
            if is_select(n) and i in (1, 2):
                assert arms[id(n)][i - 1] == home[id(a)] or home[id(a)] == home[id(n)], (
                    f"arm {i} of a select is neither sunk into its own scope "
                    f"nor left in the select's -- analysis bug"
                )
                continue
            assert ancestor_or_self(home[id(a)], home[id(n)]), (
                f"operand of {n.op} sank below its consumer -- analysis bug"
            )
    return order, home, scopes, arms


# --- emission ----------------------------------------------------------------


class _ScopedGen(flat._Gen):
    """`_Gen` with line placement by scope. Expression rendering is the
    parent's, untouched; only WHERE a `let` lands changes."""

    def __init__(self, leaf, home, arms):
        super().__init__(leaf)
        self.home, self.arms = home, arms
        self.items: dict[int, list] = {}

    def go(self, node):
        if id(node) in self.names:
            return self.names[id(node)]
        expr, is_bool = self.expr(node)
        var = f"e{self.n}"
        self.n += 1
        ty = "bool" if is_bool else "f32"
        self.items.setdefault(self.home[id(node)], []).append(f"let {var}: {ty} = {expr};")
        self.names[id(node)] = var
        if is_bool:
            self.bools.add(id(node))
        return var

    def emit_if(self, sel):
        """The reconstructed branch: a `var` for the joined result, the two
        arm scopes as real blocks, one assignment per arm."""
        cond = self.cond(sel.args[0])
        then_v, else_v = self.operand(sel.args[1]), self.operand(sel.args[2])
        var = f"e{self.n}"
        self.n += 1
        t_sc, e_sc = self.arms[id(sel)]
        self.items.setdefault(self.home[id(sel)], []).append(
            ("if", cond, then_v, else_v, var, t_sc, e_sc)
        )
        self.names[id(sel)] = var
        return var

    def render(self, sc: int, ind: str = "  ") -> list[str]:
        out = []
        for it in self.items.get(sc, []):
            if isinstance(it, str):
                out.append(ind + it)
                continue
            _, cond, t_v, e_v, var, t_sc, e_sc = it
            out.append(f"{ind}var {var}: f32;")
            out.append(f"{ind}if ({cond}) {{")
            out += self.render(t_sc, ind + "  ")
            out.append(f"{ind}  {var} = {t_v};")
            out.append(f"{ind}}} else {{")
            out += self.render(e_sc, ind + "  ")
            out.append(f"{ind}  {var} = {e_v};")
            out.append(f"{ind}}}")
        return out


def translate(art) -> tuple[str, dict]:
    """Region -> WGSL with reconstructed `if`s. Same (source, meta)
    contract as `wgsl_executor._translate`, so it drops into compile_wgsl."""
    region = art.region
    params = list(region.params)
    lattice = params[len(art.tensor_params) - 1] if art.tensor_params else None
    if lattice is None:
        raise Untranslatable("a kernel with no tensor parameters")
    axes = tuple(d.name for d in lattice.type.dims)
    if len(axes) > 2:
        raise Untranslatable("rank-3+ lattices (the 2D/1D subset translates today)")
    comp = {n: c for n, c in zip(axes, ("y", "x") if len(axes) == 2 else ("x",))}
    extents = {d.name: (d.start, d.stop) for d in lattice.type.dims}

    slots = list(art.uniforms)
    for _, _, _, base, plan, _ in art.arg_slots:
        for s in plan.slots:
            slots.append((f"arg{base}", base + s.dest.offset, s.dest.fmt))
    slot_index = {off: i for i, (_, off, _) in enumerate(sorted(slots, key=lambda s: s[1]))}

    def buf_index(p) -> str:
        strides, acc = [], 1
        for d in reversed(p.type.dims):
            strides.append(acc)
            acc *= d.size
        parts = []
        for d, s in zip(p.type.dims, reversed(strides)):
            if d.name not in comp:
                raise Untranslatable(f"a buffer dim {d.name!r} outside the launch lattice")
            parts.append(f"i32(gid.{comp[d.name]}) * {s}")
        return " + ".join(parts) or "0"

    def leaf(n, g):
        if n.op == "core.param":
            return f"buf{params.index(n)}[{buf_index(n)}]", False
        if n.op == "tl.iota":
            name = dict(n.attrs)["name"]
            if name not in comp:
                raise Untranslatable(f"an iota over dim {name!r} outside the launch lattice")
            return f"f32(gid.{comp[name]})", False
        if n.op == "abi.slot":
            return f"U[{slot_index[dict(n.attrs)['offset']]}]", False
        return None

    stores = [n for n in walk_region(region) if n.op == "tl.store"]
    for n in stores:
        if n.args[1].op != "core.param":
            raise Untranslatable("a store into a non-parameter tensor")
    roots = [n.args[2] for n in stores]
    live = live_set(stores, leaf)
    order, home, scopes, arms = assign_scopes(roots, live)

    g = _ScopedGen(leaf, home, arms)
    for n in order:  # topological drive: operands are named before consumers
        if is_select(n):
            g.emit_if(n)
        else:
            g.go(n)

    store_lines = [
        f"  buf{params.index(n.args[1])}[{buf_index(n.args[1])}] = {g.operand(n.args[2])};"
        for n in stores
    ]

    bindings = []
    for i, p in enumerate(params):
        w = i < len(art.tensor_params) and art.tensor_params[i] in art.writable
        bindings.append(
            f"@group(0) @binding({i}) var<storage, "
            f"{'read_write' if w or i >= len(art.tensor_params) else 'read'}> buf{i}: array<f32>;"
        )
    if slots:
        bindings.append(f"@group(0) @binding({len(params)}) var<storage, read> U: array<f32>;")

    wg = "@workgroup_size(8, 8, 1)" if len(axes) == 2 else "@workgroup_size(64, 1, 1)"
    guards = " || ".join(f"gid.{comp[a]} >= {extents[a][1] - extents[a][0]}u" for a in axes)
    src = "\n".join(
        [
            *bindings,
            f"@compute {wg}",
            "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {",
            f"  if ({guards}) {{ return; }}",
            *g.render(ROOT),
            *store_lines,
            "}",
        ]
    )
    meta = {
        "slots": sorted(slots, key=lambda s: s[1]),
        "axes": axes,
        "extents": extents,
        "n_params": len(params),
        "stats": {
            "nodes": len(order),
            "selects": sum(1 for n in order if is_select(n)),
            "sunk": sum(1 for n in order if home[id(n)] != ROOT),
            "scopes": len(scopes) - 1,
            "max_depth": max(s["depth"] for s in scopes),
        },
    }
    return src, meta


@contextmanager
def as_translator():
    """Reconstruction swapped into the EXISTING backend: compile_wgsl's
    buffers/uniforms/readback are untouched, the shader source is the
    only thing that changes."""
    old = flat._translate
    flat._translate = translate
    try:
        yield
    finally:
        flat._translate = old


def recon_artifact(art):
    with as_translator():
        return flat.wgpu_artifact(art)
