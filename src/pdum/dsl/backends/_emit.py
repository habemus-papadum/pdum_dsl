"""The shared dominator-placed emission walker (one copy, two renderers).

Both source-emitting backends need the same subtle algorithm: emit each node
in the DEEPEST region that dominates all its uses, so branch-exclusive work
stays lazy inside its `if`/`else` while anything shared hoists to the join
point (where some live path needs it anyway — the guard-then-divide fix,
step-8 review). The owner computation is the part most likely to drift
silently if duplicated; the house rule said extract at the third copy, and
the WGSL renderer was the third copy.

Contract: ``emit_dominated(region, statement, branch_join)`` returns
``(lines, names, result_node)``. The backend supplies only spelling:

- ``statement(node, names) -> str`` — one non-branch assignment, WITHOUT
  indentation (``v3 = a + b`` / ``let v3: f32 = a + b;``).
- ``branch_join(node, names, result_of, emit_block, path, indent) -> [str]``
  — the ``core.if`` construct; ``emit_block(subpath, indent)`` renders a
  branch's interior, ``result_of(branch_index)`` names its yielded value.
"""

from __future__ import annotations

from ..kernel.ir import Node, Region


def emit_dominated(region: Region, statement, branch_join, indent: str = "  ") -> tuple:
    topo: list[Node] = []
    seen: set[int] = set()
    users: dict[int, list[Node]] = {}
    anchors: dict[int, list] = {}
    region_result: dict[tuple, Node] = {}

    def walk(node: Node) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        for a in node.args:
            users.setdefault(id(a), []).append(node)
            walk(a)
        for i, r in enumerate(node.regions):
            for n in r.body:
                inner = n.args[0] if n.op == "core.yield" else n
                anchors.setdefault(id(inner), []).append((id(node), i))
                if n.op == "core.yield":
                    region_result[(id(node), i)] = inner
                walk(inner)
        topo.append(node)

    result = None
    for n in region.body:
        inner = n.args[0] if n.op == "core.yield" else n
        result = inner if n.op == "core.yield" else result
        anchors.setdefault(id(inner), []).append(None)  # anchored at the root
        walk(inner)

    def lca(a: tuple, b: tuple) -> tuple:
        out = []
        for x, y in zip(a, b):
            if x != y:
                break
            out.append(x)
        return tuple(out)

    # Owner = longest common prefix of all use paths; users precede their args
    # in reversed topo, so owner(user) is final when an arg is processed.
    owner: dict[int, tuple] = {}
    for node in reversed(topo):
        paths = [owner[id(u)] for u in users.get(id(node), ())]
        paths += [() if a is None else (*owner[a[0]], a) for a in anchors.get(id(node), ())]
        o = paths[0]
        for q in paths[1:]:
            o = lca(o, q)
        owner[id(node)] = o

    names = {id(n): f"v{k}" for k, n in enumerate(topo)}
    blocks: dict[tuple, list[Node]] = {}
    for n in topo:
        blocks.setdefault(owner[id(n)], []).append(n)

    def emit_block(path: tuple, ind: str) -> list[str]:
        out: list[str] = []
        for node in blocks.get(path, ()):
            if node.op == "core.if":
                out += branch_join(
                    node,
                    names,
                    lambda i, _n=node: names[id(region_result[(id(_n), i)])],
                    emit_block,
                    path,
                    ind,
                )
            else:
                out.append(f"{ind}{statement(node, names)}")
        return out

    return emit_block((), indent), names, result
