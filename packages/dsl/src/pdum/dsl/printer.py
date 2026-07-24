"""The MLIR-flavored textual form: golden-testable, and migration insurance
(keep the printed shape close enough to MLIR that adopting xDSL/MLIR later is
a refactor, not a rewrite — architecture D2).

Printing is a DAG walk: shared subexpressions get **one** definition line and
are referenced thereafter — structural sharing is visible in the text.
"""

from __future__ import annotations

from .ir import Node, Region


def print_program(region: Region, name: str = "program") -> str:
    names: dict[int, str] = {}
    counter = [0]
    out: list[str] = []

    def fmt_attrs(node: Node) -> str:
        if not node.attrs:
            return ""
        inner = ", ".join(f"{k} = {v!r}" for k, v in node.attrs)
        return " {" + inner + "}"

    def define(node: Node, indent: str) -> str:
        if id(node) in names:
            return names[id(node)]
        for a in node.args:
            define(a, indent)
        argrefs = ", ".join(define(a, indent) for a in node.args)
        regs = ""
        if node.regions:
            blocks = [render_region(r, indent + "  ") for r in node.regions]
            regs = " (" + ", ".join(blocks) + ")"
        if node.op == "core.param":
            ref = f"%p{dict(node.attrs)['index']}"
        elif node.op == "core.yield":
            out.append(f"{indent}core.yield {argrefs}".rstrip())
            return ""
        else:
            ref = f"%{counter[0]}"
            counter[0] += 1
            sep = " " if argrefs else ""
            out.append(f"{indent}{ref} = {node.op}{sep}{argrefs}{fmt_attrs(node)}{regs} : {node.type!r}")
        names[id(node)] = ref
        return ref

    def render_region(r: Region, indent: str) -> str:
        marker = len(out)
        for n in r.body:
            define(n, indent)
        block = "\n".join(out[marker:])
        del out[marker:]
        return "{\n" + block + "\n" + indent[:-2] + "}"

    params = ", ".join(f"%p{dict(p.attrs)['index']}: {p.type!r}" for p in region.params)
    for p in region.params:
        names[id(p)] = f"%p{dict(p.attrs)['index']}"
    for n in region.body:
        define(n, "  ")
    return f"{name}({params}) {{\n" + "\n".join(out) + "\n}"
