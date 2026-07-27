"""Tensor-typed lifting — the step tier's public surface (S.2).

Since the pivot's step switch (240 C4.3d), bodies lower through the ONE
dsl Lowerer with the tl dialect pack (``dialect.py``) and render back as
Programs through the migration view; ``_Lifter`` — the second lowering
engine — is DELETED. This module keeps the step tier's public surface
(``lift_step``/``LiftedStep``), the frozen layout-method table
(``_METHODS`` — the dialect consumes its packers), the structural-slot
voice, and the host-math tables.

The semantics are unchanged and pinned: unannotated parameters are
TENSOR-typed inputs; ``n: Literal[int]`` parameters are STRUCTURAL, bound
to build-time values with host arithmetic in structural slots; helpers
inline; tensors reaching structural slots refuse, naming the annotation
fix; straight-line enforced at lowering.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from pdum.dsl.types import LiteralAnnotation

from .ir import Program
from .layout import Layout
from .producer import _fn_ast
from .tensor import Tensor

_STRUCTURAL_SLOT = (
    "{what} is a STRUCTURAL slot: a runtime tensor cannot shape the lattice — "
    "pass a build-time value (annotate the parameter: `n: Literal[int]`, 200 §1.5)"
)

# method name -> (op, packer(args, kwargs) -> params); every packer input is
# host-evaluated before packing (the structural-slot discipline)
_METHODS = {
    "slice": ("slice", lambda a, kw: {"ranges": kw}),
    "select": ("select", lambda a, kw: {"coords": kw}),
    "shift": ("shift", lambda a, kw: {"deltas": kw}),
    "rename": ("rename", lambda a, kw: {"mapping": kw}),
    "repeat": ("repeat", lambda a, kw: {"name": a[0], "extent": a[1], **kw}),
    "flip": ("flip", lambda a, kw: {"name": a[0]}),
    # split's parts ORDER is the mixed-radix nesting (outer..inner) — packed
    # as a tuple of pairs so the dialect's canonical attr sort (identity)
    # cannot reorder it; _thaw_params restores the ordered dict downstream
    "split": ("split", lambda a, kw: {"name": a[0], "parts": tuple(kw.items())}),
    "merge": ("merge", lambda a, kw: {"parts": tuple(a[0]), "name": a[1], **kw}),
    "diagonal": ("diagonal", lambda a, kw: {"parts": tuple(a[0]), "name": a[1], **kw}),
    "window": ("window", lambda a, kw: dict(zip(("name", "k_name", "k", "dilation"), a)) | kw),
    "decimate": ("decimate", lambda a, kw: dict(zip(("name", "factor", "phase"), a)) | kw),
    "pad": ("pad", lambda a, kw: {"fill": a[0] if a else kw.pop("fill"), "extents": kw}),
    "stencil": ("stencil", lambda a, kw: dict(zip(("name", "k", "k_name", "fill", "dilation"), a)) | kw),
    "strip_charts": ("strip_charts", lambda a, kw: {}),
    "with_charts": ("with_charts", lambda a, kw: {"charts": kw}),
    "with_labels": ("with_labels", lambda a, kw: {"labels": kw}),
    "bind": ("bind", lambda a, kw: {"levels": kw}),
    "simplify": ("simplify", lambda a, kw: {}),
    "with_value_units": ("with_value_units", lambda a, kw: {"value_units": a[0]}),
    "round_to": ("round_to", lambda a, kw: {"encoding": a[0], **kw}),
}


@dataclass(frozen=True)
class _Intrinsic:
    """An S.1 vocabulary function: meaningful only inside a lowered body."""

    name: str

    def __call__(self, *args, **kwargs):
        raise TypeError(
            f"{self.name} is assemblage vocabulary — it lowers by inspection "
            f"inside a unit or step body; there is nothing to call"
        )


def __getattr__(name):  # lifting.contract stays importable — ONE function
    if name == "contract":
        from .compute import contract as c

        return c
    raise AttributeError(name)


@dataclass(frozen=True)
class LiftedStep:
    program: Program
    inputs: tuple[str, ...]  # tensor parameter names, in signature order
    outputs: tuple[str, ...]  # SSA vars of the returned tensors, in order


def lift_step(fn, **bindings) -> LiftedStep:
    """Lift ``fn`` to a step Program. Bind every tensor parameter to a
    Layout (or Tensor, whose layout is taken) and every ``Literal``-annotated
    parameter to a build-time value.

    Since the pivot's step switch (240 C4.3d), the body lowers through the
    ONE dsl Lowerer with the tl dialect pack and is rendered back as a
    Program through the migration view — every consumer unchanged."""
    from .dialect import export_program, lower_body, tensor_type_of_layout

    tree = _fn_ast(fn)
    anns = getattr(fn, "__annotations__", {})
    params = [a.arg for a in tree.args.args]
    inputs, arg_types, host = [], [], {}
    for p in params:
        if p not in bindings:
            raise ValueError(f"parameter {p!r} is unbound — lift_step binds every parameter by name")
        v = bindings.pop(p)
        ann = anns.get(p)
        if isinstance(ann, str):  # `from __future__ import annotations` in the def site
            ann = eval(ann, fn.__globals__)  # noqa: S307 — the def site's own namespace
        if isinstance(ann, LiteralAnnotation):
            if not isinstance(v, ann.base):
                raise ValueError(f"parameter {p!r} is Literal[{ann.base.__name__}]; got {v!r}")
            host[p] = v
            continue
        if isinstance(v, Tensor):
            v = v.layout
        if not isinstance(v, Layout):
            raise ValueError(
                f"parameter {p!r} is tensor-typed (unannotated) but received {v!r} — "
                f"structural parameters declare themselves: annotate `{p}: Literal[{type(v).__name__}]`"
            )
        arg_types.append(tensor_type_of_layout(v))
        inputs.append(p)
    if bindings:
        raise ValueError(f"unknown parameters bound: {sorted(bindings)}")
    bound_names: dict = {}
    region = lower_body(fn, tuple(arg_types), kind="step", host=host, out_names=bound_names)
    program, outs = export_program(region, tuple(inputs), names_of=bound_names)
    return LiftedStep(program, tuple(inputs), outs)


_HOST_BIN = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}
_HOST_CMP = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.LtE: lambda a, b: a <= b,
    ast.Lt: lambda a, b: a < b,
    ast.GtE: lambda a, b: a >= b,
    ast.Gt: lambda a, b: a > b,
}
