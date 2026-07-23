"""The carrier/unit signature pass — CONCERNS #16 resolved.

Markers finally have SIGNATURES: given per-operand (carrier, value-unit)
facts, every marker-DSL tree, primitive marker, reducer, and whole SSA
program propagates them forward — and refuses the nonsense the reference
layer used to evaluate happily (`exp` of micrometers, `x_volts + t_seconds`,
comparing quantities of different dimension).

Policy (lenient where unlabeled, strict where labeled):
- `None` means UNKNOWN, not dimensionless: it unifies with anything and
  taints products. Only two CONCRETE conflicting facts raise. Fully
  unlabeled programs pass through untouched — units stay opt-in metadata.
- Nonzero constants are dimensionless (ONE); the constant ZERO is
  unit-polymorphic (`where(mask, x_volts, 0)` is fine, `x_volts + 1` is
  not) — standard dimensional-analysis practice.
- exp/log/tanh demand a dimensionless argument and return a pure number.
- add/sub/maximum/minimum/where-branches/comparisons demand matching units.
- Carriers join along bool < int < rat < real < complex; true division
  lands in at least rat; transcendentals in at least real; comparisons in
  bool. A `where` condition must be bool (D13's carrier discipline).

`infer_signatures` walks a program the way `ir.infer` walks layouts; the
result makes `grad`'s `target_unit` INFERABLE (autodiff reads the target's
unit instead of being told). `marker_signature` is the single-marker entry
the reference `pointwise` now enforces at run time.

Not inferable here (honest gaps): `prod` of a dimensioned quantity (unit**n
needs a static extent), pad fills (unchecked constants), structured-dtype
unit maps (scalar Units only).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .dtypes import CARRIERS, carrier_of
from .mdsl import (
    COMPOSITE_MARKERS,
    COMPOSITE_REDUCERS,
    Arg,
    CompositeMarker,
    Const,
    Prim,
)
from .units import ONE, Unit, u


class SignatureError(ValueError):
    pass


@dataclass(frozen=True)
class VInfo:
    """What the signature pass knows about one value: the algebraic carrier
    its dtype approximates, and its value-space unit. None = unknown."""

    carrier: str | None = None
    unit: Unit | None = None


_ORDER = {c: i for i, c in enumerate(CARRIERS)}


def _join(*carriers) -> str | None:
    known = [c for c in carriers if c is not None]
    for c in known:
        if c not in _ORDER:
            raise SignatureError(f"unknown carrier {c!r} (expected one of {CARRIERS})")
    return max(known, key=_ORDER.__getitem__) if known else None


def _same_unit(op: str, a: Unit | None, b: Unit | None) -> Unit | None:
    if a is not None and b is not None and a != b:
        raise SignatureError(f"{op}: unit mismatch — {a!r} vs {b!r}")
    return a if a is not None else b


def _dimensionless(op: str, a: Unit | None) -> None:
    if a is not None and a.dims:
        raise SignatureError(f"{op}: argument must be dimensionless, got {a!r}")


def _const_info(value) -> VInfo:
    if isinstance(value, Fraction):
        carrier = "int" if value.denominator == 1 else "rat"
    elif isinstance(value, int):
        carrier = "int"
    else:
        carrier = "real"
    # zero is unit-polymorphic; any other literal is a pure number
    return VInfo(carrier, None if value == 0 else ONE)


def _prim_sig(op: str, a: tuple) -> VInfo:
    if op in ("add", "sub", "maximum", "minimum"):
        return VInfo(_join(a[0].carrier, a[1].carrier), _same_unit(op, a[0].unit, a[1].unit))
    if op == "neg":
        return a[0]
    if op == "mul":
        unit = None if a[0].unit is None or a[1].unit is None else a[0].unit * a[1].unit
        return VInfo(_join(a[0].carrier, a[1].carrier), unit)
    if op == "div":
        unit = None if a[0].unit is None or a[1].unit is None else a[0].unit / a[1].unit
        return VInfo(_join(a[0].carrier, a[1].carrier, "rat"), unit)
    if op in ("exp", "log", "tanh", "sin", "cos"):
        _dimensionless(op, a[0].unit)
        return VInfo(_join(a[0].carrier, "real"), ONE)
    if op == "sqrt":
        # sqrt of a dimensioned quantity would need fractional exponents;
        # sqrt of an UNKNOWN stays unknown (it may be dimensioned)
        _dimensionless(op, a[0].unit)
        return VInfo(_join(a[0].carrier, "real"), ONE if a[0].unit is not None else None)
    if op == "where":
        if a[0].carrier is not None and a[0].carrier != "bool":
            raise SignatureError(f"where: condition carrier must be bool, got {a[0].carrier!r}")
        return VInfo(_join(a[1].carrier, a[2].carrier), _same_unit(op, a[1].unit, a[2].unit))
    if op in ("eq", "ne", "le", "lt", "ge", "gt"):
        _same_unit(op, a[0].unit, a[1].unit)
        return VInfo("bool", None)
    # a primitive this pass has no rule for: the honest signature is UNKNOWN
    # (existence is policed by the registries and the gradient table, not here)
    return VInfo()


def _node_sig(node, args: tuple) -> VInfo:
    if isinstance(node, Arg):
        return args[node.index]
    if isinstance(node, Const):
        return _const_info(node.value)
    if isinstance(node, Prim):
        return _prim_sig(node.op, tuple(_node_sig(x, args) for x in node.args))
    raise TypeError(f"not a marker-DSL node: {node!r}")


def marker_signature(f, args) -> VInfo:
    """Signature of one marker application: `f` is a primitive marker name
    or a CompositeMarker; `args` are per-operand VInfos."""
    args = tuple(args)
    if isinstance(f, str) and f in COMPOSITE_MARKERS:
        f = COMPOSITE_MARKERS[f]
    if isinstance(f, CompositeMarker):
        if len(args) != f.arity:
            raise SignatureError(f"{f.name} takes {f.arity} operands, got {len(args)}")
        return _node_sig(f.body, args)
    return _prim_sig(f, args)


def _reducer_sig(fname: str, infos: tuple) -> VInfo:
    if fname in ("sum", "max", "min"):
        return infos[0]
    if fname == "mean":
        return VInfo(_join(infos[0].carrier, "real"), infos[0].unit)
    if fname == "prod":
        if infos[0].unit is None or infos[0].unit == ONE:
            return infos[0]
        raise SignatureError(
            "prod over a dimensioned quantity has unit u**n — not inferable "
            "without a static extent; strip or normalize units first"
        )
    f = COMPOSITE_REDUCERS.get(fname)
    if f is None:
        raise KeyError(f"no signature rule for reducer {fname!r}")
    if len(infos) != f.element:
        raise SignatureError(f"{fname} consumes {f.element} element tensor(s), got {len(infos)}")
    # state facts: seed with lift, iterate combine to the (short) fixed point
    state = tuple(_node_sig(n, infos) for n in f.lift)
    for _ in range(len(CARRIERS) + 1):
        out = tuple(_node_sig(n, state + state) for n in f.combine)
        merged = tuple(
            VInfo(_join(s.carrier, o.carrier), _same_unit(f"{fname}.combine", s.unit, o.unit))
            for s, o in zip(state, out)
        )
        if merged == state:
            break
        state = merged
    return _node_sig(f.project, state)


def _info_of(src) -> VInfo:
    if src is None:
        return VInfo()
    if isinstance(src, VInfo):
        return src
    vu = getattr(src, "value_units", None)  # Tensor; plain Layouts have neither
    return VInfo(getattr(src, "carrier", None), vu if isinstance(vu, Unit) else None)


def infer_signatures(prog, inputs=None) -> dict[str, VInfo]:
    """Propagate (carrier, unit) facts through a program — the metadata twin
    of `ir.infer`. `inputs` maps input vars to Tensors (or VInfos); missing
    or unit-less entries contribute unknowns, so unlabeled programs sail
    through while declared-unit conflicts raise SignatureError."""
    from .ir import _LAYOUT_OPS  # lazy: keeps compute -> signatures acyclic

    inputs = inputs or {}
    sigs: dict[str, VInfo] = {}
    for ins in prog.instrs:
        p = ins.params
        if ins.op == "input":
            sigs[ins.var] = _info_of(inputs.get(ins.var))
        elif ins.op == "const":
            carrier = carrier_of(np.dtype(p.get("dtype", "float64")))
            sigs[ins.var] = VInfo(carrier, None if p["value"] == 0 else ONE)
        elif ins.op == "iota":
            unit = p.get("unit")
            if unit is None:
                sigs[ins.var] = VInfo("int", None)
            else:
                sigs[ins.var] = VInfo("rat", u.parse_unit(unit) if isinstance(unit, str) else unit)
        elif ins.op == "pointwise":
            sigs[ins.var] = marker_signature(p["f"], tuple(sigs[o] for o in ins.operands))
        elif ins.op in ("reduce", "scan"):
            sigs[ins.var] = _reducer_sig(p["f"], tuple(sigs[o] for o in ins.operands))
        elif ins.op == "with_value_units":
            vu = p["value_units"]
            sigs[ins.var] = VInfo(sigs[ins.operands[0]].carrier, vu if isinstance(vu, Unit) else None)
        elif ins.op == "fold":
            state_names, elem_names = tuple(p["state"]), tuple(p["element"])
            carry = dict(p["carry"])
            sigmap = {n: sigs[o] for n, o in zip(state_names + elem_names, ins.operands)}
            ss = {}
            for _ in range(len(CARRIERS) + 1):  # carry fixed point, like reducers
                ss = infer_signatures(p["step"], sigmap)
                merged = {
                    sn: VInfo(
                        _join(sigmap[sn].carrier, ss[carry[sn]].carrier),
                        _same_unit("fold.carry", sigmap[sn].unit, ss[carry[sn]].unit),
                    )
                    for sn in state_names
                }
                if all(merged[sn] == sigmap[sn] for sn in state_names):
                    break
                sigmap.update(merged)
            sigs[ins.var] = ss[p["out"][1]]
        elif ins.op == "materialize" or ins.op in _LAYOUT_OPS:
            # layout ops move coordinates, never values (pad fills unchecked)
            sigs[ins.var] = sigs[ins.operands[0]]
        else:
            raise KeyError(f"no signature rule for op {ins.op!r}")
    return sigs
