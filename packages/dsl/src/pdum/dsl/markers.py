"""THE marker vocabulary (200 §S.2, numpy-authority amendment): one object
per primitive, named by numpy, at the language tier.

A Marker is a declared operation wrapping THE numpy function — numpy is the
naming authority (``maximum``/``minimum`` are the binary pointwise ops,
``max``/``min`` the reductions; ``where`` the ternary; exactly our
pointwise/reduce split) AND the reference semantics (IEEE non-trapping:
0/0 flows as nan, like a device — never a Python exception).

The marker OBJECT is the user-facing identity at every tier; strings exist
only as internal dict keys. One marker, several columns:

- ``fn`` — the numpy callable: the tensor tier applies it eagerly
  (``pointwise(cos, t)`` computes np.cos), and a HOST-SCALAR call applies
  it directly (``sqrt(4.0)`` is 2.0 — markers are ordinary math on plain
  numbers; per-element oracles do exactly this);
- ``op`` — the name tl's AST producer resolves bare calls by (Prim trees,
  the tl IR, the derivative-table key);
- ``value_op`` (module fn) — the value-tier IR op: ``pw.<name>`` for the
  function primitives, ``core.select`` for ``where`` (structure stays
  core-owned); intrinsics.install derives spellings from it.

Tensor-tier application is ALWAYS through the compute operators (owner
ruling, P6): calling a marker on anything that is not a host scalar
refuses, pointing at pointwise. Bare names inside scalar marker bodies and
kernel bodies lower by name — the one body language.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_HOST_SCALARS = (int, float, complex, bool, np.number, np.bool_)


@dataclass(frozen=True)
class Marker:
    """A pointwise primitive: a declared operation, not a callback."""

    name: str
    fn: object  # THE numpy callable: elementwise on arrays, direct on scalars

    @property
    def op(self) -> str:  # the AST producers resolve bare calls by this
        return self.name

    @property
    def arity(self) -> int | None:
        from .derivative import TABLE

        rules = TABLE.get(self.name)
        return None if rules is None else len(rules)

    def __call__(self, *args):
        if args and all(isinstance(a, _HOST_SCALARS) for a in args):
            return self.fn(*args)  # ordinary math on host scalars
        raise TypeError(
            f"{self.name} is a marker — tensor-tier application is spelled "
            f"pointwise({self.name}, ...) (bare names lower only inside "
            f"scalar marker bodies)"
        )

    def __repr__(self) -> str:
        return self.name


def value_op(marker: Marker) -> str:
    """The value-tier IR op this marker names. Structure stays core-owned
    (``where`` IS core.select); function primitives are the pw dialect."""
    return "core.select" if marker.name == "where" else f"pw.{marker.name}"


class pw:
    """The pointwise marker set — THE vocabulary, numpy-named."""

    add = Marker("add", np.add)
    sub = Marker("sub", np.subtract)
    mul = Marker("mul", np.multiply)
    div = Marker("div", np.divide)
    neg = Marker("neg", np.negative)
    exp = Marker("exp", np.exp)
    log = Marker("log", np.log)
    maximum = Marker("maximum", np.maximum)
    minimum = Marker("minimum", np.minimum)
    tanh = Marker("tanh", np.tanh)
    sqrt = Marker("sqrt", np.sqrt)
    sin = Marker("sin", np.sin)
    cos = Marker("cos", np.cos)
    abs = Marker("abs", np.abs)
    floor = Marker("floor", np.floor)
    where = Marker("where", np.where)  # ternary select
    eq = Marker("eq", np.equal)
    ne = Marker("ne", np.not_equal)
    le = Marker("le", np.less_equal)
    lt = Marker("lt", np.less)
    ge = Marker("ge", np.greater_equal)
    gt = Marker("gt", np.greater)


# module-level names: what scalar bodies reference and everyone imports
# (pw.abs stays class-only — shadowing the abs builtin at module level would
# surprise; the "abs" CALL name still resolves through the overload table)
add, sub, mul, div, neg = pw.add, pw.sub, pw.mul, pw.div, pw.neg
exp, log, tanh, sqrt, sin, cos = pw.exp, pw.log, pw.tanh, pw.sqrt, pw.sin, pw.cos
maximum, minimum, where, floor = pw.maximum, pw.minimum, pw.where, pw.floor
eq, ne, le, lt, ge, gt = pw.eq, pw.ne, pw.le, pw.lt, pw.ge, pw.gt
