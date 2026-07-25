"""THE marker vocabulary (200 §S.2, amended P6): one object per primitive.

A Marker is a declared operation name wrapping the numpy function — the
NAIVE BACKEND applies it directly (``pointwise(cos, t)`` computes np.cos,
eagerly, uncompiled), the compiler replaces it (the same name keys the IR,
the derivative table, the signature rules, and every device spelling).

Application is ALWAYS through the compute operators at the tensor tier —
``pointwise(cos, t)``, ``reduce(sum_r, x, dims)`` — never a bare call
(owner ruling, P6): calling a Marker refuses, pointing at pointwise. Bare
names appear only inside SCALAR marker bodies (defmarker/defreducer), where
the AST producer lowers them by name onto the Node schema.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Marker:
    """A pointwise primitive: a declared operation, not a callback."""

    name: str
    fn: object  # numpy callable applied elementwise to aligned arrays

    @property
    def op(self) -> str:  # the AST producers resolve bare calls by this
        return self.name

    @property
    def arity(self) -> int | None:
        from .derivative import TABLE

        rules = TABLE.get(self.name)
        return None if rules is None else len(rules)

    def __call__(self, *args):
        raise TypeError(
            f"{self.name} is a marker — tensor-tier application is spelled "
            f"pointwise({self.name}, ...) (bare names lower only inside "
            f"scalar marker bodies)"
        )

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Reducer:
    """A reduction primitive: a declared monoid (plus numpy ufunc for the
    reference layer). `identity` is None when no dtype-independent identity
    exists (max/min); pass `zero=` to reduce() in that case if the reduced
    extent can be empty."""

    name: str
    fn: object  # numpy ufunc; .reduce(arr, axis=...) is used
    identity: object = None
    associative: bool = True
    commutative: bool = True
    normalize: bool = False  # divide by the (static) reduced numel

    def __repr__(self) -> str:
        return self.name


class pw:
    """The initial pointwise marker set."""

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
    where = Marker("where", np.where)  # ternary select
    eq = Marker("eq", np.equal)
    ne = Marker("ne", np.not_equal)
    le = Marker("le", np.less_equal)
    lt = Marker("lt", np.less)
    ge = Marker("ge", np.greater_equal)
    gt = Marker("gt", np.greater)


class red:
    """The initial reducer set."""

    sum = Reducer("sum", np.add, identity=0)
    prod = Reducer("prod", np.multiply, identity=1)
    max = Reducer("max", np.maximum)
    min = Reducer("min", np.minimum)
    mean = Reducer("mean", np.add, identity=0, normalize=True)


# module-level names: what scalar bodies reference and everyone imports
add, sub, mul, div, neg = pw.add, pw.sub, pw.mul, pw.div, pw.neg
exp, log, tanh, sqrt, sin, cos = pw.exp, pw.log, pw.tanh, pw.sqrt, pw.sin, pw.cos
maximum, minimum, where = pw.maximum, pw.minimum, pw.where
eq, ne, le, lt, ge, gt = pw.eq, pw.ne, pw.le, pw.lt, pw.ge, pw.gt
