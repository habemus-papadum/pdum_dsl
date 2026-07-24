"""Shared zoo plumbing: the ZooModel record and reusable blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..build import Build
from ..ir import Program
from ..mdsl import defmarker, exp, tanh
from ..tensor import Tensor

# composite activations, registered once at zoo import
GELU_C = 0.7978845608028654  # sqrt(2/pi)
gelu = defmarker("zoo.gelu", 1, lambda x: 0.5 * x * (1 + tanh(GELU_C * (x + 0.044715 * x * x * x))))
sigmoid = defmarker("zoo.sigmoid", 1, lambda x: 1 / (1 + exp(-x)))
silu = defmarker("zoo.silu", 1, lambda x: x * (1 / (1 + exp(-x))))


def np_gelu(x):
    return 0.5 * x * (1 + np.tanh(GELU_C * (x + 0.044715 * x**3)))


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def np_softmax(s, axis):
    e = np.exp(s - s.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


@dataclass(frozen=True)
class ZooModel:
    program: Program
    inputs: dict[str, Tensor]
    out: str  # the output var
    ref: Callable  # dict[str, np.ndarray] -> np.ndarray, the numpy denotation
    order: tuple  # dim order matching the reference array's axes

    def numpy_inputs(self) -> dict[str, np.ndarray]:
        return {k: v.to_numpy() for k, v in self.inputs.items()}


def t_in(inputs: dict, name: str, arr, names) -> str:
    inputs[name] = Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)
    return name


def layernorm(b: Build, x: str, feat: str, extent, shape, g: str, beta: str, eps: float) -> str:
    """(x - mean) / sqrt(var + eps) * g + beta over the feat dim.
    `shape` lists x's full (name, extent) dims; g/beta are (feat,) vars."""
    others = [de for de in shape if de[0] != feat]
    mu = b.red("mean", x, (feat,), hint="mu")
    xc = b.pw("sub", x, b.bcast(mu, [(feat, extent)]), hint="xc")
    var = b.red("mean", b.pw("mul", xc, xc), (feat,), hint="var")
    ve = b.pw("add", var, b.const(eps, others, hint="eps"), hint="ve")
    sd = b.pw("sqrt", ve, hint="sd")
    xn = b.pw("div", xc, b.bcast(sd, [(feat, extent)]), hint="xn")
    return b.pw("add", b.pw("mul", xn, b.bcast(g, others)), b.bcast(beta, others), hint="ln")


def rmsnorm(b: Build, x: str, feat: str, extent, shape, g: str, eps: float) -> str:
    others = [de for de in shape if de[0] != feat]
    ms = b.red("mean", b.pw("mul", x, x), (feat,), hint="ms")
    ve = b.pw("add", ms, b.const(eps, others, hint="eps"), hint="ve")
    sd = b.pw("sqrt", ve, hint="sd")
    xn = b.pw("div", x, b.bcast(sd, [(feat, extent)]), hint="xn")
    return b.pw("mul", xn, b.bcast(g, others), hint="rms")


def np_layernorm(x, g, beta, eps, axis=-1):
    mu = x.mean(axis=axis, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=axis, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + beta


def np_rmsnorm(x, g, eps, axis=-1):
    return x / np.sqrt((x**2).mean(axis=axis, keepdims=True) + eps) * g


def contract(b: Build, x: str, y: str, x_missing, y_missing, over, hint="mm") -> str:
    """sum_over(x * y) after broadcasting each operand over the dims only
    the other carries — matmul as declaration (repeat + mul + reduce)."""
    xb = b.bcast(x, x_missing) if x_missing else x
    yb = b.bcast(y, y_missing) if y_missing else y
    return b.red("sum", b.pw("mul", xb, yb), over, hint=hint)


def causal_softmax(b: Build, sc: str, tname: str, sname: str, shape) -> str:
    """Mask s>t to -1e9 (iota comparison — masks are closed forms, not
    memory), then softmax over the key dim."""
    it = b.emit("iota", (sc,), hint="it", name=tname)
    isv = b.emit("iota", (sc,), hint="is", name=sname)
    m = b.pw("le", isv, it, hint="mask")
    neg = b.const(-1e9, shape, hint="ninf")
    sm = b.pw("where", m, sc, neg, hint="scm")
    return softmax(b, sm, sname, dict(shape)[sname], shape)


def softmax(b: Build, sm: str, sname: str, extent, shape) -> str:
    mx = b.red("max", sm, (sname,), hint="mx")
    e = b.pw("exp", b.pw("sub", sm, b.bcast(mx, [(sname, extent)])), hint="e")
    z = b.red("sum", e, (sname,), hint="z")
    return b.pw("div", e, b.bcast(z, [(sname, extent)]), hint="p")
