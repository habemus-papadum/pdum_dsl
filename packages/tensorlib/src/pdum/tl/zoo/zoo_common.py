"""Shared zoo plumbing: the ZooModel record and the parameter-blind library.

The library is S.1 (200 §6.1): plain functions from tensors to tensors —
no scope, no names, no Build. Unit bodies call them and they inline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..compute import const_like, iota, pointwise, red, reduce, repeat_like
from ..markers import exp, le, sqrt, where
from ..mdsl import defmarker, tanh
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
    region: object  # the dialect Region — THE representation
    inputs: dict[str, Tensor]
    out: str  # the output's name
    ref: Callable  # dict[str, np.ndarray] -> np.ndarray, the numpy denotation
    order: tuple  # dim order matching the reference array's axes
    names: dict  # the naming law's assignment over the region

    def numpy_inputs(self) -> dict[str, np.ndarray]:
        return {k: v.to_numpy() for k, v in self.inputs.items()}


def t_in(inputs: dict, name: str, arr, names) -> str:
    inputs[name] = Tensor.from_numpy(np.asarray(arr, dtype=np.float64), names)
    return name


def layernorm(x, g, b, *, feat, eps):
    mu = reduce(red.mean, x, feat)
    xc = x - repeat_like(mu, x)
    sd = pointwise(sqrt, reduce(red.mean, xc * xc, feat) + eps)
    return xc / repeat_like(sd, x) * repeat_like(g, x) + repeat_like(b, x)


def rmsnorm(x, g, *, feat, eps):
    ms = reduce(red.mean, x * x, feat)
    sd = pointwise(sqrt, ms + eps)
    return x / repeat_like(sd, x) * repeat_like(g, x)


def np_layernorm(x, g, beta, eps, axis=-1):
    mu = x.mean(axis=axis, keepdims=True)
    v = ((x - mu) ** 2).mean(axis=axis, keepdims=True)
    return (x - mu) / np.sqrt(v + eps) * g + beta


def np_rmsnorm(x, g, eps, axis=-1):
    return x / np.sqrt((x**2).mean(axis=axis, keepdims=True) + eps) * g


def softmax(sm, *, k):
    e = pointwise(exp, sm - repeat_like(reduce(red.max, sm, k), sm))
    return e / repeat_like(reduce(red.sum, e, k), sm)


def causal_softmax(sc, *, q="t", k="s"):
    mask = pointwise(le, iota(sc, k), iota(sc, q))
    return softmax(pointwise(where, mask, sc, const_like(sc, -1e9)), k=k)
