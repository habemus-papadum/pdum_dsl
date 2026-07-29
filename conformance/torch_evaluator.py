"""The torch substrate — the Region -> torch CHECK column (310).

The evaluator core (dims machinery, take/scatter/fold, dispatch) lives in
region_evaluator.py, shared with every framework column; this module
contributes the ~16 array hooks and the primitive marker table, plus the
column's doors: ``run_region_torch`` / ``run_named_torch``.

numpy upcasts integer inputs through float-only ufuncs and true-divides
ints to f64; torch refuses or drops to the default dtype — so float-only
rows cast integral inputs to the column's dtype first.
"""

from __future__ import annotations

import numpy as np
import torch

from region_evaluator import Field, Substrate, Untranslatable, run_named_on, run_region_on  # noqa: F401

# the executor-fingerprint seam (kernel.py's content door / 283's Pair seat):
# a version change here is a new artifact world
TORCH_FP = ("region_torch", "torch", torch.__version__.split("+")[0])

TorchField = Field  # the door's historical name


class TorchSubstrate(Substrate):
    name = "torch"

    def __init__(self, device, dtype=torch.float64):
        self.dtype = dtype
        super().__init__(torch.device(device))

    def asarray(self, x):
        t = torch.as_tensor(x if isinstance(x, torch.Tensor) else np.asarray(x)).to(self.device)
        return t.to(self.dtype) if t.is_floating_point() else t

    def to_numpy(self, x):
        return x.cpu().numpy()

    def is_array(self, x):
        return isinstance(x, torch.Tensor)

    def to_float(self, x):
        return x.to(self.dtype) if not x.is_floating_point() else x

    def as_index(self, x):
        return x.to(torch.long)

    def full(self, shape, value, kind):
        return torch.full(tuple(shape), value, dtype=torch.int64 if kind == "int" else self.dtype, device=self.device)

    def arange(self, start, stop):
        return torch.arange(start, stop, dtype=torch.int64, device=self.device)

    def permute(self, x, perm):
        return x.permute(perm)

    def expand_dims(self, x, axis):
        return x.unsqueeze(axis)

    def broadcast_to(self, x, shape):
        return x.expand(tuple(shape))

    def moveaxis(self, x, src, dst):
        return torch.movedim(x, src, dst)

    def flip(self, x, axis):
        return torch.flip(x, (axis,))

    def stack0(self, xs):
        return torch.stack(xs, dim=0)

    def paste(self, shape, fill, slices, data):
        canvas = torch.full(tuple(shape), fill, dtype=data.dtype, device=self.device)
        canvas[slices] = data
        return canvas

    def scatter_rows_add(self, shape, rows, idx, vals):
        acc = torch.zeros(tuple(shape), dtype=vals.dtype, device=self.device)
        acc.index_put_((rows, idx), vals, accumulate=True)
        return acc

    def argsort_stable(self, x):
        return torch.argsort(x, dim=-1, stable=True)

    def cum(self, name, x, axis):
        if name == "sum":
            return torch.cumsum(x, dim=axis)
        if name == "prod":
            return torch.cumprod(x, dim=axis)
        return (torch.cummax if name == "max" else torch.cummin)(x, dim=axis).values

    def red(self, name, x, axes):
        if name == "prod":  # torch.prod folds one dim at a time
            for ax in sorted(axes, reverse=True):
                x = torch.prod(x, dim=ax)
            return x
        fn = {"sum": torch.sum, "max": torch.amax, "min": torch.amin, "mean": torch.mean}[name]
        return fn(x, dim=axes)

    def pw_table(self):
        def f(x):
            return x.to(self.dtype) if isinstance(x, torch.Tensor) and not x.is_floating_point() else x

        return {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: f(a) / f(b),
            "neg": lambda a: -a,
            "exp": lambda a: torch.exp(f(a)),
            "log": lambda a: torch.log(f(a)),
            "maximum": torch.maximum,
            "minimum": torch.minimum,
            "tanh": lambda a: torch.tanh(f(a)),
            "sqrt": lambda a: torch.sqrt(f(a)),
            "sin": lambda a: torch.sin(f(a)),
            "cos": lambda a: torch.cos(f(a)),
            "abs": torch.abs,
            "floor": lambda a: torch.floor(f(a)),
            "stop_gradient": lambda a: a,
            "where": lambda c, a, b: torch.where(c.bool() if isinstance(c, torch.Tensor) else c, a, b),
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "le": lambda a, b: a <= b,
            "lt": lambda a, b: a < b,
            "ge": lambda a, b: a >= b,
            "gt": lambda a, b: a > b,
        }


def run_region_torch(region, values, *, device="cpu", dtype=torch.float64):
    """Evaluate a tensor-tier region over torch — ``values`` positional per
    param, each array in ITS PARAM's type-dim order (numpy or torch)."""
    return run_region_on(TorchSubstrate(device, dtype), region, values)


def run_named_torch(region, inputs, names, *, device="cpu", dtype=torch.float64):
    """dialect.run_named's door on the torch substrate."""
    return run_named_on(TorchSubstrate(device, dtype), region, inputs, names)
