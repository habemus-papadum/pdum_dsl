"""The jax substrate — the Region -> jax CHECK column (310).

The evaluator core lives in region_evaluator.py; this module contributes
the array hooks, the primitive marker table, and the doors
``run_region_jax`` / ``run_named_jax``.

x64 is enabled at import: the conformance columns assert the zoo's f64
denotation, and jax's f32 default would silently halve every comparison.
Constructors are committed to the substrate's device explicitly — with the
CUDA plugin installed jax's default device is the GPU, and a cpu-column
run must not leak constructor arrays onto it.
"""

from __future__ import annotations

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # before any array is built

import jax.numpy as jnp  # noqa: E402 — after the x64 switch, by design

from region_evaluator import Field, Substrate, Untranslatable, run_named_on, run_region_on  # noqa: F401, E402

# the executor-fingerprint seam (kernel.py's content door / 283's Pair seat)
JAX_FP = ("region_jax", "jax", jax.__version__)

JaxField = Field


class JaxSubstrate(Substrate):
    name = "jax"

    def __init__(self, device, dtype=jnp.float64):
        platform = "gpu" if device in ("cuda", "gpu") else "cpu"
        self.dtype = dtype
        super().__init__(jax.devices(platform)[0])

    def asarray(self, x):
        t = jax.device_put(jnp.asarray(x), self.device)
        return t.astype(self.dtype) if jnp.issubdtype(t.dtype, jnp.floating) else t

    def to_numpy(self, x):
        return np.asarray(x)

    def is_array(self, x):
        return isinstance(x, jax.Array)

    def to_float(self, x):
        return x if jnp.issubdtype(x.dtype, jnp.floating) else x.astype(self.dtype)

    def as_index(self, x):
        return x.astype(jnp.int64)

    def full(self, shape, value, kind):
        dt = jnp.int64 if kind == "int" else self.dtype
        return jax.device_put(jnp.full(tuple(shape), value, dtype=dt), self.device)

    def arange(self, start, stop):
        return jax.device_put(jnp.arange(start, stop, dtype=jnp.int64), self.device)

    def permute(self, x, perm):
        return jnp.transpose(x, perm)

    def expand_dims(self, x, axis):
        return jnp.expand_dims(x, axis)

    def broadcast_to(self, x, shape):
        return jnp.broadcast_to(x, tuple(shape))

    def moveaxis(self, x, src, dst):
        return jnp.moveaxis(x, src, dst)

    def flip(self, x, axis):
        return jnp.flip(x, axis)

    def stack0(self, xs):
        return jnp.stack(xs, axis=0)

    def paste(self, shape, fill, slices, data):
        canvas = jax.device_put(jnp.full(tuple(shape), fill, dtype=data.dtype), self.device)
        return canvas.at[slices].set(data)

    def scatter_rows_add(self, shape, rows, idx, vals):
        acc = jax.device_put(jnp.zeros(tuple(shape), dtype=vals.dtype), self.device)
        return acc.at[rows, idx].add(vals)

    def argsort_stable(self, x):
        return jnp.argsort(x, axis=-1, stable=True)

    def cum(self, name, x, axis):
        if name == "sum":
            return jnp.cumsum(x, axis=axis)
        if name == "prod":
            return jnp.cumprod(x, axis=axis)
        return (jax.lax.cummax if name == "max" else jax.lax.cummin)(x, axis=axis)

    def red(self, name, x, axes):
        fn = {"sum": jnp.sum, "prod": jnp.prod, "max": jnp.max, "min": jnp.min, "mean": jnp.mean}[name]
        return fn(x, axis=axes)

    def pw_table(self):
        def f(x):
            return x.astype(self.dtype) if isinstance(x, jax.Array) and not jnp.issubdtype(x.dtype, jnp.floating) else x

        return {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: f(a) / f(b),
            "neg": lambda a: -a,
            "exp": lambda a: jnp.exp(f(a)),
            "log": lambda a: jnp.log(f(a)),
            "maximum": jnp.maximum,
            "minimum": jnp.minimum,
            "tanh": lambda a: jnp.tanh(f(a)),
            "sqrt": lambda a: jnp.sqrt(f(a)),
            "sin": lambda a: jnp.sin(f(a)),
            "cos": lambda a: jnp.cos(f(a)),
            "abs": jnp.abs,
            "floor": lambda a: jnp.floor(f(a)),
            "stop_gradient": jax.lax.stop_gradient,  # identity forward, jax's honest spelling
            "where": lambda c, a, b: jnp.where(c.astype(bool) if isinstance(c, jax.Array) else c, a, b),
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "le": lambda a, b: a <= b,
            "lt": lambda a, b: a < b,
            "ge": lambda a, b: a >= b,
            "gt": lambda a, b: a > b,
        }


def run_region_jax(region, values, *, device="cpu", dtype=jnp.float64):
    """Evaluate a tensor-tier region over jax — ``values`` positional per
    param, each array in ITS PARAM's type-dim order (numpy or jax)."""
    return run_region_on(JaxSubstrate(device, dtype), region, values)


def run_named_jax(region, inputs, names, *, device="cpu", dtype=jnp.float64):
    """dialect.run_named's door on the jax substrate."""
    return run_named_on(JaxSubstrate(device, dtype), region, inputs, names)
