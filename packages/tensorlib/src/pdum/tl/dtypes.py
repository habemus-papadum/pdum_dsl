"""DTypes: numpy dtypes are the vocabulary.

Structured dtypes (with padding, e.g. align=True) come for free from numpy.
Extended float types (bfloat16 etc.) plug into the numpy dtype system via
ml_dtypes when it is installed.
"""

from __future__ import annotations

import numpy as np

try:  # optional
    import ml_dtypes as _ml

    bfloat16: np.dtype | None = np.dtype(_ml.bfloat16)
except ImportError:  # pragma: no cover
    bfloat16 = None


def as_dtype(spec) -> np.dtype:
    return np.dtype(spec)


# Carriers: the algebraic object a tensor's values APPROXIMATE — the
# semantic half of the value type, with the machine dtype demoted to pure
# representation (a footprint/cost resource, never semantics). Coercion
# chain: bool -> int -> rat -> real -> complex. "rat" is never inferred from
# a dtype — it arises only by declaration (iota's physical face, chart-
# derived data), which is the point: exact-rational values represented in
# floats keep their exact semantics on record.
CARRIERS = ("bool", "int", "rat", "real", "complex")


def carrier_of(dtype) -> str | None:
    """The carrier a machine dtype conventionally approximates; None for
    structured/exotic dtypes (annotate those explicitly)."""
    dt = np.dtype(dtype)
    if dt.kind == "b":
        return "bool"
    if dt.kind in "iu":
        return "int"
    if dt.kind == "f":
        return "real"
    if dt.kind == "c":
        return "complex"
    return None
