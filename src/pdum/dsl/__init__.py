"""pdum.dsl — a Python DSL compiler framework.

Core (backend-independent): capture/``Handle`` (phase A), the ``Type`` lattice +
``typeof``, and the type-keyed specialization cache + generation counter. The
WebGPU backend/runtime lives under ``pdum.dsl.webgpu`` and is imported on demand
(it pulls in ``wgpu``), so importing this package stays light for pure unit tests.
"""

from .cache import SpecCache, bump_generation, current_generation
from .jit import Handle, Program, jit, make_handle
from .lang import builtins
from .types import (
    BoolType,
    FloatType,
    FnType,
    IntType,
    NoneType,
    TupleType,
    Type,
    VecType,
    typeof,
    typeof_tuple,
)

__version__ = "0.1.0-alpha"

__all__ = [
    "__version__",
    # capture
    "jit",
    "make_handle",
    "Handle",
    "Program",
    "builtins",
    # types
    "Type",
    "IntType",
    "FloatType",
    "BoolType",
    "NoneType",
    "VecType",
    "TupleType",
    "FnType",
    "typeof",
    "typeof_tuple",
    # cache
    "SpecCache",
    "current_generation",
    "bump_generation",
]
