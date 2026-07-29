"""pdum.rt — runtimes and backend emitters (283, the third package).

The seam this package owns, proven before it was built: ONE expression
generator serves every C-family target under four lexical rules; a
backend returns (source, launch-contract), never source alone; device
compilation goes THROUGH the content door, never around it; and
selection names a generator+runtime Pair that keys the artifact and
never implies a transfer.

Importing ``pdum.rt`` is always safe — device APIs (wgpu, Metal, CUDA)
import lazily at acquire/compile time and refuse loudly when absent.
"""

from .contract import Binding, LaunchContract
from .registry import acquire, executor_for, register_acquirer, register_compiler
from .select import (
    CudaGenerator,
    CudaRuntime,
    Generator,
    MetalRuntime,
    MslGenerator,
    Pair,
    Runtime,
    WgpuRuntime,
    WgslGenerator,
    cuda,
    metal,
    on,
    webgpu,
)

# The columns import LAST because they REGISTER on import (device APIs stay
# lazy inside). The cuda package joins them and registers NOTHING: it exists
# as a NAME so a program can spell the target, and the registry refuses
# toward its arrival (283 §5). The column is the L4 team's.
# isort: split
from . import cuda as _cuda  # noqa: E402,F401
from . import msl as _msl  # noqa: E402,F401
from . import wgsl as _wgsl  # noqa: E402,F401

__all__ = [
    "Binding",
    "LaunchContract",
    "Generator",
    "Runtime",
    "Pair",
    "WgslGenerator",
    "MslGenerator",
    "CudaGenerator",
    "WgpuRuntime",
    "MetalRuntime",
    "CudaRuntime",
    "webgpu",
    "metal",
    "cuda",
    "on",
    "acquire",
    "executor_for",
    "register_acquirer",
    "register_compiler",
]
