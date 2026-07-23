"""tensorlib: a memory-layout-and-views exploration (no compute layer).

Step 1 — core family: affine map + box domain (`Layout`).
Step 2 — extension:  affine map + guards + fill (`GuardedLayout`).
Step 3 — labeling:   exact units (`Quantity`, `u`, `q`) and per-dim
                     coordinate charts (`Chart`) over the unchanged lattice.
"""

from .buffer import Buffer, FunctionalBuffer
from .build import Build
from .chart import Chart, characteristic, chart
from .compute import Marker, Reducer, iota, pointwise, pw, red, reduce, scan
from .dtypes import CARRIERS, as_dtype, bfloat16, carrier_of
from .guarded import Guard, GuardedLayout, pad_layout, stencil_layout
from .layout import Dim, Injectivity, Layout, as_range
from .mdsl import CompositeMarker, CompositeReducer, defmarker, defreducer, node_digest
from .memory import MemoryReport, peak_memory
from .opcount import ProgramOps, ops_count
from .signatures import SignatureError, VInfo, infer_signatures, marker_signature
from .tensor import Misalignment, Tensor, aligned, alignment
from .units import Quantity, Unit, UnitRegistry, q, u

__all__ = [
    "Buffer",
    "CARRIERS",
    "Build",
    "CompositeMarker",
    "CompositeReducer",
    "MemoryReport",
    "peak_memory",
    "ProgramOps",
    "SignatureError",
    "VInfo",
    "defmarker",
    "defreducer",
    "infer_signatures",
    "marker_signature",
    "node_digest",
    "ops_count",
    "FunctionalBuffer",
    "Marker",
    "carrier_of",
    "scan",
    "Misalignment",
    "Reducer",
    "iota",
    "pointwise",
    "pw",
    "red",
    "reduce",
    "aligned",
    "alignment",
    "characteristic",
    "Chart",
    "Dim",
    "Guard",
    "GuardedLayout",
    "Injectivity",
    "Layout",
    "Quantity",
    "Tensor",
    "Unit",
    "UnitRegistry",
    "as_dtype",
    "as_range",
    "bfloat16",
    "chart",
    "pad_layout",
    "q",
    "stencil_layout",
    "u",
]
