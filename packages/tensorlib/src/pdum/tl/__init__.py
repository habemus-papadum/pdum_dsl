"""pdum.tl — the assemblage tensor language (design 200 §1.1).

The exact layout algebra (affine map + box domain, guards + fill, units and
per-dim charts over the unchanged lattice), the compute primitives, the
Program/Instr IR, reverse-mode AD, the transforms (DCE, checkpointing), the
cost semantics, placement, signatures, and the model zoo. Promoted from
explorations/tensorlib at migration P2; converts onto pdum.dsl's caching,
naming, and capture at P3.

Version of record: this distribution is version-locked with
``habemus-papadum-dsl`` (scripts/_versioning.py); the anchor lives in
``pdum.dsl.__version__``.
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
from .placement import Collective, Level, Machine, TrafficReport, mesh, traffic
from .signatures import SignatureError, VInfo, infer_signatures, marker_signature
from .tensor import Misalignment, Tensor, aligned, alignment
from .transforms import CheckpointReport, checkpoint, dce
from .units import Quantity, Unit, UnitRegistry, q, u

__all__ = [
    "Buffer",
    "CARRIERS",
    "Build",
    "CheckpointReport",
    "checkpoint",
    "dce",
    "CompositeMarker",
    "CompositeReducer",
    "Collective",
    "Level",
    "Machine",
    "MemoryReport",
    "TrafficReport",
    "mesh",
    "peak_memory",
    "traffic",
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
