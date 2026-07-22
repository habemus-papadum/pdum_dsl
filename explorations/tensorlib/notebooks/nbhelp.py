"""Display helpers shared by the tensorlib notebooks.

Importing this module also puts the exploration root on sys.path so the
notebooks can `from tensorlib import ...` without installing anything.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensorlib import GuardedLayout, carrier_of  # noqa: E402


def show(t, title=""):
    """Print a tensor's layout internals: offset, dims table (with charts),
    analyses, guards, fill, and value units."""
    lay = t.layout
    guarded = isinstance(lay, GuardedLayout)
    base = lay.base if guarded else lay
    if title:
        print(f"-- {title}")
    head = f"Tensor[{t.dtype}]"
    if t.carrier != carrier_of(t.dtype):
        head = f"Tensor[{t.dtype} approximating {t.carrier}]"
    print(f"{head} on {t.buffer!r}")
    print(f"  offset : {base.offset} bytes")
    print(f"  {'dim':<6}{'stride':>8}{'start':>7}{'stop':>7}{'size':>7}  chart")
    for d in base.dims:
        if d.chart is not None:
            ch = f"{d.chart!r}"
        elif d.labels is not None:
            ch = f"#[{', '.join(d.labels)}]"
        else:
            ch = ""
        print(f"  {d.name:<6}{d.stride:>8}{d.start:>7}{d.stop:>7}{d.size:>7}  {ch}")
    print(f"  numel={t.numel}  footprint={t.footprint()}  injectivity={t.injectivity().value}")
    if guarded:
        for g in lay.guards:
            print(f"  {g!r}")
        print(f"  fill   : {t.fill!r}")
    if t.value_units is not None:
        print(f"  values : {t.value_units!r}")


def formula(t):
    """Print the layout's address map as the affine formula it is."""
    lay = t.layout
    base = lay.base if isinstance(lay, GuardedLayout) else lay
    terms = " + ".join(f"{d.stride}*{d.name}" for d in base.dims)
    body = f"{base.offset}" + (f" + {terms}" if terms else "")
    print(f"loc = {body}   (bytes)")
