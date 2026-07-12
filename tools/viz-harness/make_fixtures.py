"""Generate standalone HTML fixtures for the jsdom harness (and browser eyeballing)."""

from pdum.dsl import viz
from pdum.dsl.combinators import op, register_composition, register_role
from pdum.dsl.kernel import types as T
from pdum.dsl.kernel.api import jit
from pdum.dsl.kernel.ir import Builder, Loc, Region
from pdum.dsl.kernel.ops import CORE_OPS

register_role("device")
register_composition("pipe", "device", "device", "fuse")


def scale(k):
    @jit()
    def go(x):
        return x * k

    return go


@op
def stage(k):
    @jit()
    def go(x):
        return x + k

    return go


b = Builder(CORE_OPS)
e = b.emit("core.env", type=T.f64, slot=(0,), loc=Loc("art.py", 7))
region = Region(body=(b.emit("core.yield", b.emit("core.mul", e, e)),))

viz.save(scale(2.0), "tools/viz-harness/fixtures/handle.html", "Handle")
viz.save(region, "tools/viz-harness/fixtures/region.html", "Region")
viz.save(stage(1.0) | stage(2.0), "tools/viz-harness/fixtures/pipeline.html", "Pipeline")
print("fixtures written")
