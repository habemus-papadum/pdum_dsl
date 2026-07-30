"""Grid & launch (340 §7.2–3): the feasibility scorer and the proposal
ladder — the T=512 shared-memory crash becomes a computed decision."""

from pdum.tl.fusion import plan_region
from pdum.tl.launch import TileLevel, TileMachine, feasible, propose
from pdum.tl.zoo.tiles import flash_tile

# the 4090's opt-in shared-memory limit, as triton's OutOfResources
# reported it at the measured crash — a machine-table row, not a probe
_SHARED = TileMachine((TileLevel("shared", 128, 101376),))


def test_the_measured_crash_is_now_a_computed_refusal():
    """Flash at T=512, D=64 required 231KB against 101KB on silicon.
    The scorer decides that without compiling anything."""
    region = flash_tile(T=512, E=64, OD=64, SI=32).region
    ok_full, fp_full = feasible(region, (("t", 512),), _SHARED)
    ok_32, fp_32 = feasible(region, (("t", 32),), _SHARED)
    assert not ok_full and fp_full > _SHARED.tightest()
    assert ok_32 and fp_32 <= _SHARED.tightest()


def test_the_ladder_is_smallest_first_and_all_feasible():
    """Smallest tile above the floor = the analytic default (maximum
    oversubscription); every rung fits; the infeasible top rungs are
    simply absent."""
    region = flash_tile(T=512, E=64, OD=64, SI=32).region
    cands = propose(region, _SHARED, floor=1024)
    tiles = [dict(c)["t"] for c in cands]
    assert tiles == sorted(tiles) and tiles[0] == 16  # 16*64 elems = the floor
    assert all(feasible(region, c, _SHARED)[0] for c in cands)
    assert 512 not in tiles and 256 not in tiles  # the computed cliff


def test_plan_region_attaches_the_analytic_default():
    f = flash_tile(T=128, E=32, OD=32, SI=2)
    bare = plan_region(f.naive)
    (g0,) = bare.groups
    assert g0.launch == ()
    (g,) = plan_region(f.naive, machine=_SHARED).groups
    assert g.template == "flash" and g.launch and g.launch[0][0] == "t"
