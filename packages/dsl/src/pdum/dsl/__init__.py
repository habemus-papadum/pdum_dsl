"""pdum.dsl — the compiler-infrastructure core (design 200 §1.1).

``import pdum.dsl`` is BATTERIES INCLUDED: it wires the value language, the
scalar intrinsics, the fuse pipe, and the reference oracle into the DEFAULT
registry. Plain calls on a kind with no routed backend REFUSE — there are no
device backends until the L4 era, and oracle execution is always spelled:
``reference(f)(...)``.

The registration-free world is a hand-built ``Registry()``, populated
explicitly via ``install(registry)``.

This file is the lockstep version anchor (scripts/_versioning.py).
"""

from . import pack as _pack  # noqa: F401 — registers the marshaling aspects on BUILTINS

# Importing the package registers them, so ANY entry point gets a table that
# can plan. Without this, ``extend()`` — which snapshots the aspect registry —
# could mint a child table that is permanently unable to marshal.
from .api import jit  # noqa: E402
from .cache import no_compile  # noqa: E402
from .pipe import op  # noqa: E402
from .reference import reference  # noqa: E402
from .registry import DEFAULT, Registry  # noqa: E402
from .types import Literal  # noqa: E402 — the §1.5 annotation door

__version__ = "0.0.0+dev"


def _fold_tuple_extract(b, m):
    return m["root"].args[0].args[dict(m["root"].attrs)["index"]]


def install(registry: Registry) -> Registry:
    """Register the batteries into ``registry`` (idempotent): the value
    language's rule pack, the "device" kind + pipe fusion, the reference
    oracle, and the scalar intrinsics."""
    from . import intrinsics, pipe
    from .reference import install as _install_reference
    from .rewrite import Pat
    from .value import LOWER_RULES

    registry.lower_rules.update(LOWER_RULES)
    pipe.install(registry)  # the materializer kind + the fusion build rule
    # extract-of-tuple folds away wherever the target cannot spell tuples —
    # gated on "core.tuple" ∈ code_for_op, the same mechanism as decompositions:
    if not any(op_name == "core.tuple" for op_name, _ in registry.decompositions):
        tuple_extract = Pat("core.extract", args=("t",), guard=lambda m: m["t"].op == "core.tuple")
        registry.decompositions.append(("core.tuple", (tuple_extract, _fold_tuple_extract)))
    registry.register_kind("device", hint="the value language's neutral composable kernel")
    registry.compositions[("pipe", "device", "device")] = "fuse"
    if registry.dispatcher is None:  # never clobber a dispatcher installed first
        registry.dispatcher = lambda pipeline, value: registry.dispatch(pipeline, (value,))
    _install_reference(registry)  # the oracle: registered, never routed, never default
    intrinsics.install(registry)  # spells onto the reference record just installed
    return registry


install(DEFAULT)

__all__ = ["DEFAULT", "Registry", "__version__", "install", "Literal", "jit", "no_compile", "op", "reference"]
