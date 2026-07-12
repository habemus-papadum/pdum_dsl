"""The standard library satellite: rule packs and (later) batteries.

Everything here attaches through registration surfaces; the kernel never
imports this package. Line counts live in the SATELLITE budget bucket
(`scripts/loc_budget.py`), separately capped from the kernel.

The BASE DIALECT ships as an ``install(registry)`` function — importing this
package calls ``install(DEFAULT)`` (batteries), but a hand-built ``Registry``
can receive the exact same dialect explicitly (test isolation; multi-registry
sessions; the step-10 ``extend()`` story needs this seam anyway). Contents:
the base-language lowering pack, the "device" role (`@jit`'s default — role
vocabularies ship with their owning package, per the ch04 decision), the pipe
fusion rule, and the pipeline dispatcher (`value > pipeline` executes through
the same two-tier path as a call). Chapters before ch09 registered these by
hand as labeled stand-ins; from here on the batteries are included.
"""

from .. import combinators as _comb
from ..combinators import PIPE_BUILDERS, register_composition, register_role, set_dispatcher
from ..kernel.registry import DEFAULT, Registry
from .base_lang import LOWER_RULES


def install(registry: Registry) -> Registry:
    """Register the base dialect into ``registry`` (idempotent)."""
    registry.lower_rules.update(LOWER_RULES)
    registry.derived.update(PIPE_BUILDERS)
    register_role("device", hint="the base language's neutral composable kernel")
    register_composition("pipe", "device", "device", "fuse")
    if _comb._DISPATCHER is None:  # live check — never clobber a dispatcher installed first
        set_dispatcher(lambda pipeline, value: registry.dispatch(pipeline, (value,)))
    return registry


install(DEFAULT)
