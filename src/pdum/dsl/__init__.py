"""pdum.dsl — a Python DSL compiler framework.

``import pdum.dsl`` is BATTERIES INCLUDED: it wires the base-language rule
pack, the "device" role + pipe fusion, and the Python backend into the
DEFAULT registry, so ``@jit`` kernels are callable immediately. (Python
import semantics: any ``pdum.dsl.*`` import runs this file first, so the
batteries always ride along. The registration-free world the chapters probe
is a hand-built ``kernel.registry.Registry()``, populated explicitly via the
satellites' ``install(registry)`` functions.)

This package is the home of the redesigned framework; the redesign is being
driven by ``docs/desiderata.md``. The previous end-to-end proof of concept
(Milestone 0) is preserved intact as a frozen reference asset at
``pdum.dsl_reference`` — see ``reference/README.md`` for how to run its tests
and demo.
"""

from . import backends as _backends  # noqa: F401  — registers the Python backend
from . import stdlib as _stdlib  # noqa: F401  — registers the base dialect
from .kernel.api import jit
from .kernel.cache import no_compile

__version__ = "0.1.0-alpha"

__all__ = ["__version__", "jit", "no_compile"]
