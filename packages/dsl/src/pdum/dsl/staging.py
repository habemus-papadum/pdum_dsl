"""Staged transforms — the explicit door (240 C1, owner-ruled).

Host evaluation during body lowering is legal for STRUCTURAL results
(numbers, tuples, charts — the implicit regime, ratified) — but a host
call that returns a FUNCTION CITIZEN (an fp-carrying value: a Handle or
a Derived) is a staging act, and staging is DECLARED, never convention:
the callable must be decorated ``@staged``, or the lowering refuses
loudly.

A staged transform is an ordinary function — functional and composable
by ruling. When one executes during lowering, the lowerer records the
call as a REPLAYABLE RECIPE against its result; at launch the recipe
chain re-applies the transforms to the CURRENT parameter values, so a
warm cache hit never serves a stale capture. Recipes chain through
composition: ``t2(t1(f))`` restages through both. The contract a staged
transform must honor (device parity, 240 §I.4): its output may depend
on its inputs' IDENTITY (fp — code and types), never on captured
runtime VALUES; values flow only at execution.
"""

from __future__ import annotations


def staged(fn):
    """Declare ``fn`` a staged transform (see the module docstring)."""
    fn.__staged__ = True
    return fn
