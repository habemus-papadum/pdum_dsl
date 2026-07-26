"""The staging family — declared, never convention (240 C1/C2).

Two kinds of compile-time citizens, one doctrine: anything that runs at
LOWER TIME and shapes IR is DECLARED with a decorator from this module;
recognition is declaration-first at every tier, and the undeclared case
refuses loudly. Both are ordinary functions — functional and composable
by ruling; sequences of transformations compose from smaller ones.

- ``@macro`` — a VALUE-SPACE lowering macro: called inside a body, it
  receives the lowering context and the already-lowered argument IR and
  emits transformed IR into the same region (``with_respect_to`` is the
  archetype). IR-in, IR-out, at the call site.

- ``@staged`` — a FUNCTION-SPACE staged transform: it may be host-called
  during body lowering and returns a function CITIZEN (an fp-carrying
  Handle/Derived) whose IR comes from a registered build rule when it
  lowers (``value_and_grad`` is the archetype). Citizen-in, citizen-out;
  the "generated function" model, fp-keyed and cache-efficient. When one
  executes during lowering, the lowerer records the call as a REPLAYABLE
  RECIPE against its result; at launch the recipe chain re-applies the
  transforms to the CURRENT parameter values, so a warm hit never serves
  a stale capture. Recipes chain through composition: ``t2(t1(f))``
  restages through both.

The contract both kinds must honor (device parity, 240 §I.4): output may
depend on the inputs' IDENTITY (fp — code and types), never on captured
runtime VALUES; values flow only at execution. Structural host
evaluation (numbers, tuples, charts) stays implicit — the doctrine
governs function citizens only.
"""

from __future__ import annotations


def staged(fn):
    """Declare ``fn`` a staged transform (see the module docstring)."""
    fn.__staged__ = True
    return fn


def macro(fn):
    """Declare ``fn`` a value-space lowering macro: it receives
    ``(ctx, args, node)`` — the lowering context, the lowered argument
    nodes, and the source location — and returns the IR it emitted."""
    fn.__lower_macro__ = True
    return fn
