"""The refusal contract: guard MESSAGES are frozen behavior (the step-13
coverage ruling). A refusal whose wording drifts is an API break. The
array/transform rows died with the stdlib at migration P1 (design 200 §3.1);
their successors are pinned in pdum.tl's battery from P3 on."""

import pdum.dsl  # noqa: F401
import pytest
from pdum.dsl.api import jit
from pdum.dsl.lower import MissingRule, lower_handle
from pdum.dsl.ops import CORE_OPS
from pdum.dsl.reference import reference
from pdum.dsl.registry import DEFAULT
from pdum.dsl.types import f64


def test_derived_without_build_rule_is_a_missing_rule():
    """A DerivedValue whose tag has no build rule refuses by NAME."""
    from pdum.dsl.pipe import op

    @op
    def stage(k):
        @jit()
        def g(x):
            return x * k

        return g

    p = stage(2.0) | stage(3.0)
    with pytest.raises(MissingRule, match="no build rule for Derived template 'pipe'"):
        lower_handle(p, dict(DEFAULT.lower_rules), {**CORE_OPS, **DEFAULT.ops}, arg_types=(f64,), derived={})


def test_body_never_returns():
    @jit()
    def k(x):
        y = x * 2.0  # noqa: F841

    with pytest.raises(Exception, match="body never returns"):
        reference(k)(1.0)


def test_inline_arity_mismatch():
    @jit()
    def inner(a, b):
        return a + b

    @jit()
    def outer(x):
        return inner(x)  # one arg for a two-arg callee

    with pytest.raises(Exception, match="takes 2 args, got 1"):
        reference(outer)(1.0)
