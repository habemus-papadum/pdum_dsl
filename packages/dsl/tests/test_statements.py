"""Step 11 — statement `if`/`for`: strict joins, loop carries, single tail
return, and the refusals that define the bounded-loop subset."""

import pdum.dsl  # noqa: F401
import pytest
from pdum.dsl.api import jit
from pdum.dsl.cache import no_compile
from pdum.dsl.ir import VerifyError
from pdum.dsl.lower import MissingRule
from pdum.dsl.reference import reference


def test_if_statement_joins_and_stays_lazy():
    @jit()
    def f(x):
        y = 0.0
        if x > 0.0:
            y = 1.0 / x  # division must stay INSIDE the branch (guard-then-divide)
        else:
            y = 0.0
        return y

    assert reference(f)(4.0) == 0.25
    assert reference(f)(0.0) == 0.0  # an eager 1.0/x would have raised ZeroDivisionError


def test_if_statement_multi_name_join():
    @jit()
    def f(x):
        a = 0.0
        b = 1.0
        if x > 0.5:
            a = x
            b = x * 2.0
        else:
            a = -x
        return a + b

    assert reference(f)(1.0) == 3.0
    assert reference(f)(0.25) == 0.75  # b keeps its pre-if value on the else path


def test_for_loop_single_carry_and_zero_trip():
    def make(n):
        @jit()
        def f(x):
            acc = x
            for i in range(n):
                acc = acc + float(i)
            return acc

        return f

    assert reference(make(4))(1.0) == 1.0 + 0 + 1 + 2 + 3
    assert reference(make(0))(7.0) == 7.0  # zero-trip: the carry is the init, defined


def test_nested_loops_and_branches_match_python():
    @jit()
    def f(x):
        total = 0.0
        for i in range(3):
            row = 0.0
            for j in range(4):
                v = float(i * 4 + j)
                if v % 2.0 == 0.0:
                    row = row + v
                else:
                    row = row + x
            total = total + row
        return total

    def twin(x):
        total = 0.0
        for i in range(3):
            row = 0.0
            for j in range(4):
                v = float(i * 4 + j)
                row = row + (v if v % 2.0 == 0.0 else x)
            total = total + row
        return total

    assert abs(reference(f)(0.5) - twin(0.5)) < 1e-12


def test_loop_specializes_on_types_not_trip_count_values():
    """Captured bounds are uniforms like any capture: new bound VALUE, same
    types -> cache hit (the loop body re-executes with new staging bytes)."""

    def make(n):
        @jit()
        def f(x):
            acc = x
            for i in range(n):
                acc = acc * 2.0
            return acc

        return f

    assert reference(make(3))(1.0) == 8.0
    with no_compile():  # n is an i64 capture: 5 vs 3 is a VALUE change
        assert reference(make(5))(1.0) == 32.0


def test_single_tail_return_enforced():
    @jit()
    def f(x):
        if x > 0.0:
            return x  # refused: return inside a branch
        return -x

    with pytest.raises(MissingRule, match="single tail return"):
        reference(f)(1.0)


def test_bounded_subset_refusals():
    @jit()
    def w(x):
        while x > 0.0:
            x = x - 1.0
        return x

    with pytest.raises(MissingRule, match="while.*not in the base pack"):
        reference(w)(1.0)

    @jit()
    def b(x):
        acc = 0.0
        for i in range(3):
            acc = acc + x
            break
        return acc

    with pytest.raises(MissingRule, match="single-entry single-exit"):
        reference(b)(1.0)


def test_loop_variable_shadowing_refused():
    @jit()
    def f(x):
        i = 1.0
        for i in range(3):  # noqa: B020
            x = x + 1.0
        return x + i

    with pytest.raises(MissingRule, match="shadows an existing name"):
        reference(f)(1.0)


def test_one_sided_names_die_with_their_suite():
    """Branch-local temporaries are LEGAL (they die with the suite, like
    loop-locals die with the loop); what is refused is depending on one."""

    @jit()
    def ok(x):
        y = 0.0
        if x > 0.0:
            t = x * x  # scratch name, born and consumed inside the branch
            y = t + 1.0
        else:
            y = -x
        return y

    assert reference(ok)(2.0) == 5.0

    @jit()
    def dead_if(x):
        if x > 0.0:
            y = x  # noqa: F841 — nothing survives this if
        else:
            pass
        return x

    with pytest.raises(MissingRule, match="binds nothing that survives"):
        reference(dead_if)(1.0)

    from pdum.dsl.lower import NameFateError

    @jit()
    def reads_dead(x):
        y = 0.0
        if x > 0.0:
            t = x
            y = t
        else:
            y = 1.0
        return t + y  # noqa: F821 — t died with its suite; y survived the join

    with pytest.raises(NameFateError):
        reference(reads_dead)(1.0)


def test_nested_unsupported_construct_names_itself():
    """The carry prescan cannot see into unruled constructs; the refusal the
    user gets must still be the REAL one (review-caught masking)."""

    @jit()
    def f(x):
        acc = 0.0
        for i in range(3):
            while acc < 1.0:
                acc = acc + x
        return acc

    with pytest.raises(MissingRule, match="while.*not in the base pack"):
        reference(f)(1.0)


def test_loop_binder_keys_are_deterministic():
    """Binder indices are derived from source position only — the same
    kernel must produce the SAME content key no matter what lowered before
    it through a shared rules dict (review-caught: a shared counter made
    artifact keys depend on process history)."""
    from pdum.dsl import types as T
    from pdum.dsl.lower import lower_handle
    from pdum.dsl.ops import CORE_OPS
    from pdum.dsl.registry import DEFAULT

    def build():
        @jit()
        def f(x):
            acc = x
            for i in range(3):
                acc = acc + 1.0
            return acc

        return f

    @jit()
    def other(x):
        s = x
        for i in range(9):
            s = s * 2.0
        return s

    ops = {**CORE_OPS, **DEFAULT.ops}
    rules = dict(DEFAULT.lower_rules)
    k1 = lower_handle(build(), rules, ops, arg_types=(T.f64,)).key
    lower_handle(other, rules, ops, arg_types=(T.f64,))  # would advance any shared counter
    k2 = lower_handle(build(), rules, ops, arg_types=(T.f64,)).key
    assert k1 == k2


def test_strict_join_type_mismatch_is_loud():
    @jit()
    def f(x):
        y = 0.0
        if x > 0.0:
            y = 1
        else:
            y = 1.0
        return float(y)

    with pytest.raises((TypeError, VerifyError), match="strict join"):
        reference(f)(1.0)
