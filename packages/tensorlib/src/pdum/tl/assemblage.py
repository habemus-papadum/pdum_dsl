"""Assemblage (200 §S.1, §6): units, pipe, and the cached build.

A maker is a plain function ``(s, cfg) -> unit``; its returned tensor
function is ``@unit``-marked. ``|`` composes UNITS ONLY — build-time
function composition threading one value; a maker-level pipe would be a
third composition semantics punned onto the operator, which the spec
forbids. ``assemblage(u, **input_layouts)`` lowers the units into one
Program through the cache (the assemblage tier, §1.5): building a Program
is the compile step, and the build identity carries the units' code, the
INPUT layouts, the scope's policy map (IDENTITY-BEARING: train and eval
never collide), and the requested tap set (different tap sets never share
a derived Program).

Captured Params (s.param leaves) resolve to named Program inputs at
lowering; CAPTURE IDENTITY decides leaf identity — one Param object
captured by two units is ONE input leaf, and its gradient is the summed
contribution (the tie, zoo gate 9). Taps record their site path; requested
taps become named outputs, unrequested ones are pruned by DCE for free.
"""

from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass

from pdum.dsl.cache import Memo

from .lifting import _T, _Lifter
from .producer import _captured, _fn_ast
from .scope import Param, Scope
from .tensor import Tensor
from .transforms import dce

ASSEMBLAGES = Memo("assemblage", capacity=1 << 30)


@dataclass(frozen=True)
class Unit:
    """A tensor->tensor build fragment. ``|`` composes units only."""

    fns: tuple  # one or more plain functions, applied left to right

    def __or__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            raise TypeError(
                f"`|` composes UNITS only, got {other!r} — a maker-level pipe "
                f"would be a third composition semantics (200 §6.3); call the "
                f"maker first, then compose its unit"
            )
        return Unit(self.fns + other.fns)

    def fingerprint(self) -> tuple:
        out = []
        for fn in self.fns:
            code = fn.__code__
            out.append((code.co_qualname, hashlib.sha256(code.co_code).hexdigest()[:16]))
        return tuple(out)


def unit(fn) -> Unit:
    """Mark a maker's returned tensor function as a UNIT."""
    return Unit((fn,))


def pipe(units) -> Unit:
    """n-fold unit composition — what ``|`` folds to, and what seq returns."""
    units = list(units)
    if not units or not all(isinstance(u, Unit) for u in units):
        raise TypeError("pipe composes a non-empty sequence of units")
    out = units[0]
    for u in units[1:]:
        out = out | u
    return out


@dataclass(frozen=True)
class Assemblage:
    """The built model: one Program, named leaves, tap sites."""

    program: object
    inputs: tuple  # flowing input names, in order
    output: str  # the final SSA var
    params: dict  # flat contract name -> Param (declaration order preserved)
    taps: dict  # site path -> SSA var (every SITE; requested ones are outputs)
    outputs: tuple  # requested tap vars (+ the output) that survive in program


class _UnitLowerer(_Lifter):
    """The assemblage body lowerer: captured Params become named inputs
    (once per OBJECT — capture identity), taps record their sites, dropout
    is the mode-aware idiom."""

    def __init__(self, env: dict):
        super().__init__(env)
        self.param_vars: dict[int, _T] = {}  # id(Param) -> input _T
        self.params: dict[str, Param] = {}
        self.taps: dict[str, str] = {}

    def child(self, env: dict) -> "_UnitLowerer":
        inner = super().child(env)
        inner.param_vars, inner.params, inner.taps = self.param_vars, self.params, self.taps
        return inner

    def adopt(self, v):
        if isinstance(v, Param):
            got = self.param_vars.get(id(v))
            if got is None:
                self.b.input(v.name)
                self.shadows[v.name] = v.layout
                got = self.param_vars[id(v)] = _T(v.name, v.layout)
                self.params[v.name] = v
            return got
        return v

    def _i_tap(self, x, s: Scope):
        if not isinstance(x, _T):
            raise ValueError("tap takes a tensor and a scope site")
        self.taps[s.name] = x.var
        return x

    def _i_dropout(self, x, p, s: Scope):
        """The dropout IDIOM (§1.8): mode-aware, mask by closed-form field.
        Identity under eval — the mode branch lives here, never in user
        code; the mask is a constant field, so AD needs no new rule."""
        if not isinstance(x, _T):
            raise ValueError("dropout takes a tensor, a rate, and a scope site")
        if s.policy("mode", "train") == "eval":
            return x
        u = self.emit("random", (x.var,), "u", dist="uniform", key=s.stream())
        pc = self.const_like(float(p), x)
        kc = self.const_like(1.0 - float(p), x)
        z = self.const_like(0.0, x)
        m = self.pointwise("lt", u, pc, hint="mask")
        xs = self.pointwise("div", x, kc, hint="keep")
        return self.pointwise("where", m, z, xs, hint="drop")


def _lower(u: Unit, input_layouts: dict) -> tuple:
    lo = _UnitLowerer({})
    flow = None
    inputs = []
    for fn in u.fns:
        tree = _fn_ast(fn)
        params = [a.arg for a in tree.args.args]
        if len(params) != 1:
            raise ValueError(f"a unit takes exactly one flowing value, got {params} in {fn.__qualname__}")
        inner = lo.child(_captured(fn))
        if flow is None:  # the first unit names the program's flowing input
            name = params[0]
            layout = input_layouts.pop(name, None)
            if layout is None:
                raise ValueError(f"the flowing input {name!r} needs a layout: assemblage(u, {name}=<layout>)")
            layout = layout.layout if isinstance(layout, Tensor) else layout
            lo.b.input(name)
            lo.shadows[name] = layout
            flow = _T(name, layout)
            inputs.append(name)
        inner.env[params[0]] = flow
        outs = inner.run_body(tree)
        if len(outs) != 1:
            raise ValueError(f"a unit returns exactly one flowing value ({fn.__qualname__} returned {len(outs)})")
        flow = _T(outs[0], lo.shadows[outs[0]])
    if input_layouts:
        raise ValueError(f"unknown inputs bound: {sorted(input_layouts)}")
    return lo, tuple(inputs), flow.var


def assemblage(u: Unit, *, scope: Scope | None = None, taps: tuple = (), **input_layouts) -> Assemblage:
    """Build the Program — through the cache. Identity: the units' code,
    the input layouts, the scope's policies, and the requested tap set."""
    if not isinstance(u, Unit):
        raise TypeError(f"assemblage builds a Unit (got {u!r}) — mark the maker's function with @unit")
    layout_key = tuple(
        (k, tuple((d.name, d.start, d.stop) for d in (v.layout if isinstance(v, Tensor) else v).dims))
        for k, v in sorted(input_layouts.items())
    )
    policies = scope.policies if scope is not None else ()
    key = (u.fingerprint(), layout_key, policies, tuple(sorted(taps)))
    return ASSEMBLAGES.get_or_compile(key, lambda: _build(u, taps, dict(input_layouts)))


def _build(u: Unit, taps: tuple, input_layouts: dict) -> Assemblage:
    lo, inputs, out = _lower(u, input_layouts)
    prog = lo.b.program()
    requested = [v for site, v in sorted(lo.taps.items()) if any(fnmatch.fnmatch(site, p) for p in taps)]
    keep = (out, *requested)
    prog = dce(prog, keep) if len(prog.instrs) else prog
    return Assemblage(
        program=prog,
        inputs=inputs,
        output=out,
        params=dict(lo.params),
        taps=dict(lo.taps),
        outputs=keep,
    )
