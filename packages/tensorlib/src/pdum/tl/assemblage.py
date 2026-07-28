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

import ast as pyast
import fnmatch
import hashlib
from dataclasses import dataclass

from pdum.dsl.cache import Memo
from pdum.dsl.ir import Region
from pdum.dsl.lower import Lowerer, check_coherence
from pdum.dsl.registry import DEFAULT

from .dialect import (
    CORE_OPS,
    TL_OPS,
    TL_RULES,
    TensorType,
    _host,
    _lookup,
    _scalar_lift,
    _tl_call,
    _tl_name,
    capture_shim,
    region_names,
    tensor_type_of_layout,
)
from .lifting import _Intrinsic
from .producer import _fn_ast
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
        """Build identity: each unit's code AND its captured values — a
        different cfg knob, Param set, or placement is a different build
        (captured values are all structural at the assemblage tier)."""
        out = []
        for fn in self.fns:
            code = fn.__code__
            cells = []
            for name, cell in zip(code.co_freevars, fn.__closure__ or ()):
                try:
                    cells.append((name, _canon(cell.cell_contents)))
                except ValueError:
                    cells.append((name, "<empty>"))
            out.append((code.co_qualname, hashlib.sha256(code.co_code).hexdigest()[:16], tuple(cells)))
        return tuple(out)


def _canon(v) -> object:
    """A hashable identity for a captured value."""
    if isinstance(v, Param):
        return ("param", v.name, v.dims, v.levels)
    if isinstance(v, (int, float, str, bool, type(None), bytes)):
        return v
    if isinstance(v, tuple):
        return tuple(_canon(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _canon(x)) for k, x in v.items()))
    if callable(v) and hasattr(v, "__code__"):
        return ("fn", v.__code__.co_qualname, hashlib.sha256(v.__code__.co_code).hexdigest()[:16])
    return repr(v)  # frozen dataclasses (cfg), markers, scopes: deterministic reprs


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
    """The built model: one region, named leaves, tap sites."""

    region: object  # the dialect Region, dce'd against the requested outputs
    inputs: tuple  # flowing input names, in order
    output: str  # the final value's name
    params: dict  # flat contract name -> Param (declaration order preserved)
    taps: dict  # site path -> name (every SITE; requested ones are outputs)
    outputs: tuple  # requested tap names (+ the output) that survive the dce
    layouts: dict  # EVERY live input's layout — what virtual analyses consume
    names: dict  # the naming law's assignment (id-keyed; dce-stable)


def _u_name(ctx, node):
    """The unit layer of name resolution: a captured Param ADOPTS as a
    named input — once per OBJECT (capture identity: one Param captured by
    two units is ONE leaf; its gradient sums)."""
    return _adopt(ctx, _tl_name(ctx, node))


def _adopt(ctx, v):
    if isinstance(v, Param):
        c = ctx.context
        got = c["u.param_nodes"].get(id(v))
        if got is None:
            got = ctx.builder.param(("u", 1 + len(c["u.param_nodes"])), tensor_type_of_layout(v.layout))
            c["u.param_nodes"][id(v)] = got
            c["u.params"][v.name] = v
            c["u.param_order"].append((v.name, got))
        return got
    return v


def _u_call(ctx, node):
    """The unit layer of call resolution: scope-path taps and the
    mode-aware dropout idiom; everything else is the shared dialect pack."""
    if isinstance(node.func, pyast.Name):
        obj = _lookup(ctx, node.func.id)
        if isinstance(obj, _Intrinsic) and obj.name == "tap":
            x = ctx.lower(node.args[0])
            site = _host(ctx, node.args[1])
            if not (hasattr(x, "type") and isinstance(x.type, TensorType) and isinstance(site, Scope)):
                raise ValueError("tap takes a tensor and a scope site")
            ctx.context["u.taps"][site.name] = x
            return x
        if isinstance(obj, _Intrinsic) and obj.name == "dropout":
            x = ctx.lower(node.args[0])
            if not (hasattr(x, "type") and isinstance(x.type, TensorType)):
                raise ValueError("dropout takes a tensor, a rate, and a scope site")
            p, site = float(_host(ctx, node.args[1])), _host(ctx, node.args[2])
            if site.policy("mode", "train") == "eval":  # identity under eval — the mode
                return x  # branch lives HERE, never in user code (§1.8)
            names = ctx.context.setdefault("tl.names", {})
            u = ctx.emit("tl.random", x, node=node, dist="uniform", key=site.stream())
            names.setdefault(id(u), "u")
            m = ctx.emit("tl.pointwise", u, _scalar_lift(ctx, node, p), node=node, f="lt")
            names.setdefault(id(m), "mask")
            xs = ctx.emit("tl.pointwise", x, _scalar_lift(ctx, node, 1.0 - p), node=node, f="div")
            names.setdefault(id(xs), "keep")
            drop = ctx.emit("tl.pointwise", m, _scalar_lift(ctx, node, 0.0), xs, node=node, f="where")
            names.setdefault(id(drop), "drop")
            return drop
    return _tl_call(ctx, node)


UNIT_RULES = {**TL_RULES, pyast.Name: _u_name, pyast.Call: _u_call}


def _lower(u: Unit, input_layouts: dict) -> tuple:
    """Chain the unit fns' bodies through ONE dsl-Lowerer build (shared
    context; one lowerer per fn, nodes are values): the first unit names
    the flowing input; captured Params adopt as they are seen."""
    ops = {**CORE_OPS, **TL_OPS}
    names_led: dict = {}
    context = {
        "registry": DEFAULT,
        "tl.kind": "unit",
        "tl.names": names_led,
        "u.param_nodes": {},
        "u.params": {},
        "u.param_order": [],
        "u.taps": {},
    }
    flow = flow_param = flow_name = None
    for fn in u.fns:
        handle = capture_shim(fn)
        check_coherence(handle)
        ctx = Lowerer(handle, UNIT_RULES, ops, {}, context=context)
        params = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])
        if len(params) != 1:
            raise ValueError(f"a unit takes exactly one flowing value, got {params} in {fn.__qualname__}")
        if flow is None:  # the first unit names the program's flowing input
            name = params[0]
            layout = input_layouts.pop(name, None)
            if layout is None:
                raise ValueError(f"the flowing input {name!r} needs a layout: assemblage(u, {name}=<layout>)")
            layout = layout.layout if isinstance(layout, Tensor) else layout
            flow_param = ctx.builder.param(("u", 0), tensor_type_of_layout(layout))
            flow, flow_name = flow_param, name
        ctx.locals[params[0]] = flow
        ctx.params = (flow_param,)
        tree = _fn_ast(fn)
        if isinstance(tree, pyast.Lambda):
            result = ctx.lower(tree.body)
        else:
            for stmt in tree.body[:-1]:
                ctx.lower(stmt)
            last = tree.body[-1]
            if not isinstance(last, pyast.Return) or last.value is None:
                raise ValueError(f"a unit returns exactly one flowing value ({fn.__qualname__} returned none)")
            result = ctx.lower(last.value)
        if isinstance(result, tuple) or getattr(result, "op", None) == "core.tuple":
            n = len(result) if isinstance(result, tuple) else len(result.args)
            raise ValueError(f"a unit returns exactly one flowing value ({fn.__qualname__} returned {n})")
        flow = result
    if input_layouts:
        raise ValueError(f"unknown inputs bound: {sorted(input_layouts)}")
    c = context
    sites = sorted(c["u.taps"])
    tap_nodes = [c["u.taps"][site] for site in sites]
    yielded = flow if not tap_nodes else ctx.builder.emit("core.tuple", flow, *tap_nodes)
    region = Region(
        params=(flow_param, *(node for _, node in c["u.param_order"])),
        body=(ctx.builder.emit("core.yield", yielded),),
    )
    in_names = (flow_name, *(n for n, _ in c["u.param_order"]))
    mapping = region_names(region, in_names, names_led)
    taps = {site: mapping[id(node)] for site, node in zip(sites, tap_nodes)}
    layouts = {flow_name: region.params[0].type.layout}
    layouts.update({n: p.layout for n, p in c["u.params"].items()})
    return in_names, mapping[id(flow)], dict(c["u.params"]), taps, layouts, region, mapping


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
    in_names, out, params, tap_vars, layouts, region, mapping = _lower(u, input_layouts)
    requested = [v for site, v in sorted(tap_vars.items()) if any(fnmatch.fnmatch(site, p) for p in taps)]
    keep = (out, *requested)
    region = dce(region, keep, names=mapping)  # re-yield + reachability pruning
    live = {mapping[id(p)] for p in region.params}
    return Assemblage(
        region=region,
        inputs=(in_names[0],),
        output=out,
        params=params,
        taps=tap_vars,
        outputs=keep,
        layouts={n: v for n, v in layouts.items() if n in live},
        names=mapping,
    )
