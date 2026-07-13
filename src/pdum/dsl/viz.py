"""Rich static HTML rendering for kernel objects, in Jupyter AND marimo.

**Satellite** (zero kernel edits; design note `docs/design/060_rendering-notes.md`).
The contract: renderings are *static like a PNG* — no kernel<->JS channel.
Interactivity is CSS-only in v1 (``<details>`` collapsing, ``:hover``
attr-tooltips), which survives every notebook sandbox. Later widgets may pull
modern ESM (CodeMirror, …) from CDNs; nothing here requires it.

Composability: every renderer returns an :class:`Html` whose ``fragment`` is
style-free inner HTML; containers embed children's fragments and the single
scoped ``<style>`` is emitted once at the display root (``.pdum-viz`` class
scoping keeps notebook/site CSS untouched).

Usage: ``from pdum.dsl import viz; viz.install()`` — attaches ``_repr_html_``
to Handle/Region/Pipeline/…, so bare objects render richly. ``viz.save(obj,
path)`` writes a standalone page for the browser/jsdom dev loop
(``tools/viz-harness``).
"""

from __future__ import annotations

import html as _html

from .bench import Timeline
from .combinators import Pipeline, Stage, Terminal
from .kernel.cache import SpecializationCache, _TierCache
from .kernel.capture import Handle
from .kernel.ir import Node, Region, format_loc
from .kernel.rewrite import MatchLog
from .kernel.types import Type

_CSS = """
.pdum-viz{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;
  line-height:1.5;color-scheme:light dark}
.pdum-viz .pv-card{border:1px solid rgba(127,127,127,.35);border-radius:8px;padding:.6em .8em;
  margin:.25em 0;background:rgba(127,127,127,.06)}
.pdum-viz .pv-hd{display:flex;gap:.5em;align-items:center;flex-wrap:wrap;margin-bottom:.35em}
.pdum-viz .pv-name{font-weight:600}
.pdum-viz .pv-pill{display:inline-block;border-radius:999px;padding:0 .6em;font-size:.85em;
  background:rgba(127,127,127,.18);white-space:nowrap}
.pdum-viz .pv-pill.ty{background:rgba(70,130,220,.22)}
.pdum-viz .pv-pill.id{background:rgba(160,110,220,.22)}
.pdum-viz .pv-pill.num{background:rgba(80,170,120,.22)}
.pdum-viz .pv-pill.warn{background:rgba(220,140,60,.28)}
.pdum-viz pre{margin:.3em 0;padding:.5em .7em;border-radius:6px;background:rgba(127,127,127,.09);
  overflow-x:auto}
.pdum-viz .pv-ir{white-space:pre;overflow-x:auto}
.pdum-viz .pv-k{color:#a06edc;font-weight:600}
.pdum-viz .pv-v{color:#4682dc}
.pdum-viz .pv-ty{color:#50aa78}
.pdum-viz .pv-at{opacity:.72}
.pdum-viz .pv-dim{opacity:.6}
.pdum-viz .pv-tip{position:relative;cursor:help;border-bottom:1px dotted rgba(127,127,127,.7)}
.pdum-viz .pv-tip:hover::after{content:attr(data-tip);position:absolute;left:0;top:1.4em;z-index:99;
  background:#222;color:#eee;padding:.35em .6em;border-radius:6px;font-size:.85em;white-space:pre;
  box-shadow:0 2px 8px rgba(0,0,0,.35)}
.pdum-viz table{border-collapse:collapse;margin:.25em 0}
.pdum-viz td,.pdum-viz th{padding:.15em .7em .15em 0;text-align:left;border:none}
.pdum-viz details{margin:.3em 0}
.pdum-viz summary{cursor:pointer;opacity:.8}
.pdum-viz .pv-chip{display:inline-flex;gap:.4em;align-items:center;border:1px solid rgba(127,127,127,.4);
  border-radius:6px;padding:.15em .5em;margin:.1em}
"""


def esc(x: object) -> str:
    return _html.escape(str(x), quote=True)


class Html:
    """A composable rendering: ``fragment`` is style-free; display adds the style once."""

    def __init__(self, fragment: str):
        self.fragment = fragment

    def _repr_html_(self) -> str:
        return f'<div class="pdum-viz"><style>{_CSS}</style>{self.fragment}</div>'


_RENDERERS: dict[type, object] = {}


def renderer(cls):
    def deco(fn):
        _RENDERERS[cls] = fn
        return fn

    return deco


def render(obj) -> Html:
    for cls in type(obj).__mro__:
        fn = _RENDERERS.get(cls)
        if fn is not None:
            return Html(fn(obj))
    return Html(f"<pre>{esc(repr(obj))}</pre>")


def pill(text, kind="") -> str:
    return f'<span class="pv-pill {kind}">{esc(text)}</span>'


def tip(text, tooltip) -> str:
    return f'<span class="pv-tip" data-tip="{esc(tooltip)}">{text}</span>'


@renderer(Type)
def _type_html(t) -> str:
    return tip(pill(repr(t), "ty"), type(t).__name__)


def _source_details(handle) -> str:
    snap = handle.snapshot
    if snap is None:
        return '<div class="pv-dim">no source (REPL?)</div>'
    body = f"<pre>{esc(snap.text.rstrip())}</pre>"
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import PythonLexer

        body = highlight(snap.text, PythonLexer(), HtmlFormatter(noclasses=True, nobackground=True))
    except ImportError:
        pass
    where = f"{snap.filename.rsplit('/', 1)[-1]}:{snap.firstlineno}"
    return f"<details><summary>source · {esc(where)}</summary>{body}</details>"


@renderer(Handle)
def _handle_html(h) -> str:
    rows = "".join(
        f"<tr><td>{esc(n)}</td><td>{render(t).fragment}</td><td class='pv-dim'>{esc(repr(v)[:60])}</td></tr>"
        for (n, v), t in zip(h.env.items(), h.env_types)
    )
    env = f"<table>{rows}</table>" if rows else '<div class="pv-dim">no captures</div>'
    head = (
        f'<div class="pv-hd">{pill(h.kind)}<span class="pv-name">{esc(h.fntype.template.label)}</span>'
        f"{tip(pill('identity', 'id'), repr(h.fntype))}</div>"
    )
    return f'<div class="pv-card">{head}{env}{_source_details(h)}</div>'


def _walk_count(region, seen):
    for n in region.body:
        _walk_node_count(n, seen)


def _walk_node_count(n, seen):
    if id(n) in seen:
        return
    seen.add(id(n))
    for a in n.args:
        _walk_node_count(a, seen)
    for r in n.regions:
        _walk_count(r, seen)


def _node_tip(node: Node) -> str:
    parts = [repr(node.type)]
    if node.loc is not None:
        parts.append(format_loc(node.loc))
    return " · ".join(parts)


def _region_lines(region: Region, names: dict, counter: list, indent: str, out: list) -> None:
    def ref(n: Node) -> str:
        return tip(f'<span class="pv-v">{names[id(n)]}</span>', _node_tip(n))

    def define(n: Node) -> str:
        if id(n) in names:
            return ref(n)
        for a in n.args:
            define(a)
        argrefs = ", ".join(define(a) for a in n.args)
        if n.op == "core.param":
            names[id(n)] = f"%p{dict(n.attrs)['index']}"
            return ref(n)
        if n.op == "core.yield":
            out.append(f'{indent}<span class="pv-k">core.yield</span> {argrefs}')
            return ""
        regs = ""
        if n.regions:
            blocks = []
            for r in n.regions:
                sub: list = []
                _region_lines(r, names, counter, indent + "  ", sub)
                blocks.append("{\n" + "\n".join(sub) + "\n" + indent + "}")
            regs = " (" + ", ".join(blocks) + ")"
        name = f"%{counter[0]}"
        counter[0] += 1
        names[id(n)] = name
        attrs = ""
        if n.attrs:
            inner = ", ".join(f"{esc(k)} = {esc(repr(v))}" for k, v in n.attrs)
            attrs = f' <span class="pv-at">{{{inner}}}</span>'
        out.append(
            f"{indent}{tip(f'<span class=pv-v>{name}</span>', _node_tip(n))} = "
            f'<span class="pv-k">{esc(n.op)}</span>{" " if argrefs else ""}{argrefs}{attrs}{regs} : '
            f'<span class="pv-ty">{esc(repr(n.type))}</span>'
        )
        return ref(n)

    for n in region.body:
        define(n)


@renderer(Region)
def _region_html(region) -> str:
    seen: set = set()
    _walk_count(region, seen)
    names: dict = {}
    counter = [0]
    for p in region.params:
        names[id(p)] = f"%p{dict(p.attrs)['index']}"
    out: list = []
    _region_lines(region, names, counter, "  ", out)
    params = ", ".join(
        tip(f'<span class="pv-v">%p{dict(p.attrs)["index"]}</span>', _node_tip(p)) for p in region.params
    )
    head = (
        f'<div class="pv-hd">{pill("region")}{pill(f"{len(seen)} nodes", "num")}'
        f"{pill('key ' + region.key.hex()[:12], 'id')}</div>"
    )
    body = f'<div class="pv-ir"><pre>program({params}) {{\n' + "\n".join(out) + "\n}</pre></div>"
    return f'<div class="pv-card">{head}{body}</div>'


@renderer(Stage)
def _stage_html(s) -> str:
    cfg = f"[{', '.join(map(str, s.config))}]" if s.config else ""
    label = esc(s.handle.fntype.template.label + cfg)
    return f'<span class="pv-chip">{pill(s.kind)}{tip(label, repr(s.handle.fntype))}</span>'


@renderer(Terminal)
def _terminal_html(t) -> str:
    return f'<span class="pv-chip">{pill("materializer", "warn")}{esc(t.name)}</span>'


@renderer(Pipeline)
def _pipeline_html(p) -> str:
    chips = ' <span class="pv-dim">|</span> '.join(render(part).fragment for part in p.parts)
    head = (
        f'<div class="pv-hd">{pill("pipeline")}{tip(pill(p.fntype.template.label, "id"), repr(p.fntype))}'
        f"{pill(f'{len(p.parts)} parts', 'num')}</div>"
    )
    return f'<div class="pv-card">{head}<div>{chips}</div></div>'


@renderer(_TierCache)
def _cache_html(c) -> str:
    stats = [
        ("hits", c.hits, "num"),
        ("misses", c.misses, ""),
        ("compiles", c.compiles, "warn"),
        ("evictions", c.evictions, ""),
        ("entries", len(c), ""),
    ]
    if isinstance(c, SpecializationCache):
        stats += [
            ("guard misses", c.guard_misses, ""),
            ("retired", c.retirements, ""),
            ("generation", c.generation, "id"),
        ]
    pills = "".join(pill(f"{k}: {v}", kind) for k, v, kind in stats)
    return f'<div class="pv-card"><div class="pv-hd">{pill(c.name)}{pills}</div></div>'


@renderer(MatchLog)
def _log_html(log) -> str:
    rows = "".join(
        f"<tr><td class='pv-dim'>{esc(stage)}</td><td>{tip(esc(old.op), _node_tip(old))}</td>"
        f"<td>→</td><td>{tip(esc(new.op), _node_tip(new))}</td></tr>"
        for stage, old, new in log.entries
    )
    head = f'<div class="pv-hd">{pill("match log")}{pill(f"{len(log.entries)} firings", "num")}</div>'
    return f'<div class="pv-card">{head}<table>{rows}</table></div>'


_LANE_COLORS = {"host": "rgba(70,130,220,.55)", "gpu": "rgba(80,170,120,.65)"}


@renderer(Timeline)
def _timeline_html(tl) -> str:
    """Horizontal span bars, static like everything here: width = share of the
    total, hover = label + µs. One glance answers 'where did the frame go'."""
    total = tl.total or 1e-9
    bars = []
    for label, start, dur, lane in tl.spans:
        left, width = 100 * start / total, max(0.4, 100 * dur / total)
        color = _LANE_COLORS.get(lane, "rgba(160,110,220,.55)")
        u = f"{dur * 1e6:,.1f} µs"
        bars.append(
            f'<div style="position:relative;height:1.5em;margin:.15em 0">'
            f'<span class="pv-dim" style="position:absolute;left:0;width:9em">{esc(label)}</span>'
            f'<span class="pv-tip" data-tip="{esc(u)}" style="position:absolute;left:calc(9em + {left:.2f}% * 0.72);'
            f'width:calc({width:.2f}% * 0.72);height:1.1em;top:.15em;background:{color};border-radius:3px"></span>'
            f'<span class="pv-dim" style="position:absolute;right:0">{esc(u)}</span></div>'
        )
    total_pill = pill(f"{total * 1e3:.2f} ms total", "num")
    head = f'<div class="pv-hd">{pill("timeline")}{pill(tl.title, "id")}{total_pill}</div>'
    return f'<div class="pv-card">{head}{"".join(bars)}</div>'


def install() -> None:
    """Attach ``_repr_html_`` to the rendered classes (class-level; slots-safe)."""
    for cls in (Type, Handle, Region, Pipeline, Stage, Terminal, _TierCache, MatchLog, Timeline):
        cls._repr_html_ = lambda self: render(self)._repr_html_()


def save(obj, path: str, title: str = "pdum.dsl") -> str:
    """Write a standalone page — the browser/jsdom dev loop, no notebook needed."""
    doc = f"<!doctype html><meta charset='utf-8'><title>{esc(title)}</title>{render(obj)._repr_html_()}"
    with open(path, "w") as f:
        f.write(doc)
    return path
