# 060 — Rendering notes (rich static widgets for notebooks)

**Status:** v1 shipped 2026-07-12 (`src/pdum/dsl/viz.py`, satellite — zero
kernel edits). Serves the whole book, both notebook hosts.

## The contract

- **Static like a PNG.** A rendering is HTML handed to the host via
  `_repr_html_` (honored by Jupyter *and* marimo). No kernel↔JS channel,
  ever — later widgets may run JavaScript, but only self-contained display
  logic.
- **Interactivity, v1 = CSS only.** Collapsing via `<details>`, informative
  hovers via `:hover::after` attr-tooltips. Zero `<script>` tags — survives
  every notebook sandbox, HTML sanitizer, and the mkdocs site. (Asserted in
  tests: fixtures contain no script elements.)
- **Modern JS is sanctioned when needed** (CodeMirror-grade widgets):
  ESM modules imported from CDNs, unbundled, written against the same
  fragment contract. Not needed for v1 because we own the IR grammar and
  color it server-side.
- **Isolation** via scoped-class CSS (`.pdum-viz` prefix on every rule);
  shadow DOM is the escalation if a host ever bleeds styles in.

## Composability

`render(obj) -> Html`, where `Html.fragment` is **style-free inner HTML**
and `_repr_html_()` wraps a fragment with the single `<style>` at the
display root. Containers compose by embedding children's *fragments*
(`Pipeline` embeds `Stage` chips; `Handle` embeds `Type` pills), so nesting
never duplicates styles — asserted (`page.count("<style>") == 1`).
Renderers are a registry (`@viz.renderer(cls)`) — dialects/backends add
widgets for their own objects the same way they add everything else.

## Hooking objects without kernel edits

`viz.install()` assigns `_repr_html_` as a *class attribute* on
Handle/Region/Pipeline/caches/… (slots-safe: instances aren't touched).
Notebooks opt in with two lines; the kernel never imports viz.

## The dev loop (no notebook required)

- `viz.save(obj, path)` → standalone page for browser eyeballing.
- `tools/viz-harness/`: an npm package (jsdom + `node --test`) whose
  fixtures are generated through `uv run`; asserts DOM structure — pills
  present, tooltips carry `type · provenance`, one style block, zero
  scripts. Local-only (node optional); the CI floor is the Python-side
  structure tests in `tests/test_viz.py`.

## Current widgets

Type (pill + class tooltip) · Handle (card: kind/identity pills, env table,
collapsible highlighted source) · Region (colored IR listing; every SSA ref
hovers to `type · loc/provenance`; node-count + content-key pills) ·
Stage/Terminal/Pipeline (chips composed with `|`) · caches (counter pills) ·
MatchLog (before→after table). Fallback: `<pre>` of `repr`.

## Later

WGSL/rendered-source widgets with line↔provenance hovers (the 050 side
table, visualized); a FastRecord card; CodeMirror-backed editors if a
chapter ever wants editable source; PNG/SVG export for docs.
