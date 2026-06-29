# Driving an anywidget from Python vs. from the UI (marimo)

When you wrap an anywidget with `mo.ui.anywidget`, a trait can change in **two**
directions — and marimo reacts to them differently. This one asymmetry trips people up,
so this doc covers it twice: first conceptually (what you can and can't do), then a
technical deep dive (exactly why).

> One-sentence version: **the user interacting with the widget drives your notebook;
> your Python code writing to the widget drives the screen — but not your notebook.**

---

## Part 1 — Conceptually (what you can and can't do)

A wrapped widget value can change two ways:

- **The user interacts** with it in the browser (drags the slider, clicks, etc.).
- **Your Python code writes** to it, e.g. `w.widget.label_value = 42`.

These feel symmetric. They are not.

### ✅ What works

- **Read a widget reactively when the *user* drives it.** If the user drags the slider,
  every cell that reads `w.value` re-runs automatically. This is the normal, happy path:
  treat the widget as an **input**, and let other cells compute from it.
- **Write a trait from Python to update what's on screen.** Setting
  `w.widget.label_value = 42` *does* update the widget in the browser — the new value
  shows up immediately. Great for using the widget as a live **display/output** you push
  to from Python.

### ❌ What doesn't work (the gotcha)

- **A Python-side write does *not* re-run your other cells.** If cell A does
  `w.widget.label_value = 42`, other cells that read `w.value['label_value']` will **not**
  re-run. The new value is on the screen, but it never flows through your notebook's
  reactive graph.
- **So a cell that read the widget earlier can show a *stale* value.** It keeps displaying
  whatever it last saw, even though the widget itself (and `w.value`, if you asked again)
  shows the new number. It only "catches up" the next time it re-runs for some *other*
  reason — e.g. the user then touches the widget.
- **You can't use a Python-side widget write as a trigger** for downstream reactive work.

In short: **user → widget → notebook** is reactive. **Python → widget** updates the screen
but is a dead end as far as reactivity goes.

### What to do instead

- **Pick a single source of truth, and keep it in marimo — not inside the widget.**
  If a value needs to drive other cells, hold it in a native marimo control
  (`mo.ui.slider`, `mo.ui.number`, …) or in `mo.state`. Compute everything (including
  what you push into the widget) *from* that. Then everything stays reactive.
- **Use the widget for the direction that works.** Let *user interaction* be the input
  marimo reacts to; treat *Python writes into the widget* as display-only.
- **Don't expect a feedback loop.** Writing to the widget from Python to "kick off" a
  recompute won't work — wire the recompute to the real source of truth instead.

### Rule of thumb

| You want to… | Do this |
|---|---|
| React when the **user** changes the widget | Just read `w.value` in another cell — it works |
| **Show** a computed value in the widget | Write `w.widget.<trait> = …` from Python — it works |
| Have a Python-set value **drive other cells** | ❌ Don't route it through the widget. Keep the value in `mo.ui.*` / `mo.state` and derive from there |

---

## Part 2 — Technical deep dive (exactly what's happening)

*(Verified against marimo 0.23.x / anywidget 0.11 / ipywidgets 8.1. File\:line references
point at the installed source.)*

A trait change reaches the browser the same way in both directions, but only **one**
direction enters marimo's reactive scheduler. There are two distinct channels:

### Channel A — marimo's UI-value protocol (reactive)

When the **user** interacts with the widget, the marimo front-end sends an
`UpdateUIElementCommand` to the kernel. That lands in
`Kernel.set_ui_element_value` (`marimo/_runtime/runtime.py:1794`), whose docstring is
literally *"Runs cells that reference the UI element by name."* It:

1. applies the new value to the `UIElement` (calling the anywidget wrapper's
   `_convert_value` → `widget.set_state`), then
2. collects the cells to re-run as
   `get_referring_cells(name) - get_defining_cells(name)` (`runtime.py:1944`) — i.e.
   every cell that *reads* the variable `w`, **excluding** the cell that *created* it
   (so the widget isn't rebuilt and state isn't lost), and
3. re-runs them (`runtime.py:1972`).

**This is the only code path that schedules reactive re-runs for a widget.** It is reached
only from a front-end interaction.

### Channel B — the ipywidgets comm (display only)

When **Python** writes a trait (`w.widget.label_value = 42`), traitlets fires
`Widget.notify_change`, which calls `send_state(key='label_value')` — packaging **only the
changed trait** as an `update` message — and `MarimoComm.send`
(`marimo/_plugins/ui/_impl/comm.py:194`) ships it to the browser. The DOM updates via the
widget's own `model.on("change:label_value", …)` handler.

This channel **never calls `set_ui_element_value`.** And nothing bridges it back into
Channel A: `mo.ui.anywidget` is created with `on_change=None`
(`marimo/_plugins/ui/_impl/from_anywidget.py:213`) and registers **no traitlets observer**
that would forward Python-side changes into marimo's graph. So no cells re-run.

### Why the stale read "catches up" later

The wrapper's `.value` is computed **live** — `from_anywidget.py:276` returns
`get_anywidget_state(self.widget)`, reading the widget's current traits on every access.
So a Python-side write is never *lost*: the value is right there in `w.value`. It simply
isn't *observed* — no cell was scheduled to read it. The next time a cell re-runs for any
reason (e.g. the user then drags the slider, going through Channel A), it reads the live
value and "catches up."

### Why marimo is built this way

marimo's reactivity is a **static dataflow graph**: edges come from analyzing which
variable *names* each cell references, and re-runs are triggered by well-defined events
(a cell redefining a variable, or a UI element value arriving from the front-end).
A plain Python attribute mutation like `w.widget.label_value = 42` is invisible to that
model — marimo would have to observe arbitrary object mutation to catch it, which it
deliberately does not do. The same reason explains the related fact that the widget is one
`UIElement` whose `.value` is the *whole* trait dict, so marimo reactivity is per-variable,
not per-trait.

### See it run

`demo_marimo_asymmetry.py` makes this measurable: a "Python writes" counter climbs each
time a native slider writes `w.widget.label_value` (Channel B), while a "reader re-runs"
counter stays flat — until you drag the widget's own slider (Channel A), which bumps both.

> Related notes: `MARIMO_ANYWIDGET_NOTES.md` (§2 per-key reactivity, §3 update
> efficiency, §5 this asymmetry).
