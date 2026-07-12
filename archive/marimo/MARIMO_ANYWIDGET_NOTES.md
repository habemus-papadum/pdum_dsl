# How marimo works with anywidget — notes

Source-grounded findings on how [marimo](https://marimo.io) integrates
[anywidget](https://anywidget.dev), focused on two questions:

1. **Reactivity granularity** — when a widget trait changes, which cells re-run?
2. **Update efficiency** — when Python changes a trait, does the JS widget process
   just that change, or re-initialize?

Verified against the source of the versions installed in this repo:

| package | version |
|---------|---------|
| marimo | 0.23.11 |
| anywidget | 0.11.0 |
| ipywidgets | 8.1.8 |
| traitlets | 5.15.1 |

> **Validation note.** These findings come from reading the installed source plus a
> Python-side smoke test (`widget.py`). The live front-end behavior (DOM updates,
> cell re-runs while dragging a slider) is reasoned from the source, **not** observed
> in a running browser/kernel. The probe notebooks below are the interactive checks.

---

## TL;DR

- `mo.ui.anywidget(widget)` wraps an `AnyWidget` as a single marimo **`UIElement`**.
  Its `.value` is the **entire** trait dict — there is no per-trait surface.
- marimo reactivity is **per-variable, not per-key**. A cell that reads
  `w.value['label_value']` depends on the *name* `w`; **any** trait change re-runs it.
  marimo's graph is built by static analysis and cannot see which dict key you read.
- The Python→JS update path **is** incremental and efficient. `render()` runs **once**;
  a Python-side trait change sends **only that trait** over the comm and fires **only**
  the matching `change:<key>` event. State built in `render()` is never reinitialized.
- The efficiency boundary is the **trait**: model independently-changing state as
  separate traits. A "small change" inside one big blob trait ships the whole trait.

---

## 1. The wrapper: `mo.ui.anywidget` is a `UIElement`

`marimo/_plugins/ui/_impl/from_anywidget.py`

- `anywidget(UIElement[ModelIdRef, AnyWidgetState])` (`:157`) wraps the widget and
  proxies attribute/item access to it (`__getattr__`/`__getitem__`, `:302`/`:310`).
- `.value` (`:276`) returns `get_anywidget_state(self.widget)` — i.e. **all** synced
  traits as one dict, minus system traits like `_esm`, `_css`, `comm`, `layout`
  (`get_anywidget_state`, `:114`–`:145`).
- Frontend → Python: `_convert_value` (`:258`) receives a state update from the
  browser and applies it wholesale via `self.widget.set_state(value)`.
- Writes are forwarded: setting `w.<trait>` forwards to the underlying widget
  (`__setattr__`, `:292`), so `w.widget.label_value = 75` and `w.label_value = 75`
  both reach the trait.

**Consequence:** the unit marimo tracks is the whole widget value object, not its keys.

---

## 2. Reactivity granularity — per-variable, NOT per-key

**Question:** does a cell reading only `w.value['label_value']` re-run when only
`slider_value` changes?

**Answer: yes.** marimo's dataflow graph is built by **static analysis of variable
names**. A cell mentioning `w` depends on `w` as a whole; the `['label_value']`
subscript is a runtime value the compiler can't see. Because the widget is one
`UIElement` whose `.value` is the whole dict (§1), any trait change re-runs **every**
cell that references `w`.

There is no mechanism for finer granularity: that would require dynamically tracing
which keys each cell reads, which marimo's static model does not do.

**Probe:** `demo_marimo_reactivity.py` — two reader cells each touch one key and bump a
counter held in `mo.state`. Dragging the slider makes **both** counters climb, showing
whole-widget (not per-key) reactivity.

Patterns that matter in that probe:
- The counter lives in `mo.state` so it survives re-runs (a plain local resets each tick).
- Counters use the **functional-updater** form `set_runs(lambda d: {...})`. Calling the
  *setter* creates no dependency on the state (only calling the *getter* does), so the
  readers don't re-trigger themselves — only the scoreboard cell (which reads the
  getter) refreshes. No reactive loop.

**Practical implication:** if you want a cell to react to only one trait, don't rely on
marimo — split the relevant value into its own variable in a dedicated cell, or
structure the widget so unrelated state lives in separate widgets.

---

## 3. Python → JS updates are incremental (not a re-init)

**Question:** when a small piece of state changes from Python, does the JS widget
process just that change, or rebuild?

**Answer: just that change.** The full chain for `w.widget.label_value = 75`:

1. **`notify_change`** (ipywidgets `widget.py`) fires for the changed trait, reads
   `name = 'label_value'`, and calls `self.send_state(key=name)` — **only that key**.
2. **`send_state(key='label_value')`** builds `get_state(key='label_value')` — a state
   dict with **just that trait** — and wraps it as
   `{'method': 'update', 'state': {'label_value': 75}, ...}`.
3. **`MarimoComm.send`** (`comm.py:194`) → `_broadcast` →
   `ModelUpdate(state={'label_value': 75})` — a delta `update`, distinct from the
   one-time full-state `open` message (`comm.py:105` vs `:111`).
4. The front-end applies the delta and fires **`change:label_value` only**.
5. The handler installed in `render()` runs and does a surgical DOM update
   (e.g. `label.textContent = …`). Unrelated DOM (the slider) is untouched.

Key architectural facts:

- **`render({ model, el })` runs exactly once** per view mount. DOM nodes, child
  widgets, and closures set up there **persist** across value changes — never reinitialized.
- **Only the changed trait travels the wire.** `open` carries full state once; every
  later change is a single-key `update`.
- **Only the matching `change:<key>` handler fires.** Changing `label_value` does not
  run the slider's handler, and vice-versa.
- Echo suppression avoids redundant round-trips: a value the front-end just sent isn't
  bounced back (`echo_update` is skipped, `comm.py:122`–`:124`; `_property_lock` in
  ipywidgets `send_state`).

This is why the idiom is **"build once in `render()`, wire `model.on('change:foo', …)`
for surgical updates"** — exactly what `widget.py` does.

---

## 4. The efficiency boundary is the trait

Granularity is **per-trait**, not per-sub-field — traitlets only knows "this trait
changed," not what changed inside it.

- **Separate traits** (e.g. `label_value`, `slider_value`): each change ships only that
  trait and fires only its `change:` event. Optimal.
- **One big blob trait** (a large dict/list/array): mutating one element makes traitlets
  treat the whole trait as changed, ships the **entire** value over the comm, and your
  single `change:bigthing` handler must diff what moved.

**Guidance:** model independently-changing pieces of state as independent (or
reasonably-sized) traits. Then both layers stay efficient:
- marimo re-runs only cells that reference the widget (§2), and
- anywidget ships and re-renders only what changed (§3).

---

## 5. Two layers, kept distinct

It helps to separate the two things that happen on a change:

| | trigger | granularity | cost |
|---|---|---|---|
| **marimo cell re-run** | front-end interaction changes `w` | per-**variable** (whole widget) | re-runs every cell referencing `w` |
| **anywidget DOM update** | any trait change (either direction) | per-**trait** `change:<key>` | runs one handler; `render()` not re-run |

A Python-side trait write efficiently updates the *browser DOM* (§3). It does **not**
re-run marimo cells that read `w` — marimo's reactivity is driven by its own value-update
protocol from the front-end, not by traitlets observers. This is a genuine **asymmetry**:

- **Front-end change** (drag the slider): the browser sends an `UpdateUIElementCommand`,
  which calls `Kernel.set_ui_element_value` (`_runtime/runtime.py:1794`) — docstring:
  *"Runs cells that reference the UI element by name."* It collects
  `get_referring_cells(name) - get_defining_cells(name)` (`:1944`) and re-runs them
  (`:1972`). **This is the only path that schedules reactive re-runs for a widget.**
- **Python-side write** (`w.widget.x = v`): goes through ipywidgets' comm (§3) — updates
  the DOM but **never calls `set_ui_element_value`**. There's no traitlets-observer
  bridge: `mo.ui.anywidget` registers `on_change=None` (`from_anywidget.py:213`). So
  dependent cells do **not** re-run.

Because `w.value` is computed live (`get_anywidget_state`, §1), a Python-side write isn't
*lost* — a later front-end-triggered re-run will see the new value; it just isn't
*observed reactively* at write time.

The cell that **creates** the widget does not re-run on a value change either — line
`:1944` explicitly subtracts the defining cells (that would build a new model and lose
state); only downstream cells that reference it re-run.

**Probe:** `demo_marimo_asymmetry.py` quantifies this — a "Python writes" counter climbs
when you move a native slider that writes `w.widget.label_value`, while a "reader re-runs"
counter stays flat; dragging the *widget's* slider bumps both.

---

## Files in this repo

- `widget.py` — `LabelSliderWidget`: a minimal anywidget with two `Int` traits
  (`label_value` shown as a label, `slider_value` as a two-way slider). Demonstrates the
  build-once-in-`render`, update-via-`change:` pattern.
- `demo.ipynb` — Jupyter demo (uses an `IPython.display` handle updated in place).
- `demo_marimo.py` — marimo demo.
- `demo_marimo_reactivity.py` — the per-key reactivity probe (§2).
- `demo_marimo_asymmetry.py` — the front-end-change vs. Python-write asymmetry probe (§5).

## Source references

- `marimo/_plugins/ui/_impl/from_anywidget.py` — the `mo.ui.anywidget` wrapper.
- `marimo/_plugins/ui/_impl/comm.py` — `MarimoComm`; `open` vs `update` messages.
- `marimo/_runtime/runtime.py` — `Kernel.set_ui_element_value` (`:1794`): the only path
  that re-runs cells on a widget change; reached from the front-end command handler.
- `ipywidgets/widgets/widget.py` — `notify_change` → `send_state(key=name)` (delta send).
