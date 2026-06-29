import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # The update asymmetry: front-end change vs. Python-side write

    A widget trait can change two ways, and marimo treats them **differently**:

    | trigger | browser DOM updates? | marimo cells re-run? |
    |---------|:---:|:---:|
    | **drag the widget's slider** (front-end) | ✅ | ✅ |
    | **`w.widget.label_value = v`** (Python) | ✅ | ❌ |

    **Why** (marimo 0.23.11 source):

    - A front-end interaction sends an `UpdateUIElementCommand` to the kernel, which
      calls `Kernel.set_ui_element_value` (`_runtime/runtime.py:1794`). Its docstring:
      *"Runs cells that reference the UI element by name."* It collects
      `get_referring_cells(name) - get_defining_cells(name)` (`:1944`) and re-runs
      them (`:1972`). **This is the only path that schedules reactive re-runs for a
      widget.**
    - A Python-side trait write goes through ipywidgets' comm instead
      (`notify_change → send_state(key) → MarimoComm.send`): it ships the delta to the
      browser and updates the DOM, but **never calls `set_ui_element_value`**. There is
      no traitlets-observer bridge — `mo.ui.anywidget` registers `on_change=None`
      (`_impl/from_anywidget.py:213`).

    So: Python→browser is efficient and works, but it is a **one-way push** that does
    **not** re-trigger marimo's dataflow graph.

    ---

    ### Try it

    1. **Move the ② Python-write slider.** Watch the widget's own `label_value` label
       (in ①) update live — but the ③ reader below stays **frozen** and its
       *reader re-runs* counter does **not** move.
    2. **Now nudge the widget's slider in ①.** The ③ reader re-runs (counter climbs)
       and only *then* does its echoed `label_value` catch up to what Python wrote —
       because `w.value` is computed live, it just wasn't *observed* until a front-end
       event forced the re-run.

    (Ignore the initial `1`s; watch the **deltas** as you interact.)
    """)
    return


@app.cell
def _(LabelSliderWidget, mo):
    # ① The widget itself. Its label shows label_value live (updated via the comm).
    w = mo.ui.anywidget(LabelSliderWidget(label_value=25, slider_value=10))
    w
    return (w,)


@app.cell
def _(mo):
    # Counters: how often Python wrote a trait vs. how often the reader cell re-ran.
    get_runs, set_runs = mo.state({"python_writes": 0, "reader_reruns": 0})
    return get_runs, set_runs


@app.cell
def _(mo):
    # ② A native marimo control used to push a value into the widget FROM PYTHON.
    pydriver = mo.ui.slider(0, 100, value=25, label="② Python write → w.widget.label_value")
    pydriver
    return (pydriver,)


@app.cell
def _(mo, pydriver, set_runs, w):
    # WRITER — runs when ② moves (it depends on `pydriver`). It mutates the trait
    # with a plain Python assignment: this pushes to the browser via the comm, but
    # does NOT go through set_ui_element_value, so it does not re-run w's readers.
    w.widget.label_value = pydriver.value
    set_runs(lambda d: {**d, "python_writes": d["python_writes"] + 1})
    mo.md(f"✍️ Python just wrote **label_value = {pydriver.value}** (via `w.widget.label_value`).")
    return


@app.cell
def _(mo, set_runs, w):
    # ③ READER — references `w`, so it re-runs ONLY when a front-end interaction
    # routes through set_ui_element_value. It captures the values into locals; since
    # the cell doesn't re-run on Python-side writes, this display goes stale.
    seen_label = w.value["label_value"]
    seen_slider = w.value["slider_value"]
    set_runs(lambda d: {**d, "reader_reruns": d["reader_reruns"] + 1})
    mo.md(
        f"""
        ③ **Reader cell** (reads `w.value`)

        - last-seen `label_value` = **{seen_label}**
        - last-seen `slider_value` = **{seen_slider}**

        Compare *last-seen `label_value`* against the live label in ①: when they
        disagree, you're looking at a Python-side write the reader never reacted to.
        """
    )
    return


@app.cell(hide_code=True)
def _(get_runs, mo):
    # Scoreboard — reads the counters, so it refreshes when either changes.
    c = get_runs()
    gap = c["python_writes"] - c["reader_reruns"]
    mo.md(
        f"""
        ### Scoreboard

        | counter | value |
        |---------|------:|
        | Python writes (✍️ writer cell ran) | **{c['python_writes']}** |
        | Reader re-runs (③ cell ran) | **{c['reader_reruns']}** |

        Moving **only** the ② Python slider bumps *Python writes* but leaves *Reader
        re-runs* flat — every such gap is a DOM update that marimo's graph did **not**
        observe. (Front-end drags bump both, since the writer also references `w`.)
        """
    )
    return


@app.cell
def _():
    from widget import LabelSliderWidget

    return (LabelSliderWidget,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
