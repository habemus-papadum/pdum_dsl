import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Is marimo per-key reactive? (a probe)

    **Question:** a cell that reads only `w.value['label_value']` — does it re-run
    when only `slider_value` changes?

    **What marimo's source says** (`marimo/_plugins/ui/_impl/from_anywidget.py`):
    `mo.ui.anywidget` is a single `UIElement` whose `.value` returns the *entire*
    trait dict (`get_anywidget_state` serializes every synced trait). marimo's
    dataflow graph is built by **static analysis of variable names** — a cell that
    mentions `w` depends on `w` as a whole; it cannot see which dict key you read.

    **Prediction:** *any* trait change re-runs *every* cell that touches `w`.

    The two reader cells below each touch only one key, and bump a counter every
    time they run. Drag the slider and watch the scoreboard: if **both** counts
    climb, marimo is reacting at whole-widget granularity — not per key.
    """)
    return


@app.cell
def _(LabelSliderWidget, mo):
    w = mo.ui.anywidget(LabelSliderWidget(label_value=42, slider_value=10))
    w
    return (w,)


@app.cell
def _(mo):
    # Shared run-counter held in marimo state so it survives cell re-runs.
    get_runs, set_runs = mo.state({"reader_A_label": 0, "reader_B_slider": 0})
    return get_runs, set_runs


@app.cell
def _(mo, set_runs, w):
    # READER A — reads ONLY label_value.
    a_value = w.value["label_value"]
    # Bump this reader's counter. The functional-updater form composes within a
    # single tick and does NOT read get_runs, so it creates no reactive loop.
    set_runs(lambda d: {**d, "reader_A_label": d["reader_A_label"] + 1})
    mo.md(f"**Reader A** read `label_value` = **{a_value}**")
    return


@app.cell
def _(mo, set_runs, w):
    # READER B — reads ONLY slider_value.
    b_value = w.value["slider_value"]
    set_runs(lambda d: {**d, "reader_B_slider": d["reader_B_slider"] + 1})
    mo.md(f"**Reader B** read `slider_value` = **{b_value}**")
    return


@app.cell(hide_code=True)
def _(get_runs, mo):
    # Scoreboard — reads the counters, so it refreshes whenever they change.
    counts = get_runs()
    mo.md(
        f"""
        | cell | reads | times run |
        |------|-------|-----------|
        | Reader A | `label_value` only | **{counts['reader_A_label']}** |
        | Reader B | `slider_value` only | **{counts['reader_B_slider']}** |

        Both numbers climbing together as you drag the slider ⇒ whole-widget
        reactivity, **not** per-key.
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
