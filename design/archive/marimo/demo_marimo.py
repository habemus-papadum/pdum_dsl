import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # LabelSliderWidget demo (marimo)

    Two synced integer traits, both clamped to `[0, 100]`:

    - **`label_value`** — written from Python, rendered in JS as a read-only label.
    - **`slider_value`** — bidirectional: dragging the slider sends the value back to Python.

    In marimo, wrap the widget with `mo.ui.anywidget(...)`. Its `.value` is a dict of
    the synced traits, and any cell that reads `.value` re-runs reactively as you drag.
    """)
    return


@app.cell
def _(LabelSliderWidget, mo):
    w = mo.ui.anywidget(LabelSliderWidget(label_value=42, slider_value=10))
    w
    return (w,)


@app.cell
def _(mo, w):
    # Reactive: re-renders live as you drag the slider above.
    mo.md(f"**slider_value = {w.value['slider_value']}**  ·  label_value = {w.value['label_value']}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Drive the widget from Python
    """)
    return


@app.cell
def _(mo):
    # A native marimo slider to push values into the widget's label from Python.
    drive = mo.ui.slider(0, 100, value=75, label="set label_value")
    drive
    return (drive,)


@app.cell
def _(drive, w):
    # Writing the underlying trait pushes the value into the rendered widget.
    w.widget.label_value = drive.value
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
