"""A simple anywidget with two 0–100 integer properties.

- ``label_value``  : written from Python, rendered in JS as a read-only label.
- ``slider_value`` : read/write from both sides. Dragging the slider in JS
                     calls ``model.save_changes()``, which pushes the new value
                     back over the comm so the Python trait updates (and any
                     ``.observe()`` callback fires).
"""

import anywidget
import traitlets


class LabelSliderWidget(anywidget.AnyWidget):
    # --- shared state ----------------------------------------------------
    label_value = traitlets.Int(0).tag(sync=True)
    slider_value = traitlets.Int(0).tag(sync=True)

    # --- keep both traits clamped to [0, 100] on the Python side ---------
    @traitlets.validate("label_value", "slider_value")
    def _clamp(self, proposal):
        return max(0, min(100, proposal["value"]))

    # --- front-end (inlined ESM) -----------------------------------------
    _esm = """
    function render({ model, el }) {
      // -- label: one-way Python -> JS -------------------------------
      const label = document.createElement("div");
      label.className = "lsw-label";
      const paintLabel = () => {
        label.textContent = `label_value: ${model.get("label_value")}`;
      };
      paintLabel();
      model.on("change:label_value", paintLabel);

      // -- slider: two-way, sends back to Python on input ------------
      const row = document.createElement("div");
      row.className = "lsw-row";

      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = "0";
      slider.max = "100";
      slider.value = model.get("slider_value");

      const readout = document.createElement("span");
      readout.className = "lsw-readout";
      readout.textContent = model.get("slider_value");

      slider.addEventListener("input", () => {
        const v = Number(slider.value);
        readout.textContent = v;
        model.set("slider_value", v);
        model.save_changes();          // <-- message back to Python
      });

      // reflect Python-side writes back into the slider
      model.on("change:slider_value", () => {
        const v = model.get("slider_value");
        slider.value = v;
        readout.textContent = v;
      });

      row.appendChild(slider);
      row.appendChild(readout);
      el.appendChild(label);
      el.appendChild(row);
    }
    export default { render };
    """

    _css = """
    .lsw-label   { font: 600 14px sans-serif; margin-bottom: 6px; }
    .lsw-row     { display: flex; align-items: center; gap: 8px; }
    .lsw-readout { font: 13px monospace; min-width: 2.5em; text-align: right; }
    """


if __name__ == "__main__":
    # Quick smoke test outside a notebook: prove the trait round-trips.
    w = LabelSliderWidget(label_value=42, slider_value=10)
    w.observe(lambda ch: print(f"slider_value -> {ch['new']}"), names="slider_value")
    print(f"label_value  = {w.label_value}")
    print(f"slider_value = {w.slider_value}")
    w.slider_value = 250  # clamped to 100 by the validator
    print(f"slider_value (after clamp) = {w.slider_value}")
