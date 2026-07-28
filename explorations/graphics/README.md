# explorations/graphics — the campaign's spike ground

Spikes supporting `docs/design/280_graphics-runtime.md` (DRAFT).
Nothing here is product code; each spike exists to settle a design
question with evidence, then feeds its findings back into 280.

| spike | question it settles | status |
|---|---|---|
| `spike_controlflow/` | does select-normal-form cost real performance vs native `if` on device; is emission-time if-reconstruction cheap and correct? | DONE — see `spike_controlflow/FINDINGS.md` (branches: 1.000× parity; data-dependent loops: 2.2–4.0×; recon ~135 LOC, bitwise-correct, doesn't pay yet) |
| `spike_runner/` | what does a warm frame-loop runner (encodable factoring, canvas presentation) actually require; which seams hurt? | DONE — see `spike_runner/FINDINGS.md` (warm frame 0.135 ms vs 5.66 ms today; 5-item hurt-list; encodable sketch) |
| `spike_metal/` | can a second (Metal) backend+runtime implement the same seam; what does the program diff look like? | DONE — see `spike_metal/FINDINGS.md` (program diff = ONE identifier; WGSL↔MSL bitwise-equal ×10 subjects; 5 carve failures + the target numeric contract; `to_numpy` repack is 88–100% of a launch) |
| `demo/` | the polished dual-backend windowed demo: mouse-driven uniforms, native WebGPU (rendercanvas) and Metal (CAMetalLayer) windows, headless verification | DONE — read `demo/mouse_ripple.py`; findings in `demo/FINDINGS.md` (render-stage rows 100% portable under a 4th lexical rule; per-stage binding tables; `encode(pass)` corrected; the float/np.float64 recompile trap) |
