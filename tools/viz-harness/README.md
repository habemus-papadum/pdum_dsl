# viz-harness

The notebook-free dev loop for the viz satellite:

    npm install        # once (jsdom)
    npm test           # regenerates fixtures via uv, asserts DOM structure

`fixtures/*.html` are standalone pages — open them in a browser to eyeball
styling. Not wired into CI (node optional); the Python-side structure tests
in `tests/test_viz.py` are the CI floor.
