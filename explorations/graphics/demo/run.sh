#!/bin/sh
# mouse_ripple -- the exact commands.
#
#   ./run.sh                 a window on WebGPU (rendercanvas + glfw)
#   ./run.sh metal           a window on Metal (AppKit + CAMetalLayer)
#   ./run.sh check           headless: both backends, all verifications
#   ./run.sh check metal     headless: Metal only
#   ./run.sh dump            headless + the generated WGSL and MSL
#
# The venv lives in the main checkout; _paths.py puts THIS worktree's
# packages on sys.path, so no install is needed.
set -e
PY=/Users/nehal/src/pdum_dsl/.venv/bin/python
cd "$(dirname "$0")"
case "${1:-window}" in
  window)  exec "$PY" -B mouse_ripple.py ;;
  metal)   exec "$PY" -B mouse_ripple.py --on metal ;;
  webgpu)  exec "$PY" -B mouse_ripple.py --on webgpu ;;
  check)   exec "$PY" -B mouse_ripple.py --offscreen 8 ${2:+--on "$2"} ;;
  dump)    exec "$PY" -B mouse_ripple.py --offscreen 2 --dump ;;
  *)       echo "usage: run.sh [window|metal|webgpu|check [backend]|dump]"; exit 2 ;;
esac
