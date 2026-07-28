#!/bin/sh
# The venv's editable install points at /Users/nehal/src/pdum_dsl/src, which has
# no pdum.tl — the packages live under packages/*/src. PYTHONPATH bridges it.
# Usage:  ./run.sh offscreen   |   ./run.sh window
set -e
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
export PYTHONPATH="$ROOT/packages/dsl/src:$ROOT/packages/tensorlib/src"
exec "$ROOT/../../../.venv/bin/python" "$(dirname "$0")/${1:-offscreen}.py" "${@:2}"
