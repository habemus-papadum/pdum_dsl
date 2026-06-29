#!/bin/bash

# Notebook runner script
# This script runs Jupyter notebooks for testing
#
# Usage: ./scripts/nb.sh [--check-outputs] [--no-inplace] <notebook_path> [jupyter options]
#
# Options:
#   --check-outputs    Compare outputs before and after execution to detect changes
#   --no-inplace       Execute notebook without modifying the original file
#
# Examples:
#   ./scripts/nb.sh docs/demos/example.ipynb                    # Run notebook with default options
#   ./scripts/nb.sh --check-outputs docs/demos/example.ipynb    # Run and check if outputs changed
#   ./scripts/nb.sh --no-inplace docs/demos/example.ipynb       # Run without modifying the file
#   ./scripts/nb.sh docs/demos/example.ipynb --ExecutePreprocessor.timeout=300
#
# By default, the notebook will be executed in place, with outputs saved back to the notebook file.
# Use --no-inplace to prevent modifications (useful in CI environments).
# Execution stops on the first error encountered.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

cd "$REPO_ROOT"

# Parse flags
CHECK_OUTPUTS=false
NO_INPLACE=false
while [[ $# -gt 0 && "$1" == --* ]]; do
    case "$1" in
        --check-outputs)
            CHECK_OUTPUTS=true
            shift
            ;;
        --no-inplace)
            NO_INPLACE=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Check if notebook path is provided
if [ $# -lt 1 ]; then
    echo "Error: No notebook path provided"
    echo ""
    echo "Usage: ./scripts/nb.sh [--check-outputs] [--no-inplace] <notebook_path> [jupyter options]"
    exit 1
fi

NOTEBOOK_PATH="$1"
shift  # Remove first argument, keep any additional options

# Verify notebook exists
if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "Error: Notebook not found: $NOTEBOOK_PATH"
    exit 1
fi

# Function to extract outputs from notebook (ignoring metadata)
extract_outputs() {
    local notebook="$1"
    python3 - "$notebook" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

outputs = []
for cell in nb.get('cells', []):
    tags = cell.get('metadata', {}).get('tags', [])
    if 'nondet' in tags:
        outputs.append([])
        continue

    cell_outputs = []
    for output in cell.get('outputs', []):
        cleaned = {'output_type': output.get('output_type')}
        if 'text' in output:
            cleaned['text'] = output['text']
        if 'data' in output:
            cleaned['data'] = output['data']
        if 'name' in output:
            cleaned['name'] = output['name']
        if 'ename' in output:
            cleaned['ename'] = output['ename']
        if 'evalue' in output:
            cleaned['evalue'] = output['evalue']
        if 'traceback' in output:
            cleaned['traceback'] = output['traceback']
        cell_outputs.append(cleaned)
    outputs.append(cell_outputs)

json.dump(outputs, sys.stdout, indent=2, sort_keys=True)
PY
}

# Save outputs before execution if checking
TEMP_BEFORE=""
if [ "$CHECK_OUTPUTS" = true ]; then
    TEMP_BEFORE=$(mktemp)
    echo "Extracting outputs before execution..."
    extract_outputs "$NOTEBOOK_PATH" > "$TEMP_BEFORE"
fi

# Build jupyter command
TEMP_OUTPUT=""
TEMP_DIR=""
if [ "$NO_INPLACE" = true ]; then
    TEMP_DIR=$(mktemp -d)
    TEMP_OUTPUT="$TEMP_DIR/output.ipynb"
    JUPYTER_CMD="uv run jupyter nbconvert --execute --to notebook --output-dir=$TEMP_DIR --output=output"
else
    JUPYTER_CMD="uv run jupyter nbconvert --execute --to notebook --inplace"
fi

JUPYTER_CMD="$JUPYTER_CMD --ExecutePreprocessor.timeout=60"
JUPYTER_CMD="$JUPYTER_CMD $NOTEBOOK_PATH"

if [ $# -gt 0 ]; then
    JUPYTER_CMD="$JUPYTER_CMD $*"
fi

echo "Running notebook: $NOTEBOOK_PATH"
if [ "$NO_INPLACE" = true ]; then
    echo "Mode: Execute only (no file modification)"
else
    echo "Mode: Execute and save outputs in-place"
fi

export NOTEBOOK_CI=1

set +e
eval "$JUPYTER_CMD"
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -ne 0 ]; then
    echo "Notebook execution failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "✓ Notebook executed successfully!"
if [ "$NO_INPLACE" = true ]; then
    echo "  Original file unchanged: $NOTEBOOK_PATH"
else
    echo "  Outputs saved to: $NOTEBOOK_PATH"
fi

if [ "$CHECK_OUTPUTS" = true ]; then
    echo "Comparing outputs..."
    TEMP_AFTER=$(mktemp)
    if [ "$NO_INPLACE" = true ]; then
        extract_outputs "$TEMP_OUTPUT" > "$TEMP_AFTER"
    else
        extract_outputs "$NOTEBOOK_PATH" > "$TEMP_AFTER"
    fi

    if diff -q "$TEMP_BEFORE" "$TEMP_AFTER" > /dev/null 2>&1; then
        echo "✓ Outputs unchanged - notebook is stable"
        rm -f "$TEMP_BEFORE" "$TEMP_AFTER"
    else
        echo "⚠ Outputs changed during execution"
        diff -u "$TEMP_BEFORE" "$TEMP_AFTER" || true
        rm -f "$TEMP_BEFORE" "$TEMP_AFTER"
        exit 2
    fi
fi
