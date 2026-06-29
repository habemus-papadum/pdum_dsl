#!/bin/bash

# test_notebooks.sh - Run all demo notebooks from mkdocs.yml
#
# Usage: ./scripts/test_notebooks.sh [--check-outputs] [--no-inplace]
#
# Options:
#   --check-outputs    Verify that notebook outputs don't change during execution
#   --no-inplace       Execute notebooks without modifying the original files

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

cd "$REPO_ROOT"

NB_FLAGS=""
while [[ $# -gt 0 && "$1" == --* ]]; do
    case "$1" in
        --check-outputs)
            NB_FLAGS="$NB_FLAGS --check-outputs"
            shift
            ;;
        --no-inplace)
            NB_FLAGS="$NB_FLAGS --no-inplace"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

MKDOCS_FILE="$REPO_ROOT/mkdocs.yml"
DOCS_DIR="docs"
NB_SCRIPT="$SCRIPT_DIR/nb.sh"

if [ ! -f "$MKDOCS_FILE" ]; then
    echo "Error: mkdocs.yml not found at $MKDOCS_FILE"
    exit 1
fi

if [ ! -x "$NB_SCRIPT" ]; then
    echo "Error: nb.sh not found or not executable at $NB_SCRIPT"
    exit 1
fi

echo "==================================================================="
echo "Testing Demo Notebooks from mkdocs.yml"
if [ -n "$NB_FLAGS" ]; then
    echo "Flags:$NB_FLAGS"
else
    echo "Mode: Execute and save in-place"
    echo "(Use --no-inplace to prevent file modification in CI)"
fi
echo "==================================================================="
echo ""

NOTEBOOKS=$(grep -E '\.ipynb$' "$MKDOCS_FILE" 2>/dev/null || true)
NOTEBOOKS=$(echo "$NOTEBOOKS" | sed 's/.*: //' | tr -d ' ')

if [ -z "$NOTEBOOKS" ]; then
    echo "No notebooks found in mkdocs.yml"
    exit 0
fi

TOTAL=0
PASSED=0
FAILED=0
FAILED_NOTEBOOKS=""

while IFS= read -r notebook_path; do
    if [ -z "$notebook_path" ]; then
        continue
    fi

    TOTAL=$((TOTAL + 1))
    FULL_PATH="$DOCS_DIR/$notebook_path"

    echo "-------------------------------------------------------------------"
    echo "[$TOTAL] Running: $FULL_PATH"
    echo "-------------------------------------------------------------------"

    if "$NB_SCRIPT" $NB_FLAGS "$FULL_PATH"; then
        echo "✓ PASSED: $notebook_path"
        PASSED=$((PASSED + 1))
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 2 ]; then
            echo "⚠ OUTPUTS CHANGED: $notebook_path"
        else
            echo "✗ FAILED: $notebook_path"
        fi
        FAILED=$((FAILED + 1))
        FAILED_NOTEBOOKS="$FAILED_NOTEBOOKS  - $notebook_path (exit code: $EXIT_CODE)\n"
    fi

    echo ""
done <<< "$NOTEBOOKS"

echo "==================================================================="
echo "Test Summary"
echo "==================================================================="
echo "Total notebooks: $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "Failed notebooks:"
    echo -e "$FAILED_NOTEBOOKS"
    exit 1
else
    echo "✓ All notebooks passed!"
    exit 0
fi
