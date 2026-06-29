#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

cd "$REPO_ROOT"

echo "================================"
echo "Pre-Release Checks"
echo "================================"



echo "5. Check Git Status (must be clean)..."
if [[ -n $(git status --porcelain) ]]; then
  echo "❌ Error: Git working directory is not clean. Please commit or stash changes first."
  exit 1
fi
echo "✓ Git working directory is clean"


echo "6. Python Lint..."
uv run ruff check .


echo "7. Python Tests..."
uv run pytest

echo ""
echo "8. Documentation Build..."
uv run mkdocs build


echo "================================"
echo "✅ All pre-release checks passed!"
echo "================================"
