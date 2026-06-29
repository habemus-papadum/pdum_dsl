#!/usr/bin/env bash

# Setup script for dsl development environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

cd "$REPO_ROOT"

echo "============================================="
echo "Setting up dsl"
echo "============================================="
echo ""

check_command() {
    local cmd=$1
    local install_url=$2
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "❌ Error: $cmd is not installed"
        echo "   Install it from: $install_url"
        exit 1
    fi
    echo "   ✓ $cmd found: $($cmd --version | head -n 1)"
}

echo "1. Checking prerequisites..."
check_command uv "https://docs.astral.sh/uv/getting-started/installation/"

echo ""
echo "2. Installing Python dependencies..."
uv sync --frozen

if [ -f ".pre-commit-config.yaml" ]; then
    echo ""
    echo "5. Installing pre-commit hooks..."
    uv run pre-commit install || true
else
    echo ""
    echo "5. Skipping pre-commit hook installation (no .pre-commit-config.yaml)"
fi

echo ""
echo "============================================="
echo "✅ Setup complete!"
echo "============================================="
