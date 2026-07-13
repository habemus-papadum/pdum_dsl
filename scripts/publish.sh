#!/usr/bin/env bash
#
# publish.sh — build + publish habemus-papadum-dsl to PyPI from a maintainer box.
#
# MANUAL FALLBACK path. The PRIMARY release path is CI: the `release` workflow
# (.github/workflows/release.yml, run by a human from the Actions UI) gates on green CI, bumps +
# tags, builds from the tag, and publishes with the PYPI_API_TOKEN secret. Use THIS script only
# for out-of-band publishing: a hotfix, or when CI is down.
#
# It does NOT bump, tag, or push — it publishes the working tree as-is. Between releases the tree
# carries an X.Y.Z+dev version, which PyPI refuses (a PEP 440 local version), so an accidental
# run here fails at the upload instead of shipping an untagged build.
#
# Credentials: hatch reads HATCH_INDEX_USER / HATCH_INDEX_AUTH from the environment (it does NOT
# read ~/.pypirc).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."

cd "$REPO_ROOT"

rm -rf dist

uv run hatch build
uv run hatch publish
