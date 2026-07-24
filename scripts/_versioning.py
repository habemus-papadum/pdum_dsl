"""Shared version logic for the CI release pipeline (dependency-light: stdlib only).

The single source of truth for how ``pdum.dsl``'s **lockstep** version is discovered, read,
bumped, and written across all version-bearing files. Driven by ``.github/workflows/release.yml``
through the ``_cli`` at the bottom (``compute-release`` / ``set`` / ``latest-tag`` / ``current``).

Versioning model (see AGENTS.md):
  * **tag-as-truth** — the last release is the highest ``vX.Y.Z`` git tag; a release computes
    ``next = bump(last_tag, level)`` *at release time*, so patch/minor/major is chosen when you
    release, relative to the last real release — never guessed a release in advance;
  * between releases the working tree carries an ``X.Y.Z+dev`` marker (last release + a WIP
    flag). It is a PEP 440 *local* version, which PyPI refuses to upload — an accidental-publish
    guard. The release writes the clean ``X.Y.Z`` before building, so artifacts are never +dev;
  * **lockstep** — every published package shares one version, agreement is enforced.

This repo is a uv workspace (``[tool.uv.workspace] members = ["packages/*"]``). The ROOT is
the unpublished virtual root (design 200 §2); the published dists are the members —
packages/dsl (habemus-papadum-dsl, providing ``pdum.dsl``) and packages/tensorlib
(habemus-papadum-tl, providing ``pdum.tl``) — all sharing one lockstep version.
``discover_version_files`` globs the members, so adding one needs no edit here. The version
ANCHOR is ``packages/dsl/src/pdum/dsl/__init__.py``. A member depending on a sibling pins it
EXACTLY (``habemus-papadum-dsl==V`` in packages/tensorlib, since P3); the pin is rewritten in
lockstep with every bump and participates in the agreement check.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
INIT_PY = REPO_ROOT / "packages" / "dsl" / "src" / "pdum" / "dsl" / "__init__.py"
# Transitional mirror: the legacy tree's __version__, deleted with that tree at migration P1.
# The exists() check in discover_version_files makes the P1 deletion edit-free here.
LEGACY_INIT_PY = REPO_ROOT / "src" / "pdum" / "dsl" / "__init__.py"

_TOML_VERSION_RE = r'^(version = ")([^"]+)(")'
_INIT_VERSION_RE = r'(__version__ = ")([^"]+)(")'
# A member depending on a sibling pins it EXACTLY (lockstep, 200 §2); the pin
# is rewritten with every bump and checked by the lockstep invariant.
_INTERNAL_PIN_RE = r'("habemus-papadum-[a-z0-9-]+==)([^"]+)(")'


class VersionError(RuntimeError):
    """Raised on a malformed version or a lockstep disagreement across files."""


@dataclass(frozen=True)
class VersionFile:
    """A file whose version is bumped in lockstep with every release.

    ``kind`` selects the read/write strategy; ``published`` is False for the ``__init__.py``
    mirror (version-synced, but not itself a distribution).
    """

    path: Path
    kind: str  # "toml" | "init_py"
    name: str  # display / package name
    published: bool


def _toml_name(path: Path) -> str:
    """Best-effort project name from a pyproject.toml (falls back to the dir name)."""
    match = re.search(r'^name = "([^"]+)"', path.read_text(), flags=re.MULTILINE)
    return match.group(1) if match else path.parent.name


def discover_version_files() -> list[VersionFile]:
    """Find every version-bearing file across the workspace, dynamically.

    The (unpublished) root pyproject + every ``packages/*/pyproject.toml`` workspace member,
    plus the ``pdum.dsl.__version__`` anchor mirror. Adding a workspace member needs no edit
    here.
    """
    files = [
        VersionFile(PYPROJECT_TOML, "toml", _toml_name(PYPROJECT_TOML), False),
        VersionFile(INIT_PY, "init_py", "pdum.dsl.__version__", False),
    ]
    if LEGACY_INIT_PY.exists():  # transitional; the P1 purge deletes the legacy tree
        files.append(VersionFile(LEGACY_INIT_PY, "init_py", "legacy pdum.dsl.__version__", False))
    for pyproject in sorted((REPO_ROOT / "packages").glob("*/pyproject.toml")):
        files.append(VersionFile(pyproject, "toml", _toml_name(pyproject), True))
    return files


def _pattern_for(vf: VersionFile) -> str:
    return _INIT_VERSION_RE if vf.kind == "init_py" else _TOML_VERSION_RE


def read_version_of(vf: VersionFile) -> str:
    """Read the current version out of a discovered file."""
    match = re.search(_pattern_for(vf), vf.path.read_text(), flags=re.MULTILINE)
    if not match:
        raise VersionError(f"Could not find a version in {vf.path}")
    return match.group(2)


def write_version_of(vf: VersionFile, new_version: str) -> None:
    """Write a new version into a discovered file (internal sibling pins ride along)."""
    content = re.sub(
        _pattern_for(vf),
        rf"\g<1>{new_version}\g<3>",
        vf.path.read_text(),
        flags=re.MULTILINE,
    )
    if vf.kind == "toml":
        content = re.sub(_INTERNAL_PIN_RE, rf"\g<1>{new_version}\g<3>", content)
    vf.path.write_text(content)


def read_current_version(files: list[VersionFile] | None = None) -> str:
    """Read the version from every file and require agreement (lockstep invariant).
    Internal sibling pins (``habemus-papadum-x==V``) participate in the check."""
    files = files or discover_version_files()
    versions = {f"{vf.path.relative_to(REPO_ROOT)}": read_version_of(vf) for vf in files}
    for vf in files:
        if vf.kind == "toml":
            for m in re.finditer(_INTERNAL_PIN_RE, vf.path.read_text()):
                versions[f"{vf.path.relative_to(REPO_ROOT)} pin {m.group(1)[1:-2]}"] = m.group(2)
    unique = set(versions.values())
    if len(unique) != 1:
        lines = "\n".join(f"  {where}: {v}" for where, v in versions.items())
        raise VersionError(f"Version mismatch across packages:\n{lines}")
    return next(iter(unique))


def bump_version(version: str, level: str) -> str:
    """Bump a plain ``X.Y.Z`` version by ``patch`` / ``minor`` / ``major``."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
    if not match:
        raise VersionError(f"Invalid version format (need X.Y.Z): {version}")
    major, minor, patch = (int(g) for g in match.groups())
    if level == "patch":
        patch += 1
    elif level == "minor":
        minor, patch = minor + 1, 0
    elif level == "major":
        major, minor, patch = major + 1, 0, 0
    else:
        raise VersionError(f"Invalid bump level: {level}")
    return f"{major}.{minor}.{patch}"


def latest_release_tag() -> str | None:
    """The highest ``vX.Y.Z`` release tag as a plain ``X.Y.Z`` (rc/pre-release tags ignored),
    or None when the repo has no release tags yet. This is the source of truth for "last release"."""
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "tag", "--list", "v[0-9]*", "--sort=-version:refname"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in out.stdout.splitlines():
        m = re.match(r"^v(\d+\.\d+\.\d+)$", line.strip())
        if m:
            return m.group(1)
    return None


def compute_release_version(level: str) -> str:
    """The version this release will get: ``bump(last_release_tag, level)``. With no prior tag it
    bumps from ``0.0.0`` (so a first ``minor`` -> ``0.1.0``). This is the whole point of the
    model: the bump is applied *at release time* against the last real release."""
    return bump_version(latest_release_tag() or "0.0.0", level)


def dev_marker(release_version: str) -> str:
    """The between-releases working-tree version: ``X.Y.Z+dev`` (last release + a WIP flag)."""
    return f"{release_version}+dev"


def set_version(new_version: str, files: list[VersionFile] | None = None) -> None:
    """Write ``new_version`` into every version file."""
    for vf in files or discover_version_files():
        write_version_of(vf, new_version)


def _cli() -> None:
    """Thin CLI for release.yml. ``compute-release <bump>`` prints the version this release will
    get (``bump(last_tag, level)``); ``set <version>`` writes it — the clean ``X.Y.Z`` for the
    release commit, or ``X.Y.Z+dev`` for the finalize commit — across every file; ``latest-tag``
    and ``current`` print the last release tag and the working-tree version."""
    import argparse

    ap = argparse.ArgumentParser(description="Lockstep version helper for the CI release pipeline.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("latest-tag", help="print the last release (highest vX.Y.Z tag), else nothing")
    sub.add_parser("current", help="print the working-tree version (lockstep-checked)")
    p_cr = sub.add_parser("compute-release", help="print bump(last_tag, level)")
    p_cr.add_argument("bump", choices=["patch", "minor", "major"])
    p_set = sub.add_parser("set", help="write a version (e.g. 0.2.0 or 0.2.0+dev) across all files")
    p_set.add_argument("version")
    args = ap.parse_args()

    if args.cmd == "latest-tag":
        print(latest_release_tag() or "")
    elif args.cmd == "current":
        print(read_current_version())
    elif args.cmd == "compute-release":
        print(compute_release_version(args.bump))
    elif args.cmd == "set":
        set_version(args.version)
        print(args.version)


if __name__ == "__main__":
    _cli()
