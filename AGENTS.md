# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

A uv workspace (root project `dsl_workspace`, virtual/unpublished) hosting two published members
in LOCKSTEP versioning:

- `packages/dsl` — dist **habemus-papadum-dsl**, module `pdum.dsl`: type-keyed DSL compiler
  infrastructure (capture, two-tier cache, Node/Region IR, the value language, the fuse pipe,
  the reference evaluator).
- `packages/tensorlib` — dist **habemus-papadum-tl**, module `pdum.tl`: the assemblage tensor
  language (layout algebra, compute primitives, Program/Instr IR, AD, cost semantics, the zoo).

**THE SPECIFICATION is `docs/design/200_the-spec.md`** — the system, its principles, and the
migration plan (P0–P9). Everything numbered 010–195 in `docs/design/history/` is history.
Distilled backend knowledge: `docs/design/210_backend-notes.md`.

## Important Rules

### Version Management
**Never hand-edit version numbers.** Versions are LOCKSTEP across the root and every member,
written only by `scripts/_versioning.py` (driven by the release workflow). The anchor is
`packages/dsl/src/pdum/dsl/__init__.py`. Between releases the tree carries `X.Y.Z+dev`.
If a version change seems needed, tell the user; do not make it yourself.

### Release Management
**NEVER TRIGGER THE RELEASE WORKFLOW.** Releasing publishes ALL members to PyPI, creates a
public GitHub release, and pushes commits/tags. It is dispatched **by a human** from the
Actions UI (`release.yml`). Do not run `gh workflow run release.yml` under any circumstances;
do not suggest releasing unless the user asks about the process.

### Design discipline (from 200)
- The line budget is a tripwire, not a wall: crossing a cap in `scripts/loc_budget.py` is a
  conscious act with the reason recorded there — never silent growth.
- Refusal messages are frozen behavior (the refusal-contract battery pins wording).
- Oracle execution is always spelled: `reference(f)(...)`. Plain calls on unrouted kinds
  refuse by design — do not "fix" that refusal.
- Git history is the archive: delete, don't accrete; distill before deleting.

## Development Commands

```bash
./scripts/setup.sh          # bootstrap (uv sync --frozen, hooks)
uv run pytest               # all suites: D (packages/dsl/tests) + T/Z (packages/tensorlib/tests)
uv run ruff check .         # lint
uv run python scripts/loc_budget.py          # the budget gate
uv run mkdocs build         # docs
uv build --package habemus-papadum-dsl       # member build (never publishes)
```

Use `uv sync --frozen` for reproducible envs; re-run setup when dependencies change.
