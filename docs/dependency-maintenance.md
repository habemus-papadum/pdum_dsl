# Weekly dependency maintenance

The `weekly-dependencies` workflow runs each Tuesday at 08:47 UTC and can
be dispatched manually with `gh workflow run deps-refresh.yml`. It uses the
repository's locked Ship 0.12.0 or newer and the shared refresh action. First-party
packages are selected by `[tool.ship] ours`; normal maintenance retains manifest
floors and third-party compatibility constraints.

## Verification scope

`[tool.ship] refresh-test` owns the maintenance checks independently of release tests:

    uv run --locked pytest -q packages/dsl/tests packages/tensorlib/tests/test_axis.py packages/tensorlib/tests/test_layout.py packages/tensorlib/tests/test_indexing.py packages/tensorlib/tests/test_units.py packages/tensorlib/tests/test_transforms.py packages/tensorlib/tests/test_memory.py

The current remote main still owns versions/releases through its original lockstep scripts. Ship is added only with versioning=false; release scripts, versions and tags are untouched. Weekly checks cover the DSL compiler suite and six deterministic tensor layout/index/unit/transform/memory suites. GPU/framework conformance, notebooks, scientific benchmarks and publication remain separate.

## Outcomes and recovery

The shared action verifies the baseline before refreshing. A passing candidate
commits only intended lock/Ship-skill updates, provided main has not advanced,
and dispatches ordinary CI for that new commit. No change means no commit.
A failed valid candidate retains the exact proposed changes and diagnostics for
a draft PR; a baseline, resolution or authentication failure remains a failed
run with its report. Existing human/agent PR work is preserved. This workflow
never publishes a package or deploys a service.

The same steps can be inspected locally from an isolated clean checkout:

```sh
uv run --locked ship deps check
uv run --locked ship deps verify
uv run --locked ship deps refresh --report-dir .ship-refresh
```

Use the Node/pnpm and software prerequisites from the workflow where applicable.
Local refresh restores failed changes by default; the report retains the candidate.
Do not mistake a successful maintenance suite for checks outside the scope above.

## Rollout evidence

Prepared November 10, 2026. Manual workflow verification and any follow-ups are
recorded after the shared action release and pilot have passed.

The original frozen-lock baseline passed all 299 selected compiler/tensor tests. These checks used the existing lock before the shared Ship tooling upgrade; final workflow evidence follows below.
