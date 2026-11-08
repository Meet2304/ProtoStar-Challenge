# Evaluator checkpoint — stopped at user request

On November 8, 2026, the user requested: finish the current step, commit it,
and stop. The current step is complete: the supplementary station cohort was
preregistered, acquired with a 2018-12-31 weather cutoff, quality checked and
archived. No external model score has been opened, and no new Opus audit has
started. The last accepted Opus framework score remains **90.25/100**.

## Committed artifacts

- `e38ba3a`: V2 preregistration, separate builder, synthetic tests and frozen
  analysis runner, before V2 acquisition.
- `3433c6a`: 48 station candidates fixed before observations were fetched.
- `5c49935`: exact acquisition archive and provenance, the unscored coverage
  result and limitations. **33 stations across six strata qualify** under the
  unchanged thresholds. V1 remains a failed 5-of-24 coverage pilot, unscored.
- Archive: `evaluation/independent/snapshots/v2-acquisition.tar.gz`, SHA256
  `ff7f75a64130f461eecc7d1a7595b0d41de39362a91b9dcd4d7426a76f4755ce`.
  All 103 archived artifacts were independently checked against the manifest.

## State at stopping

The development evaluator remains on `parent-audited-v6`. Its latest full
control panel contains nine verified runs from source hash
`9e5767184eae3f975a5c74986ec8d223ba66c7a68ed149ffa26c6369c569ba2b`;
all have `requires_review=false` in their runtime input summaries.
The subsequent change to the independently owned cleaning module is recorded
in V2's preregistration. No model/scorer source changed for that amendment.

V2 binds source hash
`9faffc1689a9f5851f300cdd1bc1f9f553588626147fcc0a2b764c7f992502a3`.
Its included `src/protostar_challenge/cleaning.py` is still owned by the
cleaning thread and untracked at this checkpoint; it was not staged by the
evaluator. Before any future execution freeze, that owner must finish
verification and coordinate a committed source checkpoint. Later source
changes intentionally fail V2's preregistration guard. Do not weaken it.

The all-method execution freezes, external assessment runs, fresh Opus review
and final guide acceptance remain **not performed**. No claim of a score above
92 is made. The team guide remains a draft. The user's stop request takes
precedence over the previous review loop; no work should resume automatically.

All evaluator processes and delegated preflight work are stopped after the
already-running offline check finishes. Other teammates' dirty files are
preserved. No challenge submission was made, and `submission.csv` is unchanged.

## Completed checks

The final local offline sweep ran **236 tests: 233 passed, 3 skipped** in
310.234 seconds. This workspace sweep includes the cleaning owner's 17
synthetic tests, which remain untracked with that module. No real-data opt-in
was enabled. The standalone V2 selection/acquisition suite passes 22 synthetic
tests. The latest completed hosted core CI ran 197 tests with four skips on
both Windows and Linux; new V2 additions have not yet received a hosted run.

The existing full panel's MLflow mirrors completed. Six legacy schema-1
records cannot be mirrored because they lack `run_fingerprint`; they remain
archived and are not current-protocol results.
