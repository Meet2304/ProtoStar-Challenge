# Official eval-v1 development baseline bar

Seven fresh `controls-v1` runs were scored through `line.py score` from clean
commit `bbd231edfe163121c58eaf5cd2bc1d7241513ab2` on `Meet/eval`, after the line clock restarted.
They use locked suite-v3 `4029caa1fb85cd376baf677e5d89549655684044aac68b6b3efb368eb6e46872`,
development origins **1983, 1990 and 1997 only**, and provided training history.
All seals and worker audit receipts verify. Each run classifies as `official`
under the assembly-line rules; MLflow sync remains pending HTTP authentication.

| Baseline | Mean final raw skill | Run ID |
|---|---:|---|
| `average30` | +0.000000 | `20261108T130320Z-4c597eb838` |
| `mean10` | -0.008718 | `20261108T130405Z-3b5e492e21` |
| `ridge_pooled` | -0.022528 | `20261108T130438Z-b33d4534e2` |
| `blend` | -0.052378 | `20261108T130424Z-a94b9b4840` |
| `trend30` | -0.121566 | `20261108T130350Z-27d31d352a` |
| `starter_rf` | -0.264774 | `20261108T130517Z-fd94720094` |
| `persistence` | -0.374594 | `20261108T130336Z-3ea957a5c3` |

Rank on equal-period mean **raw** final skill. Zero is the thirty-year-average
reference; negative values are worse. Mean10 is the strongest additional control
on these periods. Inspect each period and metric; three reused periods are not a
formal significance test or an estimate of the private leaderboard. Near-zero
reference skill is floating-point noise. Nine keyed reference checks (three
reference models times three origins) match the suite forecasts exactly; CSV
row order differs. No evaluator source, controls or preregistration changed.

## Use the bar

Score every candidate through `uv run python scripts/line.py score` with the
locked `--suite outputs/evaluation/suite-v3 --phase development`. Commit its
source/assets first. Compare against the exact baseline run IDs above, rather
than a cached panel from before the assembly-line clock restarted.

The analysis-only eval data release contains CSVs by period and metric, the
plot, provenance index, seven complete portable sealed baseline directories
under `baselines/`, and the integrator-only assessment handoff. Obtain it through
`line.py pull eval-v1`; do not use the analysis release as model training input.
For a candidate already scored through the assembly line, a paired read-only
comparison can use `code/evaluate.py compare <released-baseline-dir> <candidate-run-dir>`.
The delta is candidate minus baseline.

## Assessment and release stability

Only the integrator runs assessment, once for the chosen model, across origins
2004 and 2011 together. See [the prepared handoff](../evaluation/ASSESSMENT_HANDOFF.md).
No chosen model has been frozen here, no assessment was scored, and its known
cohort access count remains zero. Counts do not include prior manual/raw/test
reads; historical labels remain reused.

Evaluator **eval-v1** stays fixed: bug fixes only, at most one evaluator release
per half-day. Publishing this analysis data release does not change the evaluator
release or reset any assessment budget. No observed 2019+ weather was acquired,
and `submission.csv` is unchanged.

## Verification

Seven end-to-end `line.py score` runs completed; all 21 development folds have
verified parent audit receipts and clear input reviews. All copied sealed runs
pass `verify_run`. The nine reference grids have zero keyed prediction
difference. The restored frozen-source/preregistration tests pass **22/22**.
A synthetic PowerShell check allowed one mock assessment launch and blocked a
repeat; no actual assessment command or freeze was executed.

**Open finding D9:** the shared MLflow service returns HTTP 401, including after
the private supplied login was tried. Runs and the analysis release remain
valid locally. The integrator needs to provide a valid login and run
`line.py sync`; do not rerun models to repair tracking.
