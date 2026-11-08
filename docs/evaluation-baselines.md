# Control baselines, model scorecard and MLflow tracking

A guide for the team. It explains how to test a new model in one command, what the numbers mean, and how
progress is tracked. The design and its review history are in
[`plans/baselines-and-tracking-plan.md`](plans/baselines-and-tracking-plan.md).

## The idea in one paragraph

We cannot test models on 2019–2025: the official FAQ forbids comparing or choosing models against those
observations. Instead, the harness replays the challenge on history. At a past year O (1983, 1990, 1997), a model
sees only data up to O and forecasts O+1 to O+7. Years O+1..O+3 stand in for the live leaderboard and O+4..O+7
for the final one, and the scoring is the judges' formula (skill against each city's 30-year average, the
10-day missing-data rule, the 10-valid-year minimum). Every new model is compared with a fixed **control panel**:
simple baselines that any useful model should beat, plus our current submission method (combo). Every result is
mirrored to **MLflow** so the team can see progress over time.

## Test your model

1. Put a `forecast(train, targets, origin, seed, **params)` function in a file under `code/` (for example
   `code/my_model.py`). Fit only on `train` and return `city_id, year, metric, predicted_value` for every row of
   `targets`. See [`evaluation/README.md`](../evaluation/README.md#integrate-a-teammates-python-model).
2. From the repository root:

   ```bash
   uv run python scripts/eval_model.py --name my-model-v1 --adapter code/my_model.py:forecast --assets evaluation/no-assets.json
   ```

   Use `evaluation/no-assets.json` only if your model reads no files besides `train`/`targets`; otherwise declare
   them (see [`evaluation/ASSETS.md`](../evaluation/ASSETS.md)). Add `--params my_params.json` for keyword
   arguments.
3. Read the one-screen summary (below), then open MLflow:

   ```bash
   uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db
   ```

The first run after a code change also builds the control panel (a few minutes); later runs reuse it.

## The control panel (`controls-v1`)

| Control | What it predicts | Why it is there |
|---|---|---|
| `average30` | each city's mean of its last 30 years | the judges' denominator; its skill is 0 by definition (a sanity check) |
| `persistence` | the last valid year, repeated | the organizers' "last year" reference; the noise floor |
| `trend30` | a straight line through the last 30 years, extended | the organizers' "linear trend" reference |
| `mean10` | the mean of the last 10 years | recency without a trend |
| `blend` | half `trend30`, half `mean10` | the strongest simple forecast in our earlier backtests |
| `ridge_pooled` | ridge regression on recent anomalies (last year, 5- and 10-year means, 30-year slope), one model per lead, pooled across cities | the "simple regression" control (logistic regression is for classes, not numbers) |
| `starter_rf` | the challenge's starter random forest on lags 1–3 | what naive ML achieves |
| *combo* (incumbent) | the method behind `submission.csv` | the model to beat |
| *nested* (incumbent) | combo's method choice redone inside each historical origin | an honest version of combo's selection |

Controls are fixed. Changing one means a new control set (`controls-v2`) and a new panel. Build or refresh it
with `uv run python scripts/run_controls.py`.

## Reading the summary

- **Headline** = mean *unclipped* final-analogue skill over the three origins. This is the ranking number.
  Positive means better than the 30-year average; 0.05 means 5% lower error on average.
- **Leaderboard formula per fold** = `100 × max(0, mean skill)`, as the judges compute it, averaged over the three
  origins. It hides how far below zero a bad model is, so we rank on the headline.
- **Per metric / per origin / by lead** show where the model gains or loses. A model that wins only on one origin,
  or only at lead 1, is fragile.
- **Bias** is the average signed error divided by the 30-year average's error. A large positive number means the
  model forecasts too warm, wet or high.
- **Robustness** gives the share of city-metric pairs where the model beats the 30-year average, the ratio of its
  90th-percentile error to the reference's, and the skill of its worst 10% of cities.
- **Paired comparisons** give the model's skill minus combo's (and minus the best control's) for each origin, plus
  the mean with each origin left out in turn.
- **Gates** are screening rules, not statistical tests:
  G1 beats the 30-year average; G2 beats the best control; G3 no metric worse than −0.02 skill;
  G4 beats combo on average, on at least 2 of 3 origins, and does not lose on 1997 (the origin closest to 2018).

The full scorecard, including climate zones and cities with station changes, is in
`outputs/evaluation/scorecards/<run_id>.json` and in MLflow.

## Rules that keep the numbers honest

- **Development origins only** (1983, 1990, 1997). The assessment origins (2004, 2011) are opened only for **one
  frozen finalist** (or a shortlist and selection rule written down first), using the harness's `freeze` workflow.
  The finalist replaces combo only if it beats combo on both. No second round against the same periods.
- **All these years are reused history.** They have shaped earlier research, and combo itself was tuned on windows
  overlapping them. Treat results as evidence for comparing methods, not as a leaderboard forecast.
- **Clean the inputs, never the labels.** Labels always come from the official annual values with the 10-day rule.
- **Never** use 2019–2025 observations, and never tune on the live leaderboard.
- Nothing here writes `submission.csv`. Regenerate it with the chosen model's own script.

## MLflow

- Local store `outputs/mlflow/` (git-ignored). Set `MLFLOW_TRACKING_URI` to use a shared server instead.
- One experiment per comparable group: `protostar/<suite id>/<phase>/<protocol>/<track>`. Compare runs only
  within one experiment.
- Each run carries the role (control, incumbent, candidate), git commit and dirty flag from the time it ran, the
  scorecard metrics, and the run's `scores.json`/`run.json`/scorecard as artifacts. It is plotted at the time the
  evaluation actually ran.
- The run folders under `outputs/evaluation/runs/` are the source of truth. `uv run python scripts/mlflow_sync.py`
  mirrors every verified run (idempotent, resumes interrupted mirrors). After copying a teammate's run folders in,
  run it again.

## Files

| File | Role |
|---|---|
| `code/evaluation/controls.py` | the control baselines (`controls-v1`) |
| `code/evaluation/panel.py` | builds, validates and reuses the control panel |
| `code/evaluation/scorecard.py` | per-run characteristics |
| `code/evaluation/tracking.py` | MLflow mirror |
| `scripts/eval_model.py` | the one-command iteration loop |
| `scripts/run_controls.py` | builds/refreshes the panel |
| `scripts/mlflow_sync.py` | backfills MLflow |
| `tests/test_controls_panel.py` | tests for all of the above |

## First results

<!-- RESULTS -->
