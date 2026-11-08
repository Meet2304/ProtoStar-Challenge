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

The first run after a code change also builds the control panel (a few minutes); later runs reuse it. The default
panel profile is `full` (all seven controls + combo + nested). If `ridge_pooled` or `starter_rf` ever break, add
`--profile A` to use only the five simple controls + combo + nested.

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

The full scorecard, including six geographic zones (a latitude/longitude split, not a climate classification) and
cities with station changes, is in
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

## First results (control panel `controls-v1`, development origins)

Built on 2026-11-08 at commit `d63347e`, protocol `audited-inputs-v5`, suite
`f2a07d24bded…` (`outputs/evaluation/suite-v3-r5`), profile `full`. Panel file:
`outputs/evaluation/controls/f2a07d24bded3db23024847598f4624b1caa2d2ce30539b6686ad3c476e531e5/development/official/panel-full.json`.
Skill is final-analogue (leads 4–7) unless marked live; 0 = the 30-year average, positive is better.

| Entry | Role | Headline | Live | Avg temp | Precip | Hottest day | CDD | 1983 | 1990 | 1997 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| average30 | control (denominator) | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| nested | incumbent | −0.007 | −0.009 | −0.008 | +0.000 | −0.021 | +0.003 | −0.004 | −0.015 | +0.000 |
| **mean10** | **best control** | −0.009 | +0.005 | +0.021 | −0.034 | −0.040 | +0.018 | −0.010 | −0.035 | +0.018 |
| combo | incumbent (submission method) | −0.013 | +0.013 | −0.019 | +0.000 | +0.000 | −0.034 | −0.037 | −0.029 | +0.026 |
| ridge_pooled | control | −0.023 | −0.001 | −0.040 | −0.016 | −0.022 | −0.012 | −0.058 | +0.001 | −0.010 |
| blend | control | −0.052 | −0.009 | −0.019 | −0.073 | −0.082 | −0.034 | −0.064 | −0.086 | −0.007 |
| trend30 | control | −0.122 | −0.037 | −0.093 | −0.135 | −0.141 | −0.118 | −0.141 | −0.163 | −0.060 |
| starter_rf | control | −0.265 | −0.189 | −0.309 | −0.349 | −0.227 | −0.173 | −0.191 | −0.307 | −0.296 |
| persistence | control | −0.375 | −0.429 | −0.517 | −0.480 | −0.223 | −0.279 | −0.314 | −0.553 | −0.257 |

Sanity gates passed: `average30` skill is below 1e-9 everywhere, and the three organizer references match the
suite's prebuilt baselines (the panel records them as byte-identical on this suite; the gate itself compares
values within 1e-6).

**What this says.**

- **The 30-year average is very hard to beat on these periods.** No control and neither incumbent has a positive
  headline on the 1983/1990/1997 origins (forecast years 1984–2004). Gains of a few hundredths of skill are the
  scale we are working at.
- **Recency helps where warming is strong, and hurts elsewhere.** `mean10` and combo gain on average temperature
  and CDD at the 1997 origin (+0.018 and +0.026 overall), and combo has the best *live* skill (+0.013). They lose on
  the earlier origins. The README's backtest (8-year windows from origins 1986/1994/2002/2010, Score clipped per
  window) showed combo positive in every window. This panel uses different origins and scores only the final-analogue
  leads 4–7 with unclipped skill, and combo comes out slightly negative on two of three. The spread between origins is
  as large as the differences between methods. Neither view is "the" answer for 2019–2025, so read every new model
  per origin, not only by its mean.
- **No evaluated method beats the average on precipitation or hottest-day temperature in these three periods.** This does not establish that those outcomes are unpredictable.
- **Naive ML is a weak control.** The starter random forest is far below the average (−0.26), and pooled ridge is
  slightly below it (−0.02). Any ML model we propose should at least clear `ridge_pooled` and `mean10`.
- **Combo vs the gates:** combo would fail G1 and G2 on these origins. On its own that is not a reason to drop it:
  three reused periods are weak evidence, and combo's strength shows on the most recent origin and in live skill.
  It does show why another declared historical assessment on 2004/2011 can inform a decision; those years also remain reused evidence.

Run IDs (all `20261108T…`): average30 `102320Z-e925f9acca`, persistence `102326Z-2de9f58382`, trend30
`102333Z-504c820bed`, mean10 `102341Z-68b6362630`, blend `102348Z-b5e41fc876`, ridge_pooled `102355Z-c8c32ad881`,
starter_rf `102419Z-fd1251f02d`, combo `102520Z-bbe55897c7`, nested `102526Z-6326cbdea0`.
