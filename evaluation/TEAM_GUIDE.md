# Team testing guide

Draft for the current implementation. Final Opus re-audit is pending; do not
interpret this guide as acceptance of the requested review-score threshold.

## What the framework does

It simulates making a seven-year forecast at earlier dates. Each model gets
only observations available at that date, then predicts four annual metrics
for all 210 cities. The evaluator uses the published challenge scoring formula
and missing-data rules. It saves predictions, errors, model/source information
and a local comparison report. Nothing is submitted to the challenge.

## Start a testing session

Run commands from the repository root, `ProtoStar-Challenge`:

```bash
uv sync --locked
uv run python code/evaluate.py verify
uv run python code/evaluate.py baselines
```

On a fresh clone, run `uv run python scripts/line.py get-suite` once before
`verify`, and do not run `prepare`: a rebuilt suite holds the same data under a
different ID, so its runs cannot be compared with the team's. Setup and logging
are in the [team MLflow guide](../docs/team-mlflow-guide.md).

## Test a teammate's model

Put a `forecast(train, targets, origin, seed, **params)` function in
`code/my_model.py`. Inside it, fit your model using only `train`, predict every
row in `targets`, and return a DataFrame with these columns in this order:

```text
city_id,year,metric,predicted_value
```

```bash
uv run python scripts/eval_model.py --name my-model-v1 --adapter code/my_model.py:forecast --assets evaluation/no-assets.json
```

The command builds/reuses standard controls, writes comparisons and scorecards,
logs to MLflow when available, and refreshes the report. Open
`outputs/evaluation/report/index.html`. Compare raw final skill, every
metric, and the individual historical periods against the baselines. Negative
skill means worse than the thirty-year-average baseline. Keep each experiment's
run ID, parameters and source. Run `uv run python code/evaluate.py diagnose --run outputs/evaluation/runs/RUN_ID` for station, CDD and precision sensitivities. To try the supplied nested-selection example,
omit `--adapter`; that does not test your own model automatically.

**Enhanced inputs:** the default is `enhance-v2` (`line.py pull enhance-v2`, then `--assets evaluation/enhance-v2-assets.json`). Such runs are exploratory; see [REWARD.md](REWARD.md#enhanced-inputs-enhance-v2-is-the-default-from-2026-11-08).

**Reward for training agents:** `line.py score` ends with a reward block (leaderboard points on the final analogue, a paired city bootstrap against the current champion, and a `promote` / `within_noise` / `worse` verdict). See [REWARD.md](REWARD.md).

Use `evaluation/no-assets.json` only for a model that reads no additional files.
For saved weights, settings or extra inputs, supply your own [asset manifest](ASSETS.md).

Training elsewhere, such as Colab? Use each development fold's `train.csv` and
`targets.csv` under `outputs/evaluation/suite-v3-r5/origin-YEAR/`. Return
`origin-1983.csv`, `origin-1990.csv` and `origin-1997.csv` in one directory:

```bash
uv run python code/evaluate.py score --name my-colab-model --predictions my_predictions --provenance model_provenance.json
```

The [full guide](README.md#external-csv-predictions-and-required-provenance)
has the required provenance JSON: author, repository, full model commit,
parameters, dependency versions and training cutoff for each origin.

## Before the official submission

Develop using the development phase. Freeze the chosen method and parameters
before opening assessment results. Commit the intended model/evaluator source
and dependencies through the normal team workflow, then reuse the chosen
development run ID:

```bash
uv run python code/evaluate.py freeze --from-run outputs/evaluation/runs/RUN_ID --output outputs/evaluation/freezes/my-model-v1.json
uv run python code/evaluate.py run --name my-model-assessment-v1 --phase assessment --freeze outputs/evaluation/freezes/my-model-v1.json
```

Assessment access is logged. Changing the model after viewing results means
those results have influenced development; a new freeze cannot undo that.
Finally, generate the actual 2019-2025 forecasts with your selected model's own
submission-generation code, then check the file:

```bash
uv run python code/evaluate.py check-submission submission.csv
```

Historical evaluation does not generate a final submission automatically. A
registered teammate submits the chosen dataset commit on Hugging Face.

## How reliable is it?

The published scoring arithmetic is verified, and loader guards, file-read tripwires, imported-helper checks, full-grid
checks, deterministic run records and regression tests catch several common
mistakes. This is a trusted-team tool: Python and ordinary Arrow reads plus network attempts are checked, but hostile
native code or interpreter manipulation can bypass these tripwires, so review every model's input sources.

Historical performance is evidence for comparing methods, not an exact
leaderboard prediction. These historical periods have already influenced team
research; nested selection and freeze records do not turn them into untouched
data. Few periods, climate shifts and station changes limit generalization.
The private answer key and exact official reference errors remain unavailable.
The evaluator's engineering review score is separate from forecasting skill.

Never use public 2019-2025 weather observations to choose or tune models. Keep
them out of this iteration workflow even if they can be downloaded elsewhere.
