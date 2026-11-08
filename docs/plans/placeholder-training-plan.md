# Plan: the "Placeholder" model series and Placeholder_v0.1

Status: **approved r5** (bar: Grok and Cursor each above 90/100. r1: Grok 78, Cursor 84; r2: Grok 93, Cursor 89; r3: Grok 88, Cursor 92; r4: Grok 93, Cursor 93. r5 applies r4's one remaining correction; changes in sections 12-14). Author: Claude Code (Claude Opus 5.5),
2026-11-08. Owner of this station: model training. Deadline: challenge closes 2026-11-10 23:59 ET.

## 0. Task, as the scorer sees it

Standing at 2018-12-31, forecast four annual metrics for 210 U.S. cities for 2019-2025 (5,880 rows). Per metric,
`skill = 1 - MAE / MAE(30-year city mean)`; `Score = 100 * max(0, mean skill)`. Live board = leads 1-3, final = leads 4-7.
All development happens on historical analogues: suite-v3, development origins 1983/1990/1997, each forecasting the next
seven years, through the team's development loop `scripts/eval_model.py` (which runs the harness behind `code/evaluate.py`).

## 1. The assembly line and this station's role

| # | Station | Current real output | What training can consume today |
|---|---|---|---|
| 1 | Cleaning (`notebooks/Meet/meet_cleaning.py`) | `data/processed/meet/annual_clean.parquet`, `daily_clean.parquet` | Not yet usable for gated runs: QC and input recipe were chosen on the full record through 2018 (baselines plan section 3). Usable only once rebuilt per origin. |
| 2 | Feature engineering | Advice only; no feature tables | Its recommended origin-anchored features are implemented inside this model for v0.1 |
| 3 | Enhancement | Research report only; nothing downloaded | Nothing yet; lat/lon from `data/city_info.csv` is the only static context |
| 4 | **Training (this plan)** | Combo + starter only | Suite `train.csv` per origin + declared static assets |
| 5 | Evaluation | Harness at commit a083840; `controls.py`/`scorecard.py`/MLflow tracking in progress | Adapter contract, official scorer, gates G1-G4 |

**Training station contract.** Input: a per-origin history table and declared assets. Output for each version: (a) a
single-file adapter `forecast(train, targets, origin, seed, **params)`, (b) a harness run on development origins with a
sealed run directory, (c) an MLflow run, (d) a model card entry with what changed and why. Each upstream station earns a
version bump only through a matched ablation against the previous version on the same folds. Training never edits a
shared evaluator file, never edits `submission.csv`, and never touches assessment origins 2004/2011 (the evaluation owner
gates those).

**Version rule.** `Placeholder_v0.N`: a new version per changed input/feature/learner group, one change group at a time,
with a pre-declared hypothesis. Trial count is limited (every harness run is a look at the three development folds, and
the report ledger counts them): at most 4 pre-declared ablation runs per version, run after a clean first run, reported
and unable to replace the version; everything else is exploratory on inner folds only.

**Handoff checklist to evaluation (checkable without the notebook):** the adapter calls `assert_no_post_cutoff` on its
input and refuses rows after its origin; loads through the worker, which sets `PROTOSTAR_FOLD_CUTOFF` before import;
reads coordinates only under `PROTOSTAR_ASSET_ROOT`; writes `placeholder_selection.json` and `placeholder_models.joblib`
under `PROTOSTAR_ARTIFACT_DIR`; and returns exactly `city_id,year,metric,predicted_value` for the full grid.

**File ownership (no shared source edited).** Harness `source_pins` cover `code/` and `src/` only, and an in-flight run
fails if those change. Placing the model outside them means Placeholder work cannot invalidate another agent's run:
`notebooks/Meet/placeholder_v0_1.py` (adapter, single file), `notebooks/Meet/placeholder-v0.1-assets.json` (manifest),
`notebooks/Meet/03-placeholder-v0.1-training.ipynb`, `notebooks/Meet/test_placeholder_v0_1.py`, this plan, and
`docs/models/placeholder.md` (model card). Promotion to `code/` happens only when a version is chosen for the submission,
coordinated with the evaluation owner.

## 2. What the data and task imply (first principles)

1. **The signal is low-dimensional.** At leads 1-7, annual weather is mostly unpredictable common-mode noise. The
   learnable part is how far a city's current climate sits from its 30-year mean, and how much of that gap and of the
   recent trend carries forward. Everything else is noise to be regularized away.
2. **Effective sample size is tiny.** Roughly 200 cities x 45 pseudo-origins x 7 leads is about 60k rows per metric, but
   neighbouring cities share weather and adjacent pseudo-origins share almost all of their windows. Independent samples
   of "climate shift" number in the dozens. This rules out high-capacity models.
3. **The score is relative to the 30-year mean, per metric, MAE.** So model the *shift from that mean*, in the metric's
   own units (keeping the scorer's city weighting), with a median-seeking loss, and make "predict zero shift" reachable
   exactly: a metric with no learnable shift should score skill 0, not negative.
4. **Precipitation and hottest day are close to noise** in every backtest so far; forcing structure there costs skill.
5. **Recursive lag models fail here** (the starter RF is worse than the 30-year mean on 3 of 4 metrics): each step feeds
   noise forward, and trees cannot extrapolate a raw level. Direct, origin-anchored forecasting avoids both problems.
6. **Regional and cross-metric pooling is the cheapest available gain.** A single city's 30-year slope is noisy; its
   neighbours' slopes are a strong prior. Warming in `annual_avg_temp` also carries information for `cdd` and hottest day.

## 3. Algorithm decision

**Direct multi-horizon pooled shift regression with an inner-fold shrinkage gate:**

`forecast(city, metric, origin + h) = ref30(city, origin) + lambda_metric * shift_hat(features, h)`

(Operationally the base is the reference chain of section 4, which equals `ref30` wherever it is defined.)

| Option | Decision | Reason |
|---|---|---|
| Per-city statistical trends (combo, nested) | Control, not candidate | No pooling; the incumbent to beat |
| Random forest with lags | Rejected | Fails in existing backtests; recursive; cannot extrapolate |
| Deep/sequence models | Rejected for v0.1 | Sample size (2.2); no regularization argument survives 2.1-2.2 |
| Pretrained forecasters | Rejected | Training-cutoff provenance is a disqualification risk |
| Full Bayesian hierarchical trend model | Deferred | Right idea, costlier; shrinkage below approximates it |
| **Pooled shift regression, two regularized learners** | **Chosen** | Matches 2.1-2.6, fast enough for honest nested validation, each step inspectable |

**Members** (no new dependency; both deterministic; single-threaded under the worker's thread limits), fixed in advance,
one model set per metric:

- `lin`: `StandardScaler` + `HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)`. Captures the persistence/trend
  structure and can extrapolate. The penalty `alpha * ||w||^2` is added to a loss *summed* over about 60k rows, so
  alpha = 1 is mild, not heavy. Fill rule: section 4.
- `gbm`: `HistGradientBoostingRegressor(loss="absolute_error", max_depth=3, learning_rate=0.05, max_iter=150,
  min_samples_leaf=500, l2_regularization=5.0, early_stopping=False, random_state=seed)`. Captures climate-dependent
  response (for example, CDD responds more to warming in warm cities). Native NaN handling.
- `shift_hat = 0.5 * lin + 0.5 * gbm`. Fixed: choosing weights with three inner folds would fit noise.

Hyperparameters are not tuned in v0.1. The single data-driven choice is the shrinkage gate (section 5). Capacity of the
boosted member comes from depth 3, leaf size 500, the learning rate and lambda; with `absolute_error`, leaf values are
recomputed and `l2_regularization` only slightly nudges split gain, so it is not relied on. For `lin`, the target is
divided by its robust scale only to condition the solver for large-unit metrics; predictions are multiplied back.

## 4. Training data and features

**Reference.** `ref30(city, t)` is exactly `common.thirty_year_mean` (calendar years t-29..t, at least 10 valid years,
otherwise missing).

**Training rows** for a fit with origin F (an inner origin or the outer origin), metric m, lead h in 1..7: pseudo-origins
t in [F-45, F-h], every year. Each row: features computed from columns `<= t` only; target
`y = value(city, t+h) - ref30(city, t)`, skipped if the target or `ref30` is invalid. Since `t + h <= F`, no row reads
past F. Row-level random splits are never used.

**Features** (raw metric units, computed at pseudo-origin t from a table truncated to `<= t`; same code for training and
prediction):

| Group | Features |
|---|---|
| Own level gap | `gap_lag1` (last valid value in t-29..t minus ref); `gap5`, `gap10`, `gap15` (mean over calendar years t-N+1..t minus ref, needing at least 3, 5 and 7 valid years, otherwise missing) |
| Own trend | `slope30` (OLS over valid years of the 30-year window, needs >= 10 valid else NaN); `trend_gap` = OLS line value at t+h minus ref |
| Climate context | `ref30`, `sigma30` (residual SD around the 30-year line), `n_valid30` |
| Regional pooling | `reg_gap10`, `reg_slope30`: mean over the 15 nearest other cities by great-circle distance (the city itself excluded; ties broken by `city_id`), using those with valid values at t; `nat_gap10`, `nat_slope30`: mean over all cities at t |
| Cross-metric | `tavg_gap10`, `tavg_trend_gap` from `annual_avg_temp` (duplicates of own features for the temperature model; harmless under regularization) |
| Horizon | `h` |

`lat`/`lon` are used only to choose neighbours, not as model inputs. They come from the declared `static_context` asset
`data/city_info.csv` (weather-independent), read through `PROTOSTAR_ASSET_ROOT` so the runtime audit accepts it.
Imputation for `lin`: anomaly and trend features (`gap*`, `slope30`, `trend_gap`, `reg_*`, `nat_*`, `tavg_*`) missing ->
0, meaning no signal; level features (`ref30`, `sigma30`, `n_valid30`) missing -> median of that fit's rows. Scaling and
medians are fitted inside each fit on that fit's rows only. `gbm` keeps NaN natively. Daily-derived features, station-change flags and cleaned inputs are out of v0.1 by design: they need
per-origin assets that do not exist yet (section 1).

**Prediction grid.** One function assembles forecasts for both inner scoring and the outer forecast, so the gate scores
exactly what is shipped: the base is `evaluation.adapters.reference(..., method="average")` on every row (30-year mean ->
mean of valid window years -> all-history mean -> metric-wide training median), the forecast is base + lambda * shift_hat
where `ref30` exists and base otherwise, and precipitation and CDD are clamped at >= 0. So lambda = 0 equals the official
30-year average on every row.

## 5. Nested validation and the shrinkage gate

For each outer origin O and metric: fit members once at each inner origin I in {O-21, O-14, O-7} using only rows with
`t+h <= I` and history `<= I`; each lambda only rescales those members' `shift_hat`, so nothing is refit per lambda. Score
the seven years after I against labels from the outer history using `evaluation.core.build_fold` and the pinned
`official_score.score` (same mechanism as `evaluation.nested`); the chosen lambda is transferred to the outer fit at O and
is never re-chosen on O+1..O+7; `lambda` grid {0, 0.25, 0.5, 0.75, 1}. Selection, per
metric: compute mean unclipped final-analogue skill per lambda across inner folds; pick the **smallest** lambda within
0.005 of the best, and lambda = 0 when the best mean is <= 0.

This gives precipitation and hottest day an honest route to exactly the 30-year mean. Then refit members on all rows with
`t+h <= O` and predict O+1..O+7. The only fallback is the pre-declared one: if the oldest inner origin has fewer than ten possible
prior calendar years (the `evaluation.nested` predicate: `history.year.min() > oldest_inner - 9`), the fold uses lambda = 0 with an explicit `fallback` record. Any other undefined inner evaluation
fails the origin, so a fold is never dropped or defaulted because of how it scored. Selection is leakage-safe within each
harness call, but the three origins' lambdas are not statistically independent: inner folds for outer origins 1990 and
1997 reuse years that are also other development origins' scored windows (section 13). Artifacts per fold under `PROTOSTAR_ARTIFACT_DIR`: `placeholder_selection.json`, fitted
bundle `placeholder_models.joblib`, feature list.

## 6. Tests and evaluation

**Before any harness run** (`notebooks/Meet/test_placeholder_v0_1.py`): information-boundary test (perturbing any value
in years > t leaves every pseudo-origin-t feature unchanged; for an inner origin I, perturbing any year > I leaves its
features, training targets and fitted predictions unchanged, and scored labels are only I+1..I+7); target rows never
exceed the fit origin; full-grid output with no NaN; determinism across two runs; lambda = 0 equals
`evaluation.adapters.reference(..., method="average")` on every row; the selection rule on a fixed table of inner skills
(margin, smallest-lambda, best <= 0 -> 0); and a synthetic fixture large enough to clear `min_samples_leaf`, with a
planted persistent shift, must recover lambda > 0.

**Asset manifest** (`notebooks/Meet/placeholder-v0.1-assets.json`, manifest schema 1): one entry,
`{"path": "data/city_info.csv", "kind": "static_context"}`; the harness pins and hashes it and supplies it to the worker
through `PROTOSTAR_ASSET_ROOT`. The adapter never opens the original `data/` path.

**Go/no-go for Tier A.** Start only after the evaluation owner posts the stable checkpoint for `audited-inputs-v5` (a
committed checkpoint with a stated runtime source hash; posted as commit d63347e), then
`uv run python code/evaluate.py verify`. If no checkpoint exists by 2026-11-09 12:00 ET, ask the owner and the team; the
guard is not bypassed. Runtime criterion before the main run: one development origin must finish in under 300 s on one
thread. Measured on origin 1997 (the largest history): see section 13. The per-origin `--timeout 900` then has a wide margin.

**Harness run, through the team's standard loop:** `uv run python scripts/eval_model.py --name Placeholder_v0.1 --adapter
notebooks/Meet/placeholder_v0_1.py:forecast --assets notebooks/Meet/placeholder-v0.1-assets.json --profile full --timeout 900`. This
verifies the suite, uses the control panel, runs the development phase only, writes the scorecard, prints paired
comparisons and gates, and mirrors to MLflow. The run's checks (origin guard, full grid, runtime I/O audit, official
scorer) apply unchanged. The adapter file is not edited while runs are in progress: the harness snapshots it once.

**Yardsticks:** the full `controls-v1` panel (all seven: average30, persistence, trend30, mean10, blend, ridge_pooled,
starter_rf) plus incumbents combo and nested, on the same development folds, reused from `panel.json` when current.
`ridge_pooled` is the closest design and the key comparison. **v0.1 hypothesis:** a gain over `ridge_pooled` and combo
from regional pooling, cross-metric temperature features, the boosted member and the shrinkage gate together.

**Pre-registered judgement (the team gates, verbatim from the baselines plan, as printed by `scripts/eval_model.py`):** G1 headline (equal-fold mean unclipped
final skill) > 0; G2 headline > best of the seven controls; G3 no metric with final skill < -0.02; G4 beats combo on
headline, on >= 2 of 3 development origins, and does not lose on 1997. Per-fold deltas and sign counts printed next to
every mean; with three folds, differences smaller than the fold spread are reported as noise. The notebook reports the
gate outcome exactly as the script prints it (the panel's best control excludes `average30`). Passing is the bar to *ask*
for a freeze and a one-shot assessment (a frozen candidate replaces combo only if it beats it on both assessment origins,
and a failed candidate is not tweaked and re-assessed). Failing is logged as a useful negative, and v0.1 still ships as
the first baseline of the series.

**Ablations (pre-declared, reported, never used to reselect v0.1):** `lin` only; `gbm` only; `lambda = 1` everywhere;
no regional and cross-metric features (ablation for 2.6). Identical folds and seeds. Exactly these 4 runs, run after the
main run is clean. They use adapter parameters, so the v0.1 definition stays the default parameter set, and each changes
one thing: `{"members": "lin"}`, `{"members": "gbm"}`, `{"gate": "none"}`, `{"features": "no_pool"}`.

**Expected null result if the gate returns lambda = 0 everywhere** (section 13): headline 0, so G1 fails (`headline > 0`);
G2 passes because the best control excludes `average30` and is negative on these folds; G3 passes; G4 fails because combo
is positive on 1997 while the reference is 0. That is a valid result for a first baseline, not a harness failure, and the
notebook reports it as such.

## 7. MLflow

`code/evaluation/tracking.py` (called by `scripts/eval_model.py`) is the single writer, so there is one MLflow record per
harness run with its schema as shipped: URI from `MLFLOW_TRACKING_URI`, else the absolute SQLite path under
`outputs/mlflow/`; experiment `protostar/<full suite_id>/<phase>/<protocol_version>/<track>`; run name = harness name
(`Placeholder_v0.1`); `role=candidate` and `git.*` from the sealed run; intrinsic metrics only, with deltas computed at
print time. A logging failure never fails the harness run or the gates. After the sync the notebook adds to that same
MLflow run: tags `series=Placeholder`, `version=v0.1`; params for hyperparameters, feature list and grid; metrics for
selected lambdas; artifacts `placeholder_selection.json`, `placeholder_models.joblib` per fold and the comparison table.
It finds the run with `tags.harness.run_id = <run_id>`, uses `MlflowClient.log_*` on that run with keys that do not collide
with those `tracking.log_run` stored, logs no paired deltas as metrics, writes nothing back into the sealed run directory,
and creates no experiment or second run.

## 8. Notebook

`notebooks/Meet/03-placeholder-v0.1-training.ipynb`, executed end to end: rationale; import adapter; inspect features on
one fold (origin 1983 training only); run the tests; run the harness; run the yardsticks and ablations; judge against
G1-G4 with paired comparisons; log to MLflow; write model card and next-steps. Keeps outputs small. Model code lives in
the adapter so the harness and the notebook run the same code. Order of work: Tier A (adapter, tests, main run, MLflow
mirror) must be clean before Tier B (ablations, model card detail). This notebook path suits development only: an
assessment freeze needs the adapter under `code/` or declared as a `source` asset, and all sources committed.

## 9. Data and integrity rules

Only suite `train.csv` for each origin plus `data/city_info.csv`. `assert_no_post_cutoff` on every table. No 2019+
weather from anywhere. No read of `origin-2004`, `origin-2011`, any `private/` directory, or legacy suites from model
code or analysis. `submission.csv` unchanged.

**Origin 2018 (pre-declared, not fitted in v0.1).** Run only after a freeze and its one-shot assessment, and only on the
team's decision: the identical frozen procedure (members on rows with `t+h <= 2018`, gate on 1997/2004/2011). Those years
are ordinary training history for a procedure with no remaining human decisions. A lambda carried over from the 1983-1997
folds would be a procedure the suite never evaluated, fixed from an older climate regime.

## 10. Risks

- Three development folds in a weaker-warming era may under-reward trend persistence relative to 2019-2025; the learner
  refits on each origin's own recent 45 years, and this is stated as a limitation rather than tuned around.
- Inner folds are noisy: mitigated by a five-point grid, the parsimony margin and lambda = 0 default.
- Shared workspace: another agent's change under `code/` mid-run fails the run by design; then rerun, nothing lost. The
  evaluation owner is moving the harness to protocol `audited-inputs-v5` (same adapter API; closes Arrow, local-filesystem,
  network and undeclared-project-file reads). No harness run starts until that owner posts a stable checkpoint.
- Runtime audit under v5: `data/city_info.csv` is read only from the declared snapshot under `PROTOSTAR_ASSET_ROOT`, never
  from the original `data/` path. The adapter stays one file (the entry point is hash-allowed; any helper outside `code/`
  or `src/` would need a declared `source` asset). Pinned imports of `common`, `combo_model`, `evaluation.core` and
  `evaluation.official_score` are allowed.
- Reused history: all folds are `reused_historical`, so results are comparative evidence only.
- `source_pins` also hash `uv.lock`, `pyproject.toml` and requirements, so no dependency is added for v0.1.

## 11. Next steps after v0.1 (separate versions, one change group each)

If the main run misses the gates, the pre-declared v0.2 is a single comparison: pseudo-origins restricted to the recent
30 years against the 45-year window, one change group, kept only if origin 1997's unclipped final skill is higher than
v0.1's (a G4 pass driven by 1983 and 1990 is not confirmation). Its rationale came from inner-fold evidence that partly overlaps
development outcomes (section 13), so a later pass on origins 1983 and 1990 cannot independently confirm it; the clean
development check is origin 1997, and a freeze still requires the one-shot assessment. Then: per-origin cleaned
inputs (needs station 1); per-lead members if v0.1 trails `ridge_pooled` at long leads (that control fits one model per
lead); v0.3 station-change and CDD-completeness covariates; v0.4 daily-derived
seasonal features; v0.5 elevation/coast/land-cover and compact ocean indices, each behind a matched ablation.

## 12. Changes in r2 (from r1 reviews: Grok 78, Cursor 84)

Adopted: `ref30` is `common.thirty_year_mean`, with lambda = 0 identical to the average reference on every row and one
assembly path for inner and outer forecasts; inner-origin boundary test and no dropped inner folds; G4's 1997 clause, G2
over all seven controls, `ridge_pooled` as the named comparison with a stated hypothesis; MLflow through the shipped
`tracking.py` only; explicit lambda semantics (no refit); ablation budget reconciled and tiered; fit-anchored
pseudo-origin windows; explicit window minimums and fill rules; lat/lon removed as model inputs; self-excluded neighbours
with a deterministic tie-break; selection rule tested on a fixed table; one-shot assessment rule cited; freeze-path note.

Declined, with reasons: (a) "`HuberRegressor(alpha=1.0)` makes the linear member nearly constant": the sklearn penalty is
added to a loss summed over rows, not averaged, so at about 60k rows alpha = 1 is mild; the test fixture will confirm
that `lin` is not near-constant. (b) Fixing origin-2018's lambda from development folds: that would ship a procedure the
suite never evaluated. Instead, the identical procedure runs only after freeze and assessment (section 9).

## 13. Changes in r3 and pre-run evidence (r2: Grok 93, Cursor 89)

Adopted from Cursor r2: concrete asset manifest; defined go/no-go with a dated escalation instead of an open wait; a
measured runtime criterion; explicit `--profile full`; MLflow lookup by `tags.harness.run_id`; `lin`/per-lead note; handoff
checklist; `scripts/eval_model.py` named as the development loop. Adopted from Grok r2: boosted capacity wording; gates as
the script reports them; the `evaluation.nested` fallback predicate; lambda transfer stated; MLflow key discipline and no
writes into a sealed run; a pre-declared v0.2 slice if the gates are missed.

**Pre-run evidence, from inner folds inside the three development `train.csv` files (corrected in r4).** No origin's own
forecast window was scored against its own outcomes, but scoring all nine inner folds did look at development outcomes:
origin 1990's inner origin 1983 has final years 1987-1990, which is origin 1983's headline window, and origin 1997's inner
origin 1990 has final years 1994-1997, which is origin 1990's headline window. Origin 1997's headline window (2001-2004) is
outside every development `train.csv`. v0.1 was not changed after this look, so its results stand as pre-registered; the
rationale for v0.2 was informed by it.
Adapter wall time for origin 1997 on one thread: 23.6 s, against the 300 s criterion. Across all nine inner folds
the model is about as good as the simple methods and sometimes better, but nothing reliably beats the 30-year mean:
final-analogue mean skill on average temperature, for example, is -0.141 for the 30-year trend, -0.060 for the trend/10-year
blend, -0.023 for the 10-year mean, and -0.007 for the ensemble at lambda = 0.5. The training windows of these eras include
mid-century cooling, so learned persistence does not pay off there; no bug was found. **Consequently the gate may pick
lambda near 0 on most metrics, and v0.1 may land close to the 30-year mean.** The pre-registered design is unchanged by this
evidence: tuning to pass on three reused folds would reduce the honesty of the result, and the regime question goes to v0.2.

## 14. Changes in r4 (r3: Grok 88, Cursor 92)

Grok must-fix adopted: section 13 now states which development outcomes the pre-run look overlapped, and section 11 limits
v0.2 to one pre-declared 30-versus-45 comparison that a later pass on 1983/1990 cannot independently confirm, with origin
1997 as the clean development check. Also adopted: the nine-fold mean final skill for every lambda and metric goes into the
model card from each run's `placeholder_selection.json`; a synthetic test that `choose_lambdas` scores only each inner
origin's next seven years and ignores anything after the outer origin. From Cursor: ablation arguments listed, section 3
base pointer, expected reading tied to section 6, checkpoint defined. v0.1's design and defaults are unchanged.
