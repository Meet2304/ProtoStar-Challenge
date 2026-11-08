# Plan: standard control baselines, characteristic scorecard and MLflow tracking

Status: **draft v4, for council review** (Sol + Grok). Author: Claude Code (Claude Opus 5.5), 2026-11-08.
Review history is in section 11 (v1: Grok 74; v2: Sol 83, Grok 80; v3: Sol 87, Grok 84). Deadline: the challenge closes
**2026-11-10 23:59 ET**, so the plan is tiered (A must ship, B deferrable, C stretch).

## 0. Goal and one correction

The team wants a rapid iteration loop: every new model is pushed through one **standard test suite** that
(a) estimates how it would score under the judges' formula, (b) profiles it along several characteristics,
(c) compares it with a fixed panel of **control baselines**, and (d) logs everything to **MLflow** so progress
is visible over time.

**Correction to the current mental model.** Models must *not* be scored on 2019–2025. The official FAQ forbids
choosing or comparing variants against public 2019–2025 observations, and `code/AGENTS.md` says the same. The
suite therefore *simulates* the task on history: at origin O the model sees data up to O and predicts O+1..O+7
(live analogue = leads 1–3, final analogue = leads 4–7), mirroring "know 2018, predict 2019–2025". The harness
for this exists (`code/evaluate.py`, `code/evaluation/`, schema-3 suite-v3: development origins 1983/1990/1997,
assessment origins 2004/2011 behind a committed freeze). **This plan extends it; it builds no parallel system.**

**Ownership.** `code/evaluate.py`, `core.py`, `protocol.py`, worker, runtime audit and report are being hardened
in parallel on branch `evaluation/opus-hardening-v4` by another agent. This plan only adds new files
(`controls.py`, `scorecard.py`, `panel.py`, `tracking.py`; `scripts/{eval_model,run_controls,mlflow_sync,_harness}.py`;
tests; docs) plus one agreed line in
`protocol.adapter_assets` (built-in allowlist). It uses the harness as a library: `run_suite`, `verify_suite`,
`verify_run`, `compare_runs`, `leaderboard`, `source_pins`, `installed_environment`, `PROTOCOL_VERSION`.

## 1. Design principles

1. **Clean the inputs, never the labels.** Labels and reference MAEs stay exactly as `build_fold` makes them
   from the raw pinned `annual_indices` with the 10-day rule.
2. **Controls are information-equivalent to models.** Every control runs through `run_suite` → worker
   (fold cutoff env, runtime I/O audit) → full-grid validation → the pinned official scorer. No shortcuts.
3. **Controls are fixed, untuned and versioned.** `CONTROL_SET = "controls-v1"`, hyperparameters fixed in code
   *from now on*. This is a frozen yardstick for future comparisons, not a prospective preregistration: `mean10`,
   `blend` and combo were already examined on overlapping historical periods. The organizers' three references
   (`average30`, `persistence`, `trend30`) are the only truly pre-specified floor. A cached panel entry is reused
   only if its run passes `verify_run` **and** its reuse key matches: hash of suite id, phase, protocol version,
   control set, `model_spec`, `installed_environment`, and the sealed source hashes (from the run's
   `source_pins`) of every file that can change a control's predictions or scoring: `common.py`,
   `combo_model.py`, `starter_model.py`, `evaluation/{adapters,controls,nested,official_score,core,worker,runtime_audit,protocol,assets,native_io,phase_policy,suite_identity}.py`,
   `uv.lock`, `pyproject.toml`, `requirements.txt`, `code/requirements.txt`. (Not *all* of `code/`: otherwise each teammate's new model file would force a
   panel rerun on every iteration.) Stale entries are rerun (a few minutes). Deliberate control changes → `controls-v2`.
4. **One headline, many diagnostics.** Headline = the harness's ranking quantity: equal-fold mean of
   *unclipped* final-analogue mean skill on development origins. Diagnostics explain; they never re-rank.
5. **Run directories are canonical; MLflow is a mirror.** Sealed run dirs are never modified after sealing.
6. **Only like-for-like comparisons**: identical suite id, phase, protocol version and input track.

## 2. Control baseline library (`controls-v1`)

`code/evaluation/controls.py`: `forecast(train, targets, origin, seed=42, *, method)` (keyword-only `method`),
registered in the `adapter_assets` built-in set as `("evaluation.controls", "forecast")` so official-track runs
need no asset manifest. Each control is a `run_suite` adapter run with params `{"method": ...}`. No file is read.

| id | Tier | Implementation | Role |
|---|---|---|---|
| `average30` | A | returns `adapters.reference(..., "average")` unchanged | Skill denominator (sanity: skill ≈ 0) |
| `persistence` | A | returns `adapters.reference(..., "last_year")` unchanged | Organizers' "last year"; noise floor |
| `trend30` | A | returns `adapters.reference(..., "trend")` unchanged | Organizers' "linear trend" |
| `mean10` | A | `combo_model.last10_forecast` + the same metric-wide training-median fill as `reference()` | "Ocarina of Ten": recency vs stability |
| `blend` | A | `combo_model.blend_forecast` (½ trend30 + ½ mean10) + the same fill | Strongest simple control in earlier backtests |
| `ridge_pooled` | B | see below | The "simple regression" control the team asked for |
| `starter_rf` | B | the challenge starter (lags 1–3, 100 trees, `random_state=42`) constructed with `n_jobs=1` **before** fit, recursive; recent lags filled 30-yr mean → all-history mean → metric median | The challenge's ML starter: what "naive ML" gets |

Because the three references return `reference()` unchanged (same row order), each origin's `adapter_output.csv`
matches the suite's `baselines/{average,last_year,trend}.csv` (byte-identical on this machine for all three
development origins; the gate compares parsed values so a last-digit float-text difference on another pandas or
platform cannot fail a correct panel). Every control yields finite values for all 5,880 rows (the same median
last resort as `reference()`). `mean10` averages however many of the last 10 years exist, then falls back to
`average_forecast`.

**`ridge_pooled`, leak-free by construction.** Logistic regression is a classifier and does not suit continuous
targets; ridge is the right linear control (`trend30` is already the per-city regression).
- One model per metric × lead k ∈ 1..7, pooled across cities.
- Pseudo-origins t ∈ [O−40, O−k]. For pseudo-origin t, **features and reference use only calendar years ≤ t**:
  ref_t = official 30-year mean ending at t (≥10 valid; else the example is skipped). Anomaly features vs ref_t,
  each with its own validity minimum and a fixed, training-free imputation of 0 when unmet (= "at the reference"
  for level features, "flat" for the slope). Pseudo-origins may have partial windows; the origin's own features
  are computed independently of the training cache:
  `lag1` = latest valid value in t−2..t (≥1); `mean5` = mean of t−4..t (≥3); `mean10` = mean of t−9..t (≥5);
  `slope30` = OLS slope over t−29..t (≥10). Target = value(t+k) − ref_t (skipped if invalid). Since t+k ≤ O,
  every number used is in `train`.
- `StandardScaler` fit in-fold, then `Ridge(alpha=1.0, solver="cholesky")`.
- Fallbacks: a lead with fewer than 50 pooled training rows, or a city whose ref_O is undefined, gets the
  `average30` prediction (then the median fill).
- No lat/lon in v1 (not in `train.csv`; reading `data/city_info.csv` would trip the worker's I/O audit unless
  declared as a `static_context` asset). Possible in `controls-v2`.

`starter_rf` falls back to `average30` only if fewer than 50 pooled lag rows exist (never on a real city history).

**Deferred to `controls-v2`:** `median30`, `ewma`, `theilsen30`, `damped_trend`. The `damped_trend` formula must be
re-derived before `controls-v2` from `trend_forecast`'s calendar-year slope and the exact midpoint of whatever
years the level averages (the v1 sketch "level + 0.5·slope·(lead + 4.5)" assumed a complete O−9..O window).

**Incumbents** (`role=incumbent`, never controls), run on suite-v3 development: `evaluation.adapters:combo`
(the method behind `submission.csv`, **the incumbent for G4**) and `evaluation.nested:forecast`.

## 3. Input tracks

| Track | Training input | Labels | Status |
|---|---|---|---|
| `official` | the suite's `train.csv` | sealed official answer key | **Tier A**: the only track for gates and model choice |
| `clean` | per-origin cleaned inputs | sealed official answer key | **Tier C stretch, diagnostic only** |

Clean track: `scripts/02_build_clean_inputs.py` runs `meet_cleaning.prepare()` once, then for each suite origin O
`build(P, end_year=O)` + `variants()` + `INPUT_RECIPE`, writing `data/processed/clean_inputs/v1/origin-<O>.csv`
(same schema as `train.csv`, years ≤ O, same city set) and `evaluation/clean-inputs-assets.json` with entries
`{"path": ..., "kind": "weather_input", "information_end_year": O, "origins": [O], "time_column": "year", "time_format": "year"}`.
A clean-track run reads the projected asset from `PROTOSTAR_ASSET_ROOT` and is scored against the existing sealed
answer key; no new suite is written.

**What it can and cannot show.** `prepare()` fits robust-z medians/MADs and neighbour-confirmed deletions on the
full daily record through 2018; `FILL_METHOD` was chosen by `simulate_fill` on the full record; `INPUT_RECIPE`
was chosen by `choose_inputs` on origins 1986/1994/2002/2010 with 8-year tests (labels 1987–2018, covering every
final-analogue window). Date checks cannot see this, because the information sits inside values dated ≤ O. So
clean-track runs are tagged `input_track_caveat=qc_fill_recipe_selected_on_full_record`, are **barred from gates
and model choice**, and the official-vs-clean gap is reported as a **contaminated descriptive comparison** (selection
on the evaluation years makes it optimistic; it is not a bound on the benefit of leak-free cleaning).

## 4. Characteristic scorecard (`code/evaluation/scorecard.py`)

Built from a verified run's `scores.json` and `origin-*/row_errors.csv` (requiring `scored` to parse as a real
boolean, as `score_fold` does), plus two evaluator-side analysis tables (`data/city_info.csv` for zones; the raw
`coverage_and_stations` parquet for station changes); nothing reaches a model. Written **outside the sealed run**
at `outputs/evaluation/scorecards/<run_id>.json` (+ flat `.csv`) with the run's `run_fingerprint` and a
`provenance` block: SHA256 of `scorecard.py`, `city_info.csv` and the coverage parquet, plus `scorecard_version`,
so a later rebuild that differs is detectable. Every ratio that can be undefined (zero or non-finite denominator,
e.g. a p90 reference error of 0) is `null`, never inf/NaN; the file is strict JSON.

**Counts.** Per-metric skills carry `n_rows`/`n_folds`; each lead carries `n_rows` and the number of folds with a
defined composite; zones carry `n_rows`/`n_cities`; bias carries `n` (final-split scored rows pooled across origins);
robustness carries `n_city_metrics`, `n_cities` and `n_cities_excluded`. City win = **strictly** lower MAE than the
30-year average (ties are not wins). p90 ratios are reported per metric and averaged only when all four are defined.

**Aggregation rule (never mixes units).** Within one metric, pool the group's scored rows and take
`skill_m = 1 − mean(|e|)/mean(|e_ref|)` against the suite's `average30` reference (undefined if the group is empty
or its reference MAE is 0). A composite is the **equal-weight mean of the four metric skills**, reported only when
all four are defined. Quantile ratios are per metric, then averaged. Origin weighting: headline and per-metric
numbers are equal-weight over folds (as the harness ranks); horizon is per fold, then equal-weight over folds; zone,
robustness and station-change sections pool rows across folds within a metric. Every grouped number carries its `n`.

| Section | Tier | Content |
|---|---|---|
| **Judge fidelity** | A | `headline_final_raw_skill`; `official_score_final_foldmean` / `_live_foldmean` (mean of per-fold clipped Scores = the leaderboard formula per fold); `selection_score_final` (clip after the cross-fold mean) |
| **Per metric** | A | Final and live skill per metric, equal-weight over folds |
| **Horizon** | A | Composite skill by lead 1..7 |
| **Stability** | A | Per-origin headline; min, max, range; worst metric × origin |
| **Bias** | A | Final mean signed error per metric, in units and ÷ reference MAE |
| **Spatial** | B | Final composite skill in 6 fixed zones (West/Central/East split at lon −104°/−87° × North/South at 37°N) |
| **Robustness** | B | Final split vs `average30`: city×metric win-rate; per-metric p90 \|error\| ratio, averaged; worst-decile city skill = 10th percentile over cities of the city composite (cities lacking a defined metric are excluded and counted) |
| **Inhomogeneity** | B | Per origin, cities with a station change in [O−10, O+7] (`data/raw/coverage_and_stations/train-00000-of-00001.parquet`, `ID`/`Year`/`Station Changes`): counts always; changed-vs-stable composite only if ≥10 changed cities. Cache absent → section marked skipped, scorecard still written |
| **Operational** | A | Total run wall time (`elapsed_seconds`; the harness records no per-fold time) |

## 5. Control panel (`code/evaluation/panel.py`, `scripts/run_controls.py`)

- Runs each control and both incumbents on the development phase of the official track and writes
  `outputs/evaluation/controls/<suite_id>/development/official/panel.json`: per entry the run id, run fingerprint,
  reuse key (principle 3), headline and per-metric skills, plus `control_set`.
- Sanity gates (fail loudly): `|average30 skill| < 1e-9` for every fold, split and metric (the reference MAE is
  computed in `build_fold` before the CSV round trip, so exact 0 is not guaranteed); each origin's
  `adapter_output.csv` for `average30` / `persistence` / `trend30` has the same key grid as
  `baselines/{average,last_year,trend}.csv` and values within 1e-6. Exact byte identity is recorded per entry
  (`reference_bytes_identical`) for information only.
- "Best control" = the control (excluding `average30`, the denominator) with the highest development headline;
  headlines equal after the harness's 1e-10 rounding go to the alphabetically **first** id.
- Panels are saved per **profile**: `A` (the five simple controls + combo + nested, the required panel) and `full`
  (A + `ridge_pooled` + `starter_rf`), as `panel-<profile>.json` with their entry lists. A Tier B failure cannot
  block or overwrite the Tier A panel. A saved panel is reused only after every entry re-verifies (`verify_run`,
  recorded fingerprint, recomputed reuse key), with summaries re-derived from the verified run.
- Assessment-phase panels only with frozen, committed controls, at the time a frozen candidate is assessed.

## 6. Rapid iteration command (`scripts/eval_model.py`)

```bash
uv run python scripts/eval_model.py --name kohav-gbm-v3 --adapter code/kohav_gbm.py:forecast --assets evaluation/no-assets.json
```

The script puts `code/` on `sys.path` itself (as the worker does). Thin `evaluate.py controls` / `mlflow-sync`
subcommands can be added by the CLI owner later. Development phase only.

1. `verify_suite`; ensures the panel is complete and current (reruns stale entries).
2. Runs the model on the development origins via `run_suite`.
3. Writes the scorecard.
4. Prints one screen: headline, per-metric skill, and `compare_runs(reference_run, candidate_run)` (returns
   candidate minus reference) against combo and against the best control: per-origin paired deltas, metric deltas,
   leave-one-period-out list. The harness's period-bootstrap range, when emitted, is shown only with its own
   interpretation (descriptive sensitivity under nested training sets, not a confidence interval).
5. Mirrors to MLflow (an MLflow failure prints a retry command and never fails the evaluation) and regenerates
   the HTML report.

**Gates** (official track only; fixed now; never auto-promote, never touch `submission.csv`; all comparisons use
the harness's 1e-10 tie tolerance): G1 headline > 0 (beats the 30-year average). G2 headline > the best control's.
G3 no metric with final skill < −0.02. G4 beats `combo` on headline, on ≥ 2 of 3 development origins, **and does
not lose on 1997** (delta ≥ −1e-10; the origin closest to a 2018 forecast).

Gates are screening heuristics, not statistical tests: three reused periods cannot establish leaderboard
performance. **Combo is not a clean comparator:** its per-metric rule and blend were chosen on tests 1987–1994,
1995–2002, 2003–2010 and 2011–2018 (`combo_model.BACKTEST_ENDS`), which overlap the development windows and cover
assessment origin 2011's whole label window. G4 (and any assessment comparison with combo) is therefore a
*replacement test against an incumbent fit on those years*, which favours combo. (The harness's combo adapter is
unrounded; `submission.csv` rounds to 3 decimals, which is immaterial to skill.)

**Predeclared assessment rule (bounded exposure).** Before *any* new assessment access, the team freezes **one
finalist** (or a finite, named shortlist together with the rule that picks among them, both written down first).
The finalist replaces combo only if its final raw skill beats combo's on both assessment origins (2004 and 2011).
There is no second round of candidates against the same assessment periods: successive candidates would turn
assessment into another selection loop. The result is described as **retrospective confirmation on reused
history**, not untouched evidence. Every attempt is in the existing access ledger.

## 7. MLflow tracking (`code/evaluation/tracking.py`, `scripts/mlflow_sync.py`)

- URI: `MLFLOW_TRACKING_URI` if set (e.g. a shared team server), else `sqlite:///<repo root>/outputs/mlflow/mlflow.db`
  resolved to an absolute path, artifacts in `outputs/mlflow/artifacts` (git-ignored). MLflow is locked at 3.18.0;
  the module imports it lazily; tests skip when it is absent (minimal CI deps) and use a temporary store, no network.
- Experiment per comparable group: `protostar/<full suite_id>/<phase>/<protocol_version>/<track>`.
- One MLflow run per verified, complete harness run, keyed by tag `harness.run_fingerprint`. Start time is set from
  `run.json` `created_utc` (parsed as `%Y%m%dT%H%M%SZ`, UTC) and end time from `elapsed_seconds`, so backfills plot
  when the evaluation ran.
- **Recoverable and serialized.** A mirror is complete only when the tag `harness.sync_complete=true` is written
  *last* (after params, metrics, artifacts and termination). A sync that finds a run with the fingerprint but
  without that tag (a crash mid-sync) deletes it and mirrors again; a complete one is skipped. Local-store syncs take
  an exclusive lock file (`outputs/mlflow/.sync.lock`, stale after 15 min) so two syncs cannot create duplicates.
- Params: model name, adapter, params JSON, track, control set, suite id, phase, seed.
- Tags: `role` derived from the sealed `model_spec` (controls adapter → control; combo/nested → incumbent; else
  candidate); `git.commit` and `git.dirty` from the sealed run metadata (`git.commit`; dirty = sealed porcelain
  `git.status` non-empty, `unknown` if it was `unavailable`; never the syncing machine); `history_status`;
  `scorecard_version`; `harness.run_id`; `input_track_caveat` (clean only). `author` only from sealed metadata if the
  harness adds it later; otherwise empty.
- Metrics: intrinsic only (fidelity, per-metric, per-lead, per-origin, bias, zones, win rate, runtime). Deltas vs
  controls/combo are computed at print time from `panel.json`, so an idempotent sync never freezes a delta.
- Artifacts: scorecard JSON/CSV, `scores.json`, `run.json`, `params.json`.
- Backfill: `uv run python scripts/mlflow_sync.py` (every verified complete run; also after copying a teammate's
  run dirs). Browse: `uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db`.

## 8. Tests

- Controls: deterministic (two runs, identical bytes); full finite grid; refuse rows after origin; keyword-only method.
- Real-data checks (skip without the raw cache): `|average30 skill| < 1e-9`; three references byte-identical.
- `ridge_pooled` boundary (unit tests on `ridge_features` and the target builder): perturbing any year > t leaves
  pseudo-origin t's features and ref_t unchanged; perturbing year t changes them; perturbing year t+k changes only
  that example's target, by exactly the perturbation; perturbing years in (t, t+k) or (t+k, O] changes neither.
  Missing recent years and an empty training design fall back to `average30` with finite output.
- `starter_rf` reproducible; finite for a city with no history.
- Scorecard: hand-computed fixture (per-metric skill, composite, bias, win rate, zones, p90); a metric in hundreds
  of units cannot dominate the composite; zero reference MAE → undefined, not an error.
- Panel: an entry whose `source_pins` / `model_spec` / environment / protocol differ is not reused.
- MLflow (skip if absent): idempotent sync into a temp store; start time = `created_utc`; refuses incomplete runs;
  a sync failure does not fail the evaluation.
- All existing tests still pass.

## 9. Order of work

1. `uv run python code/evaluate.py verify` (suite-v3 exists, id `4029caa1…`; never re-prepare it).
2. **Tier A, shippable on its own:** the five simple controls, Tier A scorecard sections, `panel.py`,
   `eval_model.py`, MLflow round trip (sync, retry, idempotence), tests. Freeze these files; tell the CLI owner.
3. Once all Python is stable (mid-run source changes deliberately fail runs), build the Tier A development panel
   with combo and nested; mirror to MLflow. **Tier A acceptance is checked here.**
4. **Tier B:** `ridge_pooled`, `starter_rf`, spatial / robustness / station-change sections (already drafted; they
   join the panel only after their tests pass, which reruns only their entries).
5. Docs: `docs/evaluation-baselines.md` (team guide + first results table), link from `evaluation/README.md`,
   README AI-tools entry.
6. Tier C stretch: clean track.

**Tier A acceptance:** all tests pass; the suite-v3 development panel with the five simple controls, combo and
nested is complete and current with sanity gates passing; MLflow shows them at their true run times;
`scripts/eval_model.py` evaluates a new adapter end to end in one command; `submission.csv` unchanged; no data
after 2018 read; no assessment origin opened. **Full acceptance** adds Tier B controls and sections.

## 10. Risks

- **Three development folds, reused history.** A mean over three reused periods is weak evidence; every comparison
  shows per-origin paired deltas and leave-one-origin-out sensitivity beside the mean.
- **Control drift.** Bound by the full reuse key; deliberate edits bump `CONTROL_SET`.
- **Parallel source edits.** Panel runs fail if another agent edits Python mid-run; runs are scheduled after both
  sides declare stable source.
- **MLflow divergence across teammates.** Run dirs are canonical; `mlflow_sync.py` rebuilds any store.

## 11. Review history

**v2 (after Grok r1, 74).** Tolerance `< 1e-9` + byte identity; references reuse existing forecasters; clean track
demoted to diagnostic; ridge pseudo-origin features, in-fold scaling, no lat/lon; controls registered as built-in;
combo/nested rerun on suite-v3, G4 names combo; control set trimmed to 7; scorecard outside the sealed run; MLflow
logs intrinsic metrics only.

**v3 (after Sol r2, 83; Grok r2, 80).**
- Ridge: per-feature validity minimums, fixed imputation, calendar-year windows, `cholesky`, explicit fallbacks;
  boundary test corrected (year t+k must change the target, and only the target).
- Scorecard: per-metric pooling then equal-weight composite everywhere; origin weighting stated; worst-decile
  defined; runtime is total, not per fold.
- Panel reuse bound to full `source_pins`, `model_spec`, environment, protocol and suite.
- Byte gate compares `adapter_output.csv`, not the re-sorted `predictions.csv`; finite values for all rows.
- suite-v3 verified, not re-prepared; keyword-only `method`; `FILL_METHOD` in the caveat; "upper bound" →
  "contaminated descriptive comparison"; clean track scored on the sealed answer key.
- MLflow: start time from `created_utc`; full suite id + protocol in the experiment name; author/dirty from sealed
  metadata only; repo-root URI; lazy import; tests skip without MLflow; sync failure non-fatal.
- `compare_runs()` for paired deltas + leave-one-out; G4 requires no loss on 1997; gates are screening heuristics;
  predeclared one-shot assessment rule.
- Tiers A/B/C; CLI in `scripts/` because `evaluate.py`/core/protocol belong to the parallel hardening work.

**v4 (after Sol r3, 87; Grok r3, 84).**
- Reference gate compares parsed values within 1e-6 on an identical key grid (bytes recorded for information):
  locally all nine origin×reference files are byte-identical, but last-digit float text can differ elsewhere.
- Ridge short-history `KeyError` fixed: origin features computed independently, partial windows allowed, tested
  with 38-, 12- and 1-year histories (12 years → exactly `average30`). Slope imputation described as "flat".
- Scorecard: every ratio guarded (`null`, strict JSON, serialization tested); per-metric p90 ratios; counts on
  per-metric, horizon, zones, bias, robustness incl. excluded cities; strict wins; `scored` must be boolean;
  provenance fingerprints for `scorecard.py` and both analysis tables.
- MLflow: completion tag written last; partial mirrors deleted and redone (tested); local lock against concurrent
  syncs; `git.dirty` from sealed porcelain status.
- Panel reuse key uses the sealed hashes of an explicit dependency list rather than all of `code/` (so new teammate
  model files don't force reruns); "best control" defined.
- `compare_runs(reference, candidate)` order stated; 1e-10 ties in every gate.
- Honesty about evidence: controls are a frozen yardstick, not prospective preregistration; only the three organizer
  references are pre-specified; combo was fit on years covering the development windows and assessment origin 2011,
  so G4 is a replacement test that favours combo; assessment exposure bounded to one frozen finalist (or a predeclared
  shortlist + rule), described as retrospective confirmation on reused history.
- Tier A is independently shippable and has its own acceptance; Tier B joins afterwards.

**v4.1 (implementation fixes after Sol r4, 91).**
- Cached panels are never trusted: `load_current_panel` re-runs `verify_run` on every entry, checks the recorded
  fingerprint and recomputed reuse key, and re-derives summaries from the verified run; a forged or corrupt
  `panel.json` is rejected (tested).
- Explicit panel **profiles**: `A` (five simple controls + combo + nested) and `full` (+ ridge, RF), each saved as
  `panel-<profile>.json` with its entry list; Tier A ships without Tier B. Tier B scorecard sections record an
  error instead of blocking the scorecard.
- `eval_model.py` refuses any non-official input track (gates are a model-choice signal).
- Tests added: hand-computed bias / zones / win rate / p90 / worst decile, ties are not wins, gate logic incl. ties
  and the 1997 rule, best-control tie-break (alphabetically first id), the real ridge example builder
  (`ridge_training_examples`), tampered panels. 25 tests.
- Per-metric `n_rows`; station subgroup row counts; metric deltas and the harness's resampling range (with its
  interpretation) printed; undefined values printed as `n/a`.
- MLflow: lock also guards any local SQLite URI; a remote server is single-writer by convention (documented); the
  control set and the scorecard CSV are logged. Zones are called *geographic* zones.
- `PANEL_SOURCES` also binds `native_io.py`, `phase_policy.py`, `suite_identity.py`, `requirements.txt`,
  `code/requirements.txt` (requested by the evaluator owner).

**v4.2 (after Grok r4, 88; most of its points were already fixed in v4.1, which it had not seen).**
- Real-data test (skips without local suite-v3): references within 1e-6 of the prebuilt baselines and
  `|average30 skill| < 1e-9` with the official scorer on all three development origins; writes only to a temp dir.
- Section 0 lists all four scripts; `starter_rf`'s 50-row fallback stated; `damped_trend` formula flagged for
  re-derivation before `controls-v2`; best-control tie direction (alphabetically first) and panel profiles stated
  in section 5.
