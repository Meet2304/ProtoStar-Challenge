# Team evaluation platform - audited runtime protocol

This is a historical model-development and assessment harness. It reproduces
published scoring arithmetic, enforces forecast-origin boundaries in supported
loaders, and records model-selection decisions and experiment provenance.

**The official targets are 2019-2025.** Public observations from those years
cannot be used to compare or tune variants under the [official FAQ](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence/blob/44b8ec9fcb6d4ad6243e14c5f72dc831d79d2ce6/content/data.md).
Internal observations and labels therefore end by 2018. A frozen forecast can
be analysed afterwards under the FAQ's declaration rule; this platform does
not fetch those observations or implement such an analysis.

## Quick start

From the repository root with Python 3.13:

```bash
uv sync --locked
uv run python code/evaluate.py prepare
uv run python code/evaluate.py baselines
uv run python code/evaluate.py run --name nested-development-v2
uv run python code/evaluate.py report
```

Open `outputs/evaluation/report/index.html`. No dashboard service or browser
server is needed. Share the complete report folder, including its copied `metadata/` links. `leaderboard.csv` is beside it. Regenerate the report after
adding runs. All commands accept `--help`.

`prepare` is a once-per-suite step and refuses to overwrite a suite. The local
workspace already has `suite-v3`, so use `verify` when resuming:

```bash
uv run python code/evaluate.py verify
```

The default is `outputs/evaluation/suite-v3`. Schema-1/2 suites and their saved
records remain archives; new runs and freezes against them are rejected. Build
a new schema-3 suite. Legacy records remain in separately labelled report groups.

Without uv, install `code/requirements.txt` and use your environment's Python.

## Data sources and scoring fidelity

The preferred source is the public CC0
[CMU-OSPO training dataset](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018/tree/bbbdebdab2a18e00f97ef87bd3f35a71e222180b),
revision `bbbdebdab2a18e00f97ef87bd3f35a71e222180b`. It contains daily and annual
records for 210 cities through 2018. The builder uses the raw annual parquet
without rounding and the tracked `data/city_info.csv` city list. It checks the
cached content/revision or fetches exactly that pinned training subset. The
known annual SHA256 is checked; supplying identical bytes with `--annual`
receives the same official-source identity, rather than a custom label.

[NOAA GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)
is a public source of station temperature and precipitation. Its catalog was
researched; no external observations were downloaded. Source revisions,
station composites, quality flags, observing periods, units and aggregation can
make reconstructed values differ from the private key. Provider requests for
model development must end by 2018-12-31, before downloading observations.

The official scorer is pinned to Space revision
`44b8ec9fcb6d4ad6243e14c5f72dc831d79d2ce6`. Source files, URLs, hashes and MIT
license are in `upstream/`. The extracted pure `score` function remains unchanged.
Tests check AST identity and numerical parity. The private answer key, exact
private mask and official reference denominators cannot be certified here.

For each origin:

1. Apply the ten-day rule to each metric-year. Temperature metrics, including
   hottest day and CDD, require at most ten missing Tmax AND Tmin days.
   Precipitation requires at most ten missing precipitation days. Unknown
   counts invalidate the metric-year; invalid all-missing `-inf` sentinels are
   masked. Other infinities are rejected.
2. Use the thirty years ending at the origin to calculate each city's baseline.
   At least ten valid years are required. Baseline eligibility stays fixed
   across the seven target years. Invalid targets are also unscored.
3. Require the entire city x seven years x four metrics grid, including unscored
   rows: 5,880 finite predictions for the default 210 cities. Reject duplicate,
   missing or unexpected keys, fractional years, NaN, infinity and booleans.
   Numeric strings in a CSV may parse to numbers; integral year values are
   required, not a particular CSV text representation.
4. Compute each metric's MAE over eligible city-year rows. Compute its reference
   MAE over the same rows, separately for the live and final analogues.
5. Skill = `1 - candidate_MAE / reference_MAE`. Official fold Score =
   `100 * max(0, mean(four_metric_skills))`. Individual skills are never clipped.
   Empty/zero reference errors fail suite construction. Preserve full precision.

The historical mask and baseline implementation reproduces the organizers'
published nine city-metrics / 63 rows without a baseline and the practice-run
CDD errors. These checks are now automated when the pinned raw cache is present.
CDD uses the supplied annual sum of `max(0, (Tmax+Tmin)/2 - 65)` in Fahrenheit
degree-days. `Days Both` can suggest stricter training-quality diagnostics; it
must not silently redefine the documented scoring mask. The private mask's
interpretation and later decimal precision remain uncertainty sources.

## Development, assessment and reused history

The new configuration declares roles before running:

| Origin | Targets | Phase | Live analogue | Final analogue |
|---|---|---|---|---|
| 1983 | 1984-1990 | Development | 1984-1986 | 1987-1990 |
| 1990 | 1991-1997 | Development | 1991-1993 | 1994-1997 |
| 1997 | 1998-2004 | Development | 1998-2000 | 2001-2004 |
| 2004 | 2005-2011 | Assessment, freeze required | 2005-2007 | 2008-2011 |
| 2011 | 2012-2018 | Assessment, freeze required | 2012-2014 | 2015-2018 |

`run`, `score` and `baselines` default to development. They do not execute or
score assessment origins. Assessment access is an explicit operation with a
matching freeze. Version-2 target windows cannot overlap, roles must partition
the origins, and development labels must end by the first assessment origin.

**All these years are reused historical evidence.** The current combo's method
choices already used every final-analogue year in suite-v1. Relabeling them or
changing fold boundaries does not make them untouched. The assessment role
adds procedural discipline going forward, not retroactive independence. Earlier
periods differ in climate regimes, baseline coverage and station changes; more
recent periods are informative diagnostics, not automatically unbiased tests.
Do not choose weights or the most favorable fold after seeing results.

## Nested selection is the default adapter

`evaluation.nested:forecast` evaluates a method-selection PROCEDURE inside each
outer training period, rather than using combo's globally chosen methods.

For outer origin O, the default inner origins are O-21, O-14 and O-7. Each has a
seven-year target window ending by O; each candidate receives only training
rows ending at its inner origin. Candidate methods are `average`, `trend`,
`last10` and `blend` from the existing forecasting code. For each metric, choose
the method with the highest mean UNCLIPPED final-analogue skill across those
inner windows. Canonical method order breaks exact ties. Refit the chosen
methods on all provided outer history, then predict the outer seven years.

Each worker saves `nested_selection.json` with inner train/target boundaries,
all candidate skills, selected methods and fallback decisions. Only the
provided outer history supplies inner labels; outer target observations are
never read by the selector. This repairs procedure-level look-ahead, while
previous research choices and repeated experimentation still limit independent
claims about generalization.

Supported parameters in a JSON file:

```json
{"inner_folds": 3, "candidates": ["average", "trend", "last10", "blend"]}
```

`inner_folds` must be 1-10; candidates must be a unique nonempty subset of the
listed methods. Unknown parameters fail clearly. If the oldest requested inner
origin cannot have ten prior calendar years, the WHOLE procedure falls back
to the thirty-year average, with an explicit artifact. It never silently drops
poor-performing folds. Other undefined reference evaluations fail.

```bash
uv run python code/evaluate.py run --name nested-custom --params my_params.json
uv run python code/evaluate.py run --name fixed-combo-diagnostic --adapter evaluation.adapters:combo
```

The fixed combo is retained for diagnostic continuity. It takes no tunable
parameters and is explicitly distinct from nested selection. Its forecast
values still match the current submission within its three-decimal rounding.
The evaluator never rewrites `submission.csv`.

## Integrate a teammate's Python model

Create an entry point such as `code/my_model.py:forecast`:

```python
def forecast(train, targets, origin, seed, **params):
    # train: city_id, year, optional name, and four metric columns.
    # targets: city_id, year, metric only; no observations.
    # Fit every transform and model using train only.
    # Return exactly: city_id,year,metric,predicted_value
    ...
```

```bash
uv run python code/evaluate.py run --name teammate-v1 --adapter code/my_model.py:forecast --assets evaluation/no-assets.json --timeout 900
```

Custom Python adapters require `--assets` with an explicit dependency manifest.
Use `evaluation/no-assets.json` only when the callback uses provided train/targets
and its code/installed libraries alone. If it loads weights, settings or other
files, list them in a model-specific manifest instead. See [asset declarations](ASSETS.md).

Declared files and metadata are pinned into the model specification. Each
worker receives only its origin-specific verified snapshot, available under
`PROTOSTAR_ASSET_ROOT`. Learned assets need a declared training-data cutoff;
weather CSV/parquet inputs also have their actual dates checked. Changes to
asset content, metadata or scope invalidate a freeze. Configuration/static
classification and true learned-data cutoffs remain author declarations.

Each fold runs in a fresh subprocess with seeds, CPU thread limits, timeout,
and `PROTOSTAR_FOLD_CUTOFF=origin` set BEFORE adapter import. The shared
`assert_no_post_cutoff`, `load_valid_annual`, and evaluation annual loader obey
this stricter context, including import-time loads. Invalid contexts fail
closed and cannot extend the 2018 ceiling. A legacy helper that reads the full
2018 file now fails in an earlier fold; port it to use the supplied `train`.

The worker's temporary current directory contains only copies of train,
targets and parameters. Its input path has no evaluator-label sibling. Model
artifacts can be written under `PROTOSTAR_ARTIFACT_DIR`. Relative paths to the
repository's full weather data no longer work by accident. Record explicit,
date-guarded paths for legitimate external forecast inputs.

This is protection against accidental leakage, not an OS security sandbox.
The Python tripwire and ordinary Arrow API wrappers catch common absolute
CSV/Parquet reads, including attempts whose exceptions the model catches. Network
connect/name-resolution/request events are denied in workers. Deliberate native I/O,
subprocesses and interpreter manipulation remain outside that boundary. All
weather inputs must use the guard; framework/GPU determinism requires its own configuration. For stronger
separation, one evaluator should distribute only train/target files and retain
labels separately. No claim of secure execution of untrusted code is made.

## External CSV predictions and required provenance

A teammate can train elsewhere and return `origin-1983.csv`, `origin-1990.csv`,
and `origin-1997.csv` for the development phase. Every file has columns exactly:

```text
city_id,year,metric,predicted_value
```

Training must be repeated for each historical origin; changing the year keys
on a 2019-2025 forecast does not produce a valid backtest. Supply a provenance
JSON, for example:

```json
{
  "author": "Team member name",
  "model_repository": "https://github.com/your-team/your-model",
  "model_commit": "REPLACE_WITH_ACTUAL_FULL_40_CHARACTER_SHA",
  "parameters": {},
  "dependencies": {"pandas": "3.1.1"},
  "training_cutoffs": {"1983": 1983, "1990": 1990, "1997": 1997, "2004": 2004, "2011": 2011}
}
```

```bash
uv run python code/evaluate.py score --name teammate-v1 --predictions my_predictions --provenance model_provenance.json
```

Each evaluated origin needs a declared training-end year no later than itself.
The run preserves the ORIGINAL CSV bytes and their hashes as well as normalized
predictions. The prediction directory is stored absolutely. External training
commits, data cutoffs and dependency declarations are not independently
certified by the scorer; the model author is responsible for their accuracy.
Document any pretrained model and its pre-2019 training cutoff too.

## Freeze a committed model before assessment

First commit the intended model/evaluator source and declared dependencies
using the team's normal Git workflow. The platform does not stage unrelated
files or make commits for you. Then:

```bash
uv run python code/evaluate.py freeze --output outputs/evaluation/freezes/model-v1.json --note "Method fixed before assessment access"
uv run python code/evaluate.py run --name nested-assessment-v1 --phase assessment --freeze outputs/evaluation/freezes/model-v1.json
```

To reuse a completed development run without repeating settings:

```bash
uv run python code/evaluate.py freeze --from-run outputs/evaluation/runs/RUN_ID --output outputs/evaluation/freezes/chosen.json
uv run python code/evaluate.py run --name chosen-assessment --phase assessment --freeze outputs/evaluation/freezes/chosen.json
```

The freeze verifies that the current suite, source and declared dependencies
still match the development record. Assessment infers frozen adapter, parameters
and assets; explicit overrides must match. External CSVs use the explicit workflow below.
For a frozen reference method, use `freeze --baseline average ...` followed by
`baselines --phase assessment --freeze PATH`. This evaluates that one method.
 `freeze` requires all Python under `code/` and `src/`, plus
project/lock/requirements files, to match their committed Git contents. It
normalizes Git text line endings for this check. It records the commit, code
pins, adapter, parameters, seed, suite identity, origins, selection rule and
historical-use label. Unrelated documentation edits do not invalidate a source
freeze. Adapters for assessment must be committed within the supported project
source tree or declared as source assets. Declared non-Python assets and their
manifest must also match committed blobs or materialized LFS oid/size.

Assessment verifies the freeze against the CURRENT committed source and
requested model. Changed parameters, code, original imported predictions,
suite or seed fail before evaluation. Freeze records refuse overwrite. Every
assessment attempt that passes the gate creates an immutable access event
under `suite-v3/assessment_access/`, including retries; completed runs retain a
copy of the freeze. This is an audit trail, not proof that nobody opened labels
manually. A new freeze cannot erase earlier exposure.

For external assessment predictions:

```bash
uv run python code/evaluate.py freeze --predictions assessment_predictions --provenance model_provenance.json --output outputs/evaluation/freezes/external-v1.json
uv run python code/evaluate.py score --name external-assessment-v1 --phase assessment --predictions assessment_predictions --provenance model_provenance.json --freeze outputs/evaluation/freezes/external-v1.json
```

The evaluator source must be committed; original assessment CSV hashes and
external author declarations are pinned as well. Their outside repository
commit remains an author declaration.

Installed Python and all installed distribution names/versions are pinned in
the freeze and rechecked at assessment. A descendant docs-only commit is allowed
when all source and declared assets are unchanged; changed source, environment
or unrelated Git ancestry fails. Package versions do not attest package binary
integrity or the true training history of weights.

## Ranking and paired comparisons

The register ranks by equal-fold mean FINAL UNCLIPPED skill, rounded to ten
decimal places for numerical ties. Negative periods lower the rank. Groups
must have identical suite, phase and protocol. Legacy runs remain separate.
The secondary internal summary is `100 * max(0, mean_raw_skill)`, clipping
ONCE after averaging. It is not another official scoring rule. The old mean
of clipped fold Scores remains in metadata only for diagnostic compatibility;
it no longer determines rank or the headline summary.

```bash
uv run python code/evaluate.py compare outputs/evaluation/runs/FIRST_RUN outputs/evaluation/runs/SECOND_RUN --output outputs/evaluation/comparison-v2.json
```

Positive raw-skill difference favors the second run. Output includes every
period's paired raw differences, metric/lead diagnostics, sign counts and
leave-one-period-out sensitivity. No individual four-year bootstrap is
reported. With at least three nonoverlapping periods, an optional bootstrap
resamples WHOLE paired periods, retaining all four leads and all cities in each.
Its percentile range is DESCRIPTIVE sensitivity conditional on observed
periods being representative, not a formal confidence interval or leaderboard
odds. Few periods, serial dependence, nested training histories, climate shifts,
and previous label reuse still limit inference. With two assessment periods,
no bootstrap range is emitted; inspect the two period differences directly.

## Artifacts, sharing and integrity

```text
outputs/evaluation/
  suite-v3/
    manifest.json
    origin-YEAR/{train.csv,targets.csv,baselines/}
    origin-YEAR/private/{answer_key.csv,reference_mae.csv}
    assessment_access/*.json
  freezes/*.json
  runs/RUN_ID/
    run.json, suite_manifest.json, params.json, scores.json
    freeze.json                         # assessment only
    code_snapshot/{code/,src/}
    asset_binding.json                  # explicitly declared dependencies
    environment/                        # lockfile and project/requirements metadata
    origin-YEAR/
      predictions.csv, scores.json, row_errors.csv, diagnostics.csv
      adapter.log, adapter_output.csv   # automatic models
      nested_selection.json             # nested adapter
      declared_assets/                  # only this origin's declared files/binding
      original_external_predictions.csv # imported models
  report/{index.html,leaderboard.csv}
```

Runs preserve failed status and logs. Timeouts produce a concise CLI error,
nonzero exit and log location. Changing source during a run invalidates it.
Reports/comparisons verify saved artifact hashes; version-2 metadata also has
a content fingerprint. Accidental changes are surfaced as errors, not silently
ranked. File owners could rewrite hashes too; this is not a signature service.

Outputs are git-ignored. Archive complete runs WITH their exact suite, freeze
and source/dependencies for reportable experiments; the HTML alone is not
provenance. Copy completed runs from teammates to a shared runs directory and
regenerate the report. Unique run IDs prevent overwrites. Failed/running runs
are excluded; a killed coordinator may leave a running record that must be
investigated. For a reproducible integration include all undeclared code/assets outside the
automatically snapshotted Python trees and all framework dependencies.

## Customize internal data

Copy `evaluation/config.json` to declare new nonoverlapping origins and phase
roles. Preserve the official seven-year horizon, lead splits, thirty-year
baseline, ten-day rule and ten-valid-year minimum. Every holdout ends by 2018.
The historical-use label remains `reused_historical`; the tool cannot certify
that a target was never seen in previous research.

```bash
uv run python code/evaluate.py prepare --config my_config.json --annual my_annual.parquet --cities my_cities.csv --suite outputs/evaluation/custom-v2
```

Custom annual CSV/parquet must use the official raw annual schema:
`ID,Year,Avg Temp,ToT Prcp,Max Tmax,cdd,Miss Tmax,Miss Tmin,Miss Prcp`, with
optional `Name`. Cities need `city_id`. Record external providers, exact
editions, licenses, units and processing in the data card. No post-2018 bypass
exists. Each immutable suite pins source bytes, city-file bytes, config, scorer
and every derived CSV.

## Human judging and official submission

The [official rubric](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence/blob/44b8ec9fcb6d4ad6243e14c5f72dc831d79d2ce6/config.py)
weights reuse 25%, interoperability 20%, reproducibility 20%, value 15%, feedback
15% and creativity 5%. Forecast Score is evidence under value and creativity,
whose combined weight is 20%; it is not a separate numerical bonus. Human
prize judging differs from forecast leaderboard ranking. The live board ranks
scores rounded to one decimal with a submission-time tie-breaker.

`rubric.json` preserves the descriptions. Copy `rubric-review.example.json`,
replace nulls with HUMAN integer ratings 1-5, and run:

```bash
uv run python code/evaluate.py rubric --ratings my_review.json
```

This applies `sum(weight * (rating-1)/4)` to your self-review. It does not predict
the judges' opinions. Finish the policy, AI-use and feedback evidence separately.

The repository's no-leaderboard-tuning rule is TEAM POLICY, stricter than the
official About text's invitation to check progress on the live board. The FAQ's
ban on selecting variants using public observed answers is an official rule.
Neither is relaxed by this implementation.

```bash
uv run python code/evaluate.py check-submission
uv run python -m unittest discover -s tests -v
```

The offline checker verifies the complete forecast grid, not Hub visibility or
the remote dataset card. A team member submits the intended dataset commit;
current configuration permits ten valid submissions per UTC day, latest counts.
This platform makes no uploads, official submissions or deployments.

## Verification and remaining limits

Regression tests cover the demonstrated ordinary-loader leak (including
import-time access), raw-skill ranking and ties, nested information boundaries,
a known winning candidate, freeze/source/parameter binding, assessment access
logging, preserved external inputs, tamper detection, timeouts/CLI exits,
invalid returns/booleans, zero references, original scoring parity and the
published real-data checks. Existing combo/submission parity is also tested.
The Windows/Linux CI workflow runs offline fixture tests after dependency
installation; raw-cache checks skip when that cache is absent. Hosted CI has
not been launched from this session.

Station-change filtering and Days Both sensitivity remain scientific diagnostic
extensions; they are not silently applied to the official-style mask. Recent
folds and pre-1984 secondary windows can be informative, but cannot restore
independence retrospectively. No historical score certifies the private answer
key or an official leaderboard position. See the Opus review and
`reports/evaluation-opus-fixes.md` for the implementation/validation record.

## Runtime checks and development sensitivities

Protocol `audited-inputs-v5` installs a Python file-open tripwire before adapter
import. Supplied history/targets/parameters and declared immutable assets are
readable; undeclared project files, suite labels and other evaluation outputs
are denied by the supported Python/Arrow paths. Ordinary Parquet, Arrow CSV,
Feather, dataset, memory-map and local-filesystem entry points are checked too.
Socket connection/name-resolution and urllib request audit events are denied.
Arbitrary extension internals and hostile monkey-patching can still bypass these checks. An attempted blocked read fails even if the adapter catches its error.
`io_audit.json` and `imported_source.json` survive ordinary callback failures.
Loaded helper modules outside the Python environment must match source pins or
declared source assets. This catches common accidental leaks, not deliberate
native I/O, subprocesses or interpreter manipulation. Review input sources.

Schema-3 suite IDs use normalized logical inputs and preparation source rather
than platform-dependent output bytes. Derived CSVs use LF. Separate manifest
and artifact hashes still detect physical changes. Windows and Linux runs of
the same logical suite can be compared, with tight baseline floating tolerance.

```bash
uv run python code/evaluate.py diagnose --run outputs/evaluation/runs/RUN_ID
```

This requires a completed current-protocol development run and identical local
annual source; it never downloads data. With verified cached station coverage,
it reports each development period under stable target stations, stable baseline
and target stations, strict CDD Days Both validity in BOTH baseline and targets,
and hypothetical whole-degree temperature labels. Predictions stay fixed.
Missing coverage is explicitly unavailable, not silently treated as stable.
Bundles are separate immutable files under `outputs/evaluation/diagnostics/`;
`metrics.csv` contains per-metric counts, MAE, references and skill changes.
These sensitivity scores never replace official-style scores or establish
station-change causality. The private key's mask and precision remain unknown.

The HTML report displays known development trial counts, assessment accesses
and unique freezes across each suite/protocol, including failed runs and retries.
Missing/unshared ledgers are disclosed as unknown totals with known lower bounds.
It cannot count experiments or label access outside the platform.

## Protection across custom suites

The default assessment target years 2005-2018 for the 210 challenge city IDs
remain reserved across suite names and custom configurations. The pinned
training source, and converted CSVs retaining these IDs, cannot assign those
city-years to development. Preparation and verification fail before scoring;
there is no silent override. The report warns about known historical exposure
across suites, including archived development runs. This does not erase the
team's earlier use of these labels or certify untouched evidence.

Successful adapter runs require both worker audits to exist and say `verified`
before scoring. A zero process exit alone cannot complete a run. Network inputs
must be obtained, reviewed and declared outside model execution before testing.

## One-command team loop

The integrated control panel and optional MLflow mirror are documented in
[the team baseline guide](../docs/evaluation-baselines.md). For a supplied-history
model, after the suite is prepared:

```bash
uv run python scripts/eval_model.py --name my-model-v1 --adapter code/my_model.py:forecast --assets evaluation/no-assets.json
```

This builds/reuses the standard controls, evaluates the candidate on development
origins only, writes characteristic scorecards and paired comparisons, mirrors
verified evidence to MLflow when available, and refreshes the local HTML report.
The gates are team screening criteria, not official judging rules or independent
validation. Controls and incumbents retain separate roles. Extra model dependencies
need their own asset manifest. No challenge submission is made.
