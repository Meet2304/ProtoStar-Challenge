# Brief: creative model strategies for the Placeholder series

Shared context for the two strategy planners and their reviewers. Written 2026-11-08 by the training station.
Deadline: challenge closes **2026-11-10 23:59 ET**, so any plan must be implementable and scored within about 36 hours of work.

## 1. The task

Forecast four annual metrics for 210 U.S. cities for 2019-2025, made as of 2018-12-31:
`annual_avg_temp` (F), `total_precip_in` (in), `hottest_day_temp` (F), `cdd` (F-days, base 65).

Score per metric: `skill = 1 - MAE / MAE(30-year city mean)`; challenge score = 100 x max(0, mean skill over metrics).
Live = 2019-2021, final = 2022-2025 (leads 4-7). A city-year counts only if <= 10 days are missing for each variable used.

The hard rule (`code/AGENTS.md`): no weather data dated after 2018-12-31 may feed a forecast, no pretrained model whose training
data reaches 2019 or has an unknown cutoff, and every value must come from code. Model selection only on years up to 2018.

## 2. Data available (all <= 2018)

- `annual_indices`: 28,052 city-years, the four metrics plus missing-day counts, from 1851 (most cities from ~1890-1950).
- `daily`: 10.2 M rows of `tmax`, `tmin`, `prcp`, 1851-2018.
- `city_info`: lat/lon (`data/city_info.csv`, declared as `static_context`).
- `coverage_and_stations`: days with both temperatures, station segment and station changes per city-year.
- Releases: `clean-v1` (gate-eligible, per-origin `annual_clean.parquet` with `_qc`, `_homog`, `_fill`, `_fill_homog` series);
  `features-v1` (**not** gate-eligible, so runs using it are labelled exploratory): per-origin annual seasonal/day-band
  features and as-of features (seasonal slopes, days >= 65 F, warm-season length, station history, similar-city pools).
- Static, weather-independent data (elevation, coastline distance, land cover, census) may come from any year if recorded in
  `data/README.md`. Climate indices (ENSO, PDO, AMO) are weather-driven: only values dated <= the forecast origin are allowed.

## 3. How a model is evaluated

- A model is **one single-file adapter** `forecast(train, targets, origin, seed=42, **params) -> DataFrame[city_id, year, metric, predicted_value]`,
  in `notebooks/Meet/`. `train` is the annual table (city_id, year, four metrics, NaN where invalid) cut at `origin`;
  `targets` is the grid of 210 cities x 7 years x 4 metrics. Helper modules outside `code/` need declaring, so keep one file.
- Extra files (coordinates, releases, saved weights) must be declared in an asset manifest JSON with kind and
  `information_end_year` <= origin, one file per origin for weather inputs. The runtime audit blocks undeclared reads.
  Daily data is not yet available as a per-origin release: a plan that needs it must build per-origin cuts and release them
  with `line.py release` (or ask the features station), and state the cost.
- Scored with `uv run python scripts/line.py score --name <Name> --adapter <file>:forecast --assets <manifest> --suite outputs/evaluation/suite-v3`
  from a clean commit (`official` label), then mirrored to MLflow with `line.py sync`.
- Test set suite-v3: **development origins 1983, 1990, 1997** (each forecasts origin+1..origin+7). Assessment origins 2004
  and 2011 are one-shot, run only by the integrator after a freeze; never read them. Headline = equal-origin mean of
  unclipped final skill (leads 4-7). Screening gates G1-G4: beat the 30-year mean, beat the best simple control
  (`mean10`, -0.0087), no metric below -0.02, beat combo (mean, >= 2 origins, not losing 1997).
- **Reward and verdict (from main `96c6257`, `evaluation/REWARD.md`; this supersedes the old keep rule).** `line.py score`
  now ends with a reward block; `uv run python code/evaluation/reward.py score <run_id>` and `reward.py ladder` give it for
  any sealed run. Reward = the host's leaderboard formula in points on the final analogue (leads 4-7 of 1983/1990/1997,
  per-metric skill unclipped). The verdict against the current **champion** comes from a paired city bootstrap (2,000 draws):
  `promote` needs P(better on final) >= 0.95, a win on >= 2 of 3 origins, live not clearly worse, and no metric below -2
  points; otherwise `within_noise` or `worse`. **Optimise in this order: verdict, then the lower bound of the 90%
  interval, then the reward.** A higher number that is `within_noise` is not progress and must not be reported as one.
- **Enhanced inputs:** the default external release is now `enhance-v2` (static elevation, coast distance, land cover;
  ERSSTv5 Nino 3.4, North Pacific and North Atlantic SST indices with per-origin anomalies, 1950-01 to 2018-12), pulled to
  `data/processed/releases/enhance-v2/` and declared by `evaluation/enhance-v2-assets.json` (copy its entries into a model's
  manifest). It is **not** gate-eligible: runs using it are `exploratory`, get a reward and a verdict, but cannot become
  champion. `enhance-v1` (static facts only) is gate-eligible. Bind only `data/processed/releases/enhance-v2/` paths.
  Since main `14bf1ef` (merged into wt/train at `0a5e6fc`), `line.lock.json` sets `enhance: enhance-v2`, and
  `code/enhance_ocean.py` provides `load_ocean_for_origin(release, origin, frequency="monthly")`, which verifies hashes
  and the fold cutoff; adapters should load ocean indices through it rather than reading the CSVs directly.
- **Combined feature tables (from the features station, 2026-11-08).** `features-v5` (not gate-eligible: includes
  enhance-v2) and its gate-eligible twin `features-v4` (identical but without the nine ocean SST columns), both pulled to
  `data/processed/releases/features-v{4,5}/`. One table per development origin, `origin-{1983,1990,1997}-combined.parquet`
  (one row per city and year up to the origin, 142 columns in v5): the four official values, lat/lon, CDD and day-band
  features, seasonal trends, station-change and missing-day history, similar-city pooled trends, the cleaning station's
  `*_qc/_fill/_homog/_nbfill/_adj` values, cleaned city facts, and (v5 only) ocean SSTs. Row t is what was known at the
  end of year t; pair row t's features with row t+h's target. Declare with the release's `assets-template.json` and read
  only the fold's own origin file through `PROTOSTAR_ASSET_ROOT`. Caveat: the adjusted columns were fitted on all data
  up to each origin, so inside a fold they have seen years after an inner pseudo-origin; inner validation must not rely
  on them as if they were causal at the pseudo-origin. **Rule for this round:** any version that must stay `official`
  (and so can become champion) uses `features-v4` or no feature release; `features-v5` is for exploratory variants only.
- Each fold must train only on years <= its origin. Inner validation (choosing hyperparameters) must use pseudo-origins
  earlier than the outer origin. The harness timeout is 900 s per origin by default.
- Hardware: 16 logical CPU cores, 15 GB RAM, RTX 3050 Ti laptop GPU (4 GB). Python 3.13 with numpy 2.5, pandas 3.1,
  scikit-learn 1.9, scipy 1.18. **PyTorch is not installed**; adding it means `uv add torch` (lock change, team-visible).
  Reproducibility is required: fixed seeds, deterministic results.

## 4. Where the series stands

| Version | Idea | Headline | 1983 | 1990 | 1997 |
|---|---|---:|---:|---:|---:|
| combo baseline | per-metric mix of average/trend/last-10 | -0.0134 | | | |
| v0.1 | pooled Huber + boosting on shift targets, shrinkage gate | -0.0008 | -0.0073 | +0.0050 | 0.0000 |
| v0.3 | + regional 30-yr warming trend W (ungated), learners fit city departures | +0.0079 | +0.0251 | -0.0386 | +0.0374 |
| **v0.5 (current best)** | 50/50 average of v0.3 and combo | **+0.0083** | +0.0027 | -0.0235 | +0.0458 |
| v0.9 | v0.5 with boosted member only | +0.0093 | +0.0031 | -0.0211 | +0.0458 |

Lessons so far (details in `docs/models/placeholder.md`, `docs/plans/placeholder-improvement-plan.md`):
1. Most fixable error is **shared warming** in temperature and CDD. Learners trained on 1940-1980 pseudo-origins learn
   mid-century cooling; removing era-wide shifts and adding a regional trend fixed 1983 and 1997.
2. The warming term **overshoots at 1990** (steep 1961-1990 slope, then a cool 1992-1997). Every version loses there.
3. **Precipitation skill is 0** everywhere (the gate always falls back to the 30-year mean). Hottest day is near 0.
4. Annual weather noise dominates at leads 4-7; skills are a few percent at best. Gains of 0.005 are meaningful.
5. All three development origins have been reused many times; only the integrator's 2004/2011 assessment is clean.
6. **Under the new reward, nothing has beaten the 30-year average yet.** The champion is still `control:average30`
   (0.000). v0.3-v0.9 score +0.17 to +0.93 final points, all `within_noise` (P(better) 0.56-0.84; v0.5 +0.834 at
   P 0.81, v0.9 +0.927 at P 0.84). v0.9's edge is all temperature, CDD and hottest day; precipitation equals
   climatology; it loses on origin 1990 (-2.1 points). On the live analogue (leads 1-3) v0.3-v0.9 are clearly better
   (+2.2 to +3.1 points, P = 1.00). The biggest open lever is final-analogue (lead 4-7) skill and the 1990-type regime.
   A plan's target is a `promote` verdict, which needs a gain that holds across cities and on at least 2 of 3 origins.

## 5. What a plan must contain

1. **Strategy and model-selection logic:** which model family and why it suits *this* data (210 cities, ~45-130 years,
   strong spatial correlation, a shared non-stationary trend, heavy annual noise), with the alternatives considered and
   why they were rejected. Creative methods are welcome (graphs, neural networks, Gaussian processes, state-space models,
   Bayesian hierarchical models, analogues, ...), but the choice must be argued from the data, not from novelty.
2. **Targets and inputs:** exactly what is predicted (level, anomaly, shift from the 30-year mean), the features, and how
   every input stays <= the pseudo-origin it is built at.
3. **Training procedure:** pseudo-origin construction, losses (MAE is scored), regularisation for small data, seeds,
   determinism, runtime per origin against the 900 s limit.
4. **Selection without peeking:** which hyperparameters are fixed in advance, which are chosen on inner origins and how,
   and a pre-declared keep rule. **Updated:** the primary rule is the reward verdict (`promote` against the champion,
   currently `control:average30`); among versions that do not promote, keep one over `Placeholder_v0.5` only if its 90%
   interval lower bound is higher AND its reward is higher. Exploratory versions (`enhance-v2`, `features-v1`) can show a
   gain but cannot become champion; say how a gain would be carried to a gate-eligible version.
5. **Versions:** a numbered sequence of single-change versions (names `Placeholder_v1.<n>` for plan A, `Placeholder_v2.<n>`
   for plan B), each scorable with `line.py score`, with at most 4 extra ablation runs per version.
6. **How each lesson in section 4 is addressed**, especially the 1990 overshoot and zero precipitation skill.
7. **Risks and fallbacks**, compute budget, and dependencies (new packages, new data releases).
8. **Tracking:** every scored run goes to MLflow via `line.py sync`; name the metrics to watch.

## 5b. Current champion (2026-11-08, after plan A's first runs)

The ladder champion is **`Placeholder_v1.2`** (run `20261108T180515Z-c2ea877eff`): the 30-year average plus a median-centred
city-minus-neighbours local term and a 60-year relative offset (plan A). Reward +1.268 final points; against average30
+1.268, 90% [+0.59, +1.94], better at 3/3 origins; per metric temperature +1.54, precipitation +0.20, hottest day +0.14,
CDD +3.19; by origin 1983 +0.02 / 1990 +2.48 / 1997 +1.31. `Placeholder_v1.1` promoted before it (+1.025);
`Placeholder_v1.3` (v1.1 + a national warm increment) scored +1.720 but was `within_noise` against v1.2 (better only at
1997). Any new version is compared with the champion recorded just before it is scored. These development numbers are
reused evidence for any design that sees them, and must be disclosed as such.

## 6. Insights and the loop rule (from main `7d3889f`, `evaluation/INSIGHTS.md`)

`line.py score` now prints, after the reward, the top 6 ranked findings (also `uv run python code/evaluation/insights.py
<run_id> [--json]`). **Loop rule:** pick one top finding per version and write the hypothesis down first; implement the fix
so it learns only from each origin's training data (never copy a fitted number); keep a version only on a `promote` verdict
(for non-promoting versions, the lower-bound-and-reward rule in section 5 still decides what is carried forward); rank
ideas by **honest** (cross-fitted) headroom, not "fit" numbers (v0.9's temperature bias fix is +1.72 fit but -2.69 honest).

What insights say about v0.9 (reward +0.93, within_noise):
1. Persistent per-city temperature offsets are 32% of squared error (honest headroom +0.98): robust or break-aware city levels.
2. Regime sensitivity: origin 1990 is -2.11; level bias flips sign by origin (temperature 1983 -0.98 F / 1990 +0.52 F /
   1997 -0.20 F; CDD -84 / +62 / -2), so a constant correction cannot help. Damp the trend, or fall back to the 30-year
   average when the trend is unstable.
3. The Central-North and East-North zones lose about 0.5 points each.
4. `control:mean10` beats v0.9 on CDD (+0.29) and CDD departures are too large (best scaling 0.72, honest +0.26): blend
   mean10 in for CDD, or shrink.
5. Precipitation equals climatology and no control beats it: low priority.

## 7. Diagnostic runs (from main `60bbde5`, `evaluation/REWARD.md` "Diagnostic runs")

`uv run python scripts/line.py score --diagnostic --name <name> ...` seals the run as `diagnostic:<name>`. It is scored and
labelled normally (`official` if it qualifies), MLflow tags it `line.role=diagnostic`, `reward.py score` and insights
report it, and `reward.py ladder` lists it as `diagnostic (<verdict>)` and **never promotes it**. Rules: declare each
diagnostic in the plan before running it; never score a model you might adopt with `--diagnostic`; if an ablation turns out
to be the better model, re-score it without the flag as a candidate version. Pre-declared development ablations (at most 4
per version) may therefore be run officially with this flag.
