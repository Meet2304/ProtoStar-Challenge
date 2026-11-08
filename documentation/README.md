# Data cleaning documentation

Index of every step taken to find flaws in the four training subsets and the strategy applied to each. The
code is [`scripts/02_clean_data.py`](../scripts/02_clean_data.py) (logic in
[`src/protostar_challenge/cleaning.py`](../src/protostar_challenge/cleaning.py)); the reviewed plan it implements is
[`notebooks/Meet/plan/cleaning-plan.md`](../notebooks/Meet/plan/cleaning-plan.md).

```bash
uv run python scripts/01_download_data.py      # raw data -> data/raw/ (pinned revision, up to 2018-12-31)
uv run python code/prepare_data.py             # official labels -> data/annual_valid.csv (already tracked)
uv run python scripts/02_clean_data.py         # cleaning + input-selection backtest -> data/processed/ (~10 min)
uv run python scripts/02_clean_data.py --skip-backtest   # reuse the saved choice if nothing changed (~4 min)
uv run python -m unittest tests.test_cleaning  # 17 acceptance tests on synthetic data (seconds)
```

## Which cleaned dataset to use

| Release | Built by | Use it for |
|---|---|---|
| **`clean-v1`** (current, gate-eligible) | `scripts/03_clean_per_origin.py`: one dataset per forecast origin, each from data up to that origin ([page 09](09-per-origin.md)) | everything scored on the assembly line: fetch it with `line.py pull`, read `data/processed/releases/clean-v1/origin-YYYY/` |
| `clean-v0` (not gate-eligible) | `scripts/02_clean_data.py`: the whole record through 2018 at once | exploration only: thresholds, outliers and station steps were fitted with later years in view, so runs on it are `exploratory` |

Pages 01–07 describe the cleaning steps, which are the same in both. Where they say "through 2018", "the end year" or
"chosen on data ≤ 1985", `clean-v1` applies the same step with the origin as the end year.

## Pages

| Page | Subset / topic | What is in it |
|---|---|---|
| [01 — daily](01-daily.md) | `daily` (10,245,754 rows) | structure, inverted days, stuck sensors, repeated rain, outliers, 2018 precision, day filling |
| [02 — annual_indices](02-annual-indices.md) | `annual_indices` (28,052 rows) | sentinels and fake zeros, official labels, the audit, QC values, fragments, percentage indices, empty-year infill |
| [03 — city_info](03-city-info.md) | `city_info` (461 rows) | schema, overlap, gaps, empty station periods, names and returning stations, short segments and clusters |
| [04 — coverage_and_stations](04-coverage-and-stations.md) | `coverage_and_stations` (28,052 rows) | verification, recomputed counts, validity, station context per year |
| [05 — station adjustment](05-station-adjustment.md) | across subsets | how station changes are detected, tested and adjusted (H1–H4) |
| [06 — input selection](06-input-selection.md) | across subsets | the rolling-origin backtest that decides which cleaned input a model should use, and its result |
| [07 — outputs](07-outputs.md) | `data/processed/` | every output file and column (data dictionary) |
| [08 — review history](08-review-history.md) | process | the plan's four review rounds with Sol (gpt-6.1-sol) and Grok (grok-4.7), scores and what changed |
| [09 — per-origin cleaning](09-per-origin.md) | `clean-v1` | one cleaned dataset per forecast origin, built only from data up to that origin |

## The flaws at a glance

| Subset | Flaw found | Evidence | Action |
|---|---|---|---|
| daily | Tmax below Tmin | 9 days, 1–11 °F apart, all inside physical limits | **swapped** (keeps the daily mean and CDD) |
| daily | stuck temperature sensor | 402 runs of ≥ 7 identical values; 5 where the neighbours varied (SD ≥ 5 °F) | **deleted** days 2…n of those 5 runs (44 readings, 1902–1931) |
| daily | repeated precipitation | 1,566 runs; 131 ≥ 0.10 in; 1 on days that were observed dry at ≥ 2 neighbours | **deleted** 2 days (Ely 1898-08-08/09); the rest kept with a reason |
| daily | outliers | 4,185 readings with robust \|z\| ≥ 5; 89.9 % confirmed by a neighbour; 424 isolated | **deleted** 1 (Greenville-Spartanburg 2018-04-28 Tmin 1.2 °F); 423 kept with reasons; review file + plots |
| daily | 2018 decimals | 47 % of 2018 Tmax, 57 % of Tmin have a decimal (°F↔°C conversion) | **kept** (the 2019–2025 answers use the same conversion) |
| daily | missing days | 416,164 empty rows, plus gaps | **filled** for inputs only (normal + neighbours' anomaly; chosen on data ≤ 1985 in `clean-v0`, on data ≤ each origin in `clean-v1`) |
| annual_indices | ±inf sentinels, 935 zero-precipitation rows (932 fully missing) | rows with every day missing | **NaN** under the official 10-day rule |
| annual_indices | detached record fragments | short blocks (< 10 years) cut off by > 20 empty years | **dropped** in QC/inputs (6 temperature, 6 precipitation blocks; 29,989 readings); long blocks kept and flagged |
| annual_indices | percentages count missing days as "not warm" | Warm Days 8.5 % with 0 missing days, 0.9 % with > 100 | **rescaled exactly** to % of valid days |
| annual_indices | empty years | city-years without a valid value | **infilled from neighbours** for inputs only (error −57 % vs the city's past 15-year mean, past-only test) |
| city_info | overlapping stations | Abilene 1943-03-01…1944-02-29 | change date = new station's start; overlap years kept out of step windows |
| city_info | station periods with no data | Asheville, Louisville, Lincoln, La Crosse | **removed** from the change list (avoids fake changes) |
| city_info | returning stations, same-name changes, short segments | 30 returns in 13 cities, Akron 1948, 17 periods < 5 years | station keys, change **clusters**, a return-pair consistency rule |
| coverage | stale after cleaning; validity not stored; station only on 1 Jan | verified exactly from `daily` and `city_info` | **recomputed** QC counts, `valid_<metric>`, station context per year |
| all | station changes (level jumps) | 229 change clusters; placebo-calibrated test | **adjusted** 42 Tmax and 61 Tmin steps (inputs only) |

## Three kinds of columns

The 2019–2025 answers are computed from the raw station record, errors included. So the processed data keep:

1. **Official labels** (`annual_avg_temp`, `total_precip_in`, `hottest_day_temp`, `cdd`): the published values with
   the official 10-day rule, equal to `data/annual_valid.csv` on every row (hard check). Never changed.
2. **QC values** (`<metric>_qc`): label + the exact effect of the repairs. Inputs only.
3. **Model inputs** (`<metric>_homog`, `_fill`, `_nbfill`, …, and the chosen `<metric>_input`): filled, adjusted or
   infilled variants. The default per metric is chosen by the backtest, where the official labels are a candidate.
   `clean-v1` has the variants but no `<metric>_input`: the training station picks per metric through the harness.

## Result of the input selection (page 06)

This table is `clean-v0`'s. Its backtests run to 2018, so these picks must not be carried over to a per-origin model.

| Metric | Shipped `<metric>_input` | Why |
|---|---|---|
| `annual_avg_temp` | `qc+homog` (station-adjusted) | +0.0032 skill over official inputs (bootstrap lower bound > 0), positive in the chronological check |
| `cdd` | `official` | the full variant led on the four origins (+0.0043), but the chronological check chose `official` in both folds, so the ship rule falls back |
| `total_precip_in` | `official` | no variant beat the official inputs |
| `hottest_day_temp` | `official` | no variant beat the official inputs |

The gains are small (+0.0032 skill on one metric ≈ +0.08 leaderboard points). The cleaning matters more for models
that fit long trends than for the current 30-year average and blend forecasters. All variant columns are kept so
later models can be tested on them with the same selection rule.

## Data rule

Only the challenge training data (weather up to 2018-12-31) is read. `assert_no_post_cutoff()` guards every table read
and written; each backtest fold is built inside a context that makes the same guard reject any year after the fold's
origin. No 2019–2025 data was used for any choice.
