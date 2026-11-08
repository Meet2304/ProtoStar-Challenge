# 07 — Outputs and data dictionary

Everything is written by `scripts/02_clean_data.py` to `data/processed/` (git-ignored; rebuild with the script). The
only tracked file it writes is the review file `data/qc/daily_outlier_decisions.csv`.

`clean-v1` has the same tables under `data/processed/releases/clean-v1/origin-YYYY/`, with rows only through that origin,
no `<metric>_input*` columns, and the Warm/Cold and Wet Days indices blank at 1983. Differences: [page 09](09-per-origin.md).

## Files

| File | Rows | Content |
|---|---:|---|
| `annual_clean.parquet` | 28,052 | one row per city-year, key (`city_id`, `year`), 1:1 with `annual_indices` |
| `coverage_clean.parquet` | 28,052 | how each city-year was measured, same key |
| `stations_clean.parquet` | 461 | one row per station period |
| `daily_clean.parquet` | 10,245,754 | one row per city-day (the record span of each city) |
| `qc/daily_qc_log.csv` | — | every repaired or deleted reading: city, date, var, old, new, action, rule |
| `qc/stuck_runs.csv`, `qc/repeated_precip_runs.csv`, `qc/precip_extremes.csv` | — | every run / extreme tested, with action and reason |
| `qc/fragments.csv`, `qc/record_breaks.csv` | — | dropped blocks and flagged breaks |
| `qc/homog_steps.csv`, `qc/homog_placebo_calibration.csv` | — | every cluster × variable tested; placebo t_crit and rates |
| `qc/fill_bakeoff.csv`, `qc/infill_evaluation.csv` | — | evidence for the fill and infill methods |
| `qc/outlier_plots/*.png` | 9 pages | the 424 isolated outliers with their neighbours, ± 10 days |
| `backtest_skills.csv`, `backtest_bootstrap.csv`, `backtest_nested.csv`, `backtest_sensitivity.csv`, `backtest_inputs.json` | — | the input selection ([page 06](06-input-selection.md)) |
| `manifest.json` | — | dataset revision, SHA-256 of inputs/code/constants/decisions, every constant, check results, counts, shipped inputs, runtime |

## `annual_clean.parquet`

| Column(s) | Meaning |
|---|---|
| `city_id`, `name`, `lat`, `lon`, `year` | key and location (current station) |
| `annual_avg_temp`, `total_precip_in`, `hottest_day_temp`, `cdd` | **official labels** = `data/annual_valid.csv` (NaN when invalid) |
| `avg_tmax`, `avg_tmin`, `min_tmax`, `max_tmin`, `min_tmin`, `max_1day_p`, `max_5day_p` | published indices, NaN under each one's 10-day rule |
| `warm_days_pct_valid`, `cold_days_pct_valid`, `warm_nights_pct_valid`, `cold_nights_pct_valid` | percentage indices rescaled to valid days (features only at origins ≥ 1987) |
| `wet_days_in` | inches from very wet days, NaN when > 10 precipitation days are missing |
| `<metric>_qc`, `<index>_qc` | QC values: label + the exact effect of repairs (inputs) |
| `<metric>_qc_invalidated` | a QC step made a valid official year invalid |
| `<metric>_homog`, `_fill`, `_fill_homog`, `_nbfill`, `_fill_homog_nbfill`, `_homog_nbfill`, `_fill_nbfill` | input variants (those that exist for the metric) |
| `<metric>_input` | the input chosen by the backtest ([page 06](06-input-selection.md)) |
| `<metric>_input_variant` | its name (`qc+homog`, `official`, …) |
| `<metric>_input_source` | `measured`, `filled days`, `neighbour infill` or `official` (label used where the variant was empty) |
| `n_deleted_<var>`, `n_swapped_<var>`, `n_filled_<var>` | days deleted / swapped / filled that year |

## `coverage_clean.parquet`

| Column(s) | Meaning |
|---|---|
| `days_in_year`, `days_in_record` | calendar days; days inside the city's record |
| `miss_tmax`, `miss_tmin`, `miss_prcp`, `days_both` | raw counts (they define the official labels) |
| `miss_*_qc`, `days_both_qc` | counts after cleaning |
| `valid_<metric>` | official validity |
| `cdd_incomplete_days` | days in the record without both temperatures |
| `segment_jan1`, `station_changes` | published station context |
| `station_key_majority`, `share_last_station`, `years_since_change`, `change_year`, `cluster_id` | station context per year |
| `in_gap_days`, `empty_station_days` | days in a gap between stations / in a station period without data |
| `fragment_dropped_temperature`, `fragment_dropped_precipitation` | the year belongs to a dropped block |
| `record_break_before_temperature`, `record_break_before_precipitation` | first year after a gap of ≥ 10 empty years |
| `offset_tmax`, `offset_tmin` | mean °F added by the station adjustment |
| `transition_days_tmax`, `transition_days_tmin` | days removed from the adjusted series |

## `stations_clean.parquet`

`city_id, name, lat, lon, stn_name, stn_name_norm, station_key, segment, start, end, days, days_with_data,
empty_period, gap_before_days, overlap_days, gap_days_with_data, same_name, returns_to_key, cluster_id`, and for the
station's cluster `step_*, t_*, t_crit_*, n_before_*, n_after_*, applied_*, status_*` for Tmax and Tmin.

## `daily_clean.parquet`

| Column(s) | Meaning |
|---|---|
| `city_id`, `date` | key |
| `tmax_raw`, `tmin_raw`, `prcp_raw` | as published |
| `tmax`, `tmin`, `prcp` | QC values |
| `tmax_filled`, `tmin_filled`, `prcp_filled` | QC values with missing days filled (inputs only) |
| `tmax_offset`, `tmin_offset` | station offset; adjusted = QC + offset |
| `tmax_flag`, `tmin_flag`, `prcp_flag` | 0 observed, 1 missing, 2 deleted by QC, 3 swapped, 4 transition day |

## Loading

```python
import pandas as pd
annual = pd.read_parquet("data/processed/annual_clean.parquet")
inputs = annual.pivot(index="city_id", columns="year", values="annual_avg_temp_input")   # cities x years
```
