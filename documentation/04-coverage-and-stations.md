# 04 — `coverage_and_stations`: how each year was measured

One row per city-year (28,052), the same rows and order as `annual_indices`: `Days Both` (days with both Tmax and
Tmin, the days `cdd` uses), `Station Segment` (station in use on 1 January), `Station Changes`. It is derived from
`daily` and `city_info`, so once those are cleaned it is stale. Plan reference: §2.

## V1 Verification (hard checks)

| Check | Result |
|---|---|
| keys and order equal `annual_indices` | yes |
| `Days Both` recomputed from `daily` | 28,052 / 28,052 equal |
| `cdd` empty ⇔ `Days Both = 0` | exact (1,206 rows) |
| `Station Segment` recomputed from `city_info` | exact |
| `Station Changes` recomputed (a 1-January change counts in that year) | exact |

The table is a pure function of the other two subsets, so recomputing it after cleaning is safe.

## V2 Stale after cleaning

The raw counts (`miss_tmax`, `miss_tmin`, `miss_prcp`, `days_both`) are kept because they define the official
labels; the counts after cleaning are added (`*_qc`). QC values and the fill eligibility use the QC counts.

## V3 Validity was not stored

`valid_<metric>` holds the official rule (≤ 10 missing days of each variable the metric uses, raw counts).
`cdd_incomplete_days = days_in_record − days_both` exposes the hidden CDD case: in 18 city-years of 1989–2018 more than
10 days lack one temperature although Tmax and Tmin each pass. Those years stay valid (that is the official rule) but a
model can see them.

## V4 Station recorded on 1 January only

A change on 1 July gives the whole year to the old station in `Station Segment`. Added: `station_key_majority` (station
supplying most days), `share_last_station` (fraction of the year after its last change), `years_since_change`,
`change_year`, `cluster_id`.

## V5 Gaps, empty periods, fragments and transitions

`in_gap_days` (112 city-years touch a gap between stations), `empty_station_days` (104 city-years inside an empty
station period), `fragment_dropped_temperature` / `_precipitation`, `record_break_before_temperature` /
`_precipitation`, `transition_days_tmax` / `_tmin` (days removed from the adjusted series inside an adopted cluster).
Nothing from a dropped block can be used by accident.

## V6 Adjustments made visible

`offset_tmax`, `offset_tmin`: the mean °F the station adjustment added that year (non-zero in 2,586 city-years for
Tmax and 3,999 for Tmin). Every adjusted input can be traced to the step that changed it in `qc/homog_steps.csv`.

## Output

`data/processed/coverage_clean.parquet`: 28,052 rows, key (`city_id`, `year`), 1:1 with `annual_indices` and
`annual_clean.parquet`.
