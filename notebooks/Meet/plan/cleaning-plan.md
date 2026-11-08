# Cleaning plan for the four training subsets (v4, final)

Author: Claude Code (Claude Opus 5.5) for Meet. Status: **final** — approved in round 4 by both reviewers
(Sol / gpt-6.1-sol 95/100, Grok / grok-4.7 97/100, no must-fix items). Implemented as `scripts/02_clean_data.py`
(logic in `src/protostar_challenge/cleaning.py`), writing to `data/processed/`. Responses to the reviews are in §9,
implementation decisions taken after round 4 in §10; reviews are in `reviews/` (round 1: Grok 68; round 2: Grok 89,
Sol 82; round 3: Grok 93, Sol 89; round 4: Sol 95, Grok 97).

## 0. Context and fixed rules

- Data: `CMU-OSPO/protostar-climate-train-to-2018` at the pinned revision, four subsets in `data/raw/`:
  `daily` (10,245,754 rows), `annual_indices` (28,052), `city_info` (461), `coverage_and_stations` (28,052).
  210 cities, 1851–2018.
- Goal: a processed dataset for later feature engineering and model training that forecasts `Avg Temp`,
  `ToT Prcp`, `Max Tmax`, `cdd` for 2019–2025. Score = 100·max(0, mean skill), skill = 1 − MAE/MAE(30-yr
  average); a year counts only if ≤ 10 days are missing for each variable the metric uses (`code/common.py`
  `METRICS`, `MAX_MISSING_DAYS = 10`); a city enters a window's baseline only with ≥ 10 valid years in the 30
  (`MIN_VALID_YEARS`).
- **Data rule:** nothing after 2018-12-31; `assert_no_post_cutoff()` on every weather table read and written; no
  method selection on 2019–2025.

### 0.1 Three kinds of columns (the central design rule)

The 2019–2025 answers are computed from the raw station record with the official 10-day rule: errors included,
nothing filled, nothing adjusted. So:

1. **Official labels** — `annual_avg_temp`, `total_precip_in`, `hottest_day_temp`, `cdd`: the **published**
   `annual_indices` values masked exactly as `code/prepare_data.py` does (`METRICS`, `MAX_MISSING_DAYS`,
   rounded to 3 decimals). They must equal `data/annual_valid.csv` exactly, NaN mask included; the script
   **hard-fails** otherwise. No cleaning step changes them.
   - Why published and not rebuilt from days: the published `Avg Temp` comes from a rounding chain (`Avg Tmax`
     and `Avg Tmin` rounded to 2 decimals, then their mean rounded to 2 decimals with the source's own half-way
     rule), so a direct daily mean differs by up to 0.0099 °F (22,219 valid rows differ by > 5e-4; checked).
   - **Audit (separate, every temperature-valid row; hard-fail):** (a) \|daily mean Tmax − published `Avg Tmax`\|
     ≤ 0.0051 °F, same for Tmin; (b) \|published `Avg Temp` − (published `Avg Tmax` + published `Avg Tmin`)/2\|
     ≤ 0.0051 °F; (c) \|direct daily `Avg Temp` − published\| ≤ 0.0101 °F (the two-stage rounding bound; it only
     catches a wrong definition, the source's tie-break being unknown); (d) `ToT Prcp`, `Max/Min Tmax/Tmin`,
     `cdd` within 1e-6. All four pass on the unmodified table (checked by both reviewers).
2. **QC values** — `<metric>_qc`, inputs only: **label + (f(QC days) − f(raw days))**, where f is our rebuild.
   So a QC value equals the label exactly unless a repair/deletion changed that year, and then differs by
   exactly the effect of the repair (precision differences cannot leak in). NaN when the QC days fail the 10-day
   rule or the year is a dropped fragment.
3. **Model inputs** — `<metric>_filled`, `_homog`, `_full`, `_nbfill`, …, the chosen `<metric>_input` and
   `<metric>_input_source ∈ {measured, filled days, neighbour infill, official}`. Every variant built from days
   uses the same **delta form** (label + f(variant days) − f(raw days)) where a label exists, so a zero offset or
   no filled day reproduces the QC value exactly; where no label exists (an invalid or empty year made usable by
   filling) the variant is f(variant days) directly. Every variant is then **completed with the official label**
   in any year where the variant is NaN and the label is finite (`input_source = official`), so no candidate covers
   fewer city-years than the labels — **except** years that a QC step itself made invalid (`qc_invalidated`, e.g. a
   deletion pushing a year over 10 missing days), which stay NaN so the error is not put back (a forecaster needs
   only some valid years in its window, so this cannot remove a forecast; §5 still hard-fails if it did). The default input is chosen by §5, where the **official labels themselves are
   a candidate** that cleaning must beat.

Every detected problem ends in an action: repair, delete (in QC values/inputs), recompute, adjust, fill, or a
keep decision with stated evidence. All thresholds are named constants fixed in this plan; none is tuned on a
backtest result. `MIN_GAIN` = 0.002 skill on one metric (= 0.05 overall leaderboard points with the other three
metrics unchanged) is carried over unchanged from v0 as a practical floor; the bootstrap and the nested ship rule
(§5) are the real guards.

Notation: "neighbours" = up to 5 nearest cities within 300 miles, inverse-distance-squared weights (IDW); "QC
neighbours" = the 3 nearest. Distances use the current station's lat/lon (the only location given).

### 0.2 Step identifiers

`C*` = `city_info`, `V*` = `coverage_and_stations`, `D*` = `daily`, `A*` = `annual_indices`, `H*` =
homogenisation (station adjustment), `E*` = evaluation. Each id is used for exactly one step.

### 0.3 Constants (each has one meaning)

| Constant | Value | Meaning |
|---|---:|---|
| `TMAX_LIMIT`, `TMIN_LIMIT`, `PRCP_LIMIT` | 134 °F, −70 °F, 43 in | physical limits (US records) |
| `STUCK_MIN_DAYS`, `STUCK_NEIGHBOUR_SD` | 7, 5.0 °F | stuck-temperature run length; neighbours' SD that proves a fault |
| `PRCP_RUN_MIN_DAYS`, `PRCP_RUN_MIN_IN` | 3, 0.10 in | repeated-precipitation run tested against neighbours |
| `Z_CANDIDATE`, `Z_SUPPORT`, `SPIKE_Z`, `SPIKE_LOOK_DAYS` | 5, 2.5, 3, 3 | outlier candidate, neighbour confirmation, one-day spike |
| `RESID_Z_DELETE`, `RESID_F_DELETE`, `MIN_NB_WITH_DATA` | 5, 25 °F, 2 | outlier delete rule |
| `FRAGMENT_GAP_YEARS`, `FRAGMENT_MAX_YEARS`, `BREAK_FLAG_YEARS` | 20, 10, 10 | drop a block of < 10 data years cut off by > 20 empty years; flag any gap ≥ 10 empty years |
| `FILL_MAX_MISSING`, `FILL_LOCAL_WINDOW`, `PRCP_RATIO_FLOOR` | 30, 15 days, 0.01 in | day filling (inputs only) |
| `CLUSTER_MIN_YEARS` | 5 | station changes closer than this form one change cluster |
| `HOMOG_WINDOW`, `HOMOG_MIN_VALID`, `HOMOG_MIN_NB` | 10, 5, 2 | years each side of a cluster; valid years needed each side; neighbours with support on both sides |
| `HOMOG_MIN_T`, `HOMOG_MIN_STEP`, `AR1_MAX`, `PLACEBO_Q` | 2.0, 0.25 °F, 0.8, 0.95 | step test floor, minimum step, cap on lag-1 autocorrelation, placebo quantile (per window class) |
| `RETURN_MAX_SUM` | 0.5 °F | a return pair whose two steps do not cancel to within this is not applied |
| `INFILL_OFFSET_YEARS`, `INFILL_MIN_NEIGHBOURS` | 15, 2 | neighbour infill of empty years |
| `MIN_GAIN`, `BOOT_REPS`, `BOOT_ALPHA`, `BOOT_CELL_DEG` | 0.002, 2000, 0.05, 5° | selection rule (§5) |
| `FILL_BLOCKS`, `FILL_MIN_NORMAL_YEARS` | 300, 10 | D6 bake-off blocks; years of data a day-of-year normal needs |

---

## 1. `city_info` — station history (C steps)

One row per station combined into a city's threaded record (`Name, ID, Lat, Lon, Stn.Name, Stn.stDate,
Stn.edDate`); 210 cities, 1–10 stations (mean 2.2); 251 changes, 65 on 1 Jan, 32 in 1989–2018. It drives H.

| # | Flaw (evidence) | Detection | Action | Why |
|---|---|---|---|---|
| C1 | dates are text; schema must hold | parse; `start ≤ end` (0 violations); last station ends 2018-12-31 (210/210); Lat/Lon constant per city and equal to `annual_indices`; 210 IDs ↔ 210 names; regression check of the card's two typo fixes (Asheville `ASHEVILLE CITY` start 1876-08-01; West Palm Beach `JUPITER-HYPOLUXO` end 1899-12-01) | parse to `datetime64`; **hard-fail** on any violation; `assert_no_post_cutoff` on `Stn.edDate` | later steps depend on these invariants |
| C2 | **overlap**: Abilene `ABILENE CITY` to 1944-02-29, `ABILENE REGIONAL AP` from 1943-03-01 (366 days inclusive) | `start − previous end < 1 day` | change date = new station's start (1943-03-01), as the source's `Station Segment` uses; the overlap years 1943 and 1944 are **excluded from H windows**; no data-driven date search | ambiguous attribution must not bias a step |
| C3 | **gaps** between stations: 7, empty days 1 (Tampa), 1 (West Palm Beach), 30 (Chicago), 244 (Colorado Springs), 776 (Kalamazoo), 1,096 (Rockford), 36,524 (Sioux City) | `start − previous end > 1 day` | verified **0 readings inside any gap, 0 outside every station window** → gap days are missing days; recorded as `in_gap_days` (V5) | `daily` and `city_info` agree |
| C4 | **station periods without data**: Asheville `ASHEVILLE CITY` 1876-08-01–1964-08-31, Louisville `LOUISVILLE WFO` 1995-11-01–2005-12-31, Lincoln `LINCOLN UNIV CAMPUS` 1954-07-01–1955-08-31, La Crosse `LA CROSSE` 1985-10-01–1986-01-31 (0 readings each); Huntsville `HUNTSVILLE` 1958-09-01–2018-12-31 has data only in 2018 (1.7 %) | readings per station period | an **empty period is removed from the change list** (its days are a gap); the change into the next station stays only if the station keys before and after differ (C5). La Crosse AP → (empty CITY) → AP: no change | a station that supplied no data cannot shift the record |
| C5 | **names inconsistent; stations return** (A→B→A: 30 rows in 13 cities after normalisation); **same-name consecutive rows** (Akron 1948) | normalise (upper case, strip punctuation, collapse spaces; `AIRPORT/ARPT/AP→AP`, `INTERNATIONAL→INTL`, `MUNICIPAL→MUNI`, `W B→WB`, `REGIONAL→RGNL`); `station_key` per distinct normalised name within a city | add `stn_name_norm`, `station_key`, `returns_to_key`, `same_name`. A same-name consecutive change (Akron 1948) has the same key on both sides, so H2's P = Q rule applies: no step (the metadata says it is the same station; a real move under one name would be undocumented, which this version does not detect). **Return pairs** (cluster k: A→B, a later cluster: B→A, both adopted): if \|s_AB + s_BA\| > `RETURN_MAX_SUM`, neither step is applied (one is mis-estimated) | name identity is metadata, not proof of an identical setup; the pair rule catches inconsistent steps |
| C6 | **short segments**: 17 periods < 5 years (9 < 1 year) in 9 cities | consecutive (non-empty, C4) changes < `CLUSTER_MIN_YEARS` apart | merge into **change clusters** (H2) | v0 gave these < 5 years a side and no step at all (e.g. West Palm Beach's city→airport move) |
| C7 | 65 changes on 1 Jan | — | a change on 1 Jan belongs to that year (the year is entirely the new station); with this convention our recomputation matches `Station Segment` and `Station Changes` exactly | removes an off-by-one ambiguity |

Output `data/processed/stations_clean.parquet` (461 rows): `city_id, name, lat, lon, stn_name, stn_name_norm,
station_key, segment, start, end, days, days_with_data, empty_period, gap_before_days, overlap_days,
returns_to_key, same_name, cluster_id`, plus per cluster from H: `step_tmax, step_tmin, t_tmax, t_tmin,
t_crit, n_before, n_after, applied_tmax, applied_tmin, reason`.

---

## 2. `coverage_and_stations` — how each city-year was measured (V steps)

Derived from `daily` and `city_info`; stale once they are cleaned.

| # | Flaw / check (evidence) | Action | Why |
|---|---|---|---|
| V1 | verify derivation: same keys/order as `annual_indices`; `Days Both` from raw days matches 28,052/28,052; `cdd` empty ⇔ `Days Both = 0` (1,206); `Station Segment` matches; `Station Changes` matches with C7 | **hard-fail** if any breaks | proves the table is a pure function of the other two |
| V2 | stale after cleaning | keep the raw counts (`miss_tmax`, `miss_tmin`, `miss_prcp`, `days_both`; they define the official labels) **and** add QC counts (`*_qc`) | official validity uses raw counts; QC values use QC counts |
| V3 | validity not stored; hidden CDD case (18 city-years in 1989–2018 with > 10 incomplete days although Tmax and Tmin each pass) | `valid_<metric>` (official rule, raw counts) and `cdd_incomplete_days = days_in_year − days_both` | labels and baselines use exactly the official validity; the CDD case stays valid but is exposed |
| V4 | station known on 1 Jan only | `segment_jan1` (published), `station_key_majority`, `share_last_station` (fraction of the year's days after its last change), `years_since_change`, `change_year`, `cluster_id` | a mixed year can be treated correctly |
| V5 | gaps, empty periods, fragments, holes, transitions invisible | `in_gap_days`, `empty_station_days` (C4), `fragment_dropped` (A3), `record_break_before` (A3), `transition_days` (H2) | nothing from a dropped block is used by accident |
| V6 | adjustments invisible | `offset_tmax`, `offset_tmin` (mean °F added that year by H) | ties every adjusted input to its step |

Output `data/processed/coverage_clean.parquet`: **28,052 rows, key (`city_id`, `year`), 1:1 with
`annual_indices` and with `annual_clean.parquet`.**

---

## 3. `daily` — station readings (D steps)

Wide matrices Date × city (61,361 × 210) per variable, with a record mask (first to last day of each city).
Every D action changes the **QC daily series only**; the raw series is kept. `qc_flag` distinguishes a reading
that was missing (`missing`) from one QC removed (`deleted:<rule>`) or repaired (`swapped`).

**Definitions used by D4.** Robust z: `z = (x − median) / (1.4826·MAD)` per city × calendar month × variable,
statistics from the data ≤ the forecast origin of the run (≤ 2018 in production; MAD of 0 → that city-month is
skipped). `spike`: the reading differs by ≥ `SPIKE_Z` z-units from both the nearest valid reading before and the
nearest after, each within `SPIKE_LOOK_DAYS` days. `nb_z` = the **equal-weight** mean of the QC neighbours' z that
day (neighbours without a reading are left out); `resid_z = z − nb_z`; expected value = the city's monthly median
+ nb_z × the city's 1.4826·MAD; `resid_f` = value − expected value, in °F; `n_nb` = QC neighbours with a reading
that day. One reduction (equal weights) is used for all three, as in v0.

| # | Flaw (evidence) | Detection | Action | Why / validation |
|---|---|---|---|---|
| D1 | structure | keys unique, one calendar, no date > 2018, dtypes | hard-fail; report 416,164 empty days | — |
| D2 | **inverted days**: 9 days with Tmax < Tmin, 1–11 °F apart (e.g. Montgomery 1917-09-06 71/72, Portland 1999-12-13 51/53, Williston 1946-01-31 −17/−6); 0 values beyond the limits | rules | **swap** Tmax and Tmin when both are within limits (a transposition; the day's mean and its CDD are unchanged; annual Avg Temp can move by ~1e-5 °F where the year's Tmax and Tmin missing counts differ, recorded by the QC delta); delete a value only if outside a limit; log each repair | deletion would change Montgomery 1917 CDD and Portland 1999 Avg Temp for no reason |
| D3a | **stuck temperatures**: 402 runs of ≥ 7 identical values | run lengths; neighbour SD = median of the QC neighbours' own SDs of the same variable over the run, each neighbour needing data on ≥ half the run; ≥ 2 such neighbours | if neighbour SD ≥ 5 °F: **delete days 2…n** (the first day may be true) → 5 runs, 44 readings, all 1902–1931 (Missoula 1903 tmax+tmin, Scottsbluff 1902, Astoria 1904, Morgantown 1931); keep the rest (calm or foggy spells mirrored by neighbours); fewer than 2 supported neighbours → keep, reason logged | — |
| D3b | **repeated precipitation**: 1,566 runs of ≥ 3 equal non-zero days, 4,853 days; median 0.01 in, 88 % ≤ 0.05 in, 131 runs ≥ 0.10 in (36 ≥ 0.25 in, max 0.76 in); small runs can add up (Portland 1911: 0.48 in in small runs, of a 33.28 in year) | runs; for runs ≥ `PRCP_RUN_MIN_IN`: QC neighbours' **observed** readings (missing ≠ dry); a neighbour "reports rain" if `prcp > 0` on the run day ± 1 day | runs < 0.10 in: **keep** (evidence of duplication is insufficient: repeated 0.01–0.05 in is normal in drizzle and trace climates; the per-year exposure is reported in the log). Runs ≥ 0.10 in: **delete days 2…n** when ≥ 2 QC neighbours have observed readings on every run day and **none** reports rain within ± 1 day of any run day; keep otherwise with the reason (96 of 131 have a wet neighbour, 32 lack two neighbours with data). Expected: 3 runs, 6 days (Beckley 1908-02-23, Bluefield 1919-02-10, Ely 1898-08-07) | a repeat on days that were dry all around is not weather |
| D4 | **outliers**: 4,185 readings with \|z\| ≥ 5 | confirmation: a QC neighbour with same-sign z ≥ 2.5 within ± 1 day → confirmed (89.9 %); 424 isolated | **delete** iff isolated ∧ spike ∧ \|resid_z\| ≥ 5 ∧ \|resid_f\| ≥ 25 °F ∧ n_nb ≥ 2 → 1 deletion (Greenville-Spartanburg 2018-04-28 Tmin 1.2 °F); keep the rest with a reason; 4 precipitation extremes kept (neighbours had rain). **Human overrides**: tracked file `data/qc/daily_outlier_decisions.csv` (seeded from `notebooks/Meet/qc/daily_outlier_decisions.csv`); columns add `decided_on`, `evidence_year` (the latest year of weather data the reviewer looked at; required for a non-empty `final_action`, default = the year of `decided_on`) and `note`; empty `final_action` = the automatic action; reruns keep human entries. At a backtest origin O the automatic action is recomputed from data ≤ O, and a human `final_action` is applied **only if `evidence_year` ≤ O** (so today's reviews enter no historical fold; production applies all). The same rule is used in §5. There are 0 overrides today | the °F test protects real coastal heat days (Eureka, San Diego: z 7–8 at 15–22 °F); the neighbour-data test protects pre-1900 readings (Spokane −30 °F, Jan 1888). MAD pooled across station segments: a 1–2 °F step against a monthly MAD of 5–8 °F changes z by < 0.3 (documented, not split) |
| D5 | **2018 precision**: 47.5 % of 2018 Tmax and 57.4 % of Tmin have one decimal (converted GHCN-D); all other years whole °F | share of non-integers by year; round trip `round(round((F−32)/1.8, 1)·1.8+32, 1)` of the nearest whole °F | **keep as published**: 87.4 % (Tmax) / 58.3 % (Tmin) of the decimals are exact round trips; mean(decimal − nearest whole °F) = +0.022 / +0.074 °F; the rest lie up to 0.3 °F from a whole °F. The 2019–2025 answers come from the same converted source, so rounding 2018 would move the training data away from the scored data. The calculation is in the script's checks | a documented no-op |
| D6 | **missing days** (inputs only) | — | `*_filled` = the city's **normal** (mean of the 30 years ending at the end year E, by day of the 365-day index — 29 Feb shares 28 Feb's slot, later days keep their 365-day position — smoothed with a 31-day circular window; a slot needs ≥ `FILL_MIN_NORMAL_YEARS` years of data, else the day is not filled) **plus an anomaly chosen by the bake-off below**. Precipitation works on ratios to the normal (normal floored at `PRCP_RATIO_FLOOR`). Only years with ≤ 30 missing days (counted after H3's transition masking; any residual missing days still face the 10-day rule). **Bake-off, fixed once on data ≤ 1985** (before every origin of §5): normals 1956–1985; candidate blocks = every complete 30-day block in 1956–1985 with ≥ 25 observed days in each variable, sorted by city id and block start; `FILL_BLOCKS` = 300 taken as every k-th, k = floor(N/300), phase 0; all chosen blocks are hidden **at once** before normals, local and neighbour anomalies are computed. Candidates: normal / normal + local anomaly (mean anomaly of the observed days within ± 15 days, hidden days excluded) / normal + neighbour anomaly (IDW mean of the neighbours' anomalies that day). Errors use the block's observed days only. **One scalar per branch**: temperature (Tmax and Tmin share the method) = lowest mean absolute error of the block-mean (Tmax+Tmin)/2; ties (< 0.01 °F) → the simpler method; block CDD error is reported, not voting. Precipitation = lowest mean absolute relative error of the block total. Fallbacks when applying: no neighbour with data that day → local anomaly; no local data → normal alone. **Observed values are never changed by filling** (assert). **`Max Tmax` is never taken from a filled day** | fills feed only inputs |
| D7 | **station offsets on days** | from H | `tmax_offset`, `tmin_offset` per day; `tmax_adj = tmax_qc + offset` | old readings on the reference station's level |

Outputs `data/processed/daily_clean.parquet` (`city_id, date, tmax_raw, tmin_raw, prcp_raw, tmax, tmin, prcp`
(QC), `tmax_filled, tmin_filled, prcp_filled, tmax_offset, tmin_offset, qc_flag_*`; float32),
`data/processed/qc/daily_qc_log.csv` (every repaired or deleted reading: city, date, var, old, new, rule),
`data/processed/qc/outlier_plots/`.

---

## 4. `annual_indices` (A steps) and homogenisation (H steps)

| # | Flaw (evidence) | Action | Why |
|---|---|---|---|
| A1 | **sentinels**: ±inf in empty years; `ToT Prcp = 0` on 935 rows: 932 with every day missing ("fake zeros"), 3 with 337/360/364 missing days | ±inf → NaN; every column NaN when its year misses > 10 days of what it uses (official rule): removes all 935 | a 0-inch year is the most damaging precipitation error |
| A2 | **official labels and audit** | labels = published values masked as `prepare_data.py` (§0.1), = `annual_valid.csv` exactly; audit rebuild from raw days within the stated tolerances | labels stay official; the audit proves the definitions |
| A2q | **QC values** | `<col>_qc` = label + (f(QC days) − f(raw days)), with QC missing counts and the 10-day rule. Expected changes: Greenville-Spartanburg 2018 Avg Temp ≈ +0.07 °F (Miss Tmin 4 → 5, still valid); Astoria 1904 (Miss Tmin 2 → 8, still valid, mean changes); Missoula 1903, Scottsbluff 1902 and Morgantown 1931 were already invalid; D3b's 3 runs move precipitation totals. Exact counts reported | corrected numbers as inputs, exact elsewhere |
| A3 | **detached fragments**, found **separately for temperature** (a year is occupied if it has any Tmax or Tmin) **and for precipitation** (any prcp), so precipitation cannot bridge a temperature gap (e.g. Athens 1858 and Manchester 1886–1893 are short temperature blocks cut off by > 20 years without temperature; the script lists every block per group). Blocks with data separated by > 20 empty years (empty years = next first year − previous last year − 1), any-variable view: Akron 1888–1952 \| 1999– (46); Asheville 1870–76 \| 1964– (87); Birmingham 1896–1903 \| 1930– (26); Ely 1893–1912 \| 1938– (25); Huntsville 1895–1958 \| 2018 (59); Manchester 1886–1972 \| 1996– (23); Natchitoches 1894–96 \| 1922– (25); Pueblo 1889–92 \| 1940– (47); Sioux City 1890–95 \| 1996– (100); Vancouver 1856 \| 1891– (34); Wheeling 1879–1953 \| 1998– (44). **Holes below the bar** (≥ 10, ≤ 20 empty years) inside scored records: Trenton 1982–1997, Louisville 1996–2005, Columbia MO 1949–1968, and others the script lists | **drop** (QC values and inputs only, for the variable group concerned) a block that has < 10 years with data **and is not the most recent block of that group**: Asheville 1870–76, Birmingham 1896–1903, Natchitoches 1894–96, Pueblo 1889–92, Sioux City 1890–95, Vancouver 1856 (`fragment_dropped`; never infilled). **Keep** every other block; `record_break_before = True` on the first year after **every gap ≥ 10 empty years** (e.g. Akron 1999, Ely 1938, Huntsville 2018, Manchester 1996, Wheeling 1998, Trenton 1998, Louisville 2006, Columbia MO 1969). Huntsville keeps 1895–1958 and its 2018 stub; it has 1 valid year in 1989–2018, so it is not scored, but the submission still needs a value and the long block is its only climatology. Holes are left empty in QC values and filled only in the `nbfill` inputs | v0 dropped everything before the last gap (64 years of Huntsville, 65 of Akron, 75 of Wheeling); only short, very old blocks distort all-history statistics. Scored cities after a break are scored on the post-break record, which is unaffected |
| A4 | **percentage indices count missing days as "not warm"** (Warm Days averages 8.5 % with 0 missing days, 7.5 % with 31–100, 0.9 % with > 100) | published-reference features (raw days; QC deletions are not removed from them): exact rescale of the published values: `pct_valid = pct × days_in_year / (days_in_year − miss_var)`; NaN if > 10 days missing. **Use as forecast features only at origins ≥ 1987**: their percentile thresholds use the start of record to 1987 (marked in the data dictionary; not used by the §5 backtest) | only valid days can be warm, so the count is recovered exactly (published to 2 decimals: < 0.02 days error) |
| A5 | `Wet Days` (inches from days above the 95th percentile) reads low with missing days; same 1987 thresholds | NaN when `miss_prcp > 10`; same origin note as A4 | same rule as the precipitation label |
| A6 | **empty years** (inputs only) | neighbour infill: value = neighbour's value that year + mean(city − neighbour) over the 15 nearest common valid years **≤ end year**; IDW average; ≥ 2 neighbours; from the city's first *kept* valid year (never in a dropped fragment); precipitation and CDD infill clipped at ≥ 0. Evidence (fixed seed): the error quoted is the **past-only** form (offsets and the comparison baseline both from years < the hidden year; baseline = mean of the previous 15 valid years). The ± 15-year form is reported as a secondary comparison labelled as using later years | complete series for models that need them; flagged `neighbour infill` |

### H. Station adjustment (inputs only; temperature)

**H1. Difference series.** For city c, variable v ∈ {Avg Tmax, Avg Tmin} (QC values, valid years, data ≤ end year
E), and a cluster's window: the reference uses a **fixed neighbour set** — the neighbours with ≥ `HOMOG_MIN_VALID`
valid years on *both* sides of the window (≥ `HOMOG_MIN_NB` = 2 required, else the cluster is "unestimable").
The realised reference is also fixed: only window years in which the city **and every neighbour of the set** have a
valid value are used, the IDW weights are fixed (not renormalised), and each neighbour's departure is taken from its
mean over those same years. `ref_y` = weighted mean of the departures; `diff_y = x_c,y − ref_y`. So the reference
cannot shift because a neighbour enters or leaves. If dropping years leaves < `HOMOG_MIN_VALID` on a side, the
neighbour with the fewest common years is removed from the set and the step retried (while ≥ 2 remain). Membership
and weights are logged. Shared weather cancels; a level shift in `diff` is the station.

**H2. Clusters, test and offsets.** After C4 the changes ≤ E are sorted; a **cluster** is a maximal run of changes
each < `CLUSTER_MIN_YEARS` after the previous one. For cluster k: first change f_k, last change l_k,
P_k = station key in use before f_k, Q_k = station key from l_k on.

- **Same endpoints (P_k = Q_k)**: not a station change. No step, no offset, no day removed (Lansing: Capital City
  AP → East Lansing 1954-08-01–1959-04-30 → Capital City AP; Huntsville: Huntsville → Madison
  1954-07-15–1958-08-31 → Huntsville; La Crosse after C4).
- **Windows**: before = up to 10 valid years ending in the year before f_k, not crossing the previous cluster's
  last change (a 1-Jan change year belongs to the new station) and excluding C2 overlap years; after = up to 10
  valid years from the year after l_k (from l_k's year if l_k is 1 Jan), not crossing the next cluster's first
  change or E. ≥ `HOMOG_MIN_VALID` valid years required each side, else "unestimable".
- **Step and test**: OLS on the window years `diff_y = a + b·(y − ȳ) + s·after_y`; the **adopted step is s**
  (the trend-adjusted coefficient, so a relative trend between city and neighbours is not written into the
  offset). SE(s) from OLS, inflated for autocorrelation: `SE_adj = SE·sqrt((1 + r)/(1 − r))`, r = lag-1
  autocorrelation of the city-variable's within-segment residuals (`diff` minus its segment mean, pairs of
  **consecutive calendar years within one segment** only, all years ≤ E), clipped to [0, `AR1_MAX`].
  t = s / SE_adj. **Critical value calibrated on placebos (data ≤ E)**: placebo changes on 1 July of every year
  that is ≥ 10 years from any real change of the city; like a mid-year real change, the placebo year itself is
  left out (before = years < Y, after = years > Y); windows are built with **exactly the real rule** (up to 10 valid
  years a side, not crossing real changes, ≥ 5 needed). Placebos are pooled over all cities per variable and split
  into two window classes: *short* (fewer than 10 valid years on either side) and *full*. For each class t_crit =
  max(`HOMOG_MIN_T`, the `PLACEBO_Q` quantile of \|t\|); a real cluster uses the t_crit of its own class (if a class
  has < 200 placebos, the short class uses the larger of the two). **Adopt** iff \|t\| ≥ t_crit and
  \|s\| ≥ `HOMOG_MIN_STEP`. Two passes (the second with neighbours adjusted by the first, recalibrated). The
  placebo adoption rates per class and pass are reported; they describe placebos, not a guaranteed error rate for
  real changes. Then the C5 return-pair rule.
- **Offsets** (reference = the station in service on E-12-31, offset 0; old readings move onto its level). Let
  clusters be indexed 1…K in time and A_k = s_k if adopted else 0. For a day t, the offset is
  `off(t) = Σ_{j : f_j > t} A_j` (all later clusters) **plus**, if t lies inside cluster k (f_k ≤ t < l_k):
  A_k if the day's station = P_k; 0 if it is Q_k; if it is a third station and A_k ≠ 0 → the day is a
  **transition day**, removed from the adjusted series; if A_k = 0 → kept with offset Σ_{j>k} A_j. Every day
  therefore belongs to exactly one interval and gets one offset (38 cities have ≥ 2 clusters).
  - *West Palm Beach* (JUPITER-HYPOLUXO 1889–1899-12-01, HYPOLUXO 1899-12-02–1929, CITY from 1930-01-01, then
    CITY/AP alternating 1938-07-01…1941-02-01, AP after): clusters {1899-12-02}, {1930-01-01} and
    {1938-07-01 … 1941-02-01} (P = CITY, Q = AP). The third's before-window = 1930–1937 (the 1 Jan 1930 change
    year belongs to CITY), after = 1942–1951; CITY days 1938–41 get A_3, AP days 0; days 1930-01-01–1938-06-30
    get A_3; days before 1930 get A_2 + A_3; days before 1899-12-02 get A_1 + A_2 + A_3.
  - *Third stations removed only when their cluster's step is adopted*: Gainesville `3 WSW`
    1957-06-01–1960-05-16 (cluster {1957-06-01, 1960-05-17}: P = UNIV OF, Q = REGIONAL AP), San Antonio
    `STINSON` 1941-01-01–1942-06-30 (P = WBO, Q = INTL AP), Louisville `BOWMAN FLD` 1945-07-01–1947-12-31
    (P = WBO, Q = STANDIFORD FIELD). The 10-day rule then decides whether that year's adjusted value exists.
    (Louisville's later WFO period is empty, C4, so Standiford → Standiford in 1995–2006 is no change.)
- Every annual input (`Avg Temp`, `Max Tmax`, `cdd`) is rebuilt from the **adjusted days**.
- *Limitation*: changes shared by all neighbours at once cancel in `diff` and cannot be seen by a pairwise method;
  undocumented-break detection is not part of this version.

**H3. Order when filling and adjusting are combined (`qc+fill+homog`).** (1) QC days; (2) H steps estimated from
QC annual values ≤ E; (3) offsets added to observed days → adjusted observed days on the reference level
(transition days removed); (4) normals and neighbour anomalies computed from **adjusted** days, missing days
(including removed transition days) filled on the reference level — the 30-missing-day eligibility is counted
after transition masking, and residual missing days face the 10-day rule; (5) the plain `qc+fill` series is built
separately from unadjusted QC days with the same method (no offsets involved). Observed values are unchanged by
(4) (assert). Annual values of every variant use the delta form of §0.1.

**H4. Coverage of late changes by the backtest.** At each origin H uses changes ≤ origin with the same rules.
Changes with 5–9 valid after-years exist at every origin (5, 3, 9 and 3 at 1986, 1994, 2002, 2010), so the
endpoint behaviour is exercised. Riverside 2009-01-01 has 2 valid after-years at origin 2010 and 10 in the 2018
run; Natchitoches 2010-08-01 has 0 at every backtest origin and 8 in the 2018 run. Both are marked "estimable only
in the 2018 run" in the step file (whether they are adopted is decided by the test).

**Precipitation adjustment is deferred**: annual totals vary by ~20 % year to year, so ratio steps under ~10 %
are not detectable in 10-year windows; a ratio variant needs its own specification and validation and is not in
this version.

Output `data/processed/annual_clean.parquet`: **28,052 rows, key (`city_id`, `year`), 1:1 with
`annual_indices`**: official labels (float64), QC values, the other indices (rescaled percentages), variant
inputs, `<metric>_input` and `_input_source`, counts of repaired/deleted/filled days, and the coverage columns.

---

## 5. Selecting inputs (E)

- **Everything is rebuilt per origin from data ≤ origin**: D2–D4 statistics and decisions, fragments, fills
  (method fixed on ≤ 1985), H steps and placebo t_crit, infill offsets. Only the constants of §0.3 and human
  decisions with `evidence_year` ≤ the origin (D4) carry across.
- **Origins** 1986, 1994, 2002, 2010, each tested on the next 8 years (1987–2018 overall). All four
  `FORECASTERS` are run for every candidate; a 7-year-horizon version is reported as a sensitivity (not used to
  choose).
- **Truth** = official `data/annual_valid.csv`; skill relative to the official 30-year average (the leaderboard
  baseline).
- **Candidates per metric**: `official` (the labels as inputs), `qc`, `qc+homog`, `qc+fill`, `qc+fill+homog`,
  `qc+nbfill`, `qc+fill+homog+nbfill` (hottest day: no `fill` variants). Each is completed with official labels
  where it is NaN (§0.1), identically in the backtest and in production.
- **Mask (full, fixed)**: per metric and origin, the scored rows are all test city-years with a finite official
  label and a finite official 30-year baseline — defined from the official data only, independent of any candidate.
  Every candidate × forecaster must give a finite forecast on **every** row of that mask; a non-finite forecast is a
  hard failure (none is expected, because candidates are completed with official labels). Mask sizes reported.
- **Decision statistic**, for forecaster m and candidate v: G_m(v) = mean over the origins used of
  [skill_m(v) − skill_m(official)] on the mask (equal origin weights).
- **Spatial block bootstrap of G** (the same statistic): cells = (floor(lat/5), floor(lon/5)) of the current
  station (42 occupied cells; 8 singletons); each draw resamples cells with replacement (2,000 draws, seed 0; a cell
  drawn twice counts its cities twice), keeps each sampled city's rows at every origin, recomputes each origin's
  candidate MAE *and* baseline MAE on the resampled rows, then G. Lower bound = the α/k quantile, α = 0.05,
  k = number of non-official candidates for the metric (Bonferroni). 3° and 8° cells reported as a sensitivity.
- **Rule R (applied to a set of origins S)**, per metric: (1) forecaster m* = the forecaster with the highest mean
  skill on `official` inputs over S (ties → the simpler: average < last10 < trend < blend); (2) a candidate is
  eligible iff G_{m*}(v) ≥ `MIN_GAIN` and its bootstrap lower bound > 0; preference order (simplest first):
  `official` < `qc` < `qc+homog` < `qc+fill` < `qc+fill+homog` < `qc+nbfill` < `qc+fill+homog+nbfill`; the choice is
  the simplest eligible candidate, replaced by a more complex eligible one only if its G is ≥ `MIN_GAIN` higher; no
  eligible candidate → `official`.
- **Nested, chronological evaluation of R**: outer origin 2002 — R on S = {1986, 1994} (test years end 2002) picks a
  forecaster *and* an input; both are frozen and scored on 2003–2010 against `official` under the same forecaster.
  Outer origin 2010 — R on S = {1986, 1994, 2002}, scored on 2011–2018. Outer gain = mean of the two folds' skill
  differences (equal weights).
- **Ship rule (fixed now)**: the production input for a metric = R on all four origins **conditional on the team's
  forecaster `combo_model.METHOD`** (the model the inputs will feed; METHOD was chosen by the team on these windows,
  disclosed). A non-official production input ships only if (a) the nested outer gain is > 0 **and** (b) at least
  one outer fold actually selected and scored a non-official input. Otherwise the metric ships `official`. Because
  the record and the v0 results have been inspected, this is not a pristine hold-out; it is the best available check
  that the procedure generalises. Thresholds are not changed in response to the outer results.
- Reported for every metric: gain over `official` and over `qc`, bootstrap bounds, mask sizes, per-origin and
  per-forecaster skills, the forecaster and input R picks in each fold, nested outer results.

## 6. Checks run by the script (hard-fail unless noted)

1. `assert_no_post_cutoff` on every table read (daily, annual, city_info end dates, coverage) and written.
2. C1 and V1 invariants; recomputed `Station Segment`/`Station Changes`.
3. Labels = `data/annual_valid.csv` exactly (NaN mask included); audit checks (a)–(d) of §0.1.
4. Row contracts: stations 461; coverage and annual 28,052 each, same keys/order as `annual_indices`; daily rows =
   record-mask rows.
5. Every QC value ≠ its label is explained by a logged repair, deletion or fragment drop (the delta form makes
   precision differences impossible).
6. Filling never changes an observed value; `Max Tmax` inputs never use a filled day; infilled precipitation and
   CDD ≥ 0.
7. Reported (not fail): D5 shares, D6 bake-off, infill errors (both forms, past-only first), adopted and
   unestimable steps, placebo rates and t_crit, return-pair sums, H4 list, §5 tables.
8. `manifest.json`: SHA-256 of the input files, of `src/protostar_challenge/cleaning.py`,
   `scripts/02_clean_data.py`, `src/protostar_challenge/config.py`, `code/common.py` and `code/combo_model.py`
   (forecasters, METHOD, skill, validity rule), of the constants and of the decisions file; dataset revision; row
   counts; counts per step; chosen inputs; runtime.
9. `--skip-backtest` reuses `backtest_inputs.csv` only if all of those stored hashes match the current ones;
   otherwise the script fails and asks for a full run.

## 7. Implementation and tests

- `src/protostar_challenge/cleaning.py` (constants at top, one function per step id), `scripts/02_clean_data.py`
  (`uv run python scripts/02_clean_data.py [--skip-backtest]`), reading `data/raw/` (`01_download_data.py`
  first).
- Outputs in `data/processed/` (git-ignored): `daily_clean.parquet`, `annual_clean.parquet`,
  `stations_clean.parquet`, `coverage_clean.parquet`, `qc/daily_qc_log.csv`, `qc/homog_steps.csv`,
  `qc/outlier_plots/`, `backtest_inputs.csv`, `backtest_bootstrap.csv`, `backtest_nested.csv`, `manifest.json`.
- Deterministic: no unseeded randomness; sums in fixed order.
- `tests/test_cleaning.py` (synthetic data, runs in seconds): swap of an inverted day; stuck run with and
  without neighbour support; zero MAD; missing neighbours; precipitation run with missing vs dry neighbours;
  first/last partial years and leap day; 1-Jan change; overlap years excluded; empty station period removed;
  P = Q cluster; third-station transition only when adopted; offsets on a 3-cluster city (one offset per day);
  injected step recovered with 5 and with 10 years a side, and no step on a trend-only series or with an uneven
  neighbour series (fixed reference); fill leaves observed values unchanged; validity masks; QC delta = 0 when
  nothing changed; completion with official labels; evidence-year filter of overrides.
- `documentation/` (after review): one page per subset (flaw → evidence → action), H, E, and an index linked from
  the root README.

## 8. Known limitations (stated, not hidden)

- Thresholds were set from physical reasoning and from inspecting cases across the whole record (e.g. coastal heat
  days, a 2016 Lander reading); they are frozen here. Labels are untouched, and `official` can win §5, so a bad QC
  gate can only cost the inputs it is used for. Its only test-window deletion (Greenville-Spartanburg 2018-04-28)
  is after every origin and is not in any backtest score.
- Repairs outside every 30-year window (stuck runs 1902–1931, D3b runs 1898–1919, fragments ending by 1903) are
  QC decisions supported by their own evidence, not by the backtest.
- Station locations before the current station are unknown; distances use the current site.
- Network-wide inhomogeneities are not removable pairwise (H2).
- Day-filled totals (CDD, precipitation) count days the official label skips; only §5 decides whether they are
  used.
- §5 proves value for the team's current forecasters and the official target; later models should reuse the same
  selection discipline (all variant columns are kept).

## 9. Responses to the reviews

**Round 1 (Grok 68)** — official labels, swap, fragment spans, a selection rule that can reject, ids/constants/rows,
step test, infill evidence: all addressed in v2.

**Round 2 (Grok 89, Sol 82)**

| Item | Change in v3 |
|---|---|
| Sol 1: Avg Temp precision | labels = published values masked like `prepare_data.py` (exact); audit rebuild with stated tolerances; QC values in delta form (§0.1) |
| Sol 2: overlapping confirmation, early calibration | rotated origins removed; nested chronological outer evaluation with a ship guard (§5); fill method fixed on ≤ 1985; A4/A5 percentile features restricted to origins ≥ 1987 |
| Sol 3: official must be able to win | `official` is a candidate and the default; common mask with a no-silent-loss check (§5) |
| Sol 4: bootstrap the actual statistic | spatial 5° cell bootstrap of G with per-origin baselines recomputed; sensitivity 3°/8° (§5) |
| Sol 5: order, precipitation ratio, reuse contract | H3 order; precipitation adjustment deferred with a reason; hash-bound `--skip-backtest`; decision-file metadata and per-origin filtering (D4, §6) |
| Grok 1: trend-adjusted step | the adopted step is the OLS coefficient on `after`; AR(1)-adjusted SE on that coefficient; placebo-calibrated t_crit (H2) |
| Grok 2: cluster rules | P = Q clusters are not changes; third stations removed only if the step is adopted; one bracketing offset per day (formula); West Palm Beach window corrected; precipitation variant removed (H2, H4) |
| Sol/Grok nice-to-haves taken | fixed neighbour set with support on both sides; lag-1 from consecutive in-segment years; return-pair rule; D3a/D3b neighbour coverage and timing rules; D4 formulas; D6 bake-off spec; holes flagged; no infill in dropped fragments; infill clipped ≥ 0; corrected D5 means and gap lengths; H4 rewritten; acceptance tests |

**Round 3 (Grok 93, Sol 89)**

| Item | Change in v4 |
|---|---|
| Sol 1 / Grok: audit tolerance fails | §0.1 audit split into (a) component means ≤ 0.0051, (b) published chain ≤ 0.0051, (c) direct daily Avg Temp ≤ 0.0101 (two-stage bound), (d) others 1e-6; delta form extended to every variant |
| Sol 2 / Grok: nest the forecaster; ship rule | §5 rule R picks the forecaster *and* the input on the inner origins; ship requires a positive mean outer gain and at least one fold that scored a non-official input |
| Sol 3 / Grok: full official mask | mask defined from official data only; candidates completed with official labels (same in production); any non-finite forecast is a hard failure; `MASK_MAX_LOSS` removed |
| Sol 4 / Grok: one D6 objective | normal = 30 years ending at E, 365-day index; k = floor(N/300), phase 0; hidden blocks masked before normals/anomalies; one scalar per branch (temperature: block-mean (Tmax+Tmin)/2 MAE; precipitation: relative total error); fallbacks; eligibility counted after transition masking |
| Sol 5 / Grok: hashes and overrides | hashes include `common.py`, `combo_model.py`, the script and config; overrides need `evidence_year` and enter a fold only if `evidence_year` ≤ origin (D4 and §5 consistent) |
| Nice-to-haves taken | fixed realised H1 reference (common years, fixed weights); placebos built with the real window rule, split short/full, per pass, no "≤ 5 % by construction" claim; fragments per variable group (Athens 1858, Manchester 1886–93 temperature); Akron 1948 = P = Q; H4 "estimable only in 2018"; A2q corrected (Astoria 1904); one equal-weight neighbour reduction in D4; bootstrap cells pinned; `MIN_GAIN` units corrected; D2 annual note; D4 "≤ forecast origin"; A4 labelled published-reference features; new acceptance tests |

## 10. Implementation decisions after round 4 (from the round-4 nice-to-haves; fixed before the final run)

| Item | Decision |
|---|---|
| D6 precipitation loss (Sol 1, Grok) | the precipitation bake-off uses the **mean absolute error of the block total in inches** (finite for dry blocks); all methods are compared on the same blocks (those where every method produced an estimate); blocks are non-overlapping 30-day blocks from 1 Jan 1956, the first 300 of the every-k sequence |
| Ship rule (Sol 2, Grok) | a non-official input ships only if, in addition, the same nested evaluation **with the production forecaster `METHOD` held fixed** has a mean outer gain > 0 (`backtest_nested.csv`, kind "conditional on METHOD") |
| Last-year-repeated reference (Sol 3) | reported for every candidate on the same official mask (`last_year`; fallback: last valid year of the 30-year window, else the 30-year average); not a selectable forecaster |
| A6 support (Sol 4) | unchanged from v0: a neighbour needs ≥ 5 common valid years; up to 15 nearest common years, ties to the earlier year (stable sort); donors and offsets come from the pre-infill table; ≥ 2 neighbours or the cell stays empty |
| H2 synthetic checks (Sol 5) | `tests/test_cleaning.py`: injected step recovered with 5+5, 5+10, 10+10 years; no bias on trend-only and uneven-neighbour series; false adoption ≤ 15 % at \|t\| ≥ 2 on 40 null cases |
| Filled pairs (Sol 6) | a filled Tmax below an observed Tmin (or the reverse) is clipped to the observed value; both filled → both to their midpoint; observed values never change (asserted) |
| Plain filled series (H3 step 5) | filled days of the plain `qc+fill` series are the reference-level estimates **minus that day's station offset** (so they sit on the recorded station's level, as Sol suggested in round 2), rather than a second fill on unadjusted days, whose normals would be on the current station's level |
| Precipitation extremes (D4) | reported in `qc/precip_extremes.csv` (wet days above 3× the city's 99.9th percentile, with the largest neighbour rain within ± 1 day); all kept |
| Completion and transitions (Grok r4) | label completion (§0.1) also skips years with station-transition days in the station-adjusted variants (e.g. Gainesville 1957–60, Louisville 1945–47 when adopted), so unadjusted values are not pasted into an adjusted series |
