# 06 — Which cleaned input should a model use? (the backtest)

Cleaning only earns its place if it predicts the **official** labels better. This page describes the rolling-origin
backtest that decides the default input per metric (`<metric>_input` in `annual_clean.parquet`) and its result. Plan
reference: §5 and §10.

## Design

- **Everything is rebuilt per origin from data ≤ the origin:** QC statistics and decisions, fragments, fills, station
  steps and their placebo calibration, infill offsets. Each fold is built inside a guard that rejects any year after
  its origin. The fill method was fixed once on data ≤ 1985, before every origin.
- **Origins** 1986, 1994, 2002, 2010, each tested on the next 8 years (1987–2018). Truth = `data/annual_valid.csv`;
  skill relative to the official 30-year average (the leaderboard baseline).
- **Mask:** all test city-years with a finite official label and an official 30-year baseline, defined from the
  official data only. Every candidate × forecaster must forecast every row (hard check); candidates are completed with
  official labels where they are empty, except where QC or a station transition removed data on purpose.
- **Candidates:** `official` (the labels as inputs), `qc`, `qc+homog`, `qc+fill`, `qc+fill+homog`, `qc+nbfill`,
  `qc+fill+homog+nbfill` (precipitation: no `homog`; hottest day: no `fill`).
- **Forecasters:** the four of `code/combo_model.py` (average, last10, trend, blend); `last_year` (last valid year
  repeated) is reported as a reference.
- **Rule R** on a set of origins: pick the forecaster with the best mean skill on official inputs (unless held fixed);
  a candidate is eligible if its mean gain G over `official` is ≥ 0.002 and the **spatial block bootstrap** lower
  bound of G is > 0 (5° lat/lon cells resampled, 2,000 draws, Bonferroni over the candidates); the simplest eligible
  candidate wins unless a more complex one adds ≥ 0.002 more.
- **Ship rule:** production uses R on all four origins with the team's forecaster (`METHOD`: blend for Avg Temp and
  CDD, average for precipitation and the hottest day). A non-official input ships only if a **nested chronological**
  evaluation agrees: R applied on the earlier origins only (1986, 1994 → scored on 2003–2010; 1986, 1994, 2002 → scored
  on 2011–2018) must give a positive mean outer gain, select a non-official input in at least one fold, and also give
  a positive mean outer gain with `METHOD` held fixed.

## Skill per origin with the team's forecaster

Skill vs the official 30-year average (higher is better). `official` and `qc` give the same skill here: inside the
windows the forecasters read (1957–2010), QC changes only one inverted day (Portland 1999, a swap that moves that
year's Avg Temp by about 1e-5 °F); the single deleted outlier (2018) is after every origin.

**Average temperature (blend)**

| Input | 1986 | 1994 | 2002 | 2010 | Mean |
|---|---:|---:|---:|---:|---:|
| official | 0.0341 | 0.1070 | 0.1192 | 0.1253 | 0.0964 |
| **qc+homog** | **0.0397** | **0.1131** | 0.1195 | 0.1261 | **0.0996** |
| qc+fill+homog+nbfill | 0.0373 | 0.1106 | 0.1202 | 0.1267 | 0.0987 |
| qc+fill+homog | 0.0344 | 0.1114 | 0.1209 | 0.1277 | 0.0986 |
| qc+fill | 0.0286 | 0.1050 | 0.1207 | 0.1271 | 0.0953 |
| qc+nbfill | 0.0339 | 0.1048 | 0.1211 | 0.1222 | 0.0955 |

**Cooling degree days (blend)**

| Input | 1986 | 1994 | 2002 | 2010 | Mean |
|---|---:|---:|---:|---:|---:|
| official | 0.0280 | 0.0465 | 0.0668 | 0.1700 | 0.0778 |
| qc+fill+homog+nbfill | 0.0309 | 0.0534 | 0.0695 | 0.1748 | 0.0822 |
| qc+fill+homog | 0.0280 | 0.0498 | 0.0694 | 0.1750 | 0.0806 |
| qc+homog | 0.0317 | 0.0502 | 0.0672 | 0.1706 | 0.0799 |
| qc+nbfill | 0.0306 | 0.0490 | 0.0691 | 0.1667 | 0.0788 |
| qc+fill | 0.0243 | 0.0460 | 0.0690 | 0.1746 | 0.0785 |

**Precipitation and hottest day (average):** every variant is within ±0.004 of `official` in every window and none has
a positive mean gain with a positive bootstrap bound.

## Decision

| Metric | Production rule | G over official | Bootstrap lower bound | Nested outer gain (inside / METHOD fixed) | **Shipped** |
|---|---|---:|---:|---|---|
| `annual_avg_temp` | `qc+homog` | +0.0032 | +0.0010 | +0.00039 / +0.00052 | **`qc+homog`** |
| `cdd` | `qc+fill+homog+nbfill` | +0.0043 | +0.0007 | 0 (both folds chose last10 + official) / +0.00048 | **`official`** |
| `total_precip_in` | `official` | — | — | 0 / 0 | **`official`** |
| `hottest_day_temp` | `official` | — | — | 0 / 0 | **`official`** |

- **Avg Temp:** the station adjustment helps in every origin and passes every gate. It ships.
- **CDD:** the full cleaning led clearly over the four origins and is positive with the forecaster held fixed, but
  when the forecaster is also chosen from earlier origins only, the procedure picked `last10` with official inputs in
  both folds, so it shows no out-of-sample evidence that *the selection procedure* adds value. The ship rule (fixed
  before the run) therefore falls back to `official`. The `cdd_fill_homog_nbfill` column is kept for later models.
- **Precipitation / hottest day:** nothing helps; they are mostly year-to-year noise.

## Other references (official inputs, mean skill over the four origins)

| Metric | average | last10 | trend | blend | last year repeated |
|---|---:|---:|---:|---:|---:|
| Avg Temp | 0 | 0.0863 | 0.0739 | 0.0964 | 0.0275 |
| CDD | 0 | 0.0759 | 0.0550 | 0.0778 | −0.0803 |
| Hottest day | 0 | −0.0047 | −0.0573 | −0.0218 | −0.3208 |
| Precipitation | 0 | −0.0229 | −0.0852 | −0.0461 | −0.3204 |

## What this means

The gain from cleaning is real but small for these forecasters: +0.0032 skill on Avg Temp is about +0.08 leaderboard
points. The station adjustment matters most for models that fit long trends. Every variant column is in
`annual_clean.parquet`, so later models can be evaluated on them with the same rule.

Files: `data/processed/backtest_skills.csv` (every origin × metric × candidate × forecaster),
`backtest_bootstrap.csv`, `backtest_nested.csv`, `backtest_sensitivity.csv` (3° and 8° cells, 7-year horizon),
`backtest_inputs.json` (the choice and the hashes it depends on).
