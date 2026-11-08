# 01 — `daily`: station readings

One row per city and day, 1851-01-01 to 2018-12-31: `tmax`, `tmin` (°F), `prcp` (inches). Handled as Date × city
matrices (61,361 × 210) with a record mask (each city's first to last day). Every action below changes the
**QC series only**; the raw series is kept, because the official labels are computed from it. Plan reference: §3.

"QC neighbours" are the 3 nearest cities; "neighbours" are up to 5 within 300 miles, inverse-distance-squared weights.

## D1 Structure

| Check | Result | Action |
|---|---|---|
| duplicate (city, date) keys | 0 | hard-fail if any |
| calendar gaps inside a record | 0 | hard-fail if any |
| dates after 2018-12-31 | none (`assert_no_post_cutoff`) | hard-fail |
| rows with no value at all | 416,164 | treated as missing days |

## D2 Inverted days (Tmax < Tmin)

- **Found:** 9 days, 1–11 °F apart, every value inside the physical limits (134 °F, −70 °F, 43 in/day; 0 values
  outside them). Examples: Montgomery 1917-09-06 71/72, Portland 1999-12-13 51/53, Williston 1946-01-31 −17/−6.
- **Strategy:** **swap** the two values. A transposition is the simplest explanation; swapping keeps the day's mean
  (so Avg Temp and CDD are unchanged) and puts the warm value in Tmax. Deleting both, as in v0, would have moved
  Montgomery 1917 CDD (1982.5 → 1976.0) and Portland 1999 Avg Temp for no reason. A value outside a limit would be
  deleted (none exist).
- **Result:** 18 values swapped (`qc_flag = 3`), logged in `qc/daily_qc_log.csv`.

## D3a Stuck temperature sensors

- **Found:** 402 runs of ≥ 7 identical Tmax or Tmin values.
- **Detection:** a sensor that stops updating repeats a value while the weather nearby keeps changing; a real calm
  or foggy spell repeats a value while the neighbours are steady too. Test: the median of the QC neighbours' own
  standard deviations of the same variable over the run, each neighbour needing data on ≥ half the run, ≥ 2 such
  neighbours.
- **Strategy:** if the neighbours varied (SD ≥ 5 °F) **delete days 2…n** of the run (the first day may be the true
  value); otherwise keep, with the reason.
- **Result:** 5 runs deleted, 44 readings: Missoula 1903 Tmax (13 days at 55 °F, neighbours SD 12.5 °F) and Tmin
  (14 days at 15 °F), Scottsbluff 1902 Tmin, Astoria 1904 Tmin, Morgantown 1931 Tmin. 375 runs kept (neighbours steady:
  real weather), 22 kept because fewer than 2 neighbours had data. `qc/stuck_runs.csv`.

## D3b Repeated precipitation

- **Found:** 1,566 runs of ≥ 3 equal non-zero days (4,853 days). Median 0.01 in; 88 % ≤ 0.05 in; 131 runs ≥ 0.10 in
  (max 0.76 in, many at Concord 1868–1889). Small runs can add up (Portland 1911: 0.48 in of a 33.28 in year).
- **Strategy:** runs < 0.10 in are **kept** — repeated 0.01–0.05 in is normal in drizzle and trace climates and there
  is no evidence of duplication; the exposure is listed. Runs ≥ 0.10 in are tested against the QC neighbours'
  **observed** readings (missing is not dry): delete days 2…n only if ≥ 2 neighbours were observed on every run day
  and none reported rain within ± 1 day of any run day.
- **Result:** 1 run deleted (Ely 1898-08-07…09, 0.10 in a day while all three neighbours were observed dry; 2 days).
  1,435 small runs kept, 123 kept because a neighbour had rain, 7 kept for lack of observed neighbours.
  `qc/repeated_precip_runs.csv`.

## D4 Outliers

- **Detection.** Robust z = (x − median) / (1.4826·MAD) per city × calendar month × variable, from data up to the
  run's end year. A reading with \|z\| ≥ 5 is a candidate (4,185). It is **confirmed** if a QC neighbour had a
  same-sign z ≥ 2.5 within ± 1 day (89.9 %: real cold waves and heat waves). For the 424 **isolated** ones the evidence
  is: `spike` (≥ 3 z-units from both the nearest reading before and after, within 3 days), `nb_z` (equal-weight mean of
  the neighbours' z that day), `resid_z = z − nb_z`, `resid_f` (value − what the neighbours imply, °F), `n_nb`.
- **Strategy:** **delete** only if isolated ∧ one-day spike ∧ \|resid_z\| ≥ 5 ∧ \|resid_f\| ≥ 25 °F ∧ ≥ 2 neighbours
  had a reading. Two lessons from the data shaped the rule: in steady coastal climates (Eureka, San Diego) a real
  offshore-wind heat day reaches z = 7–8 but only 15–22 °F above what neighbours imply, so z alone is not enough; and
  before 1900 many readings look isolated only because the neighbours had no data (Spokane −30 °F, January 1888),
  which is no evidence against them.
- **Result:** 1 deletion — Greenville-Spartanburg 2018-04-28 Tmin 1.2 °F (spike, 44.7 °F below what neighbours
  imply). Kept: 235 that last more than a day, 101 with no neighbour data, 87 spikes within the limits. Four extreme
  rain days kept (neighbours had rain). Human review: [`data/qc/daily_outlier_decisions.csv`](../data/qc/daily_outlier_decisions.csv)
  (set `final_action`, `decided_on`, `evidence_year`; empty = automatic action) and 9 gallery pages in
  `data/processed/qc/outlier_plots/`. Two kept readings worth a look: Ukiah 1995-06-06 Tmin 21 °F, Escondido
  2006-08-30 Tmin 36 °F (each a two-day dip, so not a one-day spike).
- **Leakage control:** in a backtest fold the statistics and decisions are recomputed from data ≤ the fold's origin;
  a human override applies only if its `evidence_year` ≤ the origin (there are no overrides today).

## D5 2018 precision

- **Found:** 47.5 % of 2018 Tmax and 57.4 % of Tmin values have one decimal (converted GHCN-D), every other year is in
  whole °F. 87.4 % (Tmax) and 58.3 % (Tmin) of the decimals are exact °F → 0.1 °C → °F round trips of the nearest whole
  °F; the mean difference from the nearest whole °F is +0.022 / +0.074 °F (max 0.3 °F).
- **Strategy:** **keep as published.** The 2019–2025 answers come from the same converted source, so rounding 2018
  would move the training data away from the scored data. Calculation in `manifest.json` → `checks`.

## D6 Missing days (inputs only)

- **Method:** each missing day inside a record = the city's normal for that day (mean of the 30 years ending at the
  end year, on a 365-day index where 29 Feb shares 28 Feb's slot, 31-day circular smoothing, ≥ 10 years of data per
  slot) plus an anomaly. Precipitation uses ratios to the normal. Only years with ≤ 30 missing days are filled.
- **Choice of anomaly, made once on data ≤ 1985** (before every backtest origin): 300 non-overlapping 30-day blocks
  in 1956–1985 with ≥ 25 observed days were hidden at once and refilled.

  | Method | Block mean temperature error (°F) | Block CDD error | Block rain error (in) |
  |---|---:|---:|---:|
  | normal | 2.88 | 23.9 | 1.36 |
  | normal + local anomaly (± 15 days) | 2.76 | 22.7 | 1.46 |
  | **normal + neighbour anomaly** | **0.90** | **8.6** | **0.85** |

  The neighbour anomaly wins both branches (temperature judged on the block mean, precipitation on the block total).
- **Rules:** observed values are never changed (asserted); a filled Tmax below an observed Tmin is clipped to it
  (both filled → midpoint); the hottest day is never taken from a filled day; when station adjustment is used, days
  are filled on the reference station's level and the plain filled series moves them back by the day's offset.
- **Output:** `tmax_filled`, `tmin_filled`, `prcp_filled` in `daily_clean.parquet`; annual `_fill` variants.

## D7 Station offsets on days

From the station adjustment ([page 05](05-station-adjustment.md)): `tmax_offset`, `tmin_offset` per day;
adjusted value = QC value + offset. Days from a third station inside an adopted change cluster are removed from the
adjusted series (`*_flag = 4`).

## Flags in `daily_clean.parquet`

`*_flag`: 0 observed and unchanged, 1 missing in the raw data, 2 deleted by QC, 3 swapped, 4 transition day (removed
from the adjusted series only).
