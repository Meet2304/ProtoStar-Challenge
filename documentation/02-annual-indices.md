# 02 — `annual_indices`: the yearly table

One row per city and year (28,052): averages, totals, extremes, `cdd`, percentile indices, and missing-day counts.
Four of its columns are the challenge targets. Plan reference: §0.1 and §4.

## A1 Sentinels and fake zeros

- **Found:** `±inf` in years without data (the source's empty max/min); `ToT Prcp = 0` on 935 rows — 932 with every
  day missing ("fake zeros") and 3 with 337, 360 and 364 missing days.
- **Strategy:** `±inf → NaN`; every column becomes NaN when its year misses more than 10 days of the variables it uses
  (the official rule: Tmax and Tmin for the temperature metrics and CDD, precipitation for the total). This removes all
  935 zero rows. A 0-inch year is the most damaging possible error for a precipitation model.

## A2 Official labels and the audit

- **Labels** (`annual_avg_temp`, `total_precip_in`, `hottest_day_temp`, `cdd`): the **published** values masked
  exactly as `code/prepare_data.py` does (rounded to 3 decimals). They equal `data/annual_valid.csv` on all 28,052
  rows, NaN mask included — a hard check. No cleaning step changes them, because the 2019–2025 answers are computed
  from the raw record in the same way.
- **Why published and not rebuilt:** the published `Avg Temp` is a rounding chain (`Avg Tmax` and `Avg Tmin` rounded to
  2 decimals, then their mean rounded again). A direct daily mean differs by up to 0.0099 °F.
- **Audit** (hard check, every row valid for the column) — our definitions reproduce the published table:

  | Check | Rows | Max difference | Tolerance |
  |---|---:|---:|---:|
  | daily mean Tmax vs `Avg Tmax` | 24,868 | 0.0050 | 0.0051 |
  | daily mean Tmin vs `Avg Tmin` | 24,923 | 0.0050 | 0.0051 |
  | published `Avg Temp` vs (`Avg Tmax` + `Avg Tmin`)/2 | 24,583 | 0.0050 | 0.0051 |
  | direct daily `Avg Temp` vs published | 24,583 | 0.0099 | 0.0101 |
  | `ToT Prcp`, `Max/Min Tmax`, `Max/Min Tmin`, `cdd` | 24,583–24,923 | ≤ 1.4e-14 | 1e-6 |

## A2q QC values (inputs)

- `<column>_qc` = label + (f(cleaned days) − f(raw days)), where f is our rebuild. A QC value therefore equals the
  label exactly unless a repair changed that year, and then differs by exactly the effect of the repair; precision
  differences cannot leak in. It is NaN when the cleaned days fail the 10-day rule or the year is a dropped fragment
  (`<metric>_qc_invalidated` marks years a QC step made invalid).
- **Result:** QC values differ from the labels in 23 city-years for Avg Temp, 20 for precipitation, 17 for the hottest
  day and 17 for CDD — every one in a city-year with a logged repair, deletion or fragment drop (hard check). Examples:
  Greenville-Spartanburg 2018 Avg Temp +0.07 °F (one deleted Tmin); Astoria 1904 (6 stuck days removed, still valid).

## A3 Detached record fragments

- **Detection:** per variable group (temperature: any Tmax/Tmin; precipitation: any prcp), find blocks of years with
  data separated by more than 20 empty years. Grouping separately stops precipitation from bridging a temperature gap.
- **Strategy:** **drop** (in QC values and inputs; labels untouched) a block with fewer than 10 years of data that is
  not the group's most recent block. **Keep** every other block, with `record_break_before_<group> = True` on the first
  year after any gap of ≥ 10 empty years. v0 dropped everything before the last gap, which deleted 64 years of
  Huntsville and 60+ years of Akron and Wheeling; long blocks are real history.
- **Dropped** (29,989 readings, all before 1904): temperature — Athens 1858, Birmingham 1896–1903, Manchester
  1886–1893, Natchitoches 1894–96, Pueblo 1889–92, Sioux City 1890–95; precipitation — Asheville 1870–76, Birmingham
  1896–1903, Natchitoches 1894–96, Pueblo 1889–92, Sioux City 1890–95, Vancouver 1856. Dropped years are never
  infilled.
- **Breaks flagged:** 25 (temperature), 22 (precipitation), e.g. Akron 1999, Ely 1938, Huntsville 2018 (its only year
  since 1958; the city is not scored, but the long 1895–1958 block is kept as its climatology), Manchester 1996,
  Wheeling 1998, Trenton 1998, Louisville 2006. `qc/fragments.csv`, `qc/record_breaks.csv`.

## A4 Percentage indices

- **Found:** `Warm/Cold Days/Nights` are percentages of *all* days, so missing days count as "not warm": Warm Days
  averages 8.5 % in years with no missing day, 7.5 % with 31–100 missing, 0.9 % with more than 100.
- **Strategy:** **exact rescale** to a percentage of valid days, `pct × days / (days − missing)` (only valid days can be
  warm, so the count is recovered; published values have 2 decimals, < 0.02 days of error); NaN if more than 10 days are
  missing. Columns `*_pct_valid`.
- **Caveat:** their percentile thresholds use the start of record to 1987, so they are safe as forecast features only
  at origins ≥ 1987 (not used in the backtest). They are published-reference features: QC deletions are not removed
  from them.

## A5 `Wet Days`

Inches from days above the 95th percentile; reads low with missing days and has no exact correction. NaN when more
than 10 precipitation days are missing (`wet_days_in`). Same 1987 caveat as A4.

## A6 Empty years (inputs only)

- **Method:** a missing city-year = a neighbour's value that year + the mean city − neighbour difference over the 15
  nearest common valid years (≤ the end year; ≥ 5 common years per neighbour; ties to the earlier year), averaged with
  inverse-distance weights over ≥ 2 neighbours; never before a city's first kept valid year, never in a dropped
  fragment; precipitation and CDD clipped at 0. Donors are the original (pre-infill) values.
- **Evidence** (`qc/infill_evaluation.csv`, 400 hidden known years per metric in 1989–2018, seed 0). The **past-only**
  form uses only years before the hidden one for both the offsets and the comparison baseline:

  | Metric | Infill error (past-only) | City mean of previous 15 years | Reduction |
  |---|---:|---:|---:|
  | Avg Temp (°F) | 0.50 | 1.17 | −57 % |
  | CDD (°F-days) | 76.9 | 155.4 | −51 % |
  | Hottest day (°F) | 1.50 | 2.64 | −43 % |
  | Precipitation (in) | 3.65 | 5.20 | −30 % |

  The as-used form (nearest years on both sides) is reported too; it is slightly better and uses later years.
- **Output:** `<metric>_nbfill` and the `*_nbfill` combinations; `<metric>_input_source = neighbour infill` where used.
