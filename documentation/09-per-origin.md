# 09 — Per-origin cleaning

This is release `clean-v1`, the line's current cleaned dataset. `clean-v0` cleans the whole record through 2018 at once. It is fine for exploration, but a backtest at origin 1983
would train on data whose thresholds, outliers and station adjustments were fitted with 1984–2018 in view. The
per-origin build fixes that: one complete cleaned dataset for each forecast origin in `evaluation/config.json`
(1983, 1990, 1997 and the sealed 2004 and 2011), each built from data up to 31 December of that origin only.

```bash
uv run python scripts/03_clean_per_origin.py                # all origins -> outputs/clean_per_origin/ (~15 min)
uv run python -m unittest tests.test_clean_per_origin       # truncation and guards (seconds)
```

## How the future is kept out

[`cleaning_per_origin.truncate_raw`](../src/protostar_challenge/cleaning_per_origin.py) cuts all four subsets at the
origin *before* anything else runs: later rows are gone, stations opening later do not exist, and stations still open
end on the origin date. The steps in `cleaning.py` run unchanged on that copy, inside `fold_cutoff(origin)`, so every
`assert_no_post_cutoff` guard applies the origin instead of 2018. Every output table is checked to end by the origin
before it is written.

| Fitted per origin | Full-record build | Per-origin build |
|---|---|---|
| Robust z thresholds, stuck runs, repeated rain, outliers, fragments | through 2018 | through the origin |
| Station history: empty periods, overlaps, clusters | used later readings and end dates | as known at the origin |
| Station adjustment: steps, placebo critical values | through 2018 | through the origin; the reference station is the one in use at the origin |
| Fill method (bake-off) | chosen on data ≤ 1985 | chosen on the 30 years ending at the origin |
| Normals, neighbour infill | through 2018 | through the origin |
| Warm/Cold Days and Nights, Wet Days | published, thresholds through 1987 | blank (NaN) at origin 1983; kept from 1990 |

Station-adjusted values therefore differ between origins by design: each dataset is levelled to the station in use at
its own origin.

## What is not in it

- **`<metric>_input`, `_input_source`, `_input_variant`.** The full build chooses them with backtests running to 2018,
  which is outside what an origin may know. A past-only choice at 2004 and 2011 would need to score candidates on
  years held back from assessment. Every candidate series is present (for example `annual_avg_temp_homog`,
  `cdd_fill_homog`), so the station that scores models picks one per metric through the harness.
- **Backtest and sensitivity tables, outlier plots, the infill evaluation.** They describe the full build's choices.
- **`data/qc/daily_outlier_decisions.csv` is not rewritten.** Each origin writes `qc/isolated_outliers.csv` instead, and
  a human override only applies at origins on or after its `evidence_year`.

## Outputs

`origin-YYYY/` holds `annual_clean`, `coverage_clean`, `stations_clean` and `daily_clean` (parquet, same columns as
[page 07](07-outputs.md) less the ones above), `qc/` (the same logs, plus `fill_bakeoff.csv`) and `manifest.json`
(input and code hashes, constants, checks, counts).

Cite a file in an asset manifest as `data/processed/releases/clean-v1/origin-YYYY/<file>` with
`"information_end_year": YYYY` and `"origins": [YYYY]`.
