# Data card: `data/`

The data the pipeline reads. The curated files in this folder are tracked in Git and uploaded with
the submission: together with `code/` they reproduce `submission.csv`. The raw download and scratch
builds are git-ignored.

| Path | What it holds | How it's made | Tracked |
|---|---|---|---|
| `annual_valid.csv` | Curated yearly metrics, read by the forecasting code | `python code/prepare_data.py` | Yes |
| `city_info.csv` | One row per city: ID, name, location | `python code/prepare_data.py` | Yes |
| `raw/` | The challenge dataset, exactly as downloaded (all four subsets, ~50 MB) | `uv run python scripts/01_download_data.py` | No |
| `processed/` | The cleaned dataset (`annual_clean`, `coverage_clean`, `stations_clean`, `daily_clean` parquet files, QC logs, backtest results, `manifest.json`); see [`documentation/07-outputs.md`](../documentation/07-outputs.md) | `uv run python scripts/02_clean_data.py` | No |
| `processed/releases/clean-v1/` | The line's current cleaned dataset: one folder per forecast origin (1983, 1990, 1997, 2004, 2011), each built only from data up to that origin; see [`documentation/09-per-origin.md`](../documentation/09-per-origin.md) | `uv run python scripts/line.py pull`; rebuilt by `uv run python scripts/03_clean_per_origin.py` | No |
| `qc/daily_outlier_decisions.csv` | Review file for the 424 isolated daily outliers: the evidence, the automatic action and an optional human `final_action` (with `decided_on` and `evidence_year`) | Written and updated by `scripts/02_clean_data.py`; human entries are kept on rerun | Yes |

Anything the submission code reads must be a tracked file listed in this card (move it out of `processed/`).

## Source dataset

- **Name:** Daily and annual temperature and precipitation for 210 U.S. cities, up to 2018
- **Hub:** [CMU-OSPO/protostar-climate-train-to-2018](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018)
- **Revision:** `bbbdebdab2a18e00f97ef87bd3f35a71e222180b`, pinned in `src/protostar_challenge/config.py` and `code/common.py` (keep them equal)
- **Licence:** CC0 1.0
- **Authors of the original data:** Yuchuan Lai, David Dzombak (Carnegie Mellon University)
- **Coverage:** 210 U.S. cities, 1851 to 2018-12-31. The organizers removed everything after 2018.
- **Units:** temperature in °F, precipitation in inches, cooling degree days (`cdd`) in °F-days.

## Curated files

| File | What it holds | Source | License | Years covered | How we prepared it | Used for |
|---|---|---|---|---|---|---|
| `annual_valid.csv` | One row per city and year: `city_id`, `name`, `year` and the four metrics (`annual_avg_temp`, `total_precip_in`, `hottest_day_temp`, `cdd`) | Source dataset, `annual_indices` | CC0-1.0 | 1851–2018 | Renamed columns to the metric codes; set a value to empty when that year missed more than 10 days of the data it uses (`Miss Tmax`/`Miss Tmin` for the temperature metrics and CDD, `Miss Prcp` for precipitation), the official scoring rule; rounded to 3 decimals | Forecast input |
| `city_info.csv` | One row per city: `city_id` (station `ID`), `name`, `lat`, `lon` | Source dataset, `city_info` | CC0-1.0 | n/a | Kept one row per city (the current station) | Reference (not read by the starter model) |

- **Join key:** `city_id` = the station `ID` in the challenge `city_info` table (e.g., Pittsburgh = `USW00094823`).
- **Known limitations:** each city's record joins several stations over time, so a change of station
  can shift its level (see the `coverage_and_stations` subset, not used yet). Since 1989, 271–288
  city-years per variable fail the 10-day rule and are empty here.

## Files in `raw/`

| File | Rows | One row per | Size |
|---|---:|---|---:|
| `daily/train-00000-of-00001.parquet` | 10,245,754 | city and day | 49.3 MB |
| `annual_indices/train-00000-of-00001.parquet` | 28,052 | city and year | 0.74 MB |
| `coverage_and_stations/train-00000-of-00001.parquet` | 28,052 | city and year | 0.03 MB |
| `city_info/train-00000-of-00001.parquet` | 461 | station period (a city can have several) | 0.02 MB |

The download also writes the dataset card (`raw/README.md`) and `raw/.cache/huggingface/`. The `.metadata` files in that cache record the revision of every file.

### How to fetch it

From the repo root:

```bash
uv run python scripts/01_download_data.py
```

The same thing without the project's code:

```bash
uvx hf download CMU-OSPO/protostar-climate-train-to-2018 --repo-type dataset --revision bbbdebdab2a18e00f97ef87bd3f35a71e222180b --local-dir data/raw
```

`code/prepare_data.py` does not need `raw/`: it reads the two subsets it uses straight from the Hub at the same revision.

## Data rule check

- [x] Every file marked "Forecast input" is weather data up to 2018-12-31, or data that does not respond to the weather. (`prepare_data.py` and `common.load_valid_annual()` stop if a row is dated after 2018.)
- [x] No 2019–2025 data that responds to the weather is used. (None is in this folder.)

## Internal evaluation data

The [evaluation platform](../evaluation/README.md) reads the pinned raw
`annual_indices` parquet through 2018, preserving source precision, and uses the
tracked `city_info.csv` city list. It writes versioned historical train/target
files, evaluator labels, reference errors and prediction artifacts under
`outputs/evaluation/` (git-ignored). These are derived from the same CC0 source
and have the same units. Each suite manifest records the source revision,
source/city-file SHA256, fold configuration, scoring-code hash and every derived
file hash. Labels cover only historical seven-year windows ending by 2018.
No external weather observations were added. NOAA GHCN-Daily catalog metadata
was researched as a public source; its observations were not downloaded.

The version-2 default declares development origins 1983/1990/1997 and gated
assessment origins 2004/2011. All periods are labeled `reused_historical`;
assessment freezes do not establish untouched evidence retroactively. Each
worker's supported loaders enforce its origin as well as the 2018 ceiling.
The default nested selector derives inner training/validation splits only from
the provided outer history and saves the full selection record. New suites
preserve all source precision and the documented scoring mask; previous
version-1 suites remain immutable development artifacts.

Evaluation labels are **analysis only**; each model's forecast inputs are its
origin-specific `train.csv`. Keep the exact suite with any archived run used as
evidence. The dated source and pipeline are described in
`evaluation/upstream/manifest.json` and `evaluation/README.md`.

## Citation

The dataset card asks users to cite the original records:

- Lai, Yuchuan; Dzombak, David (2019). Compiled historical daily temperature and precipitation data for selected 210 U.S. cities. Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/7890488.v6
- Lai, Yuchuan; Dzombak, David (2019). Historical changes of annual temperature and precipitation indices at selected 210 U.S. cities. Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/7961012.v6
- Lai, Y., & Dzombak, D. A. (2019). Use of historical data to assess regional climate change. Journal of Climate, 32(14), 4299–4320. https://doi.org/10.1175/JCLI-D-18-0630.1

## Supplementary station validation (analysis only)

NOAA NCEI GHCN-D Daily Summaries observations were requested only for
1950-01-01 through 2018-12-31, with explicit API start/end dates and response
validation before caching. The structural station and inventory catalogs come
from https://www.ncei.noaa.gov/pub/data/ghcn/daily/; the date-bounded API is
https://www.ncei.noaa.gov/access/services/data/v1 (see its official documentation
at https://www.ncei.noaa.gov/access/search/documentation/data-service/). These
public U.S. government observations are supplementary analysis only, not inputs
to submission.csv. V1 failed its frozen coverage minimum without model scoring;
V2 freezes 48 different WBAN station candidates and qualifies 33 across six
regions under unchanged quality thresholds. Selection and models precede
acquisition/scoring; every method and exclusion is reported. Exact raw response
bytes, provenance and aggregation inputs are archived in
evaluation/independent/snapshots/v2-acquisition.tar.gz and its SHA256 manifest.
See evaluation/independent/README.md for chronology, limits and the one-time
opening rule. No observed 2019+ weather is obtained or used.
