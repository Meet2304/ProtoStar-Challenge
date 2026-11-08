# Data card: `data/`

The data the pipeline reads. The curated files in this folder are tracked in Git and uploaded with
the submission: together with `code/` they reproduce `submission.csv`. The raw download and scratch
builds are git-ignored.

| Path | What it holds | How it's made | Tracked |
|---|---|---|---|
| `annual_valid.csv` | Curated yearly metrics, read by the forecasting code | `python code/prepare_data.py` | Yes |
| `city_info.csv` | One row per city: ID, name, location | `python code/prepare_data.py` | Yes |
| `raw/` | The challenge dataset, exactly as downloaded (all four subsets, ~50 MB) | `uv run python scripts/01_download_data.py` | No |
| `processed/` | Scratch data built from `raw/` | Later scripts | No |

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
| `ocean_indices.csv` | One row per index, year and month: `index` (`oni`, `pdo`, `amo`), `year`, `month`, `value` | NOAA Physical Sciences Laboratory climate indices (ONI = Oceanic Niño Index; PDO = Pacific Decadal Oscillation; AMO = Atlantic Multidecadal Oscillation, unsmoothed), files `oni.data`, `pdo.data`, `amon.us.data` under `www.esrl.noaa.gov/psd/data/correlation/` (now `psl.noaa.gov`) | U.S. government work (public domain) | up to the last month published in the archived copy (2018 at the latest) | `python code/prepare_ocean_indices.py`: downloads the Internet Archive's latest copy of each file captured on or before 2018-12-31 (the live files cannot be cut at 2018, so they are never called), parses it, drops missing values, and checks with `assert_no_post_cutoff()` | Forecast input candidate (being tested) |
| `ocean_indices_sources.csv` | For each index: original URL, archive capture time, archive URL, last month with a value | Written by `code/prepare_ocean_indices.py` | — | — | Provenance record for `ocean_indices.csv` | Reference |

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

## Citation

The dataset card asks users to cite the original records:

- Lai, Yuchuan; Dzombak, David (2019). Compiled historical daily temperature and precipitation data for selected 210 U.S. cities. Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/7890488.v6
- Lai, Yuchuan; Dzombak, David (2019). Historical changes of annual temperature and precipitation indices at selected 210 U.S. cities. Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/7961012.v6
- Lai, Y., & Dzombak, D. A. (2019). Use of historical data to assess regional climate change. Journal of Climate, 32(14), 4299–4320. https://doi.org/10.1175/JCLI-D-18-0630.1
