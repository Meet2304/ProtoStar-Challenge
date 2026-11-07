# Data

This folder holds the data the pipeline reads. Only this README is tracked in Git. Everything else is downloaded or generated, so `.gitignore` keeps it out.

| Folder | What it holds | How it's made |
|---|---|---|
| `raw/` | The challenge dataset, exactly as downloaded | `uv run python scripts/01_download_data.py` |
| `processed/` | Data built from `raw/` | Later scripts (not written yet) |

## Source dataset

- **Name:** Daily and annual temperature and precipitation for 210 U.S. cities, up to 2018
- **Hub:** [CMU-OSPO/protostar-climate-train-to-2018](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018)
- **Revision:** `bbbdebdab2a18e00f97ef87bd3f35a71e222180b`, pinned in `src/protostar_challenge/config.py`
- **Licence:** CC0 1.0
- **Authors of the original data:** Yuchuan Lai, David Dzombak (Carnegie Mellon University)
- **Coverage:** 210 U.S. cities, 1851 to 2018-12-31. The organizers removed everything after 2018.
- **Units:** temperature in °F, precipitation in inches, cooling degree days (`cdd`) in °F-days.

## Files in `raw/`

| File | Rows | One row per | Size |
|---|---:|---|---:|
| `daily/train-00000-of-00001.parquet` | 10,245,754 | city and day | 49.3 MB |
| `annual_indices/train-00000-of-00001.parquet` | 28,052 | city and year | 0.74 MB |
| `coverage_and_stations/train-00000-of-00001.parquet` | 28,052 | city and year | 0.03 MB |
| `city_info/train-00000-of-00001.parquet` | 461 | station period (a city can have several) | 0.02 MB |

The download also writes the dataset card (`raw/README.md`) and `raw/.cache/huggingface/`. The `.metadata` files in that cache record the revision of every file.

## How to fetch it

From the repo root:

```bash
uv run python scripts/01_download_data.py
```

The same thing without the project's code:

```bash
uvx hf download CMU-OSPO/protostar-climate-train-to-2018 --repo-type dataset --revision bbbdebdab2a18e00f97ef87bd3f35a71e222180b --local-dir data/raw
```

## Preparation

None yet. `raw/` is used as downloaded. Add a row to the table above for every file a script writes to `processed/`, saying which script makes it and from what.

## Citation

The dataset card asks users to cite the original records:

- Lai, Yuchuan; Dzombak, David (2019). Compiled historical daily temperature and precipitation data for selected 210 U.S. cities. Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/7890488.v6
- Lai, Yuchuan; Dzombak, David (2019). Historical changes of annual temperature and precipitation indices at selected 210 U.S. cities. Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/7961012.v6
- Lai, Y., & Dzombak, D. A. (2019). Use of historical data to assess regional climate change. Journal of Climate, 32(14), 4299–4320. https://doi.org/10.1175/JCLI-D-18-0630.1
