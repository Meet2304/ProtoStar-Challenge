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
| `processed/releases/clean-v2/` | The line's current cleaned dataset: `clean-v1`'s weather folders unchanged plus `static/city_static_clean.csv`, the cleaned city facts from the enhanced dataset; see [`documentation/10-enhanced.md`](../documentation/10-enhanced.md) | `uv run python scripts/line.py pull`; rebuilt by `uv run python scripts/04_clean_enhanced.py` | No |
| `processed/releases/clean-v1/` | The previous cleaned dataset: one folder per forecast origin (1983, 1990, 1997, 2004, 2011), each built only from data up to that origin; see [`documentation/09-per-origin.md`](../documentation/09-per-origin.md) | `uv run python scripts/line.py pull`; rebuilt by `uv run python scripts/03_clean_per_origin.py` | No |
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

## Enhancement releases: fixed city geography (`enhance-v1`)

**Use: forecast input, structural.** The data rule permits these fixed facts from
any year. This release describes the supplied current station coordinates, not
historical station segments or a municipal population center. No weather target,
assessment partition, or post-2018 weather observation is read by this builder.

The authoritative handoff is an immutable assembly-line release, not another
station's build folder. Run `uv run python scripts/line.py pull enhance-v1` and
read `data/processed/releases/enhance-v1/static/city_static.csv`. The release
contains all 210 `city_id` values from the tracked `city_info.csv`, its input hash,
source snapshots, a column schema, and validation results. No generated data is
committed to Git; the code and this data card are committed before publication.
For assembly-line model assets this immutable release path supersedes the older
scratch-folder guidance above; do not read another station's working directory.

| Fields | Source and edition | Preparation and units | License |
|---|---|---|---|
| `lat`, `lon`, `city_id`, `name` | Pinned challenge `city_info.csv` | Identity and observing coordinates, in degrees | CC0-1.0 |
| `elevation_m`, `elevation_resolution_m`, `elevation_acquisition_date` | USGS 3DEP Elevation Point Query Service and direct 3DEP `getSamples`, response snapshots at acquisition | Ground elevation in meters at each supplied coordinate; preserve raster resolution and source date | U.S. government public domain |
| `station_catalog_elevation_m`, `station_catalog_lat`, `station_catalog_lon`, `elevation_coordinate_offset_km` | NOAA GHCN-Daily structural station catalog, snapshot at acquisition | Match exact station ID; retain catalog elevation and coordinate offset as context, not as a substitute for ground elevation at the supplied coordinate | U.S. government public data |
| `distance_to_ocean_coast_km`, `coast_sampling_error_bound_km` | Natural Earth coastline shapefile from Git tag `v5.1.2` | Minimum spherical distance to the ocean coastline sampled at intervals no greater than 0.5 km; sampling alone overestimates by at most 0.25 km | Public domain |
| `landcover_station_class`, `landcover_dominant_class_1km`, `landcover_*_fraction_1km`, coverage fields | ESA WorldCover 2021, algorithm `v200`, approximately 10 m | Native categorical map values; pixel centers within a 1-km radius, cosine-latitude area weights, exclude nodata class 0; fractions sum to one over valid area | CC-BY-4.0 |

Source links and credits:

- USGS: https://www.usgs.gov/the-national-map-data-delivery/gis-data-download;
  point API https://epqs.nationalmap.gov/v1/json. **Data available from U.S.
  Geological Survey, National Geospatial Program.** Each response records its
  exact query coordinates, retrieval time, raster ID, source date, and resolution.
  The same 3DEP image service supports batched queries at
  https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/getSamples;
  these use bilinear interpolation and preserve original batch responses and
  location IDs. EPQS timeouts prompted this alternative. Invalid provider date
  strings are retained in `elevation_acquisition_date_raw` and normalized to
  `unknown`, without inventing a date or discarding the valid elevation.
- NOAA: https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt;
  field definitions https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt.
  Menne et al. (2012), GHCN-Daily, https://doi.org/10.7289/V5D21VHZ.
  Only the structural catalog is downloaded; no `.dly` observation file is used.
- Natural Earth: https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/;
  exact file https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/10m_physical/ne_10m_coastline.shp.
  The `10m` in this name means **1:10 million map scale, not 10-meter accuracy**.
  The Great Lakes are excluded. Source generalization and using a mean-radius
  sphere add uncertainty beyond the stated sampling bound. This is not a
  flood, shoreline-hazard, or nearest-water-body distance.
- ESA: https://esa-worldcover.org/en/data-access;
  Zanaga et al. (2022), *ESA WorldCover 10 m 2021 v200*,
  https://doi.org/10.5281/zenodo.7254221.
  **© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data
  (2021) processed by ESA WorldCover consortium.** The release retains small,
  losslessly compressed native raster windows with tile URL, extraction window,
  affine transform and SHA256. These are derived portions, not full tiles.

Class codes: 10 tree, 20 shrub, 30 grass, 40 cropland, 50 built, 60 bare,
70 snow/ice, 80 permanent water, 90 herbaceous wetland, 95 mangrove, 100 moss/lichen.
Class codes are categorical. Tree-class fraction is not canopy percentage;
built-class fraction is not impervious-surface percentage. The 2021 map is a
fixed structural snapshot permitted by the challenge, not a reconstruction of
land cover in earlier backtest years. This limitation must accompany any result.

Rebuild (only from this station):

```powershell
uv run --with rasterio==1.4.3 --with pyshp==2.3.1 python code/enhance_static.py
uv run python -m unittest discover -s code -p test_enhance_static.py -v
uv run python code/enhance_static.py --validate outputs/enhancement/static-v1
uv run --with rasterio==1.4.3 --with pyshp==2.3.1 python code/enhance_static.py --from-release data/processed/releases/enhance-v1 --output outputs/enhancement/static-v1-rebuild
```

The two extra packages are isolated by `uv run --with`; shared dependency files
are unchanged. `--offline` rebuilds from the station's saved source cache without
network access. The published source chips and point-response snapshots support
independent recalculation of the released features; provider refreshes do not
silently replace their values. The provenance file records all input checksums
and source retrieval metadata.

AI use: Codex (GPT-6.1-sol) wrote the acquisition and validation code and this
source documentation. A person should review the station-location interpretation,
source licensing, snapshot assumptions and eligibility before model adoption.
No predictive-value claim is made until a model is scored through `line.py score`.

## Ocean-temperature release (`enhance-v2`)

The requested friendly tagged copy is `enhanced/enhance-v2/`, materialized by
`uv run python scripts/materialize_enhanced.py`. `enhanced/latest.json` records
the current tag and release-manifest hash for future model-input evaluations.
The helper uses the assembly-line release downloader and validates every file.
Evaluator asset declarations keep the canonical `data/processed/releases/`
paths so lineage and eligibility remain visible. This does not change the
underlying release, its eligibility, or evaluation target data.

**Use: historical forecast-input candidates, exploratory; not certified
gate-eligible.** This cumulative release includes the unchanged `enhance-v1`
city table, links that immutable release as upstream, and adds three monthly
ocean-temperature summaries. It does not supply future observed indices.

Source: **NOAA Extended Reconstructed Sea Surface Temperature, version 5
(ERSSTv5)**, absolute SST in degrees Celsius, on a 2-degree grid. NOAA NCEI
documentation: https://www.ncei.noaa.gov/products/extended-reconstructed-sst;
data DOI https://doi.org/10.7289/V5T72FNM. Cite Huang et al. (2017), *Extended
Reconstructed Sea Surface Temperature, Version 5*, NOAA NCEI. Public U.S.
government data; attribution requested. This is explicitly v5, not the newer
v6 neural-network reconstruction.

The date-bounded service is
https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.ersst.v5/sst.mnmean.nc.
Every request specifies `time_start=1950-01-01T00:00:00Z` and
`time_end=2018-12-01T00:00:00Z`, with `var=sst`, monthly time stride 1, and the
regional bounds below. Dates are validated **before** a source snapshot is
written. Each region contains exactly 828 months, January 1950–December 2018.
Validated arrays, coordinates, exact query URLs, retrieval times and source and
snapshot hashes are retained. Provider-wide NetCDF statistics are not used.

| Index prefix | Latitude bounds | Longitude bounds, degrees east | Definition |
|---|---|---|---|
| `nino34` | 5°S–5°N | 190–240 | Niño 3.4 regional mean SST; custom per-origin anomalies, not the official ONI |
| `north_pacific` | 20–60°N | 120–260 | Broad North Pacific ocean-cell mean; not the official PDO |
| `north_atlantic` | 0–60°N | 280–360 | Broad North Atlantic ocean-cell mean; not the official AMO/AMV |

Only grid-cell centers inside each box are used. Missing/land cells are excluded,
and ocean cells are weighted by cosine of latitude. Each monthly table includes
the absolute `_sst_c` value and `_anomaly_c` value for each region. Anomalies are
computed separately at every origin against the calendar-month means of the
30 years ending at that origin. No full-1950–2018 normalization, centered
smoothing, official future-base-period anomaly, or learned PDO pattern is used.
Annual tables are equal-weight means of the 12 monthly indices, not daily or
day-weighted air-temperature targets.

There are monthly and annual snapshots for **each origin from 1979 through
2018**, enabling nested training to use its own information cutoff. For example,
`ocean/origin-1983/monthly.csv` ends in December 1983 and its reference period is
1954–1983. These paths are inputs derived from public ocean records, not opened
assessment labels or scored assessment results. Do not load the 2018 table
inside an earlier fold. `load_ocean_for_origin()` in `code/enhance_ocean.py`
checks the released file hash, the requested origin and the active
`PROTOSTAR_FOLD_CUTOFF` using the existing `assert_no_post_cutoff()` guard.

**Eligibility limitation:** trimming observation dates and computing per-origin
anomalies does not establish that NOAA's retrospective reconstruction/calibration
was itself learned only from data before each early historical origin. Source
vintage has not been independently certified here. Accordingly, `enhance-v2` is
released **without `--gate-eligible`**. This prevents presenting it as clean
gated-assessment evidence. The static-only `enhance-v1` remains separately
gate-eligible; use that release for structural assets in eligible trials. The
integrator must resolve source-vintage eligibility before using these indices
in an official gated assessment. This is separate from the final 2018 ceiling,
which every retained observation satisfies.

**Acquisition incident:** an initial NCSS probe ending at December 31, 2018
included the adjacent January 2019 coordinate because the server selects a
nearest monthly endpoint. Validation rejected the response before caching or
feature use. Subsequent requests use the exact December 1, 2018 monthly
timestamp. All retained arrays pass the full 828-month check; no 2019 observation
enters either release or a forecast. A regression test covers this boundary.

Acquire, build, or validate from the enhancement worktree:

```powershell
uv run python code/enhance_ocean.py --acquire-only
uv run python scripts/line.py pull enhance-v1
uv run python code/enhance_ocean.py --offline
uv run python code/enhance_ocean.py --validate outputs/enhancement/ocean-v2
uv run python -m unittest discover -s code -p 'test_enhance_*.py' -v
```

For handoff, use `uv run python scripts/line.py pull enhance-v2`. Its
`static/city_static.csv` matches `enhance-v1`; full structural source snapshots
remain in that upstream release. Ocean source snapshots are under
`sources/ocean/`. To rebuild solely from published snapshots, pull both releases
and pass `--cache data/processed/releases/enhance-v2/sources/ocean --offline
--static-release data/processed/releases/enhance-v1 --output <new-build-folder>`.
Never write into a published release.

No model is fitted, no assessment labels are read, and no skill improvement is
claimed by this station. Consumers must declare these release assets and score
their models with `line.py score`. Ocean currents, solar cycles and climate-model
projections were intentionally excluded from this implementation.
