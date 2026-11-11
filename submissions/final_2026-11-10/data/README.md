# Data card: `data/`

The data the pipeline reads. The curated files in this folder are tracked in Git and uploaded with
the submission: together with `code/` they reproduce `submission.csv`. The raw download and scratch
builds are git-ignored.

| Path | What it holds | How it's made | Tracked |
|---|---|---|---|
| `annual_indices_train_to_2018.parquet` | The challenge's yearly table, unchanged: **the submitted model's main weather input** | Copied from the source dataset at the pinned revision (see below) | Yes |
| `annual_valid.csv` | Curated yearly metrics, read by the earlier combo and starter models | `python code/prepare_data.py` | Yes |
| `city_info.csv` | One row per city: ID, name, location | `python code/prepare_data.py` | Yes |
| `forcing_ar6_to_2018.csv` | IPCC AR6 annual effective radiative forcing by component, 1750–2018: **the submitted model's external input** (Placeholder_F4's forced-response temperature members) | Copied from the team's `enhance-v3` release (see [Global radiative forcing](#global-radiative-forcing-enhance-v3)) | Yes |
| `daily_warm_days.csv` | Days with both Tmax and Tmin, and days with a mean above 65 °F, per city and year: Placeholder_v9-S's CDD response (F4 uses v9-S only for 49 unscored fallback rows) | `python code/prepare_warm_days.py` (from the challenge `daily` subset, up to 2018) | Yes |
| `raw/` | The challenge dataset, exactly as downloaded (all four subsets, ~50 MB) | `uv run python scripts/01_download_data.py` | No |
| `processed/f2/` | Placeholder_F4's cleaned daily features (`daily_features.parquet`, QC report) and CDD response tables, from the challenge `daily` subset up to 2018 | `python code/f2/f2_daily.py`, `python code/f2/f2_cddgrid.py` | No |
| `processed/` | The cleaned dataset (`annual_clean`, `coverage_clean`, `stations_clean`, `daily_clean` parquet files, QC logs, backtest results, `manifest.json`); see [`documentation/07-outputs.md`](../documentation/07-outputs.md) | `uv run python scripts/02_clean_data.py` | No |
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
| `annual_indices_train_to_2018.parquet` | One row per city and year (28,052 rows): the source's `annual_indices` columns, including `Avg Temp`, `ToT Prcp`, `Max Tmax`, `cdd` and the `Miss Tmax` / `Miss Tmin` / `Miss Prcp` counts | Source dataset, `annual_indices/train-00000-of-00001.parquet` | CC0-1.0 | 1851–2018 | None: a byte-for-byte copy (SHA-256 `0b5f5fa9b55ab59602dcd9884b97c74c7bec7e89ea9d23f4219703046030e411`, checked by `code/make_submission.py`). The code applies the official 10-day validity rule when it reads it (`evaluation.core.load_annual`) and refuses any row after 2018 | Forecast input (Placeholder_v1.4, the submission) |
| `annual_valid.csv` | One row per city and year: `city_id`, `name`, `year` and the four metrics (`annual_avg_temp`, `total_precip_in`, `hottest_day_temp`, `cdd`) | Source dataset, `annual_indices` | CC0-1.0 | 1851–2018 | Renamed columns to the metric codes; set a value to empty when that year missed more than 10 days of the data it uses (`Miss Tmax`/`Miss Tmin` for the temperature metrics and CDD, `Miss Prcp` for precipitation), the official scoring rule; rounded to 3 decimals | Forecast input |
| `daily_warm_days.csv` | One row per city and year (28,052 rows): `city_id`, `year`, `n_both` (days with both Tmax and Tmin, the days the dataset's `cdd` uses), `n_warm` (those days with (Tmax + Tmin) / 2 > 65 °F) | Source dataset, `daily/train-00000-of-00001.parquet` | CC0-1.0 | 1851–2018 | Counted per city and year by `code/prepare_warm_days.py`; nothing else derived; refuses any row after 2018 | Forecast input (Placeholder_v9-S, the submission: the CDD response uses the 30 years up to 2018, valid CDD years only) |
| `forcing_ar6_to_2018.csv` | One row per year (269 rows, 1750–2018): effective radiative forcing in W m⁻² by component (CO₂, CH₄, N₂O, other greenhouse gases, ozone, aerosols, land use, volcanic, solar, …) and the anthropogenic, natural and total sums | IPCC AR6 WGI Chapter 7, `data_output/AR6_ERF_1750-2019.csv` at commit `32ca11dc`, prefix through 2018 only, via the `enhance-v3` release | MIT (Chris Smith 2020); cite Forster et al. (2021) | 1750–2018 | None: the 2019 row was never acquired | Forecast input (Placeholder_F4, the submission). A physical driver, not a weather observation; published in 2021, so a later estimate of the past |
| `city_info.csv` | One row per city: `city_id` (station `ID`), `name`, `lat`, `lon` | Source dataset, `city_info` | CC0-1.0 | n/a | Kept one row per city (the current station) | Forecast input (coordinates only, for the submitted model's neighbour graph); not read by the starter model |

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

## Global radiative forcing (`enhance-v3`)

**Use: historical forecast-input candidates, exploratory, not gate-eligible.**
Requested by the deep-learning station for a forcing-aware model. This cumulative
release retains the unchanged city and ocean tables from `enhance-v2` and adds
annual global effective radiative forcing (ERF), in watts per square meter
(`W m-2`). ERF describes the change in net energy flow after rapid atmospheric
adjustments, before the surface-temperature response. It is not observed surface
temperature, Earth energy imbalance, or a future climate projection.

### Sources and publication dates

1. **Primary complete series: IPCC AR6 WGI Chapter 7 historical best-estimate
   ERF**, assessment published **2021**. Author Chris Smith's IPCC repository
   data revision is **2021-08-12 16:34:15 UTC**, commit
   `32ca11dc8a90f9eddee23686e08bff7a88f11783`:
   https://github.com/IPCC-WG1/Chapter-7/tree/32ca11dc8a90f9eddee23686e08bff7a88f11783.
   The upstream file is `data_output/AR6_ERF_1750-2019.csv`; **only its prefix
   through 2018 was acquired**. Historical forcing dataset documentation:
   https://github.com/IPCC-WG1/Chapter-7. Scientific assessment: Forster et al.
   (2021), *The Earth's Energy Budget, Climate Feedbacks, and Climate
   Sensitivity*, IPCC AR6 WGI Chapter 7,
   https://doi.org/10.1017/9781009157896.009.
   Repository license: MIT, copyright Chris Smith 2020; full notice retained in
   `forcing/sources/AR6_LICENSE.txt`. Cite both the assessment and author dataset.
2. **Separate pre-2019 reference: IPCC AR5 WGI Annex II Table AII.1.2**, assessment
   published **2013**, historical ERF **1750–2011**. The author-curated CSV used
   by the FaIR v1.3 project is pinned to its **2018-01-08 17:11:44 UTC** revision,
   commit `fcd941dc9729d16cb1df81758214bc34b20884dc`:
   https://github.com/OMS-NetZero/FAIR/blob/fcd941dc9729d16cb1df81758214bc34b20884dc/fair/ancil/cmip5_annex2_forcing.csv.
   Original assessment:
   https://www.ipcc.ch/site/assets/uploads/2017/09/WG1AR5_AnnexII_FINAL.pdf.
   Redistribution source and method: Smith et al. (2018), *FAIR v1.3*, published
   June 18, 2018, https://doi.org/10.5194/gmd-11-2273-2018.
   The FaIR repository's Apache-2.0 notice is retained in
   `forcing/sources/AR5_FAIR_LICENSE.txt`; retain IPCC and FaIR attribution.

**2012–2018 decision:** no clean pre-2019, fully historical, all-component
extension was identified in this search. GISS's CMIP5 documentation explicitly
warns that forcing after 2000/2005 includes RCP extrapolations
(https://data.giss.nasa.gov/modelforce/). Those values were not used as observed
history. We use the homogeneous, later-published **AR6 series for all 1750–2018**,
including 2012–2018, rather than splice assessments with different forcing
definitions and baselines. The AR5 series remains separate for sensitivity work.
No extrapolated forcing, RCP/SSP scenario or future weather is supplied.

### Acquisition and processing

`code/enhance_forcing.py` requests HTTP byte ranges from the pinned AR6 file.
The numeric CSV's sequential years and field counts determine conservative range
sizes. Requests stop at the end of the complete 2018 record. This acquired bytes
**0–101710 inclusive** of the 102046-byte upstream file; the 2019 record was
not requested. A server that ignores byte ranges is rejected from its headers,
before its response body is consumed. A test checks that no request crosses a
synthetic final permitted row. Only the 1750–2018 prefix is cached or released.
The complete AR5 reference is a closed historical file ending in 2011.

Exact URLs, source commits, data revision timestamps, retrieval time, byte bounds,
ETag, source hashes and license hashes are saved in
`forcing/sources/acquisition.json` and root `provenance.json`. The downloaded
prefix's SHA256 is
`e0edc9113276a231c81547cef0a3d0214026d17810bd95ff74287427e5bae700`.
The unmodified AR5 CSV SHA256 is
`059432ff3ffda9947fdee18a8b9718a7bf351088e04c7faa5b1f24eecf63f33c`.

Processing only renames component columns, computes explicitly documented sums,
and cuts rows by origin. No time interpolation, smoothing, normalization,
forecasting or fitting to station temperatures occurs. Source anthropogenic
forcing uses the 1750 convention; natural components retain their published
background offsets. In particular, **do not assume volcanic or solar forcing
must equal zero in the 1750 row**. Do not align AR5 and AR6 by treating their
absolute natural-forcing offsets as identical.

Main columns, all `W m-2`, except integer `year`:

| Column | Meaning |
|---|---|
| `wmghg_erf_wm2` | Sum of CO2, CH4, N2O and other well-mixed greenhouse gases |
| `co2_erf_wm2`, `ch4_erf_wm2`, `n2o_erf_wm2`, `other_wmghg_erf_wm2` | Individual greenhouse-gas groups |
| `tropospheric_aerosol_erf_wm2` | Sum of aerosol-radiation and aerosol-cloud interactions |
| `aerosol_radiation_erf_wm2`, `aerosol_cloud_erf_wm2` | The two aerosol contributions |
| `volcanic_erf_wm2` | Published volcanic/stratospheric-aerosol forcing |
| `solar_erf_wm2` | Published solar forcing |
| `other_anthropogenic_erf_wm2` | Ozone, stratospheric water, contrails, BC on snow and land-use contributions |
| `anthropogenic_total_erf_wm2`, `natural_total_erf_wm2`, `total_erf_wm2` | Published aggregate totals |

The smaller individual components are also present. Aggregate columns overlap
their components: **do not sum every column**. Validation checks greenhouse-gas
and aerosol grouping, anthropogenic closure, natural closure, and overall closure
against the source to numerical tolerance. The schema describes every mapping.

### Origin files, design resource and assets

Run `uv run python scripts/line.py pull enhance-v3` in the consuming worktree.
Read only declared assets from `data/processed/releases/enhance-v3/`.

- `forcing/origin-O/annual.csv`, for **O = 1983, 1990, 1997, 2004, 2011, 2018**:
  exactly one row per year from 1750 through O, no later forcing value.
- `forcing/design-to-1982/annual.csv`: complete primary series through 1982 for
  the explicitly requested design screen, with 233 rows.
- `forcing/design-to-1982/ar5-reference.csv`: separate pre-2019 AR5 design
  reference through 1982, preserving its source column names and units.
- `assets/forcing-evaluation.json`: six origin-scoped asset declarations.
- `assets/forcing-design-to-1982.json` and
  `assets/forcing-ar5-design-to-1982.json`: design-scope declarations.

Templates conservatively use `kind: weather_input`, `information_end_year: O`,
`origins: [O]`, `time_column: year` and `time_format: year`. This enforces temporal
bounds even though forcings are external drivers. Use the applicable forcing
declarations in the model's own asset manifest; combine them with any other
inputs rather than silently replacing an existing manifest. The source snapshots
are provenance, not permission to read all years inside a forecast worker.

`load_forcing_for_origin()` checks release hashes, the selected origin and the
active fold cutoff. The full 1982 design file is not origin-clean for a forecast
made in 1915: an earlier design screen must construct its own earlier cuts and
fit transformations using that screen's training information. The provided
1982 declaration intentionally cannot be bound to an earlier-origin worker.
Every forcing after the active origin, including future volcanic events, must
be forecast or marginalized by the training model itself.

### Eligibility judgment and reproducibility

**Not gate-eligible / not official.** Greenhouse gases and solar/volcanic drivers
are physically external, but that does not establish information independence
of their reconstructed historical estimates. AR6 is a 2021 retrospective
assessment; aerosol/cloud and volcanic reconstructions are not verified as
using only information available before each historical origin. AR5 is pre-2019
but likewise retrospective for the early design eras. Merely dropping later
rows does not resolve the calibration/vintage issue. This station therefore
publishes `enhance-v3` without `--gate-eligible`. A later source publication date
is disclosed explicitly, and does not become hidden by the year column.

This is a covariate experiment, not proof of the cause of the champion's errors.
No model was trained, no target or protected assessment partition was read, and
no predictive gain is claimed. Forecast and submission eligibility still require
the team's source review. Score subsequent models with `line.py score` and keep
their exploratory labels.

```powershell
uv run python code/enhance_forcing.py --acquire-only
uv run python code/enhance_forcing.py --offline
uv run python code/enhance_forcing.py --validate outputs/enhancement/forcing-v3
uv run python -m unittest discover -s code -p 'test_enhance_*.py' -v
```

An offline rebuild can use the released `forcing/sources/` as `--cache`, a pulled
`enhance-v2` as `--upstream`, and a **new** `--output` directory. Never overwrite
an immutable release. Code and provenance were prepared by Codex (GPT-6.1-sol).

### Measured greenhouse-gas add-on: screened and deferred

The training station subsequently requested a cheap, potentially official
concentration-only alternative. A source/provenance review on 2026-11-08 found
useful pre-2019 observations, but did not establish an official route for all
requested historical origins. No concentration dataset was downloaded or
released for this request; `enhance-v3` remains unchanged.

The [Law Dome NOAA archive](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=noaa-icecore-9959)
is a legitimate pre-2019 source: the underlying MacFarling Meure et al. paper
was published on 2006-07-21, and the archive citation is dated 2010-07-16
(DOI `10.25921/g6kd-k189`). It covers ice-core, firn-air and Cape Grim records
of CO2, CH4 and N2O through 2006. However, its metadata documents calibration
corrections, dating revisions using 2004 information, and gas-age spreads across
multiple years. A measured concentration assigned an old atmospheric year is
therefore not automatically information that was available at that year's
forecast origin. Those records are also not direct annual global measurements.

The [NOAA atmospheric concentration change log](https://www.gml.noaa.gov/ccgg/trends/trends_log.html)
documents retrospective changes to measurement selection, reference-gas scales
and early records. In particular, it describes recalculation of historical data
in 2011, changed background selection in 2017, and a further scale update in
2021. Downloading today's file and dropping later rows would not produce a
verified pre-2019 vintage or a verified vintage for each origin.

**Eligibility call for the proposed file alone: not certified gate-eligible.**
This is not a judgment that measured concentrations can never qualify. To make
that case, locate and pin suitable archived vintages, establish which
measurements and dating/calibration information each origin may use, and build
any interpolation or splice separately within each origin. A pre-2019
publication date alone does not certify the 1982 design cut or the 1983/1990
folds. The needed archive and processing audit goes beyond a cheap add-on, so
the user's explicit skip option was taken. The shipped AR6/AR5 experiment stays
available; no new release, gate flag, evaluation default or champion status was
changed as a result of this review.

## Processed daily data for Placeholder_F4

The two files Placeholder_F4 reads from the `daily` subset, included as built on 2026-11-10 (the build that reproduced `submission.csv` with SHA-256 `8344e656…3bc37`). Source: the challenge `daily` subset at the pinned revision (CC0-1.0), 1851-01-01 to 2018-12-31 only; downloaded by `code/download_raw.py` into `data/raw/` (not copied here). Used for: forecast input (Placeholder_F4, the submission).

| File | What it holds | How we prepared it |
|---|---|---|
| `processed/f2/daily_features.parquet` | One row per city and year (28,052 rows, 30 columns): day counts (`n_tmax`, `n_tmin`, `n_both`, `n_prcp`), cleaned yearly values (`tavg`, `cdd_raw`, `hot1`, `prcp_raw`, …), warm-season and seasonal means (`warm_tavg`, `t_djf`, `t_mam`, `t_jja`, `t_son`) and gap-filled inputs (`cdd_fill`, `prcp_fill`) | `code/f2/f2_daily.py`: sets to missing 9 days with Tmax < Tmin, 462 climatological outliers and 32 one-day spikes, then summarises each city and year; checks parity with the official `annual_indices` values |
| `processed/f2/cdd_grid.npz` | Monthly CDD response tables per city and year: how CDD changes when the observed daily mean temperatures are shifted | `code/f2/f2_cddgrid.py`, from the cleaned daily data above |
| `processed/f2/daily_qc_report.json` | Counts of what the cleaning changed (rows, missing values before and after, each rule) | Written by `code/f2/f2_daily.py` |

## Outside data (analysis and tested inputs)

| File | Content | Source | License | Years | Preparation | Role |
|---|---|---|---|---|---|---|
| `ocean_indices.csv` | `index` (`oni`, `pdo`, `amo`), `year`, `month`, `value` | ONI: NOAA Climate Prediction Center (`cpc.ncep.noaa.gov/data/indices/oni.ascii.txt`); PDO: University of Washington JISAO (`research.jisao.washington.edu/pdo/PDO.latest`, Mantua et al. 1997); AMO (unsmoothed): NOAA PSL (`esrl.noaa.gov/psd/data/correlation/amon.us.data`) | ONI, AMO: U.S. government work; PDO: freely available, cite Mantua et al. (1997) | ONI to Sep–Nov 2018, PDO and AMO to Sep 2018 | `code/prepare_ocean_indices.py`: the Internet Archive's newest copy of each file captured on or before 2018-12-31 (the live files are never called), parsed and checked with `assert_no_post_cutoff()` | ONI: forecast input tested, not used; PDO, AMO: analysis only |
| `ocean_indices_sources.csv` | Original URL, archive capture time, archive URL and last month of each index | written by `code/prepare_ocean_indices.py` | — | — | provenance record | Reference |
| `city_population.csv` | Per station: state, county, metro area and incorporated place containing it, each with its 2020 Census population; `exposure_area` / `exposure_pop_2020` (metro area, or county outside metro areas) | U.S. Census Bureau Geocoder, geographies by coordinates (`geocoding.geo.census.gov`, benchmark `Public_AR_Current`, vintage `Census2020_Current`), field `POP100` | U.S. government work | 2020 Census | `code/prepare_population.py`: one query per station's latitude/longitude; 210/210 matched | Analysis only (policy idea) |
