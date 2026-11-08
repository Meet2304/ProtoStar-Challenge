# 10 — Enhanced inputs: city facts and ocean indices

This is release `clean-v2`. The line's enhanced dataset moved to `enhance-v2` (pinned in `enhanced/latest.json`). The
same process as for the weather record was applied to it: profile every table and column, test for flaws, decide an
action for each with evidence, and write a model-ready version. The challenge weather data did not change (Hugging Face
revision `bbbdebd`), so `clean-v2` carries `clean-v1`'s per-origin weather tables byte for byte.

```bash
uv run python scripts/line.py pull enhance-v1 enhance-v2 clean-v1
uv run python scripts/04_clean_enhanced.py                  # -> outputs/clean_v2/ (about 1 min, mostly the weather copy)
uv run python -m unittest tests.test_cleaning_enhanced      # synthetic checks (seconds)
```

Code: [`cleaning_enhanced.py`](../src/protostar_challenge/cleaning_enhanced.py). Evidence and charts:
[`notebooks/Meet/04-enhanced-dataset.ipynb`](../notebooks/Meet/04-enhanced-dataset.ipynb).

## What enhance-v2 holds

| Part | Files | Gate-eligible | Content |
|---|---|---|---|
| City facts | `static/city_static.csv` (210 × 31) | yes, via `enhance-v1` (same bytes) | coordinates, USGS ground elevation, NOAA catalog elevation, distance to the ocean coast, ESA WorldCover 2021 class and 1 km fractions |
| Ocean indices | `ocean/origin-YYYY/{monthly,annual}.csv`, 1979–2018 | **no** | ERSSTv5 SST for Niño 3.4, a North Pacific and a North Atlantic box, absolute and as anomaly from the 30 years ending at the origin |

`enhance-v2` is exploratory because ERSSTv5 is a modern reconstruction fitted with data from after early origins.
Cutting rows at the origin does not remove that, and no cleaning can. `clean-v2` therefore contains the cleaned city
facts (built from `enhance-v1`) but not the ocean files.

## Flaws and actions

| Part | Flaw | Evidence | Action |
|---|---|---|---|
| city facts | 6 constant columns | `elevation_source`, `coast_sampling_error_bound_km`, `landcover_year`, `landcover_valid_fraction_1km`, snow/ice and moss/lichen fractions: one value for all 210 cities | **dropped** |
| city facts | resolution in degrees | 15 cities report `elevation_resolution_m` as 3.09e-5 or 9.26e-5 (1/9 and 1/3 arc-second in degrees) | **converted** to metres (3.44 m, 10.31 m) |
| city facts | catalog elevation disagrees | 15 cities: catalog 30–44 m higher than USGS with the two coordinates within 40 m; USGS matches the airports' field elevations | **flagged** `elevation_catalog_disagrees`; use `elevation_m` (USGS) |
| city facts | challenge coordinate not at the station | Corpus Christi 11.1 km (coordinate rounded to 0.1°), Los Angeles 6.0 km, Chicago 3.9 km from the catalog station; next largest 2.8 km | **flagged** `location_uncertain`; the facts there may describe the wrong point. Recomputing them belongs to the enhance station |
| city facts | class codes, compositional fractions | codes 10…100 are categories; the fractions sum to 1 | listed in the manifest: `categorical_columns`, `compositional_columns` |
| city facts | raw acquisition date in mixed formats | `9/1/2017`, epoch ms, `0/6/2020` | raw string dropped; the parsed `elevation_acquisition_date` kept (19 `unknown`) |
| city facts | facts describe the 2018 station | station moves in the record (page 05) | documented: a city-level fact, not a per-year one |
| ocean | none in the values | every snapshot recomputed (below) | **unchanged**; the check ships as `checks/enhance-ocean-verification.json` |
| ocean | anomalies include warming | +0.08 to +0.10 °C/decade in all three boxes; North Pacific and North Atlantic anomalies correlate 0.72 | documented |
| ocean | extremes | robust \|z\| > 4 only in Aug–Sep 2014 and Nov 2018 (North Pacific marine heat wave) | **kept**: real events |

Thresholds are constants at the top of `cleaning_enhanced.py`: 30 m for the elevation flag (about 0.2 °C at 6.5 °C/km)
and 3 km for the location flag (beyond the ~1.9 km error of an arc-minute coordinate).

## Ocean verification

`verify_ocean` fails the build on any of these, for each of the 40 origins:

- a row or date after the origin; a missing, duplicated or empty month;
- absolute SST that differs from the 2018 snapshot for the same month;
- a box mean that differs from the stored ERSSTv5 grid by more than 1e-4 °C (largest found: 3.4e-5 °C). The Niño 3.4 grid
  snapshot has one extra latitude row (6° S) outside the box, which the index correctly leaves out;
- an anomaly that is not SST minus the monthly normal of the 30 years ending at the origin;
- an annual value that is not the mean of its 12 months.

## Outputs (`clean-v2`)

| Path | Content |
|---|---|
| `static/city_static_clean.csv` | 210 rows, the original columns minus the dropped ones, plus `elevation_catalog_diff_m`, `elevation_catalog_disagrees`, `location_uncertain` |
| `static/qc/static_issues.csv` | one row per change or flag: `city_id`, `issue`, `column`, `value`, `new_value`, `action` |
| `static/manifest.json` | input and code hashes, constants, summary, `model_columns`, `categorical_columns`, `compositional_columns` |
| `checks/enhance-ocean-verification.json` | the ocean checks, grid errors, extremes, trends and correlations |
| `origin-YYYY/` | `clean-v1`'s per-origin weather tables, unchanged (page 09) |

Cite the city table in an asset manifest as `data/processed/releases/clean-v2/static/city_static_clean.csv` with kind
`static_context`. Use the ocean indices from `data/processed/releases/enhance-v2/ocean/origin-<origin>/` for that origin
only; runs using them are exploratory.

## What the notebook found for modelling

Comparisons with the targets use years up to 2004 only.

- Latitude and elevation explain 88 % of the differences in mean temperature between cities. The facts describe a city's
  level, which a skill score against the city's own 30-year mean does not reward directly. Their use is to share
  information across similar cities.
- The previous year's ocean anomalies correlate 0.2–0.3 with the next year's national-average departures, which is
  borderline. Above last year's weather: Niño 3.4 for CDD (+0.28) and the hottest day (+0.23), and North Atlantic for
  precipitation (−0.31). Average temperature is better predicted by last year's weather (+0.44).
