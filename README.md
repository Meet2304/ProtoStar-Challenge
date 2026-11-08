---
pretty_name: "Protostar Climate Challenge Submission: Team Placeholders"
license: cc-by-4.0
task_categories:
- time-series-forecasting
tags:
- climate
- forecasting
- protostar-climate-2026
configs:
- config_name: default
  data_files:
  - split: test
    path: submission.csv
---

<p align="center">
  <img src="./assets/protostar-header-v0.1.png" alt="ProtoStar Challenge" width="100%">
</p>

<h1 align="center">Team Placeholders: Protostar Climate Challenge Submission</h1>

<p align="center"><i>Climate Risk Intelligence | Team Placeholders</i></p>

**Negative result:** no

Forecasts of yearly weather for 210 U.S. cities, 2019–2025, made as of the end of 2018 (no data that reveals 2019–2025 weather), for the [Protostar Challenge: Climate Risk Intelligence](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence).

<p align="justify">
The ProtoStar Challenge, hosted by Carnegie Mellon University, explores how machine learning can improve climate forecasting using historical weather records from 210 U.S. cities. The challenge focuses on predicting four key climate indicators: average annual temperature, total annual precipitation, maximum daily temperature, and cooling degree days, translating historical climate data into actionable insights for urban resilience and adaptation.

As Team Placeholders, we are developing an iterative, data-driven forecasting pipeline that combines statistical baselines with machine learning models to improve prediction accuracy. Our workflow is built around Python and uv for reproducible dependency management, Google Colab for collaborative experimentation, and MLflow for tracking model performance, parameters, and artifacts. Through continuous evaluation and refinement, we aim to develop reliable city-level forecasts and translate our findings into a practical climate adaptation recommendation.
</p>


## Team

| Name | Hugging Face username | University, Program |
|---|---|---|
| Meet Bhatt | Meet2304 | Carnegie Mellon University, MSAIE-ETIM |
| Kohav Dey | DragonDare | Carnegie Mellon University, MISM |
| Ananya Gahalaut | Vyaghra | Carnegie Mellon University, MSAIE-IS |
| Achintya Gahalaut | agahalau | Carnegie Mellon University, MSAIE-IS |

## Method

**Current version: combined model v3** (`code/combo_model.py`). Each metric gets the simple forecast that backtests best on it:

- **Average temperature:** each city's current level (its mean over 2009–2018) carried forward at the warming rate shared by all cities (the median, across the 210 cities, of each city's 1989–2018 least-squares slope). One city's own slope is noisy: a few unusual years near the end of its window tilt it. All cities share most of the warming, so the median slope is a much steadier estimate, and the recent mean anchors each city's level.
- **Cooling degree days:** the same current level, growing at the rate shared by all cities **in percent** (the median, across cities with real cooling need, of slope ÷ 30-year mean, about 1% a year). CDD starts at zero, so the same warming adds many more degree-days in Miami than on the foggy Pacific coast: v2 gave every city the same absolute rise (Eureka, 3 CDD a year, was forecast at about 50), v3 scales the rise to each city's level.
- **Total precipitation and hottest day:** each city's 1989–2018 average. These are mostly year-to-year noise; no candidate beat the 30-year average on them in any window, and 20- or 50-year averages did no better.
- **Data and validity:** `data/annual_valid.csv` (records up to 2018), with city-years that missed more than 10 days of the data a metric uses left out (the official rule), and city coordinates from `data/city_info.csv`. Cities without enough valid years (not scored) fall back to the mean of the valid years they have, so all 5,880 rows are filled. CDD and precipitation forecasts are floored at 0.
- **How we chose it:** four backtests, each trained only on data up to the end of a window and tested on the next 8 years (1987–1994, 1995–2002, 2003–2010, 2011–2018), scored like the leaderboard (`reports/backtest_combo.csv`). Candidates per metric: 30-year average, 30-year linear trend, 10-year mean, trend/10-year blend, recent level + all-city slope, recent level + slope of the 8 nearest cities, and (precipitation and CDD only) recent level + all-city percent slope. Rule, fixed in advance: per metric, the highest mean skill across the windows; the script re-checks the choice on every run. No 2019+ observations or leaderboard scores were used to choose. The reference baselines match the organizers' practice run (CDD MAE: 30-year average 211.7, linear trend 174.6 °F-days; published: 212 / 175).

| Backtest Score (leaderboard formula) | 1987–1994 | 1995–2002 | 2003–2010 | 2011–2018 | Mean |
|---|---:|---:|---:|---:|---:|
| 30-year average | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Linear trend (30 years) | 0.0 | 0.0 | 1.0 | 5.0 | 1.5 |
| 10-year mean | 0.5 | 3.0 | 4.2 | 5.8 | 3.4 |
| Combined model v1 (trend/10-year blend for temperature and CDD) | 1.6 | 3.8 | 4.7 | 7.4 | 4.4 |
| Combined model v2 (recent level + all-city slope for temperature and CDD) | 2.7 | 4.7 | 4.6 | 8.9 | 5.2 |
| **Combined model v3** (v2, with the CDD rise in percent) | **2.7** | **4.4** | **4.8** | **9.8** | **5.4** |

| Mean skill over the 4 windows | Trend | 10-year mean | Blend (v1) | Level + 8-nearest slope | Level + all-city slope (v2) | Level + all-city % slope (v3) |
|---|---:|---:|---:|---:|---:|---:|
| Average temperature | 0.074 | 0.086 | 0.096 | 0.104 | **0.117** | — |
| Cooling degree days | 0.055 | 0.076 | 0.078 | 0.090 | 0.093 | **0.100** |

A city's own trend line wins on temperature and CDD in 2011–2018 but loses in the earlier windows; pooling the slope across cities keeps most of its gain without its noise. Testing on several windows, not just the latest, is what drove these choices.

**Tested and left out: an El Niño adjustment** (`USE_ENSO = False` in `code/combo_model.py`; `code/enso_test.py`). We joined NOAA's Oceanic Niño Index (ONI) to the city records by year and shifted each city's first forecast year by its El Niño sensitivity (fitted on earlier years only, averaged over the city and its 8 nearest cities). Over 38 one-year-ahead forecasts (1981–2018, `reports/enso_rolling_test.csv`) it cut first-year errors by about 1.2% for temperature and 1.5% for CDD, but it was better in only 21 and 23 of the 38 years, and it lowered the 4-window backtest Score from 5.4 to 5.1 (big misses after strong La Niña years, e.g. the forecast for 2011). By our selection rule it stays out. Details: *Data* and *Feedback to the Ecosystem*.

**Previous version: the challenge's starter model** (`code/starter_model.py`): one `RandomForestRegressor` per metric on lag-1/2/3 features, pooled across cities, forecast recursively. On the 2011–2018 backtest it was worse than the 30-year average on three of four metrics (Score 0.0; `reports/backtest_starter.csv`): lag features carry year-to-year noise, and a random forest cannot extrapolate a trend.

**Next steps:** forecast CDD from daily temperatures (warm each city's recent days by the forecast temperature rise and recount CDD), handle station changes with `coverage_and_stations` (a step in a city's record can shift its level), and add city covariates from outside data.

## AI Tools and Agents

- **Tools and models:** Claude Code (Anthropic).
- **What they did:** read the challenge rules and submission template; set up the repository layout, `code/AGENTS.md` / `CLAUDE.md`, the data-rule guard, `prepare_data.py`, `starter_model.py` (from the challenge's starter code) and the backtest; drafted this README and the data card. Downloaded only the challenge training dataset (data up to 2018).
- **Combined model:** Claude (Anthropic, via the Claude app) wrote `code/combo_model.py` (v1, v2 and v3), ran the four-window backtests and drafted this Method section, working only from `data/annual_valid.csv` and `data/city_info.csv` (data up to 2018). It also wrote `code/prepare_ocean_indices.py` (archived NOAA/JISAO index downloads capped at 2018), the El Niño adjustment and `code/enso_test.py`, and drafted the ocean-index parts of this README, and wrote `code/prepare_population.py` and `code/policy_cdd.py` (Census population by station, policy analysis); to find archive copies it queried the Internet Archive's index for captures on or before 2018-12-31 only.
- **What a person reviewed:** <TODO: e.g., all model code, the data-rule guard, the backtest results, the final numbers>
- **Agent instructions file:** [`code/AGENTS.md`](code/AGENTS.md) (the single agent instructions file; Claude Code loads it through `CLAUDE.md` at the repository root).

## Policy Idea

<!-- TODO. Half a page at most. -->

1. **What risk did we notice?** <...>
2. **What would we do about it, and who would do it?** <...>
3. **Why is it sustainable?** <...>

## Feedback to the Ecosystem

<!-- TODO: keep adding as we go. -->

- **What got in the way:** the challenge dataset is served from a CDN host (`us.aws.cdn.hf.co`) separate from `huggingface.co`, so sandboxed agent environments with a host allowlist failed to read the parquet files until that host was added. scikit-learn's `RandomForestRegressor` with `n_jobs=-1` gives slightly different predictions between runs (summing trees in thread order), which breaks byte-for-byte reproducibility unless prediction runs on a single thread.
- **Getting other agencies' data as of 2018:** the climate indices we needed (NOAA CPC's ONI, JISAO's PDO, NOAA PSL's AMO) are published only as files that run to the present, with no way to request data up to a date, so a team that follows the data rule strictly cannot call them. We used Internet Archive copies captured before 2019, which worked, but the archive coverage is uneven: NOAA PSL's own ONI and PDO files were last captured in early 2017, so we had to find other publishers (CPC, JISAO) with captures from December 2018. The indices also come in different layouts (PSL year-by-12-months tables with a missing-value line, CPC 3-month seasons, JISAO with `**` markers and partial years), so each needs its own parser.
- **Joining people to stations:** the stations are identified only by GHCN ID, name and coordinates, with no state or Census code, and most sit at airports outside the city they are named for (55 of 210), so a city-name join fails; we had to geocode the coordinates. The Census Bureau's data API now requires a key, while its Geocoder (no key) returns 2020 populations directly, which is what made this quick.
- **What would fix it:** list every host the dataset is served from on the dataset card, for teams running agents behind an allowlist; mention the parallel-prediction issue next to the starter code. For outside data: a snapshot of common climate indices (ONI, PDO, AMO, NAO) cut at 2018-12-31 published next to the challenge dataset, in one tidy layout (`index, year, month, value`), would let teams test them without archive workarounds; or agencies could offer an end-date parameter on index downloads. Adding each city's state and Census place/metro code to `city_info` would let teams join social and economic data without geocoding.

## Files

| File | Content |
|---|---|
| `submission.csv` | Our forecasts: `city_id,year,metric,predicted_value` |
| `code/` | Scripts that rebuild `submission.csv` (see [`code/README.md`](code/README.md)) and the agent instructions |
| `data/` | The curated data our code uses; described in its data card, [`data/README.md`](data/README.md) |
| `reports/` | Backtest results (2011–2018) written by the code |
| `notebooks/` | Each team member's exploration notebooks (not needed to reproduce `submission.csv`) |

**Metric codes:** `annual_avg_temp` (°F), `total_precip_in` (inches), `hottest_day_temp` (°F), `cdd` (°F-days).
**City IDs:** the station `ID` from the `city_info` table of the training dataset (e.g., Pittsburgh = `USW00094823`).

## How to Reproduce

1. Install Python 3.13 and the packages: `pip install -r code/requirements.txt` (or `uv sync` and prefix the commands below with `uv run`).
2. `python code/prepare_data.py`: downloads the training data and writes `data/` (optional: `data/` is included).
3. `python code/starter_model.py`: writes `reports/backtest_starter.csv` and `submission.csv` (about 15 seconds).
4. `python code/check_submission.py`: checks `submission.csv` like the challenge Space.

## Data

- **External data** (all capped at 2018; details and provenance in [`data/README.md`](data/README.md) and `data/ocean_indices_sources.csv`):
  - **Oceanic Niño Index (ONI)**, NOAA Climate Prediction Center (`cpc.ncep.noaa.gov/data/indices/oni.ascii.txt`). *Forecast input candidate, tested and not used in the submitted forecasts* (see Method).
  - **Pacific Decadal Oscillation (PDO)**, University of Washington JISAO (`research.jisao.washington.edu/pdo/PDO.latest`; Mantua et al., 1997). *Analysis only.*
  - **Atlantic Multidecadal Oscillation (AMO, unsmoothed)**, NOAA Physical Sciences Laboratory (`esrl.noaa.gov/psd/data/correlation/amon.us.data`, now `psl.noaa.gov`; Enfield et al., 2001). *Analysis only.*
  - **2020 Census population** around each station (county, metro area and city), U.S. Census Bureau Geocoder (`geocoding.geo.census.gov`, vintage `Census2020_Current`), via `code/prepare_population.py` → `data/city_population.csv`. *Analysis only (policy idea).* Joined to the challenge data by each station's latitude/longitude: all 210 stations matched; 155 lie inside city limits, the rest (mostly airports) just outside, so we use the metro area (165 stations) or, outside metro areas, the county (45). Ten areas hold more than one station; area totals count each area once.
  - **How we kept them inside the data rule:** the live files run to the present and cannot be cut at 2018, so `code/prepare_ocean_indices.py` never calls them. It downloads the Internet Archive's newest copy of each file captured on or before 2018-12-31, so no 2019+ value can be in the download, and checks every row with `assert_no_post_cutoff()`. This also means we use the values as published at the time, before later revisions.
  - **How they join the challenge data:** by year. The indices are one monthly series for a whole ocean, the challenge data one value per city and year, so we summarise each index at the end of a year (ONI: mean of its Aug–Oct and Sep–Nov seasons) and relate it to each city's next year.
  - **What we found (1951–2015, next-year departure of each city from its previous 10 years):** the year-end ONI relates to the next year's temperature in northern cities (median correlation about +0.27; El Niño winters are mild in the northern U.S.) and weakly to precipitation in southern cities (about +0.14); the PDO relates weakly (about +0.18 for northern temperature, +0.13 for southern precipitation), and the AMO hardly at all (|median r| ≤ 0.11). Index values are the year-end mean of months 9–10 (for ONI, the Aug–Oct and Sep–Nov seasons).
- **Curated data:** see the data card in [`data/README.md`](data/README.md) (each file, its source, license and preparation).

Training data: [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), from:

- Lai, Y., & Dzombak, D. A. (2019). Compiled historical daily temperature and precipitation data for selected 210 U.S. cities. Carnegie Mellon University. https://doi.org/10.1184/R1/7890488.v6
- Lai, Y., & Dzombak, D. A. (2019). *Use of historical data to assess regional climate change.* Journal of Climate, 32(14), 4299–4320. https://doi.org/10.1175/JCLI-D-18-0630.1

## Submitting (team checklist)

<!-- Delete this section before the final submission if you like; it is for the team. -->

1. Create a public Hugging Face dataset for the team (e.g., `<hf-user>/protostar-placeholders`).
2. Upload this repository's submission files: `hf upload <hf-user>/<dataset> . --repo-type dataset --include "README.md" "submission.csv" "code/*" "data/*" "reports/*" "assets/*"`.
3. A registered team member pastes the dataset link in the **Submission** tab of the challenge Space (up to 10 per day; the latest counts).
4. Record the dataset commit that was submitted, and keep `submission.csv` here in sync with it.

<p align="center"><i>Made on Earth, by Humans.</i></p>
