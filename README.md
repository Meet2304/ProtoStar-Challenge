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


## Team

| Name | Hugging Face username | University, Program |
|---|---|---|
| Meet Bhatt | Meet2304 | Carnegie Mellon University, MSAIE-ETIM |
| Kohav Dey | DragonDare | Carnegie Mellon University, MISM |
| Ananya Gahalaut | Vyaghra | Carnegie Mellon University, MSAIE-IS |
| Achintya Gahalaut | agahalau | Carnegie Mellon University, MSAIE-IS |

## Method

**Current version: combined model** (`code/combo_model.py`). Each metric gets the simple forecast that backtests best on it:

- **Average temperature and cooling degree days:** the average of two forecasts per city: a straight line through its last 30 years (1989–2018), extended to 2019–2025, and its mean of the last 10 years (2009–2018). Both metrics drift upward, so a flat 30-year average lags; the trend line catches the drift but swings with noise near the end of its window; the 10-year mean is stable but lags. Their average was the most consistent.
- **Total precipitation and hottest day:** each city's 1989–2018 average. These are mostly year-to-year noise; no candidate beat the 30-year average on them in any window.
- **Data and validity:** `data/annual_valid.csv`, with city-years that missed more than 10 days of the data a metric uses left out (the official rule). Cities without 10 valid years in the window (not scored) fall back to the mean of the valid years they have, so all 5,880 rows are filled.
- **How we chose it:** four backtests, each trained only on data up to the end of a window and tested on the next 8 years (1987–1994, 1995–2002, 2003–2010, 2011–2018), scored like the leaderboard (`reports/backtest_combo.csv`). Candidates per metric: 30-year average, 30-year linear trend, 10-year mean, and the trend/10-year blend. Rule, fixed in advance: per metric, the highest mean skill across the windows. No 2019+ observations or leaderboard scores were used to choose. The reference baselines match the organizers' practice run (CDD MAE: 30-year average 211.7, linear trend 174.6 °F-days; published: 212 / 175).

| Backtest Score (leaderboard formula) | 1987–1994 | 1995–2002 | 2003–2010 | 2011–2018 | Mean |
|---|---:|---:|---:|---:|---:|
| 30-year average | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Linear trend (30 years) | 0.0 | 0.0 | 1.0 | 5.0 | 1.5 |
| 10-year mean | 0.5 | 3.0 | 4.2 | 5.8 | 3.4 |
| Trend/10-year blend, all metrics | 0.0 | 1.0 | 4.2 | 6.4 | 2.9 |
| **Combined model** | **1.6** | **3.8** | **4.7** | **7.4** | **4.4** |

A trend line alone wins on temperature and CDD in 2011–2018 but loses in the earlier windows: testing on several windows, not just the latest, is what changed our choice.

**Previous version: the challenge's starter model** (`code/starter_model.py`): one `RandomForestRegressor` per metric on lag-1/2/3 features, pooled across cities, forecast recursively. On the 2011–2018 backtest it was worse than the 30-year average on three of four metrics (Score 0.0; `reports/backtest_starter.csv`): lag features carry year-to-year noise, and a random forest cannot extrapolate a trend.

**Next steps:** handle station changes with `coverage_and_stations` (a step in a city's record can fake a trend), shrink each city's trend toward its region's, and add city covariates.

## AI Tools and Agents

- **Tools and models:** Claude Code (Anthropic).
- **What they did:** read the challenge rules and submission template; set up the repository layout, `code/AGENTS.md` / `CLAUDE.md`, the data-rule guard, `prepare_data.py`, `starter_model.py` (from the challenge's starter code) and the backtest; drafted this README and the data card. Downloaded only the challenge training dataset (data up to 2018).
- **Combined model:** Claude (Anthropic, via the Claude app) wrote `code/combo_model.py`, ran the four-window backtest and drafted this Method section, working only from `data/annual_valid.csv` (data up to 2018).
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
- **What would fix it:** list every host the dataset is served from on the dataset card, for teams running agents behind an allowlist; mention the parallel-prediction issue next to the starter code.

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

- **External data:** none.
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
