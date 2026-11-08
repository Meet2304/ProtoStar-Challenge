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
  <img src="./assets/protostar-header-v0.1" alt="ProtoStar Challenge" width="100%">
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

<!-- TODO: fill in Hugging Face usernames and programs. -->

| Name | Hugging Face username | University, Program |
|---|---|---|
| Meet Bhatt | <@username> | Carnegie Mellon University, <program> |
| Kohav | <@username> | <University, program> |
| Ananya | <@username> | <University, program> |
| Achintya | <@username> | <University, program> |

## Method

**Current version: the challenge's starter model** (`code/starter_model.py`), a first end-to-end run of the pipeline.

- **Model:** for each metric, a `RandomForestRegressor` (100 trees, `random_state=42`) trained on all cities pooled, forecasting recursively: 2019 from 2016–2018, then each year from the three before it, using earlier forecasts.
- **Features:** the metric's own value 1, 2 and 3 years earlier (`lag_1`–`lag_3`). City-years that missed more than 10 days of data are left out of training; when one of a city's 2016–2018 values is invalid, it is replaced by the city's 1989–2018 average.
- **How we checked it:** backtest trained only on data up to 2010 and tested on 2011–2018, scored like the leaderboard (`reports/backtest_starter.csv`). Our reference baselines match the organizers' practice run (CDD MAE: 30-year average 211.7, linear trend 174.6 °F-days; published: 212 / 175).

| Backtest 2011–2018, MAE | Avg temp (°F) | Precip (in) | Hottest day (°F) | CDD (°F-days) | Score |
|---|---:|---:|---:|---:|---:|
| Last year repeated | 1.67 | 8.55 | 3.25 | 215.2 | 0.0 |
| 30-year average | 1.66 | 6.18 | 2.55 | 211.7 | 0.0 |
| Linear trend (30 years) | 1.43 | 6.53 | 2.70 | 174.6 | 5.0 |
| **Starter RF on lags** | 1.81 | 7.57 | 3.27 | 197.4 | 0.0 |

The starter beats the 30-year average on CDD only: lag features carry year-to-year noise, and a random forest cannot extrapolate a warming trend. Next steps: trend-aware and shrunk models (e.g., blend of the 30-year average and the linear trend), handling station changes with `coverage_and_stations`, and city covariates.

## AI Tools and Agents

- **Tools and models:** Claude Code (Anthropic).
- **What they did:** read the challenge rules and submission template; set up the repository layout, `AGENTS.md` / `CLAUDE.md`, the data-rule guard, `prepare_data.py`, `starter_model.py` (from the challenge's starter code) and the backtest; drafted this README and the data card. Downloaded only the challenge training dataset (data up to 2018).
- **What a person reviewed:** <TODO: e.g., all model code, the data-rule guard, the backtest results, the final numbers>
- **Agent instructions file:** [`code/AGENTS.md`](code/AGENTS.md) (also loaded through `AGENTS.md` and `CLAUDE.md` at the repository root).
- **Additional exploration:** OpenAI Codex (GPT-6.1 Terra) prepared `notebooks/Achintya/05_multi_window_validation.ipynb` to compare existing approaches across historical windows. Its code and findings await team review and do not alter the submission pipeline.

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
