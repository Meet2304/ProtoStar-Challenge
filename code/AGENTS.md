<!-- The single agent instructions file for this repository (Codex, Cursor, Claude Code, Copilot, ...).
     It lives in code/ because the challenge asks submissions to keep the agent instructions file there.
     Claude Code loads it through the root CLAUDE.md (`@code/AGENTS.md`). Do not create other copies. -->

# Instructions for AI agents

This repository is Team Placeholders' submission to the [Protostar Challenge: Climate Risk Intelligence](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence)
(CMU, Fall 2026): forecast four yearly metrics for 210 U.S. cities for 2019–2025, as of 2018-12-31.
Submission deadline: **Tuesday, November 10, 2026, 11:59 PM ET**. Follow these rules in every task,
including exploration in `notebooks/`.

## Data rule (never break it)

The one rule that can disqualify us: **no weather data dated after 2018-12-31, and no 2019–2025 data
that responds to the weather, may feed a forecast.** This includes numbers an LLM recalls or a web search returns.

- Forecasts are made as of 2018-12-31. Weather data used as a forecast input must be dated on or before 2018-12-31.
- Do not download, query or use weather observations dated 2019-01-01 or later, from any source (e.g., NOAA,
  Meteostat, Open-Meteo, ERA5, weather websites), nor 2019–2025 data driven by the weather (e.g., streamflow,
  air quality, energy demand).
- Set an end date of 2018-12-31 or earlier on every weather download. If an API cannot be limited that way, do not call it.
- Never write `predicted_value` from memory or from a web search: every value must come from the code's model.
- Do not use pretrained models whose training data reaches 2019 or later, or whose cutoff is unknown.
- If a task seems to need weather data after 2018, stop and ask a team member.
- Data that does not respond to the weather (e.g., census, land cover, elevation) may come from any year;
  record its source in `data/README.md`.

## Training data

[`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018)
(public, no token, CC0; cite the Lai & Dzombak records listed in its card). It is the KiltHub records
cut at 2018-12-31, with a yearly `cdd` column added. The revision is pinned in
`src/protostar_challenge/config.py` (`DATASET_REVISION`) and `code/common.py` (`TRAIN_REVISION`): keep them equal.

| Subset | Rows | Content |
|---|---:|---|
| `daily` | 10,245,754 | One row per city and day, 1851-01-01 to 2018-12-31: `tmax`, `tmin` (°F), `prcp` (inches). |
| `annual_indices` | 28,052 | One row per city and year: averages, totals, extremes, `cdd`, and `Miss Tmax` / `Miss Tmin` / `Miss Prcp`. |
| `city_info` | 461 | Stations combined into each city's record and the period used from each. |
| `coverage_and_stations` | 28,052 | Per city-year: `Days Both` (days with Tmax and Tmin, the days `cdd` uses), `Station Segment`, `Station Changes`. |

All subsets share the station `ID` (the submission's `city_id`, e.g., Pittsburgh = `USW00094823`) and the city `Name`.

| Metric code | `annual_indices` column | Unit |
|---|---|---|
| `annual_avg_temp` | `Avg Temp` | °F |
| `total_precip_in` | `ToT Prcp` | inches |
| `hottest_day_temp` | `Max Tmax` | °F |
| `cdd` | `cdd` | °F-days |

Things to know from the dataset card:

- **Validity:** a city-year counts only if no more than 10 days are missing for each variable the metric uses
  (Tmax and Tmin for the temperature metrics and CDD, precipitation for total precipitation); the 2019–2025
  answers are computed the same way. `cdd` uses only days with both values: in 18 city-years of 1989–2018 more
  than 10 days lack one of them although Tmax and Tmin each pass. Check `Days Both` for CDD.
- **Missing days** make totals (`ToT Prcp`, `cdd`) read low.
- **Station changes:** a record can jump when the station changes (32 changes in 1989–2018; none listed in
  2019–2023). A trend fitted across a change mixes two stations; use `coverage_and_stations` to spot them.
- **Precision:** most years are in whole °F; 2018 values partly have one decimal. No humidity, so no heat indices.

## Scoring

MAE per metric; skill = 1 − MAE / MAE(30-year average); Score = 100 × max(0, mean skill).
Live score = 2019–2021, final score = 2022–2025. A city needs ≥ 10 valid years in the 30-year window for the baseline.

## Layout

This repository has the same layout as the Hugging Face submission dataset, so it can be uploaded as is.

| Path | Content |
|---|---|
| `README.md` | The submission write-up (method, AI tools, policy idea, feedback). Contains the Hugging Face dataset YAML block: keep it, and do not move it without asking the team. |
| `CLAUDE.md` | Claude Code entry point: imports this file and adds Claude-specific notes. |
| `submission.csv` | Forecasts, 5,880 rows: `city_id,year,metric,predicted_value`. Generated: do not edit by hand. |
| `code/` | Everything that produces `submission.csv`, this `AGENTS.md`, and `check_submission.py`. `combo_model.py` is the current model; `starter_model.py` is the challenge's starter, kept for comparison. |
| `data/` | Curated data the code reads (tracked); every file is described in `data/README.md`. `data/raw/` (full download) and `data/processed/` are git-ignored. |
| `reports/` | Backtest results written by the code. |
| `notebooks/<Name>/` | Personal exploration space for Meet, Kohav, Ananya and Achintya (see `notebooks/README.md`). |
| `scripts/01_download_data.py` | Downloads all four subsets to `data/raw/` at the pinned revision. |
| `src/protostar_challenge/config.py` | Shared paths, pinned dataset revision and cutoff for scripts and notebooks. |
| `requirements.txt` | Full `uv export` of `uv.lock`, for Colab. `code/requirements.txt` is the minimal pinned set the submission code needs. |

## Commands

This repo uses uv (Python 3.13): run code with `uv run python ...` and add packages with `uv add ...`
(then regenerate `requirements.txt` with the `uv export` command in its header).

```bash
uv sync                                    # install dependencies
uv run python scripts/01_download_data.py  # optional: full raw dataset -> data/raw/ (for exploration)
uv run python code/prepare_data.py         # curated training data -> data/
uv run python code/combo_model.py          # current model: 4-window backtest -> reports/, then submission.csv
uv run python code/check_submission.py     # same checks the challenge Space runs
```

## Working rules

- **Guard every weather input in code.** Call `assert_no_post_cutoff()` from `code/common.py` on every weather
  table before it is used. It raises `DataRuleViolation` if any row is dated after 2018. Do not remove, weaken or bypass it.
- **No model selection on 2019–2025.** Choose and tune models only with backtests on years up to 2018
  (`code/combo_model.py` tests on 1987–1994, 1995–2002, 2003–2010 and 2011–2018). Scoring variants against
  2019–2025 observations counts as using the answers, even if they never enter the model.
- **Compare with the baselines.** Score every new model per metric against the reference forecasts
  (last year repeated, 30-year average, 30-year linear trend) with the official validity rule.
- **The live leaderboard is not a tuning signal.** Do not submit variants just to compare their live scores.
- **Pretrained models:** record the name, version and training-data cutoff in the README.
- **Downloads:** say what you downloaded, from where and up to which date. Put external sources in
  `data/README.md` and the README's *Data* section, marked "Forecast input" or "Analysis only".
- **Reproducibility:** `python code/prepare_data.py && python code/combo_model.py` (or the current pipeline)
  must rebuild `submission.csv` byte for byte, without an agent. Fix seeds; avoid nondeterministic parallel sums.
- **Exploration** happens in `notebooks/<Name>/`; the data rule applies there too. Anything that feeds the
  submission must end up as a script in `code/`.
- **After changing `submission.csv`**, run `python code/check_submission.py`, and update the README's
  *Method* and *AI Tools and Agents* sections. Commit code, data cards and reports together with the
  `submission.csv` they produce.
- **Submitting** (uploading to Hugging Face, pressing *Submit* in the Space) is done by a team member, not an
  agent, unless a person explicitly asks and provides a fine-grained token scoped to the team's submission dataset.
- Never write tokens, `.env` files or credentials into any file.
