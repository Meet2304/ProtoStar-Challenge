# Code

These scripts rebuild `submission.csv` from scratch, without an agent. Run them from the repository root:

```bash
pip install -r code/requirements.txt      # or: uv sync, then prefix commands with `uv run`
python code/prepare_data.py               # training data -> data/annual_valid.csv, data/city_info.csv
python code/combo_model.py                # current model: 4-window backtest -> reports/, forecasts -> submission.csv
python code/check_submission.py           # same checks as the challenge Space
```

| File | What it does |
|---|---|
| `common.py` | Shared paths, metric definitions, the 10-day validity rule, and `assert_no_post_cutoff()`, the data-rule guard that stops the pipeline if any weather input is dated after 2018. |
| `prepare_data.py` | Downloads `annual_indices` and `city_info` from [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018) (public, no token) and writes the curated files in `data/`. |
| `combo_model.py` | **Current model.** Per metric: 30-year average for precipitation and hottest day; each city's 2009–2018 mean carried forward at the median warming rate across all cities for temperature and CDD. Backtests six candidate forecasts on four windows (test 1987–1994 … 2011–2018), checks the per-metric choice against the selection rule, writes `reports/backtest_combo.csv` and `submission.csv`. Runs in a few seconds. |
| `starter_model.py` | The challenge's prescribed starter: one `RandomForestRegressor` per metric on lag-1/2/3 features, pooled across cities, forecast recursively for 2019–2025. Also backtests the model and the three reference baselines (train ≤ 2010, test 2011–2018) and writes `reports/backtest_starter.csv`. Runs in about 15 seconds. |
| `check_submission.py` | From the challenge template, unchanged: checks `submission.csv` like the Space does. |
| `requirements.txt` | Pinned packages (the same versions as `uv.lock`). |
| `AGENTS.md` | The repository's single instructions file for AI agents: the data rule, training-data notes and team rules (the root `CLAUDE.md` imports it). |

`submission.csv` is byte-for-byte reproducible: fixed seeds, and single-threaded prediction
(parallel prediction sums the trees in thread order and changes the last digits).
