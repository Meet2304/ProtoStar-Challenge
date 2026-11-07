# AGENTS.md

Instructions for AI agents (Codex, Cursor, Claude Code, Copilot, ...) working in this repository.

**Read [`code/AGENTS.md`](code/AGENTS.md) first and follow it in every task.** It holds the
challenge's data rule and the team's working rules. It lives in `code/` because the challenge
asks submissions to keep the agent instructions file there. If anything here conflicts with it,
`code/AGENTS.md` wins.

The one rule that can disqualify us: **no weather data dated after 2018-12-31, and no
2019–2025 data that responds to the weather, may feed a forecast.** This includes numbers an
LLM recalls or a web search returns. If a task seems to need it, stop and ask a team member.

## Project

Team Placeholders' entry to the [Protostar Challenge: Climate Risk Intelligence](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence)
(CMU, Fall 2026): forecast four yearly metrics for 210 U.S. cities for 2019–2025, using only data up to 2018.
Submission deadline: **Tuesday, November 10, 2026, 11:59 PM ET**.

| Metric code | `annual_indices` column | Unit |
|---|---|---|
| `annual_avg_temp` | `Avg Temp` | °F |
| `total_precip_in` | `ToT Prcp` | inches |
| `hottest_day_temp` | `Max Tmax` | °F |
| `cdd` | `cdd` | °F-days |

Scoring: MAE per metric, skill = 1 − MAE / MAE(30-year average), Score = 100 × max(0, mean skill).
Live score = 2019–2021, final score = 2022–2025.

## Layout

This repository has the same layout as the Hugging Face submission dataset, so it can be uploaded as is.

| Path | Content |
|---|---|
| `README.md` | The submission write-up (method, AI tools, policy idea, feedback). Has Hugging Face dataset YAML front matter: keep it. |
| `submission.csv` | Forecasts, 5,880 rows: `city_id,year,metric,predicted_value`. Generated: do not edit by hand. |
| `code/` | Everything that produces `submission.csv`, plus `AGENTS.md` and `check_submission.py`. |
| `data/` | Curated data the code reads; every file is described in `data/README.md`. |
| `reports/` | Backtest results written by the code. |
| `notebooks/<Name>/` | Personal exploration space for Meet, Kohav, Ananya and Achintya. |
| `src/protostar_challenge/` | Package stub from `uv init`; not used by the pipeline. |

## Commands

Python 3.13 with [uv](https://docs.astral.sh/uv/) (`pyproject.toml`, `uv.lock`); `code/requirements.txt` for pip users.

```bash
uv sync                                    # install dependencies
uv run python code/prepare_data.py         # download training data -> data/
uv run python code/starter_model.py        # backtest 2011-2018, then write submission.csv
uv run python code/check_submission.py     # same checks the challenge Space runs
```

The training dataset (`CMU-OSPO/protostar-climate-train-to-2018`) is public; no token is needed.

## Conventions

- Validate models with backtests ending at or before 2018 (default: train ≤ 2010, test 2011–2018),
  using the official validity rule: a year with more than 10 missing days of the data a metric uses
  is not scored, and a city needs ≥ 10 valid years in the 30-year window for a baseline.
- Compare every new model against the three reference baselines in `reports/` (last year repeated,
  30-year average, linear trend), per metric.
- Keep `submission.csv` reproducible: fixed seeds, deterministic code, scripts that run top to bottom.
- After changing `submission.csv`, run `code/check_submission.py`, and update the README's *Method*
  and *AI Tools and Agents* sections.
- Never commit tokens, `.env` files or credentials.
