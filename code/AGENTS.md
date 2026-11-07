<!-- Claude Code users: copy this file to CLAUDE.md, or put @code/AGENTS.md in your CLAUDE.md. -->

# Instructions for AI agents

This repository is a submission to the Protostar Challenge: Climate Risk Intelligence. Follow these rules in every task.

## Data rule (never break it)

- Forecasts are made as of 2018-12-31. Weather data used as a forecast input must be dated on or before 2018-12-31.
- Do not download, query or use weather observations dated 2019-01-01 or later, from any source (e.g., NOAA, Meteostat, Open-Meteo, ERA5, weather websites), nor 2019–2025 data driven by the weather (e.g., streamflow, air quality, energy demand).
- Set an end date of 2018-12-31 or earlier on every weather download.
- Never write `predicted_value` from memory or from a web search: every value must come from the code's model.
- If a task seems to need weather data after 2018, stop and ask a team member.
- Data that does not respond to the weather (e.g., census, land cover, elevation) may come from any year; record its source in `data/README.md`.

## Working rules

- Training data: `CMU-OSPO/protostar-climate-train-to-2018` (Hugging Face).
- Code goes in `code/`, curated data in `data/` (described in `data/README.md`).
- After changing `submission.csv`, run `python code/check_submission.py`.
- The code must reproduce `submission.csv` without an agent (scripts or notebooks that run top to bottom).
- Never write tokens or credentials into any file.
- Summarize what you did in the README's "AI Tools and Agents" section.

## Team notes

Team Placeholders (Meet, Kohav, Ananya, Achintya). These add to the rules above; they never relax them.

- **Guard every weather input in code.** Call `assert_no_post_cutoff()` from `code/common.py` on every
  weather table before it is used. It raises `DataRuleViolation` if any row is dated after 2018. Do not
  remove, weaken or bypass it.
- **No model selection on 2019–2025.** Choose and tune models only with backtests on years up to 2018
  (e.g., train up to 2010, test 2011–2018, as `code/starter_model.py` does). Scoring variants against
  2019–2025 observations counts as using the answers, even if they never enter the model.
- **The live leaderboard is not a tuning signal.** Do not submit variants just to compare their live
  scores and keep the best one.
- **Pretrained models:** only if their training data stops before 2019. Record the name, version and
  training-data cutoff in the README. If the cutoff is unknown, do not use it.
- **Downloads:** say what you downloaded, from where and up to which date. Put external sources in
  `data/README.md` and the README's *Data* section, marked "Forecast input" or "Analysis only".
- **Reproducibility:** `python code/prepare_data.py && python code/starter_model.py` (or the current
  pipeline) must rebuild `submission.csv` byte for byte. Fix seeds; avoid nondeterministic parallel sums.
- **Exploration** happens in `notebooks/<Name>/`; the data rule applies there too. Anything that feeds
  the submission must end up as a script in `code/`.
- **Submitting** (uploading to Hugging Face, pressing *Submit* in the Space) is done by a team member,
  not an agent, unless a person explicitly asks and provides a fine-grained token scoped to the team's
  submission dataset. Never write a token into any file.
