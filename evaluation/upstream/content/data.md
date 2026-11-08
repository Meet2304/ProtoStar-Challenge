## Data

One dataset, with daily and yearly weather for 210 U.S. cities up to **2018-12-31**:

**[CMU-OSPO/protostar-climate-train-to-2018](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018)**

| Subset | Rows | Content |
|---|---:|---|
| `daily` | 10,245,754 | Daily `tmax`, `tmin` (°F) and `prcp` (inches) per city, 1851–2018 |
| `annual_indices` | 28,052 | One row per city and year: averages, totals, extremes, `cdd`, and `Miss *` counts of missing days |
| `city_info` | 461 | Station `ID`, city `Name`, latitude, longitude, and the stations joined for each city |
| `coverage_and_stations` | 28,052 | One row per city and year: how many days have both temperatures (the days `cdd` uses), and which station the record comes from |

```python
from datasets import load_dataset

REPO = "CMU-OSPO/protostar-climate-train-to-2018"
daily = load_dataset(REPO, "daily", split="train").to_pandas()
annual = load_dataset(REPO, "annual_indices", split="train").to_pandas()
cities = load_dataset(REPO, "city_info", split="train").to_pandas()
coverage = load_dataset(REPO, "coverage_and_stations", split="train").to_pandas()
```

**Metric codes and their columns in `annual_indices`:** `annual_avg_temp` = `Avg Temp`, `total_precip_in` = `ToT Prcp`, `hottest_day_temp` = `Max Tmax`, `cdd` = `cdd`.

The dataset page describes each subset and its columns. Its *Limitations* section explains missing days and station changes, which can shift a city's record; `coverage_and_stations` marks both.

### Other data is welcome

The challenge dataset is a starting point, not a limit. You can combine it with data from other sources:

- ✅ **As forecast inputs:** weather records **up to 2018-12-31** from any source (e.g., NOAA, ERA5), and data from **any year** that does not respond to the year's weather (e.g., census, land cover, tree canopy, elevation, building stock).
- ❌ **Not as forecast inputs:** any **2019–2025** data that responds to the weather: weather observations from any source, and anything driven by them (e.g., streamflow, air quality, energy demand). It is still welcome in your analysis and policy idea.

The test is not whether the data is *correlated* with the weather (latitude and land cover are, and they are fine), but whether a 2019–2025 value would have been different in a hotter or wetter year. Examples are in the FAQ below. List every external source in your README.

## How to submit

**Your submission is a Hugging Face dataset.** Build it like any dataset on the Hub, then submit its link.

1. **Create your dataset** by copying the template [CMU-OSPO/protostar-climate-submission-template](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-submission-template) into your Hugging Face account.
2. **Add your forecasts** in the template's `submission.csv`: the city, year and metric of all 5,880 rows are already filled in; fill in `predicted_value` (format below). `code/check_submission.py` in the template runs the same checks as the Space (install its packages with `pip install -r code/requirements.txt`).
3. **Write your `README.md`:** your method in a few lines, your **Policy Idea** and your **Feedback to the Ecosystem** (see the *About* tab). It is your write-up.
4. **Add your code** in a `code/` folder (datasets are git repos, so code uploads like any other file).
5. **Add your curated data** in a `data/` folder: the data your code reads, as you prepared it (e.g., your processed version of the challenge data, and any external data you added). Your code and data together should reproduce your `submission.csv`. Describe each file (source, license, preparation) in the data card, `data/README.md` (the template includes one). If a source is too large or its license does not allow copying, list it there with its exact version and how to download it.
6. **Make the dataset public** and paste its link, e.g. `https://huggingface.co/datasets/your-name/your-dataset`, in the **Submission** tab.

When you submit, the Space checks that the dataset is public, contains `submission.csv` and `README.md`, and that `submission.csv` has a number for every one of the 5,880 rows; it then records its current version (commit). You can update your dataset and submit again until the deadline, **up to 10 times a day** per team; **the version recorded at your latest submission counts**, so submit again after every change you want judged. Wait a few hours for your submission to appear on the leaderboard.

### Forecast file format

`submission.csv`, 4 columns, one row per city × year × metric:

```csv
city_id,year,metric,predicted_value
USW00094823,2019,annual_avg_temp,52.4
USW00094823,2019,total_precip_in,38.2
USW00094823,2019,hottest_day_temp,91.0
USW00094823,2019,cdd,845.0
USW00094823,2020,annual_avg_temp,53.1
...
USW00024233,2025,cdd,310.0
```

- **`city_id`:** the station `ID` from `city_info` (e.g., Pittsburgh = `USW00094823`, Seattle = `USW00024233`).
- **Total rows:** 210 cities × 7 years × 4 metrics = **5,880 rows**.

## Starter code

A working model in under 30 lines (scikit-learn):

```python
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.ensemble import RandomForestRegressor

# 1. Load the yearly data (up to 2018)
ds = load_dataset("CMU-OSPO/protostar-climate-train-to-2018", "annual_indices")
train_df = ds["train"].to_pandas()

# Metric codes and their columns in annual_indices:
# annual_avg_temp = "Avg Temp", total_precip_in = "ToT Prcp",
# hottest_day_temp = "Max Tmax", cdd = "cdd"
target = "Avg Temp"

# 2. Build lag features: each city's value 1, 2 and 3 rows (years) earlier
train_df = train_df.sort_values(["ID", "Year"])
for k in (1, 2, 3):
    train_df[f"lag_{k}"] = train_df.groupby("ID")[target].shift(k)
features = ["lag_1", "lag_2", "lag_3"]
fit_df = train_df.dropna(subset=features + [target])

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(fit_df[features], fit_df[target])

# 3. Predict 2019-2025 iteratively
# ... generate predictions and save to submission.csv

# 4. Check your error locally against validation years:
def compute_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))
```

## FAQ

**Can I use data other than the challenge dataset?**
Yes, with one line drawn by *use*. Your **forecasts** may use weather records up to 2018-12-31, and data from any year that does not respond to the weather. After 2018, anything that responds to the weather gives away the answers, even when it is not weather data. The same data is welcome in your **analysis and policy idea**, where real impacts are exactly what a policy targets.

| Data (2019–2025) | Why it gives away the weather | Leaks | Forecast input? | Analysis / policy? |
|---|---|---|---|---|
| Temperature or rain from any source (NOAA, weather sites, ERA5 reanalysis, satellites) | It *is* the answer | All four metrics | ❌ | ✅ |
| Baseball home runs per game | Hot air is thinner, so balls fly farther (Callahan et al., 2023, *BAMS*, doi:10.1175/BAMS-D-22-0235.1) | Hottest day, CDD | ❌ | ✅ |
| River streamflow (USGS) | Rain becomes runoff within days | Precipitation | ❌ | ✅ |
| Ozone exceedance days (EPA) | Ozone forms faster in heat and sun | Hottest day, CDD | ❌ | ✅ |
| Spring leaf-out and bloom dates (USA-NPN) | Plants respond to accumulated warmth | Average temperature | ❌ | ✅ |
| West Nile virus cases (CDC) | Mosquitoes multiply faster in heat | CDD | ❌ | ✅ |
| Electricity demand (EIA), searches for "AC repair" | Air conditioning drives summer peaks | CDD | ❌ | ✅ |
| Census, land cover, tree canopy, building stock | Does not respond to the year's weather | — | ✅ | ✅ |
| Weather records **up to 2018** from other sources | Same cutoff as the challenge dataset | — | ✅ | ✅ |

When in doubt: if a 2019–2025 series would be different in a hotter or wetter year, keep it out of your forecasts. List every external source in your README.

**Can I score my own forecasts against 2019–2025 observations?**
The observations are public, so you could, but choosing a model by its score on them uses the answers, even if they never enter the model.

| Use | Allowed? | Why |
|---|---|---|
| Scoring model variants on 2019–2025 observations to choose or tune one | ❌ | Selecting by the answers uses them |
| Scoring a frozen, committed forecast afterwards, for analysis | ✅ | The forecast is fixed; declare it in your README with the commit hash |

**Can I use LLMs (e.g., ChatGPT, Claude, Copilot) or pretrained forecasting models?**
Yes, as tools, not as a source of answers. A model trained on data from after 2018 has already "seen" how 2019–2025 turned out, so the numbers it gives can carry the answers even if you never mention those years.

| Use | Forecast input? | Why |
|---|---|---|
| An LLM writes, debugs or explains your code | ✅ | The numbers still come from your data and your model |
| An LLM helps you brainstorm methods or features, or edit your README | ✅ | It shapes your method, not your values |
| An LLM gives you values (e.g., "What was Phoenix's CDD in 2021?", "Predict Pittsburgh's 2023 average temperature") | ❌ | Its training data includes the 2019–2025 weather |
| An LLM fills in or "corrects" `predicted_value` | ❌ | Same reason, even if your model produced the first draft |
| A pretrained time-series model (e.g., Chronos, TimesFM, Moirai) whose training data stops before 2019 | ✅ | Declare its name, version and training-data cutoff in your README |
| A pretrained model whose training data includes data from 2019 onward, or whose training data is unknown | ❌ | It may have learned the answers; check the model card first |
| AI weather or climate models (e.g., GraphCast, Pangu-Weather) trained on reanalysis from 2019 onward | ❌ | Reanalysis after 2018 *is* weather observations |

Using an LLM in your **analysis and policy idea** is welcome: for example, to summarize heat-adaptation literature or to explore 2019–2025 impacts. **Say in your README which AI tools you used and for what.** When unsure whether a model learned from data after 2018, keep it out of your forecasts or ask in the Space's [Community tab](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence/discussions).

## 🤖 AI agents: allowed, with guardrails

> **The use of AI agents is allowed.** Agents can write code, run experiments and help document your work. Set them up so they cannot fetch 2019–2025 weather (see below), and say how you used them in your README. See the [LLM Documentation Guide](https://guides.library.cmu.edu/LLMDocumentationGuide) (CMU Libraries) for ideas on how to document your LLM and agent use.

**Can we use AI agents (e.g., Claude Code, Codex, Cursor)?**
Yes: for data preparation, modeling, experiments, your README and your data card. The same line applies as for LLMs: an agent can build your method, but it cannot be the source of 2019–2025 values.

**How do we stop an agent from leaking 2019–2025 data?**
Agents search the web, call APIs and download files on their own, so set them up so they cannot fetch the answers:

- Put the data rule in the agent's instructions file (e.g., `AGENTS.md` or `CLAUDE.md`; the template includes one): only weather data up to 2018-12-31.
- Cap every weather download at that end date (e.g., NOAA, Meteostat, Open-Meteo, ERA5).
- Add a check to your code that stops if any weather input is dated after 2018.
- Review what the agent downloaded. If it fetched weather from 2019 onward, keep it out of your forecasts.

**How do we document agent use?**
In your README's *AI Tools and Agents* section: the tool, the model, the tasks it did, and what a person reviewed. Keep the agent's instructions file in `code/`. Your code must reproduce `submission.csv` **without** the agent.

**Can an agent use our Hugging Face token or submit for us?**
An agent can push to your dataset with a **fine-grained token** limited to that one repository. Never commit tokens to your dataset. Submitting in the *Submission* tab is done by a registered team member, signed in with Hugging Face.

**Does agent use affect judging?**
It is not penalized. Friction your agent ran into (missing metadata, unclear licenses, broken APIs) is good material for your **Feedback to the Ecosystem**. Values produced by an AI model that learned from later data break the data rule, disclosed or not.

## Credits

Please cite the data you use (and list any external data in your README):

- Lai, Yuchuan; Dzombak, David (2019). Compiled historical daily temperature and precipitation data for selected 210 U.S. cities. Carnegie Mellon University. https://doi.org/10.1184/R1/7890488.v6
- Lai, Yuchuan; Dzombak, David (2019). Historical changes of annual temperature and precipitation indices at selected 210 U.S. cities. Carnegie Mellon University. https://doi.org/10.1184/R1/7961012.v6
- Lai, Y., & Dzombak, D. A. (2019). Use of historical data to assess regional climate change. *Journal of Climate*, 32(14), 4299–4320. https://doi.org/10.1175/JCLI-D-18-0630.1
