# Notebooks

Personal exploration space, one folder per team member:

| Folder | Owner |
|---|---|
| [`Meet/`](Meet/) | Meet |
| [`Kohav/`](Kohav/) | Kohav |
| [`Ananya/`](Ananya/) | Ananya |
| [`Achintya/`](Achintya/) | Achintya |

[`01_explore.ipynb`](01_explore.ipynb) is the shared Colab starter: copy it into your folder to begin (its setup cell clones the repo, installs `requirements.txt` and downloads the data to `data/raw/`).

Work freely in your own folder: experiments, plots, Colab notebooks. A few shared rules:

- **The data rule applies here too.** Use weather data up to 2018-12-31 only, and never score or
  tune models on 2019–2025 observations. Call `assert_no_post_cutoff()` from `code/common.py` on any
  weather data you load (see "Loading the data" below).
- **Backtest before you promote.** Validate on years up to 2018 (default: train ≤ 2010, test
  2011–2018) and compare with the baselines in `reports/backtest_starter.csv`.
- **Promote to `code/`.** Notebooks are not part of the reproducible pipeline. When an idea beats
  the current model, turn it into a script in `code/` that regenerates `submission.csv`.
- **Clear large outputs** before committing (no embedded data dumps; keep notebooks under a few MB),
  and never commit tokens.
- Name notebooks `NN-short-topic.ipynb` (e.g., `01-cdd-trends.ipynb`) so they sort in order.

## Loading the data

From a notebook in `notebooks/<Name>/`:

```python
import sys; sys.path.insert(0, "../../code")
from common import load_valid_annual, assert_no_post_cutoff, wide

annual = load_valid_annual()          # data/annual_valid.csv, guarded against post-2018 rows
cdd = wide(annual, "cdd")             # cities x years table
```

Or straight from the Hub (public dataset, no token; works in Colab):

```python
import pandas as pd
annual = pd.read_parquet("hf://datasets/CMU-OSPO/protostar-climate-train-to-2018/annual_indices")
```
