# Meet's notebooks

Exploration space for Meet. See [`../README.md`](../README.md) for the shared rules (data rule, backtesting, promoting work to `code/`).

| Notebook | What it covers |
|---|---|
| [`01-dataset-structure.ipynb`](01-dataset-structure.ipynb) | Structure of the four training subsets, a data dictionary, key/integrity checks, coverage, the four targets, data-quality issues, station changes, feature correlations, and a cleaned city-year table (`data/processed/meet_annual_clean.parquet`, git-ignored; rebuilt by running the notebook). Challenge dataset only, no external data. |
| [`03-placeholder-v0.1-training.ipynb`](03-placeholder-v0.1-training.ipynb) | Trains and evaluates `Placeholder_v0.1`, the first model of the Placeholder series, through the team's evaluation loop and logs to MLflow. Model code is in `placeholder_v0_1.py`; card in [`docs/models/placeholder.md`](../../docs/models/placeholder.md). |
| [`04-enhanced-dataset.ipynb`](04-enhanced-dataset.ipynb) | The enhanced dataset `enhance-v2`: structure and data dictionary of the city facts and ocean indices, checks for flaws, how both relate to the targets (years up to 2004), and the cleaning that becomes release `clean-v2` (`scripts/04_clean_enhanced.py`, [`documentation/10-enhanced.md`](../../documentation/10-enhanced.md)). Needs the releases locally (`line.py pull enhance-v1 enhance-v2`), so not for Colab. |
| [`plan/cleaning-plan.md`](plan/cleaning-plan.md) | The reviewed cleaning plan for all four subsets (final, v4) and its four review rounds in [`plan/reviews/`](plan/reviews/) (Sol and Grok; final 95 and 97). Implemented as `scripts/02_clean_data.py` (release `clean-v0`) and per forecast origin as `scripts/03_clean_per_origin.py` (release `clean-v1`, the one to use for scored runs); documented in [`../../documentation/`](../../documentation/README.md). Notebook 02 and `meet_cleaning.py` are the earlier v0 exploration of `daily` and `annual_indices`. |

- **Locally:** run from this folder or the repo root with the project environment (`uv sync`, then pick the `.venv` kernel).
- **On Colab:** open the notebook and run all cells. The first cell clones the repo (branch `Meet_Exploration`, else the
  default branch) to `/content/ProtoStar-Challenge`. Colab only sees what is pushed to GitHub.
- Raw data: read from `data/raw/` if present (`uv run python scripts/01_download_data.py`), otherwise streamed from the
  Hugging Face Hub at the pinned revision (about 50 MB; no upload needed).
