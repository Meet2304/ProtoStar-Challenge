"""Project settings shared by scripts and notebooks.

Change values here, not in the scripts, so every step uses the same paths,
dataset version and cutoff.
"""

from pathlib import Path

# Repo root. This works because uv (and `pip install -e .` on Colab) install
# the package in editable mode, so this file stays inside the repo.
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # Hugging Face download, never edited
PROCESSED_DIR = DATA_DIR / "processed"  # built by scripts
OUTPUTS_DIR = ROOT / "outputs"  # submission.csv, plots

# Challenge dataset, pinned to one commit so everyone trains on the same data.
# To move to a newer version, change the hash here and rerun
# scripts/01_download_data.py.
DATASET_REPO = "CMU-OSPO/protostar-climate-train-to-2018"
DATASET_REVISION = "bbbdebdab2a18e00f97ef87bd3f35a71e222180b"
DATASET_SUBSETS = ("daily", "annual_indices", "city_info", "coverage_and_stations")

# Competition rule: forecasts may only use weather records up to this date.
CUTOFF_DATE = "2018-12-31"
CUTOFF_YEAR = 2018
