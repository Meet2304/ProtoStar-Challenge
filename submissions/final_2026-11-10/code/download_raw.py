"""Download the raw challenge dataset (all four subsets, data up to 2018-12-31) into data/raw/.

Needed once before rebuilding Placeholder_F4: code/f2/f2_daily.py and code/f2/f2_cddgrid.py read the `daily`
subset from data/raw/daily/. Same download as scripts/01_download_data.py in the team's GitHub repository, but
standalone, so it runs from this dataset with only code/requirements.txt installed. Public dataset, no token.

    python code/download_raw.py
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, TRAIN_REPO, TRAIN_REVISION  # noqa: E402

RAW_DIR = DATA_DIR / "raw"
SUBSETS = ("daily", "annual_indices", "city_info", "coverage_and_stations")


def main() -> None:
    snapshot_download(repo_id=TRAIN_REPO, repo_type="dataset", revision=TRAIN_REVISION, local_dir=RAW_DIR)
    missing = [s for s in SUBSETS if not any((RAW_DIR / s).glob("*.parquet"))]
    if missing:
        raise SystemExit(f"Download incomplete, missing subsets: {', '.join(missing)}")
    print(f"{TRAIN_REPO} @ {TRAIN_REVISION[:7]} -> {RAW_DIR}")


if __name__ == "__main__":
    main()
