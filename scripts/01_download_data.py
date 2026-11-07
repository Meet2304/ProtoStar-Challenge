"""Step 1: download the challenge dataset into data/raw at the pinned revision.

Laptop:  uv run python scripts/01_download_data.py
Colab:   python scripts/01_download_data.py   (after the notebook setup cell)

Safe to rerun: files that are already up to date are not downloaded again.
"""

from huggingface_hub import snapshot_download

from protostar_challenge.config import (
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_SUBSETS,
    RAW_DIR,
)


def main() -> None:
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        local_dir=RAW_DIR,
    )

    # Stop here, loudly, if a subset is missing, rather than failing later.
    missing = [s for s in DATASET_SUBSETS if not any((RAW_DIR / s).glob("*.parquet"))]
    if missing:
        raise SystemExit(f"Download incomplete, missing subsets: {', '.join(missing)}")

    print(f"{DATASET_REPO} @ {DATASET_REVISION[:7]} -> {RAW_DIR}")
    for path in sorted(RAW_DIR.glob("*/*.parquet")):
        size_mb = path.stat().st_size / 1e6
        print(f"  {path.relative_to(RAW_DIR).as_posix():<55} {size_mb:7.2f} MB")


if __name__ == "__main__":
    main()
