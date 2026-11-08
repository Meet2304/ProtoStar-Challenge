"""Rebuild the cleaning once per forecast origin, each from data up to that origin only.

Usage:   uv run python scripts/03_clean_per_origin.py [--origins 1983 1990] [--out outputs/clean_per_origin]

Reads data/raw/ (run scripts/01_download_data.py first). Writes one folder per origin in evaluation/config.json,
outside data/processed/ so nothing unreleased sits there; hand the result on with `line.py release`:

    origin-YYYY/annual_clean.parquet     one row per city-year through YYYY: labels, QC values, every input variant
    origin-YYYY/coverage_clean.parquet   one row per city-year through YYYY: missing days, validity, station context
    origin-YYYY/stations_clean.parquet   one row per station period known at YYYY
    origin-YYYY/daily_clean.parquet      one row per city-day through YYYY
    origin-YYYY/qc/                      every repair and deletion, station steps, calibration, fill choice
    origin-YYYY/manifest.json            hashes, constants, checks, counts

What is cut and why: documentation/09-per-origin.md. Data rule: nothing dated after the origin reaches the build.
"""

import argparse
from pathlib import Path

from protostar_challenge.cleaning_per_origin import ORIGINS, OUT_DIR, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--origins", type=int, nargs="+", default=list(ORIGINS))
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    unknown = set(args.origins) - set(ORIGINS)
    if unknown:
        parser.error(f"not an origin in evaluation/config.json: {sorted(unknown)}")
    run(tuple(args.origins), args.out)


if __name__ == "__main__":
    main()
