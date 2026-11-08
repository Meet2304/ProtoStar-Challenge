"""Step 2: clean the four training subsets and write the processed dataset to data/processed/.

Laptop:  uv run python scripts/02_clean_data.py                  (about 12 minutes, with the backtest)
         uv run python scripts/02_clean_data.py --skip-backtest  (reuses the saved input choice if nothing changed)
Colab:   python scripts/02_clean_data.py   (after the notebook setup cell and scripts/01_download_data.py)

Reads data/raw/ (run scripts/01_download_data.py first; otherwise the subsets are streamed from the Hub at the
pinned revision) and data/annual_valid.csv (code/prepare_data.py). Writes, all git-ignored:

    data/processed/annual_clean.parquet     one row per city-year (28,052): official labels, QC values,
                                            input variants, the chosen <metric>_input, coverage counts
    data/processed/coverage_clean.parquet   one row per city-year: missing days, validity, station context
    data/processed/stations_clean.parquet   one row per station period (461): keys, gaps, steps
    data/processed/daily_clean.parquet      one row per city-day: raw, QC, filled values, station offsets, flags
    data/processed/qc/                      logs of every repair and deletion, steps, calibration, plots
    data/processed/backtest_*.csv / .json   the input-selection backtest (plan section 5)
    data/processed/manifest.json            hashes, constants, checks, counts, runtime

Also updates the tracked review file data/qc/daily_outlier_decisions.csv (set `final_action`, `decided_on` and
`evidence_year` there to overrule a decision; then rerun without --skip-backtest).

What each step does and why: documentation/README.md. The reviewed plan: notebooks/Meet/plan/cleaning-plan.md.
Data rule: weather data up to 2018-12-31 only; every table passes assert_no_post_cutoff().
"""

import argparse

from protostar_challenge.cleaning import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-backtest", action="store_true",
                        help="reuse the saved input choice (fails if any input, code or constant changed)")
    args = parser.parse_args()
    run(skip_backtest=args.skip_backtest)


if __name__ == "__main__":
    main()
