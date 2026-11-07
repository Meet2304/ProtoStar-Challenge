"""Download the challenge training data and write the curated files in data/.

Usage: python code/prepare_data.py

Reads `annual_indices` and `city_info` from CMU-OSPO/protostar-climate-train-to-2018
(public, CC0; no token needed) and writes:

- data/annual_valid.csv: one row per city and year (up to 2018), the four metrics,
  with a value set to empty when that year missed more than 10 days of the data it uses
  (the official scoring rule).
- data/city_info.csv: one row per city (station ID, name, latitude, longitude).
"""

import pandas as pd

from common import (DATA_DIR, MAX_MISSING_DAYS, METRICS, TRAIN_REPO,
                    assert_no_post_cutoff)


def main() -> None:
    annual = pd.read_parquet(f"hf://datasets/{TRAIN_REPO}/annual_indices")
    assert_no_post_cutoff(annual, "Year", "annual_indices")

    out = annual[["ID", "Name", "Year"]].rename(columns={"ID": "city_id", "Name": "name", "Year": "year"})
    for metric, (column, miss_cols) in METRICS.items():
        too_many_missing = annual[miss_cols].max(axis=1) > MAX_MISSING_DAYS
        out[metric] = annual[column].mask(too_many_missing).round(3)
    out = out.sort_values(["city_id", "year"])

    cities = (pd.read_parquet(f"hf://datasets/{TRAIN_REPO}/city_info")
              [["ID", "Name", "Lat", "Lon"]].drop_duplicates("ID")
              .rename(columns={"ID": "city_id", "Name": "name", "Lat": "lat", "Lon": "lon"})
              .sort_values("city_id"))

    DATA_DIR.mkdir(exist_ok=True)
    out.to_csv(DATA_DIR / "annual_valid.csv", index=False)
    cities.to_csv(DATA_DIR / "city_info.csv", index=False)
    print(f"Wrote {len(out):,} city-years ({out['city_id'].nunique()} cities, "
          f"{out['year'].min()}–{out['year'].max()}) and {len(cities)} cities to {DATA_DIR}/")


if __name__ == "__main__":
    main()
