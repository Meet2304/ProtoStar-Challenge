"""Shared helpers: paths, metric definitions, the data-rule guard and the 10-day validity rule.

Every script in code/ imports from here, so the data rule is enforced in one place.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
SUBMISSION_PATH = ROOT / "submission.csv"

TRAIN_REPO = "CMU-OSPO/protostar-climate-train-to-2018"
# Pinned dataset commit; keep equal to DATASET_REVISION in src/protostar_challenge/config.py
TRAIN_REVISION = "bbbdebdab2a18e00f97ef87bd3f35a71e222180b"

# Forecasts are made as of this date (challenge data rule). Nothing later may feed a forecast.
CUTOFF_YEAR = 2018
FORECAST_YEARS = list(range(2019, 2026))

# metric code -> (column in annual_indices, "Miss *" columns whose days the metric uses)
METRICS = {
    "annual_avg_temp": ("Avg Temp", ["Miss Tmax", "Miss Tmin"]),
    "total_precip_in": ("ToT Prcp", ["Miss Prcp"]),
    "hottest_day_temp": ("Max Tmax", ["Miss Tmax", "Miss Tmin"]),
    "cdd": ("cdd", ["Miss Tmax", "Miss Tmin"]),
}
# Official scoring: a year is not valid if the station missed more than this many days
MAX_MISSING_DAYS = 10
# Official 30-year-average baseline needs at least this many valid years in the window
MIN_VALID_YEARS = 10


class DataRuleViolation(RuntimeError):
    """Raised when weather data exceeds the challenge or active forecast origin."""


def effective_cutoff_year() -> int:
    """The competition cutoff can only be tightened by an evaluation worker.

    Read on every call, not at import time: adapter imports must obey the active
    fold too. Invalid contexts fail closed. This is an accidental-leakage guard,
    not an OS sandbox against an adapter deliberately changing its environment.
    """
    value = os.environ.get("PROTOSTAR_FOLD_CUTOFF")
    if value is None:
        return CUTOFF_YEAR
    try:
        origin = int(value)
    except ValueError as exc:
        raise DataRuleViolation("invalid PROTOSTAR_FOLD_CUTOFF") from exc
    if str(origin) != value or not 1 <= origin <= CUTOFF_YEAR:
        raise DataRuleViolation("fold cutoff must be an integer year no later than 2018")
    return origin


def assert_no_post_cutoff(df: pd.DataFrame, year_col: str = "Year", source: str = "input") -> None:
    """Stop if any weather row is dated after the cutoff. Call this on every weather input."""
    years = pd.to_numeric(df[year_col], errors="coerce")
    if years.isna().any() or not np.isfinite(years).all() or (years % 1 != 0).any():
        raise DataRuleViolation(f"{source}: weather years must be finite integers")
    cutoff = effective_cutoff_year()
    late = years > cutoff
    if late.any():
        raise DataRuleViolation(
            f"{source}: {int(late.sum())} rows dated after {cutoff}-12-31 "
            f"(max year {int(years.max())}). Active forecast cutoff is {cutoff}-12-31; "
            "the challenge ceiling remains 2018-12-31."
        )


def load_valid_annual() -> pd.DataFrame:
    """data/annual_valid.csv (written by prepare_data.py), checked against the data rule."""
    path = DATA_DIR / "annual_valid.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `python code/prepare_data.py` first")
    df = pd.read_csv(path, dtype={"city_id": str})
    assert_no_post_cutoff(df, "year", str(path))
    return df


def wide(annual: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per city, one column per year (every year present, NaN where invalid)."""
    table = annual.pivot(index="city_id", columns="year", values=metric)
    return table.reindex(columns=range(int(table.columns.min()), int(table.columns.max()) + 1))


def thirty_year_mean(table: pd.DataFrame, last_year: int) -> pd.Series:
    """Each city's mean over the 30 years ending at last_year (needs MIN_VALID_YEARS valid years)."""
    window = table.loc[:, last_year - 29:last_year]
    mean = window.mean(axis=1)
    return mean.where(window.notna().sum(axis=1) >= MIN_VALID_YEARS)


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    ok = actual.notna() & predicted.notna()
    return float(np.mean(np.abs(actual[ok] - predicted[ok])))
