"""Combined model: for each metric, the simple forecast that backtests best across past windows.

    annual_avg_temp, cdd               -> blend: half a straight line through the city's last 30 years,
                                          half the city's mean of its last 10 years
    total_precip_in, hottest_day_temp  -> the city's 30-year average

Why: temperature and cooling degree days drift upward, so a flat 30-year average lags behind. A
30-year trend line catches the drift but swings with noise near the end of the window, and the
10-year mean is stable but lags. Their average was the best of the four candidates (30-year
average, trend, 10-year mean, blend) on mean skill across four backtest windows (test years
1987-1994, 1995-2002, 2003-2010, 2011-2018), and never the worst in any window. Precipitation and
the hottest day are mostly year-to-year noise: nothing beat the 30-year average on them.

Selection rule (fixed before looking at any 2019+ result): per metric, the candidate with the
highest mean skill over the backtest windows. All windows end before 2019; leaderboard scores
were not used.

Usage (from the repository root):
    python code/combo_model.py                # backtests -> reports/backtest_combo.csv, forecasts -> submission.csv
    python code/combo_model.py --no-backtest  # forecasts only

Inputs: data/annual_valid.csv only (records up to 2018, guarded by load_valid_annual()).
"""

import argparse

import numpy as np
import pandas as pd

from common import (CUTOFF_YEAR, FORECAST_YEARS, METRICS, MIN_VALID_YEARS, REPORTS_DIR,
                    SUBMISSION_PATH, load_valid_annual, mae, thirty_year_mean, wide)

METHOD = {
    "annual_avg_temp": "blend",
    "total_precip_in": "average",
    "hottest_day_temp": "average",
    "cdd": "blend",
}
WINDOW = 30
# Backtests: train on data up to each year, test on the 8 years after it (all before 2019).
BACKTEST_ENDS = (2010, 2002, 1994, 1986)
TEST_LENGTH = 8


def average_forecast(history: pd.DataFrame, last_year: int, years: list[int]) -> pd.DataFrame:
    """Each city's 30-year average, repeated for every forecast year.

    Cities without enough valid years for an official baseline (they are not scored) fall back
    to the mean of whatever valid years they have, so every row still gets a number.
    """
    level = thirty_year_mean(history, last_year)
    level = level.fillna(history.loc[:, last_year - WINDOW + 1:last_year].mean(axis=1))
    level = level.fillna(history.mean(axis=1))
    return pd.DataFrame({y: level for y in years})


def trend_forecast(history: pd.DataFrame, last_year: int, years: list[int]) -> pd.DataFrame:
    """A least-squares line through each city's valid years in the last 30, extended to `years`.

    Needs MIN_VALID_YEARS valid years (the same minimum as the official baseline); otherwise the
    city gets its average forecast instead.
    """
    window = history.loc[:, last_year - WINDOW + 1:last_year]
    fallback = average_forecast(history, last_year, years)
    x_future = np.array(years, dtype=float)
    rows = {}
    for city, row in window.iterrows():
        ok = row.dropna()
        if len(ok) < MIN_VALID_YEARS:
            rows[city] = fallback.loc[city].to_numpy()
            continue
        slope, intercept = np.polyfit(ok.index.to_numpy(dtype=float), ok.to_numpy(), 1)
        rows[city] = intercept + slope * x_future
    return pd.DataFrame.from_dict(rows, orient="index", columns=years).reindex(history.index)


def last10_forecast(history: pd.DataFrame, last_year: int, years: list[int]) -> pd.DataFrame:
    """Each city's mean of its last 10 years (the challenge's 'Ocarina of Ten' reference)."""
    level = history.loc[:, last_year - 9:last_year].mean(axis=1)
    level = level.fillna(average_forecast(history, last_year, years)[years[0]])
    return pd.DataFrame({y: level for y in years})


def blend_forecast(history: pd.DataFrame, last_year: int, years: list[int]) -> pd.DataFrame:
    """Equal-weight average of the 30-year trend line and the 10-year mean."""
    return 0.5 * trend_forecast(history, last_year, years) + 0.5 * last10_forecast(history, last_year, years)


FORECASTERS = {"average": average_forecast, "trend": trend_forecast, "last10": last10_forecast,
               "blend": blend_forecast}


def combo_forecast(table: pd.DataFrame, metric: str, last_year: int, years: list[int]) -> pd.DataFrame:
    history = table.loc[:, :last_year]
    return FORECASTERS[METHOD[metric]](history, last_year, years)


def backtest(annual: pd.DataFrame) -> pd.DataFrame:
    """MAE and skill of the combo and the references, on each window in BACKTEST_ENDS."""
    rows = []
    for last_year in BACKTEST_ENDS:
        years = list(range(last_year + 1, last_year + 1 + TEST_LENGTH))
        assert years[-1] <= CUTOFF_YEAR, "backtests must stay inside the training data"
        for metric in METRICS:
            table = wide(annual, metric)
            history = table.loc[:, :last_year]
            # Rows the challenge would score: valid year, and the city has a 30-year baseline
            actual = table[years].where(thirty_year_mean(history, last_year).notna(), axis=0)
            candidates = {
                "combo": combo_forecast(table, metric, last_year, years),
                "30_year_average": average_forecast(history, last_year, years),
                "linear_trend": trend_forecast(history, last_year, years),
                "last_10yr_mean": last10_forecast(history, last_year, years),
                "trend_10yr_blend": blend_forecast(history, last_year, years),
            }
            for name, pred in candidates.items():
                pred = pred.reindex(index=actual.index, columns=years)
                rows.append({"window": f"train<={last_year}, test {years[0]}-{years[-1]}",
                             "model": name, "metric": metric,
                             "mae": mae(actual.stack(future_stack=True), pred.stack(future_stack=True))})
    report = pd.DataFrame(rows)
    ref = report[report["model"] == "30_year_average"].set_index(["window", "metric"])["mae"]
    report["skill"] = 1 - report["mae"] / pd.MultiIndex.from_frame(report[["window", "metric"]]).map(ref)
    return report


def scores(report: pd.DataFrame) -> pd.DataFrame:
    """Leaderboard formula per window and model: 100 * max(0, mean skill over the 4 metrics)."""
    s = 100 * report.groupby(["window", "model"])["skill"].mean().clip(lower=0)
    return s.unstack("window").round(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-backtest", action="store_true", help="skip the backtests")
    args = parser.parse_args()

    annual = load_valid_annual()

    if not args.no_backtest:
        report = backtest(annual)
        REPORTS_DIR.mkdir(exist_ok=True)
        report.round(4).to_csv(REPORTS_DIR / "backtest_combo.csv", index=False)
        print("Skill vs the 30-year average, per metric (higher is better):")
        print(report.pivot_table(index=["metric", "model"], columns="window", values="skill")
              .round(3).to_string())
        print("\nScore (0-100, same formula as the leaderboard):")
        table = scores(report)
        table["mean"] = table.mean(axis=1).round(1)
        print(table.to_string())

    parts = []
    for metric in METRICS:
        pred = combo_forecast(wide(annual, metric), metric, CUTOFF_YEAR, FORECAST_YEARS)
        long = pred.stack().rename("predicted_value").reset_index()
        long.columns = ["city_id", "year", "predicted_value"]
        long["metric"] = metric
        parts.append(long)
    order = {m: i for i, m in enumerate(METRICS)}
    submission = (pd.concat(parts)[["city_id", "year", "metric", "predicted_value"]]
                  .sort_values(["city_id", "year", "metric"],
                               key=lambda s: s.map(order) if s.name == "metric" else s))
    submission["predicted_value"] = submission["predicted_value"].round(3)

    expected = annual["city_id"].nunique() * len(FORECAST_YEARS) * len(METRICS)
    if len(submission) != expected or not np.isfinite(submission["predicted_value"]).all():
        raise SystemExit(f"Incomplete forecasts: {len(submission)} rows (expected {expected}) "
                         "or non-finite values")
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nWrote {len(submission):,} forecasts to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
