"""Starter model: the challenge's prescribed RandomForest on lag features, for all four metrics.

Usage:
    python code/starter_model.py             # backtest (train <= 2010, test 2011-2018), then forecast 2019-2025
    python code/starter_model.py --no-backtest

For each metric, one RandomForestRegressor (100 trees, random_state=42) is trained on every
city's values 1, 2 and 3 years earlier (lag_1..lag_3), pooled across cities. Forecasts are
made recursively: 2019 from 2016-2018, 2020 from 2017-2019 (using the 2019 forecast), and so on.

Inputs: data/annual_valid.csv only (records up to 2018). Outputs:
- submission.csv (5,880 rows), checked with code/check_submission.py
- reports/backtest_starter.csv: MAE and skill of the model and the three reference
  baselines on 2011-2018, trained only on data up to 2010.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from common import (CUTOFF_YEAR, FORECAST_YEARS, METRICS, REPORTS_DIR,
                    SUBMISSION_PATH, load_valid_annual, mae, thirty_year_mean, wide)

LAGS = (1, 2, 3)
SEED = 42


def training_rows(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(X, y): each valid city-year with its three previous years as features."""
    long = table.stack(future_stack=True).rename("y").to_frame()
    for k in LAGS:
        long[f"lag_{k}"] = table.shift(k, axis=1).stack(future_stack=True)
    long = long.dropna()
    return long[[f"lag_{k}" for k in LAGS]].to_numpy(), long["y"].to_numpy()


def forecast(table: pd.DataFrame, last_year: int, years: list[int]) -> pd.DataFrame:
    """Fit on all years up to last_year and predict `years` recursively, for every city."""
    history = table.loc[:, :last_year]
    model = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    model.fit(*training_rows(history))
    # Parallel predict sums tree outputs in thread order, which changes the last digits
    # between runs; one thread keeps submission.csv byte-for-byte reproducible.
    model.set_params(n_jobs=1)

    # Fill invalid recent years with the city's 30-year mean so every city gets a forecast
    fill = thirty_year_mean(history, last_year).fillna(history.mean(axis=1))
    recent = history.loc[:, last_year - max(LAGS) + 1:last_year].apply(lambda col: col.fillna(fill))

    preds = {}
    for year in years:
        features = np.column_stack([recent[year - k] if year - k in recent else preds[year - k]
                                    for k in LAGS])
        preds[year] = pd.Series(model.predict(features), index=table.index)
        recent[year] = preds[year]
    return pd.DataFrame(preds)


def baselines(table: pd.DataFrame, last_year: int, years: list[int]) -> dict[str, pd.DataFrame]:
    """The challenge's three reference baselines, from data up to last_year."""
    history = table.loc[:, :last_year]
    window = history.loc[:, last_year - 29:last_year]

    last_valid = history.ffill(axis=1)[last_year]
    avg30 = thirty_year_mean(history, last_year)

    def trend_row(row: pd.Series) -> pd.Series:
        ok = row.dropna()
        if len(ok) < 2:
            return pd.Series(np.nan, index=years)
        slope, intercept = np.polyfit(ok.index.astype(float), ok.to_numpy(), 1)
        return pd.Series(intercept + slope * np.array(years, dtype=float), index=years)

    def repeat(values: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({y: values for y in years})

    return {"last_year_repeated": repeat(last_valid),
            "30_year_average": repeat(avg30),
            "linear_trend": window.apply(trend_row, axis=1)}


def backtest(annual: pd.DataFrame, last_year: int = 2010) -> pd.DataFrame:
    """Score the model and the baselines on last_year+1..2018, trained only up to last_year."""
    years = list(range(last_year + 1, CUTOFF_YEAR + 1))
    rows = []
    for metric in METRICS:
        table = wide(annual, metric)
        # Rows the challenge would score: valid year, and the city has a 30-year baseline
        actual = table[years].where(thirty_year_mean(table, last_year).notna(), axis=0)
        candidates = {"starter_rf_lags": forecast(table, last_year, years), **baselines(table, last_year, years)}
        for name, pred in candidates.items():
            pred = pred.reindex(index=actual.index, columns=years)
            rows.append({"model": name, "metric": metric,
                         "mae": mae(actual.stack(future_stack=True), pred.stack(future_stack=True))})
    report = pd.DataFrame(rows)
    ref = report[report["model"] == "30_year_average"].set_index("metric")["mae"]
    report["skill"] = 1 - report["mae"] / report["metric"].map(ref)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-backtest", action="store_true", help="skip the 2011-2018 backtest")
    args = parser.parse_args()

    annual = load_valid_annual()

    if not args.no_backtest:
        report = backtest(annual)
        REPORTS_DIR.mkdir(exist_ok=True)
        report.round(4).to_csv(REPORTS_DIR / "backtest_starter.csv", index=False)
        print("Backtest: train <= 2010, test 2011-2018 (skill vs the 30-year average; higher is better)")
        print(report.pivot(index="model", columns="metric", values="mae").round(2).to_string())
        score = 100 * report.groupby("model")["skill"].mean().clip(lower=0)
        print("\nScore (0-100, same formula as the leaderboard):")
        print(score.round(1).sort_values(ascending=False).to_string())

    parts = []
    for metric in METRICS:
        pred = forecast(wide(annual, metric), CUTOFF_YEAR, FORECAST_YEARS)
        long = pred.stack().rename("predicted_value").reset_index()
        long.columns = ["city_id", "year", "predicted_value"]
        long["metric"] = metric
        parts.append(long)
    submission = (pd.concat(parts)[["city_id", "year", "metric", "predicted_value"]]
                  .sort_values(["city_id", "year", "metric"], key=lambda s: s.map(list(METRICS).index)
                               if s.name == "metric" else s))
    submission["predicted_value"] = submission["predicted_value"].round(3)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nWrote {len(submission):,} forecasts to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
