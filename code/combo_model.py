"""Combined model: for each metric, the simple forecast that backtests best across past windows.

    annual_avg_temp                    -> the city's level now (mean of its last 10 years), plus the
                                          warming rate shared by all cities, carried forward
    cdd                                -> the city's level now, growing at the rate shared by all
                                          cities in percent (a city with little air-conditioning
                                          need gets a proportionally small rise)
    total_precip_in, hottest_day_temp  -> the city's 30-year average

Why: temperature and cooling degree days drift upward, so a flat average lags behind. A city's own
30-year trend line catches the drift, but one city's slope is noisy (a few odd years tilt it). All
210 cities share most of the warming, so the median slope across cities is a much steadier
estimate of how fast each city warms; the last-10-year mean gives the city's current level.
Precipitation and the hottest day are mostly year-to-year noise: nothing beat the 30-year average.

El Nino adjustment (tested, NOT used: USE_ENSO = False). It lowered the mean backtest Score from 5.2
to 5.0 (worse in 2 of 4 windows), so the selection rule keeps it out; the backtest still reports it as
"combo_enso", and code/enso_test.py runs a 38-year one-year-ahead test of it. How it works
(temperature and CDD, first forecast year only): the Oceanic Nino Index (ONI)
at the end of the last training year (mean of its Aug-Oct and Sep-Nov seasons) shifts each city's
next-year forecast by the city's El Nino sensitivity: the slope of its past one-year-ahead forecast
misses on that year-end ONI, averaged over the city and its 8 nearest cities. Only years before the
forecast origin are used to fit it. ONI comes from data/ocean_indices.csv (archived NOAA copies
from 2018 or earlier, see code/prepare_ocean_indices.py).

Candidates per metric (all scored in the backtest):
    average          city's 30-year average (the leaderboard's reference)
    trend            straight line through the city's last 30 years
    last10           city's mean of its last 10 years
    blend            half trend, half last10
    level_allcities  last10 level + median 30-year slope across all cities
    level_neighbors  last10 level + mean 30-year slope of the city and its 8 nearest cities
    level_allcities_pct  (precipitation and CDD only, which start at zero) last10 level growing at
                     the median relative slope across cities (% of the city's 30-year mean per
                     year), so cities with almost no CDD do not get the same absolute rise as Miami

Selection rule (fixed before looking at any 2019+ result): per metric, the candidate with the
highest mean skill over the backtest windows (test years 1987-1994, 1995-2002, 2003-2010,
2011-2018). All windows end before 2019; leaderboard scores were not used. METHOD below holds the
choice, and the backtest warns if it no longer matches the rule.

Usage (from the repository root):
    python code/combo_model.py                # backtests -> reports/backtest_combo.csv, forecasts -> submission.csv
    python code/combo_model.py --no-backtest  # forecasts only

Inputs: data/annual_valid.csv (records up to 2018, guarded by load_valid_annual()), the city
coordinates in data/city_info.csv, and the ONI in data/ocean_indices.csv (guarded too).
"""

import argparse

import numpy as np
import pandas as pd

from common import (CUTOFF_YEAR, DATA_DIR, FORECAST_YEARS, METRICS, MIN_VALID_YEARS, REPORTS_DIR,
                    SUBMISSION_PATH, assert_no_post_cutoff, load_valid_annual, mae,
                    thirty_year_mean, wide)

METHOD = {
    "annual_avg_temp": "level_allcities",
    "total_precip_in": "average",
    "hottest_day_temp": "average",
    "cdd": "level_allcities_pct",
}
WINDOW = 30          # years used for averages and slopes
LEVEL_YEARS = 10     # years used for a city's current level
NEIGHBORS = 8        # nearest cities pooled in level_neighbors (plus the city itself)
NON_NEGATIVE = ("total_precip_in", "cdd")
USE_ENSO = False                             # El Nino adjustment in the submitted forecasts (tested; see docstring)
ENSO_METRICS = ("annual_avg_temp", "cdd")   # metrics the El Nino adjustment applies to
ENSO_FIRST_ORIGIN = 1955                     # earliest year used to fit El Nino sensitivities
ENSO_MIN_YEARS = 15                          # years a city needs for its own sensitivity
OCEAN_PATH = DATA_DIR / "ocean_indices.csv"
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


def city_slopes(history: pd.DataFrame, last_year: int) -> pd.Series:
    """Each city's least-squares slope (units per year) over its valid years in the last 30.

    NaN when a city has fewer than MIN_VALID_YEARS valid years there.
    """
    window = history.loc[:, last_year - WINDOW + 1:last_year]
    slopes = {}
    for city, row in window.iterrows():
        ok = row.dropna()
        slopes[city] = (np.polyfit(ok.index.to_numpy(dtype=float), ok.to_numpy(), 1)[0]
                        if len(ok) >= MIN_VALID_YEARS else np.nan)
    return pd.Series(slopes)


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
    level = history.loc[:, last_year - LEVEL_YEARS + 1:last_year].mean(axis=1)
    level = level.fillna(average_forecast(history, last_year, years)[years[0]])
    return pd.DataFrame({y: level for y in years})


def blend_forecast(history: pd.DataFrame, last_year: int, years: list[int]) -> pd.DataFrame:
    """Equal-weight average of the 30-year trend line and the 10-year mean."""
    return 0.5 * trend_forecast(history, last_year, years) + 0.5 * last10_forecast(history, last_year, years)


def level_plus_slope(history: pd.DataFrame, last_year: int, years: list[int],
                     slope: pd.Series) -> pd.DataFrame:
    """City level (mean of its valid years in the last 10) carried forward at `slope` per year.

    The line passes through the level at the mean of the years it was computed from, so a city
    with a missing recent year is not shifted. Cities with no valid recent year get their average.
    """
    recent = history.loc[:, last_year - LEVEL_YEARS + 1:last_year]
    level = recent.mean(axis=1)
    center = recent.notna().mul(recent.columns.to_numpy(dtype=float), axis=1).sum(axis=1) / recent.notna().sum(axis=1)
    pred = pd.DataFrame({y: level + slope.reindex(history.index) * (y - center) for y in years})
    return pred.fillna(average_forecast(history, last_year, years))


def level_allcities_forecast(history: pd.DataFrame, last_year: int, years: list[int]) -> pd.DataFrame:
    """City level + the median 30-year slope across all cities."""
    shared = float(np.nanmedian(city_slopes(history, last_year)))
    return level_plus_slope(history, last_year, years, pd.Series(shared, index=history.index))


def level_allcities_pct_forecast(history: pd.DataFrame, last_year: int, years: list[int]) -> pd.DataFrame:
    """City level + a slope proportional to it: the median across cities of slope / 30-year mean.

    Only cities whose 30-year mean is at least a tenth of the median city's enter the median, so
    near-zero means (e.g., CDD on the foggy Pacific coast) do not produce extreme ratios.
    """
    slopes = city_slopes(history, last_year)
    means = history.loc[:, last_year - WINDOW + 1:last_year].mean(axis=1)
    usable = slopes.notna() & (means >= 0.1 * means.median()) & (means > 0)
    relative = float(np.median(slopes[usable] / means[usable]))
    level = history.loc[:, last_year - LEVEL_YEARS + 1:last_year].mean(axis=1)
    return level_plus_slope(history, last_year, years, relative * level)


def _neighbor_index() -> dict[str, list[str]]:
    """For each city, itself and its NEIGHBORS nearest cities by great-circle distance."""
    cities = pd.read_csv(DATA_DIR / "city_info.csv", dtype={"city_id": str}).set_index("city_id")
    lat, lon = np.radians(cities["lat"].to_numpy()), np.radians(cities["lon"].to_numpy())
    cos_d = (np.sin(lat[:, None]) * np.sin(lat) +
             np.cos(lat[:, None]) * np.cos(lat) * np.cos(lon[:, None] - lon))
    dist = np.arccos(np.clip(cos_d, -1, 1))
    ids = cities.index.to_numpy()
    return {ids[i]: list(ids[np.argsort(dist[i], kind="stable")]) for i in range(len(ids))}


def level_neighbors_forecast(history: pd.DataFrame, last_year: int, years: list[int]) -> pd.DataFrame:
    """City level + the mean 30-year slope of the city and its nearest cities that have one."""
    slopes = city_slopes(history, last_year)
    shared = float(np.nanmedian(slopes))
    order = _neighbor_index()
    pooled = {}
    for city in history.index:
        near = [c for c in order.get(city, [city]) if c in slopes.index and not np.isnan(slopes[c])]
        pooled[city] = slopes[near[:NEIGHBORS + 1]].mean() if near else shared
    return level_plus_slope(history, last_year, years, pd.Series(pooled))


FORECASTERS = {
    "average": average_forecast,
    "trend": trend_forecast,
    "last10": last10_forecast,
    "blend": blend_forecast,
    "level_allcities": level_allcities_forecast,
    "level_neighbors": level_neighbors_forecast,
    "level_allcities_pct": level_allcities_pct_forecast,
}
RATIO_ONLY = {"level_allcities_pct"}   # candidates that only make sense for metrics that start at zero


def forecast(table: pd.DataFrame, metric: str, method: str, last_year: int, years: list[int]) -> pd.DataFrame:
    history = table.loc[:, :last_year]
    pred = FORECASTERS[method](history, last_year, years)
    return pred.clip(lower=0) if metric in NON_NEGATIVE else pred


def load_oni_year_end() -> pd.Series:
    """ONI known at the end of each year: mean of the Aug-Oct and Sep-Nov seasons (months 9, 10)."""
    ocean = pd.read_csv(OCEAN_PATH)
    assert_no_post_cutoff(ocean, year_col="year", source=str(OCEAN_PATH))
    oni = ocean[ocean["index"] == "oni"].pivot(index="year", columns="month", values="value")
    return oni.reindex(columns=[9, 10]).mean(axis=1, skipna=False).dropna()


def enso_shift(table: pd.DataFrame, metric: str, last_year: int, oni_end: pd.Series) -> pd.Series:
    """Per-city change to the forecast for last_year + 1 from the year-end ONI of last_year.

    Fits, for each city, the slope of its one-year-ahead forecast misses (actual minus the METHOD
    forecast made the year before) on the year-end ONI, using origins ENSO_FIRST_ORIGIN to
    last_year - 1 (all outcomes known by last_year). The slope is averaged over the city and its
    NEIGHBORS nearest cities, then multiplied by the ONI at the end of last_year. Zero for other
    metrics, or when that ONI is unknown.
    """
    zero = pd.Series(0.0, index=table.index)
    if metric not in ENSO_METRICS or last_year not in oni_end.index:
        return zero
    origins = [y for y in range(ENSO_FIRST_ORIGIN, last_year) if y in oni_end.index]
    misses = pd.DataFrame({
        y: table[y + 1] - forecast(table, metric, METHOD[metric], y, [y + 1])[y + 1]
        for y in origins if y + 1 in table.columns})
    x = oni_end.reindex(misses.columns).to_numpy()
    slopes = {}
    for city, row in misses.iterrows():
        ok = ~np.isnan(row.to_numpy())
        slopes[city] = (np.polyfit(x[ok], row.to_numpy()[ok], 1)[0]
                        if ok.sum() >= ENSO_MIN_YEARS else np.nan)
    slopes = pd.Series(slopes)
    order = _neighbor_index()
    pooled = {}
    for city in table.index:
        near = [c for c in order.get(city, [city]) if c in slopes.index and not np.isnan(slopes[c])]
        pooled[city] = slopes[near[:NEIGHBORS + 1]].mean() if near else 0.0
    return pd.Series(pooled).reindex(table.index).fillna(0.0) * float(oni_end[last_year])


def combo(table: pd.DataFrame, metric: str, last_year: int, years: list[int],
          oni_end: pd.Series, enso: bool = USE_ENSO) -> pd.DataFrame:
    """The model: METHOD per metric, plus (if enso) the El Nino shift on the first forecast year."""
    pred = forecast(table, metric, METHOD[metric], last_year, years)
    if enso:
        pred[years[0]] = pred[years[0]] + enso_shift(table, metric, last_year, oni_end)
    return pred.clip(lower=0) if metric in NON_NEGATIVE else pred


def backtest(annual: pd.DataFrame) -> pd.DataFrame:
    """MAE and skill of every candidate and of the combo, on each window in BACKTEST_ENDS."""
    oni_end = load_oni_year_end()
    rows = []
    for last_year in BACKTEST_ENDS:
        years = list(range(last_year + 1, last_year + 1 + TEST_LENGTH))
        assert years[-1] <= CUTOFF_YEAR, "backtests must stay inside the training data"
        window = f"train<={last_year}, test {years[0]}-{years[-1]}"
        for metric in METRICS:
            table = wide(annual, metric)
            # Rows the challenge would score: valid year, and the city has a 30-year baseline
            actual = table[years].where(thirty_year_mean(table.loc[:, :last_year], last_year).notna(), axis=0)
            preds = {name: forecast(table, metric, name, last_year, years) for name in FORECASTERS
                     if name not in RATIO_ONLY or metric in NON_NEGATIVE}
            preds["combo"] = combo(table, metric, last_year, years, oni_end)
            preds["combo_enso"] = combo(table, metric, last_year, years, oni_end, enso=True)
            for name, pred in preds.items():
                pred = pred.reindex(index=actual.index, columns=years)
                rows.append({"window": window, "model": name, "metric": metric,
                             "mae": mae(actual.stack(future_stack=True), pred.stack(future_stack=True))})
    report = pd.DataFrame(rows)
    ref = report[report["model"] == "average"].set_index(["window", "metric"])["mae"]
    report["skill"] = 1 - report["mae"] / pd.MultiIndex.from_frame(report[["window", "metric"]]).map(ref)
    return report


def check_choice(report: pd.DataFrame) -> None:
    """Warn if METHOD is not the candidate with the highest mean skill (the selection rule)."""
    mean_skill = (report[~report["model"].str.startswith("combo")]
                  .groupby(["metric", "model"])["skill"].mean().unstack("model"))
    for metric, row in mean_skill.iterrows():
        best = row.idxmax()
        flag = "" if best == METHOD[metric] else f"   <-- WARNING: rule picks {best}"
        print(f"  {metric:18s} METHOD={METHOD[metric]:16s} best={best:16s} "
              f"(mean skill {row[best]:.3f}){flag}")


def scores(report: pd.DataFrame) -> pd.DataFrame:
    """Leaderboard formula per window and model: 100 * max(0, mean skill over the 4 metrics)."""
    s = 100 * (report[~report["model"].isin(RATIO_ONLY)]            # those cover only 2 metrics
               .groupby(["window", "model"])["skill"].mean().clip(lower=0))
    table = s.unstack("window").round(1)
    table["mean"] = table.mean(axis=1).round(1)
    return table


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
        print("\nScore (0-100, same formula as the leaderboard; 'combo' = submitted model, 'combo_enso' = with El Nino shift):")
        print(scores(report).to_string())
        print("\nSelection rule check (highest mean skill per metric):")
        check_choice(report)

    oni_end = load_oni_year_end() if USE_ENSO else pd.Series(dtype=float)
    if USE_ENSO and CUTOFF_YEAR not in oni_end.index:
        raise SystemExit(f"{OCEAN_PATH} has no ONI for Sep-Oct {CUTOFF_YEAR}: "
                         "run python code/prepare_ocean_indices.py")
    parts = []
    for metric in METRICS:
        pred = combo(wide(annual, metric), metric, CUTOFF_YEAR, FORECAST_YEARS, oni_end)
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
