"""Fixed control baselines (`controls-v1`) for the evaluation harness.

Every control is an ordinary adapter, `forecast(train, targets, origin, seed, method=...)`,
run through the same worker, origin guard, grid validation and official scorer as a team
model. Hyperparameters are frozen here from now on, as a stable yardstick for future
comparisons. This is not a prospective preregistration: mean10/blend were already examined
on overlapping history; only the organizers' three references are truly pre-specified.
Changing any control (or adding one) means bumping CONTROL_SET, which starts a new panel.

Controls use only `train` (observations up to the origin). No file is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import METRICS, assert_no_post_cutoff, thirty_year_mean, wide
from combo_model import average_forecast, blend_forecast, last10_forecast
from evaluation.adapters import reference

CONTROL_SET = "controls-v1"

# id -> (tier, one-line description). Order is the display order.
CONTROLS = {
    "average30": ("official reference", "Each city's mean of its last 30 years (the skill denominator)"),
    "persistence": ("official reference", "Last valid year repeated"),
    "trend30": ("official reference", "Least-squares line through the last 30 years, extended"),
    "mean10": ("climatology", "Mean of the last 10 years"),
    "blend": ("trend", "Half trend30, half mean10"),
    "ridge_pooled": ("simple ML", "Ridge regression on anomaly features, pooled across cities, one model per lead"),
    "starter_rf": ("simple ML", "The challenge's starter random forest on lags 1-3, recursive"),
}
# Delivery tier: A ships on its own; B (simple ML) joins the "full" panel profile.
TIER = {"average30": "A", "persistence": "A", "trend30": "A", "mean10": "A", "blend": "A",
        "ridge_pooled": "B", "starter_rf": "B"}
# The organizers' references delegate to the suite's own builder, so their predictions match
# the suite's prebuilt baselines/<name>.csv.
OFFICIAL_REFERENCE = {"average30": "average", "persistence": "last_year", "trend30": "trend"}

# Fixed hyperparameters (controls-v1)
RIDGE_ALPHA = 1.0
RIDGE_PSEUDO_ORIGINS = 40      # pseudo-origins t in [O-40, O-lead]
RIDGE_MIN_ROWS = 50            # fewer pooled training rows for a lead -> average30 for that lead
RF_TREES, RF_LAGS, RF_SEED = 100, (1, 2, 3), 42


def forecast(train, targets, origin, seed=42, *, method):
    if method not in CONTROLS:
        raise ValueError(f"unknown control {method!r}; choose one of {sorted(CONTROLS)}")
    assert_no_post_cutoff(train, "year", "control training input")
    if train.year.max() > origin:
        raise ValueError("control received observations after its origin")
    if method in OFFICIAL_REFERENCE:
        return reference(train, targets, origin, seed, method=OFFICIAL_REFERENCE[method])
    years = sorted(int(y) for y in targets.year.unique())
    if years[0] <= origin:
        raise ValueError("target years must follow the origin")
    cities = sorted(targets.city_id.unique())
    builder = {"mean10": _simple(last10_forecast), "blend": _simple(blend_forecast),
               "ridge_pooled": ridge_pooled, "starter_rf": starter_rf}[method]
    parts = []
    for metric in METRICS:
        table = wide(train, metric).reindex(cities)
        table = table.loc[:, :origin]
        pred = builder(table, origin, years)
        # Same last-resort fill as the official references: a metric-wide training median for
        # unscored early-history cities with no valid observations at all.
        pred = pred.reindex(index=cities, columns=years).fillna(np.nanmedian(table.to_numpy()))
        long = pred.rename_axis(index="city_id", columns="year").stack().rename("predicted_value").reset_index()
        long["metric"] = metric
        parts.append(long)
    return pd.concat(parts, ignore_index=True)[["city_id", "year", "metric", "predicted_value"]]


def _simple(function):
    return lambda table, origin, years: function(table, origin, years)


# ---------------------------------------------------------------------------------------------
# ridge_pooled
# ---------------------------------------------------------------------------------------------
def _window_slope(values: np.ndarray, years: np.ndarray, min_valid: int = 10) -> np.ndarray:
    """OLS slope per row over the valid (non-NaN) entries; 0 where fewer than min_valid."""
    ok = ~np.isnan(values)
    n = ok.sum(axis=1)
    x = np.where(ok, years[None, :], 0.0)
    y = np.where(ok, values, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xbar = x.sum(axis=1) / n
        ybar = y.sum(axis=1) / n
        dx = np.where(ok, years[None, :] - xbar[:, None], 0.0)
        slope = (dx * (y - ybar[:, None]) * ok).sum(axis=1) / (dx ** 2).sum(axis=1)
    return np.where((n >= min_valid) & np.isfinite(slope), slope, 0.0)


def _window_mean(history: pd.DataFrame, t: int, years: int, min_valid: int) -> pd.Series:
    window = history.reindex(columns=range(t - years + 1, t + 1))
    return window.mean(axis=1).where(window.notna().sum(axis=1) >= min_valid)


def ridge_features(table: pd.DataFrame, t: int) -> tuple[pd.DataFrame, pd.Series]:
    """Features and anomaly reference for every city at (pseudo-)origin t.

    Uses only calendar years <= t: this is the information boundary the tests check.
    Missing-feature policy (fixed, training-free): each anomaly feature needs its own
    minimum of valid years, otherwise it is set to 0, i.e. "at the city's reference".
        lag1    latest valid value in t-2..t      (>= 1 valid)
        mean5   mean of t-4..t                     (>= 3 valid)
        mean10  mean of t-9..t                     (>= 5 valid)
        slope30 OLS slope over t-29..t             (>= 10 valid, else 0)
    Returns (features, ref_t); ref_t is the official 30-year mean (>= 10 valid years) and is
    NaN where undefined; such cities are skipped in training and fall back to average30.
    """
    history = table.loc[:, :t]
    ref = thirty_year_mean(history, t)
    window = history.reindex(columns=range(t - 29, t + 1))
    recent = history.reindex(columns=range(t - 2, t + 1)).ffill(axis=1).iloc[:, -1]
    feats = pd.DataFrame({
        "lag1": recent - ref,
        "mean5": _window_mean(history, t, 5, 3) - ref,
        "mean10": _window_mean(history, t, 10, 5) - ref,
        "slope30": _window_slope(window.to_numpy(dtype=float), window.columns.to_numpy(dtype=float)),
    }, index=table.index)
    return feats.fillna(0.0), ref


def ridge_training_examples(table: pd.DataFrame, cache: dict, origin: int, lead: int):
    """Pooled (X, y, index) for one lead: for each pseudo-origin t with t + lead <= origin, the
    features at t and the target value(t + lead) - ref_t, for cities where both are defined.
    `index` holds (t, city_id) for every row, so tests can inspect individual examples."""
    X, y, index = [], [], []
    for t, (feats, ref) in cache.items():
        if t + lead > origin:
            continue
        target = table[t + lead] - ref
        ok = target.notna() & ref.notna()
        X.append(feats[ok].to_numpy())
        y.append(target[ok].to_numpy())
        index += [(t, city) for city in target.index[ok]]
    if not X:
        return np.empty((0, 4)), np.empty(0), []
    return np.vstack(X), np.concatenate(y), index


def ridge_cache(table: pd.DataFrame, origin: int) -> dict:
    first = int(table.columns.min())
    return {t: ridge_features(table, t) for t in range(max(first, origin - RIDGE_PSEUDO_ORIGINS), origin)}


def ridge_pooled(table: pd.DataFrame, origin: int, years: list[int]) -> pd.DataFrame:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    fallback = average_forecast(table, origin, years)
    out = fallback.copy()
    if table.shape[1] == 0:
        return out
    # Pseudo-origins may have partial windows; each feature applies its own validity minimum and
    # examples without a defined 30-year reference (>= 10 valid years) are dropped.
    cache = ridge_cache(table, origin)
    now, ref_now = ridge_features(table, origin)
    for year in years:
        lead = year - origin
        X, y, _ = ridge_training_examples(table, cache, origin, lead)
        if len(y) < RIDGE_MIN_ROWS:
            continue                          # too little history: keep the average fallback
        model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA, solver="cholesky"))
        model.fit(X, y)
        pred = ref_now + model.predict(now.to_numpy())
        out[year] = pred.where(ref_now.notna(), fallback[year])
    return out


# ---------------------------------------------------------------------------------------------
# starter_rf: the challenge starter (code/starter_model.py) with a single-threaded fit/predict
# ---------------------------------------------------------------------------------------------
def starter_rf(table: pd.DataFrame, origin: int, years: list[int]) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestRegressor

    history = table.loc[:, :origin]
    long = history.stack(future_stack=True).rename("y").to_frame()
    for k in RF_LAGS:
        long[f"lag_{k}"] = history.shift(k, axis=1).stack(future_stack=True)
    long = long.dropna()
    if len(long) < RIDGE_MIN_ROWS:
        return average_forecast(history, origin, years)
    model = RandomForestRegressor(n_estimators=RF_TREES, random_state=RF_SEED, n_jobs=1)
    model.fit(long[[f"lag_{k}" for k in RF_LAGS]].to_numpy(), long["y"].to_numpy())
    # Recent lags: the city's 30-year mean, else its all-history mean, else the metric-wide
    # training median (the same last resort as the official references), so features are finite.
    fill = thirty_year_mean(history, origin).fillna(history.mean(axis=1)).fillna(np.nanmedian(history.to_numpy()))
    recent = history.reindex(columns=range(origin - max(RF_LAGS) + 1, origin + 1)).apply(lambda col: col.fillna(fill))
    preds = {}
    for year in range(origin + 1, max(years) + 1):
        features = np.column_stack([recent[year - k] if year - k in recent else preds[year - k] for k in RF_LAGS])
        preds[year] = pd.Series(model.predict(features), index=table.index)
        recent[year] = preds[year]
    return pd.DataFrame({y: preds[y] for y in years})
