"""Model entry points: forecast(train, targets, origin, seed, **params).

Only train contains observations; targets is a city/year/metric grid. Return the
same grid plus predicted_value. Never read evaluator labels inside an adapter.
"""
import numpy as np
import pandas as pd

from common import METRICS, assert_no_post_cutoff, wide
from combo_model import average_forecast, trend_forecast, combo_forecast


def reference(train, targets, origin, seed=42, method="average"):
    assert_no_post_cutoff(train, "year", "adapter training input")
    if train.year.max() > origin:
        raise ValueError("adapter received observations after its origin")
    years = sorted(targets.year.unique().tolist())
    cities = sorted(targets.city_id.unique())
    parts = []
    for metric in METRICS:
        table = wide(train, metric).reindex(cities)
        if method == "average":
            pred = average_forecast(table, origin, years)
        elif method == "trend":
            pred = trend_forecast(table, origin, years)
        elif method == "combo":
            pred = combo_forecast(table, metric, origin, years)
        elif method == "last_year":
            # Last valid observation, not necessarily the origin's observation.
            level = table.ffill(axis=1).iloc[:, -1]
            pred = pd.DataFrame({year: level for year in years})
        else:
            raise ValueError(f"unknown reference method: {method}")
        # Some unscored early-history cities have no valid observations. Fill
        # their mandatory rows using a metric-wide training median only.
        fallback = np.nanmedian(table.to_numpy())
        pred = pred.fillna(fallback)
        long = pred.rename_axis(index="city_id", columns="year").stack().rename("predicted_value").reset_index()
        long["metric"] = metric
        parts.append(long)
    return pd.concat(parts, ignore_index=True)[["city_id", "year", "metric", "predicted_value"]]


def combo(train, targets, origin, seed=42, **params):
    if params:
        raise ValueError(f"fixed combo takes no tunable parameters; unknown keys: {sorted(params)}. "
                         "Use evaluation.nested:forecast for origin-specific selection.")
    return reference(train, targets, origin, seed, method="combo")
