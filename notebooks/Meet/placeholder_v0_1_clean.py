"""Placeholder_v0.1_clean: Placeholder_v0.1 with its learners trained on cleaned history from clean-v1.

    forecast(city, metric, origin + h) = ref30(city, origin) + lambda_metric * shift_hat(features, h)

ref30 is the official 30-year city mean, so lambda = 0 reproduces the reference exactly. shift_hat is
the average of two regularized learners fitted on (city, pseudo-origin, lead) rows pooled across
cities; every feature uses only years up to its pseudo-origin. One lambda per metric is chosen from
earlier inner origins, so a metric with no learnable shift falls back to skill 0.

Harness entry point: forecast(train, targets, origin, seed=42, **params), run through
scripts/eval_model.py. Design and evidence: docs/plans/placeholder-training-plan.md.

One change from v0.1: the tables behind every feature and shift target can come from the per-origin cleaned
dataset, chosen per metric with the `inputs` parameter. The anchor ref30 and the inner-fold scoring stay on the
official labels, because the official labels are what is scored. With every metric on "official" this is v0.1.

Single file on purpose: the runtime audit allows the entry point by exact hash, and helpers outside
code/ and src/ would need a declared source asset. City coordinates come only from the declared
static asset under PROTOSTAR_ASSET_ROOT.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from common import METRICS, assert_no_post_cutoff, thirty_year_mean, wide
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

MODEL_NAME = "Placeholder_v0.1_clean"

# Fixed in advance (plan section 3). Only lambda is chosen from data.
LEADS = tuple(range(1, 8))
TRAIN_YEARS = 45
N_NEIGHBOURS = 15
INNER_OFFSETS = (21, 14, 7)
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
LAMBDA_MARGIN = 0.005
MIN_VALID_WINDOW = 10
GAP_MIN_VALID = {5: 3, 10: 5, 15: 7}
NON_NEGATIVE = {"total_precip_in", "cdd"}
HUBER = {"epsilon": 1.35, "alpha": 1.0, "max_iter": 500}
GBM = {"loss": "absolute_error", "max_depth": 3, "learning_rate": 0.05, "max_iter": 150,
       "min_samples_leaf": 500, "l2_regularization": 5.0, "early_stopping": False}

OWN_FEATURES = ["gap_lag1", "gap5", "gap10", "gap15", "slope30", "trend_gap", "ref30", "sigma30", "n_valid30"]
POOLED_FEATURES = ["reg_gap10", "reg_slope30", "nat_gap10", "nat_slope30", "tavg_gap10", "tavg_trend_gap"]
LEVEL_FEATURES = {"ref30", "sigma30", "n_valid30"}   # missing -> fit median; all others missing -> 0 (no signal)

MEMBERS = ("ensemble", "lin", "gbm")
GATES = ("inner", "none")
FEATURE_SETS = ("all", "no_pool")
KEY = ["city_id", "year", "metric"]

CLEAN_RELEASE = "clean-v1"
# name of a cleaned series -> suffix on the metric's column in annual_clean.parquet
INPUT_SUFFIX = {"official": "", "qc": "_qc", "homog": "_homog", "fill": "_fill", "fill_homog": "_fill_homog"}
ALLOWED_INPUTS = {"annual_avg_temp": ("official", "qc", "homog", "fill", "fill_homog"),
                  "cdd": ("official", "qc", "homog", "fill", "fill_homog"),
                  "total_precip_in": ("official", "qc", "fill"),
                  "hottest_day_temp": ("official", "qc", "homog")}


# ---------------------------------------------------------------------------------------------
# Geography
# ---------------------------------------------------------------------------------------------
def load_coords(cities) -> pd.DataFrame:
    """Latitude and longitude from the declared snapshot; the raw data/ path is deliberately not used."""
    root = os.environ.get("PROTOSTAR_ASSET_ROOT")
    if not root:
        raise RuntimeError("PROTOSTAR_ASSET_ROOT is not set: run through the harness with the asset manifest")
    table = pd.read_csv(Path(root) / "data" / "city_info.csv", dtype={"city_id": str}, usecols=["city_id", "lat", "lon"])
    return _check_coords(table.set_index("city_id"), cities)


def _check_coords(table: pd.DataFrame, cities) -> pd.DataFrame:
    coords = table.reindex(list(cities))[["lat", "lon"]]
    if coords.isna().any(axis=None):
        raise ValueError("city coordinates are missing for some forecast cities")
    return coords


def neighbour_index(coords: pd.DataFrame, k: int = N_NEIGHBOURS) -> np.ndarray:
    """Row positions of each city's k nearest other cities. Coordinates must be sorted by city_id."""
    if list(coords.index) != sorted(coords.index):
        raise ValueError("coordinates must be sorted by city_id so distance ties break by city_id")
    lat, lon = np.radians(coords.lat.to_numpy(float)), np.radians(coords.lon.to_numpy(float))
    half_lat = (lat[:, None] - lat[None, :]) / 2
    half_lon = (lon[:, None] - lon[None, :]) / 2
    a = np.sin(half_lat) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(half_lon) ** 2
    distance = 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    np.fill_diagonal(distance, np.inf)
    order = np.argsort(distance, axis=1, kind="stable")   # stable: equal distances keep city_id order
    return order[:, : min(k, len(coords) - 1)]


# ---------------------------------------------------------------------------------------------
# Cleaned history
# ---------------------------------------------------------------------------------------------
def resolve_inputs(inputs: dict | None) -> dict[str, str]:
    """One cleaned series per metric; a metric that is not named stays on the official labels."""
    inputs = dict(inputs or {})
    unknown = set(inputs) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown metric in inputs: {sorted(unknown)}")
    resolved = {metric: inputs.get(metric, "official") for metric in METRICS}
    for metric, name in resolved.items():
        if name not in ALLOWED_INPUTS[metric]:
            raise ValueError(f"{metric} has no cleaned series {name!r}")
    return resolved


def uses_cleaned(inputs: dict[str, str]) -> bool:
    return any(name != "official" for name in inputs.values())


def load_cleaned(origin: int, inputs: dict[str, str]) -> pd.DataFrame:
    """The chosen columns of this origin's cleaned table, from the declared snapshot."""
    root = os.environ.get("PROTOSTAR_ASSET_ROOT")
    if not root:
        raise RuntimeError("PROTOSTAR_ASSET_ROOT is not set: run through the harness with the asset manifest")
    path = Path(root) / "data" / "processed" / "releases" / CLEAN_RELEASE / f"origin-{origin}" / "annual_clean.parquet"
    columns = [metric + INPUT_SUFFIX[name] for metric, name in inputs.items() if name != "official"]
    cleaned = pd.read_parquet(path, columns=["city_id", "year", *columns])
    assert_no_post_cutoff(cleaned, "year", f"{CLEAN_RELEASE} origin {origin}")
    if cleaned.year.max() > origin:
        raise ValueError(f"{CLEAN_RELEASE} origin {origin} holds years after its origin")
    return cleaned


def apply_inputs(history: pd.DataFrame, cleaned: pd.DataFrame, inputs: dict[str, str]) -> pd.DataFrame:
    """The official history with each metric's column replaced by its chosen series, for every city-year."""
    joined = history.merge(cleaned, on=["city_id", "year"], how="left", validate="one_to_one")
    for metric, name in inputs.items():
        if name != "official":
            joined[metric] = joined[metric + INPUT_SUFFIX[name]]
    return joined[list(history.columns)]


# ---------------------------------------------------------------------------------------------
# Origin-anchored state. Everything here receives only columns <= t.
# ---------------------------------------------------------------------------------------------
def _safe_mean(values: np.ndarray, axis: int, minimum: int = 1) -> np.ndarray:
    ok = ~np.isnan(values)
    count = ok.sum(axis=axis)
    total = np.where(ok, values, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count >= minimum, total / count, np.nan)


def metric_state(table: pd.DataFrame, t: int) -> dict[str, np.ndarray]:
    """Level, trend and noise of every city at pseudo-origin t, from years <= t only."""
    history = table.loc[:, :t]
    ref = thirty_year_mean(history, t).to_numpy(float)
    window = history.reindex(columns=range(t - 29, t + 1)).to_numpy(float)
    years = np.arange(t - 29, t + 1, dtype=float)
    ok = ~np.isnan(window)
    n_valid = ok.sum(axis=1)

    has_any = ok.any(axis=1)
    last_position = window.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1)
    last = np.where(has_any, window[np.arange(len(window)), last_position], np.nan)

    state = {"ref30": ref, "n_valid30": n_valid.astype(float), "gap_lag1": last - ref}
    for n, minimum in GAP_MIN_VALID.items():
        state[f"gap{n}"] = _safe_mean(window[:, -n:], axis=1, minimum=minimum) - ref

    with np.errstate(invalid="ignore", divide="ignore"):
        x_mean = np.where(ok, years[None, :], 0.0).sum(axis=1) / n_valid
        y_mean = np.where(ok, window, 0.0).sum(axis=1) / n_valid
        dx = np.where(ok, years[None, :] - x_mean[:, None], 0.0)
        dy = np.where(ok, window - y_mean[:, None], 0.0)
        slope = (dx * dy).sum(axis=1) / (dx ** 2).sum(axis=1)
        line_at_t = y_mean + slope * (t - x_mean)
        residual = np.where(ok, dy - slope[:, None] * dx, 0.0)
        sigma = np.sqrt((residual ** 2).sum(axis=1) / (n_valid - 2))
    enough = n_valid >= MIN_VALID_WINDOW
    state["slope30"] = np.where(enough & np.isfinite(slope), slope, np.nan)
    state["line_at_t"] = np.where(enough & np.isfinite(line_at_t), line_at_t, np.nan)
    state["sigma30"] = np.where(enough & np.isfinite(sigma), sigma, np.nan)
    return state


def pooled_state(states: dict[str, dict[str, np.ndarray]], metric: str, neighbours: np.ndarray) -> dict[str, np.ndarray]:
    """Regional, national and cross-metric features built from states at the same pseudo-origin."""
    own, tavg = states[metric], states["annual_avg_temp"]
    n = len(own["gap10"])
    return {
        "reg_gap10": _safe_mean(own["gap10"][neighbours], axis=1),
        "reg_slope30": _safe_mean(own["slope30"][neighbours], axis=1),
        "nat_gap10": np.full(n, _safe_mean(own["gap10"], axis=0)),
        "nat_slope30": np.full(n, _safe_mean(own["slope30"], axis=0)),
        "tavg_gap10": tavg["gap10"],
    }


def feature_matrix(state: dict, pooled: dict, tavg: dict, h: int, feature_set: str) -> np.ndarray:
    trend_gap = state["line_at_t"] + state["slope30"] * h - state["ref30"]
    tavg_trend_gap = tavg["line_at_t"] + tavg["slope30"] * h - tavg["ref30"]
    extra = {"trend_gap": trend_gap, "tavg_trend_gap": tavg_trend_gap}
    columns = list(OWN_FEATURES)
    if feature_set == "all":
        columns += POOLED_FEATURES
    n = len(state["ref30"])
    return np.column_stack([extra[c] if c in extra else (pooled[c] if c in pooled else state[c]) for c in columns]
                           + [np.full(n, float(h))])


def feature_names(feature_set: str) -> list[str]:
    return list(OWN_FEATURES) + (list(POOLED_FEATURES) if feature_set == "all" else []) + ["h"]


# ---------------------------------------------------------------------------------------------
# Training rows and members
# ---------------------------------------------------------------------------------------------
def tables_at(history: pd.DataFrame, cities, fit_origin: int) -> dict[str, pd.DataFrame]:
    """One city x year table per metric, cut at the fit origin before anything else reads it."""
    frame = history[history.year <= fit_origin]
    assert_no_post_cutoff(frame, "year", "Placeholder fit history")
    return {metric: wide(frame, metric).reindex(index=cities) for metric in METRICS}


def build_states(tables, neighbours, t_values) -> dict:
    states = {}
    for t in t_values:
        per_metric = {metric: metric_state(table, t) for metric, table in tables.items()}
        states[t] = {"metric": per_metric,
                     "pooled": {metric: pooled_state(per_metric, metric, neighbours) for metric in tables}}
    return states


def training_rows(tables, neighbours, fit_origin: int, feature_set: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Features and shift targets per metric, for pseudo-origins t in [F-45, F-h] with t + h <= F."""
    first = min(int(table.columns.min()) for table in tables.values())
    t_values = range(max(first, fit_origin - TRAIN_YEARS), fit_origin)
    states = build_states(tables, neighbours, t_values)
    rows = {}
    for metric, table in tables.items():
        X, y = [], []
        for h in LEADS:
            target_year_exists = table.columns
            for t in t_values:
                if t + h > fit_origin or t + h not in target_year_exists:
                    continue
                state = states[t]["metric"][metric]
                target = table[t + h].to_numpy(float) - state["ref30"]
                keep = np.isfinite(target) & np.isfinite(state["ref30"])
                if keep.any():
                    features = feature_matrix(state, states[t]["pooled"][metric],
                                              states[t]["metric"]["annual_avg_temp"], h, feature_set)
                    X.append(features[keep])
                    y.append(target[keep])
        if not X:
            raise ValueError(f"no training rows for {metric} at fit origin {fit_origin}")
        rows[metric] = (np.vstack(X), np.concatenate(y))
    return rows


class Members:
    """The two fixed learners of one metric. Each fits once; lambda only rescales shift_hat."""

    def __init__(self, names: list[str], members: str, seed: int):
        self.names, self.members, self.seed = names, members, seed

    def _impute(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for j, name in enumerate(self.names):
            fill = self.medians[j] if name in LEVEL_FEATURES else 0.0
            column = X[:, j]
            column[np.isnan(column)] = fill
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> Members:
        self.medians = np.nanmedian(X, axis=0)
        if self.members in ("ensemble", "lin"):
            imputed = self._impute(X)
            self.scaler = StandardScaler().fit(imputed)
            # Dividing the target by its robust scale only conditions the solver for large-unit metrics;
            # predictions are multiplied back, so the fitted model stays in the metric's own units.
            self.y_scale = max(1.4826 * float(np.median(np.abs(y - np.median(y)))), 1e-6)
            self.lin = HuberRegressor(**HUBER).fit(self.scaler.transform(imputed), y / self.y_scale)
        if self.members in ("ensemble", "gbm"):
            self.gbm = HistGradientBoostingRegressor(random_state=self.seed, **GBM).fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        parts = []
        if self.members in ("ensemble", "lin"):
            parts.append(self.lin.predict(self.scaler.transform(self._impute(X))) * self.y_scale)
        if self.members in ("ensemble", "gbm"):
            parts.append(self.gbm.predict(X))
        return np.mean(parts, axis=0)


def fit_at(history: pd.DataFrame, cities, fit_origin: int, neighbours, seed: int, members: str, feature_set: str):
    """Fit one Members per metric using history <= fit_origin, and return their shifts for the next seven years."""
    tables = tables_at(history, cities, fit_origin)
    rows = training_rows(tables, neighbours, fit_origin, feature_set)
    names = feature_names(feature_set)
    fitted = {metric: Members(names, members, seed).fit(*rows[metric]) for metric in METRICS}
    states = build_states(tables, neighbours, [fit_origin])[fit_origin]
    shifts = {}
    for metric in METRICS:
        state, pooled = states["metric"][metric], states["pooled"][metric]
        by_year = {}
        for h in LEADS:
            features = feature_matrix(state, pooled, states["metric"]["annual_avg_temp"], h, feature_set)
            by_year[fit_origin + h] = fitted[metric].predict(features)
        shift = pd.DataFrame(by_year, index=list(cities))
        shifts[metric] = shift.where(pd.Series(state["ref30"], index=shift.index).notna(), 0.0)
    return fitted, shifts, {metric: len(rows[metric][1]) for metric in METRICS}


# ---------------------------------------------------------------------------------------------
# Assembly and the shrinkage gate
# ---------------------------------------------------------------------------------------------
def assemble(base: pd.DataFrame, shifts: dict, lambdas: dict[str, float]) -> pd.DataFrame:
    """The one path from members to forecasts, used for inner scoring and the outer forecast."""
    parts = []
    for metric in METRICS:
        rows = base[base.metric == metric].copy()
        shift = shifts[metric].rename_axis(index="city_id", columns="year").stack().rename("shift").reset_index()
        rows = rows.merge(shift, on=["city_id", "year"], how="left", validate="one_to_one")
        rows["predicted_value"] = rows.predicted_value + lambdas[metric] * rows["shift"].fillna(0.0)
        if metric in NON_NEGATIVE:
            rows["predicted_value"] = rows.predicted_value.clip(lower=0.0)
        parts.append(rows[KEY + ["predicted_value"]])
    return pd.concat(parts, ignore_index=True)


def select_lambda(mean_skill: dict[float, float]) -> float:
    """Smallest lambda within the margin of the best mean inner skill; 0 when nothing beats the reference."""
    best = max(mean_skill.values())
    if best <= 0:
        return 0.0
    return min(value for value, skill in mean_skill.items() if skill >= best - LAMBDA_MARGIN)


def choose_lambdas(history, cities, targets, origin, neighbours, seed, members, feature_set, fit_history=None):
    from evaluation.adapters import reference
    from evaluation.core import build_fold, validate_predictions
    from evaluation.official_score import score

    fit_history = history if fit_history is None else fit_history   # learners read this; scoring stays official
    inner_origins = [origin - offset for offset in INNER_OFFSETS]
    artifact = {"inner_origins": inner_origins, "grid": list(LAMBDA_GRID), "margin": LAMBDA_MARGIN,
                "rule": "smallest lambda within margin of the best mean unclipped final-analogue skill; 0 if best <= 0",
                "status": "selected", "fallback_reason": None, "inner_folds": [], "mean_final_skill": {}}
    if history.year.min() > inner_origins[0] - 9 or inner_origins[0] < 1:
        artifact.update(status="fallback_insufficient_history",
                        fallback_reason="oldest inner origin has fewer than ten possible prior calendar years",
                        selected={metric: 0.0 for metric in METRICS})
        return artifact["selected"], artifact
    skills = {value: {metric: [] for metric in METRICS} for value in LAMBDA_GRID}
    for inner in inner_origins:
        inner_train, inner_targets, truth, ref = build_fold(history, cities, inner)
        if inner_train.year.max() > inner or inner_targets.year.max() > origin:
            raise ValueError("Placeholder inner schedule crossed its information boundary")
        _, shifts, row_counts = fit_at(fit_history, cities, inner, neighbours, seed, members, feature_set)
        base = reference(inner_train, inner_targets, inner, seed, method="average")
        fold = {"origin": inner, "train_rows": row_counts, "final_skill": {}}
        for value in LAMBDA_GRID:
            predictions = validate_predictions(assemble(base, shifts, {m: value for m in METRICS}), inner_targets)
            result = score(predictions, truth, ref, "final")
            if not all(np.isfinite(skill) for skill in result["skill"].values()):
                raise ValueError(f"inner origin {inner}: undefined skill for lambda {value}")
            fold["final_skill"][str(value)] = result["skill"]
            for metric in METRICS:
                skills[value][metric].append(result["skill"][metric])
        artifact["inner_folds"].append(fold)
    selected = {}
    for metric in METRICS:
        means = {value: float(np.mean(skills[value][metric])) for value in LAMBDA_GRID}
        artifact["mean_final_skill"][metric] = {str(k): v for k, v in means.items()}
        selected[metric] = select_lambda(means)
    artifact["selected"] = selected
    return selected, artifact


# ---------------------------------------------------------------------------------------------
# Harness entry point
# ---------------------------------------------------------------------------------------------
def run_forecast(train, targets, origin, seed, coords, members="ensemble", gate="inner", features="all",
                 cleaned=None, inputs=None):
    """Forecast given coordinates; `forecast` adds the asset load and writes artifacts."""
    from evaluation.adapters import reference

    if members not in MEMBERS or gate not in GATES or features not in FEATURE_SETS:
        raise ValueError(f"unknown option: members={members!r}, gate={gate!r}, features={features!r}")
    assert_no_post_cutoff(train, "year", "Placeholder training input")
    if train.year.max() > origin:
        raise ValueError("Placeholder received observations after its origin")
    cities = sorted(targets.city_id.unique())
    history = train.copy()
    history["year"] = pd.to_numeric(history.year).astype(int)
    inputs = resolve_inputs(inputs)
    fit_history = apply_inputs(history, cleaned, inputs) if uses_cleaned(inputs) else history
    neighbours = neighbour_index(_check_coords(coords, cities))
    if gate == "inner":
        lambdas, record = choose_lambdas(history, cities, targets, origin, neighbours, seed, members, features,
                                         fit_history)
    else:
        lambdas = {metric: 1.0 for metric in METRICS}
        record = {"status": "gate_disabled", "selected": lambdas}
    fitted, shifts, row_counts = fit_at(fit_history, cities, origin, neighbours, seed, members, features)
    base = reference(history, targets, origin, seed, method="average")
    output = assemble(base, shifts, lambdas)[KEY + ["predicted_value"]]
    record.update(model=MODEL_NAME, outer_origin=origin, seed=seed, members=members, gate=gate, features=features,
                  feature_names=feature_names(features), train_years=TRAIN_YEARS, n_neighbours=N_NEIGHBOURS,
                  huber=HUBER, gbm=GBM, outer_train_rows=row_counts, inputs=inputs,
                  clean_release=CLEAN_RELEASE if uses_cleaned(inputs) else None)
    return output, record, fitted


def forecast(train, targets, origin, seed=42, members="ensemble", gate="inner", features="all", inputs=None):
    cities = sorted(targets.city_id.unique())
    inputs = resolve_inputs(inputs)
    cleaned = load_cleaned(origin, inputs) if uses_cleaned(inputs) else None
    output, record, fitted = run_forecast(train, targets, origin, seed, load_coords(cities), members, gate, features,
                                          cleaned, inputs)
    directory = os.environ.get("PROTOSTAR_ARTIFACT_DIR")
    if directory:
        import joblib

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "placeholder_selection.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n",
                                                         encoding="utf-8")
        joblib.dump(fitted, path / "placeholder_models.joblib", compress=3)
    return output
