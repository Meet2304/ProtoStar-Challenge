"""Select forecasting methods inside each outer historical training period.

This evaluates a *selection procedure*, rather than reusing combo_model.METHOD,
whose choices were made with labels later used by the original outer suite.
All inner targets come from the supplied outer training frame. No data loader
or outer answer key is read here.

Parameters: inner_folds (1..10, default 3) and candidates (a unique nonempty
subset of average/trend/last10/blend). Inner windows are seven years long,
nonoverlapping, and end at the outer origin. Per metric, maximize equal-weight
mean unclipped final-analogue skill; ties follow the canonical method order.

If the requested oldest inner origin cannot have ten calendar years of prior
history, skip selection altogether and use the thirty-year-average method for
every metric. This is a predeclared training-only fallback, not performance-
based fold selection. Other unusable inner evaluations fail explicitly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import METRICS, assert_no_post_cutoff, wide
from combo_model import FORECASTERS

METHOD_ORDER = ("average", "trend", "last10", "blend")
KEY = ["city_id", "year", "metric"]


def _parameters(params):
    unknown = set(params) - {"inner_folds", "candidates"}
    if unknown:
        raise ValueError(f"nested adapter: unknown parameters {sorted(unknown)}")
    inner_folds = params.get("inner_folds", 3)
    if type(inner_folds) is not int or not 1 <= inner_folds <= 10:
        raise ValueError("nested adapter: inner_folds must be an integer from 1 to 10")
    candidates = params.get("candidates", list(METHOD_ORDER))
    if (not isinstance(candidates, (list, tuple)) or not candidates
            or any(not isinstance(c, str) or c not in METHOD_ORDER for c in candidates)
            or len(set(candidates)) != len(candidates)):
        raise ValueError("nested adapter: candidates must be a unique nonempty list drawn from "
                         "average, trend, last10, blend")
    # Canonical order makes tie behavior independent of a JSON list's ordering.
    return inner_folds, [method for method in METHOD_ORDER if method in candidates]


def _validate_inputs(train, targets, origin):
    if type(origin) is not int or not 1 <= origin <= 2018:
        raise ValueError("nested adapter: origin must be an integer no later than 2018")
    required = {"city_id", "year", *METRICS}
    if not required <= set(train) or train.empty:
        raise ValueError("nested adapter requires nonempty annual history with all four metrics")
    assert_no_post_cutoff(train, "year", "nested outer training input")
    years = pd.to_numeric(train.year, errors="raise")
    if (not np.isfinite(years).all() or (years % 1 != 0).any()
            or years.max() > origin or train.city_id.isna().any()
            or train.duplicated(["city_id", "year"]).any()):
        raise ValueError("nested adapter: invalid/duplicate training keys or observations after outer origin")
    if list(targets.columns) != KEY or targets[KEY].isna().any(axis=None):
        raise ValueError("nested adapter targets must contain only city_id, year, metric")
    target_years = pd.to_numeric(targets.year, errors="raise")
    if (not np.isfinite(target_years).all() or (target_years % 1 != 0).any()
            or set(target_years) != set(range(origin + 1, origin + 8))
            or set(targets.metric) != set(METRICS)
            or targets.duplicated(KEY).any()
            or len(targets) != targets.city_id.nunique() * 7 * len(METRICS)):
        raise ValueError("nested adapter targets require the complete city x next-seven-years x metric grid")


def _predict_methods(train, targets, origin, methods):
    """Fit each selected method using only rows at/before this fit's origin."""
    assert_no_post_cutoff(train, "year", "nested method history")
    if train.year.max() > origin:
        raise ValueError("nested method received observations after its origin")
    years = list(range(origin + 1, origin + 8))
    cities = sorted(targets.city_id.unique())
    parts = []
    for metric in METRICS:
        table = wide(train, metric).reindex(cities)
        usable = table.to_numpy(dtype=float)
        finite = usable[np.isfinite(usable)]
        if not len(finite):
            raise ValueError(f"nested adapter: no finite training values for {metric}")
        values = FORECASTERS[methods[metric]](table, origin, years)
        # Missing histories affect mandatory rows too. Fill them using only
        # this fit's metric-wide training median, never target observations.
        values = values.reindex(index=cities, columns=years).fillna(float(np.median(finite)))
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"nested adapter: method {methods[metric]} returned invalid {metric} values")
        long = (values.rename_axis(index="city_id", columns="year").stack()
                .rename("predicted_value").reset_index())
        long["metric"] = metric
        parts.append(long)
    return pd.concat(parts, ignore_index=True)[KEY + ["predicted_value"]]


def _save_artifact(artifact):
    directory = os.environ.get("PROTOSTAR_ARTIFACT_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "nested_selection.json").write_text(
            json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def forecast(train, targets, origin, seed=42, **params):
    """Return complete forecasts after choosing methods using earlier inner folds.

    The seed is accepted for the shared adapter contract; these four statistical
    candidates and the selection/tie rules are deterministic without randomness.
    """
    inner_count, candidates = _parameters(params)
    _validate_inputs(train, targets, origin)
    # Lazy imports avoid making the evaluator depend on a specific model.
    from evaluation.core import build_fold, validate_predictions
    from evaluation.official_score import score

    history = train.copy()
    history["year"] = pd.to_numeric(history.year).astype(int)
    cities = sorted(targets.city_id.unique())
    inner_origins = [origin - 7 * count for count in range(inner_count, 0, -1)]
    artifact = {
        "schema_version": 1, "outer_origin": origin, "seed": seed,
        "requested_inner_origins": inner_origins, "candidates": candidates,
        "selection_rule": "per metric: maximize equal-weight mean unclipped final (leads 4-7) skill",
        "tie_break_order": candidates,
        "history_min_year": int(history.year.min()), "history_max_year": int(history.year.max()),
        "fallback_rule": "if oldest requested inner origin has fewer than ten possible calendar history years, use average for all metrics without selecting or skipping folds",
        "inner_folds": [], "candidate_mean_final_skill": {},
        "status": "selected", "fallback_reason": None,
    }
    if history.year.min() > inner_origins[0] - 9 or inner_origins[0] < 1:
        selected = {metric: "average" for metric in METRICS}
        artifact.update(status="fallback_insufficient_history",
                        fallback_reason="requested inner schedule lacks ten possible prior calendar years")
    else:
        skills = {candidate: {metric: [] for metric in METRICS} for candidate in candidates}
        for inner_origin in inner_origins:
            # All labels in this build are inside outer history. Every candidate
            # sees only inner_train, not the surrounding history or inner truth.
            try:
                inner_train, inner_targets, truth, ref = build_fold(history, cities, inner_origin)
            except ValueError as exc:
                raise ValueError(f"nested inner origin {inner_origin} is unusable: {exc}") from exc
            if inner_train.year.max() > inner_origin or inner_targets.year.max() > origin:
                raise ValueError("nested inner schedule crossed its information boundary")
            fold_result = {
                "origin": inner_origin, "train_max_year": int(inner_train.year.max()),
                "target_min_year": int(inner_targets.year.min()),
                "target_max_year": int(inner_targets.year.max()),
                "scored_final_rows": int((truth.scored & (truth.split == "final")).sum()),
                "candidate_final_skill": {},
            }
            for candidate in candidates:
                predictions = _predict_methods(
                    inner_train, inner_targets, inner_origin, {m: candidate for m in METRICS})
                predictions = validate_predictions(predictions, inner_targets)
                result = score(predictions, truth, ref, "final")
                if not all(np.isfinite(value) for value in result["skill"].values()):
                    raise ValueError(f"nested inner origin {inner_origin}: undefined {candidate} skill")
                fold_result["candidate_final_skill"][candidate] = result["skill"]
                for metric in METRICS:
                    skills[candidate][metric].append(result["skill"][metric])
            artifact["inner_folds"].append(fold_result)
        means = {candidate: {metric: float(np.mean(values)) for metric, values in metric_skills.items()}
                 for candidate, metric_skills in skills.items()}
        selected = {metric: max(candidates, key=lambda candidate: means[candidate][metric])
                    for metric in METRICS}
        artifact["candidate_mean_final_skill"] = means
    artifact["selected_methods"] = selected
    predictions = validate_predictions(_predict_methods(history, targets, origin, selected), targets)
    _save_artifact(artifact)
    return predictions
