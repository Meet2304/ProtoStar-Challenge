"""Descriptive, training-era label sensitivities; never part of model execution.

The caller enforces the development-phase gate and verifies file provenance. This
module performs no I/O and never obtains observations. It does not redefine the
official mask, forecasts, or score. All scenarios reuse historical labels.
"""
from __future__ import annotations

import calendar
import copy
import hashlib
import json

import numpy as np
import pandas as pd

from common import METRICS, assert_no_post_cutoff, thirty_year_mean, wide
from evaluation.official_score import KEY, score


VARIANTS = (
    "default", "station_stable_target", "station_stable_baseline_target",
    "cdd_days_both_strict", "temperature_whole_f_rounding",
)
CAVEATS = [
    "Development diagnostics only; the caller must enforce phase authorization.",
    "All source labels are reused historical observations; these are not independent evidence.",
    "Exclusion changes sample composition; differences cannot establish station-change causality.",
    "Neither these scenarios nor the training precision certify the private answer key, mask, or precision.",
    "Predictions are fixed in every scenario. Diagnostic scores do not replace official-style results.",
]


def _years_and_keys(table, city, year, source):
    required = {city, year}
    if not required <= set(table):
        raise ValueError(f"{source} missing key columns: {sorted(required - set(table))}")
    out = table.copy(deep=True)
    assert_no_post_cutoff(out, year, source)
    if out[city].isna().any() or out[city].astype(str).str.strip().eq("").any():
        raise ValueError(f"{source}: city IDs must be nonempty")
    out[city] = out[city].astype(str)
    out[year] = pd.to_numeric(out[year], errors="raise").astype(int)
    if out[year].lt(1).any() or out.duplicated([city, year]).any():
        raise ValueError(f"{source}: invalid year or duplicate city-years")
    return out


def _annual(table):
    """Support raw official annual_indices or the evaluator's masked wide schema."""
    raw = {"ID", "Year"} <= set(table)
    out = _years_and_keys(table, "ID" if raw else "city_id",
                          "Year" if raw else "year", "quality annual")
    source_schema = "annual_indices" if raw else "evaluation_annual_masked"
    missing_counts = None
    if raw:
        required = {c for value, missing in METRICS.values() for c in [value, *missing]}
        if not required <= set(out):
            raise ValueError(f"annual input missing columns: {sorted(required - set(out))}")
        days = out.Year.map(lambda y: 365 + calendar.isleap(y))
        missing_counts = out[["ID", "Year", "Miss Tmax", "Miss Tmin"]].rename(
            columns={"ID": "city_id", "Year": "year"})
        missing_counts[["Miss Tmax", "Miss Tmin"]] = missing_counts[["Miss Tmax", "Miss Tmin"]].apply(
            pd.to_numeric, errors="raise")
        result = out[["ID", "Year"]].rename(columns={"ID": "city_id", "Year": "year"})
        for metric, (column, missing) in METRICS.items():
            counts = out[missing].apply(pd.to_numeric, errors="raise")
            if (counts.isna().any(axis=None) or not np.isfinite(counts).all(axis=None)
                    or (counts % 1 != 0).any(axis=None) or counts.lt(0).any(axis=None)
                    or counts.gt(days, axis=0).any(axis=None)):
                raise ValueError("annual missing-day counts must be integers within the calendar year")
            values = pd.to_numeric(out[column], errors="raise")
            valid = counts.le(10).all(axis=1)
            if np.isinf(values[valid]).any():
                raise ValueError(f"infinite valid annual value: {column}")
            result[metric] = values.where(valid)
        out = result
    else:
        if not set(METRICS) <= set(out):
            raise ValueError(f"annual input missing metrics: {sorted(set(METRICS) - set(out))}")
        out = out[["city_id", "year", *METRICS]].copy()
        for metric in METRICS:
            out[metric] = pd.to_numeric(out[metric], errors="raise")
            if np.isinf(out[metric]).any():
                raise ValueError(f"infinite annual value: {metric}")
    return out.sort_values(["city_id", "year"]).reset_index(drop=True), source_schema, missing_counts


def _fingerprint(table):
    # Canonical row/key order, independent of incoming index and line endings.
    return hashlib.sha256(table.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def _coverage(table, missing_counts):
    if table is None:
        return None, {"station": "coverage_and_stations was not provided",
                      "days_both": "coverage_and_stations was not provided"}, {}
    raw = {"ID", "Year"} <= set(table)
    out = _years_and_keys(table, "ID" if raw else "city_id",
                          "Year" if raw else "year", "quality coverage")
    if raw:
        out = out.rename(columns={"ID": "city_id", "Year": "year"})
    out = out.sort_values(["city_id", "year"]).reset_index(drop=True)
    errors = {}
    for kind, columns in (("station", ["Station Segment", "Station Changes"]),
                          ("days_both", ["Days Both"])):
        absent = set(columns) - set(out)
        if absent:
            errors[kind] = f"coverage missing columns: {sorted(absent)}"
            continue
        values = out[columns].apply(pd.to_numeric, errors="coerce")
        invalid = values.isna().any(axis=1) | ~np.isfinite(values).all(axis=1)
        invalid |= (values % 1 != 0).any(axis=1) | values.lt(0).any(axis=1)
        if kind == "station":
            invalid |= values["Station Segment"].lt(1)
            invalid |= values["Station Changes"].gt(out.year.map(lambda y: 365 + calendar.isleap(y)))
        else:
            days = out.year.map(lambda y: 365 + calendar.isleap(y))
            invalid |= values["Days Both"].gt(days)
        if invalid.any():
            errors[kind] = f"invalid {kind} metadata in {int(invalid.sum())} coverage rows"
        else:
            out[columns] = values.astype(int)
    if missing_counts is not None and "days_both" not in errors:
        matched = out.merge(missing_counts, on=["city_id", "year"], how="inner", validate="one_to_one")
        days = matched.year.map(lambda y: 365 + calendar.isleap(y))
        # The intersection cannot exceed either marginal or be below their union bound.
        impossible = (matched["Days Both"] > days - matched[["Miss Tmax", "Miss Tmin"]].max(axis=1))
        impossible |= matched["Days Both"] < days - matched["Miss Tmax"] - matched["Miss Tmin"]
        if impossible.any():
            errors["days_both"] = f"Days Both conflicts with annual missing counts in {int(impossible.sum())} rows"
    return out, errors, {"rows": len(out), "columns": list(table.columns),
                         "normalized_sha256": _fingerprint(out)}


def _answer(annual, cities, origin):
    targets = pd.MultiIndex.from_product([cities, range(origin + 1, origin + 8), list(METRICS)],
                                         names=KEY).to_frame(index=False)
    truth = annual.melt(id_vars=["city_id", "year"], value_vars=list(METRICS),
                        var_name="metric", value_name="actual_value")
    answer = targets.merge(truth, on=KEY, how="left", validate="one_to_one")
    answer["split"] = np.where(answer.year <= origin + 3, "live", "final")
    history = annual[(annual.year <= origin) & annual.city_id.isin(cities)]
    levels = []
    if not history.empty:
        for metric in METRICS:
            level = thirty_year_mean(wide(history, metric), origin).rename("baseline_value").reset_index()
            level["metric"] = metric
            levels.append(level)
    if levels:
        answer = answer.merge(pd.concat(levels), on=["city_id", "metric"], how="left", validate="many_to_one")
    else:
        answer["baseline_value"] = np.nan
    answer["scored"] = answer.actual_value.notna() & answer.baseline_value.notna()
    return targets, answer


def _evaluate(answer, predictions, unavailable=None):
    rows = answer.merge(predictions, on=KEY, validate="one_to_one")
    result = {}
    for split in ("live", "final"):
        all_rows = rows[rows.split.eq(split)]
        eligible = all_rows[all_rows.scored]
        # Same grouped reduction as the official scorer (including summation precision).
        maes = (eligible.predicted_value - eligible.actual_value).abs().groupby(eligible.metric).mean()
        metrics = {}
        for metric in METRICS:
            selected = all_rows[all_rows.metric.eq(metric) & all_rows.scored]
            reason = unavailable.get(metric) if unavailable else None
            n = len(selected) if reason is None else 0
            mae = float(maes[metric]) if n else None
            reference = float((selected.baseline_value - selected.actual_value).abs().mean()) if n else None
            if reason is None and not n:
                reason = "no eligible target rows with at least 10 valid baseline years"
            if reason is None and reference == 0:
                reason = "zero reference MAE; skill is undefined"
            metrics[metric] = {"n_rows": int(n), "mae": mae, "reference_mae": reference,
                               "skill": 1 - mae / reference if reason is None else None,
                               "available": reason is None, "unavailable_reason": reason}
        complete = all(m["available"] for m in metrics.values())
        raw = sum(m["skill"] for m in metrics.values()) / len(METRICS) if complete else None
        result[split] = {"n_rows": sum(m["n_rows"] for m in metrics.values()), "metrics": metrics,
                         "mean_skill_unclipped": raw,
                         "diagnostic_score": 100 * max(0., raw) if complete else None,
                         "available": complete,
                         "unavailable_reason": None if complete else "one or more metrics unavailable"}
        # Preserve the pure official scorer for the unchanged default-compatible calculation.
        if complete:
            ref = pd.DataFrame([{"split": split, **{m: metrics[m]["reference_mae"] for m in METRICS}}])
            result[split]["diagnostic_score"] = score(predictions, answer, ref, split)["score"]
    return result


def _station_cities(coverage, cities, first, last):
    kept, excluded = [], {"missing_coverage": [], "station_change_or_segment_difference": []}
    for city in cities:
        group = coverage[coverage.city_id.eq(city) & coverage.year.between(first, last)]
        if set(group.year) != set(range(first, last + 1)):
            excluded["missing_coverage"].append(city)
        elif group["Station Changes"].gt(0).any() or group["Station Segment"].nunique() != 1:
            excluded["station_change_or_segment_difference"].append(city)
        else:
            kept.append(city)
    return kept, excluded


def diagnose_origin(annual, coverage, origin, predictions, *, cities=None, provenance=None):
    """Return finite JSON diagnostics for one seven-year historical origin.

    ``annual`` is raw annual_indices or evaluator-normalized masked annual data;
    ``coverage`` is the corresponding coverage_and_stations table (or None).
    Predictions must cover the complete city/year/metric grid in standard order
    of columns. Inputs are copied. Missing or invalid coverage disables affected
    diagnostics explicitly; it never supplies an assumption of station stability.
    Caller-supplied provenance is descriptive and is not independently verified.
    """
    from evaluation.core import validate_predictions

    if type(origin) is not int or origin < 30 or origin + 7 > 2018:
        raise ValueError("diagnostic origin needs a 30-year baseline and targets ending by 2018")
    data, schema, missing_counts = _annual(annual)
    cov, coverage_errors, cov_source = _coverage(coverage, missing_counts)
    if cities is None:
        if "city_id" not in predictions:
            raise ValueError("predictions missing city_id")
        cities = sorted(predictions.city_id.dropna().astype(str).unique().tolist())
    else:
        cities = list(cities)
        if any(not isinstance(c, str) or not c.strip() for c in cities) or len(set(cities)) != len(cities):
            raise ValueError("cities must be unique nonempty strings")
        cities.sort()
    if not cities or set(cities) - set(data.city_id):
        raise ValueError("diagnostic cities must be nonempty and present in annual input")
    targets, default_answer = _answer(data, cities, origin)
    pred = validate_predictions(predictions, targets)
    defaults = _evaluate(default_answer, pred)
    variants = {"default": {"label": "Unchanged official-style historical mask and baseline",
                              "splits": defaults, "exclusions": {}, "coverage_status": "not_required"}}
    for name, first in (("station_stable_target", origin + 1),
                         ("station_stable_baseline_target", origin - 29)):
        reason = coverage_errors.get("station")
        exclusions = {}
        altered = default_answer.copy(deep=True)
        if reason:
            unavailable = {metric: reason for metric in METRICS}
        else:
            kept, exclusions = _station_cities(cov, cities, first, origin + 7)
            altered["scored"] &= altered.city_id.isin(kept)
            unavailable = None
        variants[name] = {"label": "Cities with complete coverage, zero listed changes and one segment in window",
                          "station_window": [first, origin + 7], "exclusions": exclusions,
                          "coverage_status": "unavailable" if reason else (
                              "partial" if exclusions.get("missing_coverage") else "complete"),
                          "coverage_unavailable_reason": reason,
                          "splits": _evaluate(altered, pred, unavailable)}
    strict = data.copy(deep=True)
    reason = coverage_errors.get("days_both")
    exclusions = {"baseline_city_years_failed_days_both": 0, "target_city_years_failed_days_both": 0}
    if not reason:
        relevant = strict.city_id.isin(cities) & strict.year.between(origin - 29, origin + 7) & strict.cdd.notna()
        joined = strict.loc[relevant, ["city_id", "year"]].merge(
            cov[["city_id", "year", "Days Both"]], on=["city_id", "year"], how="left", validate="one_to_one")
        missing = joined["Days Both"].isna()
        exclusions["missing_coverage_city_years"] = int(missing.sum())
        if missing.any():
            reason = f"Days Both missing for {int(missing.sum())} otherwise valid CDD city-years"
        else:
            joined["days_missing_either_temperature"] = joined.year.map(
                lambda y: 365 + calendar.isleap(y)) - joined["Days Both"]
            failed = joined[joined.days_missing_either_temperature.gt(10)]
            exclusions["baseline_city_years_failed_days_both"] = int(failed.year.le(origin).sum())
            exclusions["target_city_years_failed_days_both"] = int(failed.year.gt(origin).sum())
            failed_keys = pd.MultiIndex.from_frame(failed[["city_id", "year"]])
            strict.loc[pd.MultiIndex.from_frame(strict[["city_id", "year"]]).isin(failed_keys), "cdd"] = np.nan
    _, strict_answer = _answer(strict, cities, origin)
    cdd_baseline = []
    if not reason:
        for city in cities:
            old = data[data.city_id.eq(city) & data.year.between(origin - 29, origin)].cdd.dropna()
            new = strict[strict.city_id.eq(city) & strict.year.between(origin - 29, origin)].cdd.dropna()
            cdd_baseline.append({"city_id": city, "default_valid_years": len(old),
                                 "strict_valid_years": len(new),
                                 "default_baseline": float(old.mean()) if len(old) >= 10 else None,
                                 "strict_baseline": float(new.mean()) if len(new) >= 10 else None})
    variants["cdd_days_both_strict"] = {
        "label": "CDD: at most 10 days missing either Tmax or Tmin; recalculate 30-year baseline with >=10 valid years",
        "exclusions": exclusions, "coverage_status": "unavailable" if reason else "complete",
        "coverage_unavailable_reason": reason,
        "cdd_baseline_by_city": cdd_baseline,
        "splits": _evaluate(strict_answer, pred, {"cdd": reason} if reason else None)}
    rounded = data.copy(deep=True)
    rounded_metrics = ["annual_avg_temp", "hottest_day_temp"]
    rounded[rounded_metrics] = rounded[rounded_metrics].round(0)
    _, rounded_answer = _answer(rounded, cities, origin)
    variants["temperature_whole_f_rounding"] = {
        "label": "Hypothetical whole-degree F label/baseline rounding; fixed predictions",
        "rounding": {"metrics": rounded_metrics, "decimals": 0, "ties": "nearest_even",
                     "applies_to": "baseline history and target labels; predictions unchanged"},
        "exclusions": {}, "coverage_status": "not_required",
        "splits": _evaluate(rounded_answer, pred)}
    for name, variant in variants.items():
        for split, values in variant["splits"].items():
            values["default_scored_rows"] = defaults[split]["n_rows"]
            values["row_count_delta_vs_default"] = values["n_rows"] - defaults[split]["n_rows"]
            for field in ("mean_skill_unclipped", "diagnostic_score"):
                values[f"{field}_delta_vs_default"] = (
                    values[field] - defaults[split][field]
                    if values[field] is not None and defaults[split][field] is not None else None)
            for metric, result in values["metrics"].items():
                baseline = defaults[split]["metrics"][metric]
                result["n_rows_delta_vs_default"] = result["n_rows"] - baseline["n_rows"]
                for field in ("mae", "reference_mae", "skill"):
                    result[f"{field}_delta_vs_default"] = (
                        result[field] - baseline[field] if result[field] is not None and baseline[field] is not None else None)
            variant["default_eligible_rows_excluded"] = {
                key: defaults[key]["n_rows"] - variant["splits"][key]["n_rows"] for key in ("live", "final")}
        if name == "default":
            for values in variant["splits"].values():
                values["official_style_score"] = values["diagnostic_score"]
    training_years = data.loc[data.city_id.isin(cities) & data.year.le(origin), "year"]
    result = {"schema_version": 1, "kind": "development_scientific_sensitivity",
              "origin": origin, "baseline_window": [origin - 29, origin],
              "target_window": [origin + 1, origin + 7],
              "training_window": [int(training_years.min()), origin] if len(training_years) else None,
              "cities": cities,
              "source": {"annual_schema": schema, "annual_rows": len(data),
                         "annual_normalized_sha256": _fingerprint(data), "coverage": cov_source,
                         "coverage_issues": coverage_errors,
                         "caller_provenance": copy.deepcopy(provenance or {})},
              "history_status": "reused_historical", "variants": variants, "caveats": CAVEATS.copy()}
    # Fail closed if a caller's provenance or unexpected arithmetic is not finite JSON.
    json.dumps(result, allow_nan=False)
    return result


def metrics_frame(result):
    """Return a tidy per-origin/variant/split/metric table suitable for CSV."""
    return pd.DataFrame([
        {"origin": result["origin"], "variant": name, "split": split, "metric": metric, **values}
        for name, variant in result["variants"].items()
        for split, scores in variant["splits"].items()
        for metric, values in scores["metrics"].items()
    ])
