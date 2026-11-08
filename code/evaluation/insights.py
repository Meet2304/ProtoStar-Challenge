"""Model insights: where a run loses points, why, how much a fix could win back, and what to try.

The reward (reward.py) says HOW GOOD a run is. This module says WHAT TO DO NEXT. It reads only the
sealed run's ``row_errors.csv`` for development origins (years <= 2004) plus two analysis-only tables
(``data/city_info.csv`` for zones, the raw coverage table for station changes). It never changes a
score, the evaluator (eval-vN) or the reward, and it never reads 2019-2025 observations.

Every number is in HEADLINE POINTS, the reward's unit: 100 x mean over origins of mean over the four
metrics of skill. One metric's points move the headline by a quarter of their size. City and zone
contributions are additive: they sum exactly to the headline.

Sections (final analogue = leads 4-7, the reward's split, unless stated):
  headroom      per metric: current points, CEILING (a perfect constant per city over the window, i.e.
                only year-to-year weather left) and what-if gains from simple corrections.
  calibration   anomaly view vs the 30-year average: correlation, regression slope (too bold / too
                timid), share of the observed shift the model captured, mean bias and its growth by lead.
  decomposition signed error split into common-year (all cities share it: national weather of that
                year), persistent-city (a city's level is off) and residual parts.
  lead          points and bias for each lead 1-7 (live and final analogues together).
  segments      zones, station-change cities, origins, worst cities and how concentrated the losses are.
  versus        deltas against the reward champion and the best control on each metric.
  insights      ranked findings: {finding, evidence, suggestion, headroom_points, confidence}.

What-if gains are fitted on the very rows they are scored on, so they are OPTIMISTIC UPPER BOUNDS.
Never copy a fitted correction into a model: build it from training data inside each origin, then
confirm with a new run and its reward verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import METRICS, ROOT  # noqa: E402
from evaluation.core import verify_run  # noqa: E402

INSIGHTS_VERSION = "insights-v1"
INSIGHTS_DIR = ROOT / "outputs/evaluation/insights"
RUNS = ROOT / "outputs/evaluation/runs"
UNITS = {"annual_avg_temp": "F", "total_precip_in": "in", "hottest_day_temp": "F", "cdd": "F-days"}
WARMING = ("annual_avg_temp", "hottest_day_temp", "cdd")
MIN_CITIES = 10            # smallest segment reported as a finding
NOTABLE = 0.25             # headline points: smallest gain or loss worth a finding
WORST_CITIES = 10


# ---------------------------------------------------------------- loading and core arithmetic
def load_rows(run_dir) -> tuple[dict, pd.DataFrame]:
    run = Path(run_dir)
    meta = verify_run(run)
    if meta.get("phase", "development") != "development":
        raise ValueError("insights read development runs only; assessment stays sealed")
    origins = list(meta.get("origins", meta["config"]["origins"]))
    parts = []
    for origin in origins:
        frame = pd.read_csv(run / f"origin-{origin}/row_errors.csv", dtype={"city_id": str})
        parts.append(frame[frame.scored.astype(bool)])
    rows = pd.concat(parts, ignore_index=True)
    rows["error"] = rows.predicted_value - rows.actual_value                 # + means forecast too high
    rows["pred_anomaly"] = rows.predicted_value - rows.baseline_value        # departure from 30-yr average
    rows["obs_anomaly"] = rows.actual_value - rows.baseline_value
    return meta, rows


def _reference(rows: pd.DataFrame) -> pd.Series:
    """Sum of reference absolute error per (origin, metric): the denominator of every skill."""
    return rows.groupby(["origin", "metric"]).baseline_absolute_error.sum()


def metric_points(rows: pd.DataFrame, abs_error: pd.Series | None = None) -> dict:
    """Points per metric (100 x skill, mean over origins). abs_error substitutes a what-if error."""
    err = rows.absolute_error if abs_error is None else abs_error
    frame = pd.DataFrame({"origin": rows.origin, "metric": rows.metric, "e": err,
                          "r": rows.baseline_absolute_error})
    sums = frame.groupby(["metric", "origin"])[["e", "r"]].sum()
    skill = (1 - sums.e / sums.r).groupby("metric").mean()
    return {m: float(100 * skill[m]) for m in METRICS if m in skill.index}


def headline(points: dict) -> float | None:
    return float(np.mean([points[m] for m in METRICS])) if all(m in points for m in METRICS) else None


def contributions(rows: pd.DataFrame, by, reference: pd.Series | None = None) -> pd.Series:
    """Additive headline points per group: sum over groups equals the headline exactly."""
    reference = _reference(rows) if reference is None else reference
    n_origins = rows.origin.nunique()
    gain = rows.baseline_absolute_error - rows.absolute_error
    denom = reference.reindex(pd.MultiIndex.from_arrays([rows.origin, rows.metric])).to_numpy()
    share = 100 * gain.to_numpy() / denom / (len(METRICS) * n_origins)
    return pd.Series(share, index=rows.index).groupby([rows[c] for c in ([by] if isinstance(by, str) else by)]).sum()


def _whatif(rows: pd.DataFrame, new_prediction: pd.Series) -> dict:
    """Headline-point gain per metric if predictions were replaced (only metrics present in the rows)."""
    now = metric_points(rows)
    then = metric_points(rows, (new_prediction - rows.actual_value).abs())
    return {m: (then[m] - now[m]) / len(METRICS) for m in now}


# ---------------------------------------------------------------- sections
def _slope(sub: pd.DataFrame) -> float | None:
    """Least-squares scaling of the predicted departure that best matches the observed departure."""
    denominator = float((sub.pred_anomaly ** 2).sum())
    if denominator <= 0 or sub.pred_anomaly.abs().max() <= 1e-9:
        return None
    return float((sub.pred_anomaly * sub.obs_anomaly).sum() / denominator)


def _bias_corrections(final: pd.DataFrame, honest: bool) -> pd.Series:
    """Per-row mean error to subtract: from the same origin (fit) or from the OTHER origins (honest)."""
    if not honest:
        return final.groupby(["origin", "metric"]).error.transform("mean")
    sums = final.groupby(["metric", "origin"]).error.agg(["sum", "count"])
    total = sums.groupby("metric").sum()
    out = pd.Series(np.nan, index=final.index)
    for (metric, origin), row in sums.iterrows():
        rest = total.loc[metric] - row
        if rest["count"]:
            mask = (final.metric == metric) & (final.origin == origin)
            out[mask] = rest["sum"] / rest["count"]
    return out.fillna(0.0)


def _scaled(final: pd.DataFrame, honest: bool) -> pd.Series:
    """Prediction with departures rescaled: slope fitted on the same rows (fit) or the other origins (honest)."""
    out = final.predicted_value.copy()
    for metric, sub in final.groupby("metric"):
        for origin in sub.origin.unique():
            rows = sub[sub.origin == origin]
            slope = _slope(sub[sub.origin != origin] if honest else sub)
            if slope is not None:
                out[rows.index] = rows.baseline_value + slope * rows.pred_anomaly
    return out


def _city_corrections(final: pd.DataFrame, honest: bool) -> pd.Series:
    """Per-row city offset: the city's mean error in the window (fit) or in its OTHER years (honest)."""
    keys = ["origin", "metric", "city_id"]
    grouped = final.groupby(keys).error
    if not honest:
        return grouped.transform("mean")
    total, count = grouped.transform("sum"), grouped.transform("count")
    return ((total - final.error) / (count - 1)).where(count > 1, 0.0)


def headroom_section(final: pd.DataFrame) -> dict:
    """Per metric: points now, the ceiling, and what-if gains, each FIT in-sample and HONEST (cross-fitted)."""
    now = metric_points(final)
    keys = ["origin", "metric", "city_id"]
    # Ceiling: the best single number per city over the window (median minimises absolute error). It assumes
    # the future window is known, so it is an unreachable bound that only shows how much year-to-year noise is left.
    ceiling_pred = final.groupby(keys).actual_value.transform("median")
    ceiling = metric_points(final, (ceiling_pred - final.actual_value).abs())
    gains = {}
    for honest in (False, True):
        tag = "honest" if honest else "fit"
        gains[f"remove_mean_bias_{tag}"] = _whatif(final, final.predicted_value - _bias_corrections(final, honest))
        gains[f"anomaly_scaling_{tag}"] = _whatif(final, _scaled(final, honest))
        gains[f"remove_city_offsets_{tag}"] = _whatif(final, final.predicted_value - _city_corrections(final, honest))
    out = {}
    for metric in METRICS:
        out[metric] = {"points": now[metric], "headline_share": now[metric] / len(METRICS),
                       "ceiling_points": ceiling[metric],
                       "headroom_headline_points": (ceiling[metric] - now[metric]) / len(METRICS),
                       **{f"whatif_{k}": v.get(metric) for k, v in gains.items()}}
    return out


def calibration_section(final: pd.DataFrame, rows: pd.DataFrame) -> dict:
    out = {}
    for metric in METRICS:
        sub = final[final.metric == metric]
        pa, oa = sub.pred_anomaly, sub.obs_anomaly
        flat = pa.abs().max() <= 1e-9
        ref_mae = float(sub.baseline_absolute_error.mean())
        bias_by_lead = rows[rows.metric == metric].groupby("lead").error.mean()
        trend = float(np.polyfit(bias_by_lead.index, bias_by_lead.to_numpy(), 1)[0]) if len(bias_by_lead) > 2 else None
        observed_shift = float(oa.mean())
        # Pattern skill within each origin, then averaged: pooling origins would mix their mean shifts.
        within = [float(np.corrcoef(g.pred_anomaly, g.obs_anomaly)[0, 1]) for _, g in sub.groupby("origin")
                  if g.pred_anomaly.std() > 1e-9 and g.obs_anomaly.std() > 0]
        out[metric] = {
            "equals_climatology": bool(flat),
            "anomaly_correlation": None if flat or not within else float(np.mean(within)),
            "mean_bias_by_origin": {str(int(o)): float(v) for o, v in sub.groupby("origin").error.mean().items()},
            "anomaly_slope": None if flat else _slope(sub),
            "mean_bias": float(sub.error.mean()),
            "mean_bias_vs_reference_mae": float(sub.error.mean()) / ref_mae if ref_mae else None,
            "observed_shift_from_30yr": observed_shift,
            "predicted_shift_from_30yr": float(pa.mean()),
            "shift_captured": (float(pa.mean()) / observed_shift
                               if abs(observed_shift) > 0.1 * ref_mae else None),
            "bias_change_per_lead_year": trend,
            "unit": UNITS[metric],
        }
    return out


def decomposition_section(final: pd.DataFrame) -> dict:
    """Two-way split of the signed error per (origin, metric): common-year, persistent-city, residual."""
    out = {}
    for metric in METRICS:
        sub = final[final.metric == metric]
        total = year_ss = city_ss = grand_ss = 0.0
        for _, part in sub.groupby("origin"):
            e = part.error
            grand = e.mean()
            year = part.groupby("year").error.transform("mean") - grand
            city = part.groupby("city_id").error.transform("mean") - grand
            total += float((e ** 2).sum())
            grand_ss += float(len(e) * grand ** 2)
            year_ss += float((year ** 2).sum())
            city_ss += float((city ** 2).sum())
        resid = max(total - grand_ss - year_ss - city_ss, 0.0)
        out[metric] = {k: (v / total if total else None) for k, v in
                       (("mean_bias", grand_ss), ("common_year", year_ss), ("persistent_city", city_ss),
                        ("residual", resid))}
    return out


def lead_section(rows: pd.DataFrame) -> dict:
    out = {}
    for lead, part in rows.groupby("lead"):
        pts = metric_points(part)
        out[str(int(lead))] = {"headline": headline(pts), "points": pts,
                               "mean_bias": {m: float(part[part.metric == m].error.mean()) for m in METRICS
                                             if (part.metric == m).any()}}
    return out


def _zones():
    from evaluation.scorecard import zone_table
    return zone_table()


def segments_section(final: pd.DataFrame) -> dict:
    out = {}
    reference = _reference(final)
    by_origin = {}
    for origin, part in final.groupby("origin"):
        by_origin[str(int(origin))] = headline(metric_points(part))
    out["origins"] = by_origin
    try:
        zones = _zones()
        zoned = final.assign(zone=final.city_id.map(zones).fillna("unzoned"))
        contrib = contributions(zoned, "zone", reference)
        out["zones"] = {z: {"contribution": float(contrib.get(z, 0.0)),
                            "headline": headline(metric_points(part)),
                            "points": metric_points(part), "n_cities": int(part.city_id.nunique())}
                        for z, part in zoned.groupby("zone")}
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        out["zones"] = {"status": f"unavailable: {type(exc).__name__}: {exc}"}
    try:
        from evaluation.scorecard import station_change_cities
        changed_rows = []
        for origin in final.origin.unique():
            changed = station_change_cities(int(origin))
            if changed is None:
                raise FileNotFoundError("raw coverage_and_stations not present")
            changed_rows.append(final[(final.origin == origin)].city_id.isin(changed))
        flag = pd.concat(changed_rows).reindex(final.index)
        groups = {"changed": final[flag], "stable": final[~flag]}
        out["station_changes"] = {k: {"headline": headline(metric_points(v)), "n_cities": int(v.city_id.nunique()),
                                      "contribution": float(contributions(v.assign(g=k), "g", reference).sum())}
                                  for k, v in groups.items() if len(v)}
    except Exception as exc:  # noqa: BLE001
        out["station_changes"] = {"status": f"unavailable: {type(exc).__name__}: {exc}"}
    city = contributions(final, "city_id", reference).sort_values()
    by_metric = contributions(final, ["city_id", "metric"], reference)
    names = {}
    try:
        names = pd.read_csv(ROOT / "data/city_info.csv", dtype={"city_id": str}).drop_duplicates("city_id") \
            .set_index("city_id").name.to_dict()
    except Exception:  # noqa: BLE001
        pass
    losses = city[city < 0]
    out["worst_cities"] = [{"city_id": c, "name": names.get(c), "contribution": float(v),
                            "worst_metric": str(by_metric.loc[c].idxmin()),
                            "by_metric": {m: float(x) for m, x in by_metric.loc[c].items()}}
                           for c, v in city.head(WORST_CITIES).items() if v < 0]
    out["loss_concentration"] = {
        "cities_losing": int((city < 0).sum()), "cities_gaining": int((city > 0).sum()),
        "total_loss": float(losses.sum()), "total_gain": float(city[city > 0].sum()),
        "worst_10_share_of_loss": float(losses.head(WORST_CITIES).sum() / losses.sum()) if len(losses) else None}
    return out


def versus_section(final: pd.DataFrame, other_dir, label: str) -> dict:
    _, other = load_rows(other_dir)
    other = other[other.split == "final"]
    mine, theirs = metric_points(final), metric_points(other)
    zones = {}
    try:
        table = _zones()
        a = contributions(final.assign(zone=final.city_id.map(table).fillna("unzoned")), "zone")
        b = contributions(other.assign(zone=other.city_id.map(table).fillna("unzoned")), "zone")
        zones = {z: float(a.get(z, 0) - b.get(z, 0)) for z in sorted(set(a.index) | set(b.index))}
    except Exception:  # noqa: BLE001
        pass
    return {"against": Path(other_dir).name, "label": label,
            "metric_headline_delta": {m: (mine[m] - theirs[m]) / len(METRICS) for m in METRICS},
            "zone_headline_delta": zones}


def best_controls(final: pd.DataFrame, runs=RUNS) -> dict:
    """Per metric, the official control with the most points, and the gap to this run in headline points."""
    from evaluation import reward
    mine = metric_points(final)
    best = {}
    for path in reward.candidates(runs):
        meta, rows = load_rows(path)
        if not str(meta.get("model", "")).startswith("control:"):
            continue
        pts = metric_points(rows[rows.split == "final"])
        for m in METRICS:
            if m not in best or pts[m] > best[m]["points"]:
                best[m] = {"control": meta["model"], "run_id": path.name, "points": pts[m]}
    for m, entry in best.items():
        entry["gap_headline_points"] = (entry["points"] - mine[m]) / len(METRICS)
    return best


# ---------------------------------------------------------------- findings
def _finding(kind, metric, finding, evidence, suggestion, headroom, confidence):
    return {"kind": kind, "metric": metric, "finding": finding, "evidence": evidence, "suggestion": suggestion,
            "headroom_points": None if headroom is None else float(headroom), "confidence": confidence}


def findings(card: dict) -> list[dict]:
    out = []
    head, cal, dec, seg = card["headroom"], card["calibration"], card["decomposition"], card["segments"]
    controls = card.get("best_controls") or {}
    for m in METRICS:
        h, c, d, unit = head[m], cal[m], dec[m], UNITS[m]
        if c["equals_climatology"]:
            control = controls.get(m)
            gap = control["gap_headline_points"] if control and control["gap_headline_points"] > 0 else None
            out.append(_finding(
                "no_signal", m, f"{m} is forecast as the 30-year average (no departure from climatology).",
                (f"best control {control['control']} reaches {control['points']:+.1f} metric points here; " if control else "")
                + f"unreachable ceiling {h['ceiling_points']:+.1f} metric points (perfect knowledge of each city's window)",
                "Try a recent-years mean or regional trend for this metric alone, estimated per origin from training data."
                if gap else "Keeping climatology is reasonable: no control does better on this metric.",
                gap, "measured on the same folds" if gap else "descriptive"))
            continue
        bias_fit, bias_honest = h["whatif_remove_mean_bias_fit"] or 0, h["whatif_remove_mean_bias_honest"] or 0
        if bias_honest >= NOTABLE:
            direction = "too high" if c["mean_bias"] > 0 else "too low"
            growth = c["bias_change_per_lead_year"] or 0
            note = ""
            if m in WARMING and growth:
                note = {(True, True): " It grows with lead: the warming trend is under-projected.",
                        (True, False): " It shrinks with lead: the starting level is too low (anchored to a stale "
                                       "average) and the trend partly catches up.",
                        (False, False): " It grows with lead: the warming trend is over-projected.",
                        (False, True): " It shrinks with lead: the starting level is too high."}[(c["mean_bias"] < 0, growth < 0)]
            out.append(_finding(
                "level_bias", m, f"{m} forecasts are {direction} by {abs(c['mean_bias']):.2f} {unit} on average.{note}",
                f"bias {c['mean_bias_vs_reference_mae']:+.0%} of the reference MAE; change per lead year {growth:+.3f} {unit}; "
                f"the same bias holds in other origins (fit {bias_fit:+.2f}, honest {bias_honest:+.2f})",
                "Fix the level or trend: weight recent years more, fit trends over longer or regionally pooled windows, "
                "or adjust the trend by lead. Learn any correction from training years inside each origin.",
                bias_honest, "cross-fitted across origins"))
        by_origin = c["mean_bias_by_origin"]
        if bias_fit >= NOTABLE and bias_honest < 0 and min(by_origin.values()) < 0 < max(by_origin.values()):
            out.append(_finding(
                "bias_flips", m, f"{m}: the level error changes sign between origins, so no fixed correction helps.",
                "mean error by origin " + ", ".join(f"{o} {v:+.2f} {unit}" for o, v in by_origin.items())
                + f"; a constant fix gains {bias_fit:+.2f} in-sample but {bias_honest:+.2f} on other origins",
                "The level or trend method is regime-sensitive: it over-shoots after warm spells and lags after cool "
                "ones. Damp the trend, use more robust level windows, or blend toward the 30-year average when the "
                "recent trend is unstable.",
                None, "descriptive"))
        slope, scale_fit, scale_honest = (c["anomaly_slope"], h["whatif_anomaly_scaling_fit"] or 0,
                                          h["whatif_anomaly_scaling_honest"] or 0)
        if slope is not None and scale_honest >= NOTABLE and (slope < 0.8 or slope > 1.25):
            bold = slope < 0.8
            out.append(_finding(
                "amplitude", m, f"{m} departures from the 30-year average are too {'large' if bold else 'small'} "
                f"(best scaling {slope:.2f}).",
                f"pattern correlation {c['anomaly_correlation']:+.2f}; scaling gains fit {scale_fit:+.2f}, honest {scale_honest:+.2f}",
                "Shrink toward the 30-year average (blend weight, stronger regularisation, fewer features)."
                if bold else "Let forecasts move further from the 30-year average (less shrinkage, stronger trend).",
                scale_honest, "cross-fitted across origins"))
        if m in WARMING and c["shift_captured"] is not None and c["shift_captured"] < 0.6:
            out.append(_finding(
                "shift_captured", m, f"{m}: the model captures only {c['shift_captured']:.0%} of the observed shift "
                "from the 30-year average.",
                f"observed {c['observed_shift_from_30yr']:+.2f} {unit}, predicted {c['predicted_shift_from_30yr']:+.2f} {unit}",
                "The level is anchored to a stale average: weight recent years more, or add a pooled regional trend.",
                None, "descriptive"))
        city_fit, city_honest = h["whatif_remove_city_offsets_fit"] or 0, h["whatif_remove_city_offsets_honest"] or 0
        if (d["persistent_city"] or 0) >= 0.3 and city_honest >= NOTABLE:
            out.append(_finding(
                "city_offsets", m, f"{m}: {d['persistent_city']:.0%} of squared error is a persistent per-city offset.",
                f"a city's other years predict its offset: honest {city_honest:+.2f} (fit {city_fit:+.2f}, mostly noise)",
                "City levels are off: check station changes, short or broken windows and outlier years; use robust "
                "level estimates (median, trimmed mean, break-aware windows).",
                city_honest, "cross-fitted across years (upper bound: uses target years)"))
        if (d["common_year"] or 0) >= 0.4:
            out.append(_finding(
                "common_year", m, f"{m}: {d['common_year']:.0%} of squared error is shared by all cities in a year.",
                "national year-to-year weather that no station history predicts",
                "Do not chase this with city features. Only large-scale drivers (for example the enhance-v2 ocean "
                "indices) could reduce it; expect small gains.",
                None, "descriptive"))
    live = card["lead"]
    finals = [v["headline"] for k, v in live.items() if int(k) >= 4 and v["headline"] is not None]
    lives = [v["headline"] for k, v in live.items() if int(k) <= 3 and v["headline"] is not None]
    if finals and lives and np.mean(lives) - np.mean(finals) > 1.5:
        out.append(_finding(
            "lead_decay", None, f"Skill drops from {np.mean(lives):+.2f} (leads 1-3) to {np.mean(finals):+.2f} "
            "(leads 4-7). The final score uses leads 4-7.",
            "points by lead: " + ", ".join(f"{k}:{v['headline']:+.1f}" for k, v in live.items() if v["headline"] is not None),
            "Target long leads: damp or extrapolate the trend differently by lead, or blend toward the 30-year "
            "average as lead grows.", None, "descriptive"))
    origins = seg["origins"]
    if origins and min(origins.values()) < -1 < max(origins.values()):
        worst = min(origins, key=origins.get)
        out.append(_finding(
            "origin_instability", None, f"Origin {worst} is negative ({origins[worst]:+.2f}) while others gain.",
            ", ".join(f"{o}:{v:+.2f}" for o, v in origins.items()),
            "The method fails in one climate regime. Look at what was different (trend sign, station changes, "
            "drought years) and make the method fall back to the 30-year average when its own fit is weak.",
            -origins[worst] / len(origins), "descriptive"))
    zones = seg.get("zones", {})
    if "status" not in zones:
        for zone, entry in sorted(zones.items(), key=lambda kv: kv[1]["contribution"]):
            if entry["n_cities"] >= MIN_CITIES and entry["contribution"] <= -NOTABLE:
                worst_metric = min(entry["points"], key=entry["points"].get)
                out.append(_finding(
                    "zone", worst_metric, f"Zone {zone} loses {entry['contribution']:+.2f} headline points "
                    f"({entry['n_cities']} cities), mostly on {worst_metric}.",
                    ", ".join(f"{m} {v:+.1f}" for m, v in entry["points"].items()),
                    "Fit or pool this region separately, or fall back to the 30-year average there.",
                    -entry["contribution"], "descriptive"))
    stations = seg.get("station_changes", {})
    if "status" not in stations and {"changed", "stable"} <= set(stations):
        gap = stations["stable"]["headline"] - stations["changed"]["headline"]
        if gap > 2 and stations["changed"]["n_cities"] >= MIN_CITIES:
            out.append(_finding(
                "station_changes", None, f"Cities with a station change near the origin score {gap:.1f} points worse.",
                f"changed {stations['changed']['headline']:+.2f} ({stations['changed']['n_cities']} cities) vs stable "
                f"{stations['stable']['headline']:+.2f}",
                "Detect breaks with coverage_and_stations (training years only) and fit levels after the last change.",
                -stations["changed"]["contribution"] if stations["changed"]["contribution"] < 0 else None, "descriptive"))
    conc = seg["loss_concentration"]
    if conc["worst_10_share_of_loss"] and conc["worst_10_share_of_loss"] >= 0.4 and conc["total_loss"] <= -NOTABLE:
        names = ", ".join(f"{c['name'] or c['city_id']} ({c['worst_metric']})" for c in seg["worst_cities"][:5])
        out.append(_finding(
            "concentrated_losses", None, f"{conc['worst_10_share_of_loss']:.0%} of all city losses come from 10 cities.",
            names, "Inspect these cities' training series for station breaks, gaps and outliers; a per-city fallback "
            "to the 30-year average may help.", -sum(c["contribution"] for c in seg["worst_cities"]), "descriptive"))
    for m, entry in (card.get("best_controls") or {}).items():
        if entry["gap_headline_points"] >= NOTABLE:
            out.append(_finding(
                "control_beats_model", m, f"{entry['control']} beats this run on {m} by "
                f"{entry['gap_headline_points']:+.2f} headline points.",
                f"control {entry['points']:+.1f} vs this run {card['headroom'][m]['points']:+.1f} metric points",
                f"Borrow the {entry['control'].split(':')[-1]} method for {m} (or blend it in) inside your model.",
                entry["gap_headline_points"], "measured on the same folds"))
    order = {"measured on the same folds": 0}
    return sorted(out, key=lambda f: (-(f["headroom_points"] or 0), order.get(f["confidence"], 1)))


# ---------------------------------------------------------------- build, write, explain
def analyse(rows: pd.DataFrame) -> dict:
    """All row-based sections for scored rows with error/anomaly columns (see load_rows)."""
    final = rows[rows.split == "final"]
    pts = metric_points(final)
    return {"reward_points": headline(pts), "metric_points": pts,
            "headroom": headroom_section(final), "calibration": calibration_section(final, rows),
            "decomposition": decomposition_section(final), "lead": lead_section(rows),
            "segments": segments_section(final)}


def build(run_dir, against=None, controls=True, runs=RUNS) -> dict:
    meta, rows = load_rows(run_dir)
    final = rows[rows.split == "final"]
    card = {"insights_version": INSIGHTS_VERSION, "run_id": Path(run_dir).name, "model": meta.get("model"),
            "suite_id": meta["suite_id"], "origins": list(meta.get("origins", meta["config"]["origins"])),
            **analyse(rows)}
    if against is not None and Path(against).resolve() != Path(run_dir).resolve():
        card["versus_champion"] = versus_section(final, against, "champion")
    if controls:
        try:
            card["best_controls"] = best_controls(final, runs)
        except Exception as exc:  # noqa: BLE001
            card["best_controls"] = None
            card["best_controls_status"] = f"unavailable: {type(exc).__name__}: {exc}"
    card["insights"] = findings(card)
    card["caveat"] = ("What-if and ceiling gains are fitted on the rows they score: optimistic upper bounds. "
                      "Build any correction from training data inside each origin and confirm with a new run.")
    return card


def explain(card: dict, top: int = 8) -> str:
    lines = [f"INSIGHTS {card['model']} ({card['run_id']}): reward {card['reward_points']:+.3f} headline points",
             "  headroom by metric (headline points): "
             + "  ".join(f"{m} now {h['headline_share']:+.2f} / ceiling {h['ceiling_points'] / 4:+.2f}"
                         for m, h in card["headroom"].items())]
    for i, f in enumerate(card["insights"][:top], 1):
        room = "" if f["headroom_points"] is None else f"  [up to {f['headroom_points']:+.2f} pts, {f['confidence']}]"
        lines += [f"  {i}. {f['finding']}{room}", f"     evidence: {f['evidence']}", f"     try: {f['suggestion']}"]
    if not card["insights"]:
        lines.append("  No finding above the reporting thresholds.")
    return "\n".join(lines)


def markdown(card: dict) -> str:
    out = [f"# Insights: {card['model']}", "", f"Run `{card['run_id']}` · reward **{card['reward_points']:+.3f}** "
           f"headline points · {card['insights_version']}", "", f"> {card['caveat']}", "", "## Ranked findings", ""]
    for i, f in enumerate(card["insights"], 1):
        room = "" if f["headroom_points"] is None else f" (up to {f['headroom_points']:+.2f} pts, {f['confidence']})"
        out += [f"{i}. **{f['finding']}**{room}", f"   - Evidence: {f['evidence']}", f"   - Try: {f['suggestion']}"]
    out += ["", "## Headroom by metric (headline points)", "",
            "Each what-if is shown as **honest** (learned from other origins, or for city offsets the city's other "
            "years) / fit (learned on the scored rows; optimistic). Only honest gains are evidence a fix would work. "
            "The ceiling assumes each city's future window is known and cannot be reached.", "",
            "| Metric | Now | Ceiling | Remove mean bias | Rescale departures | Remove city offsets |",
            "|---|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "n/a" if v is None else f"{v:+.2f}"  # noqa: E731
    pair = lambda h, k: f"**{fmt(h[f'whatif_{k}_honest'])}** / {fmt(h[f'whatif_{k}_fit'])}"  # noqa: E731
    for m, h in card["headroom"].items():
        out.append(f"| {m} | {h['headline_share']:+.2f} | {h['ceiling_points'] / 4:+.2f} | {pair(h, 'remove_mean_bias')} "
                   f"| {pair(h, 'anomaly_scaling')} | {pair(h, 'remove_city_offsets')} |")
    out += ["", "## Calibration against the 30-year average (final analogue)", "",
            "| Metric | Bias | Anomaly corr. | Best scaling | Shift captured | Bias change / lead yr |", "|---|---:|---:|---:|---:|---:|"]
    for m, c in card["calibration"].items():
        pct = "n/a" if c["shift_captured"] is None else f"{c['shift_captured']:.0%}"
        out.append(f"| {m} | {c['mean_bias']:+.2f} {c['unit']} | {fmt(c['anomaly_correlation'])} | {fmt(c['anomaly_slope'])} "
                   f"| {pct} | {fmt(c['bias_change_per_lead_year'])} |")
    out += ["", "## Where the error comes from (share of squared error)", "",
            "| Metric | Mean bias | Common year | Persistent city | Residual |", "|---|---:|---:|---:|---:|"]
    for m, d in card["decomposition"].items():
        out.append("| " + m + " | " + " | ".join("n/a" if d[k] is None else f"{d[k]:.0%}"
                                                 for k in ("mean_bias", "common_year", "persistent_city", "residual")) + " |")
    out += ["", "## Skill by lead", "", "| Lead | Headline | " + " | ".join(METRICS) + " |", "|---" * (2 + len(METRICS)) + "|"]
    for lead, entry in card["lead"].items():
        out.append(f"| {lead} | {fmt(entry['headline'])} | " + " | ".join(fmt(entry["points"].get(m)) for m in METRICS) + " |")
    seg = card["segments"]
    out += ["", "## Segments", "", "Origins: " + ", ".join(f"{o} {v:+.2f}" for o, v in seg["origins"].items()), ""]
    if "status" not in seg.get("zones", {}):
        out += ["| Zone | Cities | Contribution | Headline |", "|---|---:|---:|---:|"]
        out += [f"| {z} | {e['n_cities']} | {e['contribution']:+.2f} | {fmt(e['headline'])} |" for z, e in seg["zones"].items()]
    out += ["", "Worst cities (headline-point contribution): " + ", ".join(
        f"{c['name'] or c['city_id']} {c['contribution']:+.3f} ({c['worst_metric']})" for c in seg["worst_cities"])]
    if card.get("versus_champion"):
        v = card["versus_champion"]
        out += ["", f"## Versus champion `{v['against']}`", "", ", ".join(
            f"{m} {d:+.2f}" for m, d in v["metric_headline_delta"].items())]
    return "\n".join(out) + "\n"


def write(card: dict, directory: Path = INSIGHTS_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{card['run_id']}.json"
    path.write_text(json.dumps(card, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (directory / f"{card['run_id']}.md").write_text(markdown(card), encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Where a run loses points and what to try next.")
    parser.add_argument("run", help="run id or folder")
    parser.add_argument("--runs", default=str(RUNS))
    parser.add_argument("--against", help="run to compare with (default: the reward champion)")
    parser.add_argument("--no-controls", action="store_true", help="skip the best-control comparison")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    from evaluation import reward
    run = reward._resolve(args.run, args.runs)
    against = reward._resolve(args.against, args.runs) if args.against else reward.current_champion(args.runs)
    card = build(run, against, controls=not args.no_controls, runs=args.runs)
    path = write(card)
    print(json.dumps(card, indent=2) if args.json else explain(card, args.top))
    print(f"  full report: {path.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
