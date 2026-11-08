"""Characteristic scorecard for a completed, verified evaluation run.

Reads only the run's saved artifacts (scores.json, origin-*/row_errors.csv) plus two
analysis-only tables the evaluator owns (data/city_info.csv for zones and the raw
coverage_and_stations subset for station changes). Nothing here feeds a model.

Aggregation never mixes units: within one metric, pool the group's scored rows and take
skill_m = 1 - mean(|e|) / mean(|e_ref|) against the suite's 30-year-average reference
(undefined if the group is empty or its reference MAE is 0). A composite is the equal-weight
mean of the four metric skills, reported only when all four are defined. Every ratio that
can be undefined is reported as null, never as inf/NaN.

The scorecard is written OUTSIDE the sealed run directory (outputs/evaluation/scorecards/),
records the run fingerprint, and fingerprints the analysis tables and this module, so a later
rebuild that differs is detectable.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import METRICS, ROOT
from evaluation.core import read_json, verify_run

SCORECARD_VERSION = 2
SCORECARD_DIR = ROOT / "outputs/evaluation/scorecards"
# Fixed GEOGRAPHIC zones (analysis only; a lat/lon split, not a climate classification):
# West / Central / East split at lon -104 and -87, North / South split at 37 N.
ZONE_LON = (-104.0, -87.0)
ZONE_LAT = 37.0
STATION_WINDOW = (10, 7)          # station change in [O-10, O+7]
MIN_GROUP_CITIES = 10
CITY_FILE = ROOT / "data/city_info.csv"
COVERAGE_FILE = ROOT / "data/raw/coverage_and_stations/train-00000-of-00001.parquet"


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _safe_ratio(numerator, denominator):
    """numerator / denominator, or None when undefined (missing, non-finite or zero denominator)."""
    if numerator is None or denominator is None:
        return None
    numerator, denominator = float(numerator), float(denominator)
    if not (math.isfinite(numerator) and math.isfinite(denominator)) or denominator == 0:
        return None
    return numerator / denominator


def _finite(value):
    """Recursively replace NaN/inf with None so every scorecard serializes as strict JSON."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _sha256(path: Path):
    if not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def analysis_provenance() -> dict:
    """Fingerprints of everything besides the sealed run that can change the scorecard."""
    source = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return {"scorecard_source_sha256": hashlib.sha256(source).hexdigest(),
            "city_info_sha256": _sha256(CITY_FILE),
            "coverage_and_stations_sha256": _sha256(COVERAGE_FILE)}


def zone_table(path: Path = CITY_FILE) -> pd.Series:
    cities = pd.read_csv(path, dtype={"city_id": str}).drop_duplicates("city_id").set_index("city_id")
    ew = np.select([cities.lon < ZONE_LON[0], cities.lon < ZONE_LON[1]], ["West", "Central"], "East")
    ns = np.where(cities.lat >= ZONE_LAT, "North", "South")
    return pd.Series([f"{a}-{b}" for a, b in zip(ew, ns)], index=cities.index, name="zone")


def station_change_cities(origin: int) -> set[str] | None:
    """Cities with a listed station change in [O-10, O+7]; None if the raw subset is absent."""
    if not COVERAGE_FILE.is_file():
        return None
    cov = pd.read_parquet(COVERAGE_FILE, columns=["ID", "Year", "Station Changes"])
    lo, hi = origin - STATION_WINDOW[0], origin + STATION_WINDOW[1]
    hit = cov[(cov["Station Changes"] > 0) & cov.Year.between(lo, hi)]
    return set(hit.ID.astype(str))


def _metric_skill(group: pd.DataFrame):
    if not len(group):
        return None
    ratio = _safe_ratio(group.absolute_error.mean(), group.baseline_absolute_error.mean())
    return None if ratio is None else 1 - ratio


def _ratio_skill(rows: pd.DataFrame) -> tuple[float | None, int]:
    """Equal-weight mean over the four metrics of the per-metric MAE-ratio skill within `rows`."""
    skills = [_metric_skill(rows[rows.metric == metric]) for metric in METRICS]
    if any(s is None for s in skills):
        return None, int(len(rows))
    return float(np.mean(skills)), int(len(rows))


def _rows(run: Path, origins: list[int]) -> pd.DataFrame:
    parts = []
    for origin in origins:
        frame = pd.read_csv(run / f"origin-{origin}/row_errors.csv", dtype={"city_id": str})
        if frame.scored.dtype != bool:
            raise ValueError(f"origin {origin}: row_errors 'scored' must parse as booleans")
        parts.append(frame[frame.scored])
    rows = pd.concat(parts, ignore_index=True)
    rows["signed_error"] = rows.predicted_value - rows.actual_value
    return rows


# ---------------------------------------------------------------------------------------------
# scorecard
# ---------------------------------------------------------------------------------------------
def build_scorecard(run_dir) -> dict:
    run = Path(run_dir).resolve()
    meta = verify_run(run)
    origins = list(meta.get("origins", meta["config"]["origins"]))
    scores = read_json(run / "scores.json")
    rows = _rows(run, origins)
    final = rows[rows.split == "final"]

    def fold(split):
        return [s for s in scores if s["split"] == split]

    card = {"scorecard_version": SCORECARD_VERSION, "run_id": meta["run_id"],
            "run_fingerprint": meta.get("run_fingerprint"), "model": meta["model"],
            "suite_id": meta["suite_id"], "phase": meta.get("phase"), "origins": origins,
            "params": meta.get("params", {}), "model_spec": meta.get("model_spec"),
            "provenance": analysis_provenance()}

    # Judge fidelity: the quantities the harness ranks and reports.
    summary = meta["summary"]
    card["fidelity"] = {
        "headline_final_raw_skill": summary["final"]["mean_skill_unclipped"],
        "live_raw_skill": summary["live"]["mean_skill_unclipped"],
        "official_score_final_foldmean": summary["final"]["mean_score"],
        "official_score_live_foldmean": summary["live"]["mean_score"],
        "selection_score_final": 100 * max(0.0, summary["final"]["mean_skill_unclipped"]),
        "n_folds": len(fold("final")),
    }
    # Per metric: equal-fold mean of each fold's metric skill (consistent with the headline).
    card["per_metric"] = {split: {m: float(np.mean([s["skill"][m] for s in fold(split)])) for m in METRICS}
                          for split in ("final", "live")}
    card["per_metric_counts"] = {
        split: {m: {"n_rows": int(((rows.split == split) & (rows.metric == m)).sum()), "n_folds": len(fold(split))}
                for m in METRICS} for split in ("final", "live")}
    # Horizon: per fold composite by lead, then equal-weight over folds.
    horizon, horizon_counts = {}, {}
    for lead in range(1, meta["config"]["horizon"] + 1):
        found = [_ratio_skill(rows[(rows.origin == o) & (rows.lead == lead)]) for o in origins]
        defined = [v for v, _ in found if v is not None]
        horizon[str(lead)] = float(np.mean(defined)) if defined else None
        horizon_counts[str(lead)] = {"n_rows": int(sum(n for _, n in found)), "n_folds": len(defined)}
    card["horizon"] = horizon
    card["horizon_counts"] = horizon_counts
    # Stability across origins.
    by_origin = {str(s["origin"]): s["mean_skill_unclipped"] for s in fold("final")}
    worst = min(((s["skill"][m], s["origin"], m) for s in fold("final") for m in METRICS))
    values = list(by_origin.values())
    card["stability"] = {"final_raw_skill_by_origin": by_origin, "min": min(values), "max": max(values),
                         "range": max(values) - min(values),
                         "worst_metric_origin": {"skill": worst[0], "origin": worst[1], "metric": worst[2]}}
    card["bias"] = bias_section(final)
    # Tier B diagnostics: a failure is recorded in the section, never blocks the scorecard.
    for name, build in (("spatial", lambda: spatial_section(final, zone_table())),
                        ("robustness", lambda: robustness_section(final)),
                        ("inhomogeneity", lambda: inhomogeneity_section(final, origins))):
        try:
            card[name] = build()
        except Exception as exc:          # noqa: BLE001 - diagnostic only
            card[name] = {"status": f"error: {type(exc).__name__}: {exc}"}
    card["operational"] = {"elapsed_seconds": meta.get("elapsed_seconds")}
    return _finite(card)


def bias_section(final: pd.DataFrame) -> dict:
    """Mean signed error per metric over final-split scored rows pooled across origins."""
    out = {}
    for metric in METRICS:
        group = final[final.metric == metric]
        mean_error = float(group.signed_error.mean()) if len(group) else None
        out[metric] = {"mean_signed_error": mean_error,
                       "relative_to_reference_mae": _safe_ratio(mean_error, group.baseline_absolute_error.mean()
                                                                if len(group) else None),
                       "n": int(len(group))}
    return out


def spatial_section(final: pd.DataFrame, zones: pd.Series) -> dict:
    """Composite skill per fixed GEOGRAPHIC zone (lat/lon split; not a climate classification)."""
    zoned = final.assign(zone=final.city_id.map(zones).fillna("unzoned"))
    out = {}
    for zone, group in zoned.groupby("zone"):
        skill, n = _ratio_skill(group)
        out[zone] = {"final_skill": skill, "n_rows": n, "n_cities": int(group.city_id.nunique())}
    return out


def robustness_section(final: pd.DataFrame) -> dict:
    """Vs the 30-year average: strict city-metric win rate, per-metric p90 ratios, worst-decile city."""
    city = final.groupby(["city_id", "metric"])[["absolute_error", "baseline_absolute_error"]].mean()
    p90 = {}
    for metric in METRICS:
        group = final[final.metric == metric]
        p90[metric] = (_safe_ratio(group.absolute_error.quantile(.9), group.baseline_absolute_error.quantile(.9))
                       if len(group) else None)
    city_skill, excluded = [], 0
    for _, group in final.groupby("city_id"):
        skill, _ = _ratio_skill(group)
        if skill is None:
            excluded += 1
        else:
            city_skill.append(skill)
    return {"city_metric_win_rate": float((city.absolute_error < city.baseline_absolute_error).mean()) if len(city) else None,
            "n_city_metrics": int(len(city)),
            "p90_abs_error_ratio_by_metric": p90,
            "p90_abs_error_ratio": (float(np.mean(list(p90.values())))
                                    if all(v is not None for v in p90.values()) else None),
            "worst_decile_city_skill": float(np.quantile(city_skill, .1)) if city_skill else None,
            "n_cities": len(city_skill), "n_cities_excluded": excluded}


def inhomogeneity_section(final: pd.DataFrame, origins: list[int]) -> dict:
    """Cities with a listed station change in [O-10, O+7], per origin."""
    out = {}
    for origin in origins:
        changed = station_change_cities(origin)
        if changed is None:
            return {"status": "raw coverage_and_stations not present; skipped"}
        sub = final[final.origin == origin]
        flag = sub.city_id.isin(changed)
        entry = {"n_changed_cities": int(sub[flag].city_id.nunique()),
                 "n_stable_cities": int(sub[~flag].city_id.nunique()),
                 "n_changed_rows": int(flag.sum()), "n_stable_rows": int((~flag).sum())}
        if entry["n_changed_cities"] >= MIN_GROUP_CITIES:
            entry["changed_skill"] = _ratio_skill(sub[flag])[0]
            entry["stable_skill"] = _ratio_skill(sub[~flag])[0]
        out[str(origin)] = entry
    return out


def flat_metrics(card: dict) -> dict[str, float]:
    """Intrinsic numbers for MLflow (no deltas against other runs). Undefined values are omitted."""
    out = dict(card["fidelity"])
    for split, metrics in card["per_metric"].items():
        out.update({f"skill.{split}.{m}": v for m, v in metrics.items()})
    out.update({f"skill.lead.{k}": v for k, v in card["horizon"].items()})
    out.update({f"skill.origin.{o}": v for o, v in card["stability"]["final_raw_skill_by_origin"].items()})
    out["stability.range"] = card["stability"]["range"]
    out.update({f"bias.{m}": v["relative_to_reference_mae"] for m, v in card["bias"].items()})
    if "status" not in card["spatial"]:
        out.update({f"skill.zone.{z}": v["final_skill"] for z, v in card["spatial"].items()})
    rob = card["robustness"]
    if "status" not in rob:
        out.update({"win_rate": rob["city_metric_win_rate"], "p90_error_ratio": rob["p90_abs_error_ratio"],
                    "worst_decile_city_skill": rob["worst_decile_city_skill"]})
        out.update({f"p90_error_ratio.{m}": v for m, v in rob["p90_abs_error_ratio_by_metric"].items()})
    out["runtime_s"] = card["operational"]["elapsed_seconds"]
    return {k: float(v) for k, v in out.items()
            if v is not None and not isinstance(v, bool) and math.isfinite(float(v))}


def write_scorecard(run_dir, directory: Path = SCORECARD_DIR) -> tuple[Path, dict]:
    card = build_scorecard(run_dir)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{card['run_id']}.json"
    path.write_text(json.dumps(card, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pd.Series(flat_metrics(card)).rename("value").to_csv(directory / f"{card['run_id']}.csv", index_label="metric")
    return path, card
