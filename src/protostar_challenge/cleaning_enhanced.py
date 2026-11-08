"""Checks and cleaning for the enhanced inputs: the fixed city facts and the ocean indices.

Run through `scripts/04_clean_enhanced.py`. Page: documentation/10-enhanced.md.

The city facts (`static/city_static.csv`, identical in enhance-v1 and enhance-v2) are cleaned into a model-ready table:
    constant columns     dropped (no information for a model)
    elevation resolution 15 cities report it in degrees, not metres: converted
    catalog elevation    NOAA catalog and USGS ground elevation disagree by > 30 m at 15 cities: flagged, USGS kept
    location             3 challenge coordinates sit > 3 km from the catalog station: flagged, facts not moved
Original values are kept under their own names; every change and flag is one row in `static/qc/static_issues.csv`.

The ocean indices (enhance-v2, `ocean/origin-YYYY/`) are not changed. `verify_ocean` recomputes them and fails on any
mismatch: no row after the origin, no gap or missing value, the same absolute SST in every snapshot, the box means
from the stored ERSSTv5 grid, the anomalies from each origin's 30-year climatology, and the annual means.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ELEVATION_DISAGREE_M = 30.0   # about 0.2 C (0.35 F) at the standard lapse rate of 6.5 C/km
COORDINATE_OFFSET_KM = 3.0    # beyond the ~1.9 km error of coordinates given to the arc-minute
DEGREES_LIMIT = 1e-3          # a resolution below this is a cell size in degrees, not metres
METRES_PER_DEGREE = 111_320.0
CLIMATOLOGY_YEARS = 30
INDICES = ("nino34", "north_pacific", "north_atlantic")
TOLERANCE_C = 1e-4            # recomputation tolerance; the stored grid is float32 at the source
LANDCOVER = "landcover_"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- city facts
def clean_static(static: pd.DataFrame, cities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Model-ready city facts, a log of every change and flag, and a summary. `cities` is data/city_info.csv."""
    if static["city_id"].duplicated().any():
        raise ValueError("city_static has duplicate city_id")
    if set(static["city_id"]) != set(cities["city_id"]):
        raise ValueError("city_static does not cover exactly the challenge cities")
    joined = static.merge(cities, on="city_id", suffixes=("", "_challenge"), validate="one_to_one")
    if not (np.allclose(joined["lat"], joined["lat_challenge"]) and np.allclose(joined["lon"], joined["lon_challenge"])):
        raise ValueError("city_static coordinates differ from data/city_info.csv")
    fractions = [c for c in static.columns if c.startswith(LANDCOVER) and c.endswith("_fraction_1km") and "valid" not in c]
    total = static[fractions].sum(axis=1)
    if not np.allclose(total, 1.0, atol=1e-6):
        raise ValueError("land-cover class fractions do not sum to 1")
    if static.drop(columns=["elevation_acquisition_date_raw"]).isna().any().any():
        raise ValueError("city_static has missing values")

    clean = static.copy()
    log = []

    def note(rows: pd.DataFrame, issue: str, action: str, column: str, before, after=None) -> None:
        for i in rows.index:
            log.append({"city_id": clean.at[i, "city_id"], "name": clean.at[i, "name"], "issue": issue, "column": column,
                        "value": before[i] if hasattr(before, "__getitem__") else before,
                        "new_value": after[i] if after is not None else np.nan, "action": action})

    # Elevation resolution reported as a cell size in degrees (1/9 or 1/3 arc-second) instead of metres.
    res = clean["elevation_resolution_m"]
    degrees = res < DEGREES_LIMIT
    metres = (res * METRES_PER_DEGREE).round(2)
    note(clean[degrees], "resolution in degrees", "converted to metres", "elevation_resolution_m", res, metres)
    clean.loc[degrees, "elevation_resolution_m"] = metres[degrees]
    clean["elevation_resolution_m"] = clean["elevation_resolution_m"].round(2)

    # Ground elevation (USGS, at the challenge coordinate) against the NOAA station catalog.
    clean["elevation_catalog_diff_m"] = clean["elevation_m"] - clean["station_catalog_elevation_m"]
    clean["elevation_catalog_disagrees"] = clean["elevation_catalog_diff_m"].abs() > ELEVATION_DISAGREE_M
    note(clean[clean["elevation_catalog_disagrees"]], f"catalog elevation differs by > {ELEVATION_DISAGREE_M:g} m",
         "flagged; elevation_m (USGS) kept", "station_catalog_elevation_m", clean["station_catalog_elevation_m"],
         clean["elevation_m"])

    # Challenge coordinate far from the catalog station: the facts describe a point that may not be the station.
    clean["location_uncertain"] = clean["elevation_coordinate_offset_km"] > COORDINATE_OFFSET_KM
    note(clean[clean["location_uncertain"]], f"coordinate > {COORDINATE_OFFSET_KM:g} km from catalog station",
         "flagged; elevation, coast distance and land cover may describe the wrong point", "elevation_coordinate_offset_km",
         clean["elevation_coordinate_offset_km"])

    # Columns with one value for every city carry nothing a model can use.
    constant = [c for c in static.columns if static[c].nunique(dropna=False) == 1]
    for c in constant:
        log.append({"city_id": "*", "name": "*", "issue": "constant column", "column": c, "value": static[c].iloc[0],
                    "new_value": np.nan, "action": "dropped"})
    clean = clean.drop(columns=constant)
    clean = clean.drop(columns=["elevation_acquisition_date_raw"])  # parsed into elevation_acquisition_date

    kept_fractions = [c for c in fractions if c not in constant]
    model_columns = ["lat", "lon", "elevation_m", "distance_to_ocean_coast_km", "landcover_station_class",
                     "landcover_dominant_class_1km", *kept_fractions]
    summary = {"rows": len(clean), "constant_columns_dropped": constant,
               "resolution_converted": int(degrees.sum()),
               "elevation_catalog_disagrees": int(clean["elevation_catalog_disagrees"].sum()),
               "location_uncertain": clean.loc[clean["location_uncertain"], "name"].tolist(),
               "elevation_catalog_diff_m": {"median": float(clean["elevation_catalog_diff_m"].median()),
                                            "min": float(clean["elevation_catalog_diff_m"].min()),
                                            "max": float(clean["elevation_catalog_diff_m"].max())},
               "model_columns": model_columns,
               "categorical_columns": ["landcover_station_class", "landcover_dominant_class_1km"],
               "compositional_columns": kept_fractions}
    return clean, pd.DataFrame(log), summary


# ---------------------------------------------------------------- ocean indices
def box_means(npz: Path, box: dict) -> np.ndarray:
    """cos(latitude)-weighted monthly mean SST over the cells inside `box`, ignoring land (NaN) cells."""
    with np.load(npz) as z:
        sst, lat, lon = z["sst"].astype(float), z["lat"].astype(float), z["lon"].astype(float)
    inside_lat = (lat >= box["south"]) & (lat <= box["north"])
    inside_lon = (lon >= box["west"]) & (lon <= box["east"])
    sst = sst[:, inside_lat][:, :, inside_lon]
    weights = np.cos(np.deg2rad(lat[inside_lat]))[None, :, None] * np.ones_like(sst)
    weights[np.isnan(sst)] = 0.0
    return np.nansum(np.nan_to_num(sst) * weights, axis=(1, 2)) / weights.sum(axis=(1, 2))


def verify_origin(monthly: pd.DataFrame, annual: pd.DataFrame, origin: int) -> list[str]:
    """Hard checks on one origin's snapshot. Returns the problems found (empty when it passes)."""
    problems = []
    first = int(monthly["year"].min())
    if monthly.isna().any().any() or annual.isna().any().any():
        problems.append("missing values")
    if monthly.duplicated(["year", "month"]).any():
        problems.append("duplicate months")
    if int(monthly["year"].max()) > origin or int(annual["year"].max()) > origin:
        problems.append("a row after the origin")
    if len(monthly) != (origin - first + 1) * 12:
        problems.append("missing months")
    if pd.to_datetime(monthly["date"]).dt.year.max() > origin:
        problems.append("a date after the origin")
    start = origin - CLIMATOLOGY_YEARS + 1
    if not ((monthly["climatology_start_year"] == start) & (monthly["climatology_end_year"] == origin)).all():
        problems.append("climatology window is not the 30 years ending at the origin")
    window = monthly[(monthly["year"] >= start) & (monthly["year"] <= origin)]
    for index in INDICES:
        normal = window.groupby("month")[f"{index}_sst_c"].mean()
        anomaly = monthly[f"{index}_sst_c"] - monthly["month"].map(normal)
        if (anomaly - monthly[f"{index}_anomaly_c"]).abs().max() > TOLERANCE_C:
            problems.append(f"{index} anomaly does not match the climatology")
        for kind in ("sst_c", "anomaly_c"):
            mean = monthly.groupby("year")[f"{index}_{kind}"].mean()
            stored = annual.set_index("year")[f"{index}_{kind}"]
            if not mean.index.equals(stored.index) or (mean - stored).abs().max() > TOLERANCE_C:
                problems.append(f"{index} annual {kind} is not the mean of its 12 months")
    if (annual["months_available"] != 12).any():
        problems.append("a year without 12 months")
    return problems


def robust_extremes(series: pd.Series, dates: pd.Series, limit: float = 4.0) -> list:
    """Months with a robust |z| above `limit` (median and scaled MAD). Reported, never removed."""
    centre = series.median()
    spread = 1.4826 * (series - centre).abs().median()
    z = (series - centre) / spread
    return [[d, round(float(v), 3), round(float(s), 2)] for d, v, s in zip(dates[z.abs() > limit], series[z.abs() > limit],
                                                                          z[z.abs() > limit])]


def verify_ocean(release: Path, origins=range(1979, 2019)) -> dict:
    """Recompute every ocean snapshot of a release; raise ValueError on any hard failure."""
    release = Path(release)
    schema = json.loads((release / "schema.json").read_text(encoding="utf-8"))
    origins = list(origins)
    latest = max(origins)
    reference = pd.read_csv(release / f"ocean/origin-{latest}/monthly.csv")
    failures, report = {}, {"origins": [min(origins), latest], "n_origins": len(origins)}
    for origin in origins:
        monthly = pd.read_csv(release / f"ocean/origin-{origin}/monthly.csv")
        annual = pd.read_csv(release / f"ocean/origin-{origin}/annual.csv")
        problems = verify_origin(monthly, annual, origin)
        same = reference[reference["year"] <= origin].reset_index(drop=True)
        for index in INDICES:
            if len(same) != len(monthly) or (same[f"{index}_sst_c"] - monthly[f"{index}_sst_c"]).abs().max() > 0:
                problems.append(f"{index} absolute SST differs from the {latest} snapshot")
        if problems:
            failures[origin] = problems
    grid = {}
    for index in INDICES:
        npz = release / f"sources/ocean/{index}.npz"
        if npz.is_file():
            recomputed = box_means(npz, schema["boxes_degrees_east"][index])
            error = float(np.abs(recomputed[:len(reference)] - reference[f"{index}_sst_c"].to_numpy()).max())
            grid[index] = error
            if len(recomputed) != len(reference) or error > TOLERANCE_C:
                failures.setdefault("grid", []).append(f"{index} box mean differs from the grid by {error:.2g} C")
    if failures:
        raise ValueError(f"ocean indices failed verification: {failures}")
    annual = pd.read_csv(release / f"ocean/origin-{latest}/annual.csv")
    report.update({
        "passed": True,
        "checks": ["no row or date after the origin", "no gap, duplicate or missing value",
                   "absolute SST identical in every snapshot", "box means recomputed from the stored ERSSTv5 grid",
                   "anomaly = SST minus the 30-year monthly climatology ending at the origin",
                   "annual values = mean of the 12 monthly values"],
        "grid_max_error_c": grid,
        "monthly_rows_latest": len(reference),
        "extreme_anomalies_latest": {i: robust_extremes(reference[f"{i}_anomaly_c"], reference["date"]) for i in INDICES},
        "largest_monthly_anomaly_change_c": {i: round(float(reference[f"{i}_anomaly_c"].diff().abs().max()), 3)
                                             for i in INDICES},
        "warming_trend_c_per_decade": {i: round(float(np.polyfit(annual["year"], annual[f"{i}_sst_c"], 1)[0] * 10), 3)
                                       for i in INDICES},
        "anomaly_correlation_latest": annual[[f"{i}_anomaly_c" for i in INDICES]].corr().round(2).to_dict()})
    return report


# ---------------------------------------------------------------- build
def run(static_release: Path, ocean_release: Path, cities_csv: Path, out_dir: Path,
        weather_release: Path | None = None) -> dict:
    """Write static/ (cleaned city facts), checks/ (ocean verification) and, if given, copy the weather release."""
    static_release, ocean_release, out_dir = Path(static_release), Path(ocean_release), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; write a new folder")
    static_path = static_release / "static/city_static.csv"
    other = ocean_release / "static/city_static.csv"
    if other.is_file() and sha256(other) != sha256(static_path):
        raise ValueError("city_static differs between the two enhanced releases")
    static = pd.read_csv(static_path, dtype={"elevation_acquisition_date_raw": str})
    clean, log, summary = clean_static(static, pd.read_csv(cities_csv))
    (out_dir / "static/qc").mkdir(parents=True)
    clean.to_csv(out_dir / "static/city_static_clean.csv", index=False)
    log.to_csv(out_dir / "static/qc/static_issues.csv", index=False)
    ocean = verify_ocean(ocean_release)
    (out_dir / "checks").mkdir()
    (out_dir / "checks/enhance-ocean-verification.json").write_text(json.dumps(ocean, indent=1, default=str) + "\n",
                                                                    encoding="utf-8")
    copied = 0
    if weather_release is not None:
        for path in sorted(Path(weather_release).rglob("*")):
            name = path.relative_to(weather_release).as_posix()
            if path.is_file() and name != "release.json":
                target = out_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
                copied += 1
    manifest = {"inputs": {"static": {"path": static_path.as_posix(), "sha256": sha256(static_path)},
                           "ocean_release": json.loads((ocean_release / "release.json").read_text())["id"],
                           "weather_release": None if weather_release is None
                           else json.loads((Path(weather_release) / "release.json").read_text())["id"],
                           "weather_files_copied": copied},
                "code": {"src/protostar_challenge/cleaning_enhanced.py": sha256(Path(__file__))},
                "constants": {"ELEVATION_DISAGREE_M": ELEVATION_DISAGREE_M, "COORDINATE_OFFSET_KM": COORDINATE_OFFSET_KM,
                              "DEGREES_LIMIT": DEGREES_LIMIT, "METRES_PER_DEGREE": METRES_PER_DEGREE},
                "static": summary, "ocean_passed": ocean["passed"]}
    (out_dir / "static/manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n", encoding="utf-8")
    return manifest
