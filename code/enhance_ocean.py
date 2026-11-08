"""Small NOAA ERSSTv5 indices with server-side cutoff and per-origin snapshots.

  uv run python code/enhance_ocean.py --acquire-only
  uv run python scripts/line.py pull enhance-v1
  uv run python code/enhance_ocean.py

Only verified 1950-01 through 2018-12 SST arrays are persisted. This release is
not certified gate-eligible: a modern reconstructed observational product does
not prove its reconstruction was learned only from data before each early
backtest origin. Origin files prevent row-date and climatology leakage but do
not settle that separate source-vintage question. No target labels are read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import requests
from scipy.io import netcdf_file

from common import DataRuleViolation, assert_no_post_cutoff, effective_cutoff_year
from enhance_static import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
URL = "https://psl.noaa.gov/thredds/ncss/grid/Datasets/noaa.ersst.v5/sst.mnmean.nc"
START = "1950-01-01T00:00:00Z"
# NCSS selects the nearest endpoint. Dec 31 can accidentally include Jan 2019.
# The exact observed monthly coordinate is Dec 1, not the last day of the month.
END = "2018-12-01T00:00:00Z"
BOXES = {
    "nino34": {"south": -5, "north": 5, "west": 190, "east": 240},
    "north_pacific": {"south": 20, "north": 60, "west": 120, "east": 260},
    "north_atlantic": {"south": 0, "north": 60, "west": 280, "east": 360},
}
ORIGINS = tuple(range(1979, 2019))
EXPECTED_MONTHS = pd.date_range("1950-01-01", "2018-12-01", freq="MS")


def parameters(box):
    return {"var": "sst", **box, "horizStride": 1, "time_start": START,
            "time_end": END, "timeStride": 1, "accept": "netcdf"}


def validate_months(dates):
    dates = pd.DatetimeIndex(dates)
    assert_no_post_cutoff(pd.DataFrame({"year": dates.year}), "year", "ERSSTv5 monthly dates")
    if not dates.equals(EXPECTED_MONTHS):
        raise DataRuleViolation("ERSST response must contain exactly January 1950 through December 2018, once per month")
    return dates


def decode_response(content):
    with netcdf_file(BytesIO(content), mmap=False) as dataset:
        variable = dataset.variables["sst"]
        if variable.dimensions != ("time", "lat", "lon"):
            raise ValueError("Unexpected ERSST dimensions")
        if getattr(variable, "dataset", b"").decode() != "NOAA Extended Reconstructed SST V5":
            raise ValueError("Unexpected product version; do not silently adopt a later reconstruction model")
        if variable.units != b"degC":
            raise ValueError("Unexpected temperature units")
        time_variable = dataset.variables["time"]
        if time_variable.units != b"days since 1800-1-1 00:00:00":
            raise ValueError("Unexpected time encoding")
        dates = validate_months([datetime(1800, 1, 1) + timedelta(days=float(x)) for x in time_variable[:]])
        lat, lon = dataset.variables["lat"][:].copy(), dataset.variables["lon"][:].copy()
        values = variable[:].astype(float).copy()
        missing = getattr(variable, "missing_value", None)
        if missing is not None:
            values[values == missing] = np.nan
        values[~np.isfinite(values)] = np.nan
        if np.any(np.isfinite(values) & ((values < -3) | (values > 45))):
            raise ValueError("Out-of-range SST or unrecognized missing-value encoding")
    return dates, lat, lon, values


def acquire(name: str, cache: Path, offline=False):
    path, meta = cache / f"{name}.npz", cache / f"{name}.json"
    expected = parameters(BOXES[name])
    if path.exists():
        entry = json.loads(meta.read_text(encoding="utf-8"))
        if entry["parameters"] != expected or entry["snapshot_sha256"] != sha(path):
            raise ValueError("Cached ocean source changed")
        with np.load(path, allow_pickle=False) as data:
            dates = validate_months(data["date"].astype(str))
            return dates, data["lat"], data["lon"], data["sst"], entry
    if offline:
        raise FileNotFoundError(f"Missing ocean snapshot: {name}")
    # Fail before making a request if code/config accidentally extends the cap.
    if pd.Timestamp(expected["time_end"]).year > 2018 or expected["time_end"] != END:
        raise DataRuleViolation("Weather request exceeds the fixed monthly cutoff")
    response = requests.get(URL, params=expected, timeout=(20, 240))
    response.raise_for_status()
    dates, lat, lon, values = decode_response(response.content)
    cache.mkdir(parents=True, exist_ok=True)
    # Do not persist provider-wide statistics from NetCDF global attributes.
    # Snapshot only the actual verified subset and the coordinates it uses.
    np.savez_compressed(path, date=dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"), lat=lat, lon=lon, sst=values)
    entry = {
        "url": response.url, "parameters": expected,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "response_sha256": hashlib.sha256(response.content).hexdigest(),
        "snapshot_sha256": sha(path), "observation_start": "1950-01-01", "observation_end": "2018-12-01",
        "product": "NOAA ERSSTv5 monthly absolute SST, degrees Celsius", "license": "U.S. government public data",
        "note": "Only validated subset arrays retained; global attributes do not enter indices",
    }
    write_json(meta, entry)
    return dates, lat, lon, values, entry


def regional_mean(lat, lon, values, box):
    latitude = (lat >= box["south"]) & (lat <= box["north"])
    normalized_lon = np.mod(lon, 360)
    longitude = (normalized_lon >= box["west"]) & (normalized_lon <= box["east"])
    # An east boundary of 360 includes the zero meridian once, if supplied as 0.
    if box["east"] == 360:
        longitude |= normalized_lon == 0
    selected = values[:, latitude][:, :, longitude]
    weights = np.cos(np.radians(lat[latitude]))[None, :, None]
    valid = np.isfinite(selected)
    counts = valid.sum(axis=(1, 2))
    if not latitude.any() or not longitude.any() or (counts == 0).any():
        raise ValueError("No ocean coverage in a requested region")
    denominator = (valid * weights).sum(axis=(1, 2))
    result = np.nansum(selected * weights, axis=(1, 2)) / denominator
    return result, {"latitude_centers": lat[latitude].tolist(), "longitude_centers": lon[longitude].tolist(),
                    "minimum_valid_cells": int(counts.min()), "maximum_valid_cells": int(counts.max()),
                    "land_mask": "Missing SST excluded independently in each month", "weight": "cos(latitude)"}


def origin_snapshot(monthly: pd.DataFrame, origin: int):
    if origin not in ORIGINS:
        raise DataRuleViolation("Supported origins are 1979 through 2018")
    history = monthly.loc[monthly.year <= origin].copy()
    assert_no_post_cutoff(history, "year", f"ocean origin {origin}")
    if history.empty or history.year.max() != origin:
        raise DataRuleViolation("Incomplete ocean history")
    if history.year.max() > origin:
        raise DataRuleViolation("Post-origin ocean row")
    reference_start = origin - 29
    reference = history.loc[history.year >= reference_start]
    if reference.groupby("month").size().ne(30).any():
        raise ValueError("Each calendar month must have 30 reference years")
    for name in BOXES:
        column = f"{name}_sst_c"
        climatology = reference.groupby("month")[column].mean()
        history[f"{name}_anomaly_c"] = history[column] - history.month.map(climatology)
    history["climatology_start_year"], history["climatology_end_year"] = reference_start, origin
    value_columns = [f"{name}_{suffix}" for name in BOXES for suffix in ("sst_c", "anomaly_c")]
    annual = history.groupby("year", as_index=False)[value_columns].mean()
    annual["climatology_start_year"], annual["climatology_end_year"] = reference_start, origin
    annual["months_available"] = history.groupby("year").size().to_numpy()
    if annual.months_available.ne(12).any():
        raise ValueError("Incomplete ocean annual average")
    return history, annual


def load_ocean_for_origin(release: Path, origin: int, frequency="monthly"):
    """Downstream loader: never open a later origin's table and truncate it."""
    if origin not in ORIGINS or origin > effective_cutoff_year():
        raise DataRuleViolation("Requested ocean origin exceeds the active fold or challenge cutoff")
    if frequency not in ("monthly", "annual"):
        raise ValueError("frequency must be monthly or annual")
    folder = Path(release)
    manifest = json.loads((folder / "release.json").read_text(encoding="utf-8"))
    name = f"ocean/origin-{origin}/{frequency}.csv"
    expected = manifest["files"].get(name)
    if expected is None or sha(folder / name) != expected["sha256"]:
        raise ValueError("Ocean release file is absent or modified")
    result = pd.read_csv(folder / name)
    assert_no_post_cutoff(result, "year", name)
    if result.year.max() > origin or (result.climatology_end_year != origin).any():
        raise DataRuleViolation("Ocean table or normalization exceeds its declared origin")
    return result


def validate(folder: Path):
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    total_rows = 0
    for origin in ORIGINS:
        for frequency, multiplier in (("monthly", 12), ("annual", 1)):
            path = folder / f"ocean/origin-{origin}/{frequency}.csv"
            frame = pd.read_csv(path)
            assert_no_post_cutoff(frame, "year", str(path))
            if len(frame) != (origin-1949)*multiplier or frame.year.min() != 1950 or frame.year.max() != origin:
                raise DataRuleViolation(f"Incomplete or post-origin data in {path}")
            if frame.isna().any().any() or not np.isfinite(frame.select_dtypes(include="number")).all().all():
                raise ValueError("Missing/nonfinite ocean index")
            if frame.climatology_end_year.ne(origin).any() or frame.climatology_start_year.ne(origin-29).any():
                raise DataRuleViolation("Wrong origin normalization")
            for name in BOXES:
                if not frame[f"{name}_sst_c"].between(-3, 45).all():
                    raise ValueError("Implausible regional SST")
            if frequency == "monthly":
                dates = pd.to_datetime(frame.date)
                expected = pd.date_range("1950-01-01", f"{origin}-12-01", freq="MS")
                if not pd.DatetimeIndex(dates).equals(expected):
                    raise ValueError("Duplicate or missing month")
                if not np.array_equal(dates.dt.year, frame.year) or not np.array_equal(dates.dt.month, frame.month):
                    raise ValueError("Date columns disagree")
                reference = frame.loc[frame.year >= origin-29]
                anomalies = reference.groupby("month")[[f"{name}_anomaly_c" for name in BOXES]].mean()
                if not np.allclose(anomalies, 0, atol=1e-8):
                    raise ValueError("Calendar-month anomalies are not centered on their training reference")
            total_rows += len(frame)
    for name, digest in provenance["files"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"Source or output hash mismatch: {name}")
    static = pd.read_csv(folder / "static/city_static.csv")
    if len(static) != 210 or static.city_id.nunique() != 210:
        raise ValueError("Cumulative release lost city facts")
    result = {"passed": True, "indices": list(BOXES), "origin_count": len(ORIGINS),
              "origin_range": [min(ORIGINS), max(ORIGINS)], "city_count": 210, "total_index_rows": total_rows,
              "observation_range": ["1950-01-01", "2018-12-01"], "post_2018_rows": 0,
              "gate_eligible": False, "reason": "Pre-origin reconstruction/calibration vintage not independently verified",
              "scope": "Data validation only; no model scoring or assessment labels accessed"}
    print(json.dumps(result, indent=2))
    return result


def build(output: Path, cache: Path, static_release: Path, offline=False):
    # Verify the upstream release in the station's own pulled release folder.
    upstream = json.loads((static_release / "release.json").read_text(encoding="utf-8"))
    if upstream["id"] != "enhance-v1":
        raise ValueError("Expected the immutable enhance-v1 upstream release")
    for name, entry in upstream["files"].items():
        if sha(static_release / name) != entry["sha256"]:
            raise ValueError(f"Upstream release changed: {name}")
    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "sources/ocean"
    source_dir.mkdir(parents=True, exist_ok=True)
    table, entries, regional_details = None, {}, {}
    for name, box in BOXES.items():
        dates, lat, lon, values, entry = acquire(name, cache, offline)
        index, details = regional_mean(lat, lon, values, box)
        if table is None:
            table = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "year": dates.year, "month": dates.month})
        table[f"{name}_sst_c"] = index
        entries[name], regional_details[name] = entry, details
        for extension in ("npz", "json"):
            shutil.copyfile(cache / f"{name}.{extension}", source_dir / f"{name}.{extension}")
        print(f"Validated ocean region: {name}, {len(dates)} months", flush=True)
    for origin in ORIGINS:
        monthly, annual = origin_snapshot(table, origin)
        folder = output / f"ocean/origin-{origin}"
        folder.mkdir(parents=True, exist_ok=True)
        monthly.to_csv(folder / "monthly.csv", index=False, float_format="%.10f")
        annual.to_csv(folder / "annual.csv", index=False, float_format="%.10f")
    (output / "static").mkdir(exist_ok=True)
    shutil.copyfile(static_release / "static/city_static.csv", output / "static/city_static.csv")
    shutil.copyfile(static_release / "schema.json", output / "static/schema.json")
    shutil.copyfile(static_release / "quality.json", output / "static/quality.json")
    write_json(output / "upstream_static_release.json", upstream)
    write_json(output / "schema.json", {
        "monthly_key": ["year", "month"], "annual_key": ["year"], "units": "degrees Celsius",
        "boxes_degrees_east": BOXES, "regional_grid_details": regional_details,
        "climatology": "Per-origin, per-calendar-month mean of the last 30 years ending at that origin",
        "annual_aggregation": "Equal-weight mean of 12 monthly values, not daily or day-weighted temperature",
        "notes": ["These are observed historical indices, not future forecasts",
                  "North Pacific box is not the official PDO; North Atlantic box is not the official AMO",
                  "No detrending, full-period standardization, centered smoothing, or future index value is supplied",
                  "The raw SST includes the warming background; anomalies are not isolated natural variability",
                  "For a training origin use its own snapshot; do not load the 2018 snapshot in an earlier fold",
                  "Modern ERSSTv5 retrospective reconstruction is not certified as pre-origin-vintage input",
                  "No improvement in forecast skill is claimed without a line.py score model trial"],
        "gate_eligible": False,
    })
    provenance = {
        "schema_version": 1, "upstream": "enhance-v1", "upstream_release_sha256": sha(static_release / "release.json"),
        "sources": entries, "license": "NOAA U.S. government public data", "doi": "10.7289/V5T72FNM",
        "source_documentation": "https://www.ncei.noaa.gov/products/extended-reconstructed-sst",
        "cutoff_request": END, "origin_specific_transformations": True, "gate_eligible": False,
        "boundary_incident": "Initial Dec-31 NCSS probe returned an adjacent January-2019 month; rejected before caching or feature use. All retained responses use exact Dec-1 endpoint and pass date guards.",
        "files": {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*"))
                  if p.is_file() and p.name not in ("provenance.json", "quality.json")},
    }
    # Include the nested static quality file too; only omit the root self-records.
    provenance["files"]["static/quality.json"] = sha(output / "static/quality.json")
    write_json(output / "provenance.json", provenance)
    write_json(output / "quality.json", validate(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/enhancement/ocean-v2")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/enhancement/ocean")
    parser.add_argument("--static-release", type=Path, default=ROOT / "data/processed/releases/enhance-v1")
    parser.add_argument("--acquire-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate:
        validate(args.validate)
    elif args.acquire_only:
        for name in BOXES:
            dates, *_ = acquire(name, args.cache, args.offline)
            print(f"{name}: validated {len(dates)} months, end {dates[-1].date()}", flush=True)
    else:
        build(args.output, args.cache, args.static_release, args.offline)


if __name__ == "__main__":
    main()
