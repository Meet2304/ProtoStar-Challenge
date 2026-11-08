"""Acquire reproducible structural city covariates; no weather observations.

Run from the enhancement worktree:
  uv run --with rasterio==1.4.3 --with pyshp==2.3.1 python code/enhance_static.py
  uv run python code/enhance_static.py --validate outputs/enhancement/static-v1

The extra packages are isolated by uv; this does not change the shared lockfile.
Source snapshots and final tables live in the ignored build directory until
scripts/line.py release publishes them. Never access an assessment partition.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
EARTH_KM = 6371.0088
COAST_STEP_KM = 0.5
BUFFER_KM = 1.0
STATION_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
COAST_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "v5.1.2/10m_physical/ne_10m_coastline.shp"
)
WORLDCOVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
ELEVATION_URL = "https://epqs.nationalmap.gov/v1/json"
ELEVATION_BULK_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/getSamples"
CLASSES = {
    10: "tree", 20: "shrub", 30: "grass", 40: "cropland", 50: "built",
    60: "bare", 70: "snow_ice", 80: "water", 90: "wetland",
    95: "mangrove", 100: "moss_lichen",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def fetch(url: str, path: Path) -> dict:
    """This helper is deliberately restricted to the two structural sources."""
    if url not in (STATION_URL, COAST_URL):
        raise ValueError("Only the declared structural sources may use this downloader")
    metadata = path.with_suffix(path.suffix + ".json")
    if path.exists():
        entry = json.loads(metadata.read_text(encoding="utf-8"))
        if entry["sha256"] != sha(path) or entry["url"] != url:
            raise ValueError(f"Changed cached source: {path}")
        return entry
    response = requests.get(url, timeout=(20, 120))
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    entry = {
        "url": url, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": sha(path), "bytes": len(response.content),
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
    }
    write_json(metadata, entry)
    return entry


def xyz(lat, lon):
    lat, lon = np.broadcast_arrays(np.radians(lat), np.radians(lon))
    return np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)


def spherical_km(lat1, lon1, lat2, lon2):
    a, b = xyz(np.asarray(lat1), np.asarray(lon1)), xyz(np.asarray(lat2), np.asarray(lon2))
    return 2 * EARTH_KM * np.arcsin(np.clip(np.linalg.norm(a - b, axis=-1) / 2, 0, 1))


def densify_segment(a, b, step=COAST_STEP_KM):
    """Spherical interpolation with no greater than step km between samples."""
    first, last = xyz(a[1], a[0]), xyz(b[1], b[0])
    angle = float(2 * np.arcsin(np.clip(np.linalg.norm(first - last) / 2, 0, 1)))
    count = max(1, int(math.ceil(angle * EARTH_KM / step)))
    if angle < 1e-12:
        return first.reshape(1, 3)
    t = np.arange(count) / count  # the next segment supplies the shared last endpoint
    return (np.sin((1 - t) * angle)[:, None] * first + np.sin(t * angle)[:, None] * last) / np.sin(angle)


def coastline_tree(path: Path):
    import shapefile
    samples = []
    # Region contains every challenge city and its potentially nearest ocean shore.
    # Inland Great Lakes shorelines are not part of this ocean-coastline layer.
    reader = shapefile.Reader(shp=io.BytesIO(path.read_bytes()))
    for shape in reader.iterShapes():
        boundaries = list(shape.parts) + [len(shape.points)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            points = shape.points[start:stop]
            for a, b in zip(points[:-1], points[1:]):
                if (max(a[0], b[0]) < -135 or min(a[0], b[0]) > -50
                        or max(a[1], b[1]) < 10 or min(a[1], b[1]) > 65):
                    continue
                samples.append(densify_segment(a, b))
                samples.append(xyz(b[1], b[0]).reshape(1, 3))
    if not samples:
        raise ValueError("No coastline geometry")
    return cKDTree(np.concatenate(samples))


def station_table(path: Path, cities: pd.DataFrame):
    selected = {}
    ids = set(cities.city_id)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line[:11] in ids:
            selected[line[:11]] = {
                "city_id": line[:11], "station_catalog_lat": float(line[12:20]),
                "station_catalog_lon": float(line[21:30]), "elevation_m": float(line[31:37]),
                "catalog_line": line,
            }
    if set(selected) != ids:
        raise ValueError(f"Station metadata missing for {sorted(ids - set(selected))}")
    table = cities.merge(pd.DataFrame(selected.values()), on="city_id", validate="one_to_one")
    if (table.elevation_m == -999.9).any() or not np.isfinite(table.elevation_m).all():
        raise ValueError("Missing station elevation; do not impute a fake elevation")
    table["elevation_coordinate_offset_km"] = spherical_km(
        table.lat, table.lon, table.station_catalog_lat, table.station_catalog_lon)
    return table


def prefetch_elevations(cities, folder: Path, workers=4):
    """Use the same public 3DEP service in coordinate-checked batches."""
    pending = [city for city in cities.itertuples(index=False) if not (folder / f"{city.city_id}.json").exists()]
    batches = [pending[n:n+20] for n in range(0, len(pending), 20)]
    snapshots = folder.parent / "usgs-elevation-batches"

    def batch_query(batch):
        geometry = {"points": [[c.lon, c.lat] for c in batch], "spatialReference": {"wkid": 4326}}
        params = {"f": "json", "geometry": json.dumps(geometry, separators=(",", ":")),
                  "geometryType": "esriGeometryMultipoint", "returnFirstValueOnly": "true",
                  "interpolation": "RSP_BilinearInterpolation", "outFields": "*"}
        response = requests.get(ELEVATION_BULK_URL, params=params, timeout=(20, 180))
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise ValueError(str(payload["error"]))
        digest = hashlib.sha256(response.content).hexdigest()
        snapshots.mkdir(parents=True, exist_ok=True)
        snapshot_name = digest + ".json"
        (snapshots / snapshot_name).write_bytes(response.content)
        found = set()
        for sample in payload["samples"]:
            i = int(sample["locationId"])
            if i in found or not 0 <= i < len(batch):
                raise ValueError("Duplicate/out-of-range elevation location ID")
            city = batch[i]
            if not np.allclose([sample["location"]["y"], sample["location"]["x"]], [city.lat, city.lon], rtol=0, atol=1e-10):
                raise ValueError("Bulk elevation response changed coordinates")
            if not -100 < float(sample["value"]) < 5000:
                raise ValueError("Bulk elevation returned missing data")
            entry = {"url": response.url, "parameters": {"x": city.lon, "y": city.lat, "wkid": 4326, "units": "Meters"},
                     "response": sample, "response_sha256": digest,
                     "batch_response_file": f"../usgs-elevation-batches/{snapshot_name}", "batch_parameters": params,
                     "retrieved_utc": datetime.now(timezone.utc).isoformat(), "license": "public domain",
                     "method": "USGS 3DEP getSamples, bilinear interpolation; source units meters"}
            write_json(folder / f"{city.city_id}.json", entry)
            found.add(i)
        if len(found) != len(batch):
            raise ValueError("Incomplete elevation batch")
        print(f"3DEP batch saved {len(found)} elevations", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(batch_query, batch) for batch in batches]
        for future in as_completed(futures):
            try:
                future.result()
            except (requests.RequestException, ValueError, KeyError) as error:
                # Missing points still go through the individually validated EPQS fallback.
                print(f"3DEP batch fallback to individual queries: {type(error).__name__}", flush=True)


def city_elevation(city, folder: Path, offline=False):
    """Query structural ground elevation at the actual challenge coordinate."""
    path = folder / f"{city.city_id}.json"
    if path.exists():
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry["parameters"]["x"] != city.lon or entry["parameters"]["y"] != city.lat:
            raise ValueError("Elevation snapshot belongs to another coordinate")
    else:
        if offline:
            raise FileNotFoundError(f"Missing offline elevation snapshot: {city.city_id}")
        params = {"x": city.lon, "y": city.lat, "wkid": 4326, "units": "Meters", "includeDate": "true"}
        for attempt in range(3):
            try:
                response = requests.get(ELEVATION_URL, params=params, timeout=(20, 90))
                response.raise_for_status()
                payload = response.json()
                elevation = float(payload["value"])
                if not -100 < elevation < 5000:
                    raise ValueError("USGS returned a missing or implausible elevation")
                break
            except (requests.RequestException, KeyError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        entry = {"url": response.url, "parameters": params, "response": payload,
                 "response_sha256": hashlib.sha256(response.content).hexdigest(),
                 "retrieved_utc": datetime.now(timezone.utc).isoformat(), "license": "public domain"}
        write_json(path, entry)
    payload = entry["response"]
    if not np.allclose([payload["location"]["y"], payload["location"]["x"]], [city.lat, city.lon], rtol=0, atol=1e-10):
        raise ValueError("USGS response coordinates differ from the request")
    if not -100 < float(payload["value"]) < 5000 or not float(payload["resolution"]) > 0:
        raise ValueError("Missing or implausible cached elevation/resolution")
    acquired = payload.get("attributes", {}).get("AcquisitionDate", "unknown")
    acquired_raw = str(acquired) if acquired is not None else "unknown"
    if isinstance(acquired, (int, float)):
        acquired = datetime.fromtimestamp(acquired / 1000, timezone.utc).date().isoformat()
    elif acquired != "unknown":
        parsed = pd.to_datetime(acquired, format="%m/%d/%Y", errors="coerce")
        acquired = parsed.date().isoformat() if pd.notna(parsed) else "unknown"
    return {"city_id": city.city_id, "elevation_m": float(payload["value"]),
            "elevation_resolution_m": float(payload["resolution"]),
            "elevation_acquisition_date": acquired,
            "elevation_acquisition_date_raw": acquired_raw,
            "elevation_source": "USGS 3DEP ground elevation"}


def tile_url(lat, lon):
    bottom, left = int(math.floor(lat / 3) * 3), int(math.floor(lon / 3) * 3)
    tile = f"{'N' if bottom >= 0 else 'S'}{abs(bottom):02d}{'E' if left >= 0 else 'W'}{abs(left):03d}"
    return WORLDCOVER_BASE + f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"


def landcover_summary(arrays, city_lat, city_lon):
    """Summarize pixel centers within a 1-km spherical radius; ignore class 0."""
    totals = {code: 0.0 for code in CLASSES}
    valid_count, circle_count, point_class = 0, 0, None
    nearest = float("inf")
    for values, transform in arrays:
        dx, _, west, _, dy, north = transform
        lat = north + (np.arange(values.shape[0]) + 0.5) * dy
        lon = west + (np.arange(values.shape[1]) + 0.5) * dx
        distances = spherical_km(lat[:, None], lon[None, :], city_lat, city_lon)
        circle = distances <= BUFFER_KM
        valid = circle & (values != 0)
        unexpected = set(np.unique(values[valid])) - set(CLASSES)
        if unexpected:
            raise ValueError(f"Unknown WorldCover codes {unexpected}")
        weights = np.broadcast_to(np.cos(np.radians(lat))[:, None], values.shape)
        for code in CLASSES:
            totals[code] += float(weights[valid & (values == code)].sum())
        circle_count += int(circle.sum())
        valid_count += int(valid.sum())
        # Find the containing station pixel, rather than a neighboring center.
        r, c = math.floor((city_lat - north) / dy), math.floor((city_lon - west) / dx)
        if 0 <= r < values.shape[0] and 0 <= c < values.shape[1]:
            if distances[r, c] < nearest:
                point_class, nearest = int(values[r, c]), float(distances[r, c])
    total = sum(totals.values())
    if total <= 0 or circle_count == 0 or point_class not in CLASSES:
        raise ValueError("No valid land-cover pixels or invalid station pixel")
    return {
        "landcover_year": 2021, "landcover_station_class": point_class,
        "landcover_dominant_class_1km": max(totals, key=totals.get),
        "landcover_valid_fraction_1km": valid_count / circle_count,
        "landcover_valid_pixels_1km": valid_count,
        **{f"landcover_{CLASSES[k]}_fraction_1km": v / total for k, v in totals.items()},
    }


def city_landcover(city, folder: Path, offline=False):
    """Save source pixel chips so an offline rerun never needs a whole tile."""
    import rasterio
    from rasterio.windows import Window, from_bounds
    import warnings
    manifest = folder / f"{city.city_id}.json"
    arrays, entries = [], []
    if manifest.exists():
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        for entry in entries:
            path = folder / entry["file"]
            if sha(path) != entry["sha256"]:
                raise ValueError(f"Changed raster chip: {path}")
            with np.load(path, allow_pickle=False) as chip:
                if not np.array_equal(chip["center"], [city.lat, city.lon]):
                    raise ValueError("Raster chip belongs to a different coordinate")
                arrays.append((chip["values"], tuple(chip["transform"])))
        return landcover_summary(arrays, city.lat, city.lon)
    if offline:
        raise FileNotFoundError(f"Missing offline WorldCover snapshot for {city.city_id}")
    dlat = math.degrees(BUFFER_KM / EARTH_KM) * 1.01
    dlon = dlat / math.cos(math.radians(city.lat))
    west, east, south, north = city.lon - dlon, city.lon + dlon, city.lat - dlat, city.lat + dlat
    urls = sorted({tile_url(lat, lon) for lat in (south, north) for lon in (west, east)})
    folder.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(), rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_TIMEOUT=90, GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=2,
    ):
        warnings.filterwarnings("ignore", message="Setting the shape", category=DeprecationWarning)
        for index, url in enumerate(urls):
            for attempt in range(3):
                try:
                    with rasterio.open(url) as dataset:
                        if str(dataset.crs) != "EPSG:4326" or dataset.nodata != 0:
                            raise ValueError("Unexpected WorldCover spatial reference or nodata")
                        window = from_bounds(max(west, dataset.bounds.left), max(south, dataset.bounds.bottom),
                                             min(east, dataset.bounds.right), min(north, dataset.bounds.top), dataset.transform)
                        left, top = math.floor(window.col_off), math.floor(window.row_off)
                        right, bottom = math.ceil(window.col_off + window.width), math.ceil(window.row_off + window.height)
                        window = Window(left, top, right - left, bottom - top)
                        values = dataset.read(1, window=window)
                        transform = tuple(dataset.window_transform(window))[:6]
                    break
                except rasterio.errors.RasterioError:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            name = f"{city.city_id}-{index}.npz"
            np.savez_compressed(folder / name, values=values, transform=np.array(transform), center=[city.lat, city.lon])
            entry = {
                "file": name, "sha256": sha(folder / name), "url": url,
                "window_col_row_width_height": [left, top, right-left, bottom-top],
                "transform": transform, "crs": "EPSG:4326", "map_year": 2021,
                "edition": "v200", "license": "CC-BY-4.0",
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            }
            entries.append(entry)
            arrays.append((values, transform))
    write_json(manifest, entries)
    return landcover_summary(arrays, city.lat, city.lon)


def validate(folder: Path, cities_path: Path):
    table = pd.read_csv(folder / "static/city_static.csv")
    cities = pd.read_csv(cities_path)
    if table.city_id.duplicated().any() or set(table.city_id) != set(cities.city_id) or len(table) != 210:
        raise ValueError("Expected exactly one row for each of the 210 challenge cities")
    expected = cities.set_index("city_id").loc[table.city_id]
    if not np.allclose(table[["lat", "lon"]], expected[["lat", "lon"]], rtol=0, atol=1e-10):
        raise ValueError("Challenge station coordinates changed")
    numeric = table.select_dtypes(include="number")
    if not np.isfinite(numeric).all().all():
        raise ValueError("Missing or nonfinite structural values")
    if not table.elevation_m.between(-100, 5000).all() or not table.distance_to_ocean_coast_km.between(0, 3000).all():
        raise ValueError("Implausible geography")
    fractions = table[[f"landcover_{name}_fraction_1km" for name in CLASSES.values()]]
    if not ((fractions >= 0) & (fractions <= 1)).all().all() or not np.allclose(fractions.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Land-cover fractions must be nonnegative and sum to one")
    if not table.landcover_valid_fraction_1km.between(.99, 1).all():
        raise ValueError("Insufficient land-cover coverage")
    source_manifest = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    if source_manifest["city_file_sha256"] != sha(cities_path):
        raise ValueError("City identity snapshot changed")
    for name, digest in source_manifest["files"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"Snapshot hash mismatch: {name}")
    result = {
        "passed": True, "city_count": len(table), "missing_values": int(table.isna().sum().sum()),
        "minimum_landcover_valid_fraction": float(table.landcover_valid_fraction_1km.min()),
        "maximum_catalog_coordinate_offset_km": float(table.elevation_coordinate_offset_km.max()),
        "elevation_offset_over_1km_city_ids": table.loc[table.elevation_coordinate_offset_km > 1, "city_id"].tolist(),
        "unknown_elevation_acquisition_dates": int(table.elevation_acquisition_date.eq("unknown").sum()),
        "distance_method": "mean-radius spherical distance to coast sampled every <=0.5 km",
        "coast_sampling_error_bound_km": COAST_STEP_KM / 2,
        "scope": "structural validation only; no forecast, target, or assessment labels accessed",
    }
    print(json.dumps(result, indent=2))
    return result


def build(output: Path, cache: Path, cities_path: Path, workers=4, offline=False, from_release=None):
    output.mkdir(parents=True, exist_ok=True)
    cities = pd.read_csv(cities_path).sort_values("city_id").reset_index(drop=True)
    if from_release:
        if output.resolve() == from_release.resolve():
            raise ValueError("Never rebuild into an existing immutable release")
        validate(from_release, cities_path)
        original = json.loads((from_release / "provenance.json").read_text(encoding="utf-8"))
        station_path = from_release / "sources/ghcnd-stations-selected.txt"
        coast_path = from_release / "sources/ne_10m_coastline.shp"
        elevation_cache = from_release / "sources/usgs-elevation"
        chips = from_release / "sources/worldcover-2021-v200"
        source_entries, offline = original["sources"], True
    else:
        station_path, coast_path = cache / "ghcnd-stations.txt", cache / "ne_10m_coastline.shp"
        if offline and (not station_path.exists() or not coast_path.exists()):
            raise FileNotFoundError("Offline structural source cache is incomplete")
        source_entries = [fetch(STATION_URL, station_path), fetch(COAST_URL, coast_path)]
        elevation_cache = cache / "usgs-elevation"
        chips = cache / "worldcover-2021-v200"
        if not offline:
            prefetch_elevations(cities, elevation_cache, workers)
    table = station_table(station_path, cities)
    table = table.rename(columns={"elevation_m": "station_catalog_elevation_m"})
    elevations = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(lambda c: city_elevation(c, elevation_cache, offline), cities.itertuples(index=False)):
            elevations.append(result)
    table = table.merge(pd.DataFrame(elevations), on="city_id", validate="one_to_one")
    print(f"USGS elevation {len(elevations)}/{len(cities)}", flush=True)
    tree = coastline_tree(coast_path)
    chord, _ = tree.query(xyz(table.lat.to_numpy(), table.lon.to_numpy()))
    table["distance_to_ocean_coast_km"] = 2 * EARTH_KM * np.arcsin(np.clip(chord / 2, 0, 1))
    table["coast_sampling_error_bound_km"] = COAST_STEP_KM / 2
    summaries = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(city_landcover, city, chips, offline): city.city_id for city in cities.itertuples(index=False)}
        for future in as_completed(futures):
            city_id = futures[future]
            summaries[city_id] = {"city_id": city_id, **future.result()}
            count = len(summaries)
            if count % 20 == 0 or count == len(cities):
                print(f"WorldCover {count}/{len(cities)}", flush=True)
    source_dir = output / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "ghcnd-stations-selected.txt").write_text("\n".join(table.catalog_line) + "\n", encoding="utf-8")
    shutil.copyfile(coast_path, source_dir / coast_path.name)
    shutil.copytree(elevation_cache, source_dir / elevation_cache.name, dirs_exist_ok=True)
    batches = elevation_cache.parent / "usgs-elevation-batches"
    if batches.is_dir():
        shutil.copytree(batches, source_dir / batches.name, dirs_exist_ok=True)
    shutil.copytree(chips, source_dir / chips.name, dirs_exist_ok=True)
    table = table.drop(columns="catalog_line").merge(pd.DataFrame(summaries.values()), on="city_id", validate="one_to_one")
    table = table.sort_values("city_id").reset_index(drop=True)
    (output / "static").mkdir(exist_ok=True)
    table.to_csv(output / "static/city_static.csv", index=False, float_format="%.10f")
    schema = {
        "key": "city_id", "structural": True, "data_rule": "Fixed city facts permitted from any year",
        "coordinates": "Current challenge observing station; not city center or earlier station segments",
        "elevation_m": "USGS 3DEP ground elevation in meters at the supplied station coordinate; EPQS or direct getSamples",
        "station_catalog_elevation_m": "Additional NOAA catalog station elevation; offset column flags coordinate differences",
        "elevation_resolution_m": "USGS response raster resolution in meters; acquisition date is recorded separately",
        "distance_to_ocean_coast_km": "Natural Earth v5.1.2-tag coastline on mean-radius sphere; Great Lakes excluded",
        "coast_sampling_error_bound_km": "Sampling alone adds <=0.25km; source generalization and spherical Earth add uncertainty",
        "landcover": "ESA WorldCover 2021 v200; 10m classes; radius 1km, cosine-latitude weighted fractions",
        "class_codes": CLASSES,
        "limitations": ["2021 structural snapshot does not reconstruct historical urban development",
                       "Tree class fraction is not percent canopy or vegetation health", "Built class is not impervious fraction",
                       "Current station coordinates and ground elevations may differ from earlier record segments",
                       "Numeric class codes are categorical, not an ordered continuous variable"],
    }
    write_json(output / "schema.json", schema)
    manifest = {
        "schema_version": 1, "city_file_sha256": sha(cities_path), "sources": source_entries,
        "worldcover": {"url": "https://esa-worldcover.org/en/data-access", "doi": "10.5281/zenodo.7254221",
                       "license": "CC-BY-4.0", "attribution": "ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium",
                       "snapshots": "sources/worldcover-2021-v200/*; per-chip URLs, windows, and hashes"},
        "licenses": {"GHCN-D metadata": "U.S. government public data", "Natural Earth": "public domain", "WorldCover": "CC-BY-4.0"},
        "elevation": {"url": ELEVATION_URL, "source": "USGS 3DEP Elevation Point Query Service",
                      "batch_url": ELEVATION_BULK_URL,
                      "license": "public domain", "snapshots": "sources/usgs-elevation/*; query parameters and responses"},
        "runtime": {"rasterio": "1.4.3", "pyshp": "2.3.1", "dependencies": "uv.lock plus uv --with extras"},
        "files": {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*"))
                  if p.is_file() and p.name not in ("provenance.json", "quality.json")},
    }
    write_json(output / "provenance.json", manifest)
    write_json(output / "quality.json", validate(output, cities_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/enhancement/static-v1")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/enhancement/static")
    parser.add_argument("--cities", type=Path, default=ROOT / "data/city_info.csv")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--from-release", type=Path, help="Rebuild offline from a pulled release's own source snapshots")
    args = parser.parse_args()
    if args.validate:
        validate(args.validate, args.cities)
    else:
        build(args.output, args.cache, args.cities, args.workers, args.offline, args.from_release)


if __name__ == "__main__":
    main()
