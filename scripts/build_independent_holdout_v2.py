"""Acquire a preregistered, analysis-only GHCN-D geographic holdout.

No model fitting or scoring occurs here. ``select`` freezes 48 stations from
structural catalogs; ``fetch`` acquires bounded daily CSVs; ``build`` aggregates
only the validated cache. All commands require a committed preregistration.
See ``--help``. Network access uses only Python's standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from common import METRICS, TRAIN_REVISION, assert_no_post_cutoff

BENCHMARK = "protostar-external-v2"
START, END = "1950-01-01", "2018-12-31"
API = "https://www.ncei.noaa.gov/access/services/data/v1"
CATALOGS = {
    "stations": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt",
    "inventory": "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt",
}
DEFAULT_OUTPUT = ROOT / "outputs/independent-v2"
PRIOR_SELECTION = "outputs/independent-v1/selection.json"
PRIOR_SELECTION_SHA256 = "2a3da7755f1a4870097ee11dfea07d82c64f19cd2b265853d5bb5d06feef5efc"
PRIOR_QUALITY = "outputs/independent-v1/quality.json"
PRIOR_QUALITY_SHA256 = "c21efcffaf163217b082d2c223a73685dc52889d827837e3384e6d3747dfe657"
RAW_CITY = ROOT / "data/raw/city_info/train-00000-of-00001.parquet"
CURATED_CITY = ROOT / "data/city_info.csv"
STRATA = tuple(f"{ns}_{ew}" for ns in ("north", "south") for ew in ("west", "central", "east"))
ALGORITHM = {
    "version": BENCHMARK, "start": START, "end": END,
    "units": "standard: Fahrenheit and inches", "data_types": ["TMAX", "TMIN", "PRCP"],
    "selection": "SHA256(protostar-external-v2|station_id), first eight in each stratum",
    "station_network": "USW prefix only (WBAN network, often airport stations; not all stations are airports)",
    "stations_per_stratum": 8,
    "prior_exclusions": {"selection": PRIOR_SELECTION, "selection_sha256": PRIOR_SELECTION_SHA256,
                         "all_ids_names_and_locations": True, "distance_km_min": 100},
    "conus_bounds": [25, 49, -125, -67], "latitude_split": 37,
    "longitude_splits": [-104, -87], "distance_km_min": 100,
    "inventory": "each of TMAX,TMIN,PRCP begins by 1950 and ends no earlier than 2018",
    "annual_avg_temp": "(mean(valid TMAX) + mean(valid TMIN))/2; published-definition approximation",
    "cdd": "sum(max((TMAX+TMIN)/2 - 65 F, 0)) over paired valid days",
    "quality": "nonempty QFLAG invalid; temperature outside [-130,140] F invalid; negative PRCP invalid; TMAX<TMIN invalidates both",
    "trace": "PRCP measurement flag T with numeric zero is valid; count recorded",
    "missing": "full calendar reindex; totals require at least one valid day",
    "station_history": "one station ID; physical moves/changes unknown; segment/change fields null",
    "eligibility": {"baseline_1982_2011": 20, "target_2012_2018": 6,
                    "live_2012_2014": 2, "final_2015_2018": 3,
                    "metric_validity": "official per-variable missing counts <=10 and finite value"},
    "minimum_benchmark": {"stations": 12, "strata": 4},
    "limitation": "Current archive QC and revisions; single-station aggregation is not the challenge city composite. Supplementary geographic generalization, never private-answer-key validation or rank.",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(data) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(path: Path) -> str:
    return sha(Path(path).read_bytes())


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_exclusive(path: Path, data: bytes):
    """Atomic publication without ever replacing an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".holdout-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # fails exclusively if destination exists
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, data):
    write_exclusive(Path(path), json.dumps(data, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_repo_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("pin path escapes repository")
    return path


def verify_prereg(path: Path):
    path = Path(path).resolve()
    relative = path.relative_to(ROOT.resolve()).as_posix()
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT,
                               capture_output=True, check=True).stdout
    if committed != path.read_bytes():
        raise ValueError("preregistration must be committed and unchanged before acquisition")
    prereg = read_json(path)
    if prereg.get("schema_version") != 1 or prereg.get("benchmark_id") != BENCHMARK:
        raise ValueError("incorrect preregistration identity")
    if prereg.get("script_sha256") != digest(Path(__file__)):
        raise ValueError("builder differs from preregistered script")
    if prereg.get("design") != ALGORITHM:
        raise ValueError("preregistration design differs from the fixed algorithm")
    if prereg.get("prior_pilot") != {
        "selection": PRIOR_SELECTION, "selection_sha256": PRIOR_SELECTION_SHA256,
        "quality": PRIOR_QUALITY, "quality_sha256": PRIOR_QUALITY_SHA256,
        "eligible_stations": 5, "selected_stations": 24, "minimum_stations": 12,
        "no_scores_computed": True,
    }:
        raise ValueError("preregistration must bind the failed V1 coverage pilot")
    if prereg.get("analysis_script_sha256") != digest(ROOT / "scripts/evaluate_independent_holdout.py"):
        raise ValueError("evaluation runner differs from preregistered analysis script")
    verify_prior_artifacts()
    pins = prereg.get("source_files", {})
    if "code/common.py" not in pins or prereg.get("source_hash") != sha(canonical(pins)):
        raise ValueError("preregistration requires canonical source_files hash and code/common.py")
    current_paths = {p for folder in ("code", "src") for p in (ROOT / folder).rglob("*.py") if p.is_file()}
    current_paths.update(ROOT / name for name in ("uv.lock", "pyproject.toml", "requirements.txt", "code/requirements.txt") if (ROOT / name).is_file())
    if {p.relative_to(ROOT).as_posix() for p in current_paths} != set(pins):
        raise ValueError("preregistered source file set changed")
    for name, expected in pins.items():
        if sha(safe_repo_path(name).read_bytes().replace(b"\r\n", b"\n")) != expected:
            raise ValueError(f"preregistered source changed: {name}")
    return prereg


def observation_url(station_id):
    if not re.fullmatch(r"US[A-Z0-9]{9}", station_id):
        raise ValueError("invalid GHCN station ID")
    url = API + "?" + urlencode({"dataset": "daily-summaries", "stations": station_id,
        "startDate": START, "endDate": END, "dataTypes": "TMAX,TMIN,PRCP",
        "units": "standard", "includeAttributes": "true", "format": "csv"})
    validate_observation_url(url)
    return url


def validate_observation_url(url):
    """Fail closed before *every* weather request, including redirects."""
    parsed = urlsplit(url)
    endpoint = urlsplit(API)
    if (parsed.scheme, parsed.netloc, parsed.path) != (endpoint.scheme, endpoint.netloc, endpoint.path) or parsed.fragment:
        raise ValueError("only the bounded NOAA Data Access Service is permitted")
    query = parse_qs(parsed.query, keep_blank_values=True)
    expected = {"dataset": ["daily-summaries"], "startDate": [START], "endDate": [END],
                "dataTypes": ["TMAX,TMIN,PRCP"], "units": ["standard"],
                "includeAttributes": ["true"], "format": ["csv"]}
    if set(query) != set(expected) | {"stations"} or any(query.get(k) != v for k, v in expected.items()):
        raise ValueError("weather request must retain every preregistered date/unit/attribute cap")
    if len(query["stations"]) != 1 or not re.fullmatch(r"US[A-Z0-9]{9}", query["stations"][0]):
        raise ValueError("request requires one explicit station")
    assert_no_post_cutoff(pd.DataFrame({"Year": [int(query["endDate"][0][:4])]}), source="NOAA request")
    return query["stations"][0]


class CappedRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_observation_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url, *, weather=False):
    if weather:
        validate_observation_url(url)
    elif url not in CATALOGS.values():
        raise ValueError("unrecognized structural catalog")
    opener = build_opener(CappedRedirect())  # catalog redirects also cannot enter an unbounded weather endpoint
    last_error = None
    for attempt in range(3):  # at most two retries; no backoff or concurrency
        try:
            with opener.open(Request(url, headers={"User-Agent": "ProtoStar-independent-analysis/1"}), timeout=45) as response:
                if weather:
                    validate_observation_url(response.geturl())
                elif response.geturl() != url:
                    raise ValueError("catalog redirect refused")
                payload = response.read(40_000_001)
                if len(payload) > 40_000_000:
                    raise ValueError("response exceeds fixed 40 MB limit")
                return payload, {"request_url": url, "retrieved_utc": utcnow(),
                    "sha256": sha(payload), "headers": {k: response.headers[k] for k in ("Last-Modified", "ETag") if response.headers.get(k)}}
        except (OSError, TimeoutError) as exc:
            last_error = exc
    raise RuntimeError("bounded NOAA download failed after two retries") from last_error


def normalize_name(name):
    return " ".join(re.findall(r"[A-Z0-9]+", str(name).upper()))


def verify_prior_artifacts():
    """Bind the unused cohort to every station in the frozen V1 coverage pilot."""
    selection_path = safe_repo_path(PRIOR_SELECTION)
    if digest(selection_path) != PRIOR_SELECTION_SHA256:
        raise ValueError("prior V1 selection changed")
    anchor = subprocess.run(["git", "show", f"HEAD:{PRIOR_SELECTION}"], cwd=ROOT,
                            capture_output=True, check=True).stdout
    if anchor != selection_path.read_bytes():
        raise ValueError("prior V1 selection must be committed and unchanged")
    if digest(safe_repo_path(PRIOR_QUALITY)) != PRIOR_QUALITY_SHA256:
        raise ValueError("prior V1 pilot quality changed")
    prior = read_json(selection_path)
    if prior.get("benchmark_id") != "protostar-external-v1" or len(prior.get("stations", [])) != 24:
        raise ValueError("prior V1 selection identity/count invalid")
    if len({r["station_id"] for r in prior["stations"]}) != 24:
        raise ValueError("prior V1 selection IDs must be unique")
    for row in prior["stations"]:
        if not re.fullmatch(r"US[A-Z0-9]{9}", row["station_id"]) or not normalize_name(row["name"]):
            raise ValueError("prior V1 station ID/name invalid")
        lat, lon = float(row["lat"]), float(row["lon"])
        if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("prior V1 station location invalid")
    return prior


def pinned_exclusions(raw_path=RAW_CITY, curated_path=CURATED_CITY):
    raw_path, curated_path = Path(raw_path), Path(curated_path)
    metadata = ROOT / "data/raw/.cache/huggingface/download/city_info/train-00000-of-00001.parquet.metadata"
    lines = metadata.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or lines[0] != TRAIN_REVISION or lines[1] != digest(raw_path):
        raise ValueError("city_info must be the revision-verified pinned subset")
    raw, curated = pd.read_parquet(raw_path), pd.read_csv(curated_path)
    required = {"ID", "Name", "Stn.Name", "Lat", "Lon"}
    if not required <= set(raw) or not {"city_id", "name", "lat", "lon"} <= set(curated):
        raise ValueError("challenge station/location exclusion schema missing")
    ids = set(raw.ID.dropna().astype(str)) | set(curated.city_id.dropna().astype(str))
    names = {normalize_name(x) for col in (raw.Name, raw["Stn.Name"], curated.name) for x in col.dropna()}
    locations = [(float(a), float(b)) for table, lat, lon in ((raw, "Lat", "Lon"), (curated, "lat", "lon")) for a, b in zip(table[lat], table[lon])]
    if not locations or any(not math.isfinite(a) or not math.isfinite(b) or not -90 <= a <= 90 or not -180 <= b <= 180 for a, b in locations):
        raise ValueError("every challenge location must be finite and valid")
    pins = {str(p.resolve().relative_to(ROOT.resolve()).as_posix()): digest(p) for p in (raw_path, curated_path, metadata)}
    prior = verify_prior_artifacts()
    for station in prior["stations"]:
        ids.add(station["station_id"])
        names.add(normalize_name(station["name"]))
        locations.append((float(station["lat"]), float(station["lon"])))
    pins.update({PRIOR_SELECTION: PRIOR_SELECTION_SHA256, PRIOR_QUALITY: PRIOR_QUALITY_SHA256})
    return ids, names, sorted(set(locations)), pins


def parse_stations(payload):
    rows = []
    for line in payload.decode("ascii").splitlines():
        if not line.strip():
            continue
        rows.append({"station_id": line[0:11], "lat": float(line[12:20]),
                     "lon": float(line[21:30]), "elevation_m": float(line[31:37]),
                     "state": line[38:40].strip(), "name": line[41:71].strip()})
    if len({r["station_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate catalog station")
    return rows


def parse_inventory(payload):
    inventory = {}
    for line in payload.decode("ascii").splitlines():
        if not line.strip():
            continue
        element = line[31:35]
        if element in ("TMAX", "TMIN", "PRCP"):
            key = (line[:11], element)
            if key in inventory:
                raise ValueError("duplicate inventory station/element")
            inventory[key] = (int(line[36:40]), int(line[41:45]))
    return inventory


def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def stratum(lat, lon):
    return ("north" if lat >= 37 else "south") + "_" + ("west" if lon < -104 else "central" if lon < -87 else "east")


def select_stations(stations, inventory, ids, names, locations):
    if not locations:
        raise ValueError("challenge exclusion locations cannot be empty")
    eligible = {s: [] for s in STRATA}
    for row in stations:
        sid, lat, lon = row["station_id"], row["lat"], row["lon"]
        if not re.fullmatch(r"USW[A-Z0-9]{8}", sid) or not 25 <= lat <= 49 or not -125 <= lon <= -67:
            continue
        if sid in ids or normalize_name(row["name"]) in names:
            continue
        if any((sid, e) not in inventory or inventory[sid, e][0] > 1950 or inventory[sid, e][1] < 2018 for e in ("TMAX", "TMIN", "PRCP")):
            continue
        distance = min(haversine_km((lat, lon), point) for point in locations)
        if distance < 100:
            continue
        group = stratum(lat, lon)
        eligible[group].append({**row, "stratum": group, "nearest_excluded_km": distance,
            "selection_key": sha(f"{BENCHMARK}|{sid}".encode()),
            "inventory": {e: list(inventory[sid, e]) for e in ("TMAX", "TMIN", "PRCP")}})
    counts = {s: len(rows) for s, rows in eligible.items()}
    if any(n < 8 for n in counts.values()):
        raise ValueError(f"preregistered selection failed; need eight in every stratum: {counts}")
    selected = [r for s in STRATA for r in sorted(eligible[s], key=lambda r: (r["selection_key"], r["station_id"]))[:8]]
    return selected, counts


def select(args):
    prereg = verify_prereg(args.preregistration)
    output = args.output.resolve()
    if (output / "selection.json").exists():
        raise FileExistsError("selection already frozen")
    ids, names, locations, exclusion_pins = pinned_exclusions()
    metadata_pins, metadata_provenance, payloads = {}, {}, {}
    for kind, provided in (("stations", args.stations_file), ("inventory", args.inventory_file)):
        if provided:
            data = provided.read_bytes()
            provenance = {"source_url": CATALOGS[kind], "local_import": True, "sha256": sha(data)}
        else:
            data, provenance = download(CATALOGS[kind])
        payloads[kind] = data
        metadata_pins[f"metadata/{kind}.txt"] = sha(data)
        metadata_provenance[kind] = provenance
    selected, counts = select_stations(parse_stations(payloads["stations"]), parse_inventory(payloads["inventory"]), ids, names, locations)
    manifest = {"schema_version": 1, "benchmark_id": BENCHMARK, "created_utc": utcnow(),
        "preregistration": str(args.preregistration.resolve().relative_to(ROOT.resolve()).as_posix()),
        "preregistration_sha256": digest(args.preregistration), "script_sha256": digest(Path(__file__)),
        "source_hash": prereg["source_hash"], "algorithm": ALGORITHM, "algorithm_sha256": sha(canonical(ALGORITHM)),
        "metadata_files": metadata_pins, "metadata_provenance": metadata_provenance,
        "exclusion_files": exclusion_pins, "eligible_counts": counts, "stations": selected}
    for kind, payload in payloads.items():
        write_exclusive(output / f"metadata/{kind}.txt", payload)
    write_json(output / "selection.json", manifest)
    # This lock is separately checked, but a committed git copy is the real immutable selection anchor.
    write_json(output / "selection.lock.json", {"selection_sha256": digest(output / "selection.json")})
    print(f"Selected {len(selected)} stations. Commit selection.json and its metadata before fetch; no observations acquired.")


def verify_selection(path, *, committed=True):
    path = Path(path).resolve()
    manifest = read_json(path)
    lock = read_json(path.with_name("selection.lock.json"))
    if lock != {"selection_sha256": digest(path)}:
        raise ValueError("selection manifest changed")
    if committed:
        relative = path.relative_to(ROOT.resolve()).as_posix()
        anchor = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, capture_output=True, check=True).stdout
        if anchor != path.read_bytes():
            raise ValueError("station selection must be committed unchanged before fetching observations")
    prereg_path = safe_repo_path(manifest["preregistration"])
    prereg = verify_prereg(prereg_path)
    if manifest["preregistration_sha256"] != digest(prereg_path) or manifest["source_hash"] != prereg["source_hash"]:
        raise ValueError("preregistration binding changed")
    if manifest.get("algorithm") != ALGORITHM or manifest.get("algorithm_sha256") != sha(canonical(ALGORITHM)) or manifest.get("script_sha256") != digest(Path(__file__)):
        raise ValueError("selection algorithm/script changed")
    for name, expected in manifest["metadata_files"].items():
        if digest(path.parent / name) != expected:
            raise ValueError(f"structural metadata changed: {name}")
    for name, expected in manifest["exclusion_files"].items():
        if digest(safe_repo_path(name)) != expected:
            raise ValueError(f"challenge exclusions changed: {name}")
    ids, names, locations, exclusion_pins = pinned_exclusions()
    chosen, counts = select_stations(parse_stations((path.parent / "metadata/stations.txt").read_bytes()),
        parse_inventory((path.parent / "metadata/inventory.txt").read_bytes()), ids, names, locations)
    if chosen != manifest["stations"] or counts != manifest["eligible_counts"] or exclusion_pins != manifest["exclusion_files"]:
        raise ValueError("selection does not reproduce exactly from structural metadata")
    return manifest


def parse_daily(payload: bytes, station_id: str):
    frame = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
    required = {"STATION", "DATE", "TMAX", "TMIN", "PRCP", "TMAX_ATTRIBUTES", "TMIN_ATTRIBUTES", "PRCP_ATTRIBUTES"}
    if not required <= set(frame):
        raise ValueError(f"NOAA CSV missing columns: {sorted(required - set(frame))}")
    if not frame.STATION.eq(station_id).all():
        raise ValueError("response station does not match request")
    if not frame.DATE.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("DATE must be exact ISO calendar dates")
    dates = pd.to_datetime(frame.DATE, format="%Y-%m-%d", errors="raise")
    if dates.duplicated().any() or (dates < pd.Timestamp(START)).any() or (dates > pd.Timestamp(END)).any():
        raise ValueError("duplicate or out-of-request DATE; response must not be persisted")
    assert_no_post_cutoff(pd.DataFrame({"Year": dates.dt.year}), source="NOAA daily response")
    out = pd.DataFrame(index=pd.DatetimeIndex(dates))
    out.index.name = "DATE"
    trace = np.zeros(len(frame), dtype=bool)
    for element in ("TMAX", "TMIN", "PRCP"):
        text = frame[element].str.strip()
        attribute_text = frame[f"{element}_ATTRIBUTES"].str.strip()
        # The service leaves attributes empty when the observation is absent.
        # Numeric values still require the advertised attribute schema.
        attribute_text = attribute_text.mask(text.eq("") & attribute_text.eq(""), ",,")
        attributes = attribute_text.str.split(",")
        if attributes.map(len).lt(3).any():
            raise ValueError("attribute fields must include measurement, quality, source flags")
        quality = attributes.map(lambda parts: parts[1].strip())
        numeric = pd.to_numeric(text.replace("", np.nan), errors="raise").astype(float)
        if np.isinf(numeric).any() or (text.ne("") & numeric.isna()).any():
            raise ValueError("non-finite NOAA observation")
        invalid = quality.ne("") | numeric.eq(-9999)
        invalid |= numeric.lt(0) if element == "PRCP" else ~numeric.between(-130, 140)
        values = numeric.mask(invalid)
        out[element] = values.to_numpy()
        if element == "PRCP":
            trace = (attributes.map(lambda parts: parts[0].strip()).eq("T") & values.eq(0)).to_numpy()
    inverted = out.TMAX.lt(out.TMIN)
    out.loc[inverted, ["TMAX", "TMIN"]] = np.nan
    out["PRCP_TRACE"] = trace
    return out.sort_index()


def fetch(args):
    manifest = verify_selection(args.selection)
    output = args.selection.resolve().parent
    for station in manifest["stations"]:
        sid = station["station_id"]
        raw, provenance = output / f"raw/{sid}.csv", output / f"raw/{sid}.provenance.json"
        if raw.exists() or provenance.exists():
            if not raw.exists() or not provenance.exists():
                raise ValueError(f"incomplete immutable station cache: {sid}")
            verify_cached(raw, provenance, sid, digest(args.selection))
            continue
        url = observation_url(sid)
        payload, record = download(url, weather=True)
        parse_daily(payload, sid)  # validate every date before writing any raw bytes
        record.update({"station_id": sid, "selection_sha256": digest(args.selection)})
        write_exclusive(raw, payload)
        write_json(provenance, record)
        print(f"Validated and saved {sid}, {START} through {END}", flush=True)


def verify_cached(raw, provenance, station_id, selection_hash):
    record = read_json(provenance)
    if record.get("sha256") != digest(raw) or record.get("station_id") != station_id or record.get("selection_sha256") != selection_hash:
        raise ValueError("cached response provenance/hash mismatch")
    if record.get("request_url") != observation_url(station_id):
        raise ValueError("cached response request changed")
    validate_observation_url(record["request_url"])
    return parse_daily(Path(raw).read_bytes(), station_id), record


def aggregate_daily(daily, station):
    if daily.index.duplicated().any() or not isinstance(daily.index, pd.DatetimeIndex):
        raise ValueError("unique daily datetime index required")
    if (daily.index < pd.Timestamp(START)).any() or (daily.index > pd.Timestamp(END)).any():
        raise ValueError("daily input out of request period")
    assert_no_post_cutoff(pd.DataFrame({"Year": daily.index.year}), source="aggregation daily input")
    daily = daily.reindex(pd.date_range(START, END, freq="D"))
    annual, coverage = [], []
    for year, group in daily.groupby(daily.index.year):
        pair = group.TMAX.notna() & group.TMIN.notna()
        cdd = ((group.TMAX + group.TMIN) / 2 - 65).clip(lower=0).sum(min_count=1)
        annual.append({"ID": station["station_id"], "Year": int(year), "Name": station["name"],
            "Avg Temp": (group.TMAX.mean() + group.TMIN.mean()) / 2,
            "ToT Prcp": group.PRCP.sum(min_count=1), "Max Tmax": group.TMAX.max(), "cdd": cdd,
            "Miss Tmax": int(group.TMAX.isna().sum()), "Miss Tmin": int(group.TMIN.isna().sum()),
            "Miss Prcp": int(group.PRCP.isna().sum())})
        coverage.append({"ID": station["station_id"], "Year": int(year), "Name": station["name"],
            "Days Both": int(pair.sum()), "Station Segment": None, "Station Changes": None,
            "Station ID Count": 1, "PRCP Trace Days": int(group.PRCP_TRACE.fillna(False).sum()),
            "Station Metadata Note": "Physical station moves and changes unknown; no HOMR history used"})
    annual, coverage = pd.DataFrame(annual), pd.DataFrame(coverage)
    assert_no_post_cutoff(annual, source="annual aggregation")
    assert_no_post_cutoff(coverage, source="coverage aggregation")
    return annual, coverage


def quality_result(annual, station):
    assert_no_post_cutoff(annual, source="quality eligibility input")
    counts, reasons = {}, []
    for metric, (value, missing) in METRICS.items():
        valid = annual[missing].le(10).all(axis=1) & np.isfinite(annual[value])
        counts[metric] = {}
        for label, start, end, minimum in (("baseline", 1982, 2011, 20), ("target", 2012, 2018, 6),
                                           ("live", 2012, 2014, 2), ("final", 2015, 2018, 3)):
            count = int((valid & annual.Year.between(start, end)).sum())
            counts[metric][label] = count
            if count < minimum:
                reasons.append(f"{metric}:{label} valid years {count} < {minimum}")
    return {"station_id": station["station_id"], "stratum": station["stratum"],
            "eligible": not reasons, "valid_years": counts, "failure_reasons": reasons}


def build(args):
    manifest = verify_selection(args.selection)
    output = args.selection.resolve().parent
    destinations = [output / name for name in ("annual_indices.csv", "cities.csv", "coverage.csv", "provenance.json", "quality.json")]
    if any(p.exists() for p in destinations):
        raise FileExistsError("built artifacts already exist; immutable output cannot be overwritten")
    annuals, coverages, quality, provenance = [], [], [], []
    for station in manifest["stations"]:  # every candidate, no performance-based replacement
        sid = station["station_id"]
        daily, record = verify_cached(output / f"raw/{sid}.csv", output / f"raw/{sid}.provenance.json", sid, digest(args.selection))
        annual, coverage = aggregate_daily(daily, station)
        annuals.append(annual)
        coverages.append(coverage)
        quality.append(quality_result(annual, station))
        provenance.append(record)
    eligible = [r for r in quality if r["eligible"]]
    enough = len(eligible) >= 12 and len({r["stratum"] for r in eligible}) >= 4
    result = {"benchmark_id": BENCHMARK, "status": "eligible_for_supplementary_evaluation" if enough else "not_validated",
        "eligible_stations": len(eligible), "eligible_strata": len({r["stratum"] for r in eligible}),
        "selection_sha256": digest(args.selection), "stations": quality,
        "limitations": ALGORITHM["limitation"], "no_scores_computed": True}
    annual, coverage = pd.concat(annuals, ignore_index=True), pd.concat(coverages, ignore_index=True)
    cities = pd.DataFrame([{"city_id": s["station_id"], "name": s["name"], "lat": s["lat"], "lon": s["lon"],
        "stratum": s["stratum"], "eligible": q["eligible"]} for s, q in zip(manifest["stations"], quality)])
    for path, table in zip(destinations[:3], (annual, cities, coverage)):
        write_exclusive(path, table.to_csv(index=False, lineterminator="\n").encode())
    write_json(destinations[3], {"selection_sha256": digest(args.selection), "responses": provenance,
        "algorithm": ALGORITHM, "output_sha256": {p.name: digest(p) for p in destinations[:3]}})
    write_json(destinations[4], result)
    print(f"{result['status']}: {len(eligible)} eligible stations; no scores computed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    selection = commands.add_parser("select", help="metadata only; freeze candidates before observations")
    selection.add_argument("--preregistration", type=Path, required=True)
    selection.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    selection.add_argument("--stations-file", type=Path, help="optional previously acquired structural catalog")
    selection.add_argument("--inventory-file", type=Path, help="optional previously acquired structural catalog")
    selection.set_defaults(func=select)
    for command, help_text, function in (("fetch", "fetch all 48 date-capped station CSVs", fetch),
                                         ("build", "aggregate validated cached CSVs; no network", build)):
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("--selection", type=Path, default=DEFAULT_OUTPUT / "selection.json")
        sub.set_defaults(func=function)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
