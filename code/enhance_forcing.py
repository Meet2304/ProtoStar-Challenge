"""Historical global ERF add-on with dated sources and origin-specific cuts.

  uv run python code/enhance_forcing.py --acquire-only
  uv run python code/enhance_forcing.py --offline

The AR6 source's last row is 2019. Acquisition reads only an HTTP byte-range
prefix ending at the complete 2018 record; it never requests the 2019 row.
AR5 is a separate <=2011 reference, not spliced into the AR6 component series.
All time-varying forcing inputs are conservatively declared weather_input.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from common import DataRuleViolation, assert_no_post_cutoff, effective_cutoff_year
from enhance_static import sha, write_json

ROOT = Path(__file__).resolve().parents[1]
AR6_COMMIT = "32ca11dc8a90f9eddee23686e08bff7a88f11783"
AR5_COMMIT = "fcd941dc9729d16cb1df81758214bc34b20884dc"
AR6_URL = f"https://raw.githubusercontent.com/IPCC-WG1/Chapter-7/{AR6_COMMIT}/data_output/AR6_ERF_1750-2019.csv"
AR5_URL = f"https://raw.githubusercontent.com/OMS-NetZero/FAIR/{AR5_COMMIT}/fair/ancil/cmip5_annex2_forcing.csv"
ORIGINS = (1983, 1990, 1997, 2004, 2011, 2018)
RELEASE_ID = "enhance-v3"
COMPONENTS = {
    "co2": "co2_erf_wm2", "ch4": "ch4_erf_wm2", "n2o": "n2o_erf_wm2",
    "other_wmghg": "other_wmghg_erf_wm2", "o3": "ozone_erf_wm2",
    "h2o_stratospheric": "stratospheric_water_erf_wm2", "contrails": "contrails_erf_wm2",
    "aerosol-radiation_interactions": "aerosol_radiation_erf_wm2",
    "aerosol-cloud_interactions": "aerosol_cloud_erf_wm2",
    "bc_on_snow": "black_carbon_on_snow_erf_wm2", "land_use": "land_use_erf_wm2",
    "volcanic": "volcanic_erf_wm2", "solar": "solar_erf_wm2",
    "total_anthropogenic": "anthropogenic_total_erf_wm2",
    "total_natural": "natural_total_erf_wm2", "total": "total_erf_wm2",
}


def dated_table(content, start, end):
    frame = pd.read_csv(BytesIO(content))
    frame = frame.rename(columns={"Year": "year"})
    assert_no_post_cutoff(frame, "year", "historical radiative forcing")
    years = frame.year.to_numpy()
    if not np.array_equal(years, np.arange(start, end + 1)):
        raise DataRuleViolation(f"Expected exactly one forcing record per year, {start}..{end}")
    if not np.isfinite(frame.select_dtypes(include="number")).all().all() or frame.isna().any().any():
        raise ValueError("Incomplete/nonfinite forcing source")
    return frame


def csv_prefix(read_range, start_year=1750, end_year=2018, initial_bytes=1024):
    """Read an ascending unquoted numeric CSV without fetching a later record.

    read_range(offset,count) must return exactly those bytes. Once the header is
    known, each request is bounded by minimum remaining record lengths (four
    year digits, delimiters, newline). At the final record, only the remaining
    delimiters, then single bytes, are requested. Thus no following row is read.
    The pinned sources use this simple ASCII numeric format, verified below.
    """
    if not 1750 <= start_year <= end_year <= 2018:
        raise DataRuleViolation("Acquisition cutoff must be no later than 2018")
    collected, pending = bytearray(), bytearray()
    last, width, header = start_year - 1, None, False
    amount = min(initial_bytes, (end_year - start_year + 1) * 5)
    requests_made = 0
    while last < end_year:
        piece = read_range(len(collected), amount)
        if len(piece) != amount:
            raise ValueError("Server returned an incomplete byte range")
        collected.extend(piece)
        pending.extend(piece)
        requests_made += 1
        while b"\n" in pending:
            line, _, rest = pending.partition(b"\n")
            pending = bytearray(rest)
            line = line.rstrip(b"\r")
            if not line:
                continue
            if not header:
                fields = line.split(b",")
                if fields[0].lower() != b"year":
                    raise ValueError("Expected forcing CSV year header")
                width, header = len(fields), True
                continue
            fields = line.split(b",")
            if len(fields) != width or b'"' in line or int(fields[0]) != last + 1:
                raise ValueError("Unexpected forcing CSV layout/year sequence")
            if not all(np.isfinite(float(value)) for value in fields):
                raise ValueError("Nonfinite forcing value")
            last += 1
            if last == end_year:
                if pending:
                    raise DataRuleViolation("Acquisition crossed the final permitted record")
                break
        if last == end_year:
            break
        if not header:
            amount = 1
        else:
            remaining_rows = end_year - last
            remaining_commas = width - 1 - pending.count(b",")
            if remaining_commas < 0:
                raise ValueError("Malformed partial forcing record")
            amount = max(1, min(8192, (remaining_rows - 1) * (width + 4) + remaining_commas))
        if requests_made % 50 == 0:
            print(f"Bounded forcing acquisition: through {last}, {len(collected)} bytes", flush=True)
    return bytes(collected), requests_made


def acquire(cache: Path, offline=False):
    cache.mkdir(parents=True, exist_ok=True)
    ar6, ar5 = cache / "ar6_erf_1750-2018.csv", cache / "ar5_erf_1750-2011.csv"
    meta = cache / "acquisition.json"
    if ar6.exists() and ar5.exists() and meta.exists():
        record = json.loads(meta.read_text(encoding="utf-8"))
        if record["ar6"]["sha256"] != sha(ar6) or record["ar5"]["sha256"] != sha(ar5):
            raise ValueError("Forcing source snapshot changed")
        dated_table(ar6.read_bytes(), 1750, 2018)
        dated_table(ar5.read_bytes(), 1750, 2011)
        return ar6, ar5, record
    if offline:
        raise FileNotFoundError("Forcing source cache is incomplete")
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=.5,
                   status_forcelist=(429, 500, 502, 503, 504))))
    total_size, etag = None, None

    def read_range(offset, count):
        nonlocal total_size, etag
        end = offset + count - 1
        with session.get(AR6_URL, params={"prefix_range": f"{offset}-{end}"},
                         headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"},
                         stream=True, timeout=(20, 60)) as response:
            # Check status/headers before consuming a body that might ignore Range.
            if response.status_code != 206:
                raise ValueError(f"Source must honor bounded ranges; received HTTP {response.status_code}")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or (int(match[1]), int(match[2])) != (offset, end):
                raise ValueError("Source returned a different byte range")
            if total_size is not None and int(match[3]) != total_size:
                raise ValueError("Source size changed during acquisition")
            if etag is not None and response.headers.get("ETag") != etag:
                raise ValueError("Source identity changed during acquisition")
            total_size, etag = int(match[3]), response.headers.get("ETag")
            return response.content

    content, count = csv_prefix(read_range)
    dated_table(content, 1750, 2018)
    ar6.write_bytes(content)
    # This fixed 2018 repository version contains historical AR5 data to 2011.
    response = session.get(AR5_URL, timeout=(20, 60))
    response.raise_for_status()
    dated_table(response.content, 1750, 2011)
    ar5.write_bytes(response.content)
    licenses = {}
    for name, url in {
        "AR6_LICENSE.txt": f"https://raw.githubusercontent.com/IPCC-WG1/Chapter-7/{AR6_COMMIT}/LICENSE",
        "AR5_FAIR_LICENSE.txt": f"https://raw.githubusercontent.com/OMS-NetZero/FAIR/{AR5_COMMIT}/LICENSE",
    }.items():
        r = session.get(url, timeout=(20, 60))
        r.raise_for_status()
        (cache / name).write_bytes(r.content)
        licenses[name] = {"url": url, "sha256": sha(cache / name)}
    record = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "ar6": {"url": AR6_URL, "commit": AR6_COMMIT, "data_commit_utc": "2021-08-12T16:34:15Z",
                "assessment_publication_year": 2021, "start_year": 1750, "end_year": 2018,
                "byte_range": [0, len(content)-1], "range_requests": count,
                "upstream_file_size": total_size, "etag": etag, "sha256": sha(ar6),
                "note": "Only prefix through complete 2018 line requested; 2019 row not downloaded"},
        "ar5": {"url": AR5_URL, "commit": AR5_COMMIT, "data_commit_utc": "2018-01-08T17:11:44Z",
                "assessment_publication_year": 2013, "start_year": 1750, "end_year": 2011,
                "sha256": sha(ar5), "note": "Separate author-curated AR5 Annex II reference, not spliced"},
        "licenses": licenses,
    }
    write_json(meta, record)
    return ar6, ar5, record


def normalize_ar6(frame):
    result = frame[["year", *COMPONENTS]].rename(columns=COMPONENTS).copy()
    result["wmghg_erf_wm2"] = frame[["co2", "ch4", "n2o", "other_wmghg"]].sum(axis=1)
    result["tropospheric_aerosol_erf_wm2"] = frame["aerosol-radiation_interactions"] + frame["aerosol-cloud_interactions"]
    result["other_anthropogenic_erf_wm2"] = frame[["o3", "h2o_stratospheric", "contrails", "bc_on_snow", "land_use"]].sum(axis=1)
    checks = [
        (result.tropospheric_aerosol_erf_wm2, frame.aerosol),
        (result.wmghg_erf_wm2 + result.tropospheric_aerosol_erf_wm2 + result.other_anthropogenic_erf_wm2, result.anthropogenic_total_erf_wm2),
        (result.solar_erf_wm2 + result.volcanic_erf_wm2, result.natural_total_erf_wm2),
        (result.anthropogenic_total_erf_wm2 + result.natural_total_erf_wm2, result.total_erf_wm2),
    ]
    if not all(np.allclose(a, b, rtol=0, atol=1e-9) for a, b in checks):
        raise ValueError("Published forcing components do not close to their totals")
    return result


def cut(frame, origin):
    if not 1750 <= origin <= min(2018, effective_cutoff_year()):
        raise DataRuleViolation("Forcing origin exceeds the active information cutoff")
    result = frame.loc[frame.year <= origin].copy()
    assert_no_post_cutoff(result, "year", f"forcing origin {origin}")
    if result.empty or result.year.max() != origin:
        raise DataRuleViolation("Forcing history does not reach its declared origin")
    return result


def declaration(origin, design=False):
    relative = "forcing/design-to-1982/annual.csv" if design else f"forcing/origin-{origin}/annual.csv"
    return {"path": f"data/processed/releases/{RELEASE_ID}/{relative}", "kind": "weather_input",
            "information_end_year": origin, "origins": [origin], "time_column": "year", "time_format": "year"}


def load_forcing_for_origin(release: Path, origin):
    if origin not in (*ORIGINS, 1982) or origin > effective_cutoff_year():
        raise DataRuleViolation("Unsupported or post-fold forcing origin")
    name = "forcing/design-to-1982/annual.csv" if origin == 1982 else f"forcing/origin-{origin}/annual.csv"
    manifest = json.loads((Path(release) / "release.json").read_text(encoding="utf-8"))
    path = Path(release) / name
    if name not in manifest["files"] or sha(path) != manifest["files"][name]["sha256"]:
        raise ValueError("Forcing release asset is missing or changed")
    result = dated_table(path.read_bytes(), 1750, origin)
    return result


def build(output: Path, cache: Path, upstream: Path, offline=False):
    ar6, ar5, acquisition = acquire(cache, offline)
    source = normalize_ar6(dated_table(ar6.read_bytes(), 1750, 2018))
    previous = json.loads((upstream / "release.json").read_text(encoding="utf-8"))
    if previous["id"] != "enhance-v2":
        raise ValueError("Cumulative release must preserve enhance-v2")
    output.mkdir(parents=True, exist_ok=True)
    for name, entry in previous["files"].items():
        if sha(upstream / name) != entry["sha256"]:
            raise ValueError(f"Upstream asset changed: {name}")
        target_name = f"upstream/enhance-v2-{name}" if name in ("schema.json", "provenance.json", "quality.json") else name
        target = output / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream / name, target)
    write_json(output / "upstream/enhance-v2-release.json", previous)
    for origin in (*ORIGINS, 1982):
        name = "forcing/design-to-1982/annual.csv" if origin == 1982 else f"forcing/origin-{origin}/annual.csv"
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cut(source, origin).to_csv(path, index=False, float_format="%.17g")
    # An alternate pre-2019 source for the design screen; no cross-source splice.
    ar5_design = cut(dated_table(ar5.read_bytes(), 1750, 2011), 1982)
    alt = output / "forcing/design-to-1982/ar5-reference.csv"
    ar5_design.to_csv(alt, index=False, float_format="%.17g")
    sources = output / "forcing/sources"
    sources.mkdir(parents=True, exist_ok=True)
    for name in (ar6.name, ar5.name, "acquisition.json", "AR6_LICENSE.txt", "AR5_FAIR_LICENSE.txt"):
        shutil.copyfile(cache / name, sources / name)
    write_json(output / "assets/forcing-evaluation.json", {"schema_version": 1, "files": [declaration(o) for o in ORIGINS]})
    write_json(output / "assets/forcing-design-to-1982.json", {"schema_version": 1, "files": [declaration(1982, design=True)]})
    write_json(output / "assets/forcing-ar5-design-to-1982.json", {"schema_version": 1, "files": [
        {**declaration(1982, design=True), "path": f"data/processed/releases/{RELEASE_ID}/forcing/design-to-1982/ar5-reference.csv"}]})
    write_json(output / "schema.json", {
        "release_id": RELEASE_ID, "upstream": "enhance-v2", "table": "forcing/origin-O/annual.csv",
        "key": "year", "units": "W m-2", "source_fields": COMPONENTS,
        "derived_fields": {"wmghg_erf_wm2": "CO2 + CH4 + N2O + other well-mixed GHGs",
                           "tropospheric_aerosol_erf_wm2": "aerosol-radiation + aerosol-cloud interactions",
                           "other_anthropogenic_erf_wm2": "ozone + stratospheric water + contrails + BC on snow + land use"},
        "reference": "Preserve IPCC source conventions: anthropogenic relative to 1750; natural components retain published background offsets. Do not assume every component is zero in 1750.",
        "transformations": "Column naming and component sums only; no interpolation, smoothing, normalization, future extension or station-temperature fitting",
        "gate_eligible": False,
        "eligibility_reason": "Retrospective 2021 forcing assessment; aerosol/volcanic reconstructions not certified origin-vintage-clean. AR5 is also retrospective for early folds.",
        "design_scope": "design-to-1982 is a full design resource, not permission to expose 1982 inputs inside earlier-origin forecast workers",
        "not_a_claim": "Forcing is a proposed covariate; supplied diagnostic does not prove causal attribution or predictive gain",
    })
    provenance = {"schema_version": 1, "release_id": RELEASE_ID, "acquisition": acquisition,
                  "upstream_release_sha256": sha(upstream / "release.json"), "gate_eligible": False,
                  "files": {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*"))
                            if p.is_file() and p.relative_to(output).as_posix() not in ("provenance.json", "quality.json")}}
    write_json(output / "provenance.json", provenance)
    result = validate(output)
    write_json(output / "quality.json", result)


def validate(folder):
    folder = Path(folder)
    max_years, rows = {}, {}
    for origin in (*ORIGINS, 1982):
        name = "forcing/design-to-1982/annual.csv" if origin == 1982 else f"forcing/origin-{origin}/annual.csv"
        frame = dated_table((folder / name).read_bytes(), 1750, origin)
        if not {"wmghg_erf_wm2", "tropospheric_aerosol_erf_wm2", "volcanic_erf_wm2", "solar_erf_wm2", "total_erf_wm2"} <= set(frame):
            raise ValueError("Missing requested forcing components")
        if not np.allclose(frame.anthropogenic_total_erf_wm2 + frame.natural_total_erf_wm2, frame.total_erf_wm2, atol=1e-9, rtol=0):
            raise ValueError("Released totals do not close")
        rows[str(origin)], max_years[str(origin)] = len(frame), int(frame.year.max())
    provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    for name, digest in provenance["files"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"Forcing release hash mismatch: {name}")
    result = {"passed": True, "forcing_rows": rows, "maximum_year_by_cut": max_years,
              "post_2018_rows": 0, "primary_source_commit_date": "2021-08-12",
              "ar5_reference_source_commit_date": "2018-01-08", "gate_eligible": False,
              "models_or_targets_read": False, "future_forcing_supplied": False}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/enhancement/forcing")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/enhancement/forcing-v3")
    parser.add_argument("--upstream", type=Path, default=ROOT / "data/processed/releases/enhance-v2")
    parser.add_argument("--acquire-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate:
        validate(args.validate)
    elif args.acquire_only:
        acquire(args.cache, args.offline)
        print("Validated forcing sources: AR6 1750-2018; AR5 reference 1750-2011")
    else:
        build(args.output, args.cache, args.upstream, args.offline)


if __name__ == "__main__":
    main()
