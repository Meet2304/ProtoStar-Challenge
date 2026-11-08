"""Explicit model dependencies bound to historical origins and committed bytes.

Standalone integration candidate; imports evaluation.core nowhere. The public
API is fingerprint_assets, project_asset_binding, verify_asset_binding, and
snapshot_assets. All take an injected project root, except the pure projection.

Manifest v1: {"schema_version": 1, "files": [{"path": "models/weights.bin",
"kind": "model_weights", "information_end_year": 1980, "origins": [1983]}]}.
Kinds are model_weights, configuration, static_context, weather_input, source.
Only learned weights/weather need information_end_year. Weather additionally
requires time_column and time_format: year (exact YYYY), iso_date (YYYY-MM-DD),
or iso_datetime (YYYY-MM-DDTHH:MM:SS[.ffffff][Z|+HH:MM|-HH:MM]). Typed parquet
date/datetime columns are also accepted for the matching format. Naive and
offset timestamps use their stated calendar year, not timezone conversion.

Origins omitted means every requested evaluation origin. Explicit scopes can
include other origins: inactive assets are not stat'ed, opened, hashed, checked
against Git or copied. Nevertheless, all declaration metadata and exact full
manifest bytes remain bound. Workers should receive a projected one-origin
binding and snapshot rather than the entire phase's assets.

Configuration/static_context are AUTHOR DECLARATIONS of weather independence.
Cutoffs on weights are AUTHOR DECLARATIONS, not verification of training data.
Declared inputs and hashes protect against accidental drift. Worker Python IO
tripwires catch common undeclared protected-path reads, but deliberate native
IO/subprocess bypasses remain outside this trust boundary; this is no OS sandbox.

Runtime bindings preserve exact bytes. Committed text manifests/config/source
and weather CSV may differ only by CRLF-to-LF conversion when both are UTF-8
without NUL. Learned weights, binary extensions and LFS always match exact raw
bytes. Materialized LFS data matches the pointer's SHA256 oid and size; no fetch
or network is used.
"""
from __future__ import annotations

from datetime import date, datetime
from contextlib import ExitStack
import csv
import gzip
import codecs
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile

KINDS = {"model_weights", "configuration", "static_context", "weather_input", "source"}
CUTOFF_KINDS = {"model_weights", "weather_input"}
TIME_FORMATS = {"year", "iso_date", "iso_datetime"}
CHUNK_SIZE = 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"
_SECRET_COMPONENT = re.compile(
    r"(^|[._-])(secrets?|credentials?|passwords?|tokens?|api[_-]?keys?|private[_-]?keys?|access[_-]?keys?|service[_-]?accounts?)([._-]|$)",
    re.IGNORECASE,
)
_FORBIDDEN_SUFFIXES = {".pem", ".key", ".token", ".p8", ".p12", ".pfx", ".ppk", ".der", ".jks", ".keystore", ".kdbx"}
_FORBIDDEN_NAMES = {".netrc", "_netrc", ".npmrc", ".pypirc", ".git-credentials",
                    "authorized_keys", "known_hosts", "id_rsa", "id_ed25519", "id_ecdsa",
                    "auth.json", "auth.yaml", "oauth.json"}
_FORBIDDEN_DIRECTORIES = {".git", ".ssh", ".aws", ".azure", ".docker"}
_RESERVED_NAMES = {"con", "prn", "aux", "nul", *{f"com{i}" for i in range(1, 10)},
                   *{f"lpt{i}" for i in range(1, 10)}}
_TEXT_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".txt", ".csv", ".tsv",
                    ".py", ".r", ".jl", ".sh", ".ps1", ".xml", ".html", ".css", ".js", ".ts", ".md"}
# SHA256 of the pinned <=2018 annual_indices parquet at TRAIN_REVISION.
# A renamed copy still needs weather-input timestamp validation.
PINNED_ANNUAL_SHA256 = "0b5f5fa9b55ab59602dcd9884b97c74c7bec7e89ea9d23f4219703046030e411"


def _validate_nonweather_classification(path, declaration, fingerprint):
    """Catch obvious mistaken declarations, without asserting training history.

    A year on a census/station table is legitimate. Temporal columns alone are
    therefore insufficient: require a recognizable weather/outcome column too.
    This is a conservative schema check, not semantic verification of all data.
    """
    if declaration["kind"] not in {"configuration", "static_context", "source"}:
        return
    name = declaration["path"].casefold()
    if (name.startswith("data/raw/") or name == "data/annual_valid.csv"
            or fingerprint["sha256"] == PINNED_ANNUAL_SHA256):
        raise AssetError(f"{declaration['path']}: known weather/label data requires weather_input, not {declaration['kind']}")
    suffix = path.suffix.casefold()
    compressed = name.endswith((".csv.gz", ".tsv.gz"))
    if suffix not in {".csv", ".tsv", ".parquet", ".feather", ".arrow"} and not compressed:
        return
    try:
        if suffix in {".csv", ".tsv"} or compressed:
            opener = gzip.open if compressed else open
            with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
                columns = next(csv.reader(stream, delimiter="\t" if name.endswith((".tsv", ".tsv.gz")) else ","), [])
        elif suffix == ".parquet":
            import pyarrow.parquet as pq
            with pq.ParquetFile(path) as parquet:
                columns = parquet.schema_arrow.names
        else:
            import pyarrow as pa
            with pa.memory_map(str(path), "r") as source:
                columns = pa.ipc.open_file(source).schema.names
    except Exception as exc:
        raise AssetError(f"{declaration['path']}: cannot inspect declared table schema") from exc
    normalized = {re.sub(r"[^a-z0-9]", "", str(column).casefold()) for column in columns}
    temporal = {"year", "date", "datetime", "timestamp", "time", "observedat", "observationdate", "observationyear"}
    outcomes = {"tmax", "tmin", "prcp", "precip", "precipitation", "temperature", "avgtemp", "avgtmax",
                "avgtmin", "maxtmax", "mintmin", "totprcp", "cdd", "annualavgtemp", "totalprecipin",
                "hottestdaytemp", "actual", "actualvalue", "observedvalue", "targetvalue", "label",
                "streamflow", "airquality", "energydemand"}
    wide_years = any(re.fullmatch(r"[0-9]{4}", str(column)) and 1800 <= int(column) <= 2200 for column in columns)
    generic_outcome = {"metric", "value"} <= normalized
    if wide_years or (normalized & temporal and (normalized & outcomes or generic_outcome)):
        raise AssetError(f"{declaration['path']}: temporal weather/actual-value table requires weather_input, not {declaration['kind']}")


class AssetError(ValueError):
    """An asset declaration or its observed/committed contents is unusable."""


def _canonical_hash(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise AssetError("asset binding must contain finite JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


def _origins(values):
    if (not isinstance(values, (list, tuple)) or not values
            or any(type(year) is not int or not 1 <= year <= 2018 for year in values)
            or len(set(values)) != len(values)):
        raise AssetError("origins must be a nonempty unique list of integer years from 1 through 2018")
    return sorted(values)


def _relative_name(value):
    # Reject credential-looking paths before stat/open or even existence checks.
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/"):
        raise AssetError("asset paths must be portable project-relative POSIX paths")
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise AssetError("asset paths cannot contain traversal, empty or dot components")
    for part in parts:
        lowered = part.casefold()
        if (lowered.startswith(".env") or lowered in _FORBIDDEN_NAMES
                or lowered in _FORBIDDEN_DIRECTORIES or _SECRET_COMPONENT.search(lowered)
                or PurePosixPath(lowered).suffix in _FORBIDDEN_SUFFIXES):
            raise AssetError(f"credential-looking asset path is forbidden: {value}")
        if (any(ord(char) < 32 or ord(char) == 127 or char in '<>:"|?*' for char in part)
                or part.endswith((" ", ".")) or part.split(".", 1)[0].casefold() in _RESERVED_NAMES):
            raise AssetError(f"asset path is not portable: {value}")
    return value


def _root_path(root):
    try:
        root = Path(root).resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise AssetError("asset root must be an existing directory") from exc
    if not root.is_dir():
        raise AssetError("asset root must be an existing directory")
    return root


def _file_path(root, name):
    name = _relative_name(name)
    candidate = root.joinpath(*name.split("/"))
    current = root
    try:
        for part in name.split("/"):
            current = current / part
            if current.is_symlink():
                raise AssetError(f"symlink asset paths are not accepted: {name}")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root) or not stat.S_ISREG(resolved.stat().st_mode):
            raise AssetError(f"asset must be an ordinary contained file: {name}")
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise AssetError(f"asset file is missing or inaccessible: {name}") from exc
    return resolved


def _stream_fingerprint(path):
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            size += len(chunk)
            hasher.update(chunk)
    return {"sha256": hasher.hexdigest(), "size": size}


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AssetError(f"duplicate JSON key in asset manifest: {key}")
        result[key] = value
    return result


def _read_manifest(root, manifest_name):
    path = _file_path(root, manifest_name)
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise AssetError("asset manifest exceeds one MiB")
    raw = path.read_bytes()
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AssetError("asset manifest must be UTF-8 JSON") from exc
    if (not isinstance(document, dict) or set(document) != {"schema_version", "files"}
            or type(document["schema_version"]) is not int or document["schema_version"] != 1
            or not isinstance(document["files"], list)):
        raise AssetError("asset manifest requires exactly schema_version=1 and a files list")
    declarations = []
    names = {manifest_name.casefold()}
    for item in document["files"]:
        if not isinstance(item, dict):
            raise AssetError("each asset declaration must be an object")
        allowed = {"path", "kind", "information_end_year", "origins", "time_column", "time_format"}
        if set(item) - allowed or not {"path", "kind"} <= set(item):
            raise AssetError("asset declaration has unknown or missing fields")
        name = _relative_name(item["path"])
        key = name.casefold()
        if any(key == other or key.startswith(other + "/") or other.startswith(key + "/") for other in names):
            raise AssetError(f"duplicate or colliding asset/manifest declaration: {name}")
        names.add(key)
        kind = item["kind"]
        if not isinstance(kind, str) or kind not in KINDS:
            raise AssetError(f"unknown asset kind: {kind!r}")
        declaration = {"path": name, "kind": kind, "origins": _origins(item["origins"]) if "origins" in item else None}
        if kind in CUTOFF_KINDS:
            end = item.get("information_end_year")
            if type(end) is not int or not 1 <= end <= 2018:
                raise AssetError(f"{name}: learned/weather assets need information_end_year <=2018")
            declaration["information_end_year"] = end
        elif "information_end_year" in item:
            raise AssetError(f"{name}: information_end_year applies only to learned/weather assets")
        if kind == "weather_input":
            column, format_name = item.get("time_column"), item.get("time_format")
            if (not isinstance(column, str) or not column.strip()
                    or not isinstance(format_name, str) or format_name not in TIME_FORMATS):
                raise AssetError(f"{name}: weather input requires time_column and supported time_format")
            if PurePosixPath(name).suffix.casefold() not in {".csv", ".parquet"}:
                raise AssetError(f"{name}: weather inputs must be CSV or parquet")
            declaration.update(time_column=column, time_format=format_name)
        elif "time_column" in item or "time_format" in item:
            raise AssetError(f"{name}: time_column/time_format apply only to weather inputs")
        declarations.append(declaration)
    return ({"path": manifest_name, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)},
            sorted(declarations, key=lambda item: item["path"]))


def _weather_years(series, format_name, source_name):
    import pandas as pd
    if series.empty:
        return pd.Series([], dtype="int64")
    if series.isna().any():
        raise AssetError(f"{source_name}: weather time column contains nulls")
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        if format_name not in {"iso_date", "iso_datetime"}:
            raise AssetError(f"{source_name}: typed datetimes require a date/datetime time_format")
        if format_name == "iso_date" and (series.dt.normalize() != series).any():
            raise AssetError(f"{source_name}: iso_date cannot contain non-midnight datetimes")
        return series.dt.year.astype("int64")
    values = series.astype("string")
    try:
        if format_name == "year":
            if not values.str.fullmatch(r"[0-9]{4}").all():
                raise AssetError(f"{source_name}: year format requires exact four-digit YYYY values")
            years = values.astype("int64")
            if (years < 1).any():
                raise AssetError(f"{source_name}: year zero is not valid")
            return years
        if format_name == "iso_date":
            if not values.str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").all():
                raise AssetError(f"{source_name}: iso_date requires exact YYYY-MM-DD values")
            return values.map(lambda value: date.fromisoformat(value).year).astype("int64")
        pattern = r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?(?:Z|[+-][0-9]{2}:[0-9]{2})?"
        if not values.str.fullmatch(pattern).all():
            raise AssetError(f"{source_name}: unsupported iso_datetime representation")
        return values.map(lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")).year).astype("int64")
    except (ValueError, OverflowError) as exc:
        if isinstance(exc, AssetError):
            raise
        raise AssetError(f"{source_name}: invalid calendar time in weather input") from exc


def _validate_weather(path, declaration, applicable_origins):
    import pandas as pd
    # Import only the shared guard, not evaluation.core/protocol.
    from common import assert_no_post_cutoff
    column = declaration["time_column"]
    bound = min(min(applicable_origins), declaration["information_end_year"], 2018)
    total = 0
    minimum, maximum = None, None
    try:
        with ExitStack() as resources:
            if path.suffix.casefold() == ".csv":
                with path.open("r", encoding="utf-8-sig", newline="") as stream:
                    header = next(csv.reader(stream), [])
                if column not in header or len(header) != len(set(header)):
                    raise AssetError(f"{declaration['path']}: missing or duplicate CSV header columns")
                batches = resources.enter_context(pd.read_csv(
                    path, usecols=[column], dtype={column: "string"}, chunksize=65536))
            else:
                import pyarrow.parquet as pq
                parquet = pq.ParquetFile(path)
                resources.callback(parquet.close)
                if column not in parquet.schema_arrow.names:
                    raise AssetError(f"{declaration['path']}: missing weather time column {column!r}")
                batches = (batch.to_pandas() for batch in parquet.iter_batches(columns=[column], batch_size=65536))
            for batch in batches:
                years = _weather_years(batch[column], declaration["time_format"], declaration["path"])
                assert_no_post_cutoff(pd.DataFrame({"year": years}), "year", declaration["path"])
                if not years.empty:
                    low, high = int(years.min()), int(years.max())
                    if high > bound:
                        raise AssetError(f"{declaration['path']}: observed year {high} exceeds information/origin bound {bound}; no subsetting is allowed")
                    minimum = low if minimum is None else min(minimum, low)
                    maximum = high if maximum is None else max(maximum, high)
                    total += len(years)
    except AssetError:
        raise
    except Exception as exc:
        raise AssetError(f"{declaration['path']}: weather time validation failed: {exc}") from exc
    if not total:
        raise AssetError(f"{declaration['path']}: weather input must have at least one timestamp")
    return {"rows_checked": total, "min_year": minimum, "max_year": maximum,
            "calendar_year_basis": "stated date or timestamp calendar year; no timezone conversion"}


def _finish_binding(manifest, declarations, origins, files):
    per_origin = {}
    for origin in origins:
        pins = {name: dict(record, applicable_origins=[origin]) for name, record in files.items()
                if origin in record["applicable_origins"]}
        record = {"origin": origin, "manifest": manifest, "files": pins}
        record["fingerprint"] = _canonical_hash(record)
        per_origin[str(origin)] = record
    payload = {"schema_version": 1, "origins": origins, "manifest": manifest,
               "declarations": declarations, "files": files, "per_origin": per_origin}
    payload["binding_id"] = _canonical_hash(payload)
    return payload


def fingerprint_assets(root, manifest_path, origins, *, commit=None):
    """Read declarations and active files, returning an origin-specific binding.

    manifest_path is a project-relative POSIX string. All declared scopes remain
    in identity; only assets applicable to requested origins are accessed.
    commit optionally requires these exact assets/manifest to match a local Git
    commit. It does not alter the binding identity.
    """
    root = _root_path(root)
    manifest_path = _relative_name(manifest_path)
    origins = _origins(origins)
    manifest, declarations = _read_manifest(root, manifest_path)
    files = {}
    for declaration in declarations:
        applicable = [origin for origin in origins if declaration["origins"] is None or origin in declaration["origins"]]
        if not applicable:
            continue
        if declaration["kind"] in CUTOFF_KINDS and declaration["information_end_year"] > min(applicable):
            raise AssetError(f"{declaration['path']}: information_end_year exceeds an applicable origin")
        path = _file_path(root, declaration["path"])
        record = {**declaration, **_stream_fingerprint(path), "applicable_origins": applicable}
        _validate_nonweather_classification(path, declaration, record)
        if declaration["kind"] == "weather_input":
            record["weather_validation"] = _validate_weather(path, declaration, applicable)
        files[declaration["path"]] = record
    binding = _finish_binding(manifest, declarations, origins, files)
    if commit is not None:
        _verify_committed(root, binding, commit)
    return binding


def _check_binding(binding):
    if not isinstance(binding, dict) or "binding_id" not in binding:
        raise AssetError("invalid asset binding")
    payload = {key: value for key, value in binding.items() if key != "binding_id"}
    if binding["binding_id"] != _canonical_hash(payload):
        raise AssetError("asset binding identity is invalid")
    if set(binding) != {"schema_version", "origins", "manifest", "declarations", "files", "per_origin", "binding_id"}:
        raise AssetError("asset binding has unsupported fields")
    if binding["schema_version"] != 1:
        raise AssetError("unsupported asset binding schema")


def project_asset_binding(binding, origins):
    """Pure projection to a subset of previously verified origins, without I/O."""
    _check_binding(binding)
    origins = _origins(origins)
    if not set(origins) <= set(binding["origins"]):
        raise AssetError("projected origins were not part of the original binding")
    files = {}
    for name, record in binding["files"].items():
        applicable = [origin for origin in origins if origin in record["applicable_origins"]]
        if applicable:
            files[name] = dict(record, applicable_origins=applicable)
    # Own all nested objects rather than exposing mutable references to a freeze.
    return json.loads(json.dumps(_finish_binding(binding["manifest"], binding["declarations"], origins, files)))


def verify_asset_binding(root, binding, *, commit=None):
    """Fail if full declarations or any active file changed since binding."""
    _check_binding(binding)
    current = fingerprint_assets(root, binding["manifest"]["path"], binding["origins"], commit=commit)
    if current != binding:
        raise AssetError("declared asset contents or metadata changed after binding")
    return current


def _git(root, args):
    result = subprocess.run(["git", *args], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise AssetError(f"local Git asset verification failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def _git_blob(root, commit, name):
    listing = _git(root, ["ls-tree", "-z", "--full-tree", commit, "--", name])
    matches = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        header, observed_name = entry.split(b"\t", 1)
        if observed_name.decode("utf-8") == name:
            matches.append(header.split())
    if len(matches) != 1 or matches[0][0] not in {b"100644", b"100755"} or matches[0][1] != b"blob":
        raise AssetError(f"asset must be an ordinary file tracked at the frozen commit: {name}")
    return matches[0][2].decode("ascii")


def _parse_lfs_pointer(raw):
    if not raw.startswith(b"version https://git-lfs.github.com/spec/"):
        return None
    # Standard v1 pointer form only; unsupported extensions fail closed.
    match = re.fullmatch(rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (0|[1-9][0-9]*)\n?", raw)
    if not match:
        raise AssetError("malformed or unsupported Git LFS pointer")
    return {"sha256": match[1].decode(), "size": int(match[2])}


def _committed_fingerprint(root, oid):
    length = int(_git(root, ["cat-file", "-s", oid]).strip())
    if length <= 1024:
        raw = _git(root, ["cat-file", "blob", oid])
        pointer = _parse_lfs_pointer(raw)
        return ({**pointer, "storage": "lfs"} if pointer else
                {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw), "storage": "blob"})
    hasher, size, prefix = hashlib.sha256(), 0, b""
    with subprocess.Popen(["git", "cat-file", "blob", oid], cwd=root,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        for chunk in iter(lambda: process.stdout.read(CHUNK_SIZE), b""):
            if not prefix:
                prefix = chunk[:100]
            hasher.update(chunk)
            size += len(chunk)
        error = process.stderr.read()
        code = process.wait()
    if code or size != length:
        raise AssetError(f"cannot read committed asset bytes: {error.decode(errors='replace').strip()}")
    if prefix.startswith(b"version https://git-lfs.github.com/spec/"):
        raise AssetError("malformed or unsupported oversized Git LFS pointer")
    return {"sha256": hasher.hexdigest(), "size": size, "storage": "blob"}


def _normalized_text_fingerprint(stream):
    """Streaming CRLF normalization, only for validated UTF-8 without NUL."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    hasher, size, pending, valid = hashlib.sha256(), 0, b"", True
    for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
        if b"\0" in chunk:
            valid = False
        if valid:
            try:
                decoder.decode(chunk, final=False)
            except UnicodeDecodeError:
                valid = False
        data = pending + chunk
        pending = b"\r" if data.endswith(b"\r") else b""
        if pending:
            data = data[:-1]
        data = data.replace(b"\r\n", b"\n")
        hasher.update(data)
        size += len(data)
    if valid:
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            valid = False
    hasher.update(pending)
    size += len(pending)
    return {"sha256": hasher.hexdigest(), "size": size} if valid else None


def _committed_text_fingerprint(root, oid):
    with subprocess.Popen(["git", "cat-file", "blob", oid], cwd=root,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        result = _normalized_text_fingerprint(process.stdout)
        error, code = process.stderr.read(), process.wait()
    if code:
        raise AssetError(f"cannot read committed text asset: {error.decode(errors='replace').strip()}")
    return result


def _text_normalization_allowed(name, record, manifest_path):
    if name == manifest_path:
        return True
    kind, suffix = record.get("kind"), PurePosixPath(name).suffix.casefold()
    if kind == "weather_input":
        return suffix == ".csv"
    return kind in {"configuration", "static_context", "source"} and suffix in _TEXT_EXTENSIONS


def _verify_committed(root, binding, commit):
    if not isinstance(commit, str) or not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", commit):
        raise AssetError("asset commit must be a full Git commit SHA")
    actual = _git(root, ["rev-parse", "--verify", f"{commit}^{{commit}}"]).decode().strip()
    if actual.casefold() != commit.casefold():
        raise AssetError("asset commit must identify a commit object directly")
    records = {binding["manifest"]["path"]: binding["manifest"], **binding["files"]}
    for name, record in records.items():
        oid = _git_blob(root, commit, name)
        expected = _committed_fingerprint(root, oid)
        if {key: expected[key] for key in ("sha256", "size")} == {key: record[key] for key in ("sha256", "size")}:
            continue
        if expected["storage"] == "blob" and _text_normalization_allowed(name, record, binding["manifest"]["path"]):
            with _file_path(root, name).open("rb") as stream:
                observed_text = _normalized_text_fingerprint(stream)
            committed_text = _committed_text_fingerprint(root, oid)
            if observed_text is not None and observed_text == committed_text:
                continue
        raise AssetError(f"asset/manifest differs from frozen Git blob or LFS oid/size: {name}")


def snapshot_assets(root, binding, run_dir, *, origin=None, commit=None):
    """Streaming-copy a verified binding to a fresh declared_assets directory.

    Returns snapshot_root (the files/ directory), project-relative manifest_path,
    binding_path and binding. The manifest is preserved at its original relative
    path inside snapshot_root, so workers can call verify_asset_binding there.
    With origin provided, only that origin's files are verified/copied; inactive
    paths may be absent and are never accessed.
    """
    root = _root_path(root)
    selected = project_asset_binding(binding, [origin]) if origin is not None else binding
    verify_asset_binding(root, selected, commit=commit)
    run_dir = Path(run_dir).resolve()
    destination = run_dir / "declared_assets"
    if destination.exists() or destination.is_symlink():
        raise AssetError("declared_assets snapshot already exists; snapshots are immutable")
    source_names = [selected["manifest"]["path"], *selected["files"]]
    sources = {name: _file_path(root, name) for name in source_names}
    if any(path.is_relative_to(destination) or destination.is_relative_to(path) for path in sources.values()):
        raise AssetError("asset snapshot overlaps a source file")
    run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".declared-assets-", dir=run_dir) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        file_root = staging / "files"
        for name, source in sources.items():
            target = file_root.joinpath(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=CHUNK_SIZE)
        verify_asset_binding(file_root, selected)
        (staging / "binding.json").write_text(json.dumps(selected, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staging.rename(destination)
    return {"snapshot_root": str(destination / "files"),
            "manifest_path": selected["manifest"]["path"],
            "binding_path": str(destination / "binding.json"), "binding": selected}
