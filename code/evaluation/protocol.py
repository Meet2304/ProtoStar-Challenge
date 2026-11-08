"""Declared phases and committed, content-pinned assessment freezes.

This is a trusted-team workflow and an audit trail, not an access-control or
cryptographic attestation service. A filesystem owner can still open labels.
"""
import hashlib
import json
import re
import subprocess
import uuid
import importlib.metadata
import platform
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT
from evaluation.assets import fingerprint_assets, verify_asset_binding

PROTOCOL_VERSION = "audited-runtime-v4"
SELECTION_RULE = "equal_fold_mean_final_unclipped_skill"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def phase_origins(manifest, phase="development"):
    config = manifest["config"]
    if phase not in ("development", "assessment"):
        raise ValueError("phase must be development or assessment")
    if config["schema_version"] < 3:
        raise ValueError("legacy suites are retired for new runs; prepare a schema-3 suite with declared phases")
    origins = config[f"{phase}_origins"]
    if not origins:
        raise ValueError(f"suite has no {phase} origins")
    return list(origins)


def history_status(manifest):
    return manifest["config"].get("history_status", "reused_historical")


def source_files(root=ROOT):
    root = Path(root).resolve()
    files = set()
    for folder in ("code", "src"):
        files.update(p for p in (root / folder).rglob("*.py") if p.is_file())
    for name in ("uv.lock", "pyproject.toml", "requirements.txt", "code/requirements.txt"):
        if (root / name).is_file():
            files.add(root / name)
    return sorted(files)


def source_pins(root=ROOT):
    """Normalize Git text endings, but keep raw run artifacts hashed separately."""
    root = Path(root).resolve()
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in source_files(root)}


def installed_environment():
    """Installed versions, including transitive packages; no paths or secrets."""
    packages = sorted({(re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower(), d.version)
                       for d in importlib.metadata.distributions() if d.metadata.get("Name")})
    inventory = {"python": platform.python_version(), "implementation": platform.python_implementation(),
                 "packages": [{"name": name, "version": version} for name, version in packages]}
    return {"inventory": inventory, "sha256": canonical_hash(inventory)}


def committed_pins(root=ROOT):
    root = Path(root).resolve()
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    if result.returncode:
        raise ValueError("assessment freeze requires a Git repository with a committed model")
    commit = result.stdout.strip()
    pins = source_pins(root)
    if not pins:
        raise ValueError("no model/evaluator source files to freeze")
    for name, expected in pins.items():
        content = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=root, capture_output=True)
        if content.returncode or hashlib.sha256(content.stdout.replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError(f"assessment freeze requires committed source matching the workspace: {name}")
    # A tracked dependency removed from the workspace is also a source change.
    tracked = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit], cwd=root,
                             capture_output=True, text=True, check=True).stdout.splitlines()
    for name in tracked:
        relevant = ((name.startswith("code/") or name.startswith("src/")) and name.endswith(".py")) or name in (
            "uv.lock", "pyproject.toml", "requirements.txt", "code/requirements.txt")
        if relevant and name not in pins:
            raise ValueError(f"committed source was removed from workspace: {name}")
    return commit, pins


def adapter_identity(adapter, root=ROOT):
    module, function = adapter.rsplit(":", 1)
    path = Path(module)
    if path.is_file():
        path = path.resolve()
        if path.is_relative_to(Path(root).resolve()):
            return {"file": path.relative_to(Path(root).resolve()).as_posix(), "function": function}
        return {"external_file": str(path), "function": function}
    if not module or not function:
        raise ValueError("adapter must be module:function or path.py:function")
    return {"module": module, "function": function}


def adapter_assets(origins, entry, assets=None, root=ROOT, commit=None):
    """Built-ins consume provided history only; custom models attest dependencies."""
    known = ((entry.get("module"), entry.get("function")) in {
        ("evaluation.nested", "forecast"), ("evaluation.adapters", "combo"),
        ("evaluation.adapters", "reference")})
    if assets is None:
        if not known:
            raise ValueError("custom adapters require --assets with an explicit dependency manifest; "
                             "use evaluation/no-assets.json only when the model uses provided train/targets alone")
        return {"mode": "builtin_provided_history_only", "schema_version": 1}
    path = Path(assets)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(Path(root).resolve())
        except ValueError as exc:
            raise ValueError("asset manifest must be inside the project") from exc
    binding = fingerprint_assets(root, path.as_posix(), origins, commit=commit)
    return {"mode": "declared", "binding": binding}


def validate_external_provenance(provenance, origins):
    if not isinstance(provenance, dict):
        raise ValueError("external predictions require a JSON provenance object")
    required = {"author", "model_repository", "model_commit", "parameters", "training_cutoffs", "dependencies"}
    if not required <= set(provenance):
        raise ValueError(f"external provenance is missing {sorted(required - set(provenance))}")
    if not isinstance(provenance["author"], str) or not provenance["author"].strip():
        raise ValueError("external provenance requires an author")
    if not isinstance(provenance["model_repository"], str) or not provenance["model_repository"].strip():
        raise ValueError("external provenance requires a model repository URL or location")
    if not isinstance(provenance["model_commit"], str) or not re.fullmatch(r"[0-9a-fA-F]{40}", provenance["model_commit"]):
        raise ValueError("external model_commit must be a full 40-character Git SHA")
    if not isinstance(provenance["parameters"], dict) or not isinstance(provenance["dependencies"], dict):
        raise ValueError("external parameters and dependencies must be JSON objects")
    cutoffs = provenance["training_cutoffs"]
    if not isinstance(cutoffs, dict):
        raise ValueError("training_cutoffs must map each origin to its training-end year")
    for origin in origins:
        cutoff = cutoffs.get(str(origin))
        if type(cutoff) is not int or not 1 <= cutoff <= min(origin, 2018):
            raise ValueError(f"external predictions for origin {origin} need a training cutoff <= {origin}")
    canonical_hash(provenance)  # rejects NaN and non-JSON values
    return provenance


def model_spec(origins, adapter=None, params=None, baseline=None, prediction_dir=None,
               provenance=None, root=ROOT, assets=None, asset_commit=None):
    from evaluation.core import digest
    params = {} if params is None else params
    if not isinstance(params, dict):
        raise ValueError("model parameters must be a JSON object")
    canonical_hash(params)
    if sum(v is not None for v in (adapter, baseline, prediction_dir)) != 1:
        raise ValueError("choose exactly one adapter, baseline or prediction directory")
    if adapter:
        entry = adapter_identity(adapter, root)
        return {"kind": "adapter", "entry_point": entry, "params": params,
                "assets": adapter_assets(origins, entry, assets, root, asset_commit)}
    if assets is not None:
        raise ValueError("local asset declarations apply only to Python adapters")
    if params:
        raise ValueError("adapter parameters apply only to adapters; use provenance for external model parameters")
    if baseline:
        if baseline not in ("average", "trend", "last_year"):
            raise ValueError("unknown baseline")
        return {"kind": "baseline", "method": baseline}
    provenance = validate_external_provenance(provenance, origins)
    files = {f"origin-{origin}.csv": digest(Path(prediction_dir) / f"origin-{origin}.csv") for origin in origins}
    return {"kind": "external_csv", "provenance": provenance, "original_prediction_sha256": files,
            "verification": "Author declarations; external training commit and cutoff are not independently certified."}


def create_freeze(suite, output, adapter=None, params=None, baseline=None,
                  prediction_dir=None, provenance=None, note="", root=ROOT, assets=None, from_run=None):
    from evaluation.core import verify_suite
    manifest = verify_suite(suite)
    if from_run is not None:
        from evaluation.core import verify_run
        previous = verify_run(from_run)
        if previous.get("phase") != "development" or previous.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("--from-run requires a completed current-protocol development run")
        if previous["suite_id"] != manifest["suite_id"] or previous["source_pins"] != source_pins(root):
            raise ValueError("development run suite/source differs from the current workspace")
        if any(v is not None for v in (adapter, baseline, prediction_dir, assets)) or params:
            raise ValueError("--from-run supplies the model settings; do not also override them")
        arguments = arguments_from_spec(previous["model_spec"], root)
        adapter, params, baseline, assets = (arguments[k] for k in ("adapter", "params", "baseline", "assets"))
        old_assets = previous["model_spec"].get("assets", {})
        if old_assets.get("mode") == "declared":
            verify_asset_binding(root, old_assets["binding"])
    origins = phase_origins(manifest, "assessment")
    commit, pins = committed_pins(root)
    spec = model_spec(origins, adapter, params, baseline, prediction_dir, provenance, root,
                      assets=assets, asset_commit=commit)
    if spec["kind"] == "adapter":
        entry = spec["entry_point"]
        if "external_file" in entry:
            raise ValueError("assessment adapters must be committed within the project")
        if "file" in entry and entry["file"] not in pins:
            declared = spec["assets"].get("binding", {}).get("files", {})
            if entry["file"] not in declared or declared[entry["file"]]["kind"] != "source":
                raise ValueError("assessment adapter must be committed source or a declared source asset")
        if "module" in entry:
            module_path = entry["module"].replace(".", "/")
            candidates = [f"{prefix}{module_path}{suffix}" for prefix in ("code/", "src/", "")
                          for suffix in (".py", "/__init__.py")]
            if not any(p in pins for p in candidates):
                raise ValueError("assessment adapter module must resolve to committed project source")
    record = {"schema_version": 1, "protocol_version": PROTOCOL_VERSION,
              "created_utc": datetime.now(timezone.utc).isoformat(), "suite_id": manifest["suite_id"],
              "assessment_origins": origins, "model_spec": spec, "seed": manifest["config"]["seed"],
              "git_commit": commit, "source_pins": pins, "selection_rule": SELECTION_RULE,
              "installed_environment": installed_environment(),
              "history_status": history_status(manifest), "note": note,
              "interpretation": "A prospective model specification freeze; this does not make previously used labels untouched."}
    if from_run is not None:
        record["development_run"] = {"run_id": previous["run_id"], "run_fingerprint": previous["run_fingerprint"]}
    record["freeze_id"] = canonical_hash(record)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: existing records are immutable.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def arguments_from_spec(spec, root=ROOT):
    """Re-use recorded settings instead of asking teammates to type them twice."""
    result = {"adapter": None, "params": {}, "baseline": None, "assets": None}
    if spec["kind"] == "baseline":
        result["baseline"] = spec["method"]
    elif spec["kind"] == "adapter":
        entry = spec["entry_point"]
        if "external_file" in entry:
            raise ValueError("move the adapter into committed project source before freezing")
        result["adapter"] = (str(Path(root) / entry["file"]) if "file" in entry else entry["module"]) + ":" + entry["function"]
        result["params"] = spec["params"]
        if spec["assets"].get("mode") == "declared":
            result["assets"] = Path(root) / spec["assets"]["binding"]["manifest"]["path"]
    else:
        raise ValueError("external CSV freezes require explicit prediction files and provenance")
    return result


def verify_freeze(path, manifest, spec, root=ROOT):
    from evaluation.core import read_json
    record = read_json(path)
    claimed = record.pop("freeze_id", None)
    if claimed != canonical_hash(record):
        raise ValueError("assessment freeze fingerprint does not match")
    record["freeze_id"] = claimed
    expected = {"protocol_version": PROTOCOL_VERSION, "suite_id": manifest["suite_id"],
                "assessment_origins": phase_origins(manifest, "assessment"), "model_spec": spec,
                "seed": manifest["config"]["seed"], "selection_rule": SELECTION_RULE}
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError("freeze does not match this suite, adapter, parameters or original predictions")
    commit, pins = committed_pins(root)
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", record["git_commit"], commit],
                              cwd=root, capture_output=True)
    if ancestry.returncode or pins != record["source_pins"]:
        raise ValueError("source or Git ancestry changed after freeze; create a new declared freeze")
    if record.get("installed_environment") != installed_environment():
        raise ValueError("installed Python/package environment changed after freeze")
    asset_info = spec.get("assets", {})
    if asset_info.get("mode") == "declared":
        verify_asset_binding(root, asset_info["binding"], commit=record["git_commit"])
    return record


def log_assessment_access(suite, record, run_id):
    """Append an immutable event before any assessment scoring is exposed."""
    folder = Path(suite) / "assessment_access"
    folder.mkdir(exist_ok=True)
    event = {"created_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
             "protocol_version": PROTOCOL_VERSION,
             "freeze_id": record["freeze_id"], "suite_id": record["suite_id"],
             "origins": record["assessment_origins"], "history_status": record["history_status"]}
    path = folder / f"{run_id}-{uuid.uuid4().hex[:8]}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(event, stream, indent=2)
        stream.write("\n")
    return path
