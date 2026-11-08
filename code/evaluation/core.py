"""Build immutable historical suites, validate predictions and record experiments."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import (ROOT, METRICS, TRAIN_REPO, TRAIN_REVISION,
                    assert_no_post_cutoff, thirty_year_mean, wide)
from evaluation.adapters import reference
from evaluation.official_score import KEY, score
from evaluation.assets import snapshot_assets, verify_asset_binding
from evaluation.protocol import (PROTOCOL_VERSION, SELECTION_RULE, phase_origins,
                                 history_status, model_spec, source_pins,
                                 verify_freeze, log_assessment_access, canonical_hash)
from evaluation.protocol import installed_environment

PRED_COLUMNS = KEY + ["predicted_value"]
SCORER = Path(__file__).with_name("official_score.py")
DEFAULT_CONFIG = ROOT / "evaluation/config.json"
PINNED_ANNUAL_SHA256 = "0b5f5fa9b55ab59602dcd9884b97c74c7bec7e89ea9d23f4219703046030e411"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rubric_score(ratings):
    rubric = read_json(ROOT / "evaluation/rubric.json")
    expected = {item["key"] for item in rubric["criteria"]}
    if not isinstance(ratings, dict) or set(ratings) != expected:
        raise ValueError(f"ratings require exactly these keys: {sorted(expected)}")
    lo, hi = rubric["scale"]
    if any(type(v) is not int or not lo <= v <= hi for v in ratings.values()):
        raise ValueError("each rating must be a human-supplied integer from 1 to 5")
    return sum(item["weight"] * (ratings[item["key"]] - lo) / (hi - lo)
               for item in rubric["criteria"])


def git_state():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")}


def validate_config(config):
    fixed = {"horizon": 7, "live_leads": [1, 2, 3],
             "final_leads": [4, 5, 6, 7], "baseline_window": 30,
             "min_valid_years": 10, "max_missing_days": 10}
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("config must retain the official seven-year, 30-year, 10-day/10-year contract")
    origins = config.get("origins", [])
    if not origins or any(type(y) is not int or y + 7 > 2018 or y < 1880 for y in origins):
        raise ValueError("origins must be integer years >=1880 whose seven-year holdouts end by 2018")
    if len(set(origins)) != len(origins):
        raise ValueError("duplicate forecast origins")
    if config.get("schema_version") not in (1, 2, 3):
        raise ValueError("unknown evaluation config version")
    if config["schema_version"] >= 2:
        development, assessment = config.get("development_origins"), config.get("assessment_origins")
        if not isinstance(development, list) or not isinstance(assessment, list) or not development:
            raise ValueError("phase configs need development_origins and assessment_origins lists")
        if any(type(y) is not int for y in development + assessment):
            raise ValueError("phase origins must be integers")
        if len(development + assessment) != len(set(development + assessment)) or set(development + assessment) != set(origins):
            raise ValueError("development and assessment must partition the declared origins")
        if any(b - a < 7 for a, b in zip(sorted(origins), sorted(origins)[1:])):
            raise ValueError("phase config target windows must not overlap")
        if assessment and max(development) + 7 > min(assessment):
            raise ValueError("development targets must end before assessment forecasting begins")
        if config.get("history_status") != "reused_historical":
            raise ValueError("historical suites must declare history_status=reused_historical; a freeze cannot certify untouched labels")
    seed = config.get("seed")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")


def load_annual(path):
    """Accept only tables with the official annual_indices schema and missing counts."""
    path = Path(path)
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype={"ID": str})
    required = {"ID", "Year"}
    for column, missing in METRICS.values():
        required.update([column, *missing])
    if not required <= set(raw):
        raise ValueError(f"annual input missing columns: {sorted(required - set(raw))}")
    year = pd.to_numeric(raw.Year, errors="raise")
    if year.isna().any() or not np.isfinite(year).all() or (year % 1 != 0).any():
        raise ValueError("annual years must be finite integers")
    raw["Year"] = year.astype(int)
    assert_no_post_cutoff(raw, "Year", str(path))
    if raw.ID.isna().any() or raw.duplicated(["ID", "Year"]).any():
        raise ValueError("annual input has empty city IDs or duplicate city-years")
    out = raw[["ID", "Year"]].rename(columns={"ID": "city_id", "Year": "year"}).copy()
    out["city_id"] = out.city_id.astype(str)
    if "Name" in raw:
        out["name"] = raw.Name
    for metric, (column, missing) in METRICS.items():
        counts = raw[missing].apply(pd.to_numeric, errors="raise")
        if (counts < 0).any(axis=None):
            raise ValueError("negative missing-day count")
        # Keep scoring validity identical to the documented per-variable rule.
        # A stricter Days Both CDD training filter must not redefine the score mask.
        valid = counts.notna().all(axis=1) & counts.le(10).all(axis=1)
        values = pd.to_numeric(raw[column], errors="raise")
        if np.isinf(values[valid]).any():
            raise ValueError(f"infinite annual value in {column}")
        out[metric] = values.where(valid)
    return out.sort_values(["city_id", "year"]).reset_index(drop=True)


def pinned_annual():
    """Use the revision-verified local cache; otherwise fetch the pinned training subset."""
    filename = "annual_indices/train-00000-of-00001.parquet"
    cached = ROOT / "data/raw" / filename
    metadata = ROOT / "data/raw/.cache/huggingface/download" / (filename + ".metadata")
    if cached.is_file() and metadata.is_file():
        lines = metadata.read_text(encoding="utf-8").splitlines()
        # Parquet LFS ETag is its SHA256, so check content as well as revision.
        if len(lines) >= 2 and lines[0] == TRAIN_REVISION and lines[1] == digest(cached):
            return cached
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(TRAIN_REPO, filename, repo_type="dataset", revision=TRAIN_REVISION))


def build_fold(annual, cities, origin):
    assert_no_post_cutoff(annual, "year", "fold source")
    if not isinstance(origin, int) or origin + 7 > 2018:
        raise ValueError("fold holdout must end by 2018")
    years = list(range(origin + 1, origin + 8))
    train = annual[(annual.year <= origin) & annual.city_id.isin(cities)].copy()
    if train.empty:
        raise ValueError(f"no training history for origin {origin}")
    assert_no_post_cutoff(train, "year", "fold history")
    targets = pd.MultiIndex.from_product([cities, years, list(METRICS)], names=KEY).to_frame(index=False)
    truth = annual.melt(id_vars=["city_id", "year"], value_vars=list(METRICS),
                        var_name="metric", value_name="actual_value")
    answer = targets.merge(truth, on=KEY, how="left", validate="one_to_one")
    answer["split"] = np.where(answer.year <= origin + 3, "live", "final")
    levels = []
    for metric in METRICS:
        level = thirty_year_mean(wide(train, metric), origin).rename("baseline_value").reset_index()
        level["metric"] = metric
        levels.append(level)
    answer = answer.merge(pd.concat(levels), on=["city_id", "metric"], how="left", validate="many_to_one")
    answer["scored"] = answer.actual_value.notna() & answer.baseline_value.notna()
    answer["exclusion_reason"] = np.select(
        [answer.baseline_value.isna(), answer.actual_value.isna()],
        ["fewer_than_10_valid_baseline_years", "missing_or_invalid_target"], default="")
    rows = []
    for split in ("live", "final"):
        row = {"split": split}
        for metric in METRICS:
            group = answer[(answer.split == split) & (answer.metric == metric) & answer.scored]
            value = float((group.actual_value - group.baseline_value).abs().mean())
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"origin {origin}, {split}, {metric}: reference MAE is empty or zero")
            row[metric] = value
        rows.append(row)
    return train, targets, answer, pd.DataFrame(rows)


def prepare(suite, config_path=DEFAULT_CONFIG, annual_path=None, cities_path=None):
    from evaluation.suite_identity import (canonical_hash, normalized_digest,
                                           preparation_source_pins, seal_suite_manifest)

    suite = Path(suite).resolve()
    if suite.exists():
        raise FileExistsError(f"suite already exists: {suite}; choose a new path to preserve past comparisons")
    config = read_json(config_path)
    validate_config(config)
    if config["schema_version"] != 3:
        raise ValueError("new suites require schema_version 3; schema-1/2 suites are readable archives only")
    source = Path(annual_path) if annual_path else pinned_annual()
    annual = load_annual(source)
    cities_path = Path(cities_path or ROOT / "data/city_info.csv")
    city_table = pd.read_csv(cities_path, dtype={"city_id": str})
    if ("city_id" not in city_table or city_table.city_id.isna().any()
            or city_table.city_id.str.strip().eq("").any()):
        raise ValueError("cities file requires nonempty city_id")
    cities = sorted(city_table.city_id.unique().tolist())
    if not cities or set(cities) - set(annual.city_id):
        raise ValueError("city list is empty or contains cities absent from annual data")
    from evaluation.phase_policy import enforce_phase_policy
    enforce_phase_policy({"config": config, "city_ids": cities,
                          "source": {"repo": TRAIN_REPO if digest(source) == PINNED_ANNUAL_SHA256 else "custom_local"}})
    suite.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".protostar-build-", dir=suite.parent) as temp:
        dest = Path(temp) / "suite"
        dest.mkdir()
        for origin in config["origins"]:
            train, targets, answer, ref = build_fold(annual, cities, origin)
            fold = dest / f"origin-{origin}"
            (fold / "private").mkdir(parents=True)
            (fold / "baselines").mkdir()
            train.to_csv(fold / "train.csv", index=False, lineterminator="\n")
            targets.to_csv(fold / "targets.csv", index=False, lineterminator="\n")
            answer.to_csv(fold / "private/answer_key.csv", index=False, lineterminator="\n")
            ref.to_csv(fold / "private/reference_mae.csv", index=False, lineterminator="\n")
            for method in ("average", "trend", "last_year"):
                reference(train, targets, origin, config["seed"], method).to_csv(
                    fold / f"baselines/{method}.csv", index=False, lineterminator="\n")
        files = {p.relative_to(dest).as_posix(): digest(p) for p in sorted(dest.rglob("*.csv"))}
        source_hash = digest(source)
        official_source = source_hash == PINNED_ANNUAL_SHA256
        if not annual_path and not official_source:
            raise ValueError("downloaded annual file does not match the pinned dataset content hash")
        manifest = {"schema_version": config["schema_version"], "config": config, "city_ids": cities,
                    "source": {"repo": TRAIN_REPO if official_source else "custom_local",
                               "revision": TRAIN_REVISION if official_source else None,
                               "annual_sha256": source_hash, "cities_sha256": digest(cities_path),
                               "annual_normalized_sha256": normalized_digest(source),
                               "cities_semantic_sha256": canonical_hash(cities)},
                    "scorer_sha256": normalized_digest(SCORER),
                    "preparation_source_pins": preparation_source_pins(), "files": files}
        manifest = seal_suite_manifest(manifest)
        (dest / "manifest.json").write_bytes(
            (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        dest.rename(suite)
    return manifest


def verify_suite(suite):
    from evaluation.suite_identity import normalized_digest, validate_suite_manifest

    suite = Path(suite).resolve()
    manifest = read_json(suite / "manifest.json")
    validate_suite_manifest(manifest)
    validate_config(manifest["config"])
    if manifest["schema_version"] >= 3:
        from evaluation.phase_policy import enforce_phase_policy
        enforce_phase_policy(manifest)
    scorer_hash = normalized_digest(SCORER) if manifest["schema_version"] >= 3 else digest(SCORER)
    if manifest["scorer_sha256"] != scorer_hash:
        raise ValueError("scorer changed; rebuild a versioned suite")
    for name, expected in manifest["files"].items():
        path = (suite / name).resolve()
        if not path.is_relative_to(suite) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"suite file changed or missing: {name}")
    return manifest


def validate_predictions(predictions, targets):
    if list(predictions.columns) != PRED_COLUMNS:
        raise ValueError(f"prediction columns must be exactly {PRED_COLUMNS}")
    pred = predictions.copy()
    for column in ("year", "predicted_value"):
        if pred[column].map(lambda v: isinstance(v, (bool, np.bool_))).any():
            raise ValueError(f"booleans are not valid {column} values")
    if pred[KEY].isna().any(axis=None):
        raise ValueError("prediction keys contain nulls")
    pred["city_id"] = pred.city_id.astype(str)
    pred["metric"] = pred.metric.astype(str)
    years = pd.to_numeric(pred.year, errors="raise")
    if not np.isfinite(years).all() or (years % 1 != 0).any():
        raise ValueError("prediction years must be finite integers")
    pred["year"] = years.astype(int)
    pred["predicted_value"] = pd.to_numeric(pred.predicted_value, errors="raise")
    if not np.isfinite(pred.predicted_value).all():
        raise ValueError("all predictions, including unscored rows, must be finite")
    if pred.duplicated(KEY).any():
        raise ValueError("duplicate prediction keys")
    actual = pd.MultiIndex.from_frame(pred[KEY])
    expected = pd.MultiIndex.from_frame(targets[KEY])
    if len(actual.difference(expected)) or len(expected.difference(actual)):
        raise ValueError("prediction grid differs from target grid (missing or unexpected city/year/metric)")
    return pred.sort_values(KEY).reset_index(drop=True)


def score_fold(predictions, fold, origin, output):
    targets = pd.read_csv(fold / "targets.csv", dtype={"city_id": str})
    pred = validate_predictions(predictions, targets)
    truth = pd.read_csv(fold / "private/answer_key.csv", dtype={"city_id": str})
    ref = pd.read_csv(fold / "private/reference_mae.csv")
    # Never coerce CSV strings with astype(bool): 'False' would become True.
    if truth.scored.dtype != bool:
        raise ValueError("truth scored must parse as booleans")
    assert_no_post_cutoff(truth, "year", "evaluation labels")
    results = []
    for split in ("live", "final"):
        result = score(pred, truth, ref, split)
        result.update({"origin": origin, "split": split,
                       "mean_skill_unclipped": sum(result["skill"].values()) / len(METRICS)})
        results.append(result)
    pred.to_csv(output / "predictions.csv", index=False)
    write_json(output / "scores.json", results)
    rows = truth.merge(pred, on=KEY, validate="one_to_one")
    rows["origin"] = origin
    rows["lead"] = rows.year - origin
    rows["absolute_error"] = (rows.predicted_value - rows.actual_value).abs()
    rows["baseline_absolute_error"] = (rows.baseline_value - rows.actual_value).abs()
    rows["gain_vs_baseline"] = rows.baseline_absolute_error - rows.absolute_error
    rows.to_csv(output / "row_errors.csv", index=False)
    diagnostics = []
    for dimension in ("year", "lead", "city_id"):
        table = rows[rows.scored].groupby(["metric", dimension]).agg(
            n_rows=("absolute_error", "size"), mae=("absolute_error", "mean"),
            baseline_mae=("baseline_absolute_error", "mean")).reset_index()
        table["skill"] = 1 - table.mae / table.baseline_mae.replace(0, np.nan)
        table["dimension"] = dimension
        table = table.rename(columns={dimension: "group"})
        diagnostics.append(table)
    pd.concat(diagnostics).to_csv(output / "diagnostics.csv", index=False)
    return results


def run_suite(suite, runs, model, adapter=None, params=None, prediction_dir=None, baseline=None, timeout=600,
              phase="development", freeze=None, provenance=None, assets=None):
    suite, runs = Path(suite).resolve(), Path(runs).resolve()
    manifest = verify_suite(suite)
    if sum(x is not None for x in (adapter, prediction_dir, baseline)) != 1:
        raise ValueError("choose one adapter, prediction directory, or baseline")
    if not model.strip():
        raise ValueError("model name must be nonempty")
    params = {} if params is None else params
    if not isinstance(params, dict):
        raise ValueError("adapter parameters must be a JSON object")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number of seconds")
    if adapter == "evaluation.adapters:combo" and params:
        raise ValueError("fixed combo takes no tunable parameters; use evaluation.nested:forecast")
    origins = phase_origins(manifest, phase)
    spec = model_spec(origins, adapter, params, baseline, prediction_dir, provenance, assets=assets)
    asset_binding = spec.get("assets", {}).get("binding")
    freeze_record = None
    if phase == "assessment":
        if freeze is None:
            raise ValueError("assessment requires --freeze with a committed, matching model specification")
        freeze_record = verify_freeze(freeze, manifest, spec)
    elif freeze is not None:
        raise ValueError("freeze records are used only for assessment runs")
    initial_source_pins = source_pins()
    if freeze_record is not None and initial_source_pins != freeze_record["source_pins"]:
        raise ValueError("source changed while verifying the committed freeze")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = runs / (stamp + "-" + uuid.uuid4().hex[:10])
    run.mkdir(parents=True)
    write_json(run / "params.json", params)
    for name in ("uv.lock", "pyproject.toml", "requirements.txt", "code/requirements.txt"):
        source = ROOT / name
        if source.is_file():
            dest = run / "environment" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())
    # Store the source tree, including uncommitted changes, so a dirty commit is
    # reviewable. This copies Python only, never data or credentials.
    for folder in ("code", "src"):
        for source in (ROOT / folder).rglob("*.py"):
            dest = run / "code_snapshot" / folder / source.relative_to(ROOT / folder)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())
    adapter_hash = None
    if adapter:
        module_name, _ = adapter.rsplit(":", 1)
        if Path(module_name).is_file():
            source = Path(module_name).resolve()
            adapter_hash = digest(source)
            (run / "adapter_snapshot.py").write_bytes(source.read_bytes())
            adapter = str(source) + ":" + adapter.rsplit(":", 1)[1]
    metadata = {"schema_version": 2, "run_id": run.name, "created_utc": stamp,
                "protocol_version": PROTOCOL_VERSION, "phase": phase, "origins": origins,
                "history_status": history_status(manifest), "selection_rule": SELECTION_RULE,
                "model_spec": spec, "source_pins": initial_source_pins,
                "status": "running", "model": model, "suite_id": manifest["suite_id"],
                "suite_path": str(suite), "config": manifest["config"],
                "adapter": adapter, "adapter_sha256": adapter_hash, "baseline": baseline,
                "prediction_dir": str(Path(prediction_dir).resolve()) if prediction_dir else None,
                "params": params, "git": git_state(),
                "python": sys.version, "platform": platform.platform(),
                "versions": {"numpy": np.__version__, "pandas": pd.__version__},
                "installed_environment": installed_environment(),
                "scorer_sha256": digest(SCORER), "artifact_sha256": {}}
    from evaluation.controls import CONTROL_SET
    metadata["control_set"] = CONTROL_SET
    try:
        from importlib.metadata import version
        metadata["versions"].update({name: version(name) for name in ("pyarrow", "scikit-learn")})
    except Exception:
        pass
    write_json(run / "run.json", metadata)
    write_json(run / "suite_manifest.json", manifest)
    if asset_binding is not None:
        write_json(run / "asset_binding.json", asset_binding)
    start = time.monotonic()
    results = []
    try:
        if freeze_record is not None:
            write_json(run / "freeze.json", freeze_record)
            from evaluation.exposure import cohort_from_manifest
            metadata["assessment_cohort"] = cohort_from_manifest(manifest, origins)
            metadata["assessment_access_record"] = str(log_assessment_access(suite, freeze_record, run.name))
            write_json(run / "run.json", metadata)
        for origin in origins:
            fold = suite / f"origin-{origin}"
            output = run / f"origin-{origin}"
            output.mkdir()
            if baseline:
                path = fold / f"baselines/{baseline}.csv"
            elif prediction_dir:
                original = Path(prediction_dir).resolve() / f"origin-{origin}.csv"
                path = output / "original_external_predictions.csv"
                path.write_bytes(original.read_bytes())
                if digest(path) != spec["original_prediction_sha256"][original.name]:
                    raise ValueError("external predictions changed after provenance/freeze validation")
            else:
                path = output / "adapter_output.csv"
                env = {**os.environ, "PYTHONHASHSEED": str(manifest["config"]["seed"]),
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "PROTOSTAR_FOLD_CUTOFF": str(origin), "PROTOSTAR_ARTIFACT_DIR": str(output)}
                for key in ("PROTOSTAR_ASSET_ROOT", "PROTOSTAR_ASSET_MANIFEST", "PROTOSTAR_ASSET_BINDING"):
                    env.pop(key, None)
                asset_snapshot = None
                if asset_binding is not None:
                    asset_snapshot = snapshot_assets(ROOT, asset_binding, output, origin=origin,
                        commit=freeze_record["git_commit"] if freeze_record else None)
                # Paths passed to the model no longer identify the evaluator's
                # fold/labels directory. Relative full-data reads fail as well.
                with tempfile.TemporaryDirectory(prefix="protostar-model-") as temp:
                    work = Path(temp)
                    for source, name in ((fold / "train.csv", "train.csv"),
                                         (fold / "targets.csv", "targets.csv"),
                                         (run / "params.json", "params.json")):
                        (work / name).write_bytes(source.read_bytes())
                    command = [sys.executable, str(Path(__file__).with_name("worker.py")),
                               "--adapter", adapter, "--train", str(work / "train.csv"),
                               "--targets", str(work / "targets.csv"), "--output", str(path),
                               "--origin", str(origin), "--seed", str(manifest["config"]["seed"]),
                               "--params", str(work / "params.json")]
                    context = {"schema_version": 1, "project_root": str(ROOT.resolve()),
                               "suite_root": str(suite), "own_output": str(output),
                               "source_pins": initial_source_pins, "adapter_file": None,
                               "declared_sources": []}
                    module_path = Path(adapter.rsplit(":", 1)[0])
                    if module_path.is_file():
                        context["adapter_file"] = {"path": str(module_path.resolve()),
                            "sha256": hashlib.sha256(module_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
                    if asset_snapshot is not None:
                        for name, info in asset_snapshot["binding"]["files"].items():
                            if info["kind"] == "source":
                                original = ROOT / name
                                context["declared_sources"].append({"original_path": str(original.resolve()),
                                    "snapshot_path": str((Path(asset_snapshot["snapshot_root"]) / name).resolve()),
                                    "sha256": hashlib.sha256(original.read_bytes().replace(b"\r\n", b"\n")).hexdigest()})
                    write_json(work / "runtime_context.json", context)
                    command += ["--runtime-context", str(work / "runtime_context.json")]
                    if asset_snapshot is not None:
                        command += ["--asset-root", asset_snapshot["snapshot_root"],
                                    "--asset-binding", asset_snapshot["binding_path"]]
                    from evaluation.audit_transport import AuditPipe
                    with AuditPipe() as audit_pipe, (output / "adapter.log").open("w", encoding="utf-8") as log:
                        command += audit_pipe.child_args
                        process = subprocess.Popen(command, cwd=work, env=env, stdout=log,
                                                   stderr=subprocess.STDOUT, **audit_pipe.popen_kwargs)
                        audit_pipe.after_spawn()
                        timed_out = False
                        try:
                            process.wait(timeout=timeout)
                        except subprocess.TimeoutExpired:
                            timed_out = True
                            process.kill()
                            process.wait()
                        receipt = audit_pipe.finish(timeout=5)
                    if output.is_symlink() or not output.resolve().is_relative_to(run):
                        raise RuntimeError("adapter replaced its output directory")
                    # Child-writable files are never the authoritative audit. The
                    # monotonic pipe stream has already reached the parent.
                    for audit_name in ("io_audit.json", "imported_source.json", "runtime_receipt.json"):
                        audit_path = output / audit_name
                        if audit_path.is_symlink() or audit_path.is_file():
                            audit_path.unlink()
                        elif audit_path.exists():
                            raise RuntimeError("adapter planted a non-file audit destination")
                    for audit_name, observed in receipt["reports"].items():
                        if audit_name in ("io_audit.json", "imported_source.json"):
                            write_json(output / audit_name, observed)
                    write_json(output / "runtime_receipt.json", receipt)
                    if timed_out:
                        raise RuntimeError(f"adapter timed out after {timeout}s for origin {origin}; see {output / 'adapter.log'}")
                    if not receipt["verified"]:
                        raise RuntimeError(f"adapter did not finish a verified parent audit for origin {origin}; see {output / 'runtime_receipt.json'}")
                    from evaluation.input_review import review_input_summary
                    if phase == "assessment" and review_input_summary(receipt["summary"])["requires_review"]:
                        raise RuntimeError("assessment adapter used undeclared external inputs; declare them before a new freeze")
                if process.returncode:
                    raise RuntimeError(f"adapter failed for origin {origin}; see {output / 'adapter.log'}")
                for audit_name in ("io_audit.json", "imported_source.json"):
                    audit_path = output / audit_name
                    if not audit_path.is_file() or read_json(audit_path).get("status") != "verified":
                        raise RuntimeError(f"adapter did not finish a verified {audit_name} for origin {origin}")
                if asset_snapshot is not None:
                    verify_asset_binding(asset_snapshot["snapshot_root"], asset_snapshot["binding"])
                    verify_asset_binding(ROOT, asset_snapshot["binding"],
                                         commit=freeze_record["git_commit"] if freeze_record else None)
            pred = pd.read_csv(path, dtype={"city_id": str, "metric": str})
            results.extend(score_fold(pred, fold, origin, output))
        write_json(run / "scores.json", results)
        if source_pins() != metadata["source_pins"]:
            raise RuntimeError("model/evaluator source changed during execution; rerun from a stable source snapshot")
        if asset_binding is not None:
            verify_asset_binding(ROOT, asset_binding, commit=freeze_record["git_commit"] if freeze_record else None)
        summary = {}
        for split in ("live", "final"):
            subset = [r for r in results if r["split"] == split]
            raw_mean = float(np.mean([r["mean_skill_unclipped"] for r in subset]))
            summary[split] = {"mean_score": float(np.mean([r["score"] for r in subset])),
                              "min_score": min(r["score"] for r in subset),
                              "max_score": max(r["score"] for r in subset),
                              "mean_skill_unclipped": raw_mean,
                              "selection_score": 100 * max(0., raw_mean),
                              "n_folds": len(subset)}
        metadata.update(status="complete", summary=summary)
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - start
        if adapter:
            metadata["runtime_audits"] = {}
            metadata["runtime_input_summary"] = {}
            for origin in origins:
                audit_summary = {}
                for file in ("io_audit.json", "imported_source.json"):
                    artifact = run / f"origin-{origin}" / file
                    if artifact.is_file():
                        observed = read_json(artifact)
                        audit_summary[file] = {k: observed[k] for k in ("status", "denied_count", "violations") if k in observed}
                    else:
                        audit_summary[file] = {"status": "not_recorded"}
                metadata["runtime_audits"][str(origin)] = audit_summary
                receipt_path = run / f"origin-{origin}" / "runtime_receipt.json"
                if receipt_path.is_file():
                    from evaluation.input_review import review_input_summary
                    metadata["runtime_input_summary"][str(origin)] = review_input_summary(read_json(receipt_path)["summary"])
        metadata["artifact_sha256"] = {p.relative_to(run).as_posix(): digest(p)
                                         for p in sorted(run.rglob("*")) if p.is_file() and p.name != "run.json"}
        metadata["run_fingerprint"] = canonical_hash(metadata)
        write_json(run / "run.json", metadata)
    return run, metadata


def verify_run(path):
    """Detect accidental changes to saved scores, inputs, logs and metadata."""
    path = Path(path).resolve()
    metadata = read_json(path / "run.json")
    if metadata.get("status") != "complete":
        raise ValueError(f"run is not complete: {path.name}")
    if metadata.get("schema_version", 1) >= 2:
        claimed = metadata.pop("run_fingerprint", None)
        if claimed != canonical_hash(metadata):
            raise ValueError(f"run metadata fingerprint changed: {path.name}")
        metadata["run_fingerprint"] = claimed
    if metadata.get("protocol_version") in ("audited-inputs-v5", "parent-audited-v6") and metadata.get("adapter"):
        for origin in metadata["origins"]:
            for audit_name in ("io_audit.json", "imported_source.json"):
                audit_path = path / f"origin-{origin}" / audit_name
                if not audit_path.is_file() or read_json(audit_path).get("status") != "verified":
                    raise ValueError(f"run has no verified worker audit: {origin}/{audit_name}")
            if metadata["protocol_version"] == "parent-audited-v6":
                receipt_path = path / f"origin-{origin}" / "runtime_receipt.json"
                if not receipt_path.is_file() or not read_json(receipt_path).get("verified"):
                    raise ValueError("run has no verified parent audit receipt")
    hashes = metadata.get("artifact_sha256", {})
    if "scores.json" not in hashes:
        raise ValueError(f"run has no score artifact fingerprint: {path.name}")
    for name, expected in hashes.items():
        artifact = (path / name).resolve()
        if not artifact.is_relative_to(path) or not artifact.is_file() or digest(artifact) != expected:
            raise ValueError(f"run artifact changed or missing: {path.name}/{name}")
    snapshot = path / "suite_manifest.json"
    if snapshot.exists():
        manifest = read_json(snapshot)
        from evaluation.suite_identity import validate_suite_manifest
        validate_suite_manifest(manifest)
        if manifest["suite_id"] != metadata["suite_id"]:
            raise ValueError(f"run's suite manifest does not match: {path.name}")
    elif metadata.get("schema_version", 1) >= 2:
        raise ValueError(f"run has no suite manifest: {path.name}")
    return metadata


def leaderboard(runs, suite_id=None):
    rows = []
    for path in sorted(Path(runs).glob("*/run.json")):
        meta = read_json(path)
        if meta.get("status") != "complete" or (suite_id and meta["suite_id"] != suite_id):
            continue
        meta = verify_run(path.parent)
        phase = meta.get("phase", "development")
        protocol = meta.get("protocol_version", "legacy-v1-unprotected")
        group = f"{meta['suite_id']}:{phase}:{protocol}"
        row = {"run_id": meta["run_id"], "model": meta["model"], "suite_id": meta["suite_id"],
               "comparison_group": group, "phase": phase, "protocol_version": protocol,
               "history_status": meta.get("history_status", "reused_historical")}
        for split, values in meta["summary"].items():
            row.update({f"{split}_{k}": v for k, v in values.items()})
            row[f"{split}_selection_score"] = 100 * max(0., values["mean_skill_unclipped"])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    # Rounding establishes an explicit 1e-10 skill tolerance for numerical ties.
    result["_ranking_skill"] = result.final_mean_skill_unclipped.round(10)
    result = result.sort_values(["comparison_group", "_ranking_skill", "run_id"], ascending=[True, False, True])
    result.insert(0, "internal_rank", result.groupby("comparison_group")["_ranking_skill"].rank(
        method="min", ascending=False).astype(int))
    return result.drop(columns="_ranking_skill")


def compare_runs(first, second, samples=2000, seed=42):
    """Paired raw-skill evidence by whole period; no four-year pseudo-CI.

    A descriptive period bootstrap keeps every lead/city in each selected period
    together. It still cannot certify generalization from few reused periods.
    """
    first, second = Path(first), Path(second)
    metas = [verify_run(path) for path in (first, second)]
    if any(m["status"] != "complete" for m in metas) or metas[0]["suite_id"] != metas[1]["suite_id"]:
        raise ValueError("comparison requires complete runs of exactly the same suite")
    if samples < 100:
        raise ValueError("use at least 100 bootstrap samples")
    for field, fallback in (("phase", "development"), ("protocol_version", "legacy-v1-unprotected")):
        if metas[0].get(field, fallback) != metas[1].get(field, fallback):
            raise ValueError("comparison requires the same phase and evaluation protocol")
    origins = [meta.get("origins", meta["config"]["origins"]) for meta in metas]
    if origins[0] != origins[1]:
        raise ValueError("comparison requires identical evaluated origins")
    evidence = []
    for origin in origins[0]:
        a, b = [pd.read_csv(path / f"origin-{origin}/row_errors.csv") for path in (first, second)]
        joined = a.merge(b[KEY + ["absolute_error", "scored", "actual_value", "baseline_absolute_error"]],
                         on=KEY, suffixes=("_a", "_b"), validate="one_to_one")
        for name in ("scored", "actual_value", "baseline_absolute_error"):
            left, right = joined[f"{name}_a"], joined[f"{name}_b"]
            equal = (np.isclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True)
                     if name == "baseline_absolute_error" else ((left == right) | (left.isna() & right.isna())))
            if not equal.all():
                raise ValueError("comparison inputs disagree on truth or baseline eligibility")
        joined = joined.rename(columns={"scored_a": "scored", "baseline_absolute_error_a": "baseline_absolute_error"})
        joined = joined[joined.scored & (joined.split == "final")]
        errors = joined.groupby("metric")[["absolute_error_a", "absolute_error_b", "baseline_absolute_error"]].mean().reindex(METRICS)
        if not np.isfinite(errors.to_numpy()).all() or (errors.baseline_absolute_error <= 0).any():
            raise ValueError("comparison has empty or zero-reference metrics")
        raw_a = 1 - errors.absolute_error_a / errors.baseline_absolute_error
        raw_b = 1 - errors.absolute_error_b / errors.baseline_absolute_error
        by_lead = []
        for lead, group in joined.groupby("lead"):
            e = group.groupby("metric")[["absolute_error_a", "absolute_error_b", "baseline_absolute_error"]].mean().reindex(METRICS)
            if np.isfinite(e.to_numpy()).all() and (e.baseline_absolute_error > 0).all():
                delta = float(((e.absolute_error_a - e.absolute_error_b) / e.baseline_absolute_error).mean())
            else:
                delta = None
            by_lead.append({"lead": int(lead), "raw_skill_delta": delta})
        evidence.append({"origin": origin, "first_raw_skill": float(raw_a.mean()),
                         "second_raw_skill": float(raw_b.mean()), "raw_skill_delta": float((raw_b - raw_a).mean()),
                         "metric_skill_deltas": {m: float(raw_b[m] - raw_a[m]) for m in METRICS},
                         "lead_skill_deltas": by_lead})
    deltas = np.array([row["raw_skill_delta"] for row in evidence])
    overlapping = any(b - a < 7 for a, b in zip(sorted(origins[0]), sorted(origins[0])[1:]))
    interval = None
    if len(deltas) >= 3 and not overlapping:
        rng = np.random.default_rng(seed)
        values = deltas[rng.integers(0, len(deltas), (samples, len(deltas)))].mean(axis=1) * 100
        interval = np.quantile(values, [.025, .975]).tolist()
    leave_one_out = [float(np.delete(deltas, i).mean() * 100) for i in range(len(deltas))] if len(deltas) > 1 else []
    return {"first_run": first.name, "second_run": second.name, "suite_id": metas[0]["suite_id"],
            "phase": metas[0].get("phase", "development"), "periods": evidence,
            "mean_final_raw_skill_delta_second_minus_first": float(deltas.mean()),
            "mean_raw_skill_delta_points": float(deltas.mean() * 100),
            "positive_periods": int((deltas > 1e-10).sum()), "negative_periods": int((deltas < -1e-10).sum()),
            "tied_periods": int((np.abs(deltas) <= 1e-10).sum()),
            "leave_one_period_out_delta_points": leave_one_out,
            "period_bootstrap_95_percent_sensitivity_interval_points": interval,
            "samples": samples if interval is not None else 0, "seed": seed,
            "interpretation": "Raw, unclipped skill; positive favors the second run. Whole-period resampling "
                              "keeps all forecast leads/cities together. The optional interval is descriptive "
                              "sensitivity conditional on treating observed periods as representative; not a "
                              "formal confidence interval or prediction of future rank. No interval with fewer "
                              "than three periods or overlapping windows. Reused labels, nested training sets, "
                              "serial dependence and climate shifts still limit inference."}
