"""Mirror verified evaluation runs into MLflow.

Run directories under outputs/evaluation/runs stay the source of truth. This module copies a
complete, fingerprint-verified run (plus its scorecard) into MLflow so the team can browse and
chart progress over time. It is idempotent: a run already mirrored (same run fingerprint) is
skipped. MLflow is imported lazily so the evaluator works without it.

Tracking store: $MLFLOW_TRACKING_URI if set (for example a shared team server), else a local
SQLite file at <repo>/outputs/mlflow/mlflow.db with artifacts in <repo>/outputs/mlflow/artifacts.
Browse:  uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import ROOT
from evaluation.controls import CONTROL_SET
from evaluation.core import read_json, verify_run
from evaluation.scorecard import SCORECARD_VERSION, build_scorecard, flat_metrics

LOCAL_STORE = ROOT / "outputs/mlflow"
FINGERPRINT_TAG = "harness.run_fingerprint"
COMPLETE_TAG = "harness.sync_complete"


def tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    LOCAL_STORE.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + (LOCAL_STORE / "mlflow.db").resolve().as_posix()


def role_of(meta: dict) -> str:
    spec = meta.get("model_spec") or {}
    entry = spec.get("entry_point") or {}
    module = entry.get("module")
    if module == "evaluation.controls":
        return "control"
    if (module, entry.get("function")) in {("evaluation.adapters", "combo"), ("evaluation.nested", "forecast")}:
        return "incumbent"
    if spec.get("kind") == "baseline":
        return "control"
    return "candidate"


def track_of(meta: dict) -> str:
    params = meta.get("params") or {}
    return str(params.get("input_track", "official"))


def experiment_name(meta: dict) -> str:
    return (f"protostar/{meta['suite_id']}/{meta.get('phase', 'development')}/"
            f"{meta.get('protocol_version', 'legacy')}/{track_of(meta)}")


def _millis(stamp: str) -> int:
    return int(datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _client():
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def _experiment(client, name: str) -> str:
    found = client.get_experiment_by_name(name)
    if found is not None:
        return found.experiment_id
    artifact_location = None
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        artifact_location = (LOCAL_STORE / "artifacts").resolve().as_uri()
    return client.create_experiment(name, artifact_location=artifact_location)


def already_logged(client, experiment_id: str, fingerprint: str) -> bool:
    """True only for a COMPLETE mirror. Partial mirrors (a crash mid-sync) are deleted for a redo."""
    hits = client.search_runs([experiment_id], filter_string=f"tags.`{FINGERPRINT_TAG}` = '{fingerprint}'",
                              max_results=50)
    complete = False
    for hit in hits:
        if hit.data.tags.get(COMPLETE_TAG) == "true" and not complete:
            complete = True
        else:
            client.delete_run(hit.info.run_id)
    return complete


def _lock_path() -> Path | None:
    """Lock beside a local SQLite store; None for a remote server (single writer by convention:
    one teammate runs scripts/mlflow_sync.py against a shared server at a time)."""
    uri = tracking_uri()
    if uri.startswith("sqlite:///"):
        db = Path(uri[len("sqlite:///"):])
        return db.with_name(db.name + ".sync.lock")
    if "://" not in uri or uri.startswith("file:"):
        return LOCAL_STORE / ".sync.lock"
    return None


@contextmanager
def sync_lock(stale_seconds: float = 900):
    """Serialize mirrors into a local store (concurrent syncs could duplicate runs)."""
    lock = _lock_path()
    if lock is None:
        yield
        return
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists() and time.time() - lock.stat().st_mtime > stale_seconds:
        lock.unlink(missing_ok=True)
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another MLflow sync is running (lock {lock}); retry when it finishes") from exc
    try:
        os.write(handle, str(os.getpid()).encode())
        yield
    finally:
        os.close(handle)
        lock.unlink(missing_ok=True)


def log_run(run_dir, client=None) -> str | None:
    """Mirror one verified run. Returns the MLflow run id, or None if it was already mirrored."""
    run_dir = Path(run_dir).resolve()
    meta = verify_run(run_dir)
    card = build_scorecard(run_dir)
    client = client or _client()
    experiment_id = _experiment(client, experiment_name(meta))
    if already_logged(client, experiment_id, meta["run_fingerprint"]):
        return None
    git = meta.get("git") or {}
    start = _millis(meta["created_utc"])
    tags = {FINGERPRINT_TAG: meta["run_fingerprint"], "harness.run_id": meta["run_id"],
            "role": role_of(meta), "track": track_of(meta), "suite_id": meta["suite_id"],
            "phase": meta.get("phase", "development"), "protocol_version": meta.get("protocol_version", ""),
            "history_status": meta.get("history_status", "reused_historical"),
            "scorecard_version": str(SCORECARD_VERSION), "git.commit": str(git.get("commit", "")),
            "git.dirty": str(bool(git.get("status"))) if git.get("status") != "unavailable" else "unknown",
            "author": str(meta.get("author", "")),
            "mlflow.runName": meta["model"]}
    if track_of(meta) == "clean":
        tags["input_track_caveat"] = "qc_fill_recipe_selected_on_full_record"
    run = client.create_run(experiment_id, start_time=start, tags=tags, run_name=meta["model"])
    rid = run.info.run_id
    entry = (meta.get("model_spec") or {}).get("entry_point") or {}
    params = {"model": meta["model"], "adapter": meta.get("adapter") or "",
              "entry_point": json.dumps(entry, sort_keys=True), "params": json.dumps(meta.get("params") or {}, sort_keys=True),
              "suite_id": meta["suite_id"], "phase": meta.get("phase", ""), "seed": meta["config"].get("seed"),
              "origins": ",".join(str(o) for o in meta.get("origins", [])), "track": track_of(meta),
              "control_set": CONTROL_SET, "scorecard_version": SCORECARD_VERSION}
    for key, value in params.items():
        client.log_param(rid, key, str(value)[:6000])
    for key, value in flat_metrics(card).items():
        client.log_metric(rid, key, value, timestamp=start, step=0)
    with tempfile.TemporaryDirectory() as temp:
        card_path = Path(temp) / "scorecard.json"
        card_path.write_text(json.dumps(card, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        client.log_artifact(rid, str(card_path))
        csv_path = Path(temp) / "scorecard.csv"
        pd.Series(flat_metrics(card)).rename("value").to_csv(csv_path, index_label="metric")
        client.log_artifact(rid, str(csv_path))
        for name in ("scores.json", "run.json", "params.json"):
            if (run_dir / name).is_file():
                client.log_artifact(rid, str(run_dir / name))
    end = start + int(1000 * float(meta.get("elapsed_seconds") or 0))
    client.set_terminated(rid, status="FINISHED", end_time=end)
    client.set_tag(rid, COMPLETE_TAG, "true")      # last: marks the mirror complete
    return rid


def sync(runs, echo=print) -> dict:
    """Mirror every complete, verified run under `runs`. Returns counts."""
    counts = {"logged": 0, "skipped_existing": 0, "skipped_incomplete": 0, "failed": 0}
    with sync_lock():
        client = _client()
        for path in sorted(Path(runs).glob("*/run.json")):
            try:
                status = read_json(path).get("status")
            except (OSError, ValueError):
                status = None
            if status != "complete":
                counts["skipped_incomplete"] += 1
                continue
            try:
                rid = log_run(path.parent, client)
            except Exception as exc:          # one bad run never stops the backfill
                counts["failed"] += 1
                echo(f"  could not mirror {path.parent.name}: {type(exc).__name__}: {exc}")
                continue
            counts["logged" if rid else "skipped_existing"] += 1
    return counts


def try_log_run(run_dir, echo=print) -> str | None:
    """Best-effort mirror after an evaluation: never raises."""
    try:
        import mlflow  # noqa: F401
    except ImportError:
        echo("MLflow is not installed; skipped mirroring. Install it with `uv sync`, then run "
             "`uv run python scripts/mlflow_sync.py`.")
        return None
    try:
        with sync_lock():
            return log_run(run_dir)
    except Exception as exc:
        echo(f"MLflow mirroring failed ({type(exc).__name__}: {exc}). The evaluation itself is saved; "
             "retry with `uv run python scripts/mlflow_sync.py`.")
        return None
