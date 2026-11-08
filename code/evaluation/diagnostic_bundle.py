"""Immutable, separate development sensitivities using verified local inputs."""
from pathlib import Path
from datetime import datetime, timezone
import uuid

import pandas as pd

from common import ROOT, TRAIN_REVISION
from evaluation.core import read_json, write_json, digest, verify_run, verify_suite
from evaluation.protocol import canonical_hash, PROTOCOL_VERSION
from evaluation.quality_diagnostics import diagnose_origin, metrics_frame

COVERAGE_SHA256 = "d52dfcacd8eae99e5bd6c807f34dfe16f8f22e7688e5718a1ba93c8d93ca07e6"


def create_diagnostics(run, suite, *, annual_path=None, coverage_path=None):
    run, suite = Path(run).resolve(), Path(suite).resolve()
    meta, manifest = verify_run(run), verify_suite(suite)
    if meta.get("phase") != "development" or meta.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("diagnostics require a completed current-protocol development run")
    if meta["suite_id"] != manifest["suite_id"]:
        raise ValueError("diagnostic suite differs from the run")
    annual_path = Path(annual_path or ROOT / "data/raw/annual_indices/train-00000-of-00001.parquet")
    if not annual_path.is_file() or digest(annual_path) != manifest["source"]["annual_sha256"]:
        raise ValueError("diagnostics require the identical local annual source used to prepare the suite")
    if coverage_path is not None and manifest["source"]["repo"] != "custom_local":
        raise ValueError("official diagnostic coverage must use the verified pinned cache")
    coverage_source = {"status": "unavailable", "reason": "local pinned coverage not present"}
    if coverage_path is None and manifest["source"]["repo"] != "custom_local":
        candidate = ROOT / "data/raw/coverage_and_stations/train-00000-of-00001.parquet"
        metadata = ROOT / "data/raw/.cache/huggingface/download/coverage_and_stations/train-00000-of-00001.parquet.metadata"
        if candidate.exists():
            lines = metadata.read_text(encoding="utf-8").splitlines() if metadata.is_file() else []
            if len(lines) < 2 or lines[0] != TRAIN_REVISION or lines[1] != COVERAGE_SHA256 or digest(candidate) != COVERAGE_SHA256:
                raise ValueError("local coverage cache does not match the pinned training revision")
            coverage_path = candidate
    coverage = None
    if coverage_path is not None:
        coverage_path = Path(coverage_path)
        coverage_source = {"status": "available", "sha256": digest(coverage_path),
                           "revision": TRAIN_REVISION if manifest["source"]["repo"] != "custom_local" else None}
        coverage = pd.read_parquet(coverage_path) if coverage_path.suffix == ".parquet" else pd.read_csv(coverage_path, dtype={"ID": str})
    annual = pd.read_parquet(annual_path) if annual_path.suffix == ".parquet" else pd.read_csv(annual_path, dtype={"ID": str})
    output = run.parent.parent / "diagnostics" / run.name / uuid.uuid4().hex
    output.mkdir(parents=True)
    tables = []
    for origin in meta["origins"]:
        predictions = pd.read_csv(run / f"origin-{origin}/predictions.csv", dtype={"city_id": str})
        result = diagnose_origin(annual, coverage, origin, predictions, cities=manifest["city_ids"],
                                 provenance={"suite_id": manifest["suite_id"], "coverage": coverage_source})
        write_json(output / f"origin-{origin}.json", result)
        tables.append(metrics_frame(result))
    pd.concat(tables, ignore_index=True).to_csv(output / "metrics.csv", index=False, lineterminator="\n")
    record = {"schema_version": 1, "run_id": meta["run_id"], "run_fingerprint": meta["run_fingerprint"],
              "suite_id": manifest["suite_id"], "phase": "development", "origins": meta["origins"],
              "created_utc": datetime.now(timezone.utc).isoformat(), "annual_sha256": digest(annual_path),
              "coverage": coverage_source, "diagnostic_source_sha256": digest(Path(__file__).with_name("quality_diagnostics.py")),
              "files": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
              "interpretation": "Descriptive sensitivities on reused development labels. Official scores and run records are unchanged."}
    record["bundle_id"] = canonical_hash(record)
    write_json(output / "manifest.json", record)
    return output
