"""Control panel: the fixed controls plus the incumbents, on one suite and the development phase.

The panel is the yardstick a new model is compared with. It is built once per suite, phase,
protocol, control set and PROFILE, and an entry is reused only while it is still current:

    reuse key = hash(suite id, phase, protocol version, control set, the entry's model
                     specification, installed environment, and the sealed source of every file
                     that can change a control's predictions or its scores — PANEL_SOURCES)

Reuse also requires the run to pass verify_run with the recorded fingerprint, and summaries are
re-derived from the verified run, never trusted from panel.json. Adding an unrelated teammate
model file under code/ does not make the panel stale.

Profiles: "A" = the five simple controls + combo + nested (ships on its own);
          "full" = A + the Tier B controls (ridge_pooled, starter_rf).
Official track, development phase only. Assessment panels use frozen controls (CLI owner).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import METRICS, ROOT
from evaluation.controls import CONTROL_SET, CONTROLS, TIER
from evaluation.core import read_json, run_suite, verify_run, verify_suite
from evaluation.protocol import (PROTOCOL_VERSION, canonical_hash, installed_environment, model_spec,
                                 phase_origins, source_pins)

PANEL_ROOT = ROOT / "outputs/evaluation/controls"
PANEL_VERSION = 2
PROFILES = ("A", "full")
# Files whose content can change a control's or incumbent's predictions or its scoring.
PANEL_SOURCES = (
    "code/common.py", "code/combo_model.py", "code/starter_model.py",
    "code/evaluation/adapters.py", "code/evaluation/controls.py", "code/evaluation/nested.py",
    "code/evaluation/official_score.py", "code/evaluation/core.py", "code/evaluation/worker.py",
    "code/evaluation/runtime_audit.py", "code/evaluation/protocol.py", "code/evaluation/assets.py",
    "code/evaluation/native_io.py", "code/evaluation/phase_policy.py", "code/evaluation/suite_identity.py",
    "uv.lock", "pyproject.toml", "requirements.txt", "code/requirements.txt",
)
INCUMBENTS = {"combo": "evaluation.adapters:combo", "nested": "evaluation.nested:forecast"}
SKILL_TOLERANCE = 1e-9
VALUE_TOLERANCE = 1e-6
TIE = 1e-10                 # the harness's numerical tie tolerance on raw skill
G3_FLOOR = -0.02
KEY = ["city_id", "year", "metric"]
REFERENCE_FILES = {"average30": "average", "persistence": "last_year", "trend30": "trend"}


def entries(profile: str = "full"):
    """(entry id, role, run name, adapter, params) in display order for a profile."""
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    rows = [(cid, "control", f"control:{cid}", "evaluation.controls:forecast", {"method": cid})
            for cid in CONTROLS if profile == "full" or TIER[cid] == "A"]
    rows += [(iid, "incumbent", f"incumbent:{iid}", adapter, {}) for iid, adapter in INCUMBENTS.items()]
    return rows


def _panel_sources(pins: dict) -> dict:
    return {name: pins.get(name) for name in PANEL_SOURCES}


def reuse_key(suite_id, phase, spec, pins, environment_sha):
    return canonical_hash({"suite_id": suite_id, "phase": phase, "protocol_version": PROTOCOL_VERSION,
                           "control_set": CONTROL_SET, "model_spec": spec,
                           "sources": _panel_sources(pins), "environment": environment_sha})


def run_key(meta) -> str:
    if meta.get("protocol_version") != PROTOCOL_VERSION:
        return ""
    return reuse_key(meta["suite_id"], meta.get("phase"), meta["model_spec"], meta["source_pins"],
                     meta["installed_environment"]["sha256"])


def _verified(run_dir: Path, key: str, fingerprint: str | None = None):
    """verify_run + reuse-key (+ fingerprint) match, else None."""
    try:
        meta = verify_run(run_dir)
    except (ValueError, OSError, KeyError):
        return None
    if run_key(meta) != key or (fingerprint is not None and meta.get("run_fingerprint") != fingerprint):
        return None
    return meta


def find_current(runs: Path, name: str, key: str):
    """Newest complete, verified run with this name and reuse key, else None."""
    for path in sorted(Path(runs).glob("*/run.json"), reverse=True):
        try:
            meta = read_json(path)
        except (OSError, ValueError):
            continue
        if meta.get("model") != name or meta.get("status") != "complete" or run_key(meta) != key:
            continue
        verified = _verified(path.parent, key)
        if verified is not None:
            return path.parent, verified
    return None


def sanity_checks(suite: Path, entry_id: str, run_dir: Path, meta: dict) -> list[str]:
    """Fail-loud checks for the official references; returns a list of problems."""
    problems = []
    if entry_id == "average30":
        for fold in read_json(run_dir / "scores.json"):
            for metric, skill in fold["skill"].items():
                if not abs(skill) < SKILL_TOLERANCE:
                    problems.append(f"average30 skill {skill!r} at origin {fold['origin']} {fold['split']} {metric}")
    if entry_id in REFERENCE_FILES:
        for origin in meta["origins"]:
            made_path = run_dir / f"origin-{origin}/adapter_output.csv"
            built_path = suite / f"origin-{origin}/baselines/{REFERENCE_FILES[entry_id]}.csv"
            # Values, not bytes: float text can differ in the last digit across pandas/platforms
            # while every score is unchanged. Exact bytes are recorded separately for information.
            made, built = (pd.read_csv(p, dtype={"city_id": str, "metric": str}).set_index(KEY).sort_index()
                           for p in (made_path, built_path))
            if not made.index.equals(built.index):
                problems.append(f"{entry_id} origin {origin}: prediction grid differs from the prebuilt baseline")
                continue
            gap = float((made.predicted_value - built.predicted_value).abs().max())
            if not gap <= VALUE_TOLERANCE:
                problems.append(f"{entry_id} origin {origin}: predictions differ from the prebuilt baseline by {gap:g}")
    return problems


def reference_bytes_identical(suite: Path, entry_id: str, run_dir: Path, meta: dict):
    if entry_id not in REFERENCE_FILES:
        return None
    return all((run_dir / f"origin-{o}/adapter_output.csv").read_bytes()
               == (suite / f"origin-{o}/baselines/{REFERENCE_FILES[entry_id]}.csv").read_bytes() for o in meta["origins"])


def summarize(run_dir: Path, meta: dict) -> dict:
    scores = [s for s in read_json(run_dir / "scores.json") if s["split"] == "final"]
    return {"headline_final_raw_skill": meta["summary"]["final"]["mean_skill_unclipped"],
            "live_raw_skill": meta["summary"]["live"]["mean_skill_unclipped"],
            "final_skill_by_origin": {str(s["origin"]): s["mean_skill_unclipped"] for s in scores},
            "final_skill_by_metric": {m: float(np.mean([s["skill"][m] for s in scores])) for m in METRICS}}


def best_control(panel_entries: dict) -> str | None:
    """Highest development headline among controls other than the average30 denominator.
    Ties within the harness's 1e-10 rounding go to the alphabetically FIRST id."""
    eligible = {k: v for k, v in panel_entries.items() if v["role"] == "control" and k != "average30"}
    if not eligible:
        return None
    return sorted(eligible, key=lambda k: (-round(eligible[k]["headline_final_raw_skill"], 10), k))[0]


def panel_path(suite_id: str, profile: str = "full", phase: str = "development", track: str = "official",
               root: Path = PANEL_ROOT) -> Path:
    return Path(root) / suite_id / phase / track / f"panel-{profile}.json"


def _describe(entry_id):
    if entry_id in CONTROLS:
        return CONTROLS[entry_id]
    return ("incumbent", "current submission method" if entry_id == "combo" else "nested per-origin method selection")


def build_panel(suite, runs, profile="full", timeout=900, echo=print, panel_root=PANEL_ROOT) -> tuple[Path, dict]:
    """Make the profile's development panel complete and current; returns (path, panel)."""
    suite, runs = Path(suite).resolve(), Path(runs).resolve()
    phase = "development"
    manifest = verify_suite(suite)
    origins = phase_origins(manifest, phase)
    pins = source_pins()
    environment_sha = installed_environment()["sha256"]
    selected = entries(profile)
    panel = {"panel_version": PANEL_VERSION, "profile": profile, "entry_ids": [e[0] for e in selected],
             "control_set": CONTROL_SET, "suite_id": manifest["suite_id"], "phase": phase, "track": "official",
             "protocol_version": PROTOCOL_VERSION, "origins": origins, "sources": _panel_sources(pins),
             "environment_sha256": environment_sha, "entries": {}}
    problems = []
    for entry_id, role, name, adapter, params in selected:
        key = reuse_key(manifest["suite_id"], phase, model_spec(origins, adapter, params), pins, environment_sha)
        found = find_current(runs, name, key)
        if found is None:
            echo(f"  running {name} ...")
            run_dir, _ = run_suite(suite, runs, name, adapter=adapter, params=params, timeout=timeout, phase=phase)
            meta = _verified(run_dir, key)
            if meta is None:
                raise RuntimeError(f"{name}: source or environment changed while the panel was running; rerun")
            reused = False
        else:
            run_dir, meta = found
            reused = True
        problems += sanity_checks(suite, entry_id, run_dir, meta)
        tier, description = _describe(entry_id)
        panel["entries"][entry_id] = {"role": role, "tier": tier, "description": description, "run_name": name,
                                      "run_id": meta["run_id"], "run_dir": str(run_dir),
                                      "run_fingerprint": meta["run_fingerprint"], "reuse_key": key,
                                      "reused": reused,
                                      "reference_bytes_identical": reference_bytes_identical(suite, entry_id, run_dir, meta),
                                      **summarize(run_dir, meta)}
        echo(f"  {'reused ' if reused else 'new    '} {name:28s} headline "
             f"{panel['entries'][entry_id]['headline_final_raw_skill']:+.4f}")
    if problems:
        raise RuntimeError("control panel sanity checks failed:\n  " + "\n  ".join(problems))
    panel["best_control"] = best_control(panel["entries"])
    panel["built_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = panel_path(manifest["suite_id"], profile, root=panel_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(panel, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path, panel


def load_current_panel(suite, runs, profile="full", panel_root=PANEL_ROOT) -> dict | None:
    """The saved panel if every entry of the profile verifies and is still current, else None.

    Each entry's run is re-verified (verify_run + recorded fingerprint + recomputed reuse key),
    and its summary is re-derived from the verified run rather than trusted from panel.json.
    """
    manifest = verify_suite(suite)
    path = panel_path(manifest["suite_id"], profile, root=panel_root)
    if not path.is_file():
        return None
    try:
        panel = read_json(path)
    except (OSError, ValueError):
        return None
    if panel.get("panel_version") != PANEL_VERSION or panel.get("profile") != profile:
        return None
    pins, environment_sha = source_pins(), installed_environment()["sha256"]
    origins = phase_origins(manifest, "development")
    selected = entries(profile)
    if panel.get("entry_ids") != [e[0] for e in selected]:
        return None
    for entry_id, role, name, adapter, params in selected:
        entry = panel["entries"].get(entry_id)
        if entry is None:
            return None
        key = reuse_key(manifest["suite_id"], "development", model_spec(origins, adapter, params), pins, environment_sha)
        run_dir = Path(entry.get("run_dir", ""))
        meta = _verified(run_dir, key, entry.get("run_fingerprint")) if run_dir.is_dir() else None
        if meta is None or entry.get("reuse_key") != key:
            return None
        entry.update(summarize(run_dir, meta))
    panel["best_control"] = best_control(panel["entries"])
    return panel


def gates(card: dict, panel: dict, vs_combo: dict) -> list[tuple[str, bool, str]]:
    """Screening gates (heuristics on three reused periods, not statistical tests)."""
    head = card["fidelity"]["headline_final_raw_skill"]
    best_id = panel["best_control"]
    best = panel["entries"][best_id]["headline_final_raw_skill"]
    deltas = {str(p["origin"]): p["raw_skill_delta"] for p in vs_combo["periods"]}
    latest = str(max(int(o) for o in deltas))
    worst_metric, worst_skill = min(card["per_metric"]["final"].items(), key=lambda kv: kv[1])
    mean_delta = vs_combo["mean_final_raw_skill_delta_second_minus_first"]
    return [
        ("G1 beats the 30-year average", head > TIE, f"headline {head:+.4f}"),
        ("G2 beats the best control", head > best + TIE, f"{best_id} {best:+.4f}"),
        ("G3 no metric below -0.02", worst_skill >= G3_FLOOR, f"worst {worst_metric} {worst_skill:+.4f}"),
        ("G4 beats combo (mean, >=2 origins, not losing the latest)",
         mean_delta > TIE and sum(d > TIE for d in deltas.values()) >= 2 and deltas[latest] >= -TIE,
         "origin deltas " + " ".join(f"{o}:{d:+.4f}" for o, d in deltas.items())),
    ]
