"""Freeze and evaluate the preregistered external-v2 station cohort once.

This supplementary benchmark never reads challenge-city assessment labels.
Run ``prepare`` after quality validation, then ``run``. No model is selected
or promoted from these results, and every preregistered result is reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import pandas as pd
from evaluation.core import prepare as prepare_suite, run_suite, verify_run, verify_suite
from evaluation.protocol import create_freeze, model_spec, canonical_hash, source_pins
from evaluation.report import render_report


def builder():
    # Import only on execution: the v2 builder is independently preregistered.
    import build_independent_holdout_v2
    return build_independent_holdout_v2


def inputs(selection):
    b = builder()
    selection = Path(selection).resolve()
    manifest = b.verify_selection(selection)
    prereg = b.verify_prereg(ROOT / manifest["preregistration"])
    home = selection.parent
    quality = b.read_json(home / "quality.json")
    provenance = b.read_json(home / "provenance.json")
    if quality["status"] != "eligible_for_supplementary_evaluation":
        raise ValueError("cohort failed its predeclared quality minimum; do not score")
    if quality["selection_sha256"] != b.digest(selection) or provenance["selection_sha256"] != b.digest(selection):
        raise ValueError("quality/provenance does not bind the frozen selection")
    for name, expected in provenance["output_sha256"].items():
        if Path(name).name != name or b.digest(home / name) != expected:
            raise ValueError("aggregated benchmark artifact changed")
    cities = pd.read_csv(home / "cities.csv")
    expected = {r["station_id"] for r in quality["stations"] if r["eligible"]}
    chosen = cities.loc[cities.eligible.eq(True)].copy()
    if set(chosen.city_id) != expected or len(chosen) != len(expected):
        raise ValueError("eligible station grid differs from the quality record")
    if len(expected) < 12 or chosen.stratum.nunique() < 4:
        raise ValueError("cohort is below preregistered coverage")
    return b, home, prereg, chosen


def prepare(selection):
    b, home, prereg, cities = inputs(selection)
    if (home / "evaluation-plan.json").exists():
        raise FileExistsError("evaluation plan already frozen")
    config = {"schema_version": 3, "origins": [1990, 2011],
        "development_origins": [1990], "assessment_origins": [2011],
        "history_status": "reused_historical", "horizon": 7,
        "live_leads": [1, 2, 3], "final_leads": [4, 5, 6, 7],
        "baseline_window": 30, "min_valid_years": 10,
        "max_missing_days": 10, "seed": 42}
    b.write_json(home / "evaluation-config.json", config)
    b.write_exclusive(home / "eligible-cities.csv", cities.to_csv(index=False, lineterminator="\n").encode())
    suite = home / "suite"
    manifest = prepare_suite(suite, home / "evaluation-config.json",
                             home / "annual_indices.csv", home / "eligible-cities.csv")
    entries = []
    # Freeze EVERY method before exposing ANY model error or skill.
    for method in [prereg["primary_method"], *prereg["secondary_methods"]]:
        freeze = home / "freezes" / (method["id"] + ".json")
        record = create_freeze(suite, freeze, adapter=method["adapter"], params=method["params"],
            note="External-v2 preregistered supplementary analysis; no model selection or submission promotion")
        entries.append({**method, "freeze": freeze.relative_to(home).as_posix(),
                        "freeze_id": record["freeze_id"]})
    plan = {"benchmark_id": prereg["benchmark_id"], "created_utc": b.utcnow(),
        "selection_sha256": b.digest(selection), "preregistration_sha256": b.digest(ROOT / b.read_json(selection)["preregistration"]),
        "quality_sha256": b.digest(home / "quality.json"), "suite_id": manifest["suite_id"],
        "source_hash": canonical_hash(source_pins()), "methods": entries,
        "primary_id": prereg["primary_method"]["id"], "no_scores_computed": True,
        "label_novelty": "User attests no prior team use of these stations; selection and methods frozen before scoring",
        "history_status_note": "Framework retains conservative reused_historical flag. New station labels supplement geographic evidence only; original challenge-city labels remain reused.",
        "development_origin_note": "1990 exists solely to satisfy suite contract; no development model runs or scores are opened."}
    b.write_json(home / "evaluation-plan.json", plan)
    print(f"All {len(entries)} methods frozen; no scores computed. Plan: {home / 'evaluation-plan.json'}")


def run(selection):
    b, home, prereg, _ = inputs(selection)
    plan = b.read_json(home / "evaluation-plan.json")
    if plan["selection_sha256"] != b.digest(selection) or plan["quality_sha256"] != b.digest(home / "quality.json"):
        raise ValueError("frozen evaluation inputs changed")
    if plan["source_hash"] != canonical_hash(source_pins()):
        raise ValueError("source changed after all-method freeze")
    methods = [prereg["primary_method"], *prereg["secondary_methods"]]
    if [{k: e[k] for k in ("id", "adapter", "params")} for e in plan["methods"]] != methods:
        raise ValueError("method list differs from preregistration")
    suite = home / "suite"
    if verify_suite(suite)["suite_id"] != plan["suite_id"]:
        raise ValueError("suite differs from frozen evaluation plan")
    # Exclusive attempt receipt: deliberately no silent repeat or outcome-based retry.
    b.write_json(home / "assessment-opened.json", {"opened_utc": b.utcnow(),
        "plan_sha256": b.digest(home / "evaluation-plan.json"), "interpretation": "These labels cease to be untouched after this opening."})
    results = []
    for method in plan["methods"]:
        try:
            path, meta = run_suite(suite, home / "runs", "external-v2:" + method["id"],
                adapter=method["adapter"], params=method["params"], phase="assessment",
                freeze=home / method["freeze"], timeout=600)
            verify_run(path)
            result = {"id": method["id"], "status": "complete", "run_id": path.name,
                "run_fingerprint": meta["run_fingerprint"], "summary": meta["summary"],
                "runtime_input_summary": meta.get("runtime_input_summary", {})}
        except Exception as exc:
            result = {"id": method["id"], "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        b.write_json(home / "results" / (method["id"] + ".json"), result)
        print(f"{method['id']}: {result['status']}", flush=True)
    b.write_json(home / "evaluation-results.json", {"benchmark_id": prereg["benchmark_id"],
        "plan_sha256": b.digest(home / "evaluation-plan.json"), "primary_id": plan["primary_id"],
        "results": results, "interpretation": "Single geographic holdout, not leaderboard prediction or model-selection evidence for future reuse. All methods reported irrespective of sign."})
    render_report(home / "runs", home / "report", plan["suite_id"])
    print(f"Report: {home / 'report/index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--selection", type=Path, default=ROOT / "outputs/independent-v2/selection.json")
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.command](args.selection)


if __name__ == "__main__":
    main()
