"""Known candidate attempts, separate from fixed controls and incumbents."""
import json
from pathlib import Path


def run_role(meta):
    spec = meta.get("model_spec", {})
    entry = spec.get("entry_point", {})
    identity = entry.get("module"), entry.get("function")
    if spec.get("kind") == "baseline" or identity in {
            ("evaluation.controls", "forecast"), ("evaluation.adapters", "reference")}:
        return "control"
    if identity in {("evaluation.adapters", "combo"), ("evaluation.nested", "forecast")}:
        return "incumbent"
    return "candidate"


def candidate_inventory(runs, suite_id):
    candidates, controls, incumbents, unreadable = {}, set(), set(), []
    for path in sorted(Path(runs).glob("*/run.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unreadable.append(path.parent.name)
            continue
        if meta.get("suite_id") != suite_id or meta.get("phase", "development") != "development":
            continue
        run_id = meta.get("run_id", path.parent.name)
        role = run_role(meta)
        if role == "candidate":
            candidates[run_id] = {"status": meta.get("status", "unknown"), "model": meta.get("model", "unknown")}
        else:
            (controls if role == "control" else incumbents).add(run_id)
    return {"known_candidate_attempts": len(candidates), "candidates": candidates,
            "known_control_runs": len(controls), "known_incumbent_runs": len(incumbents),
            "unreadable_records": unreadable,
            "interpretation": "Known attempts, including failed/running runs; not a count of independent architectures. "
                              "Deleted/unshared trials and tuning outside this suite are unknown. Repeated screening weakens gate evidence."}
