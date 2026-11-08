"""Known assessment-phase exposure by target city-year across suite versions."""
import json
from pathlib import Path
from evaluation.protocol import canonical_hash


def cohort_from_manifest(manifest, origins):
    cities = sorted(set(manifest.get("city_ids", [])))
    years = sorted({origin + lead for origin in origins for lead in range(1, 8)})
    cohort = {"city_ids": cities, "target_years": years}
    return {**cohort, "cohort_id": canonical_hash(cohort)}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def assessment_inventory(runs):
    """Known lower bounds; manual/test/raw reads are explicitly not counted."""
    events, suites, manifests, warnings = {}, set(), {}, []
    for path in sorted(Path(runs).glob("*/run.json")):
        try:
            meta = _read(path)
            suite_id = meta.get("suite_id")
            snapshot = path.parent / "suite_manifest.json"
            manifest = _read(snapshot) if snapshot.is_file() else None
            if manifest:
                manifests[suite_id] = manifest
            if meta.get("suite_path"):
                suites.add(Path(meta["suite_path"]))
            if meta.get("phase") == "assessment" and (meta.get("assessment_access_record") or
                    (meta.get("status") == "complete" and (path.parent / "freeze.json").is_file())):
                cohort = meta.get("assessment_cohort") or (cohort_from_manifest(manifest, meta.get("origins", [])) if manifest else {})
                events[meta["run_id"]] = {**cohort, "run_id": meta["run_id"], "suite_id": suite_id,
                    "freeze_id": meta.get("freeze_id"), "created_utc": meta.get("created_utc", "")}
        except (OSError, ValueError, KeyError, TypeError):
            warnings.append(f"Could not inventory assessment cohort for {path.parent.name}.")
    ledgers = {suite / "assessment_access" for suite in suites}
    ledgers.update(suite.parent / "assessment_access_registry" for suite in suites)
    for folder in sorted(ledgers):
        for path in sorted(folder.glob("*.json")):
            try:
                event = _read(path)
                identity = event["run_id"]
                if not event.get("city_ids"):
                    manifest = manifests.get(event.get("suite_id"))
                    if manifest:
                        event = {**cohort_from_manifest(manifest, event.get("origins", [])), **event}
                previous = events.get(identity, {})
                events[identity] = {**previous, **event}
            except (OSError, ValueError, KeyError, TypeError):
                warnings.append("An assessment access record could not be inventoried.")
    result = []
    for suite_id, manifest in manifests.items():
        cohort = cohort_from_manifest(manifest, manifest.get("config", {}).get("assessment_origins", []))
        cities, years = set(cohort["city_ids"]), set(cohort["target_years"])
        selected = [e for e in events.values() if cities.intersection(e.get("city_ids", []))
                    and years.intersection(e.get("target_years", []))]
        selected.sort(key=lambda e: (e.get("created_utc", ""), e["run_id"]))
        seen, repeats = set(), 0
        for event in selected:
            keys = {(city, year) for city in cities.intersection(event.get("city_ids", []))
                    for year in years.intersection(event.get("target_years", []))}
            repeats += bool(keys & seen)
            seen.update(keys)
        suite_ids = sorted({str(e.get("suite_id", "unknown")) for e in selected})
        result.append({"suite_id": suite_id, **cohort, "known_assessment_phase_accesses_across_suites": len(selected),
                       "known_repeated_city_year_accesses": repeats, "overlapping_access_suite_ids": suite_ids,
                       "known_access_run_ids": [e["run_id"] for e in selected],
                       "cross_suite_warning": len(suite_ids) > 1})
    unknown = sum(not e.get("city_ids") or not e.get("target_years") for e in events.values())
    if unknown:
        warnings.append(f"{unknown} known assessment-phase accesses have unknown city-year cohorts; matching counts are lower bounds.")
    return {"cohorts": result, "known_unique_assessment_phase_accesses": len(events),
            "accesses_with_unknown_cohort": unknown, "warnings": sorted(set(warnings)),
            "interpretation": "Known assessment-phase accesses across all available suites and protocols, deduplicated by run ID. "
                              "Counts do not include manual opens, raw-cache reads or real-data tests; missing/unshared ledgers make them lower bounds."}
