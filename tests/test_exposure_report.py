"""Synthetic-only trial/exposure reporting checks; no real assessment reads."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from evaluation.protocol import PROTOCOL_VERSION
from evaluation.report import collect_exposure_counts, render_report


class ExposureReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.runs = self.base / "runs"
        self.suite = self.base / "suite"
        self.ledger = self.suite / "assessment_access"
        self.ledger.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def run_record(self, run_id, phase="development", status="complete", protocol=PROTOCOL_VERSION,
                   freeze=None, access=False, suite="fixture", folder=None):
        meta = {"run_id": run_id, "suite_id": suite, "protocol_version": protocol,
                "suite_path": str(self.suite), "phase": phase, "status": status}
        run = self.runs / (folder or run_id)
        if access:
            meta["assessment_access_record"] = "original-machine/access.json"
        self.write_json(run / "run.json", meta)
        if freeze:
            self.write_json(run / "freeze.json", {"freeze_id": freeze, "protocol_version": protocol})
        return meta

    def event(self, name, run_id, freeze="freeze-a", protocol=PROTOCOL_VERSION, suite="fixture"):
        record = {"run_id": run_id, "suite_id": suite, "freeze_id": freeze}
        if protocol is not None:
            record["protocol_version"] = protocol
        self.write_json(self.ledger / f"{name}.json", record)

    def group(self, protocol=PROTOCOL_VERSION):
        result = collect_exposure_counts(self.runs)
        return next(g for g in result["groups"] if g["protocol_version"] == protocol)

    def test_trials_include_failed_running_and_deduplicate_run_ids(self):
        for status in ("complete", "failed", "running"):
            self.run_record("dev-" + status, status=status)
        self.run_record("dev-complete", folder="duplicate-copy")
        self.run_record("old", protocol="legacy-protocol")
        group = self.group()
        self.assertEqual(group["known_development_trials"], 3)
        self.assertEqual(group["development_status_counts"], {"complete": 1, "failed": 1, "running": 1})
        self.assertEqual(group["assessment_accesses"], 0)
        self.assertEqual(group["unique_freezes"], 0)
        self.assertEqual(self.group("legacy-protocol")["known_development_trials"], 1)

    def test_accesses_merge_ledger_run_copies_failed_attempts_and_repeated_freezes(self):
        self.run_record("dev")
        self.run_record("a", phase="assessment", freeze="freeze-a", access=True)
        self.run_record("b", phase="assessment", status="failed", freeze="freeze-a", access=True)
        self.run_record("c", phase="assessment", status="running", freeze="freeze-b", access=True)
        self.run_record("before-access-failed", phase="assessment", status="failed", freeze="freeze-c")
        self.event("a", "a", protocol=None)  # Legacy event attributed using run metadata.
        self.event("a-copy", "a")
        self.event("b", "b")
        self.event("orphan", "orphan", freeze="freeze-b")
        group = self.group()
        self.assertEqual(group["known_assessment_attempts"], 5)
        self.assertEqual(group["assessment_accesses"], 4)
        self.assertEqual(group["unique_freezes"], 2)
        self.assertEqual(group["known_repeated_assessment_accesses"], 2)
        self.assertEqual(group["ledger_event_records"], 4)

    def test_missing_ledger_never_fabricates_zero(self):
        self.run_record("dev")
        self.ledger.rmdir()
        group = self.group()
        self.assertEqual(group["known_assessment_accesses"], 0)
        self.assertIsNone(group["assessment_accesses"])
        self.assertIsNone(group["unique_freezes"])
        self.assertEqual(group["ledger_status"], "unavailable")
        self.assertTrue(group["warnings"])

    def test_missing_ledger_retains_metadata_lower_bounds(self):
        self.run_record("a", phase="assessment", freeze="freeze-a", access=True)
        self.ledger.rmdir()
        group = self.group()
        self.assertEqual(group["known_assessment_accesses"], 1)
        self.assertEqual(group["known_unique_freezes"], 1)
        self.assertIsNone(group["assessment_accesses"])

    def test_orphan_old_event_is_not_silently_assigned_a_protocol(self):
        self.run_record("dev")
        self.event("orphan", "orphan", protocol=None)
        result = collect_exposure_counts(self.runs)
        unattributed = next(g for g in result["groups"] if g["protocol_version"] == "unattributed-protocol")
        self.assertEqual(unattributed["known_assessment_accesses"], 1)
        self.assertIsNone(self.group()["assessment_accesses"])

    def test_corrupt_ledger_is_disclosed_and_does_not_establish_an_exact_total(self):
        self.run_record("dev")
        (self.ledger / "broken.json").write_text("{broken", encoding="utf-8")
        group = self.group()
        self.assertEqual(group["ledger_status"], "incomplete")
        self.assertIsNone(group["assessment_accesses"])
        self.assertIn("Invalid assessment ledger", group["warnings"][0])

    def test_report_embeds_counts_protocol_and_portable_escaped_metadata(self):
        self.run_record("shown")
        self.run_record("failed", status="failed")
        self.run_record("running", status="running")
        self.event("orphan", "access", freeze="</script><script>alert(1)</script>")
        self.write_json(self.runs / "shown/scores.json", [])
        table = pd.DataFrame([{"run_id": "shown", "model": "fixture", "suite_id": "fixture",
                               "protocol_version": PROTOCOL_VERSION, "phase": "development",
                               "comparison_group": f"fixture:development:{PROTOCOL_VERSION}"}])
        with patch("evaluation.report.leaderboard", return_value=table):
            path, _ = render_report(self.runs, self.base / "report")
        document = path.read_text(encoding="utf-8")
        payload = document.split('id="exposure-data">', 1)[1].split("</script>", 1)[0]
        groups = json.loads(payload)["groups"]
        self.assertEqual(groups[0]["known_development_trials"], 3)
        self.assertEqual(groups[0]["assessment_accesses"], 1)
        self.assertEqual(groups[0]["unique_freezes"], 1)
        self.assertIn("Known development trials:", document)
        self.assertIn("unknown total; at least", document)
        self.assertIn(f'const currentProtocol="{PROTOCOL_VERSION}"', document)
        self.assertNotIn("protocol_version==='asset-bound-v3'", document)
        self.assertNotIn("</script><script>alert(1)</script>", document)
        self.assertEqual((path.parent / "metadata/shown.json").read_bytes(),
                         (self.runs / "shown/run.json").read_bytes())
        saved = json.loads((path.parent / "exposure_counts.json").read_text())
        self.assertEqual(saved["groups"][0]["known_development_trials"], 3)

    def test_empty_report_still_discloses_failed_and_running_trial_groups(self):
        self.run_record("failed", status="failed")
        with patch("evaluation.report.leaderboard", return_value=pd.DataFrame()):
            path, table = render_report(self.runs, self.base / "report")
        self.assertTrue(table.empty)
        payload = path.read_text(encoding="utf-8").split('id="exposure-data">', 1)[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(payload)["groups"][0]["known_development_trials"], 1)


if __name__ == "__main__":
    unittest.main()
