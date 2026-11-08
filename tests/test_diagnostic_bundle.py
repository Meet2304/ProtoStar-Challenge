"""Diagnostic bundle authorization and immutable separation from run records."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from evaluation import diagnostic_bundle as bundle
from evaluation.core import build_fold, load_annual, digest, read_json
from evaluation.protocol import PROTOCOL_VERSION, canonical_hash
from test_evaluation import annual_fixture


class DiagnosticBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.run = self.base / "evaluation/runs/run-fixture"
        (self.run / "origin-2004").mkdir(parents=True)
        self.annual = self.base / "annual.csv"
        annual_fixture().to_csv(self.annual, index=False)
        _, targets, _, _ = build_fold(load_annual(self.annual), ["A", "B"], 2004)
        targets.assign(predicted_value=55.).to_csv(self.run / "origin-2004/predictions.csv", index=False)
        self.meta = {"phase": "development", "protocol_version": PROTOCOL_VERSION,
                     "suite_id": "suite", "run_id": self.run.name, "run_fingerprint": "run-fingerprint", "origins": [2004]}
        self.manifest = {"suite_id": "suite", "city_ids": ["A", "B"],
                         "source": {"repo": "custom_local", "annual_sha256": digest(self.annual)}}
        self.patches = [patch.object(bundle, "verify_run", return_value=self.meta),
                        patch.object(bundle, "verify_suite", return_value=self.manifest)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_bundle_is_separate_sealed_and_missing_coverage_explicit(self):
        before = {p.relative_to(self.run): p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        output = bundle.create_diagnostics(self.run, self.base / "suite", annual_path=self.annual)
        record = read_json(output / "manifest.json")
        identity = record.pop("bundle_id")
        self.assertEqual(identity, canonical_hash(record))
        for name, expected in record["files"].items():
            self.assertEqual(expected, digest(output / name))
        self.assertEqual(record["coverage"]["status"], "unavailable")
        detail = read_json(output / "origin-2004.json")
        self.assertEqual(detail["variants"]["station_stable_target"]["coverage_status"], "unavailable")
        self.assertEqual(before, {p.relative_to(self.run): p.read_bytes() for p in self.run.rglob("*") if p.is_file()})
        self.assertFalse(output.is_relative_to(self.run))
        second = bundle.create_diagnostics(self.run, self.base / "suite", annual_path=self.annual)
        self.assertNotEqual(output, second)
        self.assertTrue((output / "manifest.json").exists())

    def test_assessment_and_suite_mismatch_fail_before_source_read(self):
        for changes, message in (({"phase": "assessment"}, "development"), ({"suite_id": "other"}, "differs")):
            original = self.meta.copy()
            self.meta.update(changes)
            with patch.object(pd, "read_csv", side_effect=AssertionError("source read")):
                with self.assertRaisesRegex(ValueError, message):
                    bundle.create_diagnostics(self.run, self.base / "suite", annual_path=self.annual)
            self.meta.update(original)

    def test_changed_annual_input_is_rejected(self):
        self.annual.write_bytes(self.annual.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "identical local annual"):
            bundle.create_diagnostics(self.run, self.base / "suite", annual_path=self.annual)


if __name__ == "__main__":
    unittest.main()
