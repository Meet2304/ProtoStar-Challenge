"""No real labels: cross-suite exposure, candidate counts and safe test defaults."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from evaluation.core import write_json
from evaluation.exposure import assessment_inventory, cohort_from_manifest
from evaluation.report import collect_exposure_counts
from evaluation.trials import candidate_inventory
from evaluation.input_review import review_input_summary


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"

    def tearDown(self):
        self.temp.cleanup()

    def add(self, run_id, suite_id, *, phase="assessment", origin=2004, module="my_model", status="complete"):
        run = self.runs / run_id
        run.mkdir(parents=True)
        suite = self.root / suite_id
        suite.mkdir(exist_ok=True)
        manifest = {"suite_id": suite_id, "city_ids": ["SYNTHETIC"],
                    "config": {"assessment_origins": [origin]}}
        cohort = cohort_from_manifest(manifest, [origin])
        meta = {"run_id": run_id, "suite_id": suite_id, "suite_path": str(suite), "phase": phase,
                "protocol_version": "test", "status": status, "origins": [origin],
                "model_spec": {"kind": "adapter", "entry_point": {"module": module, "function": "forecast"}},
                "created_utc": run_id, "assessment_cohort": cohort}
        if phase == "assessment":
            meta["assessment_access_record"] = "logged"
            write_json(run / "freeze.json", {"freeze_id": "freeze-" + suite_id})
            ledger = suite / "assessment_access"
            ledger.mkdir(exist_ok=True)
            write_json(ledger / f"{run_id}.json", {**cohort, "run_id": run_id, "suite_id": suite_id,
                "freeze_id": "freeze-" + suite_id, "origins": [origin], "created_utc": run_id, "protocol_version": "test"})
        write_json(run / "run.json", meta)
        write_json(run / "suite_manifest.json", manifest)

    def test_combines_identical_city_years_across_suites_and_deduplicates_copies(self):
        self.add("one", "suite-a")
        self.add("two", "suite-b")
        result = assessment_inventory(self.runs)
        self.assertEqual(result["known_unique_assessment_phase_accesses"], 2)
        for cohort in result["cohorts"]:
            self.assertEqual(cohort["known_assessment_phase_accesses_across_suites"], 2)
            self.assertEqual(cohort["known_repeated_city_year_accesses"], 1)
            self.assertTrue(cohort["cross_suite_warning"])
        filtered = collect_exposure_counts(self.runs, "suite-a")
        self.assertTrue(any("across 2 known suites" in warning for warning in filtered["warnings"]))

    def test_partial_target_overlap_is_counted(self):
        self.add("one", "suite-a", origin=2004)
        self.add("two", "suite-b", origin=2007)
        self.assertTrue(all(c["known_repeated_city_year_accesses"] == 1 for c in assessment_inventory(self.runs)["cohorts"]))

    def test_candidates_exclude_controls_and_incumbents_including_failed_attempts(self):
        for i, module in enumerate(("evaluation.controls", "evaluation.nested", "my_model", "my_model")):
            self.add(str(i), "suite-a", phase="development", module=module, status="failed" if i == 3 else "complete")
        result = candidate_inventory(self.runs, "suite-a")
        self.assertEqual(result["known_candidate_attempts"], 2)
        self.assertEqual(result["known_control_runs"], 1)
        self.assertEqual(result["known_incumbent_runs"], 1)

    def test_external_input_review_allowlist_is_narrow_and_subprocess_visible(self):
        self.assertFalse(review_input_summary({"external_reads": [{"path": "/sys/fs/cgroup/cpu.max", "count": 1}]})["requires_review"])
        self.assertFalse(review_input_summary({"external_reads": [{"path": "C:\\sys\\fs\\cgroup\\cpu.max", "count": 1}]})["requires_review"])
        self.assertTrue(review_input_summary({"external_reads": [{"path": "/elsewhere/labels.csv", "count": 1}]})["requires_review"])
        self.assertTrue(review_input_summary({"subprocess_event_count": 1})["requires_review"])

    def test_default_real_data_classes_are_skipped_without_reading_labels(self):
        import os
        if os.environ.get("PROTOSTAR_RUN_REAL_DATA_TESTS") == "1":
            self.skipTest("explicit real-data run; default-only check")
        from test_opus_fixes import PublishedReferenceTests
        from test_controls_panel import RealSuiteTests
        with patch("pandas.read_parquet", side_effect=AssertionError("real read forbidden")):
            suite = unittest.TestSuite([*unittest.defaultTestLoader.loadTestsFromTestCase(PublishedReferenceTests),
                                       *unittest.defaultTestLoader.loadTestsFromTestCase(RealSuiteTests)])
            result = unittest.TestResult()
            suite.run(result)
        self.assertEqual(len(result.skipped), 3)
        self.assertFalse(result.errors or result.failures)


if __name__ == "__main__":
    unittest.main()
