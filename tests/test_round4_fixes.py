"""Cross-suite protected years and mandatory successful worker completion."""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from evaluation.core import prepare, read_json, write_json, run_suite, verify_run
from evaluation.phase_policy import PROTECTED_CITIES, enforce_phase_policy
from evaluation.protocol import PROTOCOL_VERSION, canonical_hash
from evaluation.report import collect_exposure_counts
from test_evaluation import annual_fixture


class Round4FixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_source_or_reserved_city_ids_prevent_relabelling_and_window_overlap(self):
        for source, cities in (({"repo": "CMU-OSPO/protostar-climate-train-to-2018"}, ["other"]),
                               ({"repo": "custom_local"}, [sorted(PROTECTED_CITIES)[0]])):
            for origin in (2003, 2004, 2011):
                with self.subTest(source=source, origin=origin), self.assertRaisesRegex(ValueError, "reserved assessment"):
                    enforce_phase_policy({"source": source, "city_ids": cities,
                                          "config": {"development_origins": [origin]}})
        enforce_phase_policy({"source": {"repo": "custom_local"}, "city_ids": ["SYNTHETIC"],
                              "config": {"development_origins": [2011]}})

    def test_prepare_rejects_csv_converted_reserved_station_data_before_writing_suite(self):
        city = sorted(PROTECTED_CITIES)[0]
        annual_fixture().replace({"ID": {"A": city, "B": "SYNTHETIC"}}).to_csv(self.root / "annual.csv", index=False)
        pd.DataFrame({"city_id": [city]}).to_csv(self.root / "cities.csv", index=False)
        config = read_json(ROOT / "evaluation/config.json")
        config.update(origins=[2011], development_origins=[2011], assessment_origins=[])
        write_json(self.root / "config.json", config)
        with self.assertRaisesRegex(ValueError, "reserved assessment"):
            prepare(self.root / "suite", self.root / "config.json", self.root / "annual.csv", self.root / "cities.csv")
        self.assertFalse((self.root / "suite").exists())

    def test_report_warns_of_exposure_in_another_suite_even_when_filtered(self):
        folder = self.root / "runs/earlier"
        folder.mkdir(parents=True)
        write_json(folder / "run.json", {"run_id": "earlier", "suite_id": "other-suite",
            "protocol_version": "older", "phase": "development", "status": "complete"})
        write_json(folder / "suite_manifest.json", {"city_ids": [sorted(PROTECTED_CITIES)[0]],
            "source": {"repo": "custom_local"}, "config": {"development_origins": [2011]}})
        result = collect_exposure_counts(self.root / "runs", suite_id="current-suite")
        self.assertTrue(any("Cross-suite exposure" in warning for warning in result["warnings"]))

    def test_zero_exit_without_audit_files_fails_before_scoring(self):
        annual_fixture().to_csv(self.root / "annual.csv", index=False)
        pd.DataFrame({"city_id": ["A", "B"]}).to_csv(self.root / "cities.csv", index=False)
        config = read_json(ROOT / "evaluation/config.json")
        config.update(origins=[1990], development_origins=[1990], assessment_origins=[])
        write_json(self.root / "config.json", config)
        prepare(self.root / "suite", self.root / "config.json", self.root / "annual.csv", self.root / "cities.csv")
        adapter = self.root / "early_exit.py"
        adapter.write_text("import os\nfrom pathlib import Path\ndef forecast(train, targets, origin, seed):\n"
            "    targets.assign(predicted_value=55.).to_csv(Path(os.environ['PROTOSTAR_ARTIFACT_DIR'])/'adapter_output.csv',index=False)\n"
            "    os._exit(0)\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "did not finish a verified"):
            run_suite(self.root / "suite", self.root / "runs", "exits", adapter=str(adapter) + ":forecast",
                      assets=ROOT / "evaluation/no-assets.json")
        meta = read_json(next((self.root / "runs").glob("*/run.json")))
        self.assertEqual(meta["status"], "failed")
        self.assertFalse(next((self.root / "runs").iterdir()).joinpath("scores.json").exists())

    def test_verify_run_rejects_forged_complete_without_verified_audits(self):
        run = self.root / "run"
        run.mkdir()
        meta = {"status": "complete", "schema_version": 2, "protocol_version": PROTOCOL_VERSION,
                "adapter": "example:forecast", "origins": [1990]}
        meta["run_fingerprint"] = canonical_hash(meta)
        write_json(run / "run.json", meta)
        with self.assertRaisesRegex(ValueError, "verified worker audit"):
            verify_run(run)


if __name__ == "__main__":
    unittest.main()
