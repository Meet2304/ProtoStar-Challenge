"""Offline schema-3 identity and LF/CRLF portability fixtures."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from evaluation import core
from evaluation.suite_identity import (canonical_hash, normalized_digest,
                                       preparation_source_pins, seal_suite_manifest,
                                       validate_suite_manifest)


def annual_fixture():
    records = []
    for city in ("A", "B"):
        for year in range(1982, 2019):
            value = 50 + .125 * (year - 1982) + (city == "B")
            records.append({"ID": city, "Year": year, "Miss Tmax": 0, "Miss Tmin": 0,
                            "Miss Prcp": 0, "Avg Temp": value, "ToT Prcp": value / 2,
                            "Max Tmax": value + 40, "cdd": value * 10})
    return pd.DataFrame(records)


class SuitePortabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.annual = self.root / "annual.csv"
        annual_fixture().to_csv(self.annual, index=False, lineterminator="\n")
        self.cities = self.root / "cities.csv"
        self.cities.write_bytes(b"city_id\nA\nB\n")
        self.config = core.read_json(ROOT / "evaluation/config.json")
        self.config.update(schema_version=3, origins=[2011], development_origins=[2011], assessment_origins=[])
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        # Snapshot once: concurrent source edits elsewhere cannot affect this fixture.
        self.pins = preparation_source_pins()

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, name, **kwargs):
        with patch("evaluation.suite_identity.preparation_source_pins", return_value=self.pins):
            return core.prepare(self.root / name, self.config_path,
                                kwargs.get("annual_path", self.annual), kwargs.get("cities_path", self.cities))

    def test_lf_crlf_inputs_sources_and_scorer_have_same_logical_id(self):
        scorer_bytes = core.SCORER.read_bytes().replace(b"\r\n", b"\n")
        scorer_lf, scorer_crlf = self.root / "scorer-lf.py", self.root / "scorer-crlf.py"
        scorer_lf.write_bytes(scorer_bytes)
        scorer_crlf.write_bytes(scorer_bytes.replace(b"\n", b"\r\n"))
        annual_crlf, cities_crlf = self.root / "annual-crlf.csv", self.root / "cities-crlf.csv"
        annual_crlf.write_bytes(self.annual.read_bytes().replace(b"\n", b"\r\n"))
        cities_crlf.write_bytes(self.cities.read_bytes().replace(b"\n", b"\r\n"))
        roots = [self.root / "source-lf", self.root / "source-crlf"]
        for name in self.pins:
            content = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
            for source_root, ending in zip(roots, (b"\n", b"\r\n")):
                path = source_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content.replace(b"\n", ending))
        self.assertEqual(preparation_source_pins(roots[0]), preparation_source_pins(roots[1]))
        self.pins = preparation_source_pins(roots[0])
        with patch.object(core, "SCORER", scorer_lf):
            lf = self.prepare("suite-lf")
        self.pins = preparation_source_pins(roots[1])
        with patch.object(core, "SCORER", scorer_crlf):
            crlf = self.prepare("suite-crlf", annual_path=annual_crlf, cities_path=cities_crlf)
            core.verify_suite(self.root / "suite-lf")
            core.verify_suite(self.root / "suite-crlf")
        self.assertEqual(lf["suite_id"], crlf["suite_id"])
        self.assertEqual(lf["files"], crlf["files"])
        self.assertNotEqual(lf["manifest_fingerprint"], crlf["manifest_fingerprint"])
        for suite in ("suite-lf", "suite-crlf"):
            for path in (self.root / suite).rglob("*.csv"):
                self.assertNotIn(b"\r\n", path.read_bytes())

    def test_derived_float_bytes_do_not_define_logical_identity_but_stay_sealed(self):
        ordinary = self.prepare("ordinary")
        build_fold = core.build_fold

        def slightly_different_sum(*args, **kwargs):
            train, targets, answer, reference = build_fold(*args, **kwargs)
            answer["baseline_value"] += 1e-12
            return train, targets, answer, reference

        with patch.object(core, "build_fold", side_effect=slightly_different_sum):
            changed = self.prepare("different-sum")
        self.assertEqual(ordinary["suite_id"], changed["suite_id"])
        self.assertNotEqual(ordinary["files"], changed["files"])
        self.assertNotEqual(ordinary["manifest_fingerprint"], changed["manifest_fingerprint"])
        core.verify_suite(self.root / "ordinary")
        core.verify_suite(self.root / "different-sum")
        path = self.root / "different-sum/origin-2011/private/answer_key.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "file changed or missing"):
            core.verify_suite(self.root / "different-sum")

    def test_manifest_integrity_and_logical_input_drift_are_both_detected(self):
        manifest = self.prepare("suite")
        original = copy.deepcopy(manifest)
        self.assertEqual(validate_suite_manifest(manifest), manifest)
        self.assertEqual(manifest, original)
        physical_change = copy.deepcopy(manifest)
        physical_change["files"]["origin-2011/train.csv"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "physical manifest"):
            validate_suite_manifest(physical_change)
        logical_change = copy.deepcopy(manifest)
        logical_change["config"]["seed"] += 1
        logical_change["manifest_fingerprint"] = canonical_hash(
            {k: v for k, v in logical_change.items() if k != "manifest_fingerprint"})
        with self.assertRaisesRegex(ValueError, "logical suite identity"):
            validate_suite_manifest(logical_change)
        altered_source = copy.deepcopy(manifest)
        altered_source["preparation_source_pins"]["code/common.py"] = "0" * 64
        self.assertNotEqual(seal_suite_manifest(altered_source)["suite_id"], manifest["suite_id"])

    def test_archived_schema_one_and_two_are_readable_but_not_preparable(self):
        for version in (1, 2):
            with self.subTest(version=version):
                suite = self.root / f"legacy-{version}"
                manifest = self.prepare(suite.name)
                manifest.pop("manifest_fingerprint")
                manifest.pop("preparation_source_pins")
                manifest["schema_version"] = version
                manifest["config"]["schema_version"] = version
                manifest["scorer_sha256"] = core.digest(core.SCORER)
                manifest.pop("suite_id")
                manifest["suite_id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
                (suite / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                self.assertEqual(core.verify_suite(suite)["schema_version"], version)
                old_config = dict(self.config, schema_version=version)
                self.config_path.write_text(json.dumps(old_config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "new suites require schema_version 3"):
                    self.prepare(f"rejected-{version}")
                self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def test_new_schema_rejects_actual_scorer_change(self):
        self.prepare("suite")
        altered = self.root / "modified-scorer.py"
        altered.write_bytes(core.SCORER.read_bytes() + b"\n# changed\n")
        with patch.object(core, "SCORER", altered):
            with self.assertRaisesRegex(ValueError, "scorer changed"):
                core.verify_suite(self.root / "suite")

    def test_normalization_never_rewrites_binary_input(self):
        binary = self.root / "source.parquet"
        content = b"PAR1\x00\r\n\x00PAR1"
        binary.write_bytes(content)
        self.assertEqual(normalized_digest(binary), hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    unittest.main()
