"""Current protocol regressions using synthetic metadata and isolated local Git."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from evaluation import core, protocol


class ProtocolV4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "model-repository"
        (self.root / "code").mkdir(parents=True)
        (self.root / "code/model.py").write_text(
            "def forecast(train, targets, origin, seed, level=1.0):\n"
            "    return targets.assign(predicted_value=level)\n", encoding="utf-8")
        (self.root / "evaluation").mkdir()
        (self.root / "evaluation/no-assets.json").write_text('{"schema_version":1,"files":[]}\n')
        (self.root / "README.md").write_text("Synthetic model fixture.\n")
        self.hooks = self.base / "empty-hooks"
        self.hooks.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Protocol Fixture")
        self.git("config", "user.email", "protocol-fixture@example.invalid")
        self.git("config", "core.autocrlf", "false")
        self.commit("initial synthetic model")
        self.suite = self.base / "suite"
        self.suite.mkdir()
        self.freeze_path = self.base / "freeze.json"
        self.manifest = {"suite_id": "synthetic-suite-v3", "config": {
            "schema_version": 3, "origins": [1983, 1990, 2011],
            "development_origins": [1983, 1990], "assessment_origins": [2011],
            "history_status": "reused_historical", "seed": 42}}
        inventory = {"python": "3.13.0", "implementation": "CPython",
                     "packages": [{"name": "fixture-package", "version": "1.0"}]}
        self.environment = {"inventory": inventory, "sha256": protocol.canonical_hash(inventory)}
        self.environment_patch = patch.object(protocol, "installed_environment", return_value=self.environment)
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)
        self.suite_patch = patch.object(core, "verify_suite", return_value=self.manifest)
        self.suite_patch.start()
        self.addCleanup(self.suite_patch.stop)

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        result = subprocess.run(["git", "-c", "commit.gpgsign=false", "-c",
                                 f"core.hooksPath={self.hooks}", *args], cwd=self.root,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def commit(self, message):
        self.git("add", "--all")
        self.git("commit", "-qm", message)
        return self.git("rev-parse", "HEAD")

    def freeze_baseline(self):
        return protocol.create_freeze(self.suite, self.freeze_path, baseline="average", root=self.root)

    def verify_baseline(self):
        return protocol.verify_freeze(self.freeze_path, self.manifest,
                                      protocol.model_spec([2011], baseline="average"), root=self.root)

    def test_committed_documentation_descendant_preserves_freeze(self):
        frozen = self.freeze_baseline()
        (self.root / "README.md").write_text("Updated synthetic documentation.\n")
        head = self.commit("documentation only")
        self.assertNotEqual(head, frozen["git_commit"])
        verified = self.verify_baseline()
        self.assertEqual(verified["freeze_id"], frozen["freeze_id"])
        self.assertEqual(verified["source_pins"], protocol.source_pins(self.root))
        self.assertEqual(verified["installed_environment"], self.environment)

    def test_committed_source_change_invalidates_earlier_freeze(self):
        self.freeze_baseline()
        (self.root / "code/model.py").write_text("MODEL_VERSION = 2\n")
        self.commit("change model source")
        with self.assertRaisesRegex(ValueError, "source or Git ancestry changed"):
            self.verify_baseline()

    def test_identical_source_on_nonancestor_commit_cannot_reuse_freeze(self):
        frozen = self.freeze_baseline()
        self.git("checkout", "--orphan", "unrelated-history")
        self.commit("independent root with same source")
        self.assertEqual(protocol.source_pins(self.root), frozen["source_pins"])
        with self.assertRaisesRegex(ValueError, "source or Git ancestry changed"):
            self.verify_baseline()

    def test_installed_environment_change_invalidates_earlier_freeze(self):
        frozen = self.freeze_baseline()
        changed_inventory = json.loads(json.dumps(self.environment["inventory"]))
        changed_inventory["packages"][0]["version"] = "2.0"
        changed = {"inventory": changed_inventory, "sha256": protocol.canonical_hash(changed_inventory)}
        self.assertNotEqual(changed["sha256"], frozen["installed_environment"]["sha256"])
        with patch.object(protocol, "installed_environment", return_value=changed):
            with self.assertRaisesRegex(ValueError, "installed Python/package environment changed"):
                self.verify_baseline()

    def test_legacy_schemas_reject_new_runs_before_model_run_or_ledger_creation(self):
        for schema in (1, 2):
            for phase in ("development", "assessment"):
                runs = self.base / f"runs-schema-{schema}-{phase}"
                legacy = json.loads(json.dumps(self.manifest))
                legacy["config"]["schema_version"] = schema
                with self.subTest(schema=schema, phase=phase), \
                     patch.object(core, "verify_suite", return_value=legacy), \
                     patch.object(core, "model_spec") as model_spec, \
                     patch.object(core, "log_assessment_access") as ledger, \
                     patch.object(core, "score_fold") as scorer:
                    with self.assertRaisesRegex(ValueError, "legacy suites are retired"):
                        core.run_suite(self.suite, runs, "legacy canary", baseline="average", phase=phase)
                    model_spec.assert_not_called()
                    ledger.assert_not_called()
                    scorer.assert_not_called()
                    self.assertFalse(runs.exists())
                    self.assertFalse((self.suite / "assessment_access").exists())

    def previous_run(self):
        adapter = str(self.root / "code/model.py") + ":forecast"
        spec = protocol.model_spec([1983, 1990], adapter=adapter, params={"level": 7.0},
                                   assets="evaluation/no-assets.json", root=self.root)
        return {"phase": "development", "protocol_version": protocol.PROTOCOL_VERSION,
                "suite_id": self.manifest["suite_id"], "source_pins": protocol.source_pins(self.root),
                "model_spec": spec, "run_id": "synthetic-development-run", "run_fingerprint": "a" * 64}

    def test_freeze_from_run_recovers_adapter_parameters_assets_and_run_identity(self):
        previous = self.previous_run()
        from_run = self.base / "development-run"
        with patch.object(core, "verify_run", return_value=previous) as verify_run:
            frozen = protocol.create_freeze(self.suite, self.freeze_path, from_run=from_run, root=self.root)
        verify_run.assert_called_once_with(from_run)
        self.assertEqual(frozen["model_spec"]["entry_point"], {"file": "code/model.py", "function": "forecast"})
        self.assertEqual(frozen["model_spec"]["params"], {"level": 7.0})
        self.assertEqual(frozen["model_spec"]["assets"]["binding"]["manifest"]["path"], "evaluation/no-assets.json")
        self.assertEqual(frozen["model_spec"]["assets"]["binding"]["origins"], [2011])
        self.assertEqual(frozen["development_run"], {"run_id": previous["run_id"],
                                                   "run_fingerprint": previous["run_fingerprint"]})
        self.assertEqual(protocol.verify_freeze(self.freeze_path, self.manifest, frozen["model_spec"],
                                               root=self.root)["freeze_id"], frozen["freeze_id"])

    def test_freeze_from_run_rejects_overrides_and_incompatible_development_evidence(self):
        previous = self.previous_run()
        cases = [(previous, {"params": {"level": 99.0}}, "do not also override"),
                 (dict(previous, phase="assessment"), {}, "development run"),
                 (dict(previous, protocol_version="asset-bound-v3"), {}, "development run"),
                 (dict(previous, suite_id="other-suite"), {}, "suite/source differs"),
                 (dict(previous, source_pins={}), {}, "suite/source differs")]
        for evidence, overrides, message in cases:
            with self.subTest(message=message, overrides=overrides), \
                 patch.object(core, "verify_run", return_value=evidence):
                with self.assertRaisesRegex(ValueError, message):
                    protocol.create_freeze(self.suite, self.freeze_path, from_run=self.base / "previous",
                                           root=self.root, **overrides)
                self.assertFalse(self.freeze_path.exists())


if __name__ == "__main__":
    unittest.main()
