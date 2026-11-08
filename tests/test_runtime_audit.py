"""Real worker subprocess regression checks; fixtures contain synthetic data only."""
import hashlib
import json
import os
import py_compile
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]


def normalized_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class WorkerRuntimeAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "project"
        for name in ("common.py", "evaluation/__init__.py", "evaluation/assets.py",
                     "evaluation/runtime_audit.py", "evaluation/worker.py"):
            destination = self.root / "code" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / "code" / name, destination)
        self.work = self.base / "isolated-input"
        self.work.mkdir()
        (self.work / "train.csv").write_text("city_id,year,annual_avg_temp\nA,1980,55.0\n")
        (self.work / "targets.csv").write_text("city_id,year,metric\nA,1981,annual_avg_temp\n")
        (self.work / "params.json").write_text("{}")
        self.suite = self.root / "outputs/custom-suite"
        self.suite.mkdir(parents=True)
        (self.suite / "labels.csv").write_text("synthetic_private_label\n99\n")
        self.output = self.root / "outputs/custom-run/origin-1980"
        self.output.mkdir(parents=True)
        (self.root / "data").mkdir()
        for name in ("annual.csv", "annual_valid.csv"):
            (self.root / "data" / name).write_text("year,actual_value\n1981,99\n")
        self.adapter = self.root / "labs/m/model.py"
        self.adapter.parent.mkdir(parents=True)
        self.declared = []

    def tearDown(self):
        self.temp.cleanup()

    def run_worker(self, source, *, fail=True, extra_args=()):
        self.adapter.write_text(source, encoding="utf-8")
        context = {"schema_version": 1, "project_root": str(self.root), "suite_root": str(self.suite),
                   "own_output": str(self.output),
                   "source_pins": {p.relative_to(self.root).as_posix(): normalized_hash(p)
                                   for p in (self.root / "code").rglob("*.py")},
                   "adapter_file": {"path": str(self.adapter), "sha256": normalized_hash(self.adapter)},
                   "declared_sources": self.declared}
        context_path = self.work / "runtime-context.json"
        context_path.write_text(json.dumps(context))
        command = [sys.executable, str(self.root / "code/evaluation/worker.py"),
                   "--adapter", str(self.adapter) + ":forecast", "--runtime-context", str(context_path),
                   "--origin", "1980", "--seed", "17", "--output", str(self.output / "predictions.csv")]
        for name in ("train", "targets", "params"):
            command.extend(["--" + name, str(self.work / (name + (".json" if name == "params" else ".csv")))])
        result = subprocess.run([*command, *extra_args], cwd=self.work,
                                env={**os.environ, "PROTOSTAR_FOLD_CUTOFF": "1980"},
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode != 0, fail, result.stdout + result.stderr)
        self.assertTrue((self.output / "io_audit.json").is_file(), result.stderr)
        self.assertTrue((self.output / "imported_source.json").is_file(), result.stderr)
        return (json.loads((self.output / "io_audit.json").read_text()),
                json.loads((self.output / "imported_source.json").read_text()), result)

    def test_builtin_path_and_pandas_repo_data_reads_fail_before_scoring(self):
        expressions = ["open(DATA_DIR / 'annual_valid.csv').read()",
                       "(DATA_DIR / 'annual.csv').open().read()",
                       "pd.read_csv(DATA_DIR / 'annual.csv')"]
        for expression in expressions:
            with self.subTest(expression=expression):
                io, imports, result = self.run_worker(
                    "from common import DATA_DIR\nimport pandas as pd\n"
                    "def forecast(train, targets, origin, seed):\n"
                    f"    {expression}\n    return targets.assign(predicted_value=1.0)\n")
                self.assertEqual(io["status"], "failed")
                self.assertGreater(io["denied_count"], 0)
                self.assertIn("file-open tripwire", result.stderr)
                self.assertEqual(imports["status"], "verified")

    def test_import_time_and_swallowed_opens_still_fail(self):
        for source in (
            "from common import DATA_DIR\nopen(DATA_DIR / 'annual.csv')\n",
            "from common import DATA_DIR\ndef forecast(train, targets, origin, seed):\n"
            "    try:\n        open(DATA_DIR / 'annual.csv')\n"
            "    except RuntimeError:\n        pass\n"
            "    return targets.assign(predicted_value=1.0)\n",
        ):
            with self.subTest(source=source):
                io, _, _ = self.run_worker(source)
                self.assertGreater(io["denied_count"], 0)

    def test_custom_suite_sibling_outputs_and_normalized_traversal_are_blocked(self):
        sibling = self.output.parent / "origin-1987/private.csv"
        sibling.parent.mkdir()
        sibling.write_text("synthetic_label\n42\n")
        paths = [self.suite / "labels.csv", sibling,
                 self.output / "../../custom-suite/labels.csv"]
        for path in paths:
            with self.subTest(path=path):
                io, _, _ = self.run_worker(
                    "def forecast(train, targets, origin, seed):\n"
                    f"    open({str(path)!r}).read()\n"
                    "    return targets.assign(predicted_value=1.0)\n")
                self.assertGreater(io["denied_count"], 0)

    def test_symlink_alias_to_repository_data_is_blocked(self):
        alias = self.work / "alias.csv"
        try:
            alias.symlink_to(self.root / "data/annual.csv")
        except (OSError, NotImplementedError):
            self.skipTest("platform does not permit creating symlinks")
        io, _, _ = self.run_worker(
            "def forecast(train, targets, origin, seed):\n"
            f"    open({str(alias)!r}).read()\n"
            "    return targets.assign(predicted_value=1.0)\n")
        self.assertGreater(io["denied_count"], 0)

    def test_undeclared_sibling_helper_fails_both_before_and_after_drift(self):
        helper = self.adapter.with_name("helper.py")
        helper.write_text("LEVEL = 1.0\n")
        source = "import helper\ndef forecast(train, targets, origin, seed):\n    return targets.assign(predicted_value=helper.LEVEL)\n"
        _, imports, _ = self.run_worker(source)
        self.assertTrue(any(item["module"] == "helper" and item["status"] == "undeclared"
                            for item in imports["violations"]))
        helper.write_text("LEVEL = 123.0\n")
        _, imports, _ = self.run_worker(source)
        self.assertTrue(any(item["module"] == "helper" for item in imports["violations"]))

    def test_declared_helper_hash_allows_import_then_rejects_changed_source(self):
        helper = self.adapter.with_name("helper.py")
        helper.write_text("LEVEL = 2.0\n")
        snapshot = self.output / "declared_assets/files/labs/m/helper.py"
        snapshot.parent.mkdir(parents=True)
        shutil.copyfile(helper, snapshot)
        self.declared = [{"original_path": str(helper), "snapshot_path": str(snapshot), "sha256": normalized_hash(helper)}]
        source = "import helper\ndef forecast(train, targets, origin, seed):\n    return targets.assign(predicted_value=helper.LEVEL)\n"
        _, imports, _ = self.run_worker(source, fail=False)
        self.assertEqual(imports["status"], "verified")
        helper.write_text("LEVEL = 987.0\n")
        _, imports, _ = self.run_worker(source)
        self.assertTrue(any(item["module"] == "helper" and item["status"] == "changed" for item in imports["violations"]))

    def test_valid_adapter_package_resources_and_own_output_work(self):
        io, imports, _ = self.run_worker(
            "from sklearn.linear_model import LinearRegression\n"
            "from pathlib import Path\nimport os\n"
            "def forecast(train, targets, origin, seed):\n"
            f"    Path({str(self.output / 'model-note.txt')!r}).write_text('synthetic fixture')\n"
            "    fit = LinearRegression().fit([[0.0], [1.0]], [1.0, 2.0])\n"
            "    return targets.assign(predicted_value=float(fit.predict([[0.0]])[0]))\n", fail=False)
        self.assertEqual(io["denied_count"], 0)
        self.assertEqual(imports["status"], "verified")
        self.assertTrue((self.output / "predictions.csv").is_file())

    def test_cached_source_and_namespace_children_use_explicit_source_pins(self):
        helper = self.adapter.with_name("cached_helper.py")
        helper.write_text("LEVEL = 2.0\n")
        cache = py_compile.compile(str(helper), doraise=True)
        leaf = self.adapter.parent / "namespace/leaf.py"
        leaf.parent.mkdir()
        leaf.write_text("VALUE = 3.0\n")
        for original in (helper, leaf):
            snapshot = self.output / "source-snapshots" / original.name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, snapshot)
            self.declared.append({"original_path": str(original), "snapshot_path": str(snapshot),
                                  "sha256": normalized_hash(original)})
        _, imports, _ = self.run_worker(
            "import importlib.util\nimport sys\nimport namespace.leaf\n"
            f"spec = importlib.util.spec_from_file_location('cached_helper', {cache!r})\n"
            "cached_helper = importlib.util.module_from_spec(spec)\n"
            "sys.modules['cached_helper'] = cached_helper\nspec.loader.exec_module(cached_helper)\n"
            "def forecast(train, targets, origin, seed):\n"
            "    return targets.assign(predicted_value=cached_helper.LEVEL + namespace.leaf.VALUE)\n", fail=False)
        entries = {entry["module"]: entry for entry in imports["sources"]}
        self.assertEqual(entries["cached_helper"]["path"], str(helper.resolve()))
        self.assertEqual(entries["namespace.leaf"]["status"], "verified")

    def test_callback_failure_retains_both_audits_without_exception_contents(self):
        io, imports, _ = self.run_worker("def forecast(train, targets, origin, seed):\n    raise ValueError('private-content-marker')\n")
        self.assertEqual(imports["status"], "verified")
        self.assertNotIn("private-content-marker", json.dumps([io, imports]))

    def test_verified_declared_snapshot_is_readable_but_immutable(self):
        sys.path.insert(0, str(PROJECT / "code"))
        from evaluation import assets
        census = self.root / "data/census.csv"
        census.write_text("year,population\n1980,1234\n")
        (self.root / "assets.json").write_text(json.dumps({"schema_version": 1, "files": [
            {"path": "data/census.csv", "kind": "static_context"}]}))
        binding = assets.fingerprint_assets(self.root, "assets.json", [1980])
        snapshot = assets.snapshot_assets(self.root, binding, self.output, origin=1980)
        extra_args = ["--asset-root", snapshot["snapshot_root"], "--asset-binding", snapshot["binding_path"]]
        source = ("import pandas as pd\nfrom pathlib import Path\nimport os\n"
                  "def forecast(train, targets, origin, seed):\n"
                  "    path = Path(os.environ['PROTOSTAR_ASSET_ROOT']) / 'data/census.csv'\n"
                  "    population = float(pd.read_csv(path).population.iloc[0])\n"
                  "    return targets.assign(predicted_value=population)\n")
        io, _, _ = self.run_worker(source, fail=False, extra_args=extra_args)
        self.assertEqual(io["denied_count"], 0)
        tamper = source.replace("    population =", "    path.write_text('changed')\n    population =")
        io, _, _ = self.run_worker(tamper, extra_args=extra_args)
        self.assertTrue(any(event["reason"] == "immutable_supplied_input" and event["decision"] == "deny"
                            for event in io["events"]))


if __name__ == "__main__":
    unittest.main()
