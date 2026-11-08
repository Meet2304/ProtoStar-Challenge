"""Assembly-line labels and releases. No network, no MLflow."""
import importlib.util
import socket
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("line_tool", Path(__file__).resolve().parents[1] / "scripts/line.py")
line = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(line)

PINS = {name: "a" for name in line.EVALUATOR_FILES}
LOCK = {"line_started_utc": "20261108T000000Z", "suite_id": "s1", "current_evaluator": "eval-v1",
        "evaluators": {"eval-v1": line.evaluator_hash(PINS)}, "releases": {}}
RELEASED_ASSET = "data/processed/releases/clean-v1/origin-1983.csv"


def meta(**changes):
    base = {"created_utc": "20261108T120000Z", "git": {"status": ""}, "source_pins": PINS, "suite_id": "s1",
            "model": "m", "phase": "development", "model_spec": {"kind": "adapter", "assets": {"mode": "builtin"}}}
    return {**base, **changes}


def assets(files):
    return {"kind": "adapter", "assets": {"mode": "declared", "binding": {"files": files}}}


def no_release(_):
    return None


def release_of(eligible, sha="good"):
    return {"id": "clean-v1", "gate_eligible": eligible, "files": {"origin-1983.csv": {"sha256": sha}}}


SPEC_WITH_RELEASE = assets({RELEASED_ASSET: {"kind": "weather_input", "sha256": "good"}})


class LabelTests(unittest.TestCase):
    def test_clean_run_is_official(self):
        self.assertEqual(line.classify(meta(), LOCK, no_release), ("official", []))

    def test_run_before_the_line_is_legacy(self):
        self.assertEqual(line.classify(meta(created_utc="20261107T000000Z"), LOCK, no_release)[0], "legacy")

    def test_any_gap_makes_a_run_exploratory(self):
        for change in ({"git": {"status": " M code/x.py"}}, {"git": None}, {"suite_id": "other"},
                       {"source_pins": {**PINS, "code/evaluation/core.py": "b"}}):
            with self.subTest(change=change):
                status, reasons = line.classify(meta(**change), LOCK, no_release)
                self.assertEqual(status, "exploratory")
                self.assertTrue(reasons)

    def test_a_new_model_file_does_not_change_the_evaluator(self):
        self.assertEqual(line.evaluator_hash({**PINS, "code/my_model.py": "z"}), line.evaluator_hash(PINS))
        self.assertNotEqual(line.evaluator_hash({**PINS, "code/evaluation/core.py": "z"}), line.evaluator_hash(PINS))

    def test_unversioned_weather_input_blocks_official(self):
        spec = assets({"data/processed/meet/annual.parquet": {"kind": "weather_input", "sha256": "x"}})
        self.assertEqual(line.classify(meta(model_spec=spec), LOCK, no_release)[0], "exploratory")

    def test_static_context_needs_no_release(self):
        spec = assets({"data/city_info.csv": {"kind": "static_context", "sha256": "x"}})
        self.assertEqual(line.classify(meta(model_spec=spec), LOCK, no_release)[0], "official")

    def test_eligible_release_is_official(self):
        self.assertEqual(line.classify(meta(model_spec=SPEC_WITH_RELEASE), LOCK, lambda _: release_of(True))[0], "official")

    def test_ineligible_release_is_exploratory(self):
        self.assertEqual(line.classify(meta(model_spec=SPEC_WITH_RELEASE), LOCK, lambda _: release_of(False))[0], "exploratory")

    def test_modified_release_file_is_exploratory(self):
        self.assertEqual(line.classify(meta(model_spec=SPEC_WITH_RELEASE), LOCK, lambda _: release_of(True, "bad"))[0], "exploratory")


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(holder.cleanup)
        self.tmp = Path(holder.name)
        self.root = self.tmp / "releases"

    def source(self, *names):
        folder = self.tmp / "source"
        folder.mkdir(exist_ok=True)
        for name in names:
            (folder / name).write_text("year,value\n1980,1\n")
        return folder

    def test_release_numbers_increase_and_files_are_read_only(self):
        source = self.source("origin-1983.csv")
        first, _ = line.make_release("clean", source, "a", root=self.root, commit="c")
        second, folder = line.make_release("clean", source, "b", root=self.root, commit="c")
        self.assertEqual((first["id"], second["id"]), ("clean-v1", "clean-v2"))
        self.assertFalse((folder / "origin-1983.csv").stat().st_mode & stat.S_IWRITE)
        self.assertEqual(first["files"]["origin-1983.csv"]["information_end_year"], 1983)

    def test_adopted_numbering_starts_at_zero(self):
        manifest, _ = line.make_release("clean", self.source("all.csv"), "old", lineage="reconstructed",
                                        first_version=0, root=self.root, commit="c")
        self.assertEqual(manifest["id"], "clean-v0")
        self.assertFalse(manifest["gate_eligible"])

    def test_gate_eligible_data_must_be_per_origin(self):
        with self.assertRaises(line.LineError):
            line.make_release("clean", self.source("all.csv"), "x", gate_eligible=True, root=self.root, commit="c")

    def test_tampered_release_is_detected(self):
        _, folder = line.make_release("clean", self.source("origin-1983.csv"), "a", root=self.root, commit="c")
        target = folder / "origin-1983.csv"
        target.chmod(stat.S_IWRITE)
        target.write_text("changed")
        with self.assertRaises(line.LineError):
            line.verify_release_folder(folder)

    def test_serve_refuses_a_port_already_in_use(self):
        with socket.socket() as other_server:
            other_server.bind(("127.0.0.1", 0))
            other_server.listen()
            with self.assertRaises(line.LineError):
                line.serve("127.0.0.1", other_server.getsockname()[1], "")

    def test_assessment_ledger_is_left_out_of_the_shared_suite(self):
        suite = self.tmp / "suite"
        (suite / "assessment_access").mkdir(parents=True)
        (suite / "assessment_access/event.json").write_text("{}")
        (suite / "manifest.json").write_text("{}")
        line.zip_suite(suite, self.tmp / "out.zip")
        with zipfile.ZipFile(self.tmp / "out.zip") as archive:
            self.assertEqual(archive.namelist(), ["manifest.json"])


if __name__ == "__main__":
    unittest.main()
