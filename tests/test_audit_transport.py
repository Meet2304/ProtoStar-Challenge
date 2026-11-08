"""Synthetic subprocess checks for parent-owned worker audit receipts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import test_runtime_audit as fixture_module

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from evaluation.audit_transport import AuditPipe


class AuditTransportTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.WorkerRuntimeAuditTests()
        self.fixture.setUp()
        self.root, self.work, self.output = self.fixture.root, self.fixture.work, self.fixture.output

    def tearDown(self):
        self.fixture.tearDown()

    def run_worker(self, source):
        adapter = self.fixture.adapter
        adapter.write_text(source, encoding="utf-8")
        context = {"schema_version": 1, "project_root": str(self.root), "suite_root": str(self.fixture.suite),
                   "own_output": str(self.output), "source_pins": {
                       p.relative_to(self.root).as_posix(): fixture_module.normalized_hash(p)
                       for p in (self.root / "code").rglob("*.py")},
                   "adapter_file": {"path": str(adapter), "sha256": fixture_module.normalized_hash(adapter)},
                   "declared_sources": []}
        context_path = self.work / "context.json"
        context_path.write_text(json.dumps(context), encoding="utf-8")
        command = [sys.executable, str(self.root / "code/evaluation/worker.py"),
            "--adapter", str(adapter) + ":forecast", "--runtime-context", str(context_path),
            "--origin", "1980", "--seed", "17", "--output", str(self.output / "predictions.csv")]
        for name in ("train", "targets", "params"):
            command += ["--" + name, str(self.work / (name + (".json" if name == "params" else ".csv")))]
        with AuditPipe() as pipe:
            process = subprocess.Popen(command + pipe.child_args, cwd=self.work,
                env={**os.environ, "PROTOSTAR_FOLD_CUTOFF": "1980", "PROTOSTAR_ARTIFACT_DIR": str(self.output)},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **pipe.popen_kwargs)
            pipe.after_spawn()
            try:
                stdout, stderr = process.communicate(timeout=45)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise
            receipt = pipe.finish()
        return process.returncode, receipt, stdout + stderr

    def test_success_returns_reports_over_pipe_not_model_writable_files(self):
        code, receipt, log = self.run_worker("def forecast(train, targets, origin, seed):\n"
            "    return targets.assign(predicted_value=55.)\n")
        self.assertEqual(code, 0, log)
        self.assertTrue(receipt["verified"], receipt["errors"])
        self.assertEqual(set(receipt["reports"]), {"io_audit.json", "imported_source.json"})
        self.assertFalse((self.output / "io_audit.json").exists())
        self.assertEqual(receipt["transport"], "parent_pipe_v1")

    def test_denied_event_survives_forged_files_and_zero_exit(self):
        code, receipt, log = self.run_worker("import os,json\nfrom pathlib import Path\nfrom common import DATA_DIR\n"
            "def forecast(train, targets, origin, seed):\n"
            "    try:\n        (DATA_DIR/'annual.csv').read_text()\n    except RuntimeError:\n        pass\n"
            "    for name in ('io_audit.json','imported_source.json'):\n"
            "        try:\n            (Path(os.environ['PROTOSTAR_ARTIFACT_DIR'])/name).write_text(json.dumps({'status':'verified'}))\n"
            "        except RuntimeError:\n            pass\n"
            "    os._exit(0)\n")
        self.assertEqual(code, 0, log)
        self.assertFalse(receipt["verified"])
        self.assertGreater(receipt["summary"]["denied_count"], 0)
        self.assertIn("missing final audit reports", receipt["errors"])

    def test_external_reads_and_subprocess_attempts_are_visible(self):
        outside = self.work / "unreviewed.csv"
        outside.write_text("synthetic\n", encoding="utf-8")
        source = ("from pathlib import Path\nimport subprocess,sys\n"
            "def forecast(train, targets, origin, seed):\n"
            f"    Path({str(outside)!r}).read_text()\n"
            "    try:\n        subprocess.run([sys.executable,'-c','pass'])\n"
            "    except RuntimeError:\n        pass\n"
            "    return targets.assign(predicted_value=55.)\n")
        code, receipt, log = self.run_worker(source)
        self.assertNotEqual(code, 0, log)
        self.assertFalse(receipt["verified"])
        self.assertTrue(any(item["path"] == str(outside.resolve()) for item in receipt["summary"]["external_reads"]))
        self.assertGreater(receipt["summary"]["subprocess_event_count"], 0)

    def test_protected_stdlib_copy_sqlite_and_mutations_fail(self):
        protected = self.root / "data/annual.csv"
        for operation in (f"shutil.copy2({str(protected)!r}, {str(self.work/'copy.csv')!r})",
                          f"sqlite3.connect({str(self.root/'data/model.sqlite')!r})",
                          f"os.remove({str(protected)!r})",
                          f"os.replace({str(protected)!r}, {str(self.root/'data/moved.csv')!r})"):
            with self.subTest(operation=operation):
                source = "import shutil,sqlite3,os\ndef forecast(train, targets, origin, seed):\n    " + operation + "\n    return targets.assign(predicted_value=55.)\n"
                code, receipt, log = self.run_worker(source)
                self.assertNotEqual(code, 0, log)
                self.assertFalse(receipt["verified"])
                self.assertGreater(receipt["summary"]["denied_count"], 0)
                self.assertTrue(protected.exists())

    def test_bad_or_duplicate_pipe_messages_are_rejected(self):
        for message in ({"type": "unknown"}, {"type": "report", "name": "io_audit.json", "report": {"status": "verified"}}):
            with self.subTest(message=message), AuditPipe() as pipe:
                # Unit-level malformed stream test through the actual OS pipe.
                raw = (json.dumps(message) + "\n").encode()
                os.write(pipe.write_fd, raw + raw)
                pipe.after_spawn()
                receipt = pipe.finish()
                self.assertFalse(receipt["verified"])
                self.assertTrue(receipt["errors"])

    def test_killed_worker_closes_pipe_without_waiting_for_missing_reports(self):
        with AuditPipe() as pipe:
            process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"],
                                       **pipe.popen_kwargs)
            pipe.after_spawn()
            process.kill()
            process.wait(timeout=5)
            receipt = pipe.finish(timeout=1)
            self.assertFalse(receipt["verified"])
            self.assertIn("missing final audit reports", receipt["errors"])
            self.assertNotIn("audit pipe did not close after worker termination", receipt["errors"])

    def test_nonpackage_environment_files_are_recorded(self):
        from evaluation.runtime_audit import RuntimeAudit
        environment = self.root / "synthetic-environment"
        environment.mkdir()
        target = environment / "cache/model-settings.json"
        target.parent.mkdir()
        target.write_text('{"synthetic":true}', encoding="utf-8")
        messages = []
        audit = RuntimeAudit({"schema_version": 1, "project_root": str(self.root),
            "suite_root": str(self.fixture.suite), "own_output": str(self.output),
            "source_pins": {}, "adapter_file": None, "declared_sources": []}, input_files=[], sender=messages.append)
        audit.environment = {environment.resolve()}
        audit.active = True
        audit.check_path(target)
        self.assertEqual(messages[0]["event"]["reason"], "environment_nonpackage_path")

    def test_interpreter_zip_probe_is_standard_but_not_writable(self):
        from evaluation.runtime_audit import RuntimeAudit, RuntimeAuditError
        environment = self.root / "interpreter"
        environment.mkdir()
        target = environment / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
        messages = []
        audit = RuntimeAudit({"schema_version": 1, "project_root": str(self.root),
            "suite_root": str(self.fixture.suite), "own_output": str(self.output),
            "source_pins": {}, "adapter_file": None, "declared_sources": []}, input_files=[], sender=messages.append)
        audit.environment = {environment.resolve()}
        audit.active = True
        audit.check_path(target)
        self.assertFalse(messages)
        with self.assertRaises(RuntimeAuditError):
            audit.check_path(target, writing=True)
        self.assertEqual(messages[0]["event"]["reason"], "immutable_interpreter_archive")


if __name__ == "__main__":
    unittest.main()
