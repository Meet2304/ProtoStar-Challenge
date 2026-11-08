"""Ordinary native-reader and network accident checks on synthetic fixtures."""
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

import test_runtime_audit as fixture_module


class NativeInputAuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.WorkerRuntimeAuditTests()
        self.fixture.setUp()
        self.root, self.output = self.fixture.root, self.fixture.output
        self.data = self.root / "data/raw/annual_indices"
        self.data.mkdir(parents=True)
        values = pd.DataFrame({"city_id": ["A"], "year": [1981], "actual_value": [99.]})
        self.parquet = self.data / "train-00000-of-00001.parquet"
        self.csv = self.data / "labels.csv"
        self.feather = self.data / "labels.feather"
        values.to_parquet(self.parquet)
        values.to_csv(self.csv, index=False)
        values.to_feather(self.feather)

    def tearDown(self):
        self.fixture.tearDown()

    def check_expression(self, expression, *, imports="", fail=True):
        source = ("import pandas as pd\nfrom pathlib import Path\nimport pyarrow as pa\n"
                  "import pyarrow.parquet as pq\nimport pyarrow.csv as pcsv\n"
                  "import pyarrow.feather as feather\nimport pyarrow.dataset as ds\n"
                  "import pyarrow.fs as fs\n" + imports + "\n"
                  "def forecast(train, targets, origin, seed):\n"
                  f"    {expression}\n"
                  "    return targets.assign(predicted_value=1.0)\n")
        return self.fixture.run_worker(source, fail=fail)

    def test_standard_native_readers_cannot_open_raw_cache_paths(self):
        expressions = [
            f"pd.read_parquet({str(self.parquet)!r})",
            f"pq.read_table({str(self.parquet)!r})",
            f"pq.read_pandas(source={str(self.parquet)!r})",
            f"pq.ParquetFile({str(self.parquet)!r})",
            f"pcsv.read_csv({str(self.csv)!r})",
            f"pcsv.open_csv(input_file={str(self.csv)!r})",
            f"feather.read_table({str(self.feather)!r})",
            f"pd.read_feather({str(self.feather)!r})",
            f"ds.dataset({str(self.parquet)!r})",
            f"ds.dataset([{str(self.parquet)!r}])",
            f"pa.OSFile({str(self.parquet)!r}, 'r')",
            f"pa.memory_map({str(self.parquet)!r})",
            f"pa.memory_map(path={str(self.parquet)!r})",
            f"fs.LocalFileSystem().open_input_file({str(self.parquet)!r})",
            f"fs.LocalFileSystem().open_input_stream({str(self.parquet)!r})",
            f"fs.FileSystem.from_uri({str(self.parquet)!r})[0].open_input_file({str(self.parquet)!r})",
            f"pa.input_stream({str(self.parquet)!r})",
        ]
        for expression in expressions:
            with self.subTest(expression=expression):
                io, _, result = self.check_expression(expression)
                self.assertGreater(io["denied_count"], 0, result.stderr)
                self.assertTrue(any(event["decision"] == "deny" for event in io["events"]))

    def test_project_configuration_and_unregistered_code_reads_are_denied(self):
        config = self.root / "code/cfg.json"
        config.write_text('{"level":99}')
        labels = self.root / "notebooks/labels.csv"
        labels.parent.mkdir()
        labels.write_text("year,actual_value\n1981,99\n")
        helper = self.root / "notebooks/undeclared.py"
        helper.write_text("LEVEL=99\n")
        expressions = [f"json.loads(Path({str(config)!r}).read_text())",
                       f"pd.read_csv({str(labels)!r})",
                       f"exec(Path({str(helper)!r}).read_text())",
                       f"spec=importlib.util.spec_from_file_location('unregistered',{str(helper)!r}); "
                       "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)"]
        for expression in expressions:
            with self.subTest(expression=expression):
                io, _, _ = self.check_expression(expression, imports="import json\nimport importlib.util\n")
                self.assertGreater(io["denied_count"], 0)

    def test_socket_dns_urllib_and_native_uri_access_is_denied_without_addresses_in_audit(self):
        expressions = ["socket.create_connection(('127.0.0.1', 9), timeout=.1)",
                       "socket.getaddrinfo('localhost', 9)",
                       "socket.socket().connect_ex(('127.0.0.1',9))",
                       "urllib.request.urlopen('http://127.0.0.1:9/private?credential=never-log', timeout=.1)",
                       "pq.read_table('hf://datasets/example/private.parquet')",
                       "fs.FileSystem.from_uri('s3://example/private.parquet')",
                       "fs.S3FileSystem()"]
        for expression in expressions:
            with self.subTest(expression=expression):
                io, _, _ = self.check_expression(expression, imports="import socket\nimport urllib.request\n")
                self.assertTrue(any(event["reason"] == "network_disabled" for event in io["events"]))
                text = json.dumps(io)
                self.assertNotIn("127.0.0.1", text)
                self.assertNotIn("credential=", text)

    def test_supplied_or_own_output_native_files_remain_usable(self):
        approved = self.output / "own.parquet"
        pd.DataFrame({"value": [7]}).to_parquet(approved)
        io, imported, _ = self.check_expression(
            f"assert pd.read_parquet({str(approved)!r}).value.iloc[0] == 7; "
            f"assert pq.read_table({str(approved)!r}).num_rows == 1; "
            f"assert ds.dataset({str(approved)!r}).to_table().num_rows == 1; "
            f"pa.OSFile({str(approved)!r}, 'r').close(); "
            f"pa.memory_map({str(approved)!r}).close(); "
            f"fs.LocalFileSystem().open_input_file({str(approved)!r}).close(); "
            f"fs.FileSystem.from_uri({str(approved)!r})[0].open_input_file({str(approved)!r}).close(); "
            f"pa.input_stream({str(approved)!r}).close()", fail=False)
        self.assertEqual(io["status"], "verified")
        self.assertEqual(imported["status"], "verified")

    def test_native_write_modes_cannot_modify_supplied_inputs(self):
        target = self.fixture.work / "train.csv"
        before = target.read_bytes()
        for expression in (f"pa.memory_map({str(target)!r}, 'r+')",
                           f"pa.OSFile({str(target)!r}, 'wb')",
                           f"pa.output_stream({str(target)!r})",
                           f"pa.MemoryMappedFile.create({str(target)!r}, 8)",
                           f"fs.LocalFileSystem().open_output_stream({str(target)!r})"):
            with self.subTest(expression=expression):
                io, _, _ = self.check_expression(expression)
                self.assertTrue(any(event["reason"] == "immutable_supplied_input" for event in io["events"]))
                self.assertEqual(target.read_bytes(), before)

    def test_declared_parquet_snapshot_and_supplied_csv_are_readable(self):
        sys.path.insert(0, str(fixture_module.PROJECT / "code"))
        from evaluation.assets import fingerprint_assets, snapshot_assets
        models = self.root / "models"
        models.mkdir()
        pd.DataFrame({"city_id": ["A"], "population": [100]}).to_parquet(models / "population.parquet")
        (self.root / "assets.json").write_text(json.dumps({"schema_version": 1, "files": [
            {"path": "models/population.parquet", "kind": "static_context"}]}))
        binding = fingerprint_assets(self.root, "assets.json", [1980])
        snapshot = snapshot_assets(self.root, binding, self.output, origin=1980)
        source = ("import os\nfrom pathlib import Path\nimport pyarrow.parquet as pq\nimport pyarrow.csv as pcsv\n"
                  "def forecast(train,targets,origin,seed):\n"
                  "    asset=Path(os.environ['PROTOSTAR_ASSET_ROOT'])/'models/population.parquet'\n"
                  "    assert pq.read_table(asset).column('population')[0].as_py()==100\n"
                  f"    assert pcsv.read_csv({str(self.fixture.work / 'train.csv')!r}).num_rows==1\n"
                  "    return targets.assign(predicted_value=1.)\n")
        io, imported, _ = self.fixture.run_worker(source, fail=False,
            extra_args=["--asset-root", snapshot["snapshot_root"], "--asset-binding", snapshot["binding_path"]])
        self.assertEqual(io["status"], "verified")
        self.assertEqual(imported["status"], "verified")

    def test_swallowed_native_and_network_errors_still_fail_finalization(self):
        source = ("import pandas as pd\nimport socket\n"
                  "def forecast(train,targets,origin,seed):\n"
                  f"    try:\n        pd.read_parquet({str(self.parquet)!r})\n"
                  "    except RuntimeError:\n        pass\n"
                  "    try:\n        socket.create_connection(('127.0.0.1',9))\n"
                  "    except RuntimeError:\n        pass\n"
                  "    return targets.assign(predicted_value=1.)\n")
        io, _, _ = self.fixture.run_worker(source)
        self.assertGreaterEqual(io["denied_count"], 2)


if __name__ == "__main__":
    unittest.main()
