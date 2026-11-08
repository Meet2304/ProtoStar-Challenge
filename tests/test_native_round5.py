"""R1 native path checks; synthetic files only, no labels or network."""
from pathlib import Path
import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc

import test_runtime_audit as fixtures


class NativeRound5Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.WorkerRuntimeAuditTests()
        self.fixture.setUp()
        self.root, self.output = self.fixture.root, self.fixture.output
        self.parquet = self.root / "data/labels.parquet"
        self.orc = self.root / "data/labels.orc"
        self.values = pd.DataFrame({"city_id": ["A"], "year": [1981], "value": [99.]})
        self.values.to_parquet(self.parquet)
        orc.write_table(pa.Table.from_pandas(self.values, preserve_index=False), str(self.orc))

    def tearDown(self):
        self.fixture.tearDown()

    def worker(self, expression, *, fail=True):
        source = ("from pathlib import Path\nimport pandas as pd\nimport pyarrow as pa\n"
                  "import pyarrow.parquet as pq\nimport pyarrow.parquet.core as pc\n"
                  "import pyarrow.orc as orc\nimport pyarrow.fs as fs\nimport pyarrow.dataset as ds\n"
                  "import pyarrow.csv as csv\n"
                  "def forecast(train,targets,origin,seed):\n"
                  f"    {expression}\n"
                  "    return targets.assign(predicted_value=1.)\n")
        return self.fixture.run_worker(source, fail=fail)

    def test_parquet_dataset_public_core_alias_and_orc_reads_are_blocked(self):
        for expression in (f"pq.ParquetDataset({str(self.parquet)!r}).read()",
                           f"pc.ParquetDataset({str(self.parquet)!r}).read()",
                           f"pc.read_table({str(self.parquet)!r})",
                           f"pc.ParquetFile({str(self.parquet)!r}).read()",
                           f"orc.ORCFile({str(self.orc)!r}).read()",
                           f"orc.read_table({str(self.orc)!r})",
                           f"pd.read_orc({str(self.orc)!r})"):
            with self.subTest(expression=expression):
                io, _, _ = self.worker(expression)
                self.assertGreater(io["denied_count"], 0)

    def test_subtree_native_opens_and_fragment_paths_are_blocked(self):
        for expression in (
            f"fs.SubTreeFileSystem({str(self.parquet.parent)!r},fs.LocalFileSystem()).open_input_file('labels.parquet')",
            f"pq.read_table('labels.parquet',filesystem=fs.SubTreeFileSystem({str(self.parquet.parent)!r},fs.LocalFileSystem()))",
            f"ds.ParquetFileFormat().make_fragment({str(self.parquet)!r},fs.LocalFileSystem()).to_table()",
            f"ds.ParquetFileFormat().make_fragment('labels.parquet',fs.SubTreeFileSystem({str(self.parquet.parent)!r},fs.LocalFileSystem())).to_table()",
        ):
            with self.subTest(expression=expression):
                io, _, _ = self.worker(expression)
                self.assertGreater(io["denied_count"], 0)

    def test_native_writer_constructors_do_not_plant_protected_files(self):
        table = "t=pa.table({'a':[1]}); "
        for name, expression in (
            ("planted.parquet", "pq.ParquetWriter({path},t.schema)"),
            ("core.parquet", "pc.ParquetWriter({path},t.schema)"),
            ("planted.arrow", "pa.ipc.new_file({path},t.schema)"),
            ("planted.stream", "pa.ipc.new_stream({path},t.schema)"),
            ("writer.arrow", "pa.ipc.RecordBatchFileWriter({path},t.schema)"),
            ("writer.stream", "pa.ipc.RecordBatchStreamWriter({path},t.schema)"),
            ("planted.csv", "csv.CSVWriter({path},t.schema)"),
            ("planted.orc", "orc.ORCWriter({path})"),
        ):
            path = self.root / "data" / name
            with self.subTest(name=name):
                io, _, _ = self.worker(table + expression.format(path=repr(str(path))))
                self.assertGreater(io["denied_count"], 0)
                self.assertFalse(path.exists())

    def test_own_output_readers_writers_subtrees_and_fragments_work(self):
        base = str(self.output)
        expression = (
            f"base=Path({base!r}); t=pa.table({{'a':[1]}}); "
            "pq.write_table(t,base/'own.parquet'); "
            "assert pq.ParquetDataset(base/'own.parquet').read().num_rows==1; "
            "assert pc.read_table(base/'own.parquet').num_rows==1; "
            "orc.write_table(t,str(base/'own.orc')); assert orc.ORCFile(str(base/'own.orc')).read().num_rows==1; "
            "sub=fs.SubTreeFileSystem(str(base),fs.LocalFileSystem()); "
            "assert isinstance(sub,fs.SubTreeFileSystem); assert Path(sub.base_path)==base; "
            "assert isinstance(sub.base_fs,fs.FileSystem); "
            "assert sub.open_input_file('own.parquet').size()>0; "
            "assert pq.read_table('own.parquet',filesystem=sub).num_rows==1; "
            "assert ds.ParquetFileFormat().make_fragment('own.parquet',sub).to_table().num_rows==1; "
            "w=csv.CSVWriter(str(base/'own.csv'),t.schema);w.write_table(t);w.close(); "
            "w=pa.ipc.new_file(str(base/'own.arrow'),t.schema);w.write_table(t);w.close(); "
            "w=pq.ParquetWriter(str(base/'writer.parquet'),t.schema);w.write_table(t);w.close(); "
            "fs.LocalFileSystem().create_dir(str(base/'scratch')); fs.LocalFileSystem().delete_dir(str(base/'scratch'))"
        )
        io, imports, _ = self.worker(expression, fail=False)
        self.assertEqual(io["status"], "verified")
        self.assertEqual(imports["status"], "verified")


if __name__ == "__main__":
    unittest.main()
