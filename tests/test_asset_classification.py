"""Cheap classification guards distinguish labels from structural context."""
import json
import gzip
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from evaluation import assets


class AssetClassificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def declare(self, name, kind="static_context", **metadata):
        declaration = {"path": name, "kind": kind, **metadata}
        (self.root / "assets.json").write_text(json.dumps({"schema_version": 1, "files": [declaration]}))
        return assets.fingerprint_assets(self.root, "assets.json", [1983])

    def put(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_annual_label_csv_cannot_be_mislabeled_static_configuration_or_source(self):
        self.put("models/annual.csv", "ID,Year,Avg Temp,ToT Prcp,Max Tmax,cdd\nA,1980,55,30,90,100\n")
        for kind in ("static_context", "configuration", "source"):
            with self.subTest(kind=kind), self.assertRaisesRegex(assets.AssetError, "requires weather_input"):
                self.declare("models/annual.csv", kind)

    def test_long_actual_value_table_and_parquet_weather_schema_are_rejected(self):
        self.put("labels.csv", "city_id,year,metric,actual_value\nA,1980,cdd,100\n")
        with self.assertRaisesRegex(assets.AssetError, "requires weather_input"):
            self.declare("labels.csv")
        pd.DataFrame({"date": ["1980-01-01"], "tmax": [50.0]}).to_parquet(self.root / "renamed.parquet")
        with self.assertRaisesRegex(assets.AssetError, "requires weather_input"):
            self.declare("renamed.parquet")

    def test_structural_census_and_city_info_with_year_columns_are_allowed(self):
        for name, csv in (("census.csv", "city_id,year,population\nA,1980,1234\n"),
                          ("city_info.csv", "city_id,year,station,elevation\nA,1980,S,200\n")):
            self.put(name, csv)
            with self.subTest(name=name):
                self.assertEqual(self.declare(name)["files"][name]["kind"], "static_context")

    def test_known_paths_rejected_even_when_schema_is_unrecognizable(self):
        for name in ("data/raw/city_info.csv", "data/annual_valid.csv"):
            self.put(name, "x\n1\n")
            with self.subTest(name=name), self.assertRaisesRegex(assets.AssetError, "known weather/label"):
                self.declare(name)

    def test_renamed_pinned_annual_hash_rejected_without_parsing_file(self):
        self.put("renamed.bin", "synthetic placeholder; not real observations")
        original = assets._stream_fingerprint
        def fingerprint(path):
            result = original(path)
            if path.name == "renamed.bin":
                result["sha256"] = assets.PINNED_ANNUAL_SHA256
            return result
        with patch.object(assets, "_stream_fingerprint", side_effect=fingerprint):
            with self.assertRaisesRegex(assets.AssetError, "known weather/label"):
                self.declare("renamed.bin")

    def test_weather_classification_keeps_actual_timestamp_validation(self):
        self.put("weather.csv", "year,tmax\n1980,50\n1984,60\n")
        with self.assertRaisesRegex(assets.AssetError, "exceeds information/origin bound"):
            self.declare("weather.csv", "weather_input", information_end_year=1983,
                         time_column="year", time_format="year")

    def test_metadata_rejection_precedes_snapshot_creation(self):
        self.put("labels.csv", "year,actual_value\n1980,3\n")
        with self.assertRaises(assets.AssetError):
            self.declare("labels.csv")
        self.assertFalse((self.root / "run/declared_assets").exists())

    def test_wide_year_and_generic_long_outcome_tables_are_rejected(self):
        for name, content in (("wide.csv", "city_id,1980,1981\nA,55,56\n"),
                              ("long.csv", "city_id,year,metric,value\nA,1980,cdd,100\n")):
            self.put(name, content)
            with self.subTest(name=name), self.assertRaisesRegex(assets.AssetError, "requires weather_input"):
                self.declare(name)

    def test_compressed_tsv_and_feather_weather_schemas_are_rejected(self):
        self.put("weather.tsv", "year\ttmax\n1980\t50\n")
        with gzip.open(self.root / "weather.csv.gz", "wt", encoding="utf-8") as stream:
            stream.write("year,tmax\n1980,50\n")
        pd.DataFrame({"year": [1980], "tmax": [50.]}).to_feather(self.root / "weather.feather")
        for name in ("weather.tsv", "weather.csv.gz", "weather.feather"):
            with self.subTest(name=name), self.assertRaisesRegex(assets.AssetError, "requires weather_input"):
                self.declare(name)


if __name__ == "__main__":
    unittest.main()
