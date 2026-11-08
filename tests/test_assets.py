"""Offline synthetic/local-Git tests for the standalone asset-binding component."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "code"))
from evaluation import assets


class AssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "project"
        self.root.mkdir()
        self.run = Path(self.temp.name) / "run"
        self.config = {"path": "models/settings.json", "kind": "configuration"}
        self.weights = {"path": "models/weights.bin", "kind": "model_weights", "information_end_year": 1980}
        self.put("models/settings.json", b'{"degree": 2, "regularization": 0.1}\n')
        self.put("models/weights.bin", b"synthetic-weights\x00\x01" * 1000)

    def tearDown(self):
        self.temp.cleanup()

    def put(self, name, raw):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    def manifest(self, declarations=None, name="assets.json", **extras):
        document = {"schema_version": 1, "files": declarations if declarations is not None else [self.config, self.weights], **extras}
        self.put(name, (json.dumps(document, indent=2) + "\n").encode())
        return name

    def fingerprint(self, origins=None, **kwargs):
        return assets.fingerprint_assets(self.root, "assets.json", origins or [1983, 1990], **kwargs)

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, stderr=subprocess.STDOUT).decode().strip()

    def commit(self):
        if not (self.root / ".git").exists():
            self.git("init", "-q")
            self.git("config", "user.name", "Asset Test")
            self.git("config", "user.email", "asset-test@example.invalid")
            self.git("config", "core.autocrlf", "false")
        self.git("add", "--all")
        self.git("commit", "-qm", "Synthetic asset fixture")
        return self.git("rev-parse", "HEAD")

    def test_binding_is_deterministic_includes_roles_sizes_scopes_and_manifest(self):
        self.manifest()
        first = self.fingerprint()
        second = self.fingerprint(origins=[1990, 1983])
        self.assertEqual(first, second)
        pin = first["files"]["models/weights.bin"]
        self.assertEqual(pin["sha256"], hashlib.sha256((self.root / "models/weights.bin").read_bytes()).hexdigest())
        self.assertEqual(pin["size"], (self.root / "models/weights.bin").stat().st_size)
        self.assertEqual(pin["kind"], "model_weights")
        self.assertEqual(pin["information_end_year"], 1980)
        self.assertEqual(pin["applicable_origins"], [1983, 1990])
        self.assertEqual(first["manifest"]["path"], "assets.json")
        self.assertNotEqual(first["per_origin"]["1983"]["fingerprint"], first["per_origin"]["1990"]["fingerprint"])

    def test_weight_content_change_invalidates_freeze_binding(self):
        self.manifest()
        frozen = self.fingerprint()
        self.put("models/weights.bin", b"different synthetic weights")
        with self.assertRaisesRegex(assets.AssetError, "changed after binding"):
            assets.verify_asset_binding(self.root, frozen)

    def test_kind_cutoff_and_scope_metadata_changes_alter_identity(self):
        self.manifest()
        original = self.fingerprint()
        changes = [dict(self.config, kind="static_context"), dict(self.weights, information_end_year=1981),
                   dict(self.weights, origins=[1983])]
        for changed in changes:
            declarations = [changed, self.weights] if changed["path"] == self.config["path"] else [self.config, changed]
            self.manifest(declarations)
            self.assertNotEqual(original["binding_id"], self.fingerprint()["binding_id"])
            with self.assertRaises(assets.AssetError):
                assets.verify_asset_binding(self.root, original)

    def test_declared_information_cutoff_must_fit_every_active_origin(self):
        self.manifest([dict(self.weights, information_end_year=1985)])
        with self.assertRaisesRegex(assets.AssetError, "applicable origin"):
            self.fingerprint()
        self.manifest([dict(self.weights, information_end_year=1985, origins=[1990])])
        self.assertEqual(self.fingerprint()["files"]["models/weights.bin"]["applicable_origins"], [1990])

    def test_inactive_future_assets_are_never_accessed_and_can_be_missing(self):
        future = {"path": "models/future.csv", "kind": "weather_input", "origins": [1990],
                  "information_end_year": 1989, "time_column": "date", "time_format": "iso_date"}
        self.manifest([self.config, future])
        commit = self.commit()
        # No future.csv is created: an earlier worker must not require it.
        earlier = self.fingerprint(origins=[1983], commit=commit)
        self.assertEqual(set(earlier["files"]), {"models/settings.json"})
        with self.assertRaisesRegex(assets.AssetError, "missing"):
            self.fingerprint(origins=[1990])
        self.put("models/future.csv", b"date,value\n1989-01-01,99\n")
        phase = self.fingerprint()
        with patch.object(assets, "_validate_weather", side_effect=AssertionError("inactive weather was read")):
            snapshot = assets.snapshot_assets(self.root, phase, self.run, origin=1983)
        snap_root = Path(snapshot["snapshot_root"])
        self.assertFalse((snap_root / "models/future.csv").exists())
        self.assertEqual(assets.verify_asset_binding(snap_root, snapshot["binding"]), earlier)

    def test_projection_matches_direct_fingerprint_and_binds_full_manifest(self):
        self.manifest([self.config, dict(self.weights, origins=[1990])])
        full = self.fingerprint()
        projected = assets.project_asset_binding(full, [1983])
        self.assertEqual(projected, self.fingerprint(origins=[1983]))
        self.assertEqual(set(projected["files"]), {"models/settings.json"})
        self.manifest([self.config, dict(self.weights, origins=[1990], information_end_year=1981)])
        with self.assertRaises(assets.AssetError):
            assets.verify_asset_binding(self.root, projected)

    def test_weather_year_csv_verified_against_metadata_and_origin(self):
        weather = {"path": "weather.csv", "kind": "weather_input", "information_end_year": 1980,
                   "time_column": "Year", "time_format": "year"}
        self.manifest([weather])
        self.put("weather.csv", b"Year,value\n1979,1\n1980,2\n")
        pin = self.fingerprint()["files"]["weather.csv"]
        self.assertEqual(pin["weather_validation"]["max_year"], 1980)
        self.assertEqual(pin["weather_validation"]["rows_checked"], 2)
        self.put("weather.csv", b"Year,value\n1981,1\n")
        with self.assertRaisesRegex(assets.AssetError, "bound 1980"):
            self.fingerprint()
        # Synthetic calendar rows only, never actual weather observations.
        self.put("weather.csv", b"Year,value\n2019,1\n")
        with self.assertRaisesRegex(assets.AssetError, "weather time validation failed"):
            self.fingerprint()

    def test_weather_dates_and_iso_datetimes_are_explicit_and_validated(self):
        entry = {"path": "weather.csv", "kind": "weather_input", "information_end_year": 1980,
                 "time_column": "date", "time_format": "iso_date"}
        self.manifest([entry])
        self.put("weather.csv", b"date,value\n1980-02-29,2\n")
        self.fingerprint()
        for invalid in ("1980-02-30", "1980/02/29", "1980-01-01T00:00:00", "", "2019-01-01"):
            self.put("weather.csv", f"date,value\n{invalid},2\n".encode())
            with self.subTest(invalid=invalid), self.assertRaises(assets.AssetError):
                self.fingerprint()
        self.manifest([dict(entry, time_format="iso_datetime")])
        for valid in ("1980-02-29T15:00:00", "1980-02-29T15:00:00.123456Z", "1980-12-31T23:59:59-05:00"):
            self.put("weather.csv", f"date,value\n{valid},2\n".encode())
            self.fingerprint()
        self.put("weather.csv", b"date,value\n1980-01-01 00:00:00,1\n")
        with self.assertRaises(assets.AssetError):
            self.fingerprint()

    def test_weather_parquet_column_values_are_verified(self):
        entry = {"path": "weather.parquet", "kind": "weather_input", "information_end_year": 1980,
                 "time_column": "observed_at", "time_format": "iso_datetime"}
        self.manifest([entry])
        pd.DataFrame({"observed_at": pd.to_datetime(["1980-01-01", "1980-12-31"]), "x": [1., 2.]}).to_parquet(self.root / "weather.parquet")
        self.assertEqual(self.fingerprint()["files"]["weather.parquet"]["weather_validation"]["rows_checked"], 2)
        pd.DataFrame({"observed_at": pd.to_datetime(["2019-01-01"]), "x": [1.]}).to_parquet(self.root / "weather.parquet")
        with self.assertRaises(assets.AssetError):
            self.fingerprint()

    def test_weather_missing_ambiguous_invalid_or_empty_time_columns_fail(self):
        entry = {"path": "weather.csv", "kind": "weather_input", "information_end_year": 1980,
                 "time_column": "Year", "time_format": "year"}
        self.manifest([entry])
        for raw in (b"Year,Year\n1980,2019\n", b"Other,value\n1980,1\n", b"Year,value\n",
                    b"Year,value\n,1\n", b"Year,value\n1980.0,1\n", b"Year,value\nnot-a-year,1\n"):
            self.put("weather.csv", raw)
            with self.subTest(raw=raw), self.assertRaises(assets.AssetError):
                self.fingerprint()
        self.manifest([dict(entry, time_format=[])])
        with self.assertRaises(assets.AssetError):
            self.fingerprint()

    def test_empty_explicit_manifest_and_optional_source_role(self):
        self.manifest([])
        binding = self.fingerprint()
        self.assertEqual(binding["files"], {})
        self.put("models/helper.py", b"FACTOR = 2\n")
        self.manifest([{"path": "models/helper.py", "kind": "source"}])
        binding = self.fingerprint()
        self.assertEqual(binding["files"]["models/helper.py"]["kind"], "source")
        self.manifest([dict(self.config, path="models")])
        with self.assertRaisesRegex(assets.AssetError, "ordinary"):
            self.fingerprint()

    def test_unsafe_and_credential_looking_paths_fail_without_creating_files(self):
        forbidden = [".env", ".env.local", "models/client_secret.json", "models/api-key.json", "keys/private.key",
                     "token.json", "../outside.bin", "/absolute.json", "C:/absolute.json", "models\\weights.bin",
                     "./weights.bin", "models//weights.bin", "models/../weights.bin", "models/NUL.txt", "models/trailing."]
        for name in forbidden:
            self.manifest([{**self.config, "path": name}])
            with self.subTest(name=name), self.assertRaises(assets.AssetError):
                self.fingerprint()
            if name.startswith(".env") or name in {"token.json", "models/client_secret.json", "models/api-key.json", "keys/private.key"}:
                self.assertFalse((self.root / name).is_file())

    def test_duplicate_case_colliding_and_file_directory_declarations_fail(self):
        for entries in ([self.config, self.config], [self.config, dict(self.config, path="MODELS/SETTINGS.JSON")],
                        [self.config, dict(self.config, path="models/settings.json/child.json")],
                        [dict(self.config, path="assets.json")]):
            self.manifest(entries)
            with self.assertRaisesRegex(assets.AssetError, "colliding"):
                self.fingerprint()

    def test_unknown_fields_kinds_scopes_and_ambiguous_metadata_fail(self):
        cases = [[dict(self.config, unexpected=True)], [dict(self.config, kind="learned_magic")],
                 [dict(self.config, origins=[])], [dict(self.config, origins=[1983, 1983])],
                 [dict(self.config, origins=[True])], [dict(self.config, information_end_year=1980)],
                 [dict(self.weights, information_end_year=None)], [dict(self.weights, information_end_year=2019)],
                 [dict(self.config, time_column="year")]]
        for entries in cases:
            self.manifest(entries)
            with self.subTest(entries=entries), self.assertRaises(assets.AssetError):
                self.fingerprint()
        self.manifest(extra=True)
        with self.assertRaises(assets.AssetError):
            self.fingerprint()

    def test_duplicate_json_keys_fail(self):
        self.put("assets.json", b'{"schema_version":1,"schema_version":1,"files":[]}')
        with self.assertRaisesRegex(assets.AssetError, "duplicate JSON key"):
            self.fingerprint()

    def test_symlink_escape_and_nonordinary_file_rejected(self):
        outside = Path(self.temp.name) / "outside.bin"
        outside.write_bytes(b"synthetic outside bytes")
        link = self.root / "linked.bin"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("platform does not permit creating symlinks")
        self.manifest([dict(self.weights, path="linked.bin")])
        with self.assertRaisesRegex(assets.AssetError, "symlink"):
            self.fingerprint()

    def test_git_committed_asset_and_manifest_must_match(self):
        self.manifest()
        commit = self.commit()
        binding = self.fingerprint(commit=commit)
        assets.verify_asset_binding(self.root, binding, commit=commit)
        self.put("models/settings.json", b'{"degree": 7}\n')
        with self.assertRaisesRegex(assets.AssetError, "frozen Git"):
            self.fingerprint(commit=commit)
        self.put("models/settings.json", b'{"degree": 2, "regularization": 0.1}\n')
        self.manifest([dict(self.config, kind="static_context"), self.weights])
        with self.assertRaisesRegex(assets.AssetError, "frozen Git"):
            self.fingerprint(commit=commit)

    def test_windows_git_crlf_text_equivalence_preserves_raw_freeze_bytes(self):
        self.git("init", "-q")
        self.git("config", "user.name", "Asset Test")
        self.git("config", "user.email", "asset-test@example.invalid")
        self.git("config", "core.autocrlf", "true")
        self.manifest([self.config])
        for name in ("assets.json", self.config["path"]):
            path = self.root / name
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        commit = self.commit()
        binding = self.fingerprint(commit=commit)
        raw = (self.root / self.config["path"]).read_bytes()
        self.assertIn(b"\r\n", raw)
        self.assertEqual(binding["files"][self.config["path"]]["sha256"], hashlib.sha256(raw).hexdigest())
        self.put(self.config["path"], raw.replace(b"\r\n", b"\n"))
        self.fingerprint(commit=commit)  # still matches committed logical text
        with self.assertRaisesRegex(assets.AssetError, "changed after binding"):
            assets.verify_asset_binding(self.root, binding, commit=commit)

    def test_learned_and_binary_asset_bytes_never_get_text_normalization(self):
        self.git("init", "-q")
        self.git("config", "user.name", "Asset Test")
        self.git("config", "user.email", "asset-test@example.invalid")
        self.git("config", "core.autocrlf", "true")
        learned = dict(self.weights, path="models/learned.json")
        binary = dict(self.config, path="models/blob.bin")
        self.put(learned["path"], b'{"synthetic_weight": 0.1}\r\n')
        self.put(binary["path"], b"synthetic binary extension\r\n")
        self.manifest([learned])
        commit = self.commit()
        with self.assertRaisesRegex(assets.AssetError, "frozen Git"):
            self.fingerprint(commit=commit)
        self.manifest([binary])
        self.git("add", "assets.json")
        self.git("commit", "-qm", "Synthetic binary asset role")
        with self.assertRaisesRegex(assets.AssetError, "frozen Git"):
            self.fingerprint(commit=self.git("rev-parse", "HEAD"))

    def test_untracked_asset_and_wrong_commit_fail(self):
        self.manifest([self.config])
        self.commit()
        self.put("models/new.bin", b"untracked synthetic weights")
        self.manifest([self.config, dict(self.weights, path="models/new.bin")])
        self.git("add", "assets.json")
        self.git("commit", "-qm", "Declare missing-from-commit asset")
        commit = self.git("rev-parse", "HEAD")
        with self.assertRaisesRegex(assets.AssetError, "ordinary file tracked"):
            self.fingerprint(commit=commit)
        with self.assertRaises(assets.AssetError):
            self.fingerprint(commit="HEAD")

    def test_git_lfs_materialized_content_matches_oid_and_size_without_fetch(self):
        payload = b"fake materialized binary weights\x00" * 500
        oid = hashlib.sha256(payload).hexdigest()
        pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {len(payload)}\n".encode()
        self.put("models/weights.bin", pointer)
        self.manifest([self.weights])
        commit = self.commit()
        self.put("models/weights.bin", payload)
        binding = self.fingerprint(commit=commit)
        self.assertEqual(binding["files"]["models/weights.bin"]["sha256"], oid)
        self.put("models/weights.bin", payload + b"changed")
        with self.assertRaisesRegex(assets.AssetError, "LFS oid/size"):
            self.fingerprint(commit=commit)

    def test_lfs_wrong_oid_and_size_and_unmaterialized_pointer_fail(self):
        payload = b"small fake binary weight"
        for oid, size in (("0" * 64, len(payload)), (hashlib.sha256(payload).hexdigest(), len(payload) + 1)):
            pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n".encode()
            self.put("models/weights.bin", pointer)
            self.manifest([self.weights])
            commit = self.commit()
            with self.assertRaisesRegex(assets.AssetError, "LFS oid/size"):
                self.fingerprint(commit=commit)
            self.put("models/weights.bin", payload)
            with self.assertRaisesRegex(assets.AssetError, "LFS oid/size"):
                self.fingerprint(commit=commit)

    def test_snapshot_is_complete_reverifiable_streamed_and_immutable(self):
        self.manifest()
        commit = self.commit()
        binding = self.fingerprint(commit=commit)
        # Weight hashing/copying must not call read_bytes (manifest reads are small).
        original = Path.read_bytes
        def guarded(path):
            if path.name == "weights.bin":
                raise AssertionError("large asset loaded eagerly")
            return original(path)
        with patch.object(Path, "read_bytes", guarded):
            snapshot = assets.snapshot_assets(self.root, binding, self.run, origin=1983, commit=commit)
        copied = Path(snapshot["snapshot_root"]) / "models/weights.bin"
        self.assertEqual(copied.read_bytes(), (self.root / "models/weights.bin").read_bytes())
        assets.verify_asset_binding(snapshot["snapshot_root"], snapshot["binding"])
        with self.assertRaisesRegex(assets.AssetError, "immutable"):
            assets.snapshot_assets(self.root, binding, self.run)

    def test_output_source_overlap_and_tampered_binding_fail(self):
        self.put("nested/declared_assets/weights.bin", b"fake overlapping weights")
        self.manifest([dict(self.weights, path="nested/declared_assets/weights.bin")])
        binding = self.fingerprint()
        with self.assertRaisesRegex(assets.AssetError, "immutable|overlap"):
            assets.snapshot_assets(self.root, binding, self.root / "nested")
        binding["files"]["nested/declared_assets/weights.bin"]["size"] = 1
        with self.assertRaisesRegex(assets.AssetError, "identity"):
            assets.verify_asset_binding(self.root, binding)


if __name__ == "__main__":
    unittest.main()
