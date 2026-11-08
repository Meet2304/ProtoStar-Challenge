"""The committed enhanced-input manifests stay origin-local and point at canonical release paths."""
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from evaluation.assets import fingerprint_assets  # noqa: E402

MANIFESTS = {"enhance-v2": ROOT / "evaluation/enhance-v2-assets.json",
             "enhance-v3": ROOT / "evaluation/enhance-v3-assets.json"}


class EnhancedManifestTests(unittest.TestCase):
    def test_weather_inputs_are_scoped_to_their_own_origin(self):
        for release, path in MANIFESTS.items():
            for entry in json.loads(path.read_text(encoding="utf-8"))["files"]:
                with self.subTest(release=release, path=entry["path"]):
                    self.assertTrue(entry["path"].startswith(f"data/processed/releases/{release}/"))
                    self.assertNotIn("/sources/", entry["path"])          # full uncut series never bind
                    if entry["kind"] == "weather_input":
                        origin = int(re.search(r"/origin-(\d{4})/", entry["path"]).group(1))
                        self.assertEqual(entry["origins"], [origin])
                        self.assertEqual(entry["information_end_year"], origin)

    def test_v3_declares_forcing_for_every_harness_origin(self):
        entries = json.loads(MANIFESTS["enhance-v3"].read_text(encoding="utf-8"))["files"]
        forcing = {e["origins"][0] for e in entries if "/forcing/" in e["path"]}
        self.assertEqual(forcing, {1983, 1990, 1997, 2004, 2011})

    def test_v3_binds_when_the_release_is_pulled(self):
        if not (ROOT / "data/processed/releases/enhance-v3/release.json").is_file():
            self.skipTest("enhance-v3 not pulled (line.py pull enhance-v3)")
        binding = fingerprint_assets(ROOT, "evaluation/enhance-v3-assets.json", [1983, 1990, 1997])
        for origin, part in binding["per_origin"].items():
            self.assertEqual(len(part["files"]), 4, origin)    # static + ocean annual/monthly + forcing


if __name__ == "__main__":
    unittest.main()
