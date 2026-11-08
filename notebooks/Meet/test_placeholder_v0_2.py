"""Checks for Placeholder_v0.2 on synthetic data only (no weather data is read).

v0.2 must differ from the sealed v0.1 adapter in exactly one setting, the training window. The tests prove that,
then run the whole v0.1 test suite against v0.2.

Run from the repository root:
    uv run python -m unittest notebooks/Meet/test_placeholder_v0_2.py -v
"""
import difflib
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "code")]

import placeholder_v0_2  # noqa: E402
import test_placeholder_v0_1 as base  # noqa: E402

base.model = placeholder_v0_2
from test_placeholder_v0_1 import Forecasts, Geography, InformationBoundary, SelectionRule  # noqa: E402,F401


class OneChangeFromV01(unittest.TestCase):
    def test_only_the_window_and_the_name_differ(self):
        read = lambda name: (HERE / name).read_text(encoding="utf-8").splitlines()
        changed = [line for line in difflib.ndiff(read("placeholder_v0_1.py"), read("placeholder_v0_2.py"))
                   if line[:2] in ("- ", "+ ")]
        self.assertEqual(sorted(changed), sorted([
            '- """Placeholder_v0.1: direct, pooled, multi-horizon shift regression with an inner-fold shrinkage gate.',
            '+ """Placeholder_v0.2: direct, pooled, multi-horizon shift regression with an inner-fold shrinkage gate.',
            '- MODEL_NAME = "Placeholder_v0.1"',
            '+ MODEL_NAME = "Placeholder_v0.2"',
            "- TRAIN_YEARS = 45",
            "+ TRAIN_YEARS = 30",
        ]))

    def test_window_is_thirty_years(self):
        self.assertEqual(placeholder_v0_2.TRAIN_YEARS, 30)
        self.assertEqual(placeholder_v0_2.MODEL_NAME, "Placeholder_v0.2")


if __name__ == "__main__":
    unittest.main()
