"""Checks for Placeholder_v0.7 on synthetic data only (no weather data is read).

Runs every v0.3 and v0.1 check against v0.7, plus the member default.

Run from the repository root:
    uv run python -m unittest notebooks/Meet/test_placeholder_v0_7.py -v
"""
import inspect
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "code")]

import placeholder_v0_7 as model  # noqa: E402
import test_placeholder_v0_1 as base  # noqa: E402
import test_placeholder_v0_3 as previous  # noqa: E402

base.model = model
previous.model = model
from test_placeholder_v0_1 import Geography, SelectionRule  # noqa: E402,F401
from test_placeholder_v0_3 import Forecasts, Warming  # noqa: E402,F401


class Members(unittest.TestCase):
    def test_boosted_member_is_the_default(self):
        for function in (model.forecast, model.run_forecast):
            self.assertEqual(inspect.signature(function).parameters["members"].default, "gbm")

    def test_forecast_record_names_the_boosted_member(self):
        self.assertEqual(Forecasts.result[1]["members"], "gbm")


if __name__ == "__main__":
    unittest.main()
