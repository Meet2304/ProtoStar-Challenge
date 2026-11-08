"""Checks for Placeholder_v0.4 on synthetic data only (no weather data is read).

Runs every v0.3 and v0.1 check against v0.4, plus the gate change.

Run from the repository root:
    uv run python -m unittest notebooks/Meet/test_placeholder_v0_4.py -v
"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "code")]

import placeholder_v0_4 as model  # noqa: E402
import test_placeholder_v0_1 as base  # noqa: E402
import test_placeholder_v0_3 as previous  # noqa: E402
from common import METRICS  # noqa: E402

base.model = model
previous.model = model
from test_placeholder_v0_1 import Geography, SelectionRule  # noqa: E402,F401
from test_placeholder_v0_3 import Forecasts, Warming  # noqa: E402,F401


class Gate(unittest.TestCase):
    def test_fifteen_consecutive_inner_origins(self):
        self.assertEqual(model.INNER_OFFSETS, tuple(range(21, 6, -1)))
        self.assertEqual(len(model.INNER_OFFSETS), 15)

    def test_groups_cover_every_metric_once(self):
        flat = [metric for group in model.LAMBDA_GROUPS for metric in group]
        self.assertEqual(sorted(flat), sorted(METRICS))

    def test_temperature_metrics_share_one_lambda(self):
        record = Forecasts.result[1]
        if record.get("status") != "selected":
            self.skipTest("gate fell back")
        selected = record["selected"]
        self.assertEqual(len({selected[m] for m in model.LAMBDA_GROUPS[0]}), 1)
        self.assertEqual(len(record["inner_folds"]), 15)


if __name__ == "__main__":
    unittest.main()
