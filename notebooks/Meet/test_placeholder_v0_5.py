"""Checks for Placeholder_v0.5 on synthetic data only (no weather data is read).

Run from the repository root:
    uv run python -m unittest notebooks/Meet/test_placeholder_v0_5.py -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "code")]

import placeholder_v0_3 as v03  # noqa: E402
import placeholder_v0_5 as model  # noqa: E402
from evaluation.adapters import reference  # noqa: E402
from test_placeholder_v0_1 import COORDS, N_CITIES, ORIGIN, grid, synthetic  # noqa: E402


class Blend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = synthetic(trend=0.15)
        cls.output = model.run_forecast(cls.train, grid(), ORIGIN, 42, COORDS)[0]
        cls.v03 = v03.run_forecast(cls.train, grid(), ORIGIN, 42, COORDS)[0]
        cls.combo = reference(cls.train, grid(), ORIGIN, 42, method="combo")

    def test_complete_finite_grid(self):
        self.assertEqual(len(self.output), N_CITIES * 7 * 4)
        self.assertTrue(np.isfinite(self.output.predicted_value).all())

    def test_is_the_fixed_average_of_v03_and_combo(self):
        rows = (self.output.merge(self.v03, on=model.KEY, suffixes=("", "_v03"), validate="one_to_one")
                .merge(self.combo[model.KEY + ["predicted_value"]].rename(columns={"predicted_value": "combo"}),
                       on=model.KEY, validate="one_to_one"))
        np.testing.assert_allclose(rows.predicted_value, 0.5 * rows.predicted_value_v03 + 0.5 * rows.combo)

    def test_missing_combo_rows_are_refused(self):
        with self.assertRaises(ValueError):
            model.blend_with_combo(self.v03, self.combo.iloc[1:])


if __name__ == "__main__":
    unittest.main()
