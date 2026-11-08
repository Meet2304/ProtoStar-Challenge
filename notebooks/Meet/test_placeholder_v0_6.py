"""Checks for Placeholder_v0.6 on synthetic data only (no weather data is read).

Run from the repository root:
    uv run python -m unittest notebooks/Meet/test_placeholder_v0_6.py -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "code")]

import placeholder_v0_5 as v05  # noqa: E402
import placeholder_v0_6 as model  # noqa: E402
from common import METRICS  # noqa: E402
from test_placeholder_v0_1 import CITIES, COORDS, ORIGIN, grid, neighbours, synthetic  # noqa: E402


class PhysicsResponse(unittest.TestCase):
    def setUp(self):
        self.tables = model.tables_at(synthetic(trend=0.15), CITIES, ORIGIN)
        self.per = {m: model.metric_state(table, 1960) for m, table in self.tables.items()}

    def tearDown(self):
        model._DAYS_GE_65 = None

    def test_without_a_table_it_is_v05(self):
        a = model.warming_rates(self.per, self.tables, 1960, neighbours())
        b = v05.warming_rates(self.per, self.tables, 1960, neighbours())
        for metric in METRICS:
            np.testing.assert_array_equal(a[metric], b[metric])

    def test_cdd_rate_is_temperature_rate_times_days_ge_65(self):
        days = pd.DataFrame({1960: np.full(len(CITIES), 100.0)}, index=CITIES)
        model._DAYS_GE_65 = days
        rates = model.warming_rates(self.per, self.tables, 1960, neighbours())
        np.testing.assert_allclose(rates["cdd"], rates["annual_avg_temp"] * 100.0)

    def test_missing_values_fall_back_to_the_regression_response(self):
        values = np.full(len(CITIES), 100.0)
        values[0] = np.nan
        model._DAYS_GE_65 = pd.DataFrame({1960: values}, index=CITIES)
        rates = model.warming_rates(self.per, self.tables, 1960, neighbours())
        fallback = v05.warming_rates(self.per, self.tables, 1960, neighbours())
        self.assertEqual(rates["cdd"][0], fallback["cdd"][0])

    def test_other_years_are_untouched(self):
        model._DAYS_GE_65 = pd.DataFrame({1950: np.full(len(CITIES), 100.0)}, index=CITIES)
        a = model.warming_rates(self.per, self.tables, 1960, neighbours())
        b = v05.warming_rates(self.per, self.tables, 1960, neighbours())
        np.testing.assert_array_equal(a["cdd"], b["cdd"])

    def test_forecast_runs_with_a_table(self):
        model._DAYS_GE_65 = pd.DataFrame({y: np.full(len(CITIES), 80.0) for y in range(1940, ORIGIN + 1)}, index=CITIES)
        output = model.run_forecast(synthetic(trend=0.15), grid(), ORIGIN, 42, COORDS)[0]
        self.assertTrue(np.isfinite(output.predicted_value).all())


if __name__ == "__main__":
    unittest.main()
