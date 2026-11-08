"""Checks for Placeholder_v0.8 on synthetic data only (no weather data is read).

Run from the repository root:
    uv run python -m unittest notebooks/Meet/test_placeholder_v0_8.py -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "code")]

import placeholder_v0_8 as model  # noqa: E402
from test_placeholder_v0_1 import CITIES, COORDS, N_CITIES, ORIGIN, grid, neighbours, synthetic  # noqa: E402


def station_tables(first=1940):
    years = range(first, ORIGIN + 1)
    rng = np.random.default_rng(0)
    return {name: pd.DataFrame({y: rng.integers(0, 5, len(CITIES)).astype(float) for y in years}, index=CITIES)
            for name in model.STATION_FEATURES}


class Station(unittest.TestCase):
    def tearDown(self):
        model._STATION = None

    def test_features_are_in_the_matrix(self):
        names = model.feature_names("all")
        for name in model.STATION_FEATURES:
            self.assertIn(name, names)
            self.assertIn(name, model.LEVEL_FEATURES)

    def test_state_reads_only_the_as_of_year(self):
        model._STATION = station_tables()
        values = model.station_state(1960, N_CITIES)
        for name in model.STATION_FEATURES:
            np.testing.assert_array_equal(values[name], model._STATION[name][1960].to_numpy(float))

    def test_unknown_years_are_missing(self):
        model._STATION = station_tables(first=1950)
        self.assertTrue(all(np.isnan(v).all() for v in model.station_state(1945, N_CITIES).values()))

    def test_forecast_runs_with_and_without_a_table(self):
        train = synthetic(trend=0.15)
        without = model.run_forecast(train, grid(), ORIGIN, 42, COORDS)[0]
        model._STATION = station_tables()
        with_table = model.run_forecast(train, grid(), ORIGIN, 42, COORDS)[0]
        for output in (without, with_table):
            self.assertEqual(len(output), N_CITIES * 7 * 4)
            self.assertTrue(np.isfinite(output.predicted_value).all())

    def test_training_rows_have_the_new_columns(self):
        model._STATION = station_tables()
        tables = model.tables_at(synthetic(), CITIES, ORIGIN)
        X, _ = model.training_rows(tables, neighbours(), ORIGIN, "all")["cdd"]
        self.assertEqual(X.shape[1], len(model.feature_names("all")))


if __name__ == "__main__":
    unittest.main()
