"""Checks for Placeholder_v0.3 on synthetic data only (no weather data is read).

Run from the repository root:
    uv run python -m unittest notebooks/Meet/test_placeholder_v0_3.py -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "code")]

import placeholder_v0_3 as model  # noqa: E402
import test_placeholder_v0_1 as base  # noqa: E402
from common import METRICS  # noqa: E402
from evaluation.adapters import reference  # noqa: E402

base.model = model
from test_placeholder_v0_1 import CITIES, COORDS, N_CITIES, ORIGIN, Geography, SelectionRule, grid, neighbours, synthetic  # noqa: E402,F401


class Warming(unittest.TestCase):
    def tables(self, **kw):
        return model.tables_at(synthetic(**kw), CITIES, ORIGIN)

    def rates(self, tables, t=1960):
        per = {m: model.metric_state(table, t) for m, table in tables.items()}
        return model.warming_rates(per, tables, t, neighbours())

    def test_rates_ignore_years_after_t(self):
        tables = self.tables()
        changed = {m: t.copy() for m, t in tables.items()}
        for table in changed.values():
            table.loc[:, 1961:] = table.loc[:, 1961:] + 1e6
        first, second = self.rates(tables), self.rates(changed)
        for metric in METRICS:
            np.testing.assert_array_equal(first[metric], second[metric], err_msg=metric)

    def test_precipitation_and_hottest_day_have_no_warming(self):
        rates = self.rates(self.tables(trend=0.2))
        self.assertTrue((rates["total_precip_in"] == 0).all() and (rates["hottest_day_temp"] == 0).all())

    def test_cooling_is_floored_at_zero(self):
        rates = self.rates(self.tables(trend=-0.2))
        self.assertTrue((rates["annual_avg_temp"] == 0).all())

    def test_warming_is_positive_for_a_warming_series(self):
        rates = self.rates(self.tables(trend=0.2))
        self.assertTrue((rates["annual_avg_temp"] > 0).all() and (rates["cdd"] > 0).all())

    def test_extrapolates_from_the_mean_valid_year(self):
        term = model.warming_term(np.array([0.1]), np.array([1950.0]), 1960, 5)
        self.assertAlmostEqual(float(term[0]), 1.5)

    def test_undefined_rate_or_year_gives_no_warming(self):
        self.assertEqual(float(model.warming_term(np.array([0.1]), np.array([np.nan]), 1960, 5)[0]), 0.0)


class Forecasts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = synthetic(trend=0.15)
        cls.result = model.run_forecast(cls.train, grid(), ORIGIN, 42, COORDS)

    def test_complete_finite_grid(self):
        output = self.result[0]
        self.assertEqual(len(output), N_CITIES * 7 * 4)
        self.assertTrue(np.isfinite(output.predicted_value).all())

    def test_deterministic(self):
        pd.testing.assert_frame_equal(self.result[0], model.run_forecast(self.train, grid(), ORIGIN, 42, COORDS)[0])

    def test_inner_fit_ignores_years_after_inner_origin(self):
        inner = 1969
        changed = self.train.copy()
        late = changed.year > inner
        changed.loc[late, list(METRICS)] += 5000.0
        _, a, wa, ra = model.fit_at(self.train, CITIES, inner, neighbours(), 42, "ensemble", "all")
        _, b, wb, rb = model.fit_at(changed, CITIES, inner, neighbours(), 42, "ensemble", "all")
        self.assertEqual(ra, rb)
        for metric in METRICS:
            pd.testing.assert_frame_equal(a[metric], b[metric])
            pd.testing.assert_frame_equal(wa[metric], wb[metric])

    def test_zero_warming_and_lambda_zero_is_the_reference(self):
        expected = reference(self.train, grid(), ORIGIN, 42, method="average")
        _, shifts, warming, _ = model.fit_at(self.train, CITIES, ORIGIN, neighbours(), 42, "ensemble", "all")
        zero = {m: frame * 0.0 for m, frame in warming.items()}
        output = model.assemble(expected, shifts, zero, {m: 0.0 for m in METRICS})
        merged = output.merge(expected, on=model.KEY, suffixes=("", "_ref"), validate="one_to_one")
        np.testing.assert_array_equal(merged.predicted_value, merged.predicted_value_ref)

    def test_warming_applies_even_when_lambda_is_zero(self):
        expected = reference(self.train, grid(), ORIGIN, 42, method="average")
        _, shifts, warming, _ = model.fit_at(self.train, CITIES, ORIGIN, neighbours(), 42, "ensemble", "all")
        output = model.assemble(expected, shifts, warming, {m: 0.0 for m in METRICS})
        rows = output.merge(expected, on=model.KEY, suffixes=("", "_ref"), validate="one_to_one")
        temp = rows[rows.metric == "annual_avg_temp"]
        self.assertTrue((temp.predicted_value > temp.predicted_value_ref).all())
        other = rows[rows.metric == "hottest_day_temp"]
        np.testing.assert_array_equal(other.predicted_value, other.predicted_value_ref)

    def test_residual_targets_carry_no_era_wide_shift(self):
        tables = model.tables_at(self.train, CITIES, ORIGIN)
        _, y = model.training_rows(tables, neighbours(), ORIGIN, "all")["annual_avg_temp"]
        self.assertLess(abs(float(np.median(y))), 0.05)

    def test_refuses_rows_after_origin(self):
        with self.assertRaises(ValueError):
            model.run_forecast(synthetic(last=1995), grid(), ORIGIN, 42, COORDS)


if __name__ == "__main__":
    unittest.main()
