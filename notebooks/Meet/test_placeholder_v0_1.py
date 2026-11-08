"""Checks for Placeholder_v0.1 on synthetic data only (no weather data is read).

Run from the repository root:
    uv run python -m unittest notebooks/Meet/test_placeholder_v0_1.py -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "code")]

import placeholder_v0_1 as model  # noqa: E402
from common import METRICS  # noqa: E402
from evaluation.adapters import reference  # noqa: E402
from evaluation.core import build_fold  # noqa: E402

N_CITIES = 40
FIRST, ORIGIN = 1900, 1990
CITIES = sorted(f"C{i:03d}" for i in range(N_CITIES))
COORDS = pd.DataFrame({"lat": np.linspace(30, 48, N_CITIES), "lon": np.linspace(-120, -72, N_CITIES)}, index=CITIES)


def synthetic(seed=0, trend=0.05, noise=1.0, first=FIRST, last=ORIGIN) -> pd.DataFrame:
    """Annual series with a shared persistent trend and independent noise."""
    rng = np.random.default_rng(seed)
    years = np.arange(first, last + 1)
    rows = []
    for i, city in enumerate(CITIES):
        level = 50 + i * 0.3
        for year, drift, e in zip(years, trend * (years - first), rng.normal(0, noise, (len(years), 4))):
            rows.append({"city_id": city, "year": int(year), "name": city,
                         "annual_avg_temp": level + drift + e[0],
                         "total_precip_in": max(0.0, 30 + i * 0.2 + drift + 3 * e[1]),
                         "hottest_day_temp": 95 + drift + e[2],
                         "cdd": max(0.0, 1000 + 20 * i + 25 * drift + 40 * e[3])})
    return pd.DataFrame(rows)


def grid(origin=ORIGIN) -> pd.DataFrame:
    return pd.MultiIndex.from_product([CITIES, range(origin + 1, origin + 8), list(METRICS)],
                                      names=model.KEY).to_frame(index=False)


def neighbours():
    return model.neighbour_index(COORDS.sort_index())


class InformationBoundary(unittest.TestCase):
    def test_state_ignores_years_after_t(self):
        table = model.tables_at(synthetic(), CITIES, ORIGIN)["cdd"]
        t = 1960
        changed = table.copy()
        changed.loc[:, t + 1:] = changed.loc[:, t + 1:] + 1e6
        first, second = model.metric_state(table, t), model.metric_state(changed, t)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)

    def test_inner_fit_ignores_years_after_inner_origin(self):
        history = synthetic()
        inner = 1969
        changed = history.copy()
        late = changed.year > inner
        changed.loc[late, list(METRICS)] = changed.loc[late, list(METRICS)] + 5000.0
        _, first, rows_a = model.fit_at(history, CITIES, inner, neighbours(), 42, "ensemble", "all")
        _, second, rows_b = model.fit_at(changed, CITIES, inner, neighbours(), 42, "ensemble", "all")
        self.assertEqual(rows_a, rows_b)
        for metric in METRICS:
            pd.testing.assert_frame_equal(first[metric], second[metric])

    def test_training_targets_never_pass_fit_origin(self):
        history = synthetic()
        tables = model.tables_at(history, CITIES, 1969)
        self.assertLessEqual(max(int(t.columns.max()) for t in tables.values()), 1969)
        rows = model.training_rows(tables, neighbours(), 1969, "all")
        self.assertTrue(all(len(y) for _, y in rows.values()))

    def test_scored_labels_are_the_next_seven_years(self):
        _, targets, truth, _ = build_fold(synthetic(), CITIES, 1969)
        self.assertEqual(sorted(truth.year.unique().tolist()), list(range(1970, 1977)))
        self.assertEqual(int(targets.year.max()), 1976)

    def test_gate_scores_only_inner_windows_and_ignores_years_after_outer_origin(self):
        history = synthetic(last=2005)
        changed = history.copy()
        late = changed.year > ORIGIN
        changed.loc[late, list(METRICS)] = changed.loc[late, list(METRICS)] + 5000.0
        cut = history[history.year <= ORIGIN]
        args = (CITIES, grid(), ORIGIN, neighbours(), 42, "ensemble", "all")
        selected_a, record_a = model.choose_lambdas(cut, *args)
        selected_b, record_b = model.choose_lambdas(changed, *args)
        self.assertEqual(selected_a, selected_b)
        self.assertEqual(record_a["mean_final_skill"], record_b["mean_final_skill"])
        self.assertEqual([fold["origin"] for fold in record_a["inner_folds"]], [1969, 1976, 1983])
        for inner in (1969, 1976, 1983):
            _, _, truth, _ = build_fold(cut, CITIES, inner)
            self.assertEqual(sorted(truth.year.unique().tolist()), list(range(inner + 1, inner + 8)))
            self.assertLessEqual(int(truth.year.max()), ORIGIN)

    def test_refuses_rows_after_origin(self):
        with self.assertRaises(ValueError):
            model.run_forecast(synthetic(last=1995), grid(), ORIGIN, 42, COORDS)


class Forecasts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = synthetic(trend=0.15)
        cls.first = model.run_forecast(cls.train, grid(), ORIGIN, 42, COORDS)

    def test_complete_finite_grid(self):
        output = self.first[0]
        self.assertEqual(len(output), N_CITIES * 7 * 4)
        self.assertFalse(output.duplicated(model.KEY).any())
        self.assertTrue(np.isfinite(output.predicted_value).all())

    def test_deterministic(self):
        again = model.run_forecast(self.train, grid(), ORIGIN, 42, COORDS)[0]
        pd.testing.assert_frame_equal(self.first[0], again)

    def test_planted_persistent_shift_is_recovered(self):
        selected = self.first[1]["selected"]
        self.assertGreater(selected["annual_avg_temp"], 0.0)
        self.assertGreater(selected["cdd"], 0.0)

    def test_linear_member_has_real_coefficients(self):
        members = self.first[2]["annual_avg_temp"]
        self.assertGreater(float(np.abs(members.lin.coef_).max()), 0.01)

    def test_lambda_zero_is_the_reference_on_every_row(self):
        history = self.train
        expected = reference(history, grid(), ORIGIN, 42, method="average")
        _, shifts, _ = model.fit_at(history, CITIES, ORIGIN, neighbours(), 42, "ensemble", "all")
        output = model.assemble(expected, shifts, {m: 0.0 for m in METRICS})
        merged = output.merge(expected, on=model.KEY, suffixes=("", "_ref"), validate="one_to_one")
        np.testing.assert_array_equal(merged.predicted_value, merged.predicted_value_ref)

    def test_non_negative_metrics_are_clamped(self):
        base = reference(self.train, grid(), ORIGIN, 42, method="average")
        _, shifts, _ = model.fit_at(self.train, CITIES, ORIGIN, neighbours(), 42, "ensemble", "all")
        huge = {m: shifts[m] - 1e7 for m in METRICS}
        output = model.assemble(base, huge, {m: 1.0 for m in METRICS})
        for metric in model.NON_NEGATIVE:
            self.assertTrue((output[output.metric == metric].predicted_value >= 0).all())

    def test_ablation_options_run(self):
        for options in ({"members": "lin", "gate": "none"}, {"members": "gbm", "gate": "none"},
                        {"features": "no_pool", "gate": "none"}):
            output = model.run_forecast(self.train, grid(), ORIGIN, 42, COORDS, **options)[0]
            self.assertEqual(len(output), N_CITIES * 7 * 4)

    def test_unknown_option_fails(self):
        with self.assertRaises(ValueError):
            model.run_forecast(self.train, grid(), ORIGIN, 42, COORDS, members="forest")


class SelectionRule(unittest.TestCase):
    def test_picks_smallest_lambda_within_margin(self):
        table = {0.0: 0.0, 0.25: 0.040, 0.5: 0.052, 0.75: 0.050, 1.0: 0.030}
        self.assertEqual(model.select_lambda(table), 0.5)
        table = {0.0: 0.0, 0.25: 0.048, 0.5: 0.052, 0.75: 0.050, 1.0: 0.030}
        self.assertEqual(model.select_lambda(table), 0.25)

    def test_zero_when_nothing_beats_the_reference(self):
        self.assertEqual(model.select_lambda({0.0: 0.0, 0.25: -0.01, 0.5: -0.02, 0.75: -0.03, 1.0: -0.04}), 0.0)
        self.assertEqual(model.select_lambda({0.0: -1e-17, 0.25: -0.01, 0.5: -0.02, 0.75: -0.03, 1.0: -0.04}), 0.0)

    def test_small_gain_inside_margin_keeps_zero(self):
        self.assertEqual(model.select_lambda({0.0: 0.0, 0.25: 0.004, 0.5: 0.003, 0.75: 0.0, 1.0: -0.01}), 0.0)


class Geography(unittest.TestCase):
    def test_excludes_self_and_breaks_ties_by_city_id(self):
        coords = pd.DataFrame({"lat": [40.0, 40.0, 40.0, 41.0], "lon": [-100.0, -100.0, -100.0, -100.0]},
                              index=["A", "B", "C", "D"])
        index = model.neighbour_index(coords, k=2)
        self.assertEqual(index.tolist()[0], [1, 2])
        self.assertTrue(all(i not in row for i, row in enumerate(index)))

    def test_requires_sorted_coordinates(self):
        with self.assertRaises(ValueError):
            model.neighbour_index(COORDS.sort_index(ascending=False))


if __name__ == "__main__":
    unittest.main()
