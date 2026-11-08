"""Behavioral checks for evaluating a historical method-selection procedure."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from common import METRICS
from evaluation.nested import forecast
from combo_model import FORECASTERS


def linear_history(last_year=2011, first_year=1900):
    rows = []
    for city, level in (("A", 0.), ("B", 20.)):
        for year in range(first_year, last_year + 1):
            record = {"city_id": city, "year": year}
            for number, metric in enumerate(METRICS, 1):
                record[metric] = level + 30. * number + .1 * number * (year - 1900)
            rows.append(record)
    return pd.DataFrame(rows)


def target_grid(origin=2011):
    return pd.MultiIndex.from_product([["A", "B"], range(origin + 1, origin + 8), list(METRICS)],
                                     names=["city_id", "year", "metric"]).to_frame(index=False)


class NestedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.artifacts = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"PROTOSTAR_ARTIFACT_DIR": str(self.artifacts)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def artifact(self):
        return json.loads((self.artifacts / "nested_selection.json").read_text(encoding="utf-8"))

    def test_linear_candidate_wins_and_complete_predictions_are_deterministic(self):
        train, targets = linear_history(), target_grid()
        first = forecast(train, targets, 2011, candidates=["average", "trend"])
        artifact = self.artifact()
        self.assertEqual(set(artifact["selected_methods"].values()), {"trend"})
        self.assertEqual(artifact["requested_inner_origins"], [1990, 1997, 2004])
        self.assertEqual(len(first), 56)
        self.assertTrue(np.isfinite(first.predicted_value).all())
        expected = linear_history(last_year=2018).melt(
            id_vars=["city_id", "year"], value_vars=list(METRICS),
            var_name="metric", value_name="actual")
        matched = first.merge(expected, on=["city_id", "year", "metric"], validate="one_to_one")
        self.assertTrue(np.allclose(matched.predicted_value, matched.actual, atol=1e-10))
        second = forecast(train, targets, 2011, seed=99, candidates=["trend", "average"])
        pd.testing.assert_frame_equal(first, second)

    def test_every_fit_and_inner_label_respects_its_information_boundary(self):
        calls = []
        originals = dict(FORECASTERS)
        def checking(method):
            def wrapped(table, origin, years):
                self.assertLessEqual(int(table.columns.max()), origin)
                self.assertEqual(years, list(range(origin + 1, origin + 8)))
                calls.append(origin)
                return originals[method](table, origin, years)
            return wrapped
        with patch.dict(FORECASTERS, {method: checking(method) for method in FORECASTERS}):
            forecast(linear_history(), target_grid(), 2011, candidates=["average", "trend"])
        self.assertEqual(set(calls), {1990, 1997, 2004, 2011})
        for fold in self.artifact()["inner_folds"]:
            self.assertLessEqual(fold["train_max_year"], fold["origin"])
            self.assertLessEqual(fold["target_max_year"], 2011)
            self.assertEqual(fold["target_min_year"], fold["origin"] + 1)

    def test_outer_labels_are_never_loaded_or_used(self):
        complete = linear_history(last_year=2018)
        def run():
            return forecast(complete[complete.year <= 2011].copy(), target_grid(), 2011,
                            candidates=["average", "trend"])
        with patch("pandas.read_csv", side_effect=AssertionError("selector must not load data")), \
                patch("pandas.read_parquet", side_effect=AssertionError("selector must not load data")):
            first = run()
            complete.loc[complete.year > 2011, list(METRICS)] = -99999.
            second = run()
        pd.testing.assert_frame_equal(first, second)
        with self.assertRaisesRegex(ValueError, "after outer origin"):
            forecast(complete, target_grid(), 2011)
        with self.assertRaisesRegex(ValueError, "only city_id, year, metric"):
            forecast(linear_history(), target_grid().assign(actual_value=99), 2011)

    def test_inadequate_history_uses_declared_whole_procedure_fallback(self):
        output = forecast(linear_history(first_year=2005), target_grid(), 2011)
        artifact = self.artifact()
        self.assertEqual(artifact["status"], "fallback_insufficient_history")
        self.assertEqual(artifact["inner_folds"], [])
        self.assertEqual(artifact["candidate_mean_final_skill"], {})
        self.assertEqual(set(artifact["selected_methods"].values()), {"average"})
        self.assertEqual(len(output), 56)
        self.assertTrue(np.isfinite(output.predicted_value).all())

    def test_undefined_inner_reference_fails_instead_of_skipping_fold(self):
        train = linear_history()
        train[list(METRICS)] = 50.
        with self.assertRaisesRegex(ValueError, "inner origin 1990 is unusable.*zero"):
            forecast(train, target_grid(), 2011)
        self.assertFalse((self.artifacts / "nested_selection.json").exists())

    def test_invalid_parameters_fail_clearly(self):
        invalid = [{"unknown": 2}, {"inner_folds": True}, {"inner_folds": 0}, {"inner_folds": 11},
                   {"candidates": []}, {"candidates": ["average", "average"]},
                   {"candidates": ["random_forest"]}, {"candidates": "average"}]
        for params in invalid:
            with self.subTest(params=params), self.assertRaisesRegex(ValueError, "nested adapter"):
                forecast(linear_history(), target_grid(), 2011, **params)


if __name__ == "__main__":
    unittest.main()
