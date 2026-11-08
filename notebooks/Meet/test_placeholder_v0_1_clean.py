"""Checks for Placeholder_v0.1_clean on synthetic data only (no weather data is read).

With every metric on "official" the adapter must be v0.1, so the whole v0.1 test suite runs against it. The rest
checks that cleaned series reach the learners and nothing else, and that a cleaned table cannot carry later years.

Run from the repository root:
    uv run python -m unittest notebooks/Meet/test_placeholder_v0_1_clean.py -v
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "code")]

import placeholder_v0_1
import placeholder_v0_1_clean as model
import test_placeholder_v0_1 as base
from common import METRICS
from evaluation.adapters import reference

base.model = model
from test_placeholder_v0_1 import (  # noqa: F401
    Forecasts,
    Geography,
    InformationBoundary,
    SelectionRule,
)

ORIGIN, CITIES, COORDS = base.ORIGIN, base.CITIES, base.COORDS


def cleaned_from(train: pd.DataFrame, offset: float = 0.0) -> pd.DataFrame:
    """A cleaned table with the release's column names: the official values, moved by `offset` in the homog series."""
    table = train[["city_id", "year"]].copy()
    for metric in METRICS:
        table[f"{metric}_qc"] = train[metric]
    table["annual_avg_temp_homog"] = train["annual_avg_temp"] + offset
    table["cdd_homog"] = train["cdd"] + offset
    table["hottest_day_temp_homog"] = train["hottest_day_temp"] + offset
    return table


class Inputs(unittest.TestCase):
    def test_unnamed_metrics_stay_official(self):
        resolved = model.resolve_inputs({"annual_avg_temp": "homog"})
        self.assertEqual(resolved["annual_avg_temp"], "homog")
        self.assertEqual({resolved[m] for m in METRICS if m != "annual_avg_temp"}, {"official"})
        self.assertTrue(model.uses_cleaned(resolved))
        self.assertFalse(model.uses_cleaned(model.resolve_inputs(None)))

    def test_rejects_series_a_metric_does_not_have(self):
        with self.assertRaises(ValueError):
            model.resolve_inputs({"total_precip_in": "homog"})
        with self.assertRaises(ValueError):
            model.resolve_inputs({"hottest_day_temp": "fill"})
        with self.assertRaises(ValueError):
            model.resolve_inputs({"not_a_metric": "qc"})

    def test_apply_replaces_only_the_chosen_metric(self):
        train = base.synthetic()
        out = model.apply_inputs(train, cleaned_from(train, 5.0), model.resolve_inputs({"annual_avg_temp": "homog"}))
        self.assertEqual(list(out.columns), list(train.columns))
        np.testing.assert_allclose(out["annual_avg_temp"], train["annual_avg_temp"] + 5.0)
        pd.testing.assert_series_equal(out["cdd"], train["cdd"])

    def test_nan_in_a_cleaned_series_is_kept(self):
        train = base.synthetic()
        cleaned = cleaned_from(train)
        cleaned.loc[cleaned.index[5], "annual_avg_temp_homog"] = np.nan
        out = model.apply_inputs(train, cleaned, model.resolve_inputs({"annual_avg_temp": "homog"}))
        self.assertTrue(np.isnan(out["annual_avg_temp"].iloc[5]))


class OfficialInputsAreV01(unittest.TestCase):
    def test_same_forecasts_as_the_sealed_adapter(self):
        train = base.synthetic(seed=3)
        targets = base.grid()
        new, _, _ = model.run_forecast(train, targets, ORIGIN, 42, COORDS)
        old, _, _ = placeholder_v0_1.run_forecast(train, targets, ORIGIN, 42, COORDS)
        pd.testing.assert_frame_equal(new, old)


class CleanedReachesOnlyTheLearners(unittest.TestCase):
    def setUp(self):
        self.train = base.synthetic(seed=1)
        self.targets = base.grid()

    def test_cleaned_series_changes_the_fit(self):
        train = self.train
        cleaned = cleaned_from(train, 2.0)
        cleaned["annual_avg_temp_homog"] += (train.year - 1900) * 0.05
        plain, _, _ = model.run_forecast(train, self.targets, ORIGIN, 42, COORDS, gate="none")
        moved, _, _ = model.run_forecast(train, self.targets, ORIGIN, 42, COORDS, gate="none", cleaned=cleaned,
                                         inputs={"annual_avg_temp": "homog"})
        rows = plain.metric == "annual_avg_temp"
        self.assertGreater(float((plain.predicted_value[rows] - moved.predicted_value[rows]).abs().max()), 1e-6)

    def test_lambda_zero_is_still_the_official_reference(self):
        cleaned = cleaned_from(self.train, 7.0)
        selection = {m: 0.0 for m in METRICS}
        with mock.patch.object(model, "choose_lambdas", return_value=(selection, {"selected": selection})):
            out, _, _ = model.run_forecast(self.train, self.targets, ORIGIN, 42, COORDS, cleaned=cleaned,
                                           inputs={"annual_avg_temp": "homog"})
        ref = reference(self.train, self.targets, ORIGIN, 42, method="average")
        joined = out.merge(ref, on=model.KEY, suffixes=("", "_ref"))
        np.testing.assert_allclose(joined.predicted_value, joined.predicted_value_ref)

    def test_inner_fits_read_the_cleaned_history(self):
        cleaned = cleaned_from(self.train, 3.0)
        with mock.patch.object(model, "fit_at", wraps=model.fit_at) as fit:
            model.choose_lambdas(self.train, CITIES, self.targets, ORIGIN, base.neighbours(), 42, "lin", "all",
                                 fit_history=model.apply_inputs(self.train, cleaned, model.resolve_inputs({"cdd": "homog"})))
        fitted = fit.call_args_list[0].args[0]
        self.assertGreater(float((fitted["cdd"] - self.train["cdd"]).abs().max()), 2.9)
        self.assertEqual(float((fitted["annual_avg_temp"] - self.train["annual_avg_temp"]).abs().max()), 0.0)


class LoadCleaned(unittest.TestCase):
    def write(self, root, table):
        folder = Path(root) / "data" / "processed" / "releases" / model.CLEAN_RELEASE / f"origin-{ORIGIN}"
        folder.mkdir(parents=True)
        table.to_parquet(folder / "annual_clean.parquet", index=False)

    def test_reads_only_the_chosen_columns_of_its_origin(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"PROTOSTAR_ASSET_ROOT": root}):
            self.write(root, cleaned_from(base.synthetic()))
            out = model.load_cleaned(ORIGIN, model.resolve_inputs({"cdd": "homog"}))
        self.assertEqual(list(out.columns), ["city_id", "year", "cdd_homog"])

    def test_refuses_a_table_that_passes_its_origin(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"PROTOSTAR_ASSET_ROOT": root}):
            self.write(root, cleaned_from(base.synthetic(last=ORIGIN + 1)))
            with self.assertRaises(ValueError):
                model.load_cleaned(ORIGIN, model.resolve_inputs({"cdd": "homog"}))

    def test_needs_the_harness_snapshot(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("PROTOSTAR_ASSET_ROOT", None)
            with self.assertRaises(RuntimeError):
                model.load_cleaned(ORIGIN, model.resolve_inputs({"cdd": "homog"}))


if __name__ == "__main__":
    unittest.main()
