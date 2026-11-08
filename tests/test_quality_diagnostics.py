"""Offline synthetic checks for scientific sensitivity diagnostics."""
import calendar
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from common import DataRuleViolation, METRICS
from evaluation.core import build_fold
from evaluation.official_score import score
from evaluation.quality_diagnostics import diagnose_origin, metrics_frame


def fixture():
    annual, coverage = [], []
    for city in ("A", "B", "C"):
        for year in range(1982, 2019):
            value = year - 1982 + .4 + ord(city) - ord("A")
            annual.append({"city_id": city, "year": year,
                           **{m: value * (i + 1) for i, m in enumerate(METRICS)}})
            coverage.append({"ID": city, "Year": year,
                             "Days Both": 365 + calendar.isleap(year),
                             "Station Segment": 1, "Station Changes": 0})
    annual, coverage = pd.DataFrame(annual), pd.DataFrame(coverage)
    _, targets, answer, ref = build_fold(annual, ["A", "B", "C"], 2011)
    predictions = targets.assign(predicted_value=20.25)
    return annual, coverage, predictions, answer, ref


class QualityDiagnosticTests(unittest.TestCase):
    def run_diagnostic(self, annual=None, coverage=None, predictions=None):
        data, cov, pred, _, _ = fixture()
        return diagnose_origin(data if annual is None else annual,
                               cov if coverage is None else coverage, 2011,
                               pred if predictions is None else predictions)

    def test_default_parity_with_official_scorer_and_inputs_unchanged(self):
        data, coverage, predictions, answer, ref = fixture()
        originals = [v.copy(deep=True) for v in (data, coverage, predictions)]
        result = diagnose_origin(data, coverage, 2011, predictions, provenance={"source": "synthetic"})
        for split in ("live", "final"):
            actual = result["variants"]["default"]["splits"][split]
            expected = score(predictions, answer, ref, split)
            self.assertEqual(actual["official_style_score"], expected["score"])
            self.assertEqual(actual["n_rows"], expected["n_rows"])
            for metric in METRICS:
                self.assertEqual(actual["metrics"][metric]["mae"], expected["mae"][metric])
                self.assertEqual(actual["metrics"][metric]["skill"], expected["skill"][metric])
        for old, current in zip(originals, (data, coverage, predictions)):
            assert_frame_equal(old, current)
        self.assertEqual(len(metrics_frame(result)), 5 * 2 * 4)
        self.assertEqual(result["baseline_window"], [1982, 2011])
        self.assertEqual(result["target_window"], [2012, 2018])
        json.dumps(result, allow_nan=False)

    def test_station_variants_exclude_target_and_baseline_changes(self):
        data, cov, pred, _, _ = fixture()
        cov.loc[cov.ID.eq("A") & cov.Year.eq(2013), "Station Changes"] = 1
        cov.loc[cov.ID.eq("A") & cov.Year.ge(2013), "Station Segment"] = 2
        cov.loc[cov.ID.eq("B") & cov.Year.eq(2000), "Station Changes"] = 1
        cov.loc[cov.ID.eq("B") & cov.Year.ge(2000), "Station Segment"] = 2
        result = diagnose_origin(data, cov, 2011, pred)["variants"]
        target = result["station_stable_target"]
        full = result["station_stable_baseline_target"]
        self.assertEqual(target["exclusions"]["station_change_or_segment_difference"], ["A"])
        self.assertEqual(full["exclusions"]["station_change_or_segment_difference"], ["A", "B"])
        self.assertEqual(target["splits"]["live"]["n_rows"], 2 * 3 * 4)
        self.assertEqual(full["splits"]["final"]["n_rows"], 1 * 4 * 4)

    def test_segment_difference_alone_cannot_look_stable(self):
        data, cov, pred, _, _ = fixture()
        cov.loc[cov.ID.eq("A") & cov.Year.ge(2013), "Station Segment"] = 2
        result = diagnose_origin(data, cov, 2011, pred)
        self.assertEqual(result["variants"]["station_stable_target"]["exclusions"]
                         ["station_change_or_segment_difference"], ["A"])

    def test_days_both_recalculates_baseline_and_target_eligibility(self):
        data, cov, pred, _, _ = fixture()
        # A has exactly ten valid baseline years by the original metric mask.
        data.loc[data.city_id.eq("A") & data.year.lt(2002), "cdd"] = np.nan
        # Failing one of these leaves nine: all A CDD targets become ineligible.
        cov.loc[cov.ID.eq("A") & cov.Year.eq(2002), "Days Both"] = 354
        # B retains 29 baseline years, changing the denominator, and loses one target.
        cov.loc[cov.ID.eq("B") & cov.Year.eq(1982), "Days Both"] = 354
        cov.loc[cov.ID.eq("B") & cov.Year.eq(2012), "Days Both"] = 355  # leap: 11 missing
        # Non-leap 355 = exactly ten missing, and remains eligible.
        cov.loc[cov.ID.eq("C") & cov.Year.eq(2013), "Days Both"] = 355
        result = diagnose_origin(data, cov, 2011, pred)
        variant = result["variants"]["cdd_days_both_strict"]
        baseline = {v["city_id"]: v for v in variant["cdd_baseline_by_city"]}
        self.assertEqual(baseline["A"]["default_valid_years"], 10)
        self.assertEqual(baseline["A"]["strict_valid_years"], 9)
        self.assertIsNone(baseline["A"]["strict_baseline"])
        self.assertEqual(baseline["B"]["strict_valid_years"], 29)
        expected_mean = data.loc[data.city_id.eq("B") & data.year.between(1983, 2011), "cdd"].mean()
        self.assertEqual(baseline["B"]["strict_baseline"], expected_mean)
        self.assertEqual(variant["splits"]["live"]["metrics"]["cdd"]["n_rows"], 5)
        self.assertEqual(variant["splits"]["final"]["metrics"]["cdd"]["n_rows"], 8)
        self.assertEqual(variant["exclusions"]["baseline_city_years_failed_days_both"], 2)
        self.assertEqual(variant["exclusions"]["target_city_years_failed_days_both"], 1)
        for metric in set(METRICS) - {"cdd"}:
            self.assertEqual(variant["splits"]["live"]["metrics"][metric]["skill_delta_vs_default"], 0)

    def test_missing_coverage_is_explicit(self):
        data, cov, pred, _, _ = fixture()
        result = diagnose_origin(data, None, 2011, pred)
        for name in ("station_stable_target", "station_stable_baseline_target", "cdd_days_both_strict"):
            variant = result["variants"][name]
            self.assertEqual(variant["coverage_status"], "unavailable")
            self.assertIn("not provided", variant["coverage_unavailable_reason"])
        cov = cov[~(cov.ID.eq("A") & cov.Year.eq(2012))]
        result = diagnose_origin(data, cov, 2011, pred)["variants"]
        self.assertEqual(result["station_stable_target"]["coverage_status"], "partial")
        self.assertEqual(result["station_stable_target"]["exclusions"]["missing_coverage"], ["A"])
        self.assertIn("missing", result["cdd_days_both_strict"]["coverage_unavailable_reason"])

    def test_invalid_coverage_metadata_disables_affected_variant(self):
        data, cov, pred, _, _ = fixture()
        for field, value, name in (("Station Segment", 0, "station_stable_target"),
                                   ("Station Changes", -.1, "station_stable_target"),
                                   ("Days Both", 367, "cdd_days_both_strict"),
                                   ("Days Both", np.nan, "cdd_days_both_strict")):
            altered = cov.copy()
            altered[field] = altered[field].astype(float)
            altered.loc[0, field] = value
            variant = diagnose_origin(data, altered, 2011, pred)["variants"][name]
            self.assertEqual(variant["coverage_status"], "unavailable")
            self.assertIn("invalid", variant["coverage_unavailable_reason"])
        missing = cov.drop(columns="Days Both")
        result = diagnose_origin(data, missing, 2011, pred)
        self.assertIn("missing columns", result["variants"]["cdd_days_both_strict"]["coverage_unavailable_reason"])

    def test_precision_scenario_keeps_predictions_and_default_mask(self):
        data, cov, pred, answer, ref = fixture()
        original = pred.copy(deep=True)
        result = diagnose_origin(data, cov, 2011, pred)
        scenario = result["variants"]["temperature_whole_f_rounding"]
        metric = scenario["splits"]["live"]["metrics"]["annual_avg_temp"]
        self.assertNotEqual(metric["mae_delta_vs_default"], 0)
        self.assertEqual(metric["n_rows_delta_vs_default"], 0)
        assert_frame_equal(pred, original)
        self.assertIn("predictions unchanged", scenario["rounding"]["applies_to"])

    def test_zero_reference_and_empty_scenario_are_finite_and_explicit(self):
        data, cov, pred, _, _ = fixture()
        data["annual_avg_temp"] = 1.
        cov["Station Changes"] = 1
        result = diagnose_origin(data, cov, 2011, pred)
        default = result["variants"]["default"]["splits"]["live"]
        self.assertIsNone(default["diagnostic_score"])
        self.assertIn("zero reference", default["metrics"]["annual_avg_temp"]["unavailable_reason"])
        variant = result["variants"]["station_stable_target"]["splits"]["live"]
        self.assertEqual(variant["n_rows"], 0)
        self.assertIn("no eligible", variant["metrics"]["cdd"]["unavailable_reason"])
        json.dumps(result, allow_nan=False)

    def test_raw_schema_and_cross_table_days_both_validation(self):
        data, cov, pred, _, _ = fixture()
        raw = data.rename(columns={"city_id": "ID", "year": "Year",
                                   **{m: definition[0] for m, definition in METRICS.items()}})
        for column in ("Miss Tmax", "Miss Tmin", "Miss Prcp"):
            raw[column] = 0
        original = raw.copy(deep=True)
        result = diagnose_origin(raw, cov, 2011, pred)
        normalized = diagnose_origin(data, cov, 2011, pred)
        self.assertEqual(result["source"]["annual_schema"], "annual_indices")
        self.assertEqual(result["variants"]["default"], normalized["variants"]["default"])
        assert_frame_equal(raw, original)
        # A provided intersection count inconsistent with zero marginal missing
        # counts is unavailable, rather than silently redefining completeness.
        cov.loc[0, "Days Both"] -= 1
        result = diagnose_origin(raw, cov, 2011, pred)
        self.assertIn("conflicts with annual missing counts", result["variants"]["cdd_days_both_strict"]
                      ["coverage_unavailable_reason"])
        raw.loc[0, "Miss Tmax"] = -1
        with self.assertRaises(ValueError):
            diagnose_origin(raw, cov, 2011, pred)

    def test_dates_keys_and_predictions_fail_closed(self):
        data, cov, pred, _, _ = fixture()
        for table, column in ((data, "year"), (cov, "Year")):
            bad = table.copy()
            bad.loc[0, column] = 2019
            with self.assertRaises(DataRuleViolation):
                diagnose_origin(bad if column == "year" else data,
                                bad if column == "Year" else cov, 2011, pred)
        for bad in (pd.concat([cov, cov.iloc[[0]]]), cov.assign(ID="")):
            with self.assertRaises(ValueError):
                diagnose_origin(data, bad, 2011, pred)
        with self.assertRaises(ValueError):
            diagnose_origin(data, cov, 2012, pred)
        with self.assertRaises(ValueError):
            diagnose_origin(data, cov, 2011, pred.iloc[1:])


if __name__ == "__main__":
    unittest.main()
