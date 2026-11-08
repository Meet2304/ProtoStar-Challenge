"""Offline contract tests, independent gold examples, and complete runner checks."""
import ast
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from common import DataRuleViolation, METRICS
from evaluation import official_score
from evaluation.core import (build_fold, compare_runs, digest, load_annual, prepare,
                             read_json, run_suite, validate_config, validate_predictions,
                             verify_suite, write_json, rubric_score)
from evaluation.report import render_report


def annual_fixture():
    records = []
    for city in ("A", "B"):
        for year in range(1950, 2019):
            value = 50 + .08 * (year - 1950) + math.sin(year) + (city == "B") * 10
            records.append({"ID": city, "Year": year, "Miss Tmax": 0, "Miss Tmin": 0,
                            "Miss Prcp": 0, "Avg Temp": value, "ToT Prcp": value / 2,
                            "Max Tmax": value + 40, "cdd": value * 10})
    return pd.DataFrame(records)


class ScoringTests(unittest.TestCase):
    def test_human_rubric_weights_and_scale(self):
        rubric = read_json(ROOT / "evaluation/rubric.json")
        for rating, expected in ((1, 0), (3, 50), (5, 100)):
            self.assertEqual(rubric_score({r["key"]: rating for r in rubric["criteria"]}), expected)
        with self.assertRaises(ValueError):
            rubric_score({r["key"]: None for r in rubric["criteria"]})

    def fixture(self):
        truth = pd.DataFrame([dict(city_id="A", year=2012, metric=m, actual_value=0.,
                                   split="live", scored=True) for m in METRICS])
        pred = truth[official_score.KEY].copy()
        pred["predicted_value"] = [0., 0., 0., 4.]
        ref = pd.DataFrame([dict(split="live", **{m: 1. for m in METRICS})])
        return pred, truth, ref

    def test_official_functions_are_unchanged(self):
        upstream = ast.parse((ROOT / "evaluation/upstream/scoring.py").read_text(encoding="utf-8"))
        local = ast.parse(Path(official_score.__file__).read_text(encoding="utf-8"))
        for name in ("score", "ScoringError"):
            extract = lambda tree: next(n for n in tree.body if getattr(n, "name", None) == name)
            self.assertEqual(ast.dump(extract(upstream)), ast.dump(extract(local)))

    def test_upstream_snapshots_match_recorded_hashes(self):
        base = ROOT / "evaluation/upstream"
        for file, record in read_json(base / "manifest.json")["files"].items():
            self.assertEqual(digest(base / file), record["sha256"])

    def test_gold_clip_after_averaging_not_each_metric(self):
        pred, truth, ref = self.fixture()
        result = official_score.score(pred, truth, ref, "live")
        self.assertEqual(result["score"], 0.)  # mean(1, 1, 1, -3) = 0
        self.assertEqual(result["skill"]["cdd"], -3.)
        pred["predicted_value"] = [0., 0., 0., 2.]
        self.assertEqual(official_score.score(pred, truth, ref, "live")["score"], 50.)

    def test_metric_weights_equal_despite_different_row_counts(self):
        pred, truth, ref = self.fixture()
        pred["predicted_value"] = [0., 0., 0., 2.]
        extra_truth = truth.iloc[[0]].assign(city_id="B")
        extra_pred = pred.iloc[[0]].assign(city_id="B")
        result = official_score.score(pd.concat([pred, extra_pred]), pd.concat([truth, extra_truth]), ref, "live")
        self.assertEqual(result["score"], 50.)
        self.assertEqual(result["n_rows"], 5)

    def test_invalid_truth_row_excluded(self):
        pred, truth, ref = self.fixture()
        extra = truth.iloc[[0]].assign(city_id="BAD", actual_value=999., scored=False)
        self.assertEqual(official_score.score(pred, pd.concat([truth, extra]), ref, "live")["n_rows"], 4)

    def test_missing_scored_prediction_rejected_by_official_scorer(self):
        pred, truth, ref = self.fixture()
        with self.assertRaises(official_score.ScoringError):
            official_score.score(pred.iloc[1:], truth, ref, "live")

    def test_randomized_numeric_parity_with_upstream(self):
        # Execute only the upstream pure AST, without its config/network imports.
        upstream = ast.parse((ROOT / "evaluation/upstream/scoring.py").read_text(encoding="utf-8"))
        nodes = [n for n in upstream.body if getattr(n, "name", None) in ("score", "ScoringError")]
        namespace = {"pd": pd, "math": math, "METRICS": tuple(METRICS),
                     "KEY": official_score.KEY, "SPLITS": ("live", "final")}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "upstream/scoring.py", "exec"), namespace)
        rng = np.random.default_rng(81)
        for _ in range(30):
            pred, truth, ref = self.fixture()
            pred["predicted_value"] = rng.normal(0, 5, len(pred))
            self.assertEqual(official_score.score(pred, truth, ref, "live"),
                             namespace["score"](pred, truth, ref, "live"))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "annual.csv"
        annual_fixture().to_csv(self.path, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_ten_day_boundary_and_both_temperatures_for_hottest_day(self):
        data = annual_fixture()
        data.loc[0, ["Miss Tmax", "Miss Tmin"]] = [10, 10]
        data.loc[1, "Miss Tmin"] = 11
        data.to_csv(self.path, index=False)
        clean = load_annual(self.path)
        self.assertTrue(pd.notna(clean.loc[0, "hottest_day_temp"]))
        self.assertTrue(pd.isna(clean.loc[1, "hottest_day_temp"]))
        self.assertTrue(pd.notna(clean.loc[1, "total_precip_in"]))

    def test_invalid_year_infinity_masked_valid_infinity_rejected(self):
        data = annual_fixture()
        data.loc[0, "Max Tmax"] = -np.inf
        data.loc[0, "Miss Tmax"] = 365
        data.to_csv(self.path, index=False)
        self.assertTrue(pd.isna(load_annual(self.path).loc[0, "hottest_day_temp"]))
        data.loc[0, "Miss Tmax"] = 0
        data.to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            load_annual(self.path)

    def test_post_2018_labels_and_inputs_rejected(self):
        data = annual_fixture()
        data.loc[0, "Year"] = 2019
        data.to_csv(self.path, index=False)
        with self.assertRaises(DataRuleViolation):
            load_annual(self.path)

    def test_bad_year_and_duplicate_city_year_rejected(self):
        for value in (np.nan, 2011.5):
            data = annual_fixture()
            data["Year"] = data.Year.astype(float)
            data.loc[0, "Year"] = value
            data.to_csv(self.path, index=False)
            with self.assertRaises(ValueError):
                load_annual(self.path)
        data = annual_fixture()
        pd.concat([data, data.iloc[[0]]]).to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            load_annual(self.path)

    def test_baseline_nine_invalid_ten_valid_and_horizon(self):
        data = load_annual(self.path)
        # In A's 1982-2011 window, retain only 10 years of temperature.
        mask = (data.city_id == "A") & data.year.between(1982, 2001)
        data.loc[mask, "annual_avg_temp"] = np.nan
        train, targets, truth, ref = build_fold(data, ["A", "B"], 2011)
        selected = truth[(truth.city_id == "A") & (truth.metric == "annual_avg_temp")]
        self.assertTrue(selected.scored.all())
        self.assertEqual(train.year.max(), 2011)
        self.assertEqual(set(targets.year), set(range(2012, 2019)))
        self.assertEqual(set(truth[truth.split == "final"].year), {2015, 2016, 2017, 2018})
        data.loc[(data.city_id == "A") & (data.year == 2002), "annual_avg_temp"] = np.nan
        _, _, truth, _ = build_fold(data, ["A", "B"], 2011)
        selected = truth[(truth.city_id == "A") & (truth.metric == "annual_avg_temp")]
        self.assertFalse(selected.scored.any())

    def test_prediction_contract_all_rows_not_just_scored(self):
        targets = pd.MultiIndex.from_product([["A"], [2012], list(METRICS)],
                                             names=official_score.KEY).to_frame(index=False)
        pred = targets.assign(predicted_value=1.)
        validate_predictions(pred, targets)
        for broken in (pred.iloc[1:], pd.concat([pred, pred.iloc[[0]]]),
                       pred.assign(year=2012.5), pred.assign(predicted_value=np.inf),
                       pred.assign(metric="unknown"), pred.assign(city_id=None)):
            with self.assertRaises(ValueError):
                validate_predictions(broken, targets)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.suite = self.base / "suite"
        self.runs = self.base / "runs"
        self.raw = self.base / "annual.csv"
        annual_fixture().to_csv(self.raw, index=False)
        pd.DataFrame({"city_id": ["A", "B"]}).to_csv(self.base / "cities.csv", index=False)
        config = read_json(ROOT / "evaluation/config.json")
        config["origins"] = [2011]
        config["development_origins"] = [2011]
        config["assessment_origins"] = []
        write_json(self.base / "config.json", config)
        prepare(self.suite, self.base / "config.json", self.raw, self.base / "cities.csv")

    def tearDown(self):
        self.temp.cleanup()

    def test_suite_immutable_and_tampering_detected(self):
        verify_suite(self.suite)
        with self.assertRaises(FileExistsError):
            prepare(self.suite)
        with (self.suite / "origin-2011/train.csv").open("a") as f:
            f.write("tamper")
        with self.assertRaises(ValueError):
            verify_suite(self.suite)

    def test_reference_run_external_csv_bootstrap_and_report(self):
        first, meta = run_suite(self.suite, self.runs, "reference", baseline="average")
        self.assertEqual(meta["summary"]["final"]["mean_score"], 0.)
        pred_dir = self.base / "predictions"
        pred_dir.mkdir()
        (pred_dir / "origin-2011.csv").write_bytes((first / "origin-2011/predictions.csv").read_bytes())
        provenance = {"author": "test", "model_commit": "a" * 40, "parameters": {},
                      "model_repository": "https://example.invalid/fixture",
                      "training_cutoffs": {"2011": 2011}, "dependencies": {"pandas": pd.__version__}}
        second, _ = run_suite(self.suite, self.runs, "<script>alert(1)</script>", prediction_dir=pred_dir,
                             provenance=provenance)
        comparison = compare_runs(first, second, samples=100)
        self.assertEqual(comparison["mean_final_raw_skill_delta_second_minus_first"], 0.)
        self.assertIsNone(comparison["period_bootstrap_95_percent_sensitivity_interval_points"])
        path, table = render_report(self.runs, self.base / "report")
        self.assertEqual(len(table), 2)
        self.assertNotIn("<script>alert(1)</script>", path.read_text(encoding="utf-8"))
        self.assertEqual(digest(second / "scores.json"), read_json(second / "run.json")["artifact_sha256"]["scores.json"])

    def test_actual_worker_adapter_receives_only_history_and_grid(self):
        adapter = self.base / "custom.py"
        adapter.write_text('''def predict(train, targets, origin, seed, level=55):
    assert train.year.max() == origin
    assert set(targets.columns) == {"city_id", "year", "metric"}
    assert targets.year.min() == origin + 1
    return targets.assign(predicted_value=level)
''', encoding="utf-8")
        run, meta = run_suite(self.suite, self.runs, "worker", adapter=str(adapter) + ":predict", params={"level": 60}, assets=ROOT / "evaluation/no-assets.json")
        pred = pd.read_csv(run / "origin-2011/predictions.csv")
        self.assertTrue((pred.predicted_value == 60).all())
        self.assertEqual(meta["status"], "complete")
        self.assertTrue((run / "adapter_snapshot.py").exists())

    def test_failed_adapter_not_in_leaderboard(self):
        with self.assertRaises(RuntimeError):
            run_suite(self.suite, self.runs, "bad", adapter="evaluation.adapters:missing", assets=ROOT / "evaluation/no-assets.json")
        run = next(self.runs.glob("*/run.json"))
        self.assertEqual(read_json(run)["status"], "failed")
        _, table = render_report(self.runs, self.base / "report")
        self.assertTrue(table.empty)

    def test_changed_horizon_and_late_fold_rejected(self):
        for field, value in (("horizon", 8), ("origins", [2012]), ("origins", [2011, 2011])):
            config = read_json(ROOT / "evaluation/config.json")
            config[field] = value
            with self.assertRaises(ValueError):
                validate_config(config)


if __name__ == "__main__":
    unittest.main()
