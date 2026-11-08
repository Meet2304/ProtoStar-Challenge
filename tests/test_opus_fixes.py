"""Regression checks for the failures reproduced in the independent Opus audit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from common import DataRuleViolation, METRICS, assert_no_post_cutoff, effective_cutoff_year, wide, thirty_year_mean
from combo_model import trend_forecast
from evaluation.core import (prepare, read_json, write_json, run_suite, leaderboard,
                             load_annual, build_fold, compare_runs, digest, verify_suite, verify_run)
from evaluation.protocol import (create_freeze, verify_freeze, model_spec, committed_pins,
                                 phase_origins, validate_external_provenance)
from evaluation.report import render_report
from test_evaluation import annual_fixture


def external_provenance(origins):
    return {"author": "fixture author", "model_commit": "a" * 40,
            "model_repository": "https://example.invalid/fixture",
            "parameters": {"example": True}, "dependencies": {"pandas": pd.__version__},
            "training_cutoffs": {str(y): y for y in origins}}


class GuardTests(unittest.TestCase):
    def test_origin_context_tightens_existing_guard_and_cannot_relax_2018(self):
        with patch.dict(os.environ, {"PROTOSTAR_FOLD_CUTOFF": "2011"}):
            self.assertEqual(effective_cutoff_year(), 2011)
            assert_no_post_cutoff(pd.DataFrame({"year": [2011]}), "year")
            with self.assertRaises(DataRuleViolation):
                assert_no_post_cutoff(pd.DataFrame({"year": [2012]}), "year")
        for invalid in ("2019", "", "2011.5", "unknown"):
            with patch.dict(os.environ, {"PROTOSTAR_FOLD_CUTOFF": invalid}):
                with self.assertRaises(DataRuleViolation):
                    effective_cutoff_year()

    def test_invalid_weather_years_fail_closed(self):
        for year in (None, float("inf"), 2011.5, "bad"):
            with self.assertRaises(DataRuleViolation):
                assert_no_post_cutoff(pd.DataFrame({"year": [year]}), "year")

    def test_frozen_model_parameters_cannot_silently_coerce_or_ignore_bad_values(self):
        for params in ([], {"x": float("nan")}):
            with self.assertRaises(ValueError):
                model_spec([2011], adapter="evaluation.nested:forecast", params=params)
        with self.assertRaisesRegex(ValueError, "only to adapters"):
            model_spec([2011], baseline="average", params={"x": 1})


class OpusRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.suite = self.base / "suite"
        self.runs = self.base / "runs"
        self.origins = [1990, 1997, 2004]
        raw = self.base / "annual.csv"
        annual_fixture().to_csv(raw, index=False)
        self.raw = raw
        cities = self.base / "cities.csv"
        pd.DataFrame({"city_id": ["A", "B"]}).to_csv(cities, index=False)
        config = read_json(ROOT / "evaluation/config.json")
        config.update(origins=self.origins + [2011], development_origins=self.origins,
                      assessment_origins=[2011])
        write_json(self.base / "config.json", config)
        prepare(self.suite, self.base / "config.json", raw, cities)

    def tearDown(self):
        self.temp.cleanup()

    def adapter(self, text, name="model.py"):
        path = self.base / name
        path.write_text(text, encoding="utf-8")
        return str(path) + ":forecast"

    def test_standard_loader_leak_fails_even_at_adapter_import(self):
        folder = self.base / "legacy_data"
        folder.mkdir()
        load_annual(self.raw).to_csv(folder / "annual_valid.csv", index=False)
        adapter = self.adapter(f'''from pathlib import Path
import common
common.DATA_DIR = Path({str(folder)!r})
full = common.load_valid_annual()
def forecast(train, targets, origin, seed):
    return targets.assign(predicted_value=0.)
''')
        with self.assertRaises(RuntimeError):
            run_suite(self.suite, self.runs, "accidental leak", adapter=adapter, assets=ROOT / "evaluation/no-assets.json")
        meta = read_json(next(self.runs.glob("*/run.json")))
        self.assertEqual(meta["status"], "failed")
        log = next(self.runs.glob("*/origin-1990/adapter.log")).read_text(encoding="utf-8")
        self.assertIn("DataRuleViolation", log)
        self.assertIn("1990-12-31", log)

    def test_worker_receives_no_private_label_sibling_or_repo_cwd(self):
        adapter = self.adapter('''import sys
from pathlib import Path
def forecast(train, targets, origin, seed):
    fold = Path(sys.argv[sys.argv.index("--train") + 1]).parent
    assert not (fold / "private").exists()
    assert not Path("data/annual_valid.csv").exists()
    assert train.year.max() <= origin
    return targets.assign(predicted_value=55.)
''')
        path, meta = run_suite(self.suite, self.runs, "compliant", adapter=adapter, assets=ROOT / "evaluation/no-assets.json")
        self.assertEqual(meta["origins"], self.origins)
        self.assertFalse((path / "origin-2011").exists())
        self.assertTrue((path / "code_snapshot/src/protostar_challenge/config.py").exists())

    def test_combo_rejects_unknown_params_before_creating_a_run(self):
        with self.assertRaisesRegex(ValueError, "no tunable parameters"):
            run_suite(self.suite, self.runs, "bad parameters", adapter="evaluation.adapters:combo", params={"x": 1})
        self.assertFalse(self.runs.exists())

    def test_custom_adapter_requires_explicit_dependency_declaration(self):
        adapter = self.adapter('''def forecast(train, targets, origin, seed):
    return targets.assign(predicted_value=55.)
''')
        with self.assertRaisesRegex(ValueError, "custom adapters require --assets"):
            run_suite(self.suite, self.runs, "undeclared", adapter=adapter)
        self.assertFalse(self.runs.exists())

    def test_report_links_are_self_contained_without_cross_drive_relative_paths(self):
        path, meta = run_suite(self.suite, self.runs, "reference", baseline="average")
        with patch("os.path.relpath", side_effect=ValueError("different drives")):
            report, table = render_report(self.runs, self.base / "report")
        copied = report.parent / "metadata" / f"{meta['run_id']}.json"
        self.assertEqual(copied.read_bytes(), (path / "run.json").read_bytes())
        self.assertIn(f"metadata/{meta['run_id']}.json", report.read_text(encoding="utf-8"))

    def test_timeout_has_clean_cli_error_and_failed_record(self):
        adapter = self.adapter('''import time
def forecast(train, targets, origin, seed):
    time.sleep(30)
    return targets.assign(predicted_value=1.)
''')
        result = subprocess.run([sys.executable, str(ROOT / "code/evaluate.py"), "run", "--suite", str(self.suite),
                                 "--runs", str(self.runs), "--name", "timeout", "--adapter", adapter,
                                 "--timeout", "0.1", "--assets", str(ROOT / "evaluation/no-assets.json")], cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 1)
        self.assertIn("timed out", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        meta = read_json(next(self.runs.glob("*/run.json")))
        self.assertEqual(meta["status"], "failed")
        self.assertIn("adapter.log", meta["error"])

    def test_boolean_adapter_values_are_rejected_before_csv_coercion(self):
        for expression in ("targets.assign(predicted_value=True)",
                           "targets.assign(predicted_value=55., year=True)"):
            adapter = self.adapter(f'''def forecast(train, targets, origin, seed):
    return {expression}
''')
            with self.assertRaises(RuntimeError):
                run_suite(self.suite, self.runs, "broken bool", adapter=adapter, assets=ROOT / "evaluation/no-assets.json")

    def test_invalid_return_and_nan_are_preserved_as_failed_runs(self):
        for expression in ("None", "targets.assign(predicted_value=float('nan'))"):
            adapter = self.adapter(f'''def forecast(train, targets, origin, seed):
    return {expression}
''')
            with self.assertRaises((ValueError, RuntimeError)):
                run_suite(self.suite, self.runs, "invalid output", adapter=adapter, assets=ROOT / "evaluation/no-assets.json")
        self.assertTrue(all(read_json(p)["status"] == "failed" for p in self.runs.glob("*/run.json")))
        self.assertTrue(leaderboard(self.runs).empty)

    def test_rank_preserves_negative_skill_and_has_numerical_ties(self):
        def record(name, raw, clipped):
            folder = self.runs / name
            folder.mkdir(parents=True)
            write_json(folder / "scores.json", [])
            write_json(folder / "run.json", {"status": "complete", "run_id": name, "model": name,
                "suite_id": "example", "artifact_sha256": {"scores.json": digest(folder / "scores.json")},
                "summary": {"final": {"mean_skill_unclipped": raw,
                "mean_score": clipped, "min_score": 0., "max_score": clipped, "n_folds": 5}}})
        record("baseline", 0., 0.)
        record("tiny-rounding", 1e-12, 1e-10)
        record("volatile-worse", -.1, 20.)
        table = leaderboard(self.runs).set_index("model")
        self.assertEqual(table.loc["baseline", "internal_rank"], 1)
        self.assertEqual(table.loc["tiny-rounding", "internal_rank"], 1)
        self.assertGreater(table.loc["volatile-worse", "internal_rank"], 1)

    def test_external_csv_requires_cutoffs_and_retains_original_bytes(self):
        directory = self.base / "predictions"
        directory.mkdir()
        for origin in self.origins:
            (directory / f"origin-{origin}.csv").write_bytes((self.suite / f"origin-{origin}/baselines/average.csv").read_bytes())
        with self.assertRaisesRegex(ValueError, "provenance"):
            run_suite(self.suite, self.runs, "anonymous", prediction_dir=directory)
        provenance = external_provenance(self.origins)
        provenance["training_cutoffs"]["1990"] = 1991
        with self.assertRaisesRegex(ValueError, "cutoff"):
            run_suite(self.suite, self.runs, "future", prediction_dir=directory, provenance=provenance)
        provenance = external_provenance(self.origins)
        path, meta = run_suite(self.suite, self.runs, "declared", prediction_dir=directory, provenance=provenance)
        original = path / "origin-1990/original_external_predictions.csv"
        self.assertEqual(original.read_bytes(), (directory / "origin-1990.csv").read_bytes())
        self.assertEqual(meta["model_spec"]["original_prediction_sha256"]["origin-1990.csv"], digest(original))
        self.assertEqual(meta["prediction_dir"], str(directory.resolve()))

    def test_comparison_uses_raw_period_differences_when_both_models_clip_to_zero(self):
        paths = []
        for offset in (1000., 500.):
            directory = self.base / f"predictions-{offset}"
            directory.mkdir()
            for origin in self.origins:
                frame = pd.read_csv(self.suite / f"origin-{origin}/baselines/average.csv")
                frame.predicted_value += offset
                frame.to_csv(directory / f"origin-{origin}.csv", index=False)
            path, meta = run_suite(self.suite, self.runs, str(offset), prediction_dir=directory,
                                   provenance=external_provenance(self.origins))
            self.assertEqual(meta["summary"]["final"]["mean_score"], 0.)
            paths.append(path)
        result = compare_runs(*paths, samples=200)
        self.assertGreater(result["mean_final_raw_skill_delta_second_minus_first"], 0.)
        self.assertEqual(result["positive_periods"], 3)
        self.assertEqual(len(result["periods"]), 3)
        self.assertIsNotNone(result["period_bootstrap_95_percent_sensitivity_interval_points"])
        self.assertTrue(all(len(p["lead_skill_deltas"]) == 4 for p in result["periods"]))
        self.assertNotIn("bootstrap_95_percent_interval", result)

    def test_assessment_is_not_evaluated_without_a_freeze(self):
        with self.assertRaisesRegex(ValueError, "freeze"):
            run_suite(self.suite, self.runs, "unfrozen", baseline="average", phase="assessment")
        self.assertFalse(self.runs.exists())

    def test_reports_and_comparisons_reject_changed_scores_and_metadata(self):
        first, _ = run_suite(self.suite, self.runs, "reference", baseline="average")
        second, _ = run_suite(self.suite, self.runs, "other", baseline="trend")
        verify_run(first)
        (first / "scores.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact"):
            leaderboard(self.runs)
        with self.assertRaisesRegex(ValueError, "artifact"):
            compare_runs(first, second)
        metadata = read_json(second / "run.json")
        metadata["summary"]["final"]["mean_skill_unclipped"] = 1.
        write_json(second / "run.json", metadata)
        with self.assertRaisesRegex(ValueError, "metadata fingerprint"):
            verify_run(second)

    def init_committed_model(self):
        repo = self.base / "model-repo"
        (repo / "code").mkdir(parents=True)
        (repo / "code/model.py").write_text("MODEL_VERSION = 1\n", encoding="utf-8")
        (repo / "evaluation").mkdir()
        (repo / "evaluation/no-assets.json").write_bytes((ROOT / "evaluation/no-assets.json").read_bytes())
        for command in (["git", "init", "-q"], ["git", "add", "code/model.py", "evaluation/no-assets.json"],
                        ["git", "-c", "user.name=Evaluation Test", "-c", "user.email=evaluation@example.invalid",
                         "-c", "commit.gpgsign=false", "commit", "-qm", "fixture model"]):
            subprocess.run(command, cwd=repo, capture_output=True, check=True)
        return repo

    def test_committed_freeze_rejects_changed_params_source_and_untracked_dependencies(self):
        repo = self.init_committed_model()
        manifest = verify_suite(self.suite)
        freeze = self.base / "freeze.json"
        record = create_freeze(self.suite, freeze, adapter="model:forecast", params={"x": 1}, root=repo, assets="evaluation/no-assets.json")
        spec = model_spec([2011], adapter="model:forecast", params={"x": 1}, root=repo, assets="evaluation/no-assets.json")
        self.assertEqual(verify_freeze(freeze, manifest, spec, root=repo)["freeze_id"], record["freeze_id"])
        with self.assertRaisesRegex(ValueError, "match"):
            verify_freeze(freeze, manifest, model_spec([2011], adapter="model:forecast", params={"x": 2}, assets="evaluation/no-assets.json"), root=repo)
        (repo / "code/model.py").write_text("MODEL_VERSION = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "committed"):
            verify_freeze(freeze, manifest, spec, root=repo)
        (repo / "code/model.py").write_text("MODEL_VERSION = 1\n", encoding="utf-8")
        (repo / "code/new_dependency.py").write_text("VALUE = 1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "committed"):
            committed_pins(root=repo)

    def test_every_assessment_access_is_logged_before_scoring(self):
        repo = self.init_committed_model()
        freeze = self.base / "freeze.json"
        record = create_freeze(self.suite, freeze, baseline="average", root=repo)
        # Real Git/content verification above, injected model-repo location only.
        def verify_at_fixture(path, manifest, spec):
            return verify_freeze(path, manifest, spec, root=repo)
        with patch("evaluation.core.verify_freeze", side_effect=verify_at_fixture), \
             patch("evaluation.core.source_pins", return_value=record["source_pins"]):
            for _ in range(2):
                path, meta = run_suite(self.suite, self.runs, "frozen", baseline="average",
                                       phase="assessment", freeze=freeze)
                self.assertEqual(meta["origins"], [2011])
                self.assertTrue((path / "freeze.json").exists())
                self.assertFalse((path / "origin-1990").exists())
        events = list((self.suite / "assessment_access").glob("*.json"))
        self.assertEqual(len(events), 2)
        self.assertTrue(all(read_json(p)["freeze_id"] == record["freeze_id"] for p in events))

    def test_zero_reference_is_an_explicit_error(self):
        annual = load_annual(self.raw)
        annual[list(METRICS)] = 1.
        with self.assertRaisesRegex(ValueError, "zero"):
            build_fold(annual, ["A", "B"], 2011)


@unittest.skipUnless(os.environ.get("PROTOSTAR_RUN_REAL_DATA_TESTS") == "1",
                     "real-data tests require explicit PROTOSTAR_RUN_REAL_DATA_TESTS=1; may read reserved years")
class PublishedReferenceTests(unittest.TestCase):
    """No downloads in tests; use the already-present pinned cache when available."""
    @unittest.skipUnless((ROOT / "data/raw/annual_indices/train-00000-of-00001.parquet").exists(), "pinned raw cache unavailable")
    def test_published_baseline_counts_and_practice_cdd(self):
        annual = load_annual(ROOT / "data/raw/annual_indices/train-00000-of-00001.parquet")
        missing = sum(int(thirty_year_mean(wide(annual, m), 2018).isna().sum()) for m in METRICS)
        self.assertEqual(missing, 9)
        self.assertEqual(missing * 7, 63)
        table = wide(annual, "cdd")
        history = table.loc[:, :2010]
        baseline = thirty_year_mean(history, 2010)
        actual = table[list(range(2011, 2019))].where(baseline.notna(), axis=0)
        base_error = actual.sub(baseline, axis=0).abs().to_numpy()
        trend_error = (actual - trend_forecast(history, 2010, list(range(2011, 2019)))).abs().to_numpy()
        self.assertAlmostEqual(float(np.nanmean(base_error)), 211.69, delta=.02)
        self.assertAlmostEqual(float(np.nanmean(trend_error)), 174.60, delta=.02)

    @unittest.skipUnless((ROOT / "data/annual_valid.csv").exists() and (ROOT / "submission.csv").exists(), "curated submission fixture unavailable")
    def test_existing_combo_adapter_preserves_the_submission_forecast(self):
        from common import load_valid_annual
        from evaluation.adapters import combo
        train = load_valid_annual()
        targets = pd.MultiIndex.from_product([sorted(train.city_id.unique()), range(2019, 2026), list(METRICS)],
                                             names=["city_id", "year", "metric"]).to_frame(index=False)
        predictions = combo(train, targets, 2018)
        existing = pd.read_csv(ROOT / "submission.csv")
        joined = existing.merge(predictions, on=["city_id", "year", "metric"], validate="one_to_one", suffixes=("_old", "_new"))
        self.assertEqual(len(joined), 5880)
        self.assertLessEqual(float((joined.predicted_value_old - joined.predicted_value_new).abs().max()), .00050001)


class ReferenceSanityTests(unittest.TestCase):
    def test_last_year_uses_last_valid_value_for_each_metric(self):
        from evaluation.adapters import reference
        rows = []
        for year, value in ((2009, 1.), (2010, 3.), (2011, np.nan)):
            rows.append({"city_id": "A", "year": year, **{m: value for m in METRICS}})
        targets = pd.MultiIndex.from_product([["A"], range(2012, 2019), list(METRICS)],
                                             names=["city_id", "year", "metric"]).to_frame(index=False)
        actual = reference(pd.DataFrame(rows), targets, 2011, method="last_year")
        self.assertTrue((actual.predicted_value == 3.).all())


if __name__ == "__main__":
    unittest.main()
