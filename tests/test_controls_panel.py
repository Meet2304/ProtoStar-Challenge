"""Controls (controls-v1), scorecard, control panel and MLflow mirror."""
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from common import METRICS, wide
from evaluation import controls, scorecard
from evaluation.core import prepare, read_json, write_json
from evaluation.panel import best_control, build_panel, entries, gates, load_current_panel, reuse_key, run_key

CITIES = [f"C{i}" for i in range(8)]


def annual_fixture(first=1900, last=2018, seed=3):
    rng = np.random.default_rng(seed)
    records = []
    for i, city in enumerate(CITIES):
        for year in range(first, last + 1):
            base = 45 + 3 * i + .03 * (year - first) + rng.normal(0, 1)
            miss = 30 if (city == "C7" and year % 9 == 0) else 0       # some invalid years
            records.append({"ID": city, "Year": year, "Miss Tmax": miss, "Miss Tmin": 0, "Miss Prcp": 0,
                            "Avg Temp": base, "ToT Prcp": 30 + 2 * i + rng.normal(0, 4),
                            "Max Tmax": base + 40 + rng.normal(0, 2), "cdd": 900 + 60 * i + 15 * (base - 45)})
    return pd.DataFrame(records)


def train_table(annual):
    out = annual.rename(columns={"ID": "city_id", "Year": "year", "Avg Temp": "annual_avg_temp",
                                 "ToT Prcp": "total_precip_in", "Max Tmax": "hottest_day_temp"})
    return out[["city_id", "year", *METRICS]]


def targets_for(origin, cities=CITIES):
    return pd.MultiIndex.from_product([cities, range(origin + 1, origin + 8), list(METRICS)],
                                      names=["city_id", "year", "metric"]).to_frame(index=False)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.train = train_table(annual_fixture(last=1997))
        self.targets = targets_for(1997)

    def test_every_control_is_finite_complete_and_deterministic(self):
        for method in controls.CONTROLS:
            a = controls.forecast(self.train, self.targets, 1997, 42, method=method)
            b = controls.forecast(self.train, self.targets, 1997, 42, method=method)
            self.assertEqual(list(a.columns), ["city_id", "year", "metric", "predicted_value"], method)
            self.assertEqual(len(a), len(self.targets), method)
            self.assertTrue(np.isfinite(a.predicted_value).all(), method)
            pd.testing.assert_frame_equal(a, b)

    def test_method_is_keyword_only_and_validated(self):
        with self.assertRaises(TypeError):
            controls.forecast(self.train, self.targets, 1997, 42, "average30")
        with self.assertRaises(ValueError):
            controls.forecast(self.train, self.targets, 1997, 42, method="oracle")

    def test_refuses_rows_after_origin(self):
        with self.assertRaises(ValueError):
            controls.forecast(self.train, self.targets, 1990, 42, method="mean10")

    def test_city_without_history_still_gets_finite_values(self):
        train = pd.concat([self.train, pd.DataFrame({"city_id": ["EMPTY"], "year": [1997],
                                                     **{m: [np.nan] for m in METRICS}})])
        targets = targets_for(1997, CITIES + ["EMPTY"])
        for method in ("ridge_pooled", "starter_rf", "mean10", "blend"):
            out = controls.forecast(train, targets, 1997, 42, method=method)
            self.assertTrue(np.isfinite(out.predicted_value).all(), method)


class RidgeBoundaryTests(unittest.TestCase):
    """Features at pseudo-origin t may use only years <= t; the target is year t+k."""

    def setUp(self):
        self.table = wide(train_table(annual_fixture(last=1997)), "annual_avg_temp")
        self.t, self.k, self.origin = 1985, 3, 1997

    def features(self, table):
        return controls.ridge_features(table, self.t)

    def perturbed(self, year, delta=7.0):
        table = self.table.copy()
        table[year] = table[year] + delta
        return table

    def test_years_after_t_do_not_change_features_or_reference(self):
        feats, ref = self.features(self.table)
        for year in range(self.t + 1, self.origin + 1):
            f2, r2 = self.features(self.perturbed(year))
            pd.testing.assert_frame_equal(feats, f2)
            pd.testing.assert_series_equal(ref, r2)

    def test_year_t_changes_features(self):
        feats, _ = self.features(self.table)
        f2, _ = self.features(self.perturbed(self.t))
        self.assertFalse(feats.equals(f2))

    def examples(self, table):
        """Pseudo-origin t's rows from the real training-example builder used by ridge_pooled."""
        cache = controls.ridge_cache(table, self.origin)
        X, y, index = controls.ridge_training_examples(table, cache, self.origin, self.k)
        rows = [i for i, (t, _) in enumerate(index) if t == self.t]
        return X[rows], y[rows]

    def test_target_moves_only_with_year_t_plus_k(self):
        X, y = self.examples(self.table)
        X2, y2 = self.examples(self.perturbed(self.t + self.k, 7.0))
        np.testing.assert_allclose(X2, X)
        np.testing.assert_allclose(y2 - y, 7.0)
        for year in [y for y in range(self.t + 1, self.origin + 1) if y != self.t + self.k]:
            X3, y3 = self.examples(self.perturbed(year))
            np.testing.assert_allclose(X3, X)
            np.testing.assert_allclose(y3, y)

    def test_builder_never_uses_targets_after_origin(self):
        cache = controls.ridge_cache(self.table, self.origin)
        for lead in range(1, 8):
            _, _, index = controls.ridge_training_examples(self.table, cache, self.origin, lead)
            self.assertTrue(all(t + lead <= self.origin for t, _ in index))

    def test_missing_recent_years_and_tiny_history_fall_back(self):
        table = self.table.copy()
        table.loc[:, 1995:1997] = np.nan
        feats, ref = controls.ridge_features(table, 1997)
        self.assertTrue(np.isfinite(feats.to_numpy()).all())
        years = list(range(1998, 2005))
        for first in (1960, 1986, 1997):               # 38, 12 and 1 years of history
            tiny = self.table.loc[:, first:1997]
            out = controls.ridge_pooled(tiny, 1997, years)
            self.assertTrue(np.isfinite(out.to_numpy()).all(), first)
        # 12 years: a valid 30-year reference exists but too few training rows -> exactly average30
        tiny = self.table.loc[:, 1986:1997]
        pd.testing.assert_frame_equal(controls.ridge_pooled(tiny, 1997, years),
                                      controls.average_forecast(tiny, 1997, years))


class ScorecardAggregationTests(unittest.TestCase):
    def test_composite_equal_weights_metrics_regardless_of_units(self):
        rows = []
        for metric, scale in zip(METRICS, (1, 1, 1, 1000)):
            for i in range(10):
                good = metric != "cdd"
                rows.append({"metric": metric, "absolute_error": scale * (0.5 if good else 2.0),
                             "baseline_absolute_error": scale * 1.0})
        skill, n = scorecard._ratio_skill(pd.DataFrame(rows))
        self.assertAlmostEqual(skill, (0.5 * 3 + (-1.0)) / 4)
        self.assertEqual(n, 40)

    def test_zero_reference_is_undefined_not_an_error(self):
        frame = pd.DataFrame([{"metric": m, "absolute_error": 1.0, "baseline_absolute_error": 0.0} for m in METRICS])
        self.assertIsNone(scorecard._ratio_skill(frame)[0])

    def final_rows(self):
        """Two cities, hand-computable errors. Reference error is 1 everywhere."""
        rows = []
        for city, err in (("A", 0.5), ("B", 2.0)):
            for metric in METRICS:
                for signed in (err, err):                          # two rows each, same signed error
                    rows.append({"city_id": city, "metric": metric, "origin": 1983,
                                 "absolute_error": abs(signed), "baseline_absolute_error": 1.0,
                                 "signed_error": signed})
        return pd.DataFrame(rows)

    def test_hand_computed_bias_zones_win_rate_and_p90(self):
        final = self.final_rows()
        bias = scorecard.bias_section(final)
        self.assertAlmostEqual(bias["cdd"]["mean_signed_error"], 1.25)      # (0.5+0.5+2+2)/4
        self.assertAlmostEqual(bias["cdd"]["relative_to_reference_mae"], 1.25)
        self.assertEqual(bias["cdd"]["n"], 4)
        zones = pd.Series({"A": "West-North", "B": "East-South"})
        spatial = scorecard.spatial_section(final, zones)
        self.assertAlmostEqual(spatial["West-North"]["final_skill"], 0.5)    # 1 - 0.5/1 on every metric
        self.assertAlmostEqual(spatial["East-South"]["final_skill"], -1.0)
        rob = scorecard.robustness_section(final)
        self.assertAlmostEqual(rob["city_metric_win_rate"], 0.5)            # A wins all 4, B none
        self.assertEqual(rob["n_city_metrics"], 8)
        # p90 of |e| over {0.5,0.5,2,2} = 2.0 (linear interpolation), reference p90 = 1
        self.assertAlmostEqual(rob["p90_abs_error_ratio_by_metric"]["cdd"], 2.0)
        self.assertAlmostEqual(rob["worst_decile_city_skill"], -1.0 + 0.1 * 1.5)   # quantile(-1, 0.5; .1)

    def test_ties_are_not_wins(self):
        final = self.final_rows()
        final["absolute_error"] = 1.0
        self.assertEqual(scorecard.robustness_section(final)["city_metric_win_rate"], 0.0)

    def test_undefined_ratios_serialize_as_null(self):
        self.assertIsNone(scorecard._safe_ratio(3.0, 0.0))     # e.g. a p90 reference error of 0
        self.assertIsNone(scorecard._safe_ratio(float("nan"), 1.0))
        card = scorecard._finite({"a": float("inf"), "b": [np.float64("nan"), 1.5], "c": np.int64(2)})
        self.assertEqual(json.loads(json.dumps(card, allow_nan=False)), {"a": None, "b": [None, 1.5], "c": 2})


class PanelAndTrackingTests(unittest.TestCase):
    """A real (synthetic-data) suite, panel subset, scorecard and MLflow mirror."""

    @classmethod
    def setUpClass(cls):
        # MLflow's SQLite engine can keep the store open on Windows until the process exits.
        cls.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        base = Path(cls.temp.name)
        annual_fixture().to_csv(base / "annual.csv", index=False)
        pd.DataFrame({"city_id": CITIES}).to_csv(base / "cities.csv", index=False)
        config = read_json(ROOT / "evaluation/config.json")
        write_json(base / "config.json", config)
        cls.suite, cls.runs, cls.panels = base / "suite", base / "runs", base / "panels"
        prepare(cls.suite, base / "config.json", base / "annual.csv", base / "cities.csv")
        cls.subset = [e[0] for e in entries("A")]
        cls.path, cls.panel = build_panel(cls.suite, cls.runs, profile="A", echo=lambda *_: None,
                                          panel_root=cls.panels)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_tier_a_panel_ships_alone_and_passes_sanity_gates(self):
        self.assertEqual(sorted(self.panel["entries"]), sorted(self.subset))
        self.assertNotIn("ridge_pooled", self.panel["entries"])
        self.assertEqual(self.panel["profile"], "A")
        loaded = load_current_panel(self.suite, self.runs, "A", panel_root=self.panels)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["best_control"], self.panel["best_control"])
        self.assertIsNone(load_current_panel(self.suite, self.runs, "full", panel_root=self.panels))
        for fold in read_json(Path(self.panel["entries"]["average30"]["run_dir"]) / "scores.json"):
            for skill in fold["skill"].values():
                self.assertLess(abs(skill), 1e-9)

    def test_second_build_reuses_current_runs(self):
        _, again = build_panel(self.suite, self.runs, profile="A", echo=lambda *_: None, panel_root=self.panels)
        for entry_id in self.subset:
            self.assertTrue(again["entries"][entry_id]["reused"])
            self.assertEqual(again["entries"][entry_id]["run_id"], self.panel["entries"][entry_id]["run_id"])

    def test_tampered_or_mismatched_panel_is_not_trusted(self):
        path = self.path
        original = path.read_bytes()
        try:
            panel = json.loads(original)
            panel["entries"]["mean10"]["run_fingerprint"] = "0" * 64        # wrong fingerprint
            panel["entries"]["mean10"]["headline_final_raw_skill"] = 9.9     # forged summary
            path.write_text(json.dumps(panel), encoding="utf-8")
            self.assertIsNone(load_current_panel(self.suite, self.runs, "A", panel_root=self.panels))
            panel = json.loads(original)
            panel["entries"]["mean10"]["headline_final_raw_skill"] = 9.9     # forged summary only
            path.write_text(json.dumps(panel), encoding="utf-8")
            loaded = load_current_panel(self.suite, self.runs, "A", panel_root=self.panels)
            self.assertNotEqual(loaded["entries"]["mean10"]["headline_final_raw_skill"], 9.9)  # re-derived
            path.write_text("{not json", encoding="utf-8")
            self.assertIsNone(load_current_panel(self.suite, self.runs, "A", panel_root=self.panels))
        finally:
            path.write_bytes(original)

    def test_reuse_key_changes_with_spec_sources_environment(self):
        meta = read_json(Path(self.panel["entries"]["mean10"]["run_dir"]) / "run.json")
        key = run_key(meta)
        self.assertEqual(key, self.panel["entries"]["mean10"]["reuse_key"])
        pins = dict(meta["source_pins"])
        env = meta["installed_environment"]["sha256"]
        self.assertNotEqual(key, reuse_key(meta["suite_id"], "development", {**meta["model_spec"], "params": {"method": "blend"}}, pins, env))
        self.assertNotEqual(key, reuse_key(meta["suite_id"], "development", meta["model_spec"], {**pins, "code/combo_model.py": "x"}, env))
        self.assertNotEqual(key, reuse_key(meta["suite_id"], "development", meta["model_spec"], pins, "other-env"))
        # An unrelated teammate model file does not make the panel stale.
        self.assertEqual(key, reuse_key(meta["suite_id"], "development", meta["model_spec"], {**pins, "code/teammate.py": "y"}, env))

    def test_scorecard_is_written_outside_the_run_and_is_consistent(self):
        run_dir = Path(self.panel["entries"]["trend30"]["run_dir"])
        before = (run_dir / "run.json").read_bytes()
        out = Path(self.temp.name) / "cards"
        path, card = scorecard.write_scorecard(run_dir, out)
        self.assertEqual((run_dir / "run.json").read_bytes(), before)
        self.assertTrue(path.is_file() and path.parent == out)
        meta = read_json(run_dir / "run.json")
        self.assertEqual(card["fidelity"]["headline_final_raw_skill"], meta["summary"]["final"]["mean_skill_unclipped"])
        self.assertEqual(card["run_fingerprint"], meta["run_fingerprint"])
        self.assertTrue(all(math.isfinite(v) for v in scorecard.flat_metrics(card).values()))

    @unittest.skipUnless(importlib.util.find_spec("mlflow"), "MLflow not installed (optional backend)")
    def test_mlflow_mirror_is_idempotent_and_backdated(self):
        from evaluation import tracking
        store = Path(self.temp.name) / "mlflow"
        store.mkdir()
        old = os.environ.get("MLFLOW_TRACKING_URI")
        os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + (store / "m.db").as_posix()
        try:
            run_dir = Path(self.panel["entries"]["persistence"]["run_dir"])
            first = tracking.log_run(run_dir)
            self.assertIsNotNone(first)
            self.assertIsNone(tracking.log_run(run_dir))
            client = tracking._client()
            meta = read_json(run_dir / "run.json")
            run = client.get_run(first)
            self.assertEqual(run.info.start_time, tracking._millis(meta["created_utc"]))
            self.assertEqual(run.data.tags["role"], "control")
            self.assertEqual(run.data.tags[tracking.COMPLETE_TAG], "true")
            self.assertIn("headline_final_raw_skill", run.data.metrics)
            # A partial mirror (crash before the completion tag) is replaced on retry.
            other = Path(self.panel["entries"]["trend30"]["run_dir"])
            other_meta = read_json(other / "run.json")
            experiment = client.get_experiment_by_name(tracking.experiment_name(other_meta)).experiment_id
            partial = client.create_run(experiment, tags={tracking.FINGERPRINT_TAG: other_meta["run_fingerprint"]})
            redo = tracking.log_run(other)
            self.assertIsNotNone(redo)
            self.assertNotEqual(redo, partial.info.run_id)
            self.assertEqual(client.get_run(partial.info.run_id).info.lifecycle_stage, "deleted")
            self.assertIsNone(tracking.log_run(other))
        finally:
            if old is None:
                os.environ.pop("MLFLOW_TRACKING_URI", None)
            else:
                os.environ["MLFLOW_TRACKING_URI"] = old

    def test_mirror_failure_never_raises(self):
        from evaluation import tracking
        self.assertIsNone(tracking.try_log_run(Path(self.temp.name) / "missing-run", echo=lambda *_: None))


SUITE_V3 = ROOT / "outputs/evaluation/suite-v3"


@unittest.skipUnless((SUITE_V3 / "manifest.json").is_file(), "local suite-v3 not present")
class RealSuiteTests(unittest.TestCase):
    """On the real suite-v3 development origins: references match the prebuilt baselines within
    1e-6, and average30 scores |skill| < 1e-9 with the official scorer. Writes only to a temp dir."""

    def test_references_and_denominator_on_suite_v3(self):
        from evaluation.core import score_fold, verify_suite
        from evaluation.protocol import phase_origins
        manifest = verify_suite(SUITE_V3)
        with tempfile.TemporaryDirectory() as temp:
            for origin in phase_origins(manifest, "development"):
                fold = SUITE_V3 / f"origin-{origin}"
                train = pd.read_csv(fold / "train.csv", dtype={"city_id": str})
                targets = pd.read_csv(fold / "targets.csv", dtype={"city_id": str, "metric": str})
                for method, name in (("average30", "average"), ("persistence", "last_year"), ("trend30", "trend")):
                    made = controls.forecast(train, targets, origin, 42, method=method)
                    built = pd.read_csv(fold / f"baselines/{name}.csv", dtype={"city_id": str, "metric": str})
                    key = ["city_id", "year", "metric"]
                    joined = made.merge(built, on=key, suffixes=("", "_built"), validate="one_to_one")
                    self.assertEqual(len(joined), len(built))
                    self.assertLessEqual((joined.predicted_value - joined.predicted_value_built).abs().max(), 1e-6)
                out = Path(temp) / str(origin)
                out.mkdir()
                pred = controls.forecast(train, targets, origin, 42, method="average30")
                for result in score_fold(pred, fold, origin, out):
                    for skill in result["skill"].values():
                        self.assertLess(abs(skill), 1e-9)


class EntryListTests(unittest.TestCase):
    def test_profiles(self):
        full = [e[0] for e in entries("full")]
        self.assertEqual(full[:len(controls.CONTROLS)], list(controls.CONTROLS))
        tier_a = [e[0] for e in entries("A")]
        self.assertEqual(tier_a, ["average30", "persistence", "trend30", "mean10", "blend", "combo", "nested"])
        with self.assertRaises(ValueError):
            entries("B")


class GateTests(unittest.TestCase):
    def card(self, head, metric_skills):
        return {"fidelity": {"headline_final_raw_skill": head},
                "per_metric": {"final": dict(zip(METRICS, metric_skills))}}

    def panel(self, best=0.02):
        return {"best_control": "blend", "entries": {"blend": {"role": "control", "headline_final_raw_skill": best}}}

    def comparison(self, deltas):
        return {"periods": [{"origin": o, "raw_skill_delta": d} for o, d in zip((1983, 1990, 1997), deltas)],
                "mean_final_raw_skill_delta_second_minus_first": sum(deltas) / len(deltas)}

    def test_all_gates_pass(self):
        results = gates(self.card(0.05, [0.05, 0.0, -0.01, 0.1]), self.panel(), self.comparison([0.01, 0.02, 0.0]))
        self.assertTrue(all(ok for _, ok, _ in results), results)

    def test_each_gate_can_fail(self):
        g = lambda *a: [ok for _, ok, _ in gates(*a)]
        self.assertFalse(g(self.card(0.0, [0] * 4), self.panel(-1), self.comparison([1, 1, 1]))[0])   # G1 tie
        self.assertFalse(g(self.card(0.02, [0] * 4), self.panel(0.02), self.comparison([1, 1, 1]))[1])  # G2 tie
        self.assertFalse(g(self.card(0.1, [0.3, 0, 0, -0.03]), self.panel(), self.comparison([1, 1, 1]))[2])
        self.assertFalse(g(self.card(0.1, [0] * 4), self.panel(), self.comparison([0.05, 0.05, -0.001]))[3])  # loses 1997
        self.assertFalse(g(self.card(0.1, [0] * 4), self.panel(), self.comparison([0.2, -0.01, -0.0]))[3])    # 1 win

    def test_best_control_excludes_denominator_and_breaks_ties_alphabetically(self):
        entries_ = {"average30": {"role": "control", "headline_final_raw_skill": 0.0},
                    "mean10": {"role": "control", "headline_final_raw_skill": -0.1},
                    "trend30": {"role": "control", "headline_final_raw_skill": -0.05},
                    "blend": {"role": "control", "headline_final_raw_skill": -0.05},
                    "combo": {"role": "incumbent", "headline_final_raw_skill": 0.5}}
        self.assertEqual(best_control(entries_), "blend")


if __name__ == "__main__":
    unittest.main()
