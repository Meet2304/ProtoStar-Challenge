"""Tests for code/evaluation/insights.py: arithmetic identities and that each diagnosis fires on its cause."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from common import METRICS  # noqa: E402
from evaluation import insights, reward  # noqa: E402

ORIGINS = (1983, 1990, 1997)


def synthetic(error_fn, cities=60, seed=0, anomaly=0.0):
    """Scored rows for 3 origins x 7 leads x cities x 4 metrics. error_fn(origin, city, year, rng) -> error."""
    rng = np.random.default_rng(seed)
    rows = []
    for origin in ORIGINS:
        year_shock = {y: rng.normal(0, 0.3) for y in range(origin + 1, origin + 8)}
        for c in range(cities):
            base = rng.uniform(40, 70)
            for lead in range(1, 8):
                year = origin + lead
                for metric in METRICS:
                    actual = base + anomaly + year_shock[year] + rng.normal(0, 1.0)
                    predicted = actual + error_fn(origin, c, year, rng)
                    rows.append({"city_id": f"C{c:03d}", "year": year, "metric": metric, "origin": origin,
                                 "lead": lead, "split": "live" if lead <= 3 else "final", "scored": True,
                                 "actual_value": actual, "baseline_value": base, "predicted_value": predicted})
    frame = pd.DataFrame(rows)
    frame["absolute_error"] = (frame.predicted_value - frame.actual_value).abs()
    frame["baseline_absolute_error"] = (frame.baseline_value - frame.actual_value).abs()
    frame["error"] = frame.predicted_value - frame.actual_value
    frame["pred_anomaly"] = frame.predicted_value - frame.baseline_value
    frame["obs_anomaly"] = frame.actual_value - frame.baseline_value
    return frame


def card_for(rows):
    card = insights.analyse(rows)
    card["insights"] = insights.findings(card)
    return card


def kinds(card, metric="annual_avg_temp"):
    return {f["kind"] for f in card["insights"] if f["metric"] in (metric, None)}


class Arithmetic(unittest.TestCase):
    def test_city_and_zone_contributions_sum_to_the_headline(self):
        rows = synthetic(lambda o, c, y, r: r.normal(0, 0.8))
        final = rows[rows.split == "final"]
        head = insights.headline(insights.metric_points(final))
        self.assertAlmostEqual(insights.contributions(final, "city_id").sum(), head, places=9)
        self.assertAlmostEqual(insights.contributions(final, "metric").sum(), head, places=9)

    def test_decomposition_shares_sum_to_one(self):
        card = card_for(synthetic(lambda o, c, y, r: 0.5 + r.normal(0, 0.5)))
        for shares in card["decomposition"].values():
            self.assertAlmostEqual(sum(shares.values()), 1.0, places=9)

    def test_points_match_the_reward_on_a_sealed_run(self):
        try:
            paths = reward.candidates()
        except reward.RewardError as exc:
            self.skipTest(str(exc))
        if not paths:
            self.skipTest("no official runs in this checkout")
        _, rows = insights.load_rows(paths[-1])
        expected = reward.summary(reward.Run(paths[-1]))["reward"]
        self.assertAlmostEqual(insights.analyse(rows)["reward_points"], expected, places=9)


class Diagnoses(unittest.TestCase):
    def test_consistent_bias_is_a_level_bias_with_honest_gain(self):
        card = card_for(synthetic(lambda o, c, y, r: -0.8 + r.normal(0, 0.4)))
        self.assertIn("level_bias", kinds(card))
        self.assertGreater(card["headroom"]["annual_avg_temp"]["whatif_remove_mean_bias_honest"], 0.25)

    def test_bias_that_flips_between_origins_is_not_a_fixable_level_bias(self):
        sign = {1983: -1.0, 1990: 1.0, 1997: -1.0}
        card = card_for(synthetic(lambda o, c, y, r: sign[o] + r.normal(0, 0.4)))
        found = kinds(card)
        self.assertIn("bias_flips", found)
        self.assertNotIn("level_bias", found)

    def test_noise_city_offsets_are_rejected_by_cross_fitting(self):
        card = card_for(synthetic(lambda o, c, y, r: r.normal(0, 1.0)))
        h = card["headroom"]["annual_avg_temp"]
        self.assertGreater(h["whatif_remove_city_offsets_fit"], 0)       # in-sample looks like a win...
        self.assertLessEqual(h["whatif_remove_city_offsets_honest"], 0)  # ...but other years do not confirm it
        self.assertNotIn("city_offsets", kinds(card))

    def test_persistent_city_offsets_are_found(self):
        offsets = {c: np.random.default_rng(c).normal(0, 1.5) for c in range(60)}
        card = card_for(synthetic(lambda o, c, y, r: offsets[c] + r.normal(0, 0.3)))
        self.assertIn("city_offsets", kinds(card))

    def test_overconfident_departures_call_for_shrinkage(self):
        # Prediction = base + 2.5 x (true anomaly signal) + noise: departures are too large.
        def error(o, c, y, r):
            return r.normal(0, 1.5)
        rows = synthetic(error)
        final = rows.split == "final"
        rows.loc[final, "predicted_value"] = rows.baseline_value + 2.5 * rows.obs_anomaly
        rows["error"] = rows.predicted_value - rows.actual_value
        rows["absolute_error"] = rows.error.abs()
        rows["pred_anomaly"] = rows.predicted_value - rows.baseline_value
        card = card_for(rows)
        self.assertLess(card["calibration"]["annual_avg_temp"]["anomaly_slope"], 0.8)
        self.assertIn("amplitude", kinds(card))

    def test_climatology_metric_is_flagged_without_a_fake_headroom(self):
        rows = synthetic(lambda o, c, y, r: r.normal(0, 0.5))
        mask = rows.metric == "total_precip_in"
        rows.loc[mask, "predicted_value"] = rows.loc[mask, "baseline_value"]
        rows["error"] = rows.predicted_value - rows.actual_value
        rows["absolute_error"] = rows.error.abs()
        rows["pred_anomaly"] = rows.predicted_value - rows.baseline_value
        card = card_for(rows)
        flagged = [f for f in card["insights"] if f["kind"] == "no_signal"]
        self.assertEqual([f["metric"] for f in flagged], ["total_precip_in"])
        self.assertIsNone(flagged[0]["headroom_points"])                 # no control data: no invented number


if __name__ == "__main__":
    unittest.main()
