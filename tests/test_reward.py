"""Tests for code/evaluation/reward.py and for the scorer's agreement with the public leaderboard.

The leaderboard rows below are the per-metric MAEs and Scores the Space displayed on 2026-11-08. They are
used only to check arithmetic (that our pinned scorer turns those MAEs into the displayed Score); no
model is chosen or tuned with them, and no 2019-2025 observation is read.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from common import METRICS  # noqa: E402
from evaluation import reward  # noqa: E402
from evaluation.official_score import score  # noqa: E402

# name, displayed Score, MAE annual_avg_temp, total_precip_in, hottest_day_temp, cdd
LEADERBOARD_20261108 = """PstarAlu,16.1,0.990,5.956,2.174,136.8
TROUBLESHOOTERS,15.9,0.981,5.824,2.233,138.4
GuaGuaBear,15.5,1.006,5.831,2.172,143.6
Ashley boys,14.9,1.026,5.987,2.173,140.8
Team Tatakae,14.4,1.061,5.987,2.173,140.2
unicorns,14.1,1.064,5.987,2.173,141.9
Klepon Nusantara,14.1,1.093,6.001,2.165,138.6
Evanguo12,13.8,1.080,5.993,2.174,142.0
MKDir,13.6,1.068,5.987,2.173,145.2
LEGENDS OF THE FALL,13.1,1.156,5.830,2.200,140.1
Carbon Copies,12.9,1.089,5.987,2.185,146.9
Cloudbreak,12.9,1.126,5.987,2.176,143.2
GW OSPO,12.8,1.098,5.975,2.177,147.3
Zhoey,12.8,1.106,5.987,2.173,146.2
Placeholder,12.5,1.128,5.987,2.176,146.0
Team Hurricane,11.7,1.018,6.942,2.185,134.1
Ocarina of Ten,10.4,1.191,5.782,2.295,149.6
Linear trend,9.6,1.095,6.122,2.413,147.4
30-year average,0.0,1.477,5.987,2.176,198.0
Last year repeated,0.0,1.146,8.466,2.751,175.8"""
REFERENCE = dict(zip(METRICS, (1.477, 5.987, 2.176, 198.0)))
# Display rounding: Score to 0.1 (0.05) plus MAEs to 3 decimals / CDD to 0.1 (<= 0.025 points).
TOLERANCE = 0.08


def _one_row_per_metric(maes):
    key = pd.DataFrame({"city_id": "X", "year": 2019, "metric": list(METRICS), "actual_value": 0.0,
                        "split": "live", "scored": True})
    sub = key[["city_id", "year", "metric"]].assign(predicted_value=[maes[m] for m in METRICS])
    return sub, key


class LeaderboardFormulaTests(unittest.TestCase):
    def test_pinned_scorer_reproduces_every_displayed_score(self):
        ref = pd.DataFrame([{"split": "live", **REFERENCE}])
        for line in LEADERBOARD_20261108.splitlines():
            name, shown, *values = line.split(",")
            sub, key = _one_row_per_metric(dict(zip(METRICS, map(float, values))))
            with self.subTest(team=name):
                self.assertAlmostEqual(score(sub, key, ref, "live")["score"], float(shown), delta=TOLERANCE)

    def test_skill_is_not_clipped_per_metric(self):
        # Per-metric clipping would give Team Hurricane 15.8 and last-year-repeated 8.4; the board shows 11.7 and 0.0.
        ref = pd.DataFrame([{"split": "live", **REFERENCE}])
        sub, key = _one_row_per_metric(dict(zip(METRICS, (1.018, 6.942, 2.185, 134.1))))
        self.assertLess(score(sub, key, ref, "live")["skill"]["total_precip_in"], 0)


def _fake_run(origins, error, ref, name="fake", path="fake"):
    run = reward.Run.__new__(reward.Run)
    run.origins, run.name, run.path = origins, name, Path(path)
    run.meta = {"suite_id": "s"}
    run.cities = [f"c{i}" for i in range(error.shape[0])]
    run.cells = [(o, s, m) for o in origins for s in reward.SPLITS for m in METRICS]
    run.error, run.reference = error, ref
    keys = pd.MultiIndex.from_product([run.cities, [1], ["m"]], names=["city_id", "year", "metric"])
    run.rows = pd.DataFrame({"baseline_absolute_error": 1.0}, index=keys).reset_index()
    return run


class LadderTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.origins = [1983, 1990, 1997]
        cells = len(self.origins) * len(reward.SPLITS) * len(METRICS)
        self.ref = rng.uniform(5, 10, (210, cells))
        self.base = self.ref * rng.uniform(0.9, 1.1, (210, cells))

    def test_identical_runs_are_within_noise(self):
        a = _fake_run(self.origins, self.base, self.ref)
        result = reward.compare(a, _fake_run(self.origins, self.base.copy(), self.ref, path="b"), samples=200)
        self.assertEqual(result["final"]["delta_points"], 0)
        self.assertEqual(result["verdict"], "within_noise")

    def test_clear_gain_everywhere_is_promoted(self):
        champ = _fake_run(self.origins, self.base, self.ref)
        better = _fake_run(self.origins, self.base * 0.9, self.ref, path="b")
        result = reward.compare(better, champ, samples=500)
        self.assertEqual(result["verdict"], "promote")
        ratio = (self.base.sum(0) / self.ref.sum(0)).reshape(len(self.origins), len(reward.SPLITS), len(METRICS))
        self.assertAlmostEqual(result["final"]["delta_points"], 100 * 0.1 * ratio[:, 1, :].mean(), 9)

    def test_clear_loss_is_worse(self):
        champ = _fake_run(self.origins, self.base, self.ref)
        result = reward.compare(_fake_run(self.origins, self.base * 1.1, self.ref, path="b"), champ, samples=500)
        self.assertEqual(result["verdict"], "worse")

    def test_gain_on_one_origin_only_is_not_promoted(self):
        error = self.base.copy()
        per_origin = len(reward.SPLITS) * len(METRICS)
        error[:, 2 * per_origin:] *= 0.5            # huge gain on 1997 only
        result = reward.compare(_fake_run(self.origins, error, self.ref, path="b"),
                                _fake_run(self.origins, self.base, self.ref), samples=500)
        self.assertGreater(result["final"]["p_better"], 0.95)
        self.assertFalse(result["checks"]["wins_most_origins"])
        self.assertEqual(result["verdict"], "within_noise")


class SealedRunParityTests(unittest.TestCase):
    def test_reward_matches_sealed_scores(self):
        try:
            paths = reward.candidates()
        except reward.RewardError as exc:
            self.skipTest(str(exc))
        if not paths:
            self.skipTest("no official runs in this checkout")
        run = reward.Run(paths[-1])
        out = reward.summary(run)
        sealed = {}
        for origin in run.origins:
            for entry in json.loads((run.path / f"origin-{origin}/scores.json").read_text(encoding="utf-8")):
                sealed.setdefault(entry["split"], []).append(entry["mean_skill_unclipped"])
        for split in reward.SPLITS:
            self.assertAlmostEqual(out[split]["points"], 100 * np.mean(sealed[split]), places=9)


if __name__ == "__main__":
    unittest.main()
