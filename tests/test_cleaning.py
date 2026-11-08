"""Acceptance tests for the cleaning pipeline (synthetic data; plan §7). Run: uv run python -m unittest tests.test_cleaning"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # works without an editable install
from protostar_challenge import cleaning as cl  # noqa: E402


def days(start, n):
    return pd.date_range(start, periods=n)


def mats(index, cols, value=np.nan):
    return pd.DataFrame(value, index=index, columns=cols, dtype=float)


class DailyQC(unittest.TestCase):
    def test_inverted_day_is_swapped_and_limits_deleted(self):
        idx = days("2000-01-01", 3)
        W = {"tmax": mats(idx, ["A"], 50.0), "tmin": mats(idx, ["A"], 40.0), "prcp": mats(idx, ["A"], 0.0)}
        W["tmax"].iloc[1, 0], W["tmin"].iloc[1, 0] = 51.0, 53.0      # inverted
        W["tmax"].iloc[2, 0] = 140.0                                  # beyond the US record
        out, log = cl.repair_impossible(W)
        self.assertEqual((out["tmax"].iloc[1, 0], out["tmin"].iloc[1, 0]), (53.0, 51.0))
        self.assertTrue(np.isnan(out["tmax"].iloc[2, 0]))
        self.assertEqual(set(log["action"]), {"swap", "delete"})

    def _stuck(self, nb_values):
        idx = days("1990-01-01", 30)
        cols = ["A", "B", "C", "D"]
        W = {v: mats(idx, cols) for v in cl.VARS}
        rng = np.random.default_rng(0)
        for c in cols:
            W["tmax"][c] = 50 + rng.normal(0, 1, 30).round()
        W["tmax"].iloc[5:15, 0] = 61.0                                # 10 identical days
        for c, v in zip(["B", "C", "D"], nb_values):
            W["tmax"][c] = v
        return cl.stuck_runs(W, {"A": ["B", "C", "D"], "B": ["A", "C", "D"], "C": ["A", "B", "D"], "D": ["A", "B", "C"]})

    def test_stuck_run_needs_varying_neighbours(self):
        rng = np.random.default_rng(1)
        varying = [50 + rng.normal(0, 8, 30) for _ in range(3)]
        steady = [np.full(30, 55.0) + rng.normal(0, 0.5, 30) for _ in range(3)]
        sparse = [np.where(np.arange(30) < 3, 50.0, np.nan)] * 3
        get = lambda r: r[(r.city_id == "A") & (r.days == 10)].iloc[0]  # noqa: E731
        self.assertEqual(get(self._stuck(varying))["action"], "delete")
        self.assertEqual(get(self._stuck(steady))["action"], "keep")
        r = get(self._stuck(sparse))
        self.assertEqual(r["action"], "keep")
        self.assertIn("fewer than 2", r["reason"])

    def test_repeated_precip_missing_is_not_dry(self):
        idx = days("1900-01-01", 10)
        cols = ["A", "B", "C", "D"]
        W = {v: mats(idx, cols, 0.0) for v in cl.VARS}
        W["prcp"].iloc[3:6, 0] = 0.3
        nbrs = {"A": ["B", "C", "D"], "B": ["A"], "C": ["A"], "D": ["A"]}
        self.assertEqual(cl.prcp_runs(W, nbrs).iloc[0]["action"], "delete")       # dry and observed
        W["prcp"].iloc[3:6, 1:3] = np.nan                                           # two neighbours missing
        self.assertEqual(cl.prcp_runs(W, nbrs).iloc[0]["action"], "keep")
        W["prcp"].iloc[3:6, 1:3] = 0.0
        W["prcp"].iloc[6, 3] = 0.2                                                  # rain the day after
        self.assertEqual(cl.prcp_runs(W, nbrs).iloc[0]["action"], "keep")

    def test_zero_mad_is_skipped(self):
        idx = days("2001-01-01", 31)
        M = mats(idx, ["A"], 70.0)
        M.iloc[10, 0] = 90.0
        z, _, scale = cl.robust_z(M)
        self.assertTrue(z["A"].isna().all())


class Calendar(unittest.TestCase):
    def test_leap_day_shares_feb28(self):
        idx = pd.DatetimeIndex(["2000-02-28", "2000-02-29", "2000-03-01", "2001-03-01", "2000-12-31"])
        self.assertEqual(list(cl.doy365(idx)), [59, 59, 60, 60, 365])


class Filling(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("1960-01-01", "1990-12-31")
        cols = ["A", "B", "C"]
        rng = np.random.default_rng(2)
        season = 20 * np.sin(2 * np.pi * idx.day_of_year / 365)
        common = rng.normal(0, 4, len(idx))
        self.W = {"tmax": pd.DataFrame({c: 60 + season + common + rng.normal(0, 1, len(idx)) for c in cols}, index=idx),
                  "prcp": pd.DataFrame({c: rng.exponential(0.1, len(idx)) for c in cols}, index=idx)}
        self.W["tmin"] = self.W["tmax"] - 15
        self.rec = pd.DataFrame(True, index=idx, columns=cols)
        self.w = pd.DataFrame([[0, 1, 1], [1, 0, 1], [1, 1, 0]], index=cols, columns=cols, dtype=float)

    def test_fill_keeps_observed_values(self):
        W = {v: m.copy() for v, m in self.W.items()}
        W["tmax"].iloc[100:130, 0] = np.nan
        F, was = cl.fill_daily(W, self.rec, self.w, 1990, "normal + neighbour anomaly")
        obs = W["tmax"].notna()
        self.assertTrue(((F["tmax"] == W["tmax"]) | ~obs).all().all())
        self.assertTrue(was["tmax"].iloc[100:130, 0].all())
        err = (F["tmax"].iloc[100:130, 0] - self.W["tmax"].iloc[100:130, 0]).abs().mean()
        self.assertLess(err, 2.0)

    def test_filled_member_is_clipped_to_observed_member(self):
        W = {v: m.copy() for v, m in self.W.items()}
        W["tmax"].iloc[200, 0] = np.nan
        W["tmin"].iloc[200, 0] = 200.0              # absurd observed tmin: the filled tmax must not sit below it
        F, _ = cl.fill_daily(W, self.rec, self.w, 1990, "normal")
        self.assertEqual(F["tmin"].iloc[200, 0], 200.0)
        self.assertGreaterEqual(F["tmax"].iloc[200, 0], 200.0)


class Stations(unittest.TestCase):
    def _st(self, rows):
        st = pd.DataFrame(rows, columns=["city_id", "stn_name", "start", "end", "station_key", "empty_period"])
        st["start"], st["end"] = pd.to_datetime(st["start"]), pd.to_datetime(st["end"])
        return st

    def test_empty_period_and_same_key_merge(self):
        st = self._st([("L", "CITY", "1873-01-01", "1950-12-31", 1, False),
                       ("L", "AP", "1951-01-01", "1985-09-30", 2, False),
                       ("L", "CITY", "1985-10-01", "1986-01-31", 1, True),
                       ("L", "AP", "1986-02-01", "2018-12-31", 2, False)])
        ch = cl.effective_changes(st)
        self.assertEqual(len(ch), 1)
        self.assertEqual(ch.iloc[0]["date"], pd.Timestamp("1951-01-01"))

    def test_clusters_same_endpoints_and_overlap(self):
        st = self._st([("H", "H", "1900-01-01", "1954-07-14", 1, False),
                       ("H", "M", "1954-07-15", "1958-08-31", 2, False),
                       ("H", "H", "1958-09-01", "2018-12-31", 1, False),
                       ("A", "C", "1886-01-01", "1944-02-29", 1, False),
                       ("A", "AP", "1943-03-01", "2018-12-31", 2, False)])
        cls = cl.make_clusters(cl.effective_changes(st), 2018)
        h = cls[cls.city_id == "H"].iloc[0]
        self.assertTrue(h["same_endpoints"])
        a = cls[cls.city_id == "A"].iloc[0]
        self.assertEqual(a["overlap_years"], [1943, 1944])
        self.assertEqual(a["f"], pd.Timestamp("1943-03-01"))

    def test_offsets_one_per_day_and_transition_only_when_adopted(self):
        idx = pd.date_range("1900-01-01", "1960-12-31")
        keys = pd.DataFrame(0, index=idx, columns=["X"], dtype=np.int16)
        spans = [("1900-01-01", "1919-12-31", 1), ("1920-01-01", "1921-06-30", 2), ("1921-07-01", "1939-12-31", 3),
                 ("1940-01-01", "1960-12-31", 4)]
        for a, b, k in spans:
            keys.loc[a:b, "X"] = k
        clusters = pd.DataFrame([
            {"city_id": "X", "cluster_id": 0, "f": pd.Timestamp("1920-01-01"), "l": pd.Timestamp("1921-07-01"),
             "P": 1, "Q": 3},
            {"city_id": "X", "cluster_id": 1, "f": pd.Timestamp("1940-01-01"), "l": pd.Timestamp("1940-01-01"),
             "P": 3, "Q": 4}])
        steps = pd.DataFrame([{"city_id": "X", "var": "Avg Tmax", "cluster_id": 0, "step": 1.0, "applied": True},
                              {"city_id": "X", "var": "Avg Tmax", "cluster_id": 1, "step": -2.0, "applied": True}])
        off, trans = cl.daily_offsets(steps, "tmax", keys, clusters)
        o = off["X"]
        self.assertAlmostEqual(o["1910-06-01"], -1.0)      # before both: 1 + (-2)
        self.assertAlmostEqual(o["1930-06-01"], -2.0)      # between: the later step only
        self.assertAlmostEqual(o["1950-06-01"], 0.0)       # reference station
        self.assertTrue(trans["X"]["1920-01-01":"1921-06-30"].all())
        self.assertFalse(trans["X"]["1921-07-01":].any())
        steps.loc[0, "applied"] = False
        off2, trans2 = cl.daily_offsets(steps, "tmax", keys, clusters)
        self.assertFalse(trans2.values.any())
        self.assertAlmostEqual(off2["X"]["1910-06-01"], -2.0)


class Homogenisation(unittest.TestCase):
    def _case(self, step, n_before, n_after, trend=0.0, missing_nb=False, seed=0):
        rng = np.random.default_rng(seed)
        years = np.arange(1900, 1961)
        cities = ["A", "B", "C", "D"]
        common = rng.normal(0, 1, len(years))
        T = pd.DataFrame({c: 50 + common + rng.normal(0, 0.3, len(years)) for c in cities}, index=years).T
        T.loc["A"] += trend * (years - 1930)
        T.loc["A", years >= 1930] += step
        T.loc["A", (years < 1930 - n_before) | (years >= 1930 + n_after)] = np.nan
        if missing_nb:
            T.loc["B", years[::3]] = np.nan
        clusters = pd.DataFrame([{"city_id": "A", "cluster_id": 0, "f": pd.Timestamp("1930-01-01"),
                                  "l": pd.Timestamp("1930-01-01"), "P": 1, "Q": 2, "n_changes": 1,
                                  "same_endpoints": False, "middle_keys": [], "overlap_years": [], "label": "x"}])
        w = pd.DataFrame(1.0, index=cities, columns=cities) - np.eye(4)
        H = cl.Homogeniser({"Avg Tmax": T, "Avg Tmin": T}, clusters, w, 1960)
        b, a = H.windows("A", 0)
        return H.estimate("Avg Tmax", "A", b, a, H.X["Avg Tmax"], 0.0)

    def test_injected_step_recovered(self):
        for nb, na in [(10, 10), (5, 10), (5, 5)]:
            e = self._case(1.5, nb, na)
            self.assertEqual(e["status"], "estimated")
            self.assertAlmostEqual(e["step"], 1.5, delta=0.5)
            self.assertGreater(abs(e["t"]), 2)

    def test_no_step_on_trend_or_uneven_neighbours(self):
        for kw in [{"trend": 0.05}, {"missing_nb": True}]:
            es = [self._case(0.0, 10, 10, seed=s, **kw) for s in range(20)]
            self.assertLess(abs(np.mean([e["step"] for e in es])), 0.15)       # unbiased: the trend is not a step
            self.assertLessEqual(sum(abs(e["t"]) >= 2 for e in es), 4)

    def test_false_adoption_rate_is_low(self):
        hits = sum(abs(self._case(0.0, 10, 10, seed=s)["t"]) >= 2 for s in range(40))
        self.assertLessEqual(hits, 6)


class Annual(unittest.TestCase):
    def test_delta_form_is_exact_when_nothing_changed(self):
        idx = pd.Index([1, 2, 3])
        label = pd.Series([69.32, np.nan, 50.0], index=idx)
        f = pd.Series([69.3101092896, 40.0, 50.004], index=idx)
        out = cl.delta_form(label, f, f, pd.Series(True, index=idx))
        self.assertEqual(out[1], 69.32)
        self.assertEqual(out[2], 40.0)          # no label: the variant itself
        self.assertEqual(out[3], 50.0)

    def test_fragment_rule(self):
        idx = pd.date_range("1880-01-01", "2018-12-31")
        W = {v: mats(idx, ["A"]) for v in cl.VARS}
        for y in [1880, 1881, 1882]:                       # short old block
            W["tmax"].loc[str(y), "A"] = 50.0
        for y in range(1950, 2019):
            W["tmax"].loc[str(y), "A"] = 50.0
        drops, breaks = cl.fragments(W)
        t = drops[drops.group == "temperature"]
        self.assertEqual(list(t["first"]), [1880])
        self.assertIn(1950, list(breaks[breaks.group == "temperature"]["year"]))

    def test_infill_never_fills_dropped_years(self):
        T = pd.DataFrame({y: [10.0 + (y % 3), 12.0 + (y % 3), 11.0 + (y % 3)] for y in range(1950, 1980)},
                         index=["A", "B", "C"])
        T.loc["A", 1960] = np.nan
        T.loc["A", 1961] = np.nan
        w = pd.DataFrame(1.0, index=T.index, columns=T.index) - np.eye(3)
        never = pd.DataFrame(False, index=T.index, columns=T.columns)
        never.loc["A", 1961] = True
        out = cl.neighbour_infill(T, w, 1979, never=never)
        self.assertAlmostEqual(out.loc["A", 1960], 10.0 + 1960 % 3)
        self.assertTrue(np.isnan(out.loc["A", 1961]))


class Overrides(unittest.TestCase):
    def test_evidence_year_filters_overrides(self):
        cand = pd.DataFrame({"city_id": ["A"], "date": [pd.Timestamp("1990-05-01")], "var": ["tmax"],
                             "value": [99.0], "proposed_action": ["keep"]})
        ov = pd.DataFrame({"city_id": ["A"], "date": ["1990-05-01"], "var": ["tmax"], "final_action": ["delete"],
                           "decided_on": ["2026-11-08"], "evidence_year": [2018], "note": [""]})
        self.assertEqual(len(cl.outlier_log(cand, ov, 2010)), 0)
        self.assertEqual(len(cl.outlier_log(cand, ov, 2018)), 1)


if __name__ == "__main__":
    unittest.main()
