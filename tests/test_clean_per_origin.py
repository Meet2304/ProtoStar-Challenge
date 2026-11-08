"""Tests that the per-origin build cannot see past its origin (synthetic data). Run: uv run python -m unittest tests.test_clean_per_origin"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # works without an editable install
from protostar_challenge import cleaning_per_origin as cpo


def raw(future=1.0):
    """Two cities, 1980-1985. City B's second station starts after the 1982 origin; `future` scales later values."""
    idx = pd.date_range("1980-01-01", "1985-12-31")
    later = idx.year > 1982
    W = {v: pd.DataFrame({"A": np.where(later, 70.0 * future, 50.0), "B": np.where(later, 71.0 * future, 51.0)}, index=idx)
         for v in ["tmax", "tmin", "prcp"]}
    stations = pd.DataFrame({"ID": ["A", "B", "B"], "Stn.stDate": pd.to_datetime(["1980-01-01", "1980-01-01", "1984-01-01"]),
                             "Stn.edDate": pd.to_datetime(["1985-12-31", "1983-12-31", "1985-12-31"])})
    years = pd.DataFrame({"ID": ["A", "B"] * 6, "Year": sorted(list(range(1980, 1986)) * 2)})
    return {"W": W, "record": pd.DataFrame(True, index=idx, columns=["A", "B"]),
            "annual": years.assign(value=np.where(years["Year"] > 1982, future, 0.0)),
            "coverage": years.assign(days=np.where(years["Year"] > 1982, future, 0.0)),
            "city_info": stations, "cities": pd.DataFrame({"lat": [1.0, 2.0]}, index=["A", "B"])}


class Truncation(unittest.TestCase):
    def test_no_table_reaches_past_the_origin(self):
        cut = cpo.truncate_raw(raw(), 1982)
        for v in ["tmax", "tmin", "prcp"]:
            self.assertEqual(cut["W"][v].index.max(), pd.Timestamp("1982-12-31"))
        self.assertEqual(cut["record"].index.max(), pd.Timestamp("1982-12-31"))
        self.assertEqual(cut["annual"]["Year"].max(), 1982)
        self.assertEqual(cut["coverage"]["Year"].max(), 1982)

    def test_stations_are_as_known_at_the_origin(self):
        cut = cpo.truncate_raw(raw(), 1982)["city_info"]
        self.assertEqual(list(cut["ID"]), ["A", "B"])           # B's 1984 station does not exist yet
        self.assertTrue((cut["Stn.edDate"] == pd.Timestamp("1982-12-31")).all())   # open stations end at the origin

    def test_later_values_cannot_change_the_cut_copy(self):
        a, b = cpo.truncate_raw(raw(future=1.0), 1982), cpo.truncate_raw(raw(future=9.0), 1982)
        for v in ["tmax", "tmin", "prcp"]:
            pd.testing.assert_frame_equal(a["W"][v], b["W"][v])
        pd.testing.assert_frame_equal(a["annual"], b["annual"])
        pd.testing.assert_frame_equal(a["coverage"], b["coverage"])
        pd.testing.assert_frame_equal(a["city_info"], b["city_info"])

    def test_full_input_is_not_modified(self):
        full = raw()
        before = full["city_info"].copy()
        cpo.truncate_raw(full, 1982)
        pd.testing.assert_frame_equal(full["city_info"], before)


class PercentileIndices(unittest.TestCase):
    def table(self):
        return pd.DataFrame({c: [1.0] for c in cpo.PERCENTILE_COLS} | {"other": [2.0]})

    def test_blanked_before_the_thresholds_end(self):
        out = cpo.blank_percentile_indices(self.table(), 1983)
        self.assertTrue(out[cpo.PERCENTILE_COLS].isna().all().all())
        self.assertEqual(out["other"].iloc[0], 2.0)

    def test_kept_once_the_thresholds_are_in_the_past(self):
        out = cpo.blank_percentile_indices(self.table(), 1990)
        self.assertTrue(out[cpo.PERCENTILE_COLS].notna().all().all())


class LatestYear(unittest.TestCase):
    def test_reads_date_and_year_columns(self):
        self.assertEqual(cpo.latest_year(pd.DataFrame({"date": ["1981-05-01"], "year": [1979]})), 1981)
        self.assertEqual(cpo.latest_year(pd.DataFrame({"f": pd.to_datetime(["1990-01-01"])})), 1990)

    def test_no_time_column_gives_nan(self):
        self.assertTrue(np.isnan(cpo.latest_year(pd.DataFrame({"x": [1]}))))

    def test_origins_match_the_evaluation_config(self):
        self.assertEqual(cpo.ORIGINS, (1983, 1990, 1997, 2004, 2011))


if __name__ == "__main__":
    unittest.main()
