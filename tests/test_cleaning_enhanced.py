"""Checks for the enhanced-input cleaning on synthetic data. Run: uv run python -m unittest tests.test_cleaning_enhanced"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # works without an editable install
from protostar_challenge import cleaning_enhanced as ce


def static():
    """Three cities: one reports its resolution in degrees, one has a far catalog station, one a far catalog elevation."""
    return pd.DataFrame({
        "city_id": ["A", "B", "C"], "name": ["a", "b", "c"], "lat": [30.0, 40.0, 45.0], "lon": [-90.0, -100.0, -110.0],
        "station_catalog_elevation_m": [10.0, 500.0, 1000.0], "elevation_coordinate_offset_km": [0.1, 5.0, 0.2],
        "elevation_m": [12.0, 498.0, 950.0], "elevation_resolution_m": [1.0, 3.08642e-05, 1.0],
        "elevation_acquisition_date": ["2020-01-01", "unknown", "2019-01-01"],
        "elevation_acquisition_date_raw": ["1/1/2020", "0/6/2020", "1/1/2019"],
        "distance_to_ocean_coast_km": [5.0, 900.0, 1200.0], "landcover_year": [2021] * 3,
        "landcover_station_class": [50, 30, 10], "landcover_dominant_class_1km": [50, 30, 10],
        "landcover_valid_fraction_1km": [1.0] * 3,
        "landcover_built_fraction_1km": [0.6, 0.2, 0.0], "landcover_grass_fraction_1km": [0.4, 0.8, 0.3],
        "landcover_tree_fraction_1km": [0.0, 0.0, 0.7], "landcover_snow_ice_fraction_1km": [0.0] * 3})


CITIES = pd.DataFrame({"city_id": ["A", "B", "C"], "name": ["a", "b", "c"], "lat": [30.0, 40.0, 45.0],
                       "lon": [-90.0, -100.0, -110.0]})


def snapshot(origin, first=1950, shift=0.0):
    """One origin's monthly and annual tables, built the way the release defines them."""
    dates = pd.date_range(f"{first}-01-01", f"{origin}-12-01", freq="MS")
    monthly = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "year": dates.year, "month": dates.month})
    t = np.arange(len(dates))
    for k, index in enumerate(ce.INDICES):
        monthly[f"{index}_sst_c"] = 20 + k + 2 * np.sin(2 * np.pi * monthly["month"] / 12) + 0.001 * t + shift
    start = origin - ce.CLIMATOLOGY_YEARS + 1
    window = monthly[monthly["year"] >= start]
    for index in ce.INDICES:
        normal = window.groupby("month")[f"{index}_sst_c"].mean()
        monthly[f"{index}_anomaly_c"] = monthly[f"{index}_sst_c"] - monthly["month"].map(normal)
    monthly["climatology_start_year"], monthly["climatology_end_year"] = start, origin
    annual = monthly.groupby("year", as_index=False)[[c for c in monthly.columns if c.endswith("_c")]].mean()
    annual["climatology_start_year"], annual["climatology_end_year"], annual["months_available"] = start, origin, 12
    return monthly, annual


class CityFacts(unittest.TestCase):
    def setUp(self):
        self.clean, self.log, self.summary = ce.clean_static(static(), CITIES)

    def test_degrees_become_metres(self):
        self.assertAlmostEqual(self.clean.set_index("city_id").at["B", "elevation_resolution_m"], 3.44, places=2)
        self.assertEqual(self.summary["resolution_converted"], 1)

    def test_flags(self):
        flags = self.clean.set_index("city_id")
        self.assertEqual(flags["location_uncertain"].tolist(), [False, True, False])
        self.assertEqual(flags["elevation_catalog_disagrees"].tolist(), [False, False, True])
        self.assertEqual(flags.at["C", "elevation_m"], 950.0)   # USGS value kept, not replaced

    def test_constant_columns_dropped_and_logged(self):
        dropped = {"landcover_year", "landcover_valid_fraction_1km", "landcover_snow_ice_fraction_1km"}
        self.assertTrue(dropped.isdisjoint(self.clean.columns))
        self.assertEqual(set(self.log.loc[self.log["action"] == "dropped", "column"]), dropped)
        self.assertNotIn("landcover_snow_ice_fraction_1km", self.summary["model_columns"])

    def test_rejects_wrong_city_set_and_bad_fractions(self):
        with self.assertRaises(ValueError):
            ce.clean_static(static().iloc[:2], CITIES)
        bad = static()
        bad.loc[0, "landcover_built_fraction_1km"] = 0.9
        with self.assertRaises(ValueError):
            ce.clean_static(bad, CITIES)


class OceanSnapshots(unittest.TestCase):
    def test_a_correct_snapshot_passes(self):
        self.assertEqual(ce.verify_origin(*snapshot(1990), 1990), [])

    def test_a_row_after_the_origin_fails(self):
        monthly, annual = snapshot(1991)
        problems = ce.verify_origin(monthly, annual, 1990)
        self.assertIn("a row after the origin", problems)

    def test_a_full_period_climatology_fails(self):
        monthly, annual = snapshot(1990)
        for index in ce.INDICES:   # normalised with every year instead of the 30 ending at the origin
            normal = monthly.groupby("month")[f"{index}_sst_c"].mean()
            monthly[f"{index}_anomaly_c"] = monthly[f"{index}_sst_c"] - monthly["month"].map(normal)
        problems = ce.verify_origin(monthly, annual, 1990)
        self.assertTrue(any("climatology" in p for p in problems))

    def test_a_missing_month_fails(self):
        monthly, annual = snapshot(1990)
        self.assertIn("missing months", ce.verify_origin(monthly.drop(index=5), annual, 1990))


if __name__ == "__main__":
    unittest.main()
