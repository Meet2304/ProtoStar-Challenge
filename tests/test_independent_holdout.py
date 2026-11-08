"""Offline independent-station acquisition and aggregation contract tests."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_independent_holdout as holdout


SID = "USW00000001"


def csv_bytes(rows):
    columns = ["STATION", "DATE", "TMAX", "TMAX_ATTRIBUTES", "TMIN", "TMIN_ATTRIBUTES", "PRCP", "PRCP_ATTRIBUTES"]
    return pd.DataFrame(rows, columns=columns).to_csv(index=False).encode()


def observation(date, tmax="80", tmin="60", prcp="1", *, txflag=",,W", tnflag=",,W", pflag=",,W", station=SID):
    return [station, date, tmax, txflag, tmin, tnflag, prcp, pflag]


def station_fixture():
    rows, inventory = [], {}
    number = 1
    for north, lat in ((False, 30), (True, 43)):
        for lon in (-115, -97, -77):
            for offset in range(6):
                sid = f"USW{number:08d}"
                number += 1
                rows.append({"station_id": sid, "lat": lat + offset * 1.1, "lon": lon,
                    "elevation_m": 100.0, "state": "XX", "name": f"Fixture Station {number}"})
                for element in ("TMAX", "TMIN", "PRCP"):
                    inventory[sid, element] = (1940, 2018)
    return rows, inventory


class EndpointTests(unittest.TestCase):
    def change(self, url, **updates):
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        for name, value in updates.items():
            if value is None:
                query.pop(name, None)
            else:
                query[name] = [value]
        return urlunsplit(parsed._replace(query=urlencode(query, doseq=True)))

    def test_exact_date_caps_units_attributes_and_endpoint(self):
        url = holdout.observation_url(SID)
        self.assertEqual(holdout.validate_observation_url(url), SID)
        for updates in ({"endDate": None}, {"startDate": None}, {"endDate": "2019-01-01"},
                        {"startDate": "1949-01-01"}, {"units": "metric"},
                        {"includeAttributes": "false"}, {"format": "json"},
                        {"dataTypes": "TMAX,TMIN"}, {"stations": SID + ",USW00000002"}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                holdout.validate_observation_url(self.change(url, **updates))
        for url in ("https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USW00000001.dly",
                    "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd_all.tar.gz",
                    holdout.observation_url(SID).replace("https:", "http:")):
            with self.assertRaises(ValueError):
                holdout.validate_observation_url(url)

    def test_request_guard_runs_before_network(self):
        with patch.object(holdout, "build_opener") as opener:
            with self.assertRaises(ValueError):
                holdout.download(self.change(holdout.observation_url(SID), endDate="2025-12-31"), weather=True)
            opener.assert_not_called()

    def test_response_date_station_duplicates_and_infinity_fail_closed(self):
        bad_rows = [[observation("2019-01-01")], [observation("1949-12-31")],
                    [observation("2018-12-31", station="USW00000002")],
                    [observation("2018-12-31"), observation("2018-12-31")],
                    [observation("2018-12-31", tmax="inf")],
                    [observation("2018-02-30")], [observation("2018-01-01T00:00:00")]]
        for rows in bad_rows:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                holdout.parse_daily(csv_bytes(rows), SID)

    def test_bad_response_is_never_persisted(self):
        with tempfile.TemporaryDirectory() as name:
            selection = Path(name) / "selection.json"
            selection.write_text("{}")
            manifest = {"stations": [{"station_id": SID}]}
            with patch.object(holdout, "verify_selection", return_value=manifest), \
                 patch.object(holdout, "download", return_value=(csv_bytes([observation("2019-01-01")]), {})):
                with self.assertRaises(ValueError):
                    holdout.fetch(type("Args", (), {"selection": selection})())
            self.assertFalse((Path(name) / "raw").exists())

    def test_preregistration_required_before_metadata_network(self):
        with patch.object(holdout, "verify_prereg", side_effect=ValueError("not committed")), \
             patch.object(holdout, "download") as network:
            with self.assertRaises(ValueError):
                holdout.select(type("Args", (), {"preregistration": Path("missing.json")})())
            network.assert_not_called()


class AggregationTests(unittest.TestCase):
    def test_header_only_response_remains_a_quality_failure_without_replacement(self):
        daily = holdout.parse_daily(csv_bytes([]), SID)
        station = {"station_id": SID, "name": "Empty", "stratum": "north_west"}
        annual, coverage = holdout.aggregate_daily(daily, station)
        self.assertEqual(len(annual), 69)
        result = holdout.quality_result(annual, station)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["valid_years"]["cdd"]["target"], 0)

    def test_flags_missing_days_leap_year_trace_and_fahrenheit_cdd(self):
        daily = holdout.parse_daily(csv_bytes([
            observation("2016-02-28", tmax="80", tmin="60", prcp="1.25"),
            observation("2016-02-29", tmax="90", tmin="70", prcp="0", pflag="T,,W"),
            observation("2016-03-01", tmax="85", tmin="65", prcp="2", txflag=",X,W"),
            observation("2016-03-02", tmax="50", tmin="60", prcp="-1"),
            observation("2016-03-03", tmax="141", tmin="-131", prcp="3", pflag=",Q,W"),
            observation("2016-03-04", tmax="", tmin="", prcp=""),
            observation("2016-03-05", tmax="-9999", tmin="-9999", prcp="-9999"),
        ]), SID)
        annual, coverage = holdout.aggregate_daily(daily, {"station_id": SID, "name": "Fixture"})
        year = annual.set_index("Year").loc[2016]
        cov = coverage.set_index("Year").loc[2016]
        self.assertEqual(len(annual), 69)
        self.assertEqual(year["Miss Tmax"], 364)
        self.assertEqual(year["Miss Tmin"], 363)
        self.assertEqual(year["Miss Prcp"], 363)
        self.assertEqual(year["Avg Temp"], (85 + 65) / 2)
        self.assertEqual(year["ToT Prcp"], 3.25)
        self.assertEqual(year["Max Tmax"], 90)
        self.assertEqual(year.cdd, 20)  # (70-65)+(80-65), no Celsius conversion
        self.assertEqual(cov["Days Both"], 2)
        self.assertEqual(cov["PRCP Trace Days"], 1)
        self.assertTrue(pd.isna(cov["Station Changes"]))
        self.assertTrue(pd.isna(cov["Station Segment"]))
        self.assertEqual(cov["Station ID Count"], 1)
        absent = annual.set_index("Year").loc[2015]
        self.assertEqual(absent["Miss Tmax"], 365)
        self.assertTrue(pd.isna(absent.cdd))
        self.assertTrue(pd.isna(absent["ToT Prcp"]))

    def test_nonempty_qflag_invalidates_and_attribute_schema_is_required(self):
        daily = holdout.parse_daily(csv_bytes([observation("2018-01-01", tnflag=",I,W")]), SID)
        self.assertTrue(pd.isna(daily.TMIN.iloc[0]))
        with self.assertRaises(ValueError):
            holdout.parse_daily(csv_bytes([observation("2018-01-01", pflag="T")]), SID)

    def test_missing_values_may_have_empty_attributes(self):
        row = {"STATION": SID, "DATE": "2018-01-01", "TMAX": "", "TMIN": "", "PRCP": "",
               "TMAX_ATTRIBUTES": "", "TMIN_ATTRIBUTES": "", "PRCP_ATTRIBUTES": ""}
        daily = holdout.parse_daily(csv_bytes([row]), SID)
        self.assertTrue(daily[["TMAX", "TMIN", "PRCP"]].isna().all(axis=None))

    def test_annual_schema_is_accepted_by_official_loader(self):
        from evaluation.core import load_annual
        daily = holdout.parse_daily(csv_bytes([observation("2018-01-01")]), SID)
        annual, _ = holdout.aggregate_daily(daily, {"station_id": SID, "name": "Fixture"})
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "annual.csv"
            annual.to_csv(path, index=False)
            loaded = load_annual(path)
        self.assertEqual(len(loaded), 69)
        self.assertTrue(loaded.cdd.isna().all())

    def test_fixed_eligibility_keeps_failure_reasons(self):
        years = np.arange(1950, 2019)
        table = pd.DataFrame({"Year": years, "Avg Temp": 60., "ToT Prcp": 20., "Max Tmax": 100.,
                              "cdd": 200., "Miss Tmax": 0, "Miss Tmin": 0, "Miss Prcp": 0})
        station = {"station_id": SID, "stratum": "south_west"}
        self.assertTrue(holdout.quality_result(table, station)["eligible"])
        table.loc[table.Year.between(2012, 2015), "Miss Prcp"] = 11
        result = holdout.quality_result(table, station)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["valid_years"]["total_precip_in"]["target"], 3)
        self.assertTrue(any("total_precip_in:live" in reason for reason in result["failure_reasons"]))


class SelectionTests(unittest.TestCase):
    def test_deterministic_four_per_stratum_with_no_score_access(self):
        rows, inventory = station_fixture()
        for i, row in enumerate(rows):
            row["irrelevant_weather_score"] = i  # must never affect selection
        chosen, counts = holdout.select_stations(rows, inventory, set(), set(), [(0, 0)])
        altered = [{**r, "irrelevant_weather_score": -r["irrelevant_weather_score"]} for r in reversed(rows)]
        other, _ = holdout.select_stations(altered, inventory, set(), set(), [(0, 0)])
        # Input metadata extras can be retained but cannot influence station identity/order.
        self.assertEqual([r["station_id"] for r in chosen], [r["station_id"] for r in other])
        self.assertEqual(len(chosen), 24)
        self.assertEqual(set(counts), set(holdout.STRATA))
        self.assertTrue(all(sum(r["stratum"] == s for r in chosen) == 4 for s in holdout.STRATA))

    def test_all_ids_names_and_locations_are_excluded(self):
        rows, inventory = station_fixture()
        excluded = [rows[0], rows[6], rows[12]]
        chosen, _ = holdout.select_stations(rows, inventory, {excluded[0]["station_id"]},
            {holdout.normalize_name(excluded[1]["name"])}, [(excluded[2]["lat"], excluded[2]["lon"]), (0, 0)])
        self.assertTrue({r["station_id"] for r in chosen}.isdisjoint({r["station_id"] for r in excluded}))

    def test_inventory_and_stratum_minimum_fail_without_relaxation(self):
        rows, inventory = station_fixture()
        with self.assertRaisesRegex(ValueError, "four in every stratum"):
            holdout.select_stations(rows[3:], inventory, set(), set(), [(0, 0)])
        for row in rows[:3]:
            inventory[row["station_id"], "PRCP"] = (1951, 2018)
        with self.assertRaises(ValueError):
            holdout.select_stations(rows, inventory, set(), set(), [(0, 0)])
        self.assertLess(holdout.haversine_km((40, -100), (40.1, -100)), 100)
        self.assertGreater(holdout.haversine_km((40, -100), (42, -100)), 100)

    def test_official_fixed_width_parsers(self):
        station = f"{SID} {40.1234:8.4f} {-100.1234:9.4f} {123.4:6.1f} KS {'TEST STATION':30s}"
        parsed = holdout.parse_stations(station.encode())[0]
        self.assertEqual(parsed["name"], "TEST STATION")
        self.assertEqual(parsed["lat"], 40.1234)
        inventory = f"{SID} {40.1234:8.4f} {-100.1234:9.4f} TMAX 1940 2018"
        self.assertEqual(holdout.parse_inventory(inventory.encode())[SID, "TMAX"], (1940, 2018))

    def test_exclusive_artifacts_and_cache_hashes(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "artifact"
            holdout.write_exclusive(path, b"first")
            with self.assertRaises(FileExistsError):
                holdout.write_exclusive(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")
            raw, provenance = Path(name) / "raw.csv", Path(name) / "provenance.json"
            raw.write_bytes(csv_bytes([observation("2018-01-01")]))
            record = {"sha256": holdout.digest(raw), "station_id": SID, "selection_sha256": "fixed",
                      "request_url": holdout.observation_url(SID)}
            provenance.write_text(json.dumps(record))
            holdout.verify_cached(raw, provenance, SID, "fixed")
            raw.write_bytes(csv_bytes([observation("2018-01-02")]))
            with self.assertRaises(ValueError):
                holdout.verify_cached(raw, provenance, SID, "fixed")


if __name__ == "__main__":
    unittest.main()
