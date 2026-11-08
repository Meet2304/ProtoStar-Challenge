"""Population around each of the 210 weather stations, from the 2020 Census (analysis only).

For each station's latitude/longitude (data/city_info.csv), the U.S. Census Bureau Geocoder
(geographies by coordinates, 2020 Census vintage) returns the county, metropolitan statistical
area (metro) and incorporated place the point falls in, each with its 2020 Census population
(POP100). No API key is needed. Most stations are at airports, often just outside the city limits,
so the metro area is the unit we use for "people served by this station"; stations outside any metro
fall back to their county.

Data rule: population does not respond to the weather, so any year is allowed. These numbers are
used for the policy analysis only, not as forecast inputs.

Usage (from the repository root):
    python code/prepare_population.py

Writes data/city_population.csv (one row per city).
"""

import json
import time
import urllib.parse
import urllib.request

import pandas as pd

from common import DATA_DIR

GEOCODER = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
USER_AGENT = "protostar-placeholders-data-prep/1.0"
PARAMS = {"benchmark": "Public_AR_Current", "vintage": "Census2020_Current",
          "layers": "all", "format": "json"}


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except (OSError, json.JSONDecodeError):
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def _first(geographies: dict, layer: str) -> dict:
    items = geographies.get(layer) or []
    return items[0] if items else {}


def geographies_for(lat: float, lon: float) -> dict:
    """County, metro, place and state (with 2020 populations) containing a point."""
    query = urllib.parse.urlencode({"x": lon, "y": lat, **PARAMS})
    g = _get_json(f"{GEOCODER}?{query}")["result"]["geographies"]
    county, metro = _first(g, "Counties"), _first(g, "Metropolitan Statistical Areas")
    place, state = _first(g, "Incorporated Places"), _first(g, "States")
    row = {
        "state": state.get("STUSAB"),
        "county_geoid": county.get("GEOID"), "county_name": county.get("NAME"),
        "county_pop_2020": county.get("POP100"),
        "metro_geoid": metro.get("GEOID"), "metro_name": metro.get("NAME"),
        "metro_pop_2020": metro.get("POP100"),
        "place_name": place.get("NAME"), "place_pop_2020": place.get("POP100"),
    }
    if row["metro_geoid"]:
        row.update(exposure_area=row["metro_name"], exposure_geoid=f"metro:{row['metro_geoid']}",
                   exposure_pop_2020=row["metro_pop_2020"])
    else:
        row.update(exposure_area=row["county_name"], exposure_geoid=f"county:{row['county_geoid']}",
                   exposure_pop_2020=row["county_pop_2020"])
    return row


def main() -> None:
    cities = pd.read_csv(DATA_DIR / "city_info.csv", dtype={"city_id": str})
    rows = []
    for i, city in enumerate(cities.itertuples(index=False), start=1):
        try:
            row = geographies_for(city.lat, city.lon)
        except Exception as error:          # keep going; the city is reported as unmatched
            row = {"error": f"{type(error).__name__}: {error}"}
        rows.append({"city_id": city.city_id, "name": city.name, "lat": city.lat, "lon": city.lon, **row})
        if i % 25 == 0:
            print(f"  {i}/{len(cities)} stations looked up")
        time.sleep(0.2)                     # be polite to the Census Bureau's server

    out = pd.DataFrame(rows)
    columns = ["city_id", "name", "lat", "lon", "state", "county_geoid", "county_name", "county_pop_2020",
               "metro_geoid", "metro_name", "metro_pop_2020", "place_name", "place_pop_2020",
               "exposure_area", "exposure_geoid", "exposure_pop_2020", "error"]
    out = out.reindex(columns=columns)
    out.to_csv(DATA_DIR / "city_population.csv", index=False)

    matched = out["exposure_pop_2020"].notna()
    shared = out[matched].groupby("exposure_geoid")["city_id"].count()
    print(f"\nWrote {DATA_DIR / 'city_population.csv'}: {matched.sum()}/{len(out)} stations matched; "
          f"{out['metro_geoid'].notna().sum()} in a metro area, {(matched & out['metro_geoid'].isna()).sum()} "
          f"county only; {out['place_name'].notna().sum()} inside city limits; "
          f"{int((shared > 1).sum())} metro/county areas hold more than one station.")
    if (~matched).any():
        print("Not matched:", ", ".join(out.loc[~matched, "name"]))


if __name__ == "__main__":
    main()
