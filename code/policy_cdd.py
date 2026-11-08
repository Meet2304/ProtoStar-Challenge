"""Policy analysis: where our forecasts put the biggest rise in air-conditioning need, and how many people live there.

Combines three things, all already in the repository:
    submission.csv            our CDD (cooling degree days) forecasts for 2019-2025
    data/annual_valid.csv     each city's observed CDD, 1989-2018 (valid years only)
    data/city_population.csv  2020 Census population of the metro area (or county) around each station
                              (python code/prepare_population.py)

For each city: the forecast 2019-2025 mean CDD minus its 1989-2018 mean (the 30-year average the
challenge uses as "normal"), in degree-days and in percent. For each metro area: the average rise of
its stations, its population, and "person-degree-days" (population x rise), a rough measure of how
much extra cooling a place needs in total. Stations whose record changed station during 2009-2018 are
flagged when the raw coverage_and_stations table is available, since a station move can shift the
recent level the forecast starts from.

Analysis only: nothing here feeds submission.csv.

Usage (from the repository root):
    python code/policy_cdd.py

Writes reports/policy_cdd_by_city.csv and reports/policy_cdd_by_area.csv, and prints a summary.
"""

import pandas as pd

from common import DATA_DIR, REPORTS_DIR, SUBMISSION_PATH, load_valid_annual

NORMAL_YEARS = (1989, 2018)
RECENT_CHANGE_YEARS = (2009, 2018)
TOP = 10
MIN_CDD_FOR_PERCENT = 500   # percent rises of cities with little cooling need are not meaningful


def station_changes() -> pd.Series | None:
    """city_id -> number of station changes in 2009-2018, if the raw table is on disk."""
    raw = DATA_DIR / "raw"
    for path in (raw / "coverage_and_stations", raw / "coverage_and_stations.parquet"):
        if path.exists():
            cov = pd.read_parquet(path)
            recent = cov[cov["Year"].between(*RECENT_CHANGE_YEARS)]
            return recent.groupby("ID")["Station Changes"].sum()
    return None


def main() -> None:
    annual = load_valid_annual()
    normal = (annual[annual["year"].between(*NORMAL_YEARS)]
              .groupby("city_id")["cdd"].agg(cdd_1989_2018="mean", valid_years="count"))
    normal = normal[normal["valid_years"] >= 10]

    sub = pd.read_csv(SUBMISSION_PATH, dtype={"city_id": str})
    fc = sub[sub["metric"] == "cdd"].pivot(index="city_id", columns="year", values="predicted_value")
    city = normal.join(pd.DataFrame({"cdd_forecast_2019_2025": fc.mean(axis=1),
                                     "cdd_forecast_2025": fc[2025]}), how="inner")
    city["rise"] = city["cdd_forecast_2019_2025"] - city["cdd_1989_2018"]
    city["rise_pct"] = 100 * city["rise"] / city["cdd_1989_2018"]

    pop = pd.read_csv(DATA_DIR / "city_population.csv", dtype={"city_id": str}).set_index("city_id")
    city = city.join(pop[["name", "state", "exposure_area", "exposure_geoid", "exposure_pop_2020",
                          "place_pop_2020"]], how="left")
    changes = station_changes()
    if changes is not None:
        city["station_changes_2009_2018"] = changes.reindex(city.index).fillna(0).astype(int)

    REPORTS_DIR.mkdir(exist_ok=True)
    city.round(2).sort_values("rise", ascending=False).to_csv(REPORTS_DIR / "policy_cdd_by_city.csv")

    known = city.dropna(subset=["exposure_geoid"])
    area = known.groupby("exposure_geoid").agg(
        area=("exposure_area", "first"), population_2020=("exposure_pop_2020", "first"),
        stations=("name", lambda s: ", ".join(s)), cdd_1989_2018=("cdd_1989_2018", "mean"),
        cdd_forecast_2019_2025=("cdd_forecast_2019_2025", "mean"), rise=("rise", "mean"))
    area["rise_pct"] = 100 * area["rise"] / area["cdd_1989_2018"]
    area["person_degree_days_millions"] = area["population_2020"] * area["rise"] / 1e6
    area = area.sort_values("person_degree_days_millions", ascending=False)
    area.round(2).to_csv(REPORTS_DIR / "policy_cdd_by_area.csv")

    total_pop = area["population_2020"].sum()
    weighted = (area["population_2020"] * area["rise"]).sum() / total_pop
    print(f"{len(city)} cities, {len(area)} metro/county areas, {total_pop / 1e6:.0f} million people (2020 Census).")
    print(f"Population-weighted forecast rise in CDD, 2019-2025 vs 1989-2018: {weighted:+.0f} degree-days a year "
          f"(median city {city['rise'].median():+.0f}, {city['rise_pct'].median():+.0f}%).")
    print(f"\nTop {TOP} areas by total extra cooling need (population x CDD rise):")
    cols = ["area", "population_2020", "cdd_1989_2018", "cdd_forecast_2019_2025", "rise", "rise_pct",
            "person_degree_days_millions"]
    print(area[cols].head(TOP).round(1).to_string(index=False))
    top_pop = area.head(TOP)["population_2020"].sum()
    print(f"\nThese {TOP} areas hold {top_pop / 1e6:.1f} million people ({100 * top_pop / total_pop:.0f}% of the total).")
    print(f"\nTop {TOP} cities by percent rise (cities with at least {MIN_CDD_FOR_PERCENT} CDD a year in 1989-2018):")
    cols = ["name", "state", "cdd_1989_2018", "cdd_forecast_2019_2025", "rise", "rise_pct"]
    if "station_changes_2009_2018" in city:
        cols.append("station_changes_2009_2018")
    big = city[city["cdd_1989_2018"] >= MIN_CDD_FOR_PERCENT]
    print(big.sort_values("rise_pct", ascending=False)[cols].head(TOP).round(1).to_string())
    if changes is None:
        print("\n(Station-change flags skipped: data/raw/coverage_and_stations not found; "
              "run python scripts/01_download_data.py to add them.)")


if __name__ == "__main__":
    main()
