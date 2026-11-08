# 03 — `city_info`: station history

One row per station combined into a city's threaded record (461 rows): `Name, ID, Lat, Lon, Stn.Name, Stn.stDate,
Stn.edDate`. 210 cities, 1–10 stations each (mean 2.2), 251 listed changes (65 on 1 January, 32 in 1989–2018). It is
the only metadata for station changes, the largest non-climatic signal in the record, so it drives the station
adjustment ([page 05](05-station-adjustment.md)). Plan reference: §1.

## C1 Schema and invariants (hard checks)

Dates parse; no station starts after it ends; every city's last station ends 2018-12-31 (210/210); lat/lon are
constant per city and equal to `annual_indices`; 210 IDs ↔ 210 names; the dataset card's two typo fixes are present
(Asheville `ASHEVILLE CITY` starts 1876-08-01; West Palm Beach `JUPITER-HYPOLUXO` ends 1899-12-01). `Stn.edDate` passes
`assert_no_post_cutoff`.

## C2 Overlapping stations

- **Found:** Abilene — `ABILENE CITY` ends 1944-02-29 but `ABILENE REGIONAL AP` starts 1943-03-01 (366 days listed
  for both).
- **Strategy:** the change date is the new station's start, as the source's `Station Segment` uses; the overlap years
  (1943, 1944) are kept out of the step windows. No data-driven date search (it would be fitted on the series it then
  adjusts).

## C3 Gaps between stations

- **Found:** 7 gaps — Tampa 1 day, West Palm Beach 1, Chicago 30, Colorado Springs 244, Kalamazoo 776, Rockford 1,096,
  Sioux City 36,524 (1896–1995).
- **Verified:** no daily reading falls inside a gap and none falls outside every station window (hard check). A gap is
  just missing days; recorded per year as `in_gap_days` in the coverage table.

## C4 Station periods without data

- **Found:** four periods with no reading at all — Asheville `ASHEVILLE CITY` 1876-08-01–1964-08-31, Louisville
  `LOUISVILLE WFO` 1995-11-01–2005-12-31, Lincoln `LINCOLN UNIV CAMPUS` 1954-07-01–1955-08-31, La Crosse `LA CROSSE`
  1985-10-01–1986-01-31. (Huntsville `HUNTSVILLE` 1958–2018 has data only in 2018.)
- **Strategy:** an empty period is **removed from the change list** — a station that supplied no data cannot shift the
  record. A change stays only if the stations before and after differ: La Crosse airport → (empty city) → airport and
  Louisville Standiford → (empty WFO) → Standiford are no change at all. Without this, the adjustment would test fake
  steps there.
- **Result:** 251 listed changes → 244 effective changes.

## C5 Names, returning stations and same-name changes

- **Found:** inconsistent names (`AP`/`ARPT`/`AIRPORT`/`Airport`, `INTL`/`INTERNATIONAL`, `W B`/`WB`, mixed case);
  30 rows in 13 cities where a station returns (A → B → A, e.g. West Palm Beach alternates 7 times in 1938–41);
  Akron 1948 changes to a station with the same name.
- **Strategy:** normalised names (`stn_name_norm`) and one `station_key` per distinct name within a city. A same-name
  change has the same key on both sides, so it is not a change (Akron 1948). **Return-pair rule:** if the adopted steps
  A → B and the later B → A do not cancel to within 0.5 °F, neither is applied (one is mis-estimated). It fired for
  Gainesville Tmax (+1.16 / −2.23 °F) and Omaha Tmax (−3.25 / +2.23 °F).

## C6 Short segments and change clusters

- **Found:** 17 station periods shorter than 5 years (9 shorter than 1 year) in 9 cities. v0 stopped every step window
  at the next change, so these got fewer than 5 years a side and no step at all — West Palm Beach's city → airport move
  was never adjusted.
- **Strategy:** consecutive changes less than 5 years apart form one **cluster**, adjusted as one step between the
  station before it (P) and the station after it (Q). 244 changes → 229 clusters; 7 have more than one change:

  | City | Cluster | P → Q | Treatment |
  |---|---|---|---|
  | West Palm Beach | 1938-07-01 … 1941-02-01 (7 changes) | City → Intl airport | one step; city days inside the cluster get the city's offset |
  | Moline | 1932-01-01 … 1937-02-18 (5) | Davenport → Moline airport | one step |
  | Gainesville | 1957-06-01, 1960-05-17 | Univ. → Regional AP | `3 WSW` days removed from the adjusted series if the step is adopted (adopted for Tmin: 1957-06 … 1960-05) |
  | San Antonio | 1941-01-01, 1942-07-01 | WBO → Intl AP | `Stinson` days, same rule (adopted for Tmin: 1941-01 … 1942-06) |
  | Louisville | 1945-07-01, 1948-01-01 | WBO → Standiford | `Bowman Fld` days, same rule (adopted for Tmax and Tmin: 1945-07 … 1947-12) |
  | Huntsville | 1954-07-15, 1958-09-01 | Huntsville → Huntsville | same endpoints: not a change |
  | Lansing | 1954-08-01, 1959-05-01 | Capital City AP → Capital City AP | same endpoints: not a change |

## C7 Changes on 1 January

A change on 1 January belongs to that year (the whole year is the new station). With this convention our
recomputation reproduces `Station Segment` and `Station Changes` in `coverage_and_stations` exactly.

## Output

`data/processed/stations_clean.parquet` (461 rows): the station period, normalised name, key, days with data, empty /
gap / overlap / return / same-name flags, the cluster, and for its cluster the step, t, critical t, years used and
the decision for Tmax and Tmin.
