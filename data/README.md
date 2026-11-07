# Data card: `data/`

The curated data the code in `code/` reads. `python code/prepare_data.py` rebuilds both files
from the challenge dataset; together with `code/` they reproduce `submission.csv`.

## Files

| File | What it holds | Source | License | Years covered | How we prepared it | Used for |
|---|---|---|---|---|---|---|
| `annual_valid.csv` | One row per city and year: `city_id`, `name`, `year` and the four metrics (`annual_avg_temp`, `total_precip_in`, `hottest_day_temp`, `cdd`) | [Challenge dataset](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), `annual_indices` | CC0-1.0 | 1851–2018 | Renamed columns to the metric codes; set a value to empty when that year missed more than 10 days of the data it uses (`Miss Tmax`/`Miss Tmin` for the temperature metrics and CDD, `Miss Prcp` for precipitation), the official scoring rule; rounded to 3 decimals | Forecast input |
| `city_info.csv` | One row per city: `city_id` (station `ID`), `name`, `lat`, `lon` | [Challenge dataset](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), `city_info` | CC0-1.0 | n/a | Kept one row per city (the current station) | Reference (not read by the starter model) |

## How the files fit together

- **Join key:** `city_id` = the station `ID` in the challenge `city_info` table (e.g., Pittsburgh = `USW00094823`).
- **Units:** °F, inches, °F-days, as in the source; no conversions.
- **Known limitations:** each city's record joins several stations over time, so a change of station
  can shift its level (see the `coverage_and_stations` subset, not used yet). 271–288 city-years since
  1989 fail the 10-day rule and are empty here.

## Data rule check

- [x] Every file marked "Forecast input" is weather data up to 2018-12-31, or data that does not respond to the weather. (`prepare_data.py` and `common.load_valid_annual()` stop if a row is dated after 2018.)
- [x] No 2019–2025 data that responds to the weather is used. (None is in this folder.)

## Citation

- Lai, Yuchuan; Dzombak, David (2019). Compiled historical daily temperature and precipitation data for selected 210 U.S. cities. Carnegie Mellon University. https://doi.org/10.1184/R1/7890488.v6
- Lai, Yuchuan; Dzombak, David (2019). Historical changes of annual temperature and precipitation indices at selected 210 U.S. cities. Carnegie Mellon University. https://doi.org/10.1184/R1/7961012.v6
