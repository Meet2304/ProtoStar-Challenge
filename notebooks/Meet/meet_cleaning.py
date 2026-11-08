"""Cleaning pipeline for the `daily` and `annual_indices` subsets (Meet's exploration).

Walk-through with explanations and charts: `02-cleaning-daily-annual.ipynb` (same folder).
Rebuild the processed files without the notebook (from the repo root):

    uv run python notebooks/Meet/meet_cleaning.py

Writes to `data/processed/meet/` (git-ignored):
    daily_clean.parquet    one row per city-day: cleaned, filled and station-adjusted values
    annual_clean.parquet   one row per city-year: official-rule targets plus input variants
    qc_log.csv             every daily value the pipeline deleted, with the rule that deleted it
    homog_steps.csv        every station change tested, with its estimated step and decision
Reads the tracked review file `notebooks/Meet/qc/daily_outlier_decisions.csv` (written on the first
run; edit its `final_action` column to override a proposed decision).

Data rule: only the challenge training data (weather up to 2018-12-31) is read, and every weather
table passes `assert_no_post_cutoff()`. Functions that build model inputs take `end_year`, so a
backtest that stops at year O uses no data after O.

Overview of the steps (details in each function's docstring and in the notebook):
    D1 structure checks          D2 impossible values      D3 sensor faults (stuck runs)
    D4 outliers vs neighbours    D5 annual table rebuilt from the cleaned days
    D6 missing days filled (inputs only)
    A1 sentinels and validity    A2 detached record fragments
    A3 station-change adjustment (inputs only)            A4 empty years filled from neighbours (inputs only)
    E  rolling-origin backtest of every input variant
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "code"))
from common import (CUTOFF_YEAR, MAX_MISSING_DAYS, METRICS, TRAIN_REPO,  # noqa: E402
                    TRAIN_REVISION, assert_no_post_cutoff)

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed" / "meet"
QC_DIR = HERE / "qc"
DECISIONS_PATH = QC_DIR / "daily_outlier_decisions.csv"
VARS = ["tmax", "tmin", "prcp"]

# ---------------------------------------------------------------------------------------------
# Settings. Every threshold is here so the notebook can show and test it.
# ---------------------------------------------------------------------------------------------
# D2: contiguous-U.S. records (NOAA State Climate Extremes Committee): 134 °F Death Valley 1913,
# -70 °F Rogers Pass MT 1954; 24-hour rain record about 42 in (Alvin TX 1979, rounded up).
TMAX_LIMIT, TMIN_LIMIT, PRCP_LIMIT = 134.0, -70.0, 43.0
# D3: a run of identical temperatures this long is tested against the neighbours
STUCK_MIN_DAYS = 7
STUCK_NEIGHBOUR_SD = 5.0      # °F: neighbours this variable over the same days -> the run is a fault
# D4: outliers
Z_CANDIDATE = 5.0             # robust z (city x calendar month) that makes a reading a candidate
Z_SUPPORT = 2.5               # a neighbour this unusual, same sign, within +-1 day, confirms it
QC_NEIGHBOURS = 3             # nearest cities used to confirm
SPIKE_Z = 3.0                 # robust-z jump from both the day before and the day after
RESID_Z_DELETE = 5.0          # still this far (z units) from what the neighbours imply ...
RESID_F_DELETE = 25.0         # ... and this far in °F -> delete (coastal heat days reach ~22 °F and are real)
MIN_NB_WITH_DATA = 2          # neighbours that must have a reading that day before anything is deleted
SPIKE_LOOK_DAYS = 3           # "day before / after" = nearest reading within this many days
# D6 / A4: neighbour weighting for infill and homogenisation
NB_K, NB_RADIUS_MI = 5, 300.0
FILL_MAX_MISSING = 30         # inputs only: a year may miss up to this many days and still be filled
LOCAL_WINDOW = 15             # days each side used for the local anomaly fallback
# A3: station-change adjustment
HOMOG_WINDOW = 10             # years each side of a change
HOMOG_MIN_YEARS = 5           # valid years needed on each side
HOMOG_MIN_T = 2.0             # |t| of the step needed to adjust
HOMOG_MIN_STEP = 0.25         # °F: smaller steps are left alone
# A2: a run of record years separated from the rest by this many empty years is dropped
FRAGMENT_GAP_YEARS = 30
# A4: neighbour infill of empty years
INFILL_OFFSET_YEARS = 15      # common years used to estimate the city-neighbour offset
INFILL_MIN_NEIGHBOURS = 2

TARGET_COL = {m: col for m, (col, _) in METRICS.items()}      # metric code -> annual_indices column
MISS_RULE = {  # annual_indices column -> the missing-day columns of the variables it uses
    **{c: ["Miss Tmax"] for c in ["Avg Tmax", "Min Tmax", "Warm Days", "Cold Days"]},
    **{c: ["Miss Tmin"] for c in ["Avg Tmin", "Max Tmin", "Min Tmin", "Warm Nights", "Cold Nights"]},
    **{c: ["Miss Prcp"] for c in ["Max 1-day P", "Max 5-day P", "Wet Days"]},
    **{col: miss for col, miss in METRICS.values()},
}


# =============================================================================================
# Loading
# =============================================================================================
def read_subset(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    """A raw subset from data/raw/ if downloaded, else streamed from the Hub at the pinned revision."""
    local = RAW_DIR / name
    if local.exists():
        return pd.read_parquet(local, columns=columns)
    return pd.read_parquet(f"hf://datasets/{TRAIN_REPO}@{TRAIN_REVISION}/{name}", columns=columns)


def load_daily_long() -> pd.DataFrame:
    d = read_subset("daily", ["ID", "Date", *VARS])
    d["Date"] = pd.to_datetime(d["Date"])
    d["ID"] = d["ID"].astype(str)
    assert_no_post_cutoff(d.assign(Year=d["Date"].dt.year), "Year", "daily")
    return d


def to_wide(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """{var: Date x city table}. Dates outside a city's record stay NaN (see `record_mask`)."""
    return {v: d.pivot(index="Date", columns="ID", values=v).sort_index(axis=1) for v in VARS}


def record_mask(d: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """True on the days inside each city's record (its first to last row in `daily`)."""
    span = d.groupby("ID")["Date"].agg(["min", "max"]).reindex(columns)
    dates = index.values[:, None]
    return pd.DataFrame((dates >= span["min"].values) & (dates <= span["max"].values),
                        index=index, columns=columns)


def load_annual() -> pd.DataFrame:
    """annual_indices joined row for row with coverage_and_stations."""
    a = read_subset("annual_indices")
    cv = read_subset("coverage_and_stations", ["ID", "Year", "Days Both", "Station Segment", "Station Changes"])
    for t, name in [(a, "annual_indices"), (cv, "coverage_and_stations")]:
        t["ID"] = t["ID"].astype(str)
        assert_no_post_cutoff(t, "Year", name)
    a = a.merge(cv, on=["ID", "Year"], how="left", validate="one_to_one")
    return a.sort_values(["ID", "Year"]).reset_index(drop=True)


def load_cities() -> pd.DataFrame:
    """city_id, name, lat, lon (curated data/city_info.csv, else built from the raw city_info)."""
    path = ROOT / "data" / "city_info.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"city_id": str}).set_index("city_id").sort_index()
    st = read_subset("city_info", ["ID", "Name", "Lat", "Lon"]).drop_duplicates("ID")
    return (st.rename(columns={"ID": "city_id", "Name": "name", "Lat": "lat", "Lon": "lon"})
              .set_index("city_id").sort_index())


def load_station_changes() -> pd.DataFrame:
    """One row per station change: city_id, the first day of the new station, its segment number.

    Used only as a lookup of change dates for step A3 (the stations table itself is out of scope).
    Abilene's old station is listed as ending 1944-02-29 although the next starts 1943-03-01: the
    start date of the new station is used, so the overlap does not matter here.
    """
    st = read_subset("city_info", ["ID", "Stn.stDate"]).rename(columns={"ID": "city_id", "Stn.stDate": "start"})
    st["city_id"] = st["city_id"].astype(str)
    st["start"] = pd.to_datetime(st["start"])
    st = st.sort_values(["city_id", "start"])
    st["segment"] = st.groupby("city_id").cumcount() + 1
    return st[st["segment"] > 1].reset_index(drop=True)


# =============================================================================================
# Neighbours
# =============================================================================================
def distance_matrix(cities: pd.DataFrame) -> pd.DataFrame:
    """Great-circle distance between cities, miles."""
    lat, lon = np.radians(cities["lat"].values), np.radians(cities["lon"].values)
    cosang = (np.sin(lat[:, None]) * np.sin(lat) + np.cos(lat[:, None]) * np.cos(lat)
              * np.cos(lon[:, None] - lon))
    return pd.DataFrame(np.arccos(np.clip(cosang, -1, 1)) * 3958.8, index=cities.index, columns=cities.index)


def nearest(dist: pd.DataFrame, k: int) -> dict[str, list[str]]:
    return {c: list(dist.loc[c].drop(c).nsmallest(k).index) for c in dist.index}


def idw_weights(dist: pd.DataFrame, k: int = NB_K, radius: float = NB_RADIUS_MI) -> pd.DataFrame:
    """Row c: inverse-distance-squared weights of c's k nearest cities within `radius` (0 elsewhere)."""
    w = pd.DataFrame(0.0, index=dist.index, columns=dist.columns)
    for c in dist.index:
        nb = dist.loc[c].drop(c).nsmallest(k)
        nb = nb[nb <= radius]
        w.loc[c, nb.index] = 1.0 / nb.values ** 2
    return w


def neighbour_mean(values: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weighted mean of each city's neighbours, row by row, skipping neighbours with NaN.

    values: rows = time, columns = cities. Returns (mean, number of neighbours that had a value).
    """
    cols = values.columns
    W = weights.loc[cols, cols].values
    have = values.notna().values.astype(float)
    num = np.nan_to_num(values.values) @ W.T
    den = have @ W.T
    n = have @ (W.T > 0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return pd.DataFrame(out, index=values.index, columns=cols), pd.DataFrame(n, index=values.index, columns=cols)


# =============================================================================================
# D1. Structure
# =============================================================================================
def structure_checks(d: pd.DataFrame) -> pd.DataFrame:
    """Keys unique, calendar complete inside each record, nothing after the cutoff."""
    span = d.groupby("ID")["Date"].agg(["min", "max", "size"])
    expected = (span["max"] - span["min"]).dt.days + 1
    rows = {
        "rows": len(d),
        "cities": d["ID"].nunique(),
        "first date": d["Date"].min().date(),
        "last date": d["Date"].max().date(),
        "duplicate (city, date) keys": int(d.duplicated(["ID", "Date"]).sum()),
        "cities with missing calendar days": int((expected != span["size"]).sum()),
        "days with no value at all (row present, all NaN)": int(d[VARS].isna().all(axis=1).sum()),
        "records starting on 1 Jan": int((span["min"].dt.day_of_year == 1).sum()),
        "records ending 2018-12-31": int((span["max"] == pd.Timestamp(f"{CUTOFF_YEAR}-12-31")).sum()),
    }
    return pd.Series(rows, name="value").to_frame()


# =============================================================================================
# D2. Impossible values
# =============================================================================================
def impossible_values(W: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Readings that cannot be right. Action: delete.

    - above / below the contiguous-U.S. records (tmax > 134, tmin < -70, prcp < 0 or > 43)
    - tmax below tmin on the same day: one of the two is wrong and the data cannot say which, so
      both temperatures of that day are deleted (precipitation is kept).
    """
    rules = {
        "tmax above U.S. record": ("tmax", W["tmax"] > TMAX_LIMIT),
        "tmin below U.S. record": ("tmin", W["tmin"] < TMIN_LIMIT),
        "tmax below U.S. record low": ("tmax", W["tmax"] < TMIN_LIMIT),
        "tmin above U.S. record high": ("tmin", W["tmin"] > TMAX_LIMIT),
        "prcp negative": ("prcp", W["prcp"] < 0),
        "prcp above U.S. 24-h record": ("prcp", W["prcp"] > PRCP_LIMIT),
    }
    out = [_cells(mask, W[var], var, rule) for rule, (var, mask) in rules.items()]
    order = W["tmax"] < W["tmin"]
    for var in ["tmax", "tmin"]:
        out.append(_cells(order, W[var], var, "tmax below tmin (both deleted)"))
    log = pd.concat(out, ignore_index=True)
    log["action"] = "delete"
    return log


def _cells(mask: pd.DataFrame, values: pd.DataFrame, var: str, rule: str) -> pd.DataFrame:
    s = values.where(mask).stack().dropna()
    return pd.DataFrame({"city_id": s.index.get_level_values(1), "date": s.index.get_level_values(0),
                         "var": var, "value": s.values, "rule": rule})


# =============================================================================================
# D3. Sensor faults: stuck temperature sensors
# =============================================================================================
def run_lengths(M: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each cell: the length of the run of identical values it belongs to, and its position (1 = first)."""
    same = (M == M.shift()) & M.notna()
    run_id = (~same).cumsum()
    pos = same.astype(int)
    length = pd.DataFrame(index=M.index, columns=M.columns, dtype=float)
    position = pd.DataFrame(index=M.index, columns=M.columns, dtype=float)
    for c in M.columns:
        g = pos[c].groupby(run_id[c])
        position[c] = g.cumsum() + 1
        length[c] = g.transform("sum") + 1
    return length.where(M.notna()), position.where(M.notna())


def stuck_runs(W: dict[str, pd.DataFrame], nbrs: dict[str, list[str]]) -> pd.DataFrame:
    """Runs of >= STUCK_MIN_DAYS identical tmax or tmin values, each tested against the neighbours.

    A sensor that stops updating repeats one value while the weather nearby keeps changing. A real
    steady spell (a Gulf Coast summer) repeats a value while the neighbours are steady too. Test:
    the median, over the 3 nearest cities, of their day-to-day standard deviation during the run.
    Action: if the neighbours varied by >= STUCK_NEIGHBOUR_SD °F, delete the run after its first day
    (the first reading may well be real); otherwise keep.
    """
    rows = []
    for var in ["tmax", "tmin"]:
        M = W[var]
        length, position = run_lengths(M)
        ends = (length >= STUCK_MIN_DAYS) & (position == length)
        for date, city in ends.stack()[lambda s: s].index:
            n = int(length.at[date, city])
            days = pd.date_range(end=date, periods=n)
            sds = [M.loc[days, nb].std() for nb in nbrs[city] if M.loc[days, nb].count() >= n // 2]
            nb_sd = float(np.median(sds)) if sds else np.nan
            rows.append({"city_id": city, "var": var, "start": days[0], "end": date, "days": n,
                         "value": M.at[date, city], "neighbour_sd": nb_sd,
                         "action": "delete" if nb_sd >= STUCK_NEIGHBOUR_SD else "keep"})
    return pd.DataFrame(rows)


def stuck_log(runs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in runs[runs["action"] == "delete"].itertuples():
        days = pd.date_range(r.start, r.end)[1:]
        out.append(pd.DataFrame({"city_id": r.city_id, "date": days, "var": r.var, "value": r.value,
                                 "rule": f"stuck sensor ({r.days} identical days, neighbours sd {r.neighbour_sd:.1f} °F)",
                                 "action": "delete"}))
    cols = ["city_id", "date", "var", "value", "rule", "action"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)


def repeated_precip(W: dict[str, pd.DataFrame], min_days: int = 3) -> pd.DataFrame:
    """Runs of identical non-zero precipitation (a multi-day total split evenly, or a repeated entry).

    Kept: an evenly split total leaves the annual total (the target) unchanged; only the 1-day
    maximum would be affected. Listed so the decision is on record.
    """
    P = W["prcp"].where(W["prcp"] > 0)
    length, position = run_lengths(P)
    ends = (length >= min_days) & (position == length)
    s = ends.stack()[lambda x: x]
    out = pd.DataFrame({"city_id": s.index.get_level_values(1), "end": s.index.get_level_values(0)})
    out["days"] = [int(length.at[d, c]) for d, c in zip(out["end"], out["city_id"])]
    out["value"] = [P.at[d, c] for d, c in zip(out["end"], out["city_id"])]
    out["start"] = out["end"] - pd.to_timedelta(out["days"] - 1, unit="D")
    out["action"] = "keep"
    return out[["city_id", "start", "end", "days", "value", "action"]]


# =============================================================================================
# D4. Outliers, confirmed or not by the neighbours
# =============================================================================================
def robust_z(M: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """z = (x - median) / (1.4826 * MAD) within each city and calendar month, over all years.

    Median and MAD instead of mean and SD so that the outliers being looked for do not widen the
    yardstick. Returns (z, median, scale).
    """
    month = M.index.month
    med = M.groupby(month).transform("median")
    scale = (M - med).abs().groupby(month).transform("median") * 1.4826
    scale = scale.where(scale > 0)
    return (M - med) / scale, med, scale


def outlier_candidates(W: dict[str, pd.DataFrame], nbrs: dict[str, list[str]]) -> pd.DataFrame:
    """Every tmax / tmin reading with robust |z| > Z_CANDIDATE, with the evidence used to decide it.

    Columns:
        z           robust z of the reading
        nb_support  the most extreme same-sign z among the 3 nearest cities, on the day or +-1 day
        nb_z        mean z of the neighbours on the same day (what the weather nearby was doing)
        resid_z     z - nb_z: how far the reading is from what the neighbours imply
        n_nb        how many of the 3 nearest cities have a reading that day
        spike       the reading jumps >= SPIKE_Z (in z units) away from both the nearest reading before and
                    after (within SPIKE_LOOK_DAYS days, so a missing day next to it does not hide a spike)
        year_extreme  the reading is its city's hottest tmax / coldest tmin of that year
    """
    rows = []
    for var in ["tmax", "tmin"]:
        M = W[var]
        Z, med, scale = robust_z(M)
        cand = Z.abs() > Z_CANDIDATE
        if not cand.values.any():
            continue
        # neighbour support: max same-sign |z| over 3 neighbours x 3 days
        pos_sup = pd.DataFrame(0.0, index=Z.index, columns=Z.columns)
        neg_sup = pd.DataFrame(0.0, index=Z.index, columns=Z.columns)
        nb_same = pd.DataFrame(index=Z.index, columns=Z.columns, dtype=float)
        nb_n = pd.DataFrame(0.0, index=Z.index, columns=Z.columns)
        for c in Z.columns:
            if not cand[c].any():
                continue
            nbz = Z[nbrs[c]]
            nb_same[c] = nbz.mean(axis=1)
            nb_n[c] = nbz.notna().sum(axis=1)
            stack = pd.concat([nbz.shift(k) for k in (-1, 0, 1)], axis=1)
            pos_sup[c] = stack.clip(lower=0).max(axis=1).fillna(0)
            neg_sup[c] = (-stack).clip(lower=0).max(axis=1).fillna(0)
        prev_day = M.shift(1).ffill(limit=SPIKE_LOOK_DAYS - 1)
        next_day = M.shift(-1).bfill(limit=SPIKE_LOOK_DAYS - 1)
        jump_prev = (M - prev_day) / scale
        jump_next = (M - next_day) / scale
        spike = ((jump_prev >= SPIKE_Z) & (jump_next >= SPIKE_Z)) | ((jump_prev <= -SPIKE_Z) & (jump_next <= -SPIKE_Z))
        year = M.index.year
        extreme = M.groupby(year).transform("max") if var == "tmax" else M.groupby(year).transform("min")
        s = Z.where(cand).stack().dropna()
        dates, cities = s.index.get_level_values(0), s.index.get_level_values(1)
        get = lambda T: T.stack(future_stack=True).reindex(s.index).values  # noqa: E731
        z = s.values
        rows.append(pd.DataFrame({
            "city_id": cities, "date": dates, "var": var, "value": get(M), "z": z,
            "nb_support": np.where(z > 0, get(pos_sup), get(neg_sup)),
            "nb_z": get(nb_same), "n_nb": get(nb_n).astype(int), "spike": get(spike).astype(bool),
            "year_extreme": get(M == extreme).astype(bool),
            "median": get(med), "scale": get(scale),
            "prev_day": get(prev_day), "next_day": get(next_day),
        }))
    out = pd.concat(rows, ignore_index=True)
    out["resid_z"] = out["z"] - out["nb_z"].fillna(0)
    out["expected"] = out["median"] + out["nb_z"].fillna(0) * out["scale"]   # what the neighbours imply, °F
    out["resid_f"] = out["value"] - out["expected"]
    out["class"] = np.where(out["nb_support"] >= Z_SUPPORT, "confirmed by neighbours", "isolated")
    return out


def propose_actions(cand: pd.DataFrame) -> pd.DataFrame:
    """Turn the evidence into an action for every candidate (keep / delete) and a reason.

    1. Confirmed by a neighbour                                          -> keep (real weather)
    2. Isolated, a one-day spike, >= RESID_Z_DELETE (z units) and >= RESID_F_DELETE °F
       from what the neighbours imply, and >= MIN_NB_WITH_DATA neighbours
       had a reading that day                                            -> delete
       (nobody nearby saw it, and the city itself was normal the day before and after: the
       signature of a keying or transmission error)
    3. Isolated but lasting more than a day, or close to what the
       neighbours imply                                                  -> keep
       (a local event that lasts, e.g. a coastal heat day or a cold-air pool, is real weather;
       a sensor that is wrong for days is caught by D3 instead)
    The °F test matters in steady coastal climates (Eureka, San Diego, Astoria): their spread is so
    small that a real one-day offshore-wind heat event reaches z = 7-8 but only 15-22 °F.
    The neighbour-data test matters before 1900: many readings are "isolated" only because the
    neighbours had no data (e.g. the January 1888 cold wave in Spokane and Boise), which is no
    evidence against them.
    """
    c = cand.copy()
    rule2 = ((c["class"] == "isolated") & c["spike"] & (c["resid_z"].abs() >= RESID_Z_DELETE)
             & (c["resid_f"].abs() >= RESID_F_DELETE) & (c["n_nb"] >= MIN_NB_WITH_DATA))
    c["proposed_action"] = np.where(rule2, "delete", "keep")
    c["reason"] = np.select(
        [c["class"] == "confirmed by neighbours", rule2,
         c["n_nb"] < MIN_NB_WITH_DATA, ~c["spike"]],
        ["a nearby city was unusual the same way within a day",
         "one-day spike, no nearby city saw it, far from what neighbours imply",
         "neighbours had no data that day: no evidence against it",
         "lasts more than a day (weather, or a fault D3 would catch)"],
        f"one-day spike, but within {RESID_F_DELETE:g} °F or {RESID_Z_DELETE:g} z of what neighbours imply")
    return c


def decisions_file(cand: pd.DataFrame, path: Path = DECISIONS_PATH) -> pd.DataFrame:
    """The review file for the isolated candidates (confirmed ones are kept without review).

    On first use it is written with `final_action` empty. A reviewer may set `final_action` to
    keep / delete and add a `note`; reruns keep those entries and add any new candidates. The
    pipeline applies `final_action` where set, else `proposed_action`.
    """
    iso = cand[cand["class"] == "isolated"].copy()
    iso["date"] = pd.to_datetime(iso["date"]).dt.date.astype(str)
    cols = ["city_id", "date", "var", "prev_day", "value", "next_day", "expected", "z", "nb_support",
            "nb_z", "n_nb", "resid_z", "resid_f", "spike", "year_extreme", "proposed_action", "reason"]
    iso = iso[cols].round(2)
    if path.exists():
        old = pd.read_csv(path, dtype={"city_id": str, "final_action": str, "note": str})
        keep = old[["city_id", "date", "var", "final_action", "note"]]
        iso = iso.merge(keep, on=["city_id", "date", "var"], how="left")
    else:
        iso["final_action"], iso["note"] = "", ""
    iso[["final_action", "note"]] = iso[["final_action", "note"]].fillna("")
    path.parent.mkdir(parents=True, exist_ok=True)
    iso.sort_values(["date", "city_id", "var"]).to_csv(path, index=False)
    return iso


def outlier_log(decisions: pd.DataFrame) -> pd.DataFrame:
    act = decisions["final_action"].where(decisions["final_action"].isin(["keep", "delete"]),
                                          decisions["proposed_action"])
    dl = decisions[act == "delete"]
    return pd.DataFrame({"city_id": dl["city_id"], "date": pd.to_datetime(dl["date"]), "var": dl["var"],
                         "value": dl["value"],
                         "rule": np.where(dl["final_action"] == "delete", "outlier (reviewer)",
                                          "outlier (isolated one-day spike)"),
                         "action": "delete"})


def apply_log(W: dict[str, pd.DataFrame], log: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """A copy of W with every logged deletion set to NaN."""
    out = {v: m.copy() for v, m in W.items()}
    for var, g in log[log["action"] == "delete"].groupby("var"):
        M = out[var]
        r = M.index.get_indexer(pd.to_datetime(g["date"]))
        c = M.columns.get_indexer(g["city_id"])
        vals = M.to_numpy(copy=True)
        vals[r, c] = np.nan
        out[var] = pd.DataFrame(vals, index=M.index, columns=M.columns)
    return out


def precip_extremes(W: dict[str, pd.DataFrame], nbrs: dict[str, list[str]]) -> pd.DataFrame:
    """Wet days above 3x the city's 99.9th percentile of wet days, with neighbour support.

    Action: keep (the physical limit is enforced in D2). Heavy convective rain is often local, so
    a missing neighbour signal is not evidence of an error, and deleting a real downpour would make
    the yearly total (a target) read low. Listed for the record.
    """
    P = W["prcp"]
    q = P.where(P > 0).quantile(0.999)
    s = P.where(P > 3 * q).stack().dropna()
    rows = []
    for (date, city), val in s.items():
        win = P.loc[date - pd.Timedelta(days=1):date + pd.Timedelta(days=1), nbrs[city]]
        rows.append({"city_id": city, "date": date, "prcp": val, "city_p999": q[city],
                     "max_neighbour_prcp_pm1day": win.max().max()})
    out = pd.DataFrame(rows)
    out["action"] = "keep"
    return out


# =============================================================================================
# D5. Annual table rebuilt from the (cleaned) days
# =============================================================================================
RECOMPUTED = ["Miss Tmax", "Miss Tmin", "Miss Prcp", "Days Both", "Avg Tmax", "Avg Tmin", "Avg Temp",
              "ToT Prcp", "Max Tmax", "Min Tmax", "Max Tmin", "Min Tmin", "cdd", "Max 1-day P", "Max 5-day P"]


def annual_from_daily(W: dict[str, pd.DataFrame], record: pd.DataFrame) -> pd.DataFrame:
    """The annual_indices columns that are plain functions of the days, one row per city-year.

    Definitions (checked against the published table in the notebook):
      Avg Temp = (mean daily tmax + mean daily tmin) / 2, each over the days that have it
      ToT Prcp = sum of daily prcp over days with a value;  Max Tmax = warmest daily tmax
      cdd      = sum of max(0, (tmax + tmin) / 2 - 65) over days with both values
      Miss X   = days in the record that year without X;  Days Both = days with tmax and tmin
    The percentile-based indices (Warm/Cold Days/Nights, Wet Days) need the original reference
    period and are taken from annual_indices instead.
    """
    tx, tn, pr = W["tmax"], W["tmin"], W["prcp"]
    rec = record.loc[tx.index]
    yr = tx.index.year
    days = rec.groupby(yr).sum()
    both = tx.notna() & tn.notna()
    tavg = ((tx + tn) / 2).where(both)
    out = {
        "Miss Tmax": days - tx.groupby(yr).count(), "Miss Tmin": days - tn.groupby(yr).count(),
        "Miss Prcp": days - pr.groupby(yr).count(), "Days Both": both.groupby(yr).sum(),
        "Avg Tmax": tx.groupby(yr).mean(), "Avg Tmin": tn.groupby(yr).mean(),
        "ToT Prcp": pr.groupby(yr).sum(min_count=1),
        "Max Tmax": tx.groupby(yr).max(), "Min Tmax": tx.groupby(yr).min(),
        "Max Tmin": tn.groupby(yr).max(), "Min Tmin": tn.groupby(yr).min(),
        "cdd": (tavg - 65).clip(lower=0).groupby(yr).sum(min_count=1),
        "Max 1-day P": pr.groupby(yr).max(),
        "Max 5-day P": pr.rolling(5, min_periods=1).sum().where(pr.notna()).groupby(yr).max(),
    }
    out["Avg Temp"] = (out["Avg Tmax"] + out["Avg Tmin"]) / 2
    long = pd.concat({k: v.stack(future_stack=True) for k, v in out.items()}, axis=1)
    long.index.names = ["Year", "ID"]
    long["days_in_year"] = days.stack(future_stack=True)
    long = long[long["days_in_year"] > 0].reset_index()
    return long[["ID", "Year", "days_in_year", *RECOMPUTED]].sort_values(["ID", "Year"]).reset_index(drop=True)


def apply_validity(a: pd.DataFrame, rule: dict[str, list[str]] = MISS_RULE,
                   max_missing: int = MAX_MISSING_DAYS) -> pd.DataFrame:
    """±inf -> NaN, then each column -> NaN when its year misses > max_missing days of what it uses."""
    a = a.copy()
    cols = [c for c in rule if c in a.columns]
    a[cols] = a[cols].replace([np.inf, -np.inf], np.nan)
    for col in cols:
        a.loc[a[rule[col]].max(axis=1) > max_missing, col] = np.nan
    return a


# =============================================================================================
# A2. Detached record fragments
# =============================================================================================
def record_fragments(annual: pd.DataFrame, gap: int = FRAGMENT_GAP_YEARS) -> pd.DataFrame:
    """Blocks of years with data that are cut off from the rest of the record by > `gap` empty years.

    Only the block that reaches the present is kept. Example: Sioux City has 1890-1895 from an old
    station, then a century with no data, then the airport from 1996. Six years from a different
    site, a hundred years earlier, cannot inform a 2019-2025 forecast and would distort any
    'all-history' statistic. Action: delete those years (daily and annual).
    """
    has = annual[annual[["Miss Tmax", "Miss Tmin", "Miss Prcp"]].min(axis=1) < annual["days_in_year"]]
    rows = []
    for city, g in has.groupby("ID"):
        yrs = np.sort(g["Year"].to_numpy())
        breaks = np.where(np.diff(yrs) > gap)[0]
        if len(breaks):
            cut = yrs[breaks[-1] + 1]
            rows.append({"city_id": city, "first_dropped": int(yrs[0]), "last_dropped": int(yrs[breaks[-1]]),
                         "years_with_data_dropped": int((yrs < cut).sum()), "record_resumes": int(cut)})
    return pd.DataFrame(rows, columns=["city_id", "first_dropped", "last_dropped",
                                       "years_with_data_dropped", "record_resumes"])


def drop_fragments(W: dict[str, pd.DataFrame], record: pd.DataFrame,
                   frags: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    W = {v: m.copy() for v, m in W.items()}
    record = record.copy()
    for r in frags.itertuples():
        upto = f"{r.record_resumes - 1}-12-31"
        for m in W.values():
            m.loc[:upto, r.city_id] = np.nan
        record.loc[:upto, r.city_id] = False
    return W, record


# =============================================================================================
# D6. Missing days filled (inputs only)
# =============================================================================================
def doy365(index: pd.DatetimeIndex) -> np.ndarray:
    """Day of year on a 365-day calendar: 29 Feb shares 28 Feb's slot (1..365)."""
    doy = index.day_of_year.to_numpy().copy()
    after_feb28 = (index.month > 2) | ((index.month == 2) & (index.day == 29))
    doy[np.asarray(index.is_leap_year & after_feb28)] -= 1
    return doy


def climatology(M: pd.DataFrame, end_year: int, years: int = 30, smooth: int = 31) -> pd.DataFrame:
    """Each city's normal for every calendar day: mean over the `years` years ending at end_year,
    smoothed with a circular `smooth`-day moving average. A city with fewer than 5 years of data in
    that window falls back to all its years up to end_year."""
    M = M.loc[:f"{end_year}-12-31"]
    doy = doy365(M.index)
    recent = np.asarray(M.index.year > end_year - years)

    def norm(mask):
        clim = M[mask].groupby(doy[mask]).mean().reindex(range(1, 366))
        ext = pd.concat([clim.iloc[-smooth:], clim, clim.iloc[:smooth]])
        return ext.rolling(smooth, center=True, min_periods=smooth // 2).mean().iloc[smooth:-smooth]

    clim = norm(recent)
    thin = M[recent].notna().sum() < 5 * 365
    if thin.any():
        full = norm(np.ones(len(M), bool))
        clim.loc[:, thin] = full.loc[:, thin]
    return clim


def expand(clim: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(clim.loc[doy365(index)].to_numpy(), index=index, columns=clim.columns)


FILL_METHODS = ["normal", "normal + local anomaly", "normal + neighbour anomaly"]


def fill_daily(W: dict[str, pd.DataFrame], record: pd.DataFrame, weights: pd.DataFrame, end_year: int,
               method: str = "normal + neighbour anomaly") -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Estimate every missing day inside a record (inputs only: targets keep the official rule).

    normal                        the city's normal for that calendar day
    normal + local anomaly        + how far the city ran above/below normal in the 15 days either side
    normal + neighbour anomaly    + how far nearby cities (<= 300 mi, inverse-distance weights) ran
                                  above/below *their* normal that same day; falls back to the local
                                  anomaly, then to the normal
    Precipitation uses ratios instead of differences (the neighbours' rain as a multiple of their
    normal), because rain cannot go below zero.
    Returns (filled values, mask of filled cells). Uses data up to end_year only.
    """
    filled, was_filled = {}, {}
    rec = record.loc[:f"{end_year}-12-31"]
    for var in VARS:
        M = W[var].loc[:f"{end_year}-12-31"]
        C = expand(climatology(M, end_year), M.index)
        ratio = var == "prcp"
        A = M / C.where(C > 0.005) if ratio else M - C
        if method == "normal":
            adj = None
        else:
            adj = A.rolling(2 * LOCAL_WINDOW + 1, center=True, min_periods=5).mean()
            if method == "normal + neighbour anomaly":
                adj = neighbour_mean(A, weights)[0].fillna(adj)
        if ratio:
            est = C if adj is None else C * adj.fillna(1.0)
        else:
            est = C if adj is None else C + adj.fillna(0.0)
        need = M.isna() & rec
        filled[var] = M.where(~need, est)
        was_filled[var] = need & est.notna()
    tx, tn = filled["tmax"], filled["tmin"]           # keep tmax >= tmin on filled days
    swap = (tx < tn) & (was_filled["tmax"] | was_filled["tmin"])
    mid = (tx + tn) / 2
    filled["tmax"], filled["tmin"] = tx.where(~swap, mid), tn.where(~swap, mid)
    return filled, was_filled


def annual_filled(Wf: dict[str, pd.DataFrame], was_filled: dict[str, pd.DataFrame],
                  record: pd.DataFrame, qc_annual: pd.DataFrame, max_missing: int = FILL_MAX_MISSING) -> pd.DataFrame:
    """Avg Temp, Avg Tmax, Avg Tmin, ToT Prcp and cdd from filled days, for years missing <= max_missing.

    Max Tmax (hottest day) is never filled: a filled day is an average-like estimate, and the hottest
    day is exactly the kind of extreme an estimate cannot recover.
    """
    a = annual_from_daily(Wf, record)
    nf = pd.concat({f"n_filled_{v}": was_filled[v].groupby(was_filled[v].index.year).sum().stack()
                    for v in VARS}, axis=1)
    nf.index.names = ["Year", "ID"]
    a = a.merge(nf.reset_index(), on=["ID", "Year"], how="left")
    miss = qc_annual.set_index(["ID", "Year"])[["Miss Tmax", "Miss Tmin", "Miss Prcp"]].add_prefix("qc ")
    a = a.join(miss, on=["ID", "Year"])
    t_ok = a[["qc Miss Tmax", "qc Miss Tmin"]].max(axis=1) <= max_missing
    p_ok = a["qc Miss Prcp"] <= max_missing
    out = a[["ID", "Year", "n_filled_tmax", "n_filled_tmin", "n_filled_prcp"]].copy()
    for col in ["Avg Temp", "Avg Tmax", "Avg Tmin", "cdd"]:
        out[col] = a[col].where(t_ok)
    out["ToT Prcp"] = a["ToT Prcp"].where(p_ok)
    return out


def simulate_fill(W: dict[str, pd.DataFrame], record: pd.DataFrame, weights: pd.DataFrame,
                  n_samples: int = 600, seed: int = 0, years: tuple[int, int] = (1989, 2018)) -> pd.DataFrame:
    """Hide real days in complete city-years and compare each fix with the truth.

    For k = 10 and 30 hidden days, as one block or scattered: % error of the yearly total
    (precipitation, CDD) and °F error of the yearly mean temperature, for: raw (no fix), scaling
    the observed days up to a full year, and the three fill methods.
    """
    rng = np.random.default_rng(seed)
    lo, hi = f"{years[0]}-01-01", f"{years[1]}-12-31"
    Ws = {v: W[v].loc[lo:hi] for v in VARS}
    rec = record.loc[lo:hi]
    yr = Ws["tmax"].index.year
    complete = Ws["tmax"].notna() & Ws["tmin"].notna() & Ws["prcp"].notna()
    complete = complete.groupby(yr).all().stack()
    complete = complete[complete].index.to_list()
    picks = [complete[i] for i in rng.choice(len(complete), min(n_samples, len(complete)), replace=False)]

    def stats(tx, tn, pr):
        both = tx.notna() & tn.notna()
        return {"Avg Temp": (tx.mean() + tn.mean()) / 2, "ToT Prcp": pr.sum(),
                "cdd": ((tx + tn) / 2 - 65).clip(lower=0)[both].sum()}

    rows = []
    for k in (10, 30):
        for pattern in ("block", "scattered"):
            masked = {v: m.copy() for v, m in Ws.items()}
            hidden = []
            for y, c in picks:
                idx = np.where(yr == y)[0]
                if pattern == "block":
                    s = rng.integers(0, len(idx) - k)
                    days = idx[s:s + k]
                else:
                    days = np.sort(rng.choice(idx, k, replace=False))
                j = masked["tmax"].columns.get_loc(c)
                for v in VARS:
                    masked[v].iloc[days, j] = np.nan
                hidden.append((y, c, idx))
            fills = {m: fill_daily(masked, rec, weights, years[1], m)[0] for m in FILL_METHODS}
            for y, c, idx in hidden:
                t = stats(*(Ws[v].iloc[idx][c] for v in VARS))
                obs = stats(*(masked[v].iloc[idx][c] for v in VARS))
                share = len(idx) / masked["prcp"].iloc[idx][c].notna().sum()
                cand = {"raw (no fix)": obs,
                        "scale up to a full year": {"Avg Temp": obs["Avg Temp"], "ToT Prcp": obs["ToT Prcp"] * share,
                                                    "cdd": obs["cdd"] * share}}
                for m, F in fills.items():
                    cand[m] = stats(*(F[v].iloc[idx][c] for v in VARS))
                for meth, st in cand.items():
                    for metric in ("Avg Temp", "ToT Prcp", "cdd"):
                        if metric != "Avg Temp" and t[metric] <= 0:
                            continue
                        err = st[metric] - t[metric]
                        rows.append({"hidden days": k, "pattern": pattern, "method": meth, "metric": metric,
                                     "error": err if metric == "Avg Temp" else 100 * err / t[metric]})
    return pd.DataFrame(rows)


# =============================================================================================
# A3. Station-change adjustment (inputs only)
# =============================================================================================
def estimate_steps(annual: pd.DataFrame, changes: pd.DataFrame, weights: pd.DataFrame, end_year: int,
                   iterations: int = 2) -> pd.DataFrame:
    """Size of the jump in Avg Tmax and Avg Tmin at every station change, measured against neighbours.

    For a change in year Y: the city's yearly values minus a reference built from its neighbours
    (inverse-distance mean of each neighbour's departure from its own mean over the window), in the
    HOMOG_WINDOW years before and after (Y itself excluded unless the change is on 1 Jan; windows stop
    at the city's other changes and at end_year). Shared weather cancels in the difference, so a shift
    in the difference is the station. Step = mean after - mean before; adopted when |t| >= HOMOG_MIN_T
    and |step| >= HOMOG_MIN_STEP °F with >= HOMOG_MIN_YEARS valid years on each side.
    A second pass repeats the estimate with neighbours that were themselves adjusted in the first.
    `annual` must hold valid (10-day rule) Avg Tmax / Avg Tmin.
    """
    ch = changes[changes["start"].dt.year <= end_year].copy()
    ch["year"] = ch["start"].dt.year
    ch["jan1"] = (ch["start"].dt.month == 1) & (ch["start"].dt.day == 1)
    tables = {v: annual.pivot(index="Year", columns="ID", values=v).loc[:end_year] for v in ["Avg Tmax", "Avg Tmin"]}
    adjusted = {v: t.copy() for v, t in tables.items()}
    steps = pd.DataFrame()
    for _ in range(iterations):
        rows = []
        for var, T in tables.items():
            ref_T = adjusted[var]
            for city, g in ch.groupby("city_id"):
                if city not in T.columns:
                    continue
                w = weights.loc[city]
                nbs = w[w > 0].index.intersection(T.columns)
                ww = w[nbs].to_numpy()
                yrs_all = sorted(g["year"].tolist())
                for r in g.itertuples():
                    prev_ch = max([y for y in yrs_all if y < r.year], default=-10_000)
                    next_ch = min([y for y in yrs_all if y > r.year], default=10_000)
                    after_start = r.year if r.jan1 else r.year + 1
                    before = list(range(max(r.year - HOMOG_WINDOW, prev_ch + 1), r.year))
                    after = list(range(after_start, min(after_start + HOMOG_WINDOW, next_ch, end_year + 1)))
                    row = {"city_id": city, "var": var, "change_date": r.start, "year": r.year,
                           "n_neighbours": len(nbs), "step": np.nan, "t": np.nan, "n_before": 0, "n_after": 0}
                    if len(nbs) and before and after:
                        win = ref_T.index.intersection(before + after)
                        nb = ref_T.loc[win, nbs]
                        nb_anom = nb - nb.mean()
                        have = nb_anom.notna().to_numpy().astype(float)
                        den = have @ ww
                        with np.errstate(invalid="ignore", divide="ignore"):
                            ref = np.where(den > 0, (nb_anom.fillna(0).to_numpy() @ ww) / den, np.nan)
                        dif = (T.loc[win, city] - ref).dropna()
                        b, a = dif[dif.index < r.year], dif[dif.index >= after_start]
                        row.update(n_before=len(b), n_after=len(a))
                        if len(b) >= HOMOG_MIN_YEARS and len(a) >= HOMOG_MIN_YEARS:
                            step = a.mean() - b.mean()
                            se = np.sqrt(b.var(ddof=1) / len(b) + a.var(ddof=1) / len(a))
                            row.update(step=step, t=step / se if se > 0 else np.inf)
                    rows.append(row)
        steps = pd.DataFrame(rows)
        steps["applied"] = (steps["t"].abs() >= HOMOG_MIN_T) & (steps["step"].abs() >= HOMOG_MIN_STEP)
        for var, T in tables.items():       # neighbours for the next pass, already adjusted
            adjusted[var] = T + annual_offsets(steps[steps["var"] == var], T.index, T.columns)
    return steps


def _offset_by_date(steps: pd.DataFrame, dates: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """Offset to add on each date. Zero from the last applied change on; before a change, the sum of
    the steps of that and all later applied changes (old readings move onto today's station level)."""
    out = pd.DataFrame(0.0, index=dates, columns=columns)
    for city, g in steps[steps["applied"]].groupby("city_id"):
        if city not in columns:
            continue
        g = g.sort_values("change_date")
        cum = np.append(g["step"].to_numpy()[::-1].cumsum()[::-1], 0.0)
        pos = np.searchsorted(g["change_date"].to_numpy(), dates.to_numpy(), side="right")
        out[city] = cum[pos]
    return out


def annual_offsets(steps: pd.DataFrame, years: pd.Index, columns: pd.Index) -> pd.DataFrame:
    dates = pd.date_range(f"{years.min()}-01-01", f"{years.max()}-12-31")
    return _offset_by_date(steps, dates, columns).groupby(dates.year).mean().reindex(years)


def homogenise_daily(W: dict[str, pd.DataFrame], steps: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Daily tmax / tmin with the applied steps added (prcp unchanged). Returns (values, offsets)."""
    out, offs = dict(W), {}
    for var, col in [("tmax", "Avg Tmax"), ("tmin", "Avg Tmin")]:
        off = _offset_by_date(steps[steps["var"] == col], W[var].index, W[var].columns)
        offs[var] = off
        out[var] = W[var] + off
    return out, offs


# =============================================================================================
# A4. Empty years filled from neighbours (inputs only)
# =============================================================================================
def neighbour_infill(T: pd.DataFrame, weights: pd.DataFrame, end_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill a city's missing years from its neighbours' values that year plus the usual difference.

    T: city x year table of valid values. For a missing year y and each neighbour n (<= 300 mi) with a
    value in y: estimate = x_n,y + mean(x_city - x_n) over the INFILL_OFFSET_YEARS common valid years
    closest to y (all <= end_year, at least 5). Estimates are averaged with inverse-distance weights;
    at least INFILL_MIN_NEIGHBOURS are required. Only years from the city's first valid year to
    end_year are filled (never before a record starts). Returns (filled table, mask of filled cells).
    """
    T = T.loc[:, :end_year]
    years = T.columns.to_numpy()
    X = T.to_numpy(dtype=float)
    pos = {c: i for i, c in enumerate(T.index)}
    out = X.copy()
    for i, c in enumerate(T.index):
        valid = ~np.isnan(X[i])
        if not valid.any():
            continue
        targets = np.where(~valid & (years >= years[valid][0]))[0]
        if not len(targets):
            continue
        w = weights.loc[c]
        num, den, cnt = np.zeros(len(targets)), np.zeros(len(targets)), np.zeros(len(targets))
        for n, wt in w[w > 0].items():
            j = pos.get(n)
            if j is None:
                continue
            both = valid & ~np.isnan(X[j])
            if both.sum() < 5:
                continue
            diff, yb = X[i][both] - X[j][both], years[both]
            for k, t in enumerate(targets):
                if np.isnan(X[j, t]):
                    continue
                near = np.argsort(np.abs(yb - years[t]), kind="stable")[:INFILL_OFFSET_YEARS]
                num[k] += wt * (X[j, t] + diff[near].mean())
                den[k] += wt
                cnt[k] += 1
        ok = cnt >= INFILL_MIN_NEIGHBOURS
        out[i, targets[ok]] = num[ok] / den[ok]
    filled = pd.DataFrame(out, index=T.index, columns=T.columns)
    return filled, filled.notna() & T.isna()


def leave_one_out_infill(T: pd.DataFrame, weights: pd.DataFrame, years: tuple[int, int] = (1989, 2018),
                         n_samples: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Hide known years one at a time; error of neighbour infill vs two simple fills."""
    rng = np.random.default_rng(seed)
    T = T.loc[:, :years[1]]
    cells = list(T.loc[:, years[0]:years[1]].stack().index)
    picks = [cells[i] for i in rng.choice(len(cells), min(n_samples, len(cells)), replace=False)]
    rows = []
    for c, y in picks:
        truth = T.at[c, y]
        nbs = [n for n in weights.columns[weights.loc[c] > 0] if n in T.index]
        sub = T.loc[[c, *nbs]].copy()
        sub.at[c, y] = np.nan
        row = sub.loc[c]
        prev = row.loc[:y - 1].dropna()
        est = neighbour_infill(sub, weights, years[1])[0].at[c, y]
        rows.append({"city mean, 15 years either side": row.loc[y - 15:y + 15].mean() - truth,
                     "previous valid year": (prev.iloc[-1] - truth) if len(prev) else np.nan,
                     "neighbour infill": est - truth})
    return pd.DataFrame(rows)


# =============================================================================================
# Pipeline
# =============================================================================================
FILL_METHOD = "normal + neighbour anomaly"     # chosen by simulate_fill (notebook section D6)
METRIC_COL = {"annual_avg_temp": "Avg Temp", "total_precip_in": "ToT Prcp",
              "hottest_day_temp": "Max Tmax", "cdd": "cdd"}
# Which input variant feeds `<metric>_input` in annual_clean.parquet: the result of choose_inputs()
# on the backtest (notebook section E checks that this constant still matches). Targets are never
# taken from a variant.
INPUT_RECIPE = {"annual_avg_temp": "qc+homog", "total_precip_in": "qc", "hottest_day_temp": "qc",
                "cdd": "qc+fill+homog+nbfill"}
MIN_GAIN, MIN_WINS = 0.002, 3     # selection rule, see choose_inputs()


def prepare(write_decisions: bool = True, keep_raw: bool = False, verbose: bool = True) -> dict:
    """D1-D4 and A2: load, check, decide and delete. Returns everything later steps need."""
    say = print if verbose else (lambda *a, **k: None)
    d = load_daily_long()
    checks = structure_checks(d)
    W = to_wide(d)
    record = record_mask(d, W["tmax"].index, W["tmax"].columns)
    del d
    cities = load_cities().loc[W["tmax"].columns]
    dist = distance_matrix(cities)
    nbrs = nearest(dist, QC_NEIGHBOURS)
    weights = idw_weights(dist)
    say("D2 impossible values ...")
    impossible = impossible_values(W)
    say("D3 stuck sensors ...")
    runs = stuck_runs(W, nbrs)
    rep_prcp = repeated_precip(W)
    say("D4 outliers ...")
    cand = propose_actions(outlier_candidates(W, nbrs))
    decisions = decisions_file(cand) if write_decisions else cand[cand["class"] == "isolated"].assign(final_action="", note="")
    pext = precip_extremes(W, nbrs)
    log = pd.concat([impossible, stuck_log(runs), outlier_log(decisions)], ignore_index=True)
    log["date"] = pd.to_datetime(log["date"])
    log = log.drop_duplicates(["city_id", "date", "var"])
    Wc = apply_log(W, log)
    say("A2 record fragments ...")
    raw_rebuilt = annual_from_daily(W, record)
    frags = record_fragments(raw_rebuilt)
    Wc, record_c = drop_fragments(Wc, record, frags)
    P = dict(checks=checks, W_clean=Wc, record=record_c, record_raw=record, cities=cities, dist=dist, nbrs=nbrs, weights=weights,
             annual_raw=load_annual(), raw_rebuilt=raw_rebuilt, impossible=impossible, stuck=runs,
             repeated_prcp=rep_prcp, candidates=cand, decisions=decisions, precip_extremes=pext, log=log,
             fragments=frags, changes=load_station_changes())
    if keep_raw:
        P["W_raw"] = W
    return P


def build(P: dict, end_year: int = CUTOFF_YEAR, fill_method: str = FILL_METHOD) -> dict:
    """D5, D6, A3 for data up to end_year. Each output is a long (ID, Year) annual table."""
    end = f"{end_year}-12-31"
    Wc = {v: m.loc[:end] for v, m in P["W_clean"].items()}
    rec = P["record"].loc[:end]
    qc_raw = annual_from_daily(Wc, rec)
    qc = apply_validity(qc_raw)
    steps = estimate_steps(qc, P["changes"], P["weights"], end_year)
    Wh, offs = homogenise_daily(Wc, steps)
    homog = apply_validity(annual_from_daily(Wh, rec))
    Wf, wf = fill_daily(Wc, rec, P["weights"], end_year, fill_method)
    # Fill estimates sit on today's station level (normals from the last 30 years). Put them on the
    # recorded station's level for the plain filled series, and add the offsets to observed days
    # only for the adjusted one.
    W_fill = {v: (Wf[v] - offs[v].where(wf[v], 0.0)) if v in offs else Wf[v] for v in VARS}
    W_fill_h = {v: (Wf[v] + offs[v].where(~wf[v], 0.0)) if v in offs else Wf[v] for v in VARS}
    filled = annual_filled(W_fill, wf, rec, qc_raw)
    filled_h = annual_filled(W_fill_h, wf, rec, qc_raw)
    return dict(end_year=end_year, qc=qc, steps=steps, homog=homog, filled=filled, filled_homog=filled_h,
                daily=dict(clean=Wc, filled=W_fill, was_filled=wf, offsets=offs))


def _wide(long: pd.DataFrame, col: str, end_year: int, first: int = 1851) -> pd.DataFrame:
    t = long.pivot(index="ID", columns="Year", values=col)
    return t.reindex(columns=range(first, end_year + 1))


VARIANTS = ["qc", "qc+fill", "qc+homog", "qc+fill+homog", "qc+nbfill", "qc+fill+homog+nbfill"]


def variants(B: dict) -> dict[str, dict[str, pd.DataFrame]]:
    """Every input variant as {variant: {metric: city x year table}}, all built from data <= end_year.
    B must carry the neighbour weights as B["weights"]."""
    E = B["end_year"]
    src = {
        "qc": {m: (B["qc"], c) for m, c in METRIC_COL.items()},
        "qc+fill": {"annual_avg_temp": (B["filled"], "Avg Temp"), "total_precip_in": (B["filled"], "ToT Prcp"),
                    "hottest_day_temp": (B["qc"], "Max Tmax"), "cdd": (B["filled"], "cdd")},
        "qc+homog": {"annual_avg_temp": (B["homog"], "Avg Temp"), "total_precip_in": (B["qc"], "ToT Prcp"),
                     "hottest_day_temp": (B["homog"], "Max Tmax"), "cdd": (B["homog"], "cdd")},
        "qc+fill+homog": {"annual_avg_temp": (B["filled_homog"], "Avg Temp"),
                          "total_precip_in": (B["filled"], "ToT Prcp"),
                          "hottest_day_temp": (B["homog"], "Max Tmax"), "cdd": (B["filled_homog"], "cdd")},
    }
    V = {name: {m: _wide(t, c, E) for m, (t, c) in spec.items()} for name, spec in src.items()}
    for base, name in [("qc", "qc+nbfill"), ("qc+fill+homog", "qc+fill+homog+nbfill")]:
        V[name] = {m: neighbour_infill(V[base][m], B["weights"], E)[0] for m in METRIC_COL}
    return V


def backtest(P: dict, origins: tuple[int, ...] = (1986, 1994, 2002, 2010), test_length: int = 8,
             verbose: bool = True) -> pd.DataFrame:
    """Rolling-origin backtest of every input variant x forecaster, scored like the leaderboard.

    For each origin O, inputs are rebuilt from data <= O only (cleaning decisions excepted: the
    outlier rules were set once on the whole record). Forecasts for O+1..O+8 are scored against the
    *official* values (data/annual_valid.csv), on the city-years the challenge would score, with
    skill relative to the official 30-year average. So a variant only wins by predicting the
    official numbers better, not by changing them.
    """
    import combo_model as cm
    from common import load_valid_annual, mae, thirty_year_mean, wide
    official = load_valid_annual()
    rows = []
    for origin in origins:
        if verbose:
            print(f"origin {origin} ...")
        years = list(range(origin + 1, origin + 1 + test_length))
        assert years[-1] <= CUTOFF_YEAR
        B = build(P, origin)
        B["weights"] = P["weights"]
        V = variants(B)
        V["official"] = {m: wide(official, m).loc[:, :origin] for m in METRIC_COL}
        for m in METRIC_COL:
            tab = wide(official, m)
            hist_off = tab.loc[:, :origin]
            actual = tab[years].where(thirty_year_mean(hist_off, origin).notna(), axis=0)
            base = cm.average_forecast(hist_off, origin, years).reindex(index=actual.index, columns=years)
            ref = mae(actual.stack(future_stack=True), base.stack(future_stack=True))
            for vname, tabs in V.items():
                hist = tabs[m].reindex(index=tab.index, columns=range(tab.columns.min(), origin + 1))
                for model in ["average", "last10", "trend", "blend"]:
                    pred = cm.FORECASTERS[model](hist, origin, years).reindex(index=actual.index, columns=years)
                    err = mae(actual.stack(future_stack=True), pred.stack(future_stack=True))
                    rows.append({"origin": origin, "variant": vname, "model": model, "metric": m,
                                 "mae": err, "skill": 1 - err / ref})
    return pd.DataFrame(rows)


def choose_inputs(bt: pd.DataFrame, min_gain: float = MIN_GAIN, min_wins: int = MIN_WINS) -> pd.DataFrame:
    """Per metric, the input variant to use, judged with the team's current forecaster for that metric
    (code/combo_model.py METHOD). A variant replaces plain `qc` only if its mean skill over the origins
    is at least `min_gain` higher (0.2 leaderboard points) and it beats `qc` in at least `min_wins` of
    the origins; among those, the highest mean skill wins."""
    import combo_model as cm
    combo = bt[(bt["model"] == bt["metric"].map(cm.METHOD)) & (bt["variant"] != "official")]
    per = combo.pivot_table(index=["metric", "variant"], columns="origin", values="skill")
    rows = []
    for m in METRIC_COL:
        t = per.loc[m]
        base = t.loc["qc"]
        gain = t.mean(axis=1) - base.mean()
        wins = (t.sub(base, axis=1) > 1e-9).sum(axis=1)
        ok = (gain >= min_gain) & (wins >= min_wins)
        pick = gain[ok].idxmax() if ok.any() else "qc"
        rows.append({"metric": m, "chosen": pick, "mean skill qc": base.mean(),
                     "mean skill chosen": t.loc[pick].mean(), "gain": gain[pick], "origins won": int(wins[pick])})
    return pd.DataFrame(rows).set_index("metric")


# =============================================================================================
# Outputs
# =============================================================================================
def snake(c: str) -> str:
    return c.lower().replace(" ", "_").replace("-", "").replace(".", "")


# Variant columns written per metric (only the ones that change something: precipitation has no
# station adjustment, the hottest day is never filled from days).
VARIANT_COLUMNS = {
    "annual_avg_temp": {"qc+fill": "filled", "qc+homog": "homog", "qc+fill+homog+nbfill": "full"},
    "cdd": {"qc+fill": "filled", "qc+homog": "homog", "qc+fill+homog+nbfill": "full"},
    "total_precip_in": {"qc+fill": "filled", "qc+fill+homog+nbfill": "full"},
    "hottest_day_temp": {"qc+homog": "homog", "qc+fill+homog+nbfill": "full"},
}


def assemble_annual(P: dict, B: dict, V: dict) -> pd.DataFrame:
    """The processed city-year table (see the data dictionary in the notebook)."""
    E = B["end_year"]
    qc = B["qc"].copy()
    raw = apply_validity(P["annual_raw"])
    keep_raw = ["ID", "Year", "Warm Days", "Warm Nights", "Cold Days", "Cold Nights", "Wet Days",
                "Station Segment", "Station Changes"]
    out = qc.merge(raw[keep_raw], on=["ID", "Year"], how="left")
    out = out.rename(columns={METRIC_COL[m]: m for m in METRIC_COL})
    out = out.rename(columns=snake).rename(columns={"id": "city_id", "max_1day_p": "max_1day_p"})
    # how many readings the cleaning deleted
    lg = P["log"].assign(year=lambda x: x["date"].dt.year)
    n = lg.groupby(["city_id", "year", "var"]).size().unstack("var", fill_value=0)
    n = n.reindex(columns=VARS, fill_value=0).add_prefix("n_deleted_")
    out = out.join(n, on=["city_id", "year"])
    out[n.columns] = out[n.columns].fillna(0).astype(int)
    # input variants
    def long(tab, name):
        s = tab.loc[:, :E].stack()
        s.index.names = ["city_id", "year"]
        return s.rename(name)
    extra = []
    for m in METRIC_COL:
        for vname, suffix in VARIANT_COLUMNS[m].items():
            extra.append(long(V[vname][m], f"{m}_{suffix}"))
        recipe = INPUT_RECIPE[m]
        extra.append(long(V[recipe][m], f"{m}_input"))
        fill_tab = V["qc+fill"][m] if "fill" in recipe else V["qc"][m]
        src = pd.DataFrame(np.select([V["qc"][m].notna(), fill_tab.notna(), V[recipe][m].notna()],
                                     ["measured", "filled days", "neighbour infill"], ""),
                           index=V["qc"][m].index, columns=V["qc"][m].columns)
        extra.append(long(src.where(src != ""), f"{m}_input_source"))
    out = out.join(pd.concat(extra, axis=1), on=["city_id", "year"])
    nf = B["filled"][["ID", "Year", "n_filled_tmax", "n_filled_tmin", "n_filled_prcp"]].rename(
        columns={"ID": "city_id", "Year": "year"})
    out = out.merge(nf, on=["city_id", "year"], how="left")
    cities = P["cities"]
    out.insert(1, "name", cities["name"].reindex(out["city_id"]).values)
    out.insert(2, "lat", cities["lat"].reindex(out["city_id"]).values)
    out.insert(3, "lon", cities["lon"].reindex(out["city_id"]).values)
    assert_no_post_cutoff(out, "year", "annual_clean")
    return out.sort_values(["city_id", "year"]).reset_index(drop=True)


def assemble_daily(B: dict) -> pd.DataFrame:
    dl = B["daily"]
    parts = {"tmax": dl["clean"]["tmax"], "tmin": dl["clean"]["tmin"], "prcp": dl["clean"]["prcp"],
             "tmax_filled": dl["filled"]["tmax"], "tmin_filled": dl["filled"]["tmin"],
             "prcp_filled": dl["filled"]["prcp"],
             "tmax_offset": dl["offsets"]["tmax"], "tmin_offset": dl["offsets"]["tmin"]}
    long = pd.concat({k: v.astype("float32").stack(future_stack=True) for k, v in parts.items()}, axis=1)
    long.index.names = ["date", "city_id"]
    long = long.reset_index()
    rec = dl["clean"]["tmax"].notna() | dl["filled"]["tmax"].notna() | dl["clean"]["prcp"].notna()
    long = long[rec.stack(future_stack=True).to_numpy()]
    assert_no_post_cutoff(long.assign(year=long["date"].dt.year), "year", "daily_clean")
    return long[["city_id", "date", *parts]].sort_values(["city_id", "date"]).reset_index(drop=True)


def run(write: bool = True, verbose: bool = True) -> dict:
    P = prepare(verbose=verbose)
    B = build(P, CUTOFF_YEAR)
    B["weights"] = P["weights"]
    V = variants(B)
    annual = assemble_annual(P, B, V)
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        annual.to_parquet(OUT_DIR / "annual_clean.parquet", index=False)
        assemble_daily(B).to_parquet(OUT_DIR / "daily_clean.parquet", index=False)
        P["log"].to_csv(OUT_DIR / "qc_log.csv", index=False)
        B["steps"].to_csv(OUT_DIR / "homog_steps.csv", index=False)
        if verbose:
            print(f"wrote {OUT_DIR.relative_to(ROOT)}: annual_clean.parquet {annual.shape}, daily_clean.parquet, "
                  "qc_log.csv, homog_steps.csv")
    return dict(P=P, B=B, V=V, annual=annual)


if __name__ == "__main__":
    run()
