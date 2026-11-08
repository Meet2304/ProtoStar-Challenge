"""Cleaning of the four training subsets (daily, annual_indices, city_info, coverage_and_stations).

Run through `scripts/02_clean_data.py`. The plan this code implements, with the evidence behind every
step and the reviews it passed, is `notebooks/Meet/plan/cleaning-plan.md`; the step-by-step write-up is
in `documentation/`. Step ids (C*, V*, D*, A*, H*, E*) match the plan.

Three kinds of columns (plan §0.1):
    official labels   the published values masked with the official 10-day rule, equal to
                      data/annual_valid.csv. Never changed.
    QC values         label + (f(cleaned days) - f(raw days)): the label plus exactly the effect of the
                      repairs. Inputs only.
    model inputs      filled / station-adjusted / neighbour-infilled variants; the default per metric
                      (`<metric>_input`) is chosen by the backtest (E), where the labels themselves are a
                      candidate.

Data rule: only the challenge training data (weather up to 2018-12-31) is read; `assert_no_post_cutoff`
guards every weather table, and every backtest fold is built inside `fold_cutoff(origin)`, which makes
the same guard reject any row after the origin.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from protostar_challenge.config import (CUTOFF_YEAR, DATASET_REPO, DATASET_REVISION, PROCESSED_DIR,
                                        RAW_DIR, ROOT)

sys.path.insert(0, str(ROOT / "code"))
from common import MAX_MISSING_DAYS, METRICS, assert_no_post_cutoff  # noqa: E402

# =============================================================================================
# Constants (plan §0.3). Each has one meaning.
# =============================================================================================
TMAX_LIMIT, TMIN_LIMIT, PRCP_LIMIT = 134.0, -70.0, 43.0      # US records (°F, °F, in/day)
STUCK_MIN_DAYS, STUCK_NEIGHBOUR_SD = 7, 5.0
PRCP_RUN_MIN_DAYS, PRCP_RUN_MIN_IN = 3, 0.10
Z_CANDIDATE, Z_SUPPORT, SPIKE_Z, SPIKE_LOOK_DAYS = 5.0, 2.5, 3.0, 3
RESID_Z_DELETE, RESID_F_DELETE, MIN_NB_WITH_DATA = 5.0, 25.0, 2
QC_NEIGHBOURS = 3
NB_K, NB_RADIUS_MI = 5, 300.0
FRAGMENT_GAP_YEARS, FRAGMENT_MAX_YEARS, BREAK_FLAG_YEARS = 20, 10, 10
FILL_MAX_MISSING, FILL_LOCAL_WINDOW, PRCP_RATIO_FLOOR = 30, 15, 0.01
FILL_BLOCKS, FILL_MIN_NORMAL_YEARS, FILL_BAKEOFF_END = 300, 10, 1985
CLUSTER_MIN_YEARS = 5
HOMOG_WINDOW, HOMOG_MIN_VALID, HOMOG_MIN_NB = 10, 5, 2
HOMOG_MIN_T, HOMOG_MIN_STEP, AR1_MAX, PLACEBO_Q, PLACEBO_MIN = 2.0, 0.25, 0.8, 0.95, 200
RETURN_MAX_SUM = 0.5
INFILL_OFFSET_YEARS, INFILL_MIN_NEIGHBOURS = 15, 2
MIN_GAIN, BOOT_REPS, BOOT_ALPHA, BOOT_CELL_DEG = 0.002, 2000, 0.05, 5.0
ORIGINS, TEST_LENGTH = (1986, 1994, 2002, 2010), 8
NESTED = {2002: (1986, 1994), 2010: (1986, 1994, 2002)}

VARS = ["tmax", "tmin", "prcp"]
TEMP_VARS = ["tmax", "tmin"]
METRIC_COL = {m: col for m, (col, _) in METRICS.items()}
MISS_RULE = {  # annual_indices column -> missing-day columns of the variables it uses
    **{c: ["Miss Tmax"] for c in ["Avg Tmax", "Min Tmax", "Warm Days", "Cold Days"]},
    **{c: ["Miss Tmin"] for c in ["Avg Tmin", "Max Tmin", "Min Tmin", "Warm Nights", "Cold Nights"]},
    **{c: ["Miss Prcp"] for c in ["Max 1-day P", "Max 5-day P", "Wet Days"]},
    **{col: miss for col, miss in METRICS.values()},
}
REBUILT = ["Avg Tmax", "Avg Tmin", "Avg Temp", "ToT Prcp", "Max Tmax", "Min Tmax", "Max Tmin", "Min Tmin",
           "cdd", "Max 1-day P", "Max 5-day P"]
PCT_COLS = {"Warm Days": "Miss Tmax", "Cold Days": "Miss Tmax", "Warm Nights": "Miss Tmin",
            "Cold Nights": "Miss Tmin"}
FLAG = {"observed": 0, "missing": 1, "deleted": 2, "swapped": 3, "transition": 4}
CANDIDATES = {
    "annual_avg_temp": ["official", "qc", "qc+homog", "qc+fill", "qc+fill+homog", "qc+nbfill", "qc+fill+homog+nbfill"],
    "cdd": ["official", "qc", "qc+homog", "qc+fill", "qc+fill+homog", "qc+nbfill", "qc+fill+homog+nbfill"],
    "total_precip_in": ["official", "qc", "qc+fill", "qc+nbfill", "qc+fill+nbfill"],
    "hottest_day_temp": ["official", "qc", "qc+homog", "qc+nbfill", "qc+homog+nbfill"],
}
PREFERENCE = ["official", "qc", "qc+homog", "qc+fill", "qc+fill+homog", "qc+nbfill", "qc+homog+nbfill",
              "qc+fill+nbfill", "qc+fill+homog+nbfill"]
FORECASTER_ORDER = ["average", "last10", "trend", "blend"]

DECISIONS_PATH = ROOT / "data" / "qc" / "daily_outlier_decisions.csv"
DECISIONS_SEED = ROOT / "notebooks" / "Meet" / "qc" / "daily_outlier_decisions.csv"
QC_OUT = PROCESSED_DIR / "qc"


def constants() -> dict:
    """Every constant above, for the manifest and the reuse hash."""
    g = globals()
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()
            if k.isupper() and isinstance(v, (int, float, tuple, str)) and not k.startswith("_")
            and k not in {"CUTOFF_YEAR", "DATASET_REPO", "DATASET_REVISION"}}


@contextlib.contextmanager
def fold_cutoff(origin: int):
    """Inside this block `assert_no_post_cutoff` rejects any year after `origin` (code/common.py)."""
    old = os.environ.get("PROTOSTAR_FOLD_CUTOFF")
    os.environ["PROTOSTAR_FOLD_CUTOFF"] = str(origin)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("PROTOSTAR_FOLD_CUTOFF", None)
        else:
            os.environ["PROTOSTAR_FOLD_CUTOFF"] = old


def say(*a):
    print(*a, flush=True)


# =============================================================================================
# Loading
# =============================================================================================
def read_subset(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    local = RAW_DIR / name
    if local.exists() and any(local.glob("*.parquet")):
        return pd.read_parquet(local, columns=columns)
    return pd.read_parquet(f"hf://datasets/{DATASET_REPO}@{DATASET_REVISION}/{name}", columns=columns)


def load_raw() -> dict:
    """All four subsets, guarded, plus the daily data as Date x city matrices."""
    d = read_subset("daily", ["ID", "Date", *VARS])
    d["Date"] = pd.to_datetime(d["Date"])
    d["ID"] = d["ID"].astype(str)
    assert_no_post_cutoff(d.assign(Year=d["Date"].dt.year), "Year", "daily")
    span = d.groupby("ID")["Date"].agg(["min", "max", "size"])
    n_rows = len(d)
    dup = int(d.duplicated(["ID", "Date"]).sum())
    W = {v: d.pivot(index="Date", columns="ID", values=v).sort_index(axis=1) for v in VARS}
    idx, cols = W["tmax"].index, W["tmax"].columns
    full = pd.date_range(idx.min(), idx.max())
    W = {v: m.reindex(full) for v, m in W.items()}
    span = span.reindex(cols)
    record = pd.DataFrame((full.values[:, None] >= span["min"].values) & (full.values[:, None] <= span["max"].values),
                          index=full, columns=cols)
    empty_days = int(d[VARS].isna().all(axis=1).sum())
    del d
    annual = read_subset("annual_indices")
    cov = read_subset("coverage_and_stations")
    ci = read_subset("city_info")
    for t, name in [(annual, "annual_indices"), (cov, "coverage_and_stations")]:
        t["ID"] = t["ID"].astype(str)
        assert_no_post_cutoff(t, "Year", name)
    ci["ID"] = ci["ID"].astype(str)
    for c in ["Stn.stDate", "Stn.edDate"]:
        ci[c] = pd.to_datetime(ci[c])
    assert_no_post_cutoff(ci.assign(Year=ci["Stn.edDate"].dt.year), "Year", "city_info")
    cities = (ci[["ID", "Name", "Lat", "Lon"]].drop_duplicates("ID")
              .rename(columns={"ID": "city_id", "Name": "name", "Lat": "lat", "Lon": "lon"})
              .set_index("city_id").sort_index())
    structure = {"rows": n_rows, "cities": int(len(cols)), "duplicate keys": dup,
                 "calendar gaps": int(((span["max"] - span["min"]).dt.days + 1 != span["size"]).sum()),
                 "empty days": empty_days, "first date": str(full.min().date()), "last date": str(full.max().date())}
    if dup or structure["calendar gaps"]:
        raise ValueError(f"D1: daily structure broken: {structure}")
    return dict(W=W, record=record, annual=annual, coverage=cov, city_info=ci, cities=cities.loc[cols],
                structure=structure)


def file_hashes(paths: list[Path]) -> dict:
    out = {}
    for p in paths:
        if p.is_dir():
            for q in sorted(p.glob("*.parquet")):
                out[str(q.relative_to(ROOT).as_posix())] = hashlib.sha256(q.read_bytes()).hexdigest()
        elif p.exists():
            out[str(p.relative_to(ROOT).as_posix())] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# =============================================================================================
# Neighbours
# =============================================================================================
def distance_matrix(cities: pd.DataFrame) -> pd.DataFrame:
    lat, lon = np.radians(cities["lat"].values), np.radians(cities["lon"].values)
    cosang = np.sin(lat[:, None]) * np.sin(lat) + np.cos(lat[:, None]) * np.cos(lat) * np.cos(lon[:, None] - lon)
    return pd.DataFrame(np.arccos(np.clip(cosang, -1, 1)) * 3958.8, index=cities.index, columns=cities.index)


def nearest(dist: pd.DataFrame, k: int) -> dict[str, list[str]]:
    return {c: list(dist.loc[c].drop(c).nsmallest(k).index) for c in dist.index}


def idw_weights(dist: pd.DataFrame, k: int = NB_K, radius: float = NB_RADIUS_MI) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=dist.index, columns=dist.columns)
    for c in dist.index:
        nb = dist.loc[c].drop(c).nsmallest(k)
        nb = nb[nb <= radius]
        w.loc[c, nb.index] = 1.0 / nb.values ** 2
    return w


def neighbour_mean(values: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Rows = time, columns = cities: each city's weighted neighbour mean, skipping NaN neighbours."""
    cols = values.columns
    Wt = weights.loc[cols, cols].values.T
    have = values.notna().values.astype(float)
    num = np.nan_to_num(values.values) @ Wt
    den = have @ Wt
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame(np.where(den > 0, num / den, np.nan), index=values.index, columns=cols)


# =============================================================================================
# C. city_info: station history
# =============================================================================================
_SYN = [(r"\bINTERNATIONAL\b", "INTL"), (r"\bMUNICIPAL\b", "MUNI"), (r"\bREGIONAL\b", "RGNL"),
        (r"\b(AIRPORT|ARPT|AP)\b", "AP"), (r"\bW B\b", "WB")]


def normalise_name(s: str) -> str:
    s = re.sub(r"[^A-Z0-9 ]", " ", str(s).upper())
    s = re.sub(r"\s+", " ", s).strip()
    for a, b in _SYN:
        s = re.sub(a, b, s)
    return re.sub(r"\s+", " ", s).strip()


def check_city_info(ci: pd.DataFrame, annual: pd.DataFrame) -> dict:
    """C1: hard invariants of the station table."""
    errors = []
    if len(ci) != 461:
        errors.append(f"{len(ci)} rows (expected 461)")
    if (ci["Stn.stDate"] > ci["Stn.edDate"]).any():
        errors.append("a station starts after it ends")
    last = ci.groupby("ID")["Stn.edDate"].max()
    if (last != pd.Timestamp(f"{CUTOFF_YEAR}-12-31")).any():
        errors.append("a city's last station does not end on 2018-12-31")
    if (ci.groupby("ID")[["Lat", "Lon", "Name"]].nunique() > 1).any().any():
        errors.append("lat/lon/name vary within a city")
    a = annual.groupby("ID")[["Lat", "Lon"]].first()
    c = ci.groupby("ID")[["Lat", "Lon"]].first().reindex(a.index)
    if not np.allclose(a.values, c.values):
        errors.append("lat/lon differ from annual_indices")
    if ci["ID"].nunique() != 210 or ci["Name"].nunique() != 210:
        errors.append("not 210 cities")
    ash = ci[(ci["Name"] == "Asheville") & (ci["Stn.Name"] == "ASHEVILLE CITY")]["Stn.stDate"]
    wpb = ci[(ci["Name"] == "WestPalmBeach") & (ci["Stn.Name"] == "JUPITER-HYPOLUXO")]["Stn.edDate"]
    if list(ash) != [pd.Timestamp("1876-08-01")] or list(wpb) != [pd.Timestamp("1899-12-01")]:
        errors.append("the dataset card's two typo fixes are not present")
    if errors:
        raise ValueError("C1: " + "; ".join(errors))
    return {"rows": len(ci), "cities": int(ci["ID"].nunique()), "changes": int(len(ci) - ci["ID"].nunique())}


def build_stations(ci: pd.DataFrame, W: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """C2-C6: one row per station period with names normalised, keys, data coverage, gaps, overlaps."""
    anyd = W["tmax"].notna() | W["tmin"].notna() | W["prcp"].notna()
    cum = anyd.cumsum()
    st = ci.sort_values(["ID", "Stn.stDate"]).reset_index(drop=True).rename(
        columns={"ID": "city_id", "Name": "name", "Lat": "lat", "Lon": "lon", "Stn.Name": "stn_name",
                 "Stn.stDate": "start", "Stn.edDate": "end"})
    st["stn_name_norm"] = st["stn_name"].map(normalise_name)
    st["segment"] = st.groupby("city_id").cumcount() + 1
    st["station_key"] = 0
    for city, g in st.groupby("city_id"):
        keys = {}
        st.loc[g.index, "station_key"] = [keys.setdefault(n, len(keys) + 1) for n in g["stn_name_norm"]]
    st["days"] = (st["end"] - st["start"]).dt.days + 1
    idx = cum.index

    def count(city, a, b):
        col = cum[city]
        hi = col.iloc[idx.get_indexer([b])[0]] if b <= idx[-1] else col.iloc[-1]
        lo_pos = idx.get_indexer([a - pd.Timedelta(days=1)])[0]
        lo = col.iloc[lo_pos] if lo_pos >= 0 else 0
        return int(hi - lo)

    st["days_with_data"] = [count(r.city_id, r.start, r.end) for r in st.itertuples()]
    st["empty_period"] = st["days_with_data"] == 0
    prev_end = st.groupby("city_id")["end"].shift()
    gap = (st["start"] - prev_end).dt.days - 1
    st["gap_before_days"] = gap.clip(lower=0).fillna(0).astype(int)
    st["overlap_days"] = (-gap).clip(lower=0).fillna(0).astype(int)
    st["gap_days_with_data"] = [count(r.city_id, pe + pd.Timedelta(days=1), r.start - pd.Timedelta(days=1))
                                if g > 0 else 0 for r, pe, g in zip(st.itertuples(), prev_end, st["gap_before_days"])]
    prev_key = st.groupby("city_id")["station_key"].shift()
    st["same_name"] = st["station_key"] == prev_key
    seen = st.groupby("city_id")["station_key"].transform(lambda s: s.duplicated())
    st["returns_to_key"] = seen & ~st["same_name"]
    if st["gap_days_with_data"].sum():
        raise ValueError("C3: readings found inside a gap between stations")
    inside = pd.DataFrame(False, index=anyd.index, columns=anyd.columns)
    for r in st.itertuples():
        inside.loc[r.start:r.end, r.city_id] = True
    if (anyd & ~inside).values.sum():
        raise ValueError("C3: readings outside every station window")
    return st


def effective_changes(st: pd.DataFrame) -> pd.DataFrame:
    """C4/C5: station changes after removing empty periods and merging consecutive periods of one key.

    One row per change: city_id, date (first day of the new station), from_key, to_key, names, and the
    overlap years (C2) to keep out of step windows."""
    rows = []
    for city, g in st.groupby("city_id"):
        g = g[~g["empty_period"]].sort_values("start")
        prev = None
        for r in g.itertuples():
            if prev is not None and r.station_key != prev.station_key:
                ov = []
                if r.start <= prev.end:
                    ov = list(range(r.start.year, prev.end.year + 1))
                rows.append({"city_id": city, "date": r.start, "from_key": prev.station_key, "to_key": r.station_key,
                             "from_name": prev.stn_name, "to_name": r.stn_name, "overlap_years": ov})
            if prev is None or r.station_key != prev.station_key:
                prev = r
            else:   # same key continues: extend its end
                prev = prev._replace(end=max(prev.end, r.end))
    return pd.DataFrame(rows, columns=["city_id", "date", "from_key", "to_key", "from_name", "to_name",
                                       "overlap_years"])


def station_key_by_day(st: pd.DataFrame, index: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """Key of the (non-empty) station in use on each day; 0 where none."""
    K = np.zeros((len(index), len(columns)), dtype=np.int16)
    pos = {c: i for i, c in enumerate(columns)}
    for r in st[~st["empty_period"]].itertuples():
        a, b = index.searchsorted(r.start), index.searchsorted(r.end, side="right")
        K[a:b, pos[r.city_id]] = r.station_key
    return pd.DataFrame(K, index=index, columns=columns)


def make_clusters(changes: pd.DataFrame, end_year: int) -> pd.DataFrame:
    """C6/H2: consecutive changes < CLUSTER_MIN_YEARS apart form one cluster; P = station before, Q = after."""
    ch = changes[changes["date"].dt.year <= end_year].sort_values(["city_id", "date"])
    rows = []
    for city, g in ch.groupby("city_id"):
        cid, cur = 0, []
        for r in g.itertuples():
            if cur and (r.date - cur[-1].date).days >= CLUSTER_MIN_YEARS * 365.25:
                rows.append(_cluster_row(city, cid, cur))
                cid, cur = cid + 1, []
            cur.append(r)
        if cur:
            rows.append(_cluster_row(city, cid, cur))
    return pd.DataFrame(rows, columns=["city_id", "cluster_id", "f", "l", "P", "Q", "n_changes", "same_endpoints",
                                       "middle_keys", "overlap_years", "label"])


def _cluster_row(city, cid, cur):
    P, Q = cur[0].from_key, cur[-1].to_key
    middle = sorted({r.to_key for r in cur[:-1]} | {r.from_key for r in cur[1:]})
    ov = sorted({y for r in cur for y in r.overlap_years})
    return {"city_id": city, "cluster_id": cid, "f": cur[0].date, "l": cur[-1].date, "P": P, "Q": Q,
            "n_changes": len(cur), "same_endpoints": P == Q, "middle_keys": [k for k in middle if k not in (P, Q)],
            "overlap_years": ov, "label": f"{cur[0].from_name} -> {cur[-1].to_name}"}


# =============================================================================================
# V. coverage_and_stations
# =============================================================================================
def check_coverage(cov: pd.DataFrame, annual: pd.DataFrame, W: dict, record: pd.DataFrame, ci: pd.DataFrame) -> dict:
    """V1: the published coverage table is a pure function of daily and city_info."""
    errors = []
    if not cov[["ID", "Year"]].reset_index(drop=True).equals(annual[["ID", "Year"]].reset_index(drop=True)):
        errors.append("rows differ from annual_indices")
    both = (W["tmax"].notna() & W["tmin"].notna())
    db = both.groupby(both.index.year).sum().stack()
    db.index.names = ["Year", "ID"]
    c = cov.set_index(["ID", "Year"])
    calc = db.swaplevel().reindex(c.index).fillna(0)
    if (calc.values != c["Days Both"].values).any():
        errors.append("Days Both differs from daily")
    a = annual.set_index(["ID", "Year"]).reindex(c.index)
    if (a["cdd"].isna().values != (c["Days Both"] == 0).values).any():
        errors.append("cdd empty != Days Both 0")
    seg, chg = [], []
    for city, g in ci.sort_values("Stn.stDate").groupby("ID"):
        starts = g["Stn.stDate"].sort_values().to_numpy()
        yrs = c.loc[city].index.to_numpy()
        jan1 = pd.to_datetime([f"{y}-01-01" for y in yrs]).to_numpy()
        dec31 = pd.to_datetime([f"{y}-12-31" for y in yrs]).to_numpy()
        s = np.maximum(np.searchsorted(starts, jan1, side="right"), 1)
        later = starts[1:]
        n = ((later[None, :] >= jan1[:, None]) & (later[None, :] <= dec31[:, None])).sum(axis=1)
        seg.append(pd.Series(s, index=pd.MultiIndex.from_product([[city], yrs])))
        chg.append(pd.Series(n, index=pd.MultiIndex.from_product([[city], yrs])))
    seg, chg = pd.concat(seg).reindex(c.index), pd.concat(chg).reindex(c.index)
    if (seg.values != c["Station Segment"].values).any():
        errors.append(f"Station Segment differs ({int((seg.values != c['Station Segment'].values).sum())} rows)")
    if (chg.values != c["Station Changes"].values).any():
        errors.append(f"Station Changes differs ({int((chg.values != c['Station Changes'].values).sum())} rows)")
    if errors:
        raise ValueError("V1: " + "; ".join(errors))
    return {"rows": len(cov), "days_both_rebuilt": True, "segments_rebuilt": True}


# =============================================================================================
# Annual values from days (the rebuild f)
# =============================================================================================
def annual_from_daily(W: dict[str, pd.DataFrame], record: pd.DataFrame) -> pd.DataFrame:
    """REBUILT columns, missing counts and Days Both per city-year, from day matrices (long, ID x Year)."""
    tx, tn, pr = W["tmax"], W["tmin"], W["prcp"]
    rec = record.loc[tx.index]
    yr = tx.index.year
    days = rec.groupby(yr).sum()
    both = tx.notna() & tn.notna()
    tavg = ((tx + tn) / 2).where(both)
    out = {
        "days_in_record": days,
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
    return long.swaplevel().sort_index()


def valid_mask(a: pd.DataFrame, col: str, max_missing: int = MAX_MISSING_DAYS) -> pd.Series:
    return a[MISS_RULE[col]].max(axis=1) <= max_missing


# =============================================================================================
# Official labels and the audit (A1, A2)
# =============================================================================================
def official_labels(annual: pd.DataFrame) -> pd.DataFrame:
    """Published values masked as code/prepare_data.py does (sentinels removed by the same mask)."""
    a = annual.set_index(["ID", "Year"])
    out = pd.DataFrame(index=a.index)
    for m, (col, miss) in METRICS.items():
        bad = a[miss].max(axis=1) > MAX_MISSING_DAYS
        out[m] = a[col].mask(bad).round(3)
    return out


def check_labels(labels: pd.DataFrame, valid_csv: Path) -> int:
    v = pd.read_csv(valid_csv, dtype={"city_id": str}).set_index(["city_id", "year"])
    assert_no_post_cutoff(v.reset_index(), "year", str(valid_csv))
    L = labels.copy()
    L.index.names = ["city_id", "year"]
    v = v.reindex(L.index)
    for m in METRICS:
        a, b = L[m].to_numpy(), v[m].to_numpy()
        same = (np.isnan(a) & np.isnan(b)) | (a == b)
        if not same.all():
            raise ValueError(f"labels: {m} differs from {valid_csv.name} on {int((~same).sum())} rows")
    return len(L)


def audit_rebuild(annual: pd.DataFrame, f_raw: pd.DataFrame) -> dict:
    """§0.1 (a)-(d): our definitions reproduce the published values (on rows valid for each column)."""
    a = annual.set_index(["ID", "Year"]).replace([np.inf, -np.inf], np.nan)
    f = f_raw.reindex(a.index)
    res, errors = {}, []
    tv = (a["Miss Tmax"] <= MAX_MISSING_DAYS) & (a["Miss Tmin"] <= MAX_MISSING_DAYS)
    checks = {
        "a: Avg Tmax": ((f["Avg Tmax"] - a["Avg Tmax"])[a["Miss Tmax"] <= MAX_MISSING_DAYS], 0.0051),
        "a: Avg Tmin": ((f["Avg Tmin"] - a["Avg Tmin"])[a["Miss Tmin"] <= MAX_MISSING_DAYS], 0.0051),
        "b: published chain": ((a["Avg Temp"] - (a["Avg Tmax"] + a["Avg Tmin"]) / 2)[tv], 0.0051),
        "c: Avg Temp direct": ((f["Avg Temp"] - a["Avg Temp"])[tv], 0.0101),
    }
    for col in ["ToT Prcp", "Max Tmax", "Min Tmax", "Max Tmin", "Min Tmin", "cdd"]:
        checks[f"d: {col}"] = ((f[col] - a[col])[valid_mask(a, col)], 1e-6)
    for name, (diff, tol) in checks.items():
        d = diff.dropna().abs()
        res[name] = {"rows": int(len(d)), "max_abs_diff": float(d.max()) if len(d) else 0.0, "tolerance": tol}
        if (d > tol).any():
            errors.append(f"{name}: {int((d > tol).sum())} rows over {tol}")
    if errors:
        raise ValueError("audit: " + "; ".join(errors))
    return res


# =============================================================================================
# D2-D4. Daily quality control
# =============================================================================================
def _log(mask: pd.DataFrame, values: pd.DataFrame, var: str, rule: str, action: str, new=None) -> pd.DataFrame:
    s = values.where(mask).stack().dropna()
    out = pd.DataFrame({"city_id": s.index.get_level_values(1), "date": s.index.get_level_values(0),
                        "var": var, "old": s.values, "rule": rule, "action": action})
    out["new"] = np.nan if new is None else new.where(mask).stack().reindex(s.index).values
    return out


def repair_impossible(W: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    """D2: swap inverted days whose values are both inside the limits; delete limit violations."""
    tx, tn, pr = W["tmax"].copy(), W["tmin"].copy(), W["prcp"].copy()
    logs = []
    lim = {"tmax": (tx > TMAX_LIMIT) | (tx < TMIN_LIMIT), "tmin": (tn < TMIN_LIMIT) | (tn > TMAX_LIMIT),
           "prcp": (pr < 0) | (pr > PRCP_LIMIT)}
    for v, M in [("tmax", tx), ("tmin", tn), ("prcp", pr)]:
        logs.append(_log(lim[v], M, v, "outside physical limits", "delete"))
    inv = (tx < tn) & ~lim["tmax"] & ~lim["tmin"]
    logs.append(_log(inv, tx, "tmax", "tmax below tmin: swapped", "swap", new=tn))
    logs.append(_log(inv, tn, "tmin", "tmax below tmin: swapped", "swap", new=tx))
    tx2, tn2 = tx.where(~inv, tn), tn.where(~inv, tx)
    out = {"tmax": tx2.mask(lim["tmax"]), "tmin": tn2.mask(lim["tmin"]), "prcp": pr.mask(lim["prcp"])}
    return out, pd.concat(logs, ignore_index=True)


def run_lengths(M: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    """D3a: runs of >= STUCK_MIN_DAYS identical temperatures tested against the QC neighbours."""
    rows = []
    for var in TEMP_VARS:
        M = W[var]
        length, position = run_lengths(M)
        ends = (length >= STUCK_MIN_DAYS) & (position == length)
        for date, city in ends.stack()[lambda s: s].index:
            n = int(length.at[date, city])
            days = pd.date_range(end=date, periods=n)
            sds = [M.loc[days, nb].std() for nb in nbrs[city] if M.loc[days, nb].count() >= n / 2]
            nb_sd = float(np.median(sds)) if len(sds) >= 2 else np.nan
            act = "delete" if nb_sd >= STUCK_NEIGHBOUR_SD else "keep"
            reason = ("neighbours varied: sensor stuck" if act == "delete" else
                      "fewer than 2 neighbours with data" if len(sds) < 2 else "neighbours steady too: real weather")
            rows.append({"city_id": city, "var": var, "start": days[0], "end": date, "days": n,
                         "value": M.at[date, city], "n_neighbours": len(sds), "neighbour_sd": nb_sd,
                         "action": act, "reason": reason})
    return pd.DataFrame(rows, columns=["city_id", "var", "start", "end", "days", "value", "n_neighbours",
                                       "neighbour_sd", "action", "reason"])


def prcp_runs(W: dict[str, pd.DataFrame], nbrs: dict[str, list[str]]) -> pd.DataFrame:
    """D3b: runs of >= PRCP_RUN_MIN_DAYS equal non-zero precipitation days, tested against neighbours."""
    P = W["prcp"].where(W["prcp"] > 0)
    length, position = run_lengths(P)
    ends = (length >= PRCP_RUN_MIN_DAYS) & (position == length)
    s = ends.stack()[lambda x: x]
    rows = []
    allp = W["prcp"]
    for date, city in s.index:
        n = int(length.at[date, city])
        start = date - pd.Timedelta(days=n - 1)
        v = float(P.at[date, city])
        row = {"city_id": city, "start": start, "end": date, "days": n, "value": v, "total": v * n}
        if v < PRCP_RUN_MIN_IN:
            row.update(n_observed_neighbours=np.nan, wet_neighbours=np.nan, action="keep",
                       reason=f"small amount (< {PRCP_RUN_MIN_IN} in): normal in drizzle climates")
        else:
            run = allp.loc[start:date, nbrs[city]]
            win = allp.loc[start - pd.Timedelta(days=1):date + pd.Timedelta(days=1), nbrs[city]]
            n_obs = int(run.notna().all().sum())
            wet = int((win > 0).any().sum())
            dele = n_obs >= 2 and wet == 0
            row.update(n_observed_neighbours=n_obs, wet_neighbours=wet, action="delete" if dele else "keep",
                       reason=("repeat on days that were dry all around" if dele else
                               "a neighbour had rain" if wet else "fewer than 2 neighbours observed every day"))
        rows.append(row)
    return pd.DataFrame(rows)


def runs_log(stuck: pd.DataFrame, pruns: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in stuck[stuck["action"] == "delete"].itertuples():
        out.append(pd.DataFrame({"city_id": r.city_id, "date": pd.date_range(r.start, r.end)[1:], "var": r.var,
                                 "old": r.value, "new": np.nan, "action": "delete",
                                 "rule": f"stuck sensor ({r.days} days, neighbours sd {r.neighbour_sd:.1f} °F)"}))
    for r in pruns[pruns["action"] == "delete"].itertuples():
        out.append(pd.DataFrame({"city_id": r.city_id, "date": pd.date_range(r.start, r.end)[1:], "var": "prcp",
                                 "old": r.value, "new": np.nan, "action": "delete",
                                 "rule": f"repeated precipitation ({r.days} days of {r.value} in, neighbours dry)"}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["city_id", "date", "var", "old", "new",
                                                                             "action", "rule"])


def robust_z(M: pd.DataFrame):
    month = M.index.month
    med = M.groupby(month).transform("median")
    scale = (M - med).abs().groupby(month).transform("median") * 1.4826
    scale = scale.where(scale > 0)
    return (M - med) / scale, med, scale


def outlier_candidates(W: dict[str, pd.DataFrame], nbrs: dict[str, list[str]]) -> pd.DataFrame:
    """D4: every |z| >= Z_CANDIDATE temperature with its evidence (definitions in plan §3)."""
    rows = []
    for var in TEMP_VARS:
        M = W[var]
        Z, med, scale = robust_z(M)
        cand = Z.abs() >= Z_CANDIDATE
        pos_sup = pd.DataFrame(0.0, index=Z.index, columns=Z.columns)
        neg_sup = pd.DataFrame(0.0, index=Z.index, columns=Z.columns)
        nb_z = pd.DataFrame(np.nan, index=Z.index, columns=Z.columns)
        nb_n = pd.DataFrame(0.0, index=Z.index, columns=Z.columns)
        for c in Z.columns:
            if not cand[c].any():
                continue
            nbz = Z[nbrs[c]]
            nb_z[c] = nbz.mean(axis=1)
            nb_n[c] = nbz.notna().sum(axis=1)
            stack = pd.concat([nbz.shift(k) for k in (-1, 0, 1)], axis=1)
            pos_sup[c] = stack.clip(lower=0).max(axis=1).fillna(0)
            neg_sup[c] = (-stack).clip(lower=0).max(axis=1).fillna(0)
        prev_day = M.shift(1).ffill(limit=SPIKE_LOOK_DAYS - 1)
        next_day = M.shift(-1).bfill(limit=SPIKE_LOOK_DAYS - 1)
        jp, jn = (M - prev_day) / scale, (M - next_day) / scale
        spike = ((jp >= SPIKE_Z) & (jn >= SPIKE_Z)) | ((jp <= -SPIKE_Z) & (jn <= -SPIKE_Z))
        yr = M.index.year
        extreme = M.groupby(yr).transform("max" if var == "tmax" else "min")
        s = Z.where(cand).stack().dropna()
        get = lambda T: T.stack(future_stack=True).reindex(s.index).values  # noqa: E731
        z = s.values
        rows.append(pd.DataFrame({
            "city_id": s.index.get_level_values(1), "date": s.index.get_level_values(0), "var": var,
            "value": get(M), "z": z, "nb_support": np.where(z > 0, get(pos_sup), get(neg_sup)),
            "nb_z": get(nb_z), "n_nb": get(nb_n).astype(int), "spike": get(spike).astype(bool),
            "year_extreme": get(M == extreme).astype(bool), "median": get(med), "scale": get(scale),
            "prev_day": get(prev_day), "next_day": get(next_day)}))
    out = pd.concat(rows, ignore_index=True)
    out["resid_z"] = out["z"] - out["nb_z"].fillna(0)
    out["expected"] = out["median"] + out["nb_z"].fillna(0) * out["scale"]
    out["resid_f"] = out["value"] - out["expected"]
    out["class"] = np.where(out["nb_support"] >= Z_SUPPORT, "confirmed by neighbours", "isolated")
    rule = ((out["class"] == "isolated") & out["spike"] & (out["resid_z"].abs() >= RESID_Z_DELETE)
            & (out["resid_f"].abs() >= RESID_F_DELETE) & (out["n_nb"] >= MIN_NB_WITH_DATA))
    out["proposed_action"] = np.where(rule, "delete", "keep")
    out["reason"] = np.select(
        [out["class"] == "confirmed by neighbours", rule, out["n_nb"] < MIN_NB_WITH_DATA, ~out["spike"]],
        ["a nearby city was unusual the same way within a day",
         "one-day spike, no nearby city saw it, far from what neighbours imply",
         "neighbours had no data that day: no evidence against it",
         "lasts more than a day (weather, or a fault D3 would catch)"],
        f"one-day spike, but within {RESID_F_DELETE:g} °F or {RESID_Z_DELETE:g} z of what neighbours imply")
    return out


def precip_extremes(W: dict[str, pd.DataFrame], nbrs: dict[str, list[str]]) -> pd.DataFrame:
    """D4 (precipitation, report only): wet days above 3x the city's 99.9th percentile of wet days, with the
    largest neighbour rain within +-1 day. Kept: heavy convective rain is often local, the physical limit is
    enforced in D2, and deleting a real downpour would make a yearly total read low."""
    P = W["prcp"]
    q = P.where(P > 0).quantile(0.999)
    s = P.where(P > 3 * q).stack().dropna()
    rows = []
    for (date, city), val in s.items():
        win = P.loc[date - pd.Timedelta(days=1):date + pd.Timedelta(days=1), nbrs[city]]
        rows.append({"city_id": city, "date": date, "prcp": val, "city_p999": q[city],
                     "max_neighbour_prcp_pm1day": win.max().max(), "action": "keep"})
    return pd.DataFrame(rows, columns=["city_id", "date", "prcp", "city_p999", "max_neighbour_prcp_pm1day", "action"])


DECISION_COLS = ["city_id", "date", "var", "prev_day", "value", "next_day", "expected", "z", "nb_support", "nb_z",
                 "n_nb", "resid_z", "resid_f", "spike", "year_extreme", "proposed_action", "reason",
                 "final_action", "decided_on", "evidence_year", "note"]


def read_overrides(path: Path = DECISIONS_PATH) -> pd.DataFrame:
    """Human entries of the review file (rows with a final_action)."""
    src = path if path.exists() else DECISIONS_SEED
    if not src.exists():
        return pd.DataFrame(columns=["city_id", "date", "var", "final_action", "decided_on", "evidence_year", "note"])
    d = pd.read_csv(src, dtype={"city_id": str, "final_action": str, "note": str, "decided_on": str})
    for c in ["decided_on", "evidence_year", "note"]:
        if c not in d.columns:
            d[c] = np.nan
    d["final_action"] = d["final_action"].fillna("").str.strip().str.lower()
    d = d[d["final_action"].isin(["keep", "delete"])].copy()
    if len(d):
        ev = pd.to_numeric(d["evidence_year"], errors="coerce")
        dec = pd.to_datetime(d["decided_on"], errors="coerce").dt.year
        if ev.isna().any() and dec[ev.isna()].isna().any():
            raise ValueError("decisions file: a final_action needs decided_on or evidence_year")
        d["evidence_year"] = ev.fillna(dec).astype(int)
    return d[["city_id", "date", "var", "final_action", "decided_on", "evidence_year", "note"]]


def write_decisions(cand: pd.DataFrame, overrides: pd.DataFrame, path: Path = DECISIONS_PATH) -> pd.DataFrame:
    """The review file: every isolated candidate (production run), keeping human entries."""
    iso = cand[cand["class"] == "isolated"].copy()
    iso["date"] = pd.to_datetime(iso["date"]).dt.date.astype(str)
    iso = iso[DECISION_COLS[:17]].round(2)
    iso = iso.merge(overrides, on=["city_id", "date", "var"], how="left")
    for c in ["final_action", "decided_on", "note"]:
        iso[c] = iso[c].fillna("")
    path.parent.mkdir(parents=True, exist_ok=True)
    iso = iso[DECISION_COLS].sort_values(["date", "city_id", "var"])
    iso.to_csv(path, index=False)
    return iso


def outlier_log(cand: pd.DataFrame, overrides: pd.DataFrame, end_year: int) -> pd.DataFrame:
    """Automatic actions, replaced by human ones whose evidence_year <= end_year."""
    c = cand.copy()
    c["date_s"] = pd.to_datetime(c["date"]).dt.date.astype(str)
    ov = overrides[overrides["evidence_year"] <= end_year] if len(overrides) else overrides
    act = c["proposed_action"].copy()
    rule = np.where(act == "delete", "outlier (isolated one-day spike)", "")
    if len(ov):
        m = c.merge(ov, left_on=["city_id", "date_s", "var"], right_on=["city_id", "date", "var"], how="left")
        has = m["final_action"].notna().to_numpy()
        act = pd.Series(np.where(has, m["final_action"], act), index=c.index)
        rule = np.where(has & (act == "delete"), "outlier (reviewer)", rule)
    dl = c[act == "delete"]
    return pd.DataFrame({"city_id": dl["city_id"], "date": pd.to_datetime(dl["date"]), "var": dl["var"],
                         "old": dl["value"], "new": np.nan, "action": "delete", "rule": rule[act == "delete"]})


def apply_log(W: dict[str, pd.DataFrame], log: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {v: m.copy() for v, m in W.items()}
    for var, g in log[log["action"] == "delete"].groupby("var"):
        M = out[var]
        vals = M.to_numpy(copy=True)
        vals[M.index.get_indexer(pd.to_datetime(g["date"])), M.columns.get_indexer(g["city_id"])] = np.nan
        out[var] = pd.DataFrame(vals, index=M.index, columns=M.columns)
    return out


# =============================================================================================
# A3. Detached fragments (per variable group)
# =============================================================================================
def find_blocks(years: np.ndarray, gap: int) -> list[np.ndarray]:
    years = np.sort(years)
    if not len(years):
        return []
    cuts = np.where(np.diff(years) - 1 > gap)[0]
    return np.split(years, cuts + 1)


def fragments(W: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blocks per group (temperature: any tmax/tmin; precipitation: any prcp).

    Returns (dropped blocks, record breaks): a block is dropped if it has < FRAGMENT_MAX_YEARS years with
    data, is separated from the next block by > FRAGMENT_GAP_YEARS empty years, and is not the group's most
    recent block. A break is flagged on the first year after every gap >= BREAK_FLAG_YEARS empty years."""
    groups = {"temperature": W["tmax"].notna() | W["tmin"].notna(), "precipitation": W["prcp"].notna()}
    drops, breaks = [], []
    for grp, has in groups.items():
        occ = has.groupby(has.index.year).any()
        for city in occ.columns:
            yrs = occ.index[occ[city]].to_numpy()
            for blk in find_blocks(yrs, BREAK_FLAG_YEARS - 1)[1:]:
                breaks.append({"city_id": city, "group": grp, "year": int(blk[0])})
            blocks = find_blocks(yrs, FRAGMENT_GAP_YEARS)
            for i, blk in enumerate(blocks[:-1]):
                if len(blk) < FRAGMENT_MAX_YEARS:
                    drops.append({"city_id": city, "group": grp, "first": int(blk[0]), "last": int(blk[-1]),
                                  "years_with_data": len(blk), "next_block_starts": int(blocks[i + 1][0]),
                                  "empty_years_after": int(blocks[i + 1][0] - blk[-1] - 1)})
    return (pd.DataFrame(drops, columns=["city_id", "group", "first", "last", "years_with_data", "next_block_starts",
                                         "empty_years_after"]),
            pd.DataFrame(breaks, columns=["city_id", "group", "year"]))


def drop_fragments(W: dict[str, pd.DataFrame], drops: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    out = {v: m.copy() for v, m in W.items()}
    logs = []
    for r in drops.itertuples():
        vs = TEMP_VARS if r.group == "temperature" else ["prcp"]
        for v in vs:
            sl = out[v].loc[f"{r.first}-01-01":f"{r.last}-12-31", r.city_id]
            s = sl.dropna()
            logs.append(pd.DataFrame({"city_id": r.city_id, "date": s.index, "var": v, "old": s.values, "new": np.nan,
                                      "action": "delete", "rule": f"detached fragment {r.first}-{r.last}"}))
            out[v].loc[f"{r.first}-01-01":f"{r.last}-12-31", r.city_id] = np.nan
    log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=["city_id", "date", "var", "old",
                                                                                "new", "action", "rule"])
    return out, log


def fragment_years(drops: pd.DataFrame) -> dict[str, set]:
    """(city, year) pairs dropped per group."""
    out = {"temperature": set(), "precipitation": set()}
    for r in drops.itertuples():
        out[r.group] |= {(r.city_id, y) for y in range(r.first, r.last + 1)}
    return out


# =============================================================================================
# QC driver for one end year (D2-D4 + A3)
# =============================================================================================
def quality_control(R: dict, end_year: int, nbrs: dict, overrides: pd.DataFrame, verbose: bool = True) -> dict:
    end = f"{end_year}-12-31"
    W = {v: m.loc[:end] for v, m in R["W"].items()}
    W2, log_imp = repair_impossible(W)
    if verbose:
        say(f"  [{end_year}] D3 runs ...")
    stuck = stuck_runs(W2, nbrs)
    pruns = prcp_runs(W2, nbrs)
    log_runs = runs_log(stuck, pruns)
    W3 = apply_log(W2, log_runs)
    if verbose:
        say(f"  [{end_year}] D4 outliers ...")
    cand = outlier_candidates(W3, nbrs)
    log_out = outlier_log(cand, overrides, end_year)
    W4 = apply_log(W3, log_out)
    drops, breaks = fragments(W4)
    W5, log_frag = drop_fragments(W4, drops)
    log = pd.concat([log_imp, log_runs, log_out, log_frag], ignore_index=True)
    log["date"] = pd.to_datetime(log["date"])
    return dict(W=W5, log=log, stuck=stuck, prcp_runs=pruns, candidates=cand, fragments=drops, breaks=breaks)


# =============================================================================================
# QC values and the delta form (§0.1)
# =============================================================================================
def published_masked(annual: pd.DataFrame) -> pd.DataFrame:
    """Every rebuilt and percentage column of annual_indices, ±inf removed, masked by its own 10-day rule."""
    a = annual.set_index(["ID", "Year"]).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame(index=a.index)
    for col in REBUILT + ["Wet Days"] + list(PCT_COLS):
        out[col] = a[col].where(valid_mask(a, col))
    return out


def delta_form(label: pd.Series, f_var: pd.Series, f_raw: pd.Series, ok: pd.Series) -> pd.Series:
    """label + (f_var - f_raw) where a label exists, f_var where it does not; NaN where `ok` is False."""
    val = np.where(label.notna() & f_raw.notna(), label + (f_var - f_raw), f_var)
    return pd.Series(val, index=label.index).where(ok)


# =============================================================================================
# D6. Day filling (inputs only)
# =============================================================================================
def doy365(index: pd.DatetimeIndex) -> np.ndarray:
    """1..365; 29 Feb shares 28 Feb's slot and later days keep their 365-day position."""
    doy = index.day_of_year.to_numpy().copy()
    after = (index.month > 2) | ((index.month == 2) & (index.day == 29))
    doy[np.asarray(index.is_leap_year & after)] -= 1
    return doy


def normals(M: pd.DataFrame, end_year: int, years: int = 30, smooth: int = 31) -> pd.DataFrame:
    """Mean of the `years` years ending at end_year per 365-day slot, circularly smoothed; a slot needs
    >= FILL_MIN_NORMAL_YEARS observations."""
    W = M.loc[f"{end_year - years + 1}-01-01":f"{end_year}-12-31"]
    doy = doy365(W.index)
    g = W.groupby(doy)
    mean = g.mean().reindex(range(1, 366))
    cnt = g.count().reindex(range(1, 366)).fillna(0)
    mean = mean.where(cnt >= FILL_MIN_NORMAL_YEARS)
    ext = pd.concat([mean.iloc[-smooth:], mean, mean.iloc[:smooth]])
    return ext.rolling(smooth, center=True, min_periods=smooth // 2 + 1).mean().iloc[smooth:-smooth]


def expand(clim: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(clim.loc[doy365(index)].to_numpy(), index=index, columns=clim.columns)


FILL_METHODS = ["normal", "normal + local anomaly", "normal + neighbour anomaly"]


def fill_daily(W: dict, record: pd.DataFrame, weights: pd.DataFrame, end_year: int, method: str) -> tuple[dict, dict]:
    """Every missing day inside a record estimated (inputs only). Observed values are never changed."""
    filled, was = {}, {}
    rec = record.loc[W["tmax"].index]
    for var in VARS:
        M = W[var]
        C = expand(normals(M, end_year), M.index)
        ratio = var == "prcp"
        A = M / C.clip(lower=PRCP_RATIO_FLOOR) if ratio else M - C
        adj = None
        if method != "normal":
            adj = A.rolling(2 * FILL_LOCAL_WINDOW + 1, center=True, min_periods=5).mean()
            if method == "normal + neighbour anomaly":
                adj = neighbour_mean(A, weights).fillna(adj)
        est = C if adj is None else (C.clip(lower=0) * adj.fillna(1.0) if ratio else C + adj.fillna(0.0))
        need = M.isna() & rec
        filled[var] = M.where(~need, est)
        was[var] = need & est.notna()
    # a filled member is clipped to the observed one; if both were filled, both move to their midpoint
    tx, tn = filled["tmax"], filled["tmin"]
    bad = tx < tn
    fx, fn = was["tmax"], was["tmin"]
    mid = (tx + tn) / 2
    new_x = tx.where(~(bad & fx & ~fn), tn).where(~(bad & fx & fn), mid)
    new_n = tn.where(~(bad & fn & ~fx), tx).where(~(bad & fx & fn), mid)
    filled["tmax"], filled["tmin"] = new_x, new_n
    for v in TEMP_VARS:
        M = W[v]
        if not ((filled[v] == M) | M.isna()).all().all():
            raise AssertionError("fill changed an observed value")
    return filled, was


def fill_bakeoff(W: dict, weights: pd.DataFrame, end_year: int = FILL_BAKEOFF_END) -> tuple[str, str, pd.DataFrame]:
    """D6 bake-off on data <= end_year (plan §3 D6). Returns (temperature method, precipitation method, table)."""
    lo = end_year - 29
    Wt = {v: m.loc[f"{lo}-01-01":f"{end_year}-12-31"] for v, m in W.items()}
    idx = Wt["tmax"].index
    starts = np.arange(0, len(idx) - 29, 30)       # non-overlapping 30-day blocks from 1 Jan of the first year
    obs = {v: Wt[v].notna().to_numpy() for v in VARS}
    blocks = []
    for j, c in enumerate(Wt["tmax"].columns):
        for s in starts:
            if all(obs[v][s:s + 30, j].sum() >= 25 for v in VARS):
                blocks.append((c, s))
    blocks.sort(key=lambda b: (b[0], b[1]))
    k = max(1, len(blocks) // FILL_BLOCKS)
    chosen = blocks[::k][:FILL_BLOCKS]
    masked = {v: m.copy() for v, m in Wt.items()}
    for c, s in chosen:
        j = masked["tmax"].columns.get_loc(c)
        for v in VARS:
            masked[v].iloc[s:s + 30, j] = np.nan
    rec = pd.DataFrame(True, index=idx, columns=Wt["tmax"].columns)
    rows = []
    for meth in FILL_METHODS:
        F, _ = fill_daily(masked, rec, weights, end_year, meth)
        for c, s in chosen:
            sl = slice(s, s + 30)
            tx, tn, pr = (Wt[v][c].iloc[sl] for v in VARS)
            fx, fn, fp = (F[v][c].iloc[sl] for v in VARS)
            both = tx.notna() & tn.notna()
            true_t, est_t = ((tx + tn) / 2)[both], ((fx + fn) / 2)[both]
            true_cdd, est_cdd = (true_t - 65).clip(lower=0).sum(), (est_t - 65).clip(lower=0).sum()
            pok = pr.notna()
            tp, ep = pr[pok].sum(), fp[pok].sum()
            rows.append({"method": meth, "city_id": c, "block_start": idx[s].date(),
                         "temp_abs_err": abs(est_t.mean() - true_t.mean()), "cdd_abs_err": abs(est_cdd - true_cdd),
                         "prcp_abs_err_in": abs(ep - tp)})
    t = pd.DataFrame(rows)
    # common evaluation set: blocks where every method produced an estimate
    for col in ["temp_abs_err", "cdd_abs_err", "prcp_abs_err_in"]:
        okb = t.pivot_table(index=["city_id", "block_start"], columns="method", values=col).notna().all(axis=1)
        keep = set(okb[okb].index)
        t.loc[~t.set_index(["city_id", "block_start"]).index.isin(keep), col] = np.nan
    summ = t.groupby("method")[["temp_abs_err", "cdd_abs_err", "prcp_abs_err_in"]].agg(["mean", "count"])
    summ.columns = [f"{a}_{b}" if b == "count" else a for a, b in summ.columns]
    summ = summ.reindex(FILL_METHODS)

    def pick(col, tol):
        best = summ[col].min()
        for m in FILL_METHODS:      # simplest first
            if summ.at[m, col] <= best + tol:
                return m
    temp_m = pick("temp_abs_err", 0.01)
    prcp_m = pick("prcp_abs_err_in", 0.001)
    summ["n_blocks"] = len(chosen)
    return temp_m, prcp_m, summ.reset_index()


def fill_all(W: dict, record: pd.DataFrame, weights: pd.DataFrame, end_year: int, methods: dict) -> tuple[dict, dict]:
    """Temperature and precipitation filled with their own bake-off methods."""
    Ft, wt = fill_daily(W, record, weights, end_year, methods["temperature"])
    if methods["precipitation"] == methods["temperature"]:
        return Ft, wt
    Fp, wp = fill_daily(W, record, weights, end_year, methods["precipitation"])
    return {"tmax": Ft["tmax"], "tmin": Ft["tmin"], "prcp": Fp["prcp"]}, {"tmax": wt["tmax"], "tmin": wt["tmin"],
                                                                           "prcp": wp["prcp"]}


# =============================================================================================
# H. Station adjustment
# =============================================================================================
def _ols_step(yrs, diff, after):
    """OLS diff ~ a + b (y - ybar) + s after. Returns (s, se) or (nan, nan)."""
    X = np.column_stack([np.ones(len(yrs)), yrs - yrs.mean(), after.astype(float)])
    beta, *_ = np.linalg.lstsq(X, diff, rcond=None)
    resid = diff - X @ beta
    dof = len(yrs) - 3
    if dof <= 0:
        return np.nan, np.nan
    s2 = resid @ resid / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    return beta[2], np.sqrt(max(cov[2, 2], 0))


class Homogeniser:
    """H1-H2 for one end year: steps per cluster and variable, placebo-calibrated, two passes."""

    VARS_H = ["Avg Tmax", "Avg Tmin"]

    def __init__(self, T: dict[str, pd.DataFrame], clusters: pd.DataFrame, weights: pd.DataFrame, end_year: int):
        self.E = end_year
        self.T = {v: t.loc[:, :end_year] for v, t in T.items()}      # city x year, valid QC values
        self.cities = list(self.T["Avg Tmax"].index)
        self.years = self.T["Avg Tmax"].columns.to_numpy()
        self.y0 = int(self.years[0])
        self.X = {v: t.to_numpy(dtype=float) for v, t in self.T.items()}
        self.ci = {c: i for i, c in enumerate(self.cities)}
        w = weights.reindex(index=self.cities, columns=self.cities).fillna(0.0)
        self.nb = {}
        for c in self.cities:
            row = w.loc[c]
            nbs = list(row[row > 0].index)
            self.nb[c] = (np.array([self.ci[n] for n in nbs], dtype=int), row[nbs].to_numpy(), nbs)
        self.cl = clusters
        self.bounds = {c: self._bounds(c) for c in self.cities}

    def _bounds(self, city):
        g = self.cl[self.cl["city_id"] == city].sort_values("f")
        out = []
        for r in g.itertuples():
            after_start = r.l.year if (r.l.month == 1 and r.l.day == 1) else r.l.year + 1
            out.append((r.f.year, after_start, r))
        return out

    def windows(self, city, k):
        """Plan H2: up to HOMOG_WINDOW years each side, not crossing other clusters or E, overlap years out."""
        b = self.bounds[city]
        f_year, after_start, r = b[k]
        lo = b[k - 1][1] if k > 0 else -10_000
        hi = b[k + 1][0] - 1 if k + 1 < len(b) else 10_000
        before = range(max(f_year - HOMOG_WINDOW, lo), f_year)
        after = range(after_start, min(after_start + HOMOG_WINDOW - 1, hi, self.E) + 1)
        ov = set(r.overlap_years)
        return [y for y in before if y not in ov], [y for y in after if y not in ov]

    def _idx(self, yrs):
        i = np.array([y - self.y0 for y in yrs], dtype=int)
        return i[(i >= 0) & (i < len(self.years))]

    def estimate(self, var, city, before, after, R, r1):
        """Fixed-reference, trend-adjusted step and its AR(1)-adjusted t (R: neighbours' value matrix)."""
        x = self.X[var][self.ci[city]]
        res = {"step": np.nan, "se": np.nan, "t": np.nan, "n_before": 0, "n_after": 0, "neighbours": ""}
        bi, ai = self._idx(before), self._idx(after)
        bi, ai = bi[np.isfinite(x[bi])], ai[np.isfinite(x[ai])]
        res.update(n_before=len(bi), n_after=len(ai))
        if len(bi) < HOMOG_MIN_VALID or len(ai) < HOMOG_MIN_VALID:
            res["status"] = "unestimable: too few valid years"
            return res
        nbi, nbw, names = self.nb[city]
        win = np.concatenate([bi, ai])
        N = R[nbi][:, win] if len(nbi) else np.zeros((0, len(win)))
        okN = np.isfinite(N)
        nb_ok = (okN[:, :len(bi)].sum(axis=1) >= HOMOG_MIN_VALID) & (okN[:, len(bi):].sum(axis=1) >= HOMOG_MIN_VALID)
        sel = list(np.where(nb_ok)[0])
        common = None
        while len(sel) >= HOMOG_MIN_NB:
            common = okN[sel].all(axis=0)
            if common[:len(bi)].sum() >= HOMOG_MIN_VALID and common[len(bi):].sum() >= HOMOG_MIN_VALID:
                break
            fewest = min(sel, key=lambda j: (okN[j].sum(), names[j]))
            sel.remove(fewest)
        if len(sel) < HOMOG_MIN_NB:
            res["status"] = "unestimable: fewer than 2 neighbours with support"
            return res
        Ns = N[sel][:, common]
        ww = nbw[sel]
        dep = Ns - Ns.mean(axis=1, keepdims=True)
        ref = (ww @ dep) / ww.sum()
        cy = win[common]
        diff = x[cy] - ref
        aft = np.concatenate([np.zeros(len(bi), bool), np.ones(len(ai), bool)])[common]
        s, se = _ols_step(cy.astype(float), diff, aft)
        se_adj = se * np.sqrt((1 + r1) / (1 - r1))
        res.update(step=s, se=se_adj, t=s / se_adj if se_adj > 0 else np.nan, n_before=int((~aft).sum()),
                   n_after=int(aft.sum()), neighbours=",".join(names[j] for j in sel), status="estimated")
        return res

    def ar1(self, var, R):
        """Lag-1 autocorrelation of within-segment residuals of the city - neighbour difference, per city."""
        X = self.X[var]
        with np.errstate(invalid="ignore"):
            anom = R - np.nanmean(R, axis=1, keepdims=True)
        out = {}
        for city in self.cities:
            i = self.ci[city]
            nbi, nbw, _ = self.nb[city]
            if not len(nbi):
                out[city] = np.nan
                continue
            A = anom[nbi]
            have = np.isfinite(A)
            den = nbw @ have
            with np.errstate(invalid="ignore", divide="ignore"):
                ref = np.where(den > 0, (nbw @ np.nan_to_num(A)) / den, np.nan)
            d = X[i] - ref
            seg = np.zeros(len(self.years), dtype=int)
            for k, (f_year, after_start, _) in enumerate(self.bounds[city]):
                seg[self.years >= f_year] = 2 * k + 1
                seg[self.years >= after_start] = 2 * k + 2
            res = np.full(len(d), np.nan)
            for sg in np.unique(seg):
                m = (seg == sg) & np.isfinite(d)
                if m.any():
                    res[m] = d[m] - d[m].mean()
            a, b = res[:-1], res[1:]
            ok = np.isfinite(a) & np.isfinite(b) & (seg[:-1] == seg[1:])
            out[city] = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() >= 10 else np.nan
        s = pd.Series(out)
        return s.fillna(s.median()).clip(0, AR1_MAX)

    def placebos(self, var, R, r1):
        """Placebo changes on 1 July of every year >= HOMOG_WINDOW years from a real change, same window rule."""
        rows = []
        X = self.X[var]
        for city in self.cities:
            x = X[self.ci[city]]
            valid = self.years[np.isfinite(x)]
            if len(valid) < 2 * HOMOG_MIN_VALID:
                continue
            b = self.bounds[city]
            ch_years = [y for f, a, _ in b for y in (f, a - 1)]
            for Y in range(int(valid.min()) + HOMOG_MIN_VALID, min(int(valid.max()), self.E) - HOMOG_MIN_VALID + 1):
                if any(abs(Y - cy) < HOMOG_WINDOW for cy in ch_years):
                    continue
                lo = max([a for f, a, _ in b if a <= Y], default=-10_000)
                hi = min([f - 1 for f, a, _ in b if f > Y], default=10_000)
                before = range(max(Y - HOMOG_WINDOW, lo), Y)
                after = range(Y + 1, min(Y + HOMOG_WINDOW, hi, self.E) + 1)
                e = self.estimate(var, city, before, after, R, r1[city])
                if e["status"] == "estimated":
                    rows.append((e["t"], min(e["n_before"], e["n_after"]) < HOMOG_WINDOW))
        return pd.DataFrame(rows, columns=["t", "short"])

    def run(self, passes: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
        R = {v: x.copy() for v, x in self.X.items()}
        steps, calib = None, []
        for p in range(1, passes + 1):
            rows = []
            for var in self.VARS_H:
                r1 = self.ar1(var, R[var])
                pl = self.placebos(var, R[var], r1)
                full = pl.loc[~pl["short"], "t"].abs()
                short = pl.loc[pl["short"], "t"].abs()
                tc_full = max(HOMOG_MIN_T, float(full.quantile(PLACEBO_Q))) if len(full) else HOMOG_MIN_T
                tc_short = max(HOMOG_MIN_T, float(short.quantile(PLACEBO_Q))) if len(short) else tc_full
                if len(short) < PLACEBO_MIN:
                    tc_short = max(tc_short, tc_full)
                calib.append({"pass": p, "var": var, "placebos_full": len(full), "placebos_short": len(short),
                              "t_crit_full": tc_full, "t_crit_short": tc_short,
                              "placebo_rate_full": float((full >= tc_full).mean()) if len(full) else np.nan,
                              "placebo_rate_short": float((short >= tc_short).mean()) if len(short) else np.nan,
                              "median_ar1": float(r1.median())})
                for city in self.cities:
                    for k, (f_year, after_start, r) in enumerate(self.bounds[city]):
                        row = {"city_id": city, "var": var, "cluster_id": r.cluster_id, "f": r.f, "l": r.l,
                               "P": r.P, "Q": r.Q, "label": r.label, "n_changes": r.n_changes, "ar1": r1[city]}
                        if r.same_endpoints:
                            row.update(status="same station before and after: not a change", step=np.nan, se=np.nan,
                                       t=np.nan, t_crit=np.nan, n_before=0, n_after=0, neighbours="")
                        else:
                            before, after = self.windows(city, k)
                            e = self.estimate(var, city, before, after, R[var], r1[city])
                            short_w = min(e["n_before"], e["n_after"]) < HOMOG_WINDOW
                            row.update(e, t_crit=tc_short if short_w else tc_full)
                        rows.append(row)
            steps = pd.DataFrame(rows)
            steps["applied"] = ((steps["status"] == "estimated") & (steps["t"].abs() >= steps["t_crit"])
                                & (steps["step"].abs() >= HOMOG_MIN_STEP))
            steps = return_pair_rule(steps)
            if p < passes:
                for var in self.VARS_H:
                    R[var] = self.X[var] + annual_offsets(steps[steps["var"] == var], self.T[var]).to_numpy()
        steps["pass"] = passes
        return steps, pd.DataFrame(calib)


def return_pair_rule(steps: pd.DataFrame) -> pd.DataFrame:
    """C5: an adopted A->B step and a later adopted B->A step must cancel to within RETURN_MAX_SUM."""
    steps = steps.copy()
    steps["return_pair_sum"] = np.nan
    for (city, var), g in steps[steps["applied"]].groupby(["city_id", "var"]):
        g = g.sort_values("f")
        for i, a in g.iterrows():
            for j, b in g.iterrows():
                if b["f"] > a["f"] and b["P"] == a["Q"] and b["Q"] == a["P"]:
                    s = a["step"] + b["step"]
                    steps.loc[[i, j], "return_pair_sum"] = s
                    if abs(s) > RETURN_MAX_SUM:
                        steps.loc[[i, j], "applied"] = False
                        steps.loc[[i, j], "status"] = "estimated; return pair does not cancel: not applied"
    return steps


def annual_offsets(steps: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    """Approximate yearly offsets (for the second pass's neighbour reference): sum of later adopted steps,
    plus the cluster's own step for years entirely before it."""
    off = pd.DataFrame(0.0, index=T.index, columns=T.columns)
    yrs = T.columns.to_numpy()
    for city, g in steps[steps["applied"]].groupby("city_id"):
        if city not in off.index:
            continue
        v = np.zeros(len(yrs))
        for r in g.itertuples():
            v[yrs < r.f.year] += r.step
        off.loc[city] = v
    return off


def daily_offsets(steps: pd.DataFrame, var: str, keys: pd.DataFrame, clusters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H2 offsets per day for one daily variable, and the transition-day mask (plan §4 H2 formula)."""
    col = "Avg Tmax" if var == "tmax" else "Avg Tmin"
    st = steps[steps["var"] == col].set_index(["city_id", "cluster_id"])
    idx = keys.index
    off = np.zeros(keys.shape)
    trans = np.zeros(keys.shape, dtype=bool)
    for j, city in enumerate(keys.columns):
        g = clusters[clusters["city_id"] == city].sort_values("f")
        if not len(g):
            continue
        K = keys[city].to_numpy()
        A = np.array([st.at[(city, r.cluster_id), "step"] if (city, r.cluster_id) in st.index
                      and bool(st.at[(city, r.cluster_id), "applied"]) else 0.0 for r in g.itertuples()])
        later = np.append(A[::-1].cumsum()[::-1], 0.0)            # later[k] = sum of A_j, j >= k
        f = g["f"].to_numpy()
        pos = np.searchsorted(f, idx.to_numpy(), side="right")     # clusters with f_j <= t
        o = later[pos]                                             # sum over clusters with f_j > t
        for k, r in enumerate(g.itertuples()):
            inside = (idx >= r.f) & (idx < r.l)
            if not inside.any():
                continue
            base = later[k + 1]
            kk = K[inside]
            vals = np.full(kk.shape, base)
            vals[kk == r.P] = base + A[k]
            third = (kk != r.P) & (kk != r.Q) & (kk != 0)
            if A[k] != 0:
                trans_idx = np.where(inside)[0][third]
                trans[trans_idx, j] = True
            o[inside] = vals
        off[:, j] = o
    return pd.DataFrame(off, index=idx, columns=keys.columns), pd.DataFrame(trans, index=idx, columns=keys.columns)


# =============================================================================================
# A6. Neighbour infill of empty years
# =============================================================================================
def neighbour_infill(T: pd.DataFrame, weights: pd.DataFrame, end_year: int, never: pd.DataFrame | None = None,
                     clip0: bool = False) -> pd.DataFrame:
    T = T.loc[:, :end_year]
    years = T.columns.to_numpy()
    X = T.to_numpy(dtype=float)
    pos = {c: i for i, c in enumerate(T.index)}
    out = X.copy()
    nev = never.reindex(index=T.index, columns=T.columns).fillna(False).to_numpy(bool) if never is not None else None
    for i, c in enumerate(T.index):
        valid = ~np.isnan(X[i])
        if not valid.any():
            continue
        targets = np.where(~valid & (years >= years[valid][0]))[0]
        if nev is not None:
            targets = targets[~nev[i, targets]]
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
    res = pd.DataFrame(out, index=T.index, columns=T.columns)
    return res.clip(lower=0) if clip0 else res


def infill_evaluation(T: pd.DataFrame, weights: pd.DataFrame, years=(1989, 2018), n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """A6 evidence: hide known years; (b) past-only (offsets and baseline from earlier years) and (a) as used."""
    rng = np.random.default_rng(seed)
    T = T.loc[:, :years[1]]
    cells = list(T.loc[:, years[0]:years[1]].stack().index)
    picks = [cells[i] for i in rng.choice(len(cells), min(n, len(cells)), replace=False)]
    rows = []
    for c, y in picks:
        truth = T.at[c, y]
        nbs = [k for k in weights.columns[weights.loc[c] > 0] if k in T.index]
        sub = T.loc[[c, *nbs]].copy()
        sub.at[c, y] = np.nan
        est_a = neighbour_infill(sub, weights, years[1]).at[c, y]
        past = sub.loc[:, :y]
        est_b = neighbour_infill(past, weights, y).at[c, y]
        prev = sub.loc[c, :y - 1].dropna()
        rows.append({"city_id": c, "year": y, "past_only_infill": est_b - truth,
                     "past_only_baseline_prev15": (prev.iloc[-15:].mean() - truth) if len(prev) else np.nan,
                     "as_used_infill": est_a - truth,
                     "as_used_baseline_pm15": sub.loc[c, y - 15:y + 15].mean() - truth})
    return pd.DataFrame(rows)


# =============================================================================================
# Build everything for one end year
# =============================================================================================
def build(R: dict, S: dict, end_year: int, overrides: pd.DataFrame, fill_methods: dict, verbose: bool = True) -> dict:
    """QC, fills, station adjustment and every input variant, from data <= end_year only."""
    E = end_year
    end = f"{E}-12-31"
    rec = R["record"].loc[:end]
    Wraw = {v: m.loc[:end] for v, m in R["W"].items()}
    Q = quality_control(R, E, S["nbrs"], overrides, verbose)
    Wc = Q["W"]
    f_raw = annual_from_daily(Wraw, rec)
    f_qc = annual_from_daily(Wc, rec)
    keys = S["annual_index"]
    keys = keys[keys.get_level_values(1) <= E]
    f_raw, f_qc = f_raw.reindex(keys), f_qc.reindex(keys)
    pub = S["published"].reindex(keys)
    labels = S["labels"].reindex(keys)
    fr = fragment_years(Q["fragments"])
    dropped = {g: pd.Series([(c, y) in s for c, y in keys], index=keys) for g, s in fr.items()}
    grp_of = lambda col: "precipitation" if "Prcp" in col or col.endswith(" P") else "temperature"  # noqa: E731
    qc = pd.DataFrame(index=keys)
    for col in REBUILT:
        ok = valid_mask(f_qc, col) & ~dropped[grp_of(col)]
        qc[col] = delta_form(pub[col], f_qc[col], f_raw[col], ok)
    for m, col in METRIC_COL.items():
        qc[m] = delta_form(labels[m], f_qc[col], f_raw[col], valid_mask(f_qc, col) & ~dropped[grp_of(col)])
    qc_invalid = pd.DataFrame({m: labels[m].notna() & qc[m].isna() for m in METRICS})
    # H: station adjustment from QC Avg Tmax / Avg Tmin
    if verbose:
        say(f"  [{E}] H station adjustment ...")
    T = {c: qc[c].unstack("Year").reindex(columns=range(int(keys.get_level_values(1).min()), E + 1))
         for c in ["Avg Tmax", "Avg Tmin"]}
    clusters = make_clusters(S["changes"], E)
    steps, calib = Homogeniser(T, clusters, S["weights"], E).run()
    kd = S["keys_by_day"].loc[:end]
    offs, trans = {}, {}
    for v in TEMP_VARS:
        offs[v], trans[v] = daily_offsets(steps, v, kd, clusters)
    Wh = {"tmax": (Wc["tmax"] + offs["tmax"]).mask(trans["tmax"]), "tmin": (Wc["tmin"] + offs["tmin"]).mask(trans["tmin"]),
          "prcp": Wc["prcp"]}
    f_h = annual_from_daily(Wh, rec).reindex(keys)
    ty = (trans["tmax"] | trans["tmin"]).groupby(trans["tmax"].index.year).any().stack()
    ty.index.names = ["Year", "ID"]
    trans_year = ty.swaplevel().reindex(keys).fillna(False).astype(bool)
    # D6: fill on adjusted days (reference level); plain fill = the same estimates moved back by the offsets
    if verbose:
        say(f"  [{E}] D6 fill ...")
    Fh, was = fill_all(Wh, rec, S["weights"], E, fill_methods)
    Fp = {"tmax": Wc["tmax"].where(Wc["tmax"].notna(), Fh["tmax"] - offs["tmax"]),
          "tmin": Wc["tmin"].where(Wc["tmin"].notna(), Fh["tmin"] - offs["tmin"]), "prcp": Fh["prcp"]}
    was_p = {v: Wc[v].isna() & Fp[v].notna() & rec for v in VARS}
    f_fh = annual_from_daily(Fh, rec).reindex(keys)
    f_fp = annual_from_daily(Fp, rec).reindex(keys)
    fill_ok_p = {col: f_qc[MISS_RULE[col]].max(axis=1) <= FILL_MAX_MISSING for col in METRIC_COL.values()}
    fill_ok_h = {col: f_h[MISS_RULE[col]].max(axis=1) <= FILL_MAX_MISSING for col in METRIC_COL.values()}
    V = {}

    def wide(s):
        return s.unstack("Year").reindex(columns=range(1851, E + 1))

    for m, col in METRIC_COL.items():
        g = grp_of(col)
        nd = ~dropped[g]
        L = labels[m]
        var = {"official": L, "qc": qc[m]}
        homog = delta_form(L, f_h[col], f_raw[col], valid_mask(f_h, col) & nd) if g == "temperature" else qc[m]
        filled = delta_form(L, f_fp[col], f_raw[col], fill_ok_p[col] & nd)
        filled_h = delta_form(L, f_fh[col], f_raw[col], fill_ok_h[col] & nd) if g == "temperature" else filled
        var.update({"qc+homog": homog, "qc+fill": filled, "qc+fill+homog": filled_h})
        never = qc_invalid[m] | dropped[g]
        out = {}
        for name in CANDIDATES[m]:
            base = name.replace("+nbfill", "")
            s = homog if (base == "qc+homog" and m == "hottest_day_temp") else var[base]
            # completion with official labels, except where QC or a station transition removed data on purpose
            nv = never | trans_year if ("homog" in base and g == "temperature") else never
            s = s.where(s.notna() | nv, L)
            out[name] = wide(s)
        for name in CANDIDATES[m]:
            if name.endswith("+nbfill"):
                out[name] = neighbour_infill(out[name.replace("+nbfill", "")], S["weights"], E,
                                             never=wide(dropped[g]), clip0=m in ("cdd", "total_precip_in"))
        V[m] = out
    with fold_cutoff(E):
        for m in V:
            for name, t in V[m].items():
                assert_no_post_cutoff(pd.DataFrame({"Year": t.columns}), "Year", f"variant {m}/{name}")
    return dict(E=E, Q=Q, qc=qc, qc_invalid=qc_invalid, f_raw=f_raw, f_qc=f_qc, f_h=f_h, steps=steps, calib=calib,
                clusters=clusters, offsets=offs, transition=trans, Wh=Wh, Fp=Fp, Fh=Fh, was_filled=was_p, V=V,
                labels=labels, dropped=dropped)


def setup(R: dict) -> dict:
    """Static structures: neighbours, stations, changes, labels."""
    cities = R["cities"]
    dist = distance_matrix(cities)
    st = build_stations(R["city_info"], R["W"])
    changes = effective_changes(st)
    ann = R["annual"].set_index(["ID", "Year"])
    return dict(dist=dist, nbrs=nearest(dist, QC_NEIGHBOURS), weights=idw_weights(dist), stations=st,
                changes=changes, keys_by_day=station_key_by_day(st, R["W"]["tmax"].index, R["W"]["tmax"].columns),
                labels=official_labels(R["annual"]), published=published_masked(R["annual"]),
                annual_index=ann.index)


# =============================================================================================
# E. Selecting inputs
# =============================================================================================
def backtest_errors(builds: dict, labels: pd.DataFrame) -> pd.DataFrame:
    """Absolute errors per (origin, metric, candidate, forecaster, city, year) on the full official mask,
    plus the baseline (official 30-year average) errors."""
    import combo_model as cm
    from common import thirty_year_mean
    rows = []
    for origin, B in builds.items():
        years = list(range(origin + 1, origin + 1 + TEST_LENGTH))
        assert years[-1] <= CUTOFF_YEAR
        for m in METRICS:
            truth = labels[m].unstack("Year").reindex(columns=range(1851, CUTOFF_YEAR + 1))
            hist_off = truth.loc[:, :origin]
            base_ok = thirty_year_mean(hist_off, origin).notna()
            actual = truth[years].where(base_ok, axis=0)
            mask = actual.notna()
            base = cm.average_forecast(hist_off, origin, years).reindex(index=actual.index, columns=years)
            be = (actual - base).abs()
            if base.where(mask).isna().values[mask.values].any():
                raise ValueError("E: baseline missing on the mask")
            for name, tab in B["V"][m].items():
                hist = tab.reindex(index=truth.index).loc[:, :origin]
                for fc in FORECASTER_ORDER + ["last_year"]:
                    if fc == "last_year":
                        pred = last_year_forecast(hist, origin, years).reindex(index=actual.index, columns=years)
                    else:
                        pred = cm.FORECASTERS[fc](hist, origin, years).reindex(index=actual.index, columns=years)
                    if pred.isna().values[mask.values].any():
                        raise ValueError(f"E: {m}/{name}/{fc} at {origin}: non-finite forecast on the official mask")
                    err = (actual - pred).abs()
                    s = err.where(mask).stack()
                    rows.append(pd.DataFrame({"origin": origin, "metric": m, "candidate": name, "forecaster": fc,
                                              "city_id": s.index.get_level_values(0), "year": s.index.get_level_values(1),
                                              "abs_err": s.values,
                                              "base_err": be.where(mask).stack().reindex(s.index).values}))
    return pd.concat(rows, ignore_index=True)


def last_year_forecast(history: pd.DataFrame, origin: int, years: list[int]) -> pd.DataFrame:
    """Reference (reported only): the last valid year of the 30-year window, else the 30-year average."""
    import combo_model as cm
    win = history.loc[:, origin - 29:origin]
    last = win.ffill(axis=1).iloc[:, -1]
    last = last.fillna(cm.average_forecast(history, origin, years)[years[0]])
    return pd.DataFrame({y: last for y in years})


def _sums(err: pd.DataFrame, cities: list[str], horizon: int = TEST_LENGTH) -> pd.DataFrame:
    e = err[err["year"] - err["origin"] <= horizon]
    g = e.groupby(["origin", "metric", "candidate", "forecaster", "city_id"])
    s = g.agg(err=("abs_err", "sum"), base=("base_err", "sum"), n=("abs_err", "size")).reset_index()
    return s


def skill_table(sums: pd.DataFrame) -> pd.DataFrame:
    g = sums.groupby(["origin", "metric", "candidate", "forecaster"])[["err", "base", "n"]].sum()
    g["skill"] = 1 - g["err"] / g["base"]
    return g.reset_index()


def bootstrap_G(sums: pd.DataFrame, cities: pd.DataFrame, metric: str, forecaster: str, origins: tuple,
                candidates: list[str], cell_deg: float = BOOT_CELL_DEG, reps: int = BOOT_REPS, seed: int = 0) -> pd.DataFrame:
    """Spatial-block bootstrap of G(v) = mean over origins of skill(v) - skill(official)."""
    cell = (np.floor(cities["lat"] / cell_deg).astype(int).astype(str) + "_"
            + np.floor(cities["lon"] / cell_deg).astype(int).astype(str))
    cell_ids = sorted(cell.unique())
    city_list = list(cities.index)
    M = np.zeros((len(cell_ids), len(city_list)))
    for j, c in enumerate(city_list):
        M[cell_ids.index(cell[c]), j] = 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cell_ids), size=(reps, len(cell_ids)))
    counts = np.zeros((reps, len(cell_ids)))
    for r in range(reps):
        counts[r] = np.bincount(draws[r], minlength=len(cell_ids))
    Wc = counts @ M                                                   # reps x cities: weight of each city
    s = sums[(sums["metric"] == metric) & (sums["forecaster"] == forecaster) & sums["origin"].isin(origins)]

    def per_origin(cand):
        out = []
        for o in origins:
            t = s[(s["candidate"] == cand) & (s["origin"] == o)].set_index("city_id").reindex(city_list).fillna(0)
            err, base = Wc @ t["err"].to_numpy(), Wc @ t["base"].to_numpy()
            out.append(1 - err / base)
        return np.mean(out, axis=0)

    ref = per_origin("official")
    k = max(1, len([c for c in candidates if c != "official"]))
    rows = []
    for cand in candidates:
        if cand == "official":
            continue
        d = per_origin(cand) - ref
        rows.append({"metric": metric, "forecaster": forecaster, "candidate": cand, "cell_deg": cell_deg,
                     "G_boot_mean": float(d.mean()), "lower": float(np.quantile(d, BOOT_ALPHA / k)),
                     "alpha_over_k": BOOT_ALPHA / k})
    return pd.DataFrame(rows)


def G_point(skills: pd.DataFrame, metric: str, forecaster: str, origins: tuple) -> pd.Series:
    t = skills[(skills["metric"] == metric) & (skills["forecaster"] == forecaster) & skills["origin"].isin(origins)]
    p = t.pivot(index="candidate", columns="origin", values="skill")
    return (p.sub(p.loc["official"], axis=1)).mean(axis=1)


def rule_R(sums: pd.DataFrame, skills: pd.DataFrame, cities: pd.DataFrame, metric: str, origins: tuple,
           forecaster: str | None = None) -> dict:
    """§5 rule R on a set of origins: forecaster (unless given), then input."""
    if forecaster is None:
        t = skills[(skills["metric"] == metric) & (skills["candidate"] == "official") & skills["origin"].isin(origins)]
        ms = t.groupby("forecaster")["skill"].mean().reindex(FORECASTER_ORDER)
        forecaster = next(f for f in FORECASTER_ORDER if ms[f] >= ms.max() - 1e-12)
    G = G_point(skills, metric, forecaster, origins)
    boot = bootstrap_G(sums, cities, metric, forecaster, origins, CANDIDATES[metric])
    lower = boot.set_index("candidate")["lower"]
    eligible = [c for c in PREFERENCE if c in CANDIDATES[metric] and c != "official"
                and G[c] >= MIN_GAIN and lower[c] > 0]
    choice = "official"
    for c in eligible:                 # simplest first; a more complex one must add >= MIN_GAIN
        if choice == "official" or G[c] >= G[choice] + MIN_GAIN:
            choice = c
    return {"metric": metric, "origins": list(origins), "forecaster": forecaster, "choice": choice,
            "G": G.to_dict(), "lower": lower.to_dict(), "eligible": eligible}


def select_inputs(err: pd.DataFrame, cities: pd.DataFrame) -> dict:
    import combo_model as cm
    sums = _sums(err, list(cities.index))
    skills = skill_table(sums)
    sums7 = _sums(err, list(cities.index), horizon=7)
    skills7 = skill_table(sums7)
    out = {"production": {}, "nested": [], "ship": {}, "sensitivity": []}
    for m in METRICS:
        prod = rule_R(sums, skills, cities, m, ORIGINS, forecaster=cm.METHOD[m])
        out["production"][m] = prod
        folds = []
        for outer, inner in NESTED.items():
            r = rule_R(sums, skills, cities, m, inner)
            t = skills[(skills["metric"] == m) & (skills["origin"] == outer) & (skills["forecaster"] == r["forecaster"])]
            t = t.set_index("candidate")["skill"]
            gain = float(t[r["choice"]] - t["official"])
            folds.append({"metric": m, "outer_origin": outer, "inner_origins": list(inner),
                          "forecaster": r["forecaster"], "choice": r["choice"], "outer_gain": gain})
        cond = []
        for outer, inner in NESTED.items():       # the same nest with the production forecaster held fixed
            r = rule_R(sums, skills, cities, m, inner, forecaster=cm.METHOD[m])
            t = skills[(skills["metric"] == m) & (skills["origin"] == outer) & (skills["forecaster"] == cm.METHOD[m])]
            t = t.set_index("candidate")["skill"]
            cond.append({"metric": m, "outer_origin": outer, "inner_origins": list(inner), "forecaster": cm.METHOD[m],
                         "choice": r["choice"], "outer_gain": float(t[r["choice"]] - t["official"]),
                         "kind": "conditional on METHOD"})
        for f in folds:
            f["kind"] = "forecaster and input chosen inside"
        out["nested"] += folds + cond
        mean_gain = float(np.mean([f["outer_gain"] for f in folds]))
        any_non_off = any(f["choice"] != "official" for f in folds)
        cond_gain = float(np.mean([f["outer_gain"] for f in cond]))
        ok = mean_gain > 0 and any_non_off and cond_gain > 0
        ship = prod["choice"] if (prod["choice"] == "official" or ok) else "official"
        out["ship"][m] = {"production_rule_choice": prod["choice"], "nested_mean_outer_gain": mean_gain,
                          "nested_any_non_official": any_non_off, "conditional_mean_outer_gain": cond_gain,
                          "shipped": ship, "forecaster": cm.METHOD[m]}
        for deg in (3.0, 8.0):
            out["sensitivity"].append(bootstrap_G(sums, cities, m, cm.METHOD[m], ORIGINS, CANDIDATES[m], cell_deg=deg))
        g7 = G_point(skills7, m, cm.METHOD[m], ORIGINS)
        out["sensitivity"].append(pd.DataFrame({"metric": m, "forecaster": cm.METHOD[m], "candidate": g7.index,
                                                "cell_deg": np.nan, "G_7yr": g7.values}))
    out["skills"] = skills
    out["sums"] = sums
    return out


# =============================================================================================
# Outputs
# =============================================================================================
def snake(c: str) -> str:
    return c.lower().replace(" ", "_").replace("-", "").replace(".", "")


def assemble_annual(R: dict, S: dict, B: dict, ship: dict) -> pd.DataFrame:
    keys = S["annual_index"]
    a = R["annual"].set_index(["ID", "Year"]).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame(index=keys)
    for m in METRICS:
        out[m] = S["labels"][m]
    pub = S["published"]
    for col in REBUILT:
        if col not in METRIC_COL.values():
            out[snake(col)] = pub[col]
    for col, miss in PCT_COLS.items():
        days = B["f_raw"]["days_in_record"].reindex(keys)
        out[f"{snake(col)}_pct_valid"] = (a[col] * days / (days - a[miss])).where(a[miss] <= MAX_MISSING_DAYS)
    out["wet_days_in"] = a["Wet Days"].where(a["Miss Prcp"] <= MAX_MISSING_DAYS)
    qc = B["qc"].reindex(keys)
    for m in METRICS:
        out[f"{m}_qc"] = qc[m]
    for col in REBUILT:
        if col not in METRIC_COL.values():
            out[f"{snake(col)}_qc"] = qc[col]
    for m in METRICS:
        out[f"{m}_qc_invalidated"] = B["qc_invalid"][m].reindex(keys).fillna(False)

    def long(t):
        s = t.stack()
        s.index.names = ["ID", "Year"]
        return s.reindex(keys)

    for m in METRICS:
        for name, t in B["V"][m].items():
            if name in ("official", "qc"):
                continue
            out[f"{m}_{name.replace('qc+', '').replace('+', '_')}"] = long(t)
        chosen = ship[m]["shipped"]
        out[f"{m}_input"] = long(B["V"][m][chosen])
        src = np.where(B["labels"][m].reindex(keys).notna() & qc[m].isna() & out[f"{m}_input"].notna()
                       & (chosen != "official"), "official", "measured")
        if "fill" in chosen:
            n_f = (B["f_qc"][MISS_RULE[METRIC_COL[m]]].max(axis=1).reindex(keys) > 0)
            src = np.where(n_f & out[f"{m}_input"].notna(), "filled days", src)
        if "nbfill" in chosen:
            base = long(B["V"][m][chosen.replace("+nbfill", "")])
            src = np.where(base.isna() & out[f"{m}_input"].notna(), "neighbour infill", src)
        out[f"{m}_input_source"] = pd.Series(src, index=keys).where(out[f"{m}_input"].notna())
        out[f"{m}_input_variant"] = chosen
    lg = B["Q"]["log"].assign(year=lambda x: x["date"].dt.year)
    for act in ["delete", "swap"]:
        n = lg[lg["action"] == act].groupby(["city_id", "year", "var"]).size().unstack("var", fill_value=0)
        n = n.reindex(columns=VARS, fill_value=0).add_prefix(f"n_{'deleted' if act == 'delete' else 'swapped'}_")
        n.index.names = ["ID", "Year"]
        out = out.join(n)
    for v in VARS:
        w = B["was_filled"][v]
        n = w.groupby(w.index.year).sum().stack()
        n.index.names = ["Year", "ID"]
        out[f"n_filled_{v}"] = n.swaplevel().reindex(keys).fillna(0).astype(int)
    cnt = [c for c in out.columns if c.startswith(("n_deleted_", "n_swapped_"))]
    out[cnt] = out[cnt].fillna(0).astype(int)
    cities = R["cities"]
    out = out.reset_index().rename(columns={"ID": "city_id", "Year": "year"})
    out.insert(1, "name", cities["name"].reindex(out["city_id"]).values)
    out.insert(2, "lat", cities["lat"].reindex(out["city_id"]).values)
    out.insert(3, "lon", cities["lon"].reindex(out["city_id"]).values)
    return out


def assemble_coverage(R: dict, S: dict, B: dict) -> pd.DataFrame:
    keys = S["annual_index"]
    cv = R["coverage"].set_index(["ID", "Year"])
    a = R["annual"].set_index(["ID", "Year"])
    f_qc = B["f_qc"].reindex(keys)
    out = pd.DataFrame(index=keys)
    out["days_in_year"] = [366 if pd.Timestamp(f"{y}-12-31").day_of_year == 366 else 365 for y in keys.get_level_values(1)]
    out["days_in_record"] = B["f_raw"]["days_in_record"].reindex(keys).fillna(0).astype(int)
    out["miss_tmax"], out["miss_tmin"], out["miss_prcp"] = a["Miss Tmax"], a["Miss Tmin"], a["Miss Prcp"]
    out["days_both"] = cv["Days Both"]
    for c in ["Miss Tmax", "Miss Tmin", "Miss Prcp", "Days Both"]:
        out[f"{snake(c)}_qc"] = f_qc[c].fillna(0).astype(int)
    for m, (col, miss) in METRICS.items():
        out[f"valid_{m}"] = a[miss].max(axis=1) <= MAX_MISSING_DAYS
    out["cdd_incomplete_days"] = out["days_in_record"] - out["days_both"]
    out["segment_jan1"] = cv["Station Segment"]
    out["station_changes"] = cv["Station Changes"]
    kd = S["keys_by_day"]
    rec = R["record"]
    maj = {}
    K = kd.where(rec, 0)
    for city in kd.columns:
        s = K[city]
        g = s[s > 0].groupby(s[s > 0].index.year)
        maj[city] = g.agg(lambda x: x.value_counts().index[0])
    maj = pd.concat(maj).rename_axis(["ID", "Year"])
    out["station_key_majority"] = maj.reindex(keys)
    cl = B["clusters"]
    chg = S["changes"]
    last_change = {}
    for r in chg.itertuples():
        last_change.setdefault(r.city_id, []).append(r.date)
    ysc, cy, share = [], [], []
    for city, y in keys:
        ds = [d for d in last_change.get(city, []) if d.year <= y]
        ysc.append(y - ds[-1].year if ds else np.nan)
        this = [d for d in last_change.get(city, []) if d.year == y]
        cy.append(bool(this))
        share.append(((pd.Timestamp(f"{y}-12-31") - max(this)).days + 1) / out.at[(city, y), "days_in_year"]
                     if this else 1.0)
    out["years_since_change"], out["change_year"], out["share_last_station"] = ysc, cy, share
    cid = []
    for city, y in keys:
        g = cl[(cl["city_id"] == city) & (cl["f"].dt.year <= y) & (cl["l"].dt.year >= y)]
        cid.append(int(g["cluster_id"].iloc[0]) if len(g) else -1)
    out["cluster_id"] = cid
    st = S["stations"]
    gap_days = pd.Series(0, index=keys)
    empty_days = pd.Series(0, index=keys)
    for r in st.itertuples():
        if r.gap_before_days > 0:
            d = pd.date_range(r.start - pd.Timedelta(days=r.gap_before_days), r.start - pd.Timedelta(days=1))
            for y, n in pd.Series(1, index=d).groupby(d.year).sum().items():
                if (r.city_id, y) in gap_days.index:
                    gap_days[(r.city_id, y)] += n
        if r.empty_period:
            d = pd.date_range(r.start, r.end)
            for y, n in pd.Series(1, index=d).groupby(d.year).sum().items():
                if (r.city_id, y) in empty_days.index:
                    empty_days[(r.city_id, y)] += n
    out["in_gap_days"], out["empty_station_days"] = gap_days, empty_days
    fd = B["dropped"]
    out["fragment_dropped_temperature"] = fd["temperature"].reindex(keys).fillna(False)
    out["fragment_dropped_precipitation"] = fd["precipitation"].reindex(keys).fillna(False)
    br = B["Q"]["breaks"]
    for grp in ["temperature", "precipitation"]:
        b = set(map(tuple, br[br["group"] == grp][["city_id", "year"]].to_numpy()))
        out[f"record_break_before_{grp}"] = [(c, y) in b for c, y in keys]
    for v in TEMP_VARS:
        o = B["offsets"][v].where(B["Wh"][v].notna())
        oy = o.groupby(o.index.year).mean().stack()
        oy.index.names = ["Year", "ID"]
        out[f"offset_{v}"] = oy.swaplevel().reindex(keys).fillna(0.0)
        t = B["transition"][v]
        ty = t.groupby(t.index.year).sum().stack()
        ty.index.names = ["Year", "ID"]
        out[f"transition_days_{v}"] = ty.swaplevel().reindex(keys).fillna(0).astype(int)
    out = out.reset_index().rename(columns={"ID": "city_id", "Year": "year"})
    return out


def assemble_stations(S: dict, B: dict) -> pd.DataFrame:
    st = S["stations"].copy()
    cl = B["clusters"]
    st["cluster_id"] = -1
    for r in cl.itertuples():
        m = (st["city_id"] == r.city_id) & (st["start"] >= r.f) & (st["start"] <= r.l)
        st.loc[m, "cluster_id"] = r.cluster_id
    steps = B["steps"]
    for var, sfx in [("Avg Tmax", "tmax"), ("Avg Tmin", "tmin")]:
        s = steps[steps["var"] == var].set_index(["city_id", "cluster_id"])
        for col in ["step", "t", "t_crit", "n_before", "n_after", "applied", "status"]:
            st[f"{col}_{sfx}"] = [s.at[(c, k), col] if k >= 0 and (c, k) in s.index else np.nan
                                  for c, k in zip(st["city_id"], st["cluster_id"])]
    return st


def assemble_daily(R: dict, B: dict) -> pd.DataFrame:
    Wr = {v: m.loc[:f"{CUTOFF_YEAR}-12-31"] for v, m in R["W"].items()}
    Wc = B["Q"]["W"]
    log = B["Q"]["log"]
    parts = {}
    for v in VARS:
        parts[f"{v}_raw"] = Wr[v]
        parts[v] = Wc[v]
        parts[f"{v}_filled"] = B["Fp"][v]
    for v in TEMP_VARS:
        parts[f"{v}_offset"] = B["offsets"][v]
    rec = R["record"]
    long = pd.concat({k: m.astype("float32").where(rec).stack(future_stack=True) for k, m in parts.items()}, axis=1)
    long.index.names = ["date", "city_id"]
    keep = rec.stack(future_stack=True)
    long = long[keep.reindex(long.index).to_numpy()]
    for v in VARS:
        flag = np.where(long[f"{v}_raw"].isna(), FLAG["missing"], FLAG["observed"]).astype(np.int8)
        lv = log[log["var"] == v]
        idx = pd.MultiIndex.from_arrays([pd.to_datetime(lv["date"]), lv["city_id"]])
        pos = long.index.get_indexer(idx)
        code = np.where(lv["action"] == "swap", FLAG["swapped"], FLAG["deleted"])
        flag[pos[pos >= 0]] = code[pos >= 0]
        if v in TEMP_VARS:
            t = B["transition"][v].stack()
            t = t[t]
            p2 = long.index.get_indexer(t.index)
            flag[p2[p2 >= 0]] = FLAG["transition"]
        long[f"{v}_flag"] = flag
    long = long.reset_index()
    cols = ["city_id", "date"] + [c for c in long.columns if c not in ("city_id", "date")]
    long = long[cols].sort_values(["city_id", "date"]).reset_index(drop=True)
    assert_no_post_cutoff(long.assign(year=long["date"].dt.year), "year", "daily_clean")
    return long


def plot_outliers(cand: pd.DataFrame, W: dict, nbrs: dict, out_dir: Path, per_page: int = 48) -> int:
    """Gallery of the isolated outliers: the city (black) and its 3 QC neighbours (grey), ±10 days."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in out_dir.glob("*.png"):
        p.unlink()
    iso = cand[cand["class"] == "isolated"].sort_values(["date", "city_id"]).reset_index(drop=True)
    pages = int(np.ceil(len(iso) / per_page))
    for p in range(pages):
        sub = iso.iloc[p * per_page:(p + 1) * per_page]
        fig, axes = plt.subplots(8, 6, figsize=(18, 20))
        for ax, r in zip(axes.flat, sub.itertuples()):
            M = W[r.var]
            d0 = pd.Timestamp(r.date)
            win = M.loc[d0 - pd.Timedelta(days=10):d0 + pd.Timedelta(days=10)]
            for nb in nbrs[r.city_id]:
                ax.plot(win.index, win[nb], color="0.7", lw=1)
            ax.plot(win.index, win[r.city_id], color="k", lw=1.2)
            ax.scatter([d0], [r.value], color="tab:red" if r.proposed_action == "delete" else "tab:blue", zorder=3)
            ax.set_title(f"{r.city_id} {d0.date()} {r.var} {r.value:g}\n{r.proposed_action}: z={r.z:.1f} "
                         f"resid={r.resid_f:.0f}°F", fontsize=7)
            ax.tick_params(labelsize=5)
            ax.set_xticks([])
        for ax in axes.flat[len(sub):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"isolated_outliers_{p + 1:02d}.png", dpi=70)
        plt.close(fig)
    return pages


# =============================================================================================
# Pipeline
# =============================================================================================
HASHED_CODE = ["src/protostar_challenge/cleaning.py", "scripts/02_clean_data.py", "src/protostar_challenge/config.py",
               "code/common.py", "code/combo_model.py"]


def reuse_hashes() -> dict:
    """Inputs, code, constants and the decisions file that a saved backtest choice depends on (§6 check 9)."""
    h = file_hashes([RAW_DIR / s for s in ["daily", "annual_indices", "city_info", "coverage_and_stations"]])
    h.update(file_hashes([ROOT / p for p in HASHED_CODE]))
    h.update(file_hashes([DECISIONS_PATH if DECISIONS_PATH.exists() else DECISIONS_SEED]))
    h["constants"] = hashlib.sha256(json.dumps(constants(), sort_keys=True, default=str).encode()).hexdigest()
    return h


def trim(B: dict) -> dict:
    """What a backtest fold keeps: the input variants and a few counts (H4)."""
    st = B["steps"]
    late = st[st["applied"] & (st["n_after"] < HOMOG_WINDOW)]
    return {"V": B["V"], "E": B["E"], "n_steps_applied": int(st["applied"].sum()),
            "n_steps_applied_short_after_window": int(len(late)),
            "qc_log": {k: int(v) for k, v in B["Q"]["log"].groupby("rule").size().items()}}


def precision_2018(W: dict) -> dict:
    """D5 evidence: share of decimals in 2018 and the GHCN round trip of the nearest whole °F."""
    out = {}
    for v in TEMP_VARS:
        x = W[v].loc["2018"].stack().dropna()
        dec = x[(x % 1).abs() > 1e-9]
        F = dec.round()
        rt = (((F - 32) / 1.8).round(1) * 1.8 + 32).round(1)
        out[v] = {"share_decimal": round(len(dec) / len(x), 4),
                  "share_round_trip": round(float((rt == dec.round(1)).mean()), 4),
                  "mean_decimal_minus_whole": round(float((dec - F).mean()), 4),
                  "max_abs": round(float((dec - F).abs().max()), 3)}
    return out


def explain_qc_differences(annual: pd.DataFrame, log: pd.DataFrame) -> dict:
    """§6 check 5: every QC value that differs from its label is in a city-year with a logged change."""
    touched = set(zip(log["city_id"], log["date"].dt.year))
    res = {}
    for m in METRICS:
        a, q = annual[m], annual[f"{m}_qc"]
        diff = ~((a.isna() & q.isna()) | (a == q))
        keys = set(zip(annual.loc[diff, "city_id"], annual.loc[diff, "year"]))
        unexplained = keys - touched
        if unexplained:
            raise ValueError(f"{m}: {len(unexplained)} QC differences without a logged change, "
                             f"e.g. {sorted(unexplained)[:3]}")
        res[m] = len(keys)
    return res


def run(skip_backtest: bool = False, verbose: bool = True) -> dict:
    """The whole pipeline: checks, bake-off, backtest folds, production build, outputs."""
    import time
    t0 = time.time()
    out_dir = PROCESSED_DIR
    QC_OUT.mkdir(parents=True, exist_ok=True)
    say("Loading the four subsets ...")
    R = load_raw()
    checks = {"D1 structure": R["structure"], "C1 city_info": check_city_info(R["city_info"], R["annual"]),
              "V1 coverage": check_coverage(R["coverage"], R["annual"], R["W"], R["record"], R["city_info"])}
    S = setup(R)
    checks["labels == annual_valid.csv (rows)"] = check_labels(S["labels"], ROOT / "data" / "annual_valid.csv")
    checks["audit"] = audit_rebuild(R["annual"], annual_from_daily(R["W"], R["record"]))
    checks["D5 2018 precision"] = precision_2018(R["W"])
    overrides = read_overrides()
    say(f"D6 fill bake-off on data <= {FILL_BAKEOFF_END} ...")
    with fold_cutoff(FILL_BAKEOFF_END):
        Q85 = quality_control(R, FILL_BAKEOFF_END, S["nbrs"], overrides, verbose)
        tm, pm, bake = fill_bakeoff(Q85["W"], S["weights"])
    del Q85
    methods = {"temperature": tm, "precipitation": pm}
    bake.to_csv(QC_OUT / "fill_bakeoff.csv", index=False)
    hashes = reuse_hashes()
    saved = out_dir / "backtest_inputs.json"
    if skip_backtest:
        if not saved.exists():
            raise SystemExit("--skip-backtest: no saved backtest_inputs.json; run without the flag")
        prev = json.loads(saved.read_text())
        stale = [k for k in set(hashes) | set(prev["hashes"]) if hashes.get(k) != prev["hashes"].get(k)]
        if stale:
            raise SystemExit(f"--skip-backtest: the saved choice is stale ({', '.join(sorted(stale))}); "
                             "run without --skip-backtest")
        ship = prev["ship"]
        say("Reusing the saved input choice (all hashes match).")
    else:
        errs, fold_info = [], {}
        for o in ORIGINS:
            say(f"E backtest fold: data <= {o} ...")
            with fold_cutoff(o):
                Bo = trim(build(R, S, o, overrides, methods, verbose))
            fold_info[o] = {k: v for k, v in Bo.items() if k != "V"}
            errs.append(backtest_errors({o: Bo}, S["labels"]))
            del Bo
        err = pd.concat(errs, ignore_index=True)
        say("E selecting inputs ...")
        sel = select_inputs(err, R["cities"])
        ship = sel["ship"]
        sel["skills"].to_csv(out_dir / "backtest_skills.csv", index=False)
        boots = []
        for m in METRICS:
            p = sel["production"][m]
            boots.append(pd.DataFrame({"metric": m, "forecaster": p["forecaster"], "candidate": list(p["G"]),
                                       "G": list(p["G"].values()),
                                       "lower": [p["lower"].get(c, np.nan) for c in p["G"]],
                                       "eligible": [c in p["eligible"] for c in p["G"]]}))
        pd.concat(boots).to_csv(out_dir / "backtest_bootstrap.csv", index=False)
        pd.concat(sel["sensitivity"], ignore_index=True).to_csv(out_dir / "backtest_sensitivity.csv", index=False)
        pd.DataFrame(sel["nested"]).to_csv(out_dir / "backtest_nested.csv", index=False)
        saved.write_text(json.dumps({"ship": ship, "hashes": hashes, "folds": fold_info,
                                     "production": sel["production"]}, indent=1, default=str))
    say(f"Production build: data <= {CUTOFF_YEAR} ...")
    B = build(R, S, CUTOFF_YEAR, overrides, methods, verbose)
    say("Writing outputs ...")
    dec = write_decisions(B["Q"]["candidates"], overrides)
    pages = plot_outliers(B["Q"]["candidates"], {v: R["W"][v] for v in TEMP_VARS}, S["nbrs"],
                          QC_OUT / "outlier_plots")
    log = B["Q"]["log"].sort_values(["city_id", "date", "var"])
    log.to_csv(QC_OUT / "daily_qc_log.csv", index=False)
    B["Q"]["stuck"].to_csv(QC_OUT / "stuck_runs.csv", index=False)
    B["Q"]["prcp_runs"].to_csv(QC_OUT / "repeated_precip_runs.csv", index=False)
    B["Q"]["fragments"].to_csv(QC_OUT / "fragments.csv", index=False)
    B["Q"]["breaks"].to_csv(QC_OUT / "record_breaks.csv", index=False)
    pext = precip_extremes(B["Q"]["W"], S["nbrs"])
    pext.to_csv(QC_OUT / "precip_extremes.csv", index=False)
    B["steps"].to_csv(QC_OUT / "homog_steps.csv", index=False)
    B["calib"].to_csv(QC_OUT / "homog_placebo_calibration.csv", index=False)
    annual = assemble_annual(R, S, B, ship)
    coverage = assemble_coverage(R, S, B)
    stations = assemble_stations(S, B)
    say("  infill evaluation ...")
    ev = []
    for m in METRICS:
        e = infill_evaluation(B["V"][m]["qc"], S["weights"], n=400)
        e["metric"] = m
        ev.append(e)
    ev = pd.concat(ev, ignore_index=True)
    ev.to_csv(QC_OUT / "infill_evaluation.csv", index=False)
    for t, name in [(annual, "annual_clean"), (coverage, "coverage_clean")]:
        assert_no_post_cutoff(t, "year", name)
        if len(t) != len(R["annual"]):
            raise ValueError(f"{name}: {len(t)} rows, expected {len(R['annual'])}")
    if len(stations) != len(R["city_info"]):
        raise ValueError("stations_clean: row count differs from city_info")
    explained = explain_qc_differences(annual, log)
    annual.to_parquet(out_dir / "annual_clean.parquet", index=False)
    coverage.to_parquet(out_dir / "coverage_clean.parquet", index=False)
    stations.to_parquet(out_dir / "stations_clean.parquet", index=False)
    say("  daily table ...")
    daily = assemble_daily(R, B)
    daily.to_parquet(out_dir / "daily_clean.parquet", index=False)
    st = B["steps"]
    manifest = {
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION, "cutoff": f"{CUTOFF_YEAR}-12-31"},
        "hashes": hashes, "constants": constants(), "checks": checks, "fill_methods": methods,
        "rows": {"daily_clean": len(daily), "annual_clean": len(annual), "coverage_clean": len(coverage),
                 "stations_clean": len(stations)},
        "qc_rules": {k: int(v) for k, v in log.groupby("rule").size().items()},
        "outlier_candidates": {"total": int(len(B["Q"]["candidates"])),
                               "isolated": int((B["Q"]["candidates"]["class"] == "isolated").sum()),
                               "deleted": int((dec["proposed_action"] == "delete").sum()),
                               "review_file": DECISIONS_PATH.relative_to(ROOT).as_posix(), "plot_pages": pages,
                               "precip_extremes_kept": int(len(pext))},
        "homogenisation": {"clusters": int(len(B["clusters"])),
                           "applied": {k: int(v) for k, v in st.groupby("var")["applied"].sum().items()},
                           "status": {f"{a} | {b}": int(n) for (a, b), n in st.groupby(["var", "status"]).size().items()},
                           "placebo_calibration": B["calib"].to_dict(orient="records")},
        "qc_differences_explained": explained,
        "infill_mae": {m: {c: round(float(g[c].abs().mean()), 4) for c in
                           ["past_only_infill", "past_only_baseline_prev15", "as_used_infill", "as_used_baseline_pm15"]}
                       for m, g in ev.groupby("metric")},
        "inputs_shipped": ship,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    say(f"Done in {manifest['runtime_seconds']:.0f} s. Inputs shipped: "
        + ", ".join(f"{m}={v['shipped']}" for m, v in ship.items()))
    return dict(R=R, S=S, B=B, annual=annual, coverage=coverage, stations=stations, manifest=manifest)
