"""One complete cleaned dataset per forecast origin, built only from data up to that origin.

Run through `scripts/03_clean_per_origin.py`. The step-by-step logic is `cleaning.py`, unchanged; this module
decides what each step may see. `truncate_raw` cuts every subset at 31 December of the origin, and everything
after it (neighbour QC, thresholds, outliers, station history, fill method, station adjustment, infill) runs on
that cut copy inside `fold_cutoff(origin)`. Years after the origin are therefore not reachable, not just unused.

Three things in the full-record build reach past an origin and are handled here:
    station history     empty periods and overlaps depend on later readings and end dates: rebuilt from the cut copy
    fill method         chosen on data <= 1985 in the full build: chosen again on data <= origin
    percentile indices  published with thresholds through 1987: blanked at origins before 1987

`<metric>_input` is not produced. The full build picks it with backtests running to 2018, and a past-only pick
for the 2004 and 2011 origins would score candidates on years held back from assessment. Every candidate series
is in the table, so the choice stays with the station that scores models. Page: documentation/09-per-origin.md.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from protostar_challenge import cleaning as cl
from protostar_challenge.config import CUTOFF_YEAR, DATASET_REPO, DATASET_REVISION, ROOT

ORIGINS = tuple(json.loads((ROOT / "evaluation" / "config.json").read_text(encoding="utf-8"))["origins"])
OUT_DIR = ROOT / "outputs" / "clean_per_origin"
PERCENTILE_THRESHOLDS_END = 1987
PERCENTILE_COLS = [f"{cl.snake(c)}_pct_valid" for c in cl.PCT_COLS] + ["wet_days_in"]
DATE_COLS = ("date", "start", "end", "f", "l")
YEAR_COLS = ("year", "first", "last", "next_block_starts")
UNDATED_TABLES = ("fill_bakeoff.csv", "homog_placebo_calibration.csv")


def truncate_raw(R: dict, origin: int) -> dict:
    """Every subset as it stood on 31 December of `origin`. Stations starting later vanish; open ones end there."""
    end = pd.Timestamp(f"{origin}-12-31")
    stations = R["city_info"][R["city_info"]["Stn.stDate"] <= end].copy()
    stations["Stn.edDate"] = stations["Stn.edDate"].clip(upper=end)
    return {"W": {v: m.loc[:end] for v, m in R["W"].items()},
            "record": R["record"].loc[:end],
            "annual": R["annual"][R["annual"]["Year"] <= origin].reset_index(drop=True),
            "coverage": R["coverage"][R["coverage"]["Year"] <= origin].reset_index(drop=True),
            "city_info": stations.reset_index(drop=True),
            "cities": R["cities"]}


def _guard(R: dict, origin: int) -> None:
    """Inside fold_cutoff(origin): the standard data-rule guard on every table the build receives."""
    cl.assert_no_post_cutoff(pd.DataFrame({"Year": R["W"]["tmax"].index.year.unique()}), "Year", f"daily {origin}")
    cl.assert_no_post_cutoff(R["annual"], "Year", f"annual_indices {origin}")
    cl.assert_no_post_cutoff(R["coverage"], "Year", f"coverage_and_stations {origin}")
    ends = R["city_info"]["Stn.edDate"].dt.year
    cl.assert_no_post_cutoff(pd.DataFrame({"Year": ends}), "Year", f"city_info {origin}")


def blank_percentile_indices(annual: pd.DataFrame, origin: int) -> pd.DataFrame:
    """The published Warm/Cold and Wet Days indices use thresholds through 1987 (plan A4/A5)."""
    if origin >= PERCENTILE_THRESHOLDS_END:
        return annual
    return annual.assign(**{c: np.nan for c in PERCENTILE_COLS})


def build_origin(full: dict, origin: int, overrides: pd.DataFrame) -> dict:
    """Every output table for one origin. `full` is cut before anything else touches it."""
    R = truncate_raw(full, origin)
    with cl.fold_cutoff(origin):
        _guard(R, origin)
        reported = R["annual"]["ID"].unique()
        known = R["city_info"][R["city_info"]["ID"].isin(reported)]
        checks = {"V1 coverage": cl.check_coverage(R["coverage"], R["annual"], R["W"], R["record"], known)}
        S = cl.setup(R)
        cl.say(f"[{origin}] choosing the fill method on data <= {origin}")
        Q = cl.quality_control(R, origin, S["nbrs"], overrides, verbose=False)
        temperature, precipitation, bakeoff = cl.fill_bakeoff(Q["W"], S["weights"], end_year=origin)
        del Q
        methods = {"temperature": temperature, "precipitation": precipitation}
        cl.say(f"[{origin}] QC, station adjustment, fill, infill")
        B = cl.build(R, S, origin, overrides, methods, verbose=False)
        annual = cl.assemble_annual(R, S, B, {m: {"shipped": "official"} for m in cl.METRICS})
        annual = annual.drop(columns=[c for c in annual.columns if c.endswith(("_input", "_input_source", "_input_variant"))])
        annual = blank_percentile_indices(annual, origin)
        coverage = cl.assemble_coverage(R, S, B)
        stations = cl.assemble_stations(S, B)
        log = B["Q"]["log"].sort_values(["city_id", "date", "var"])
        checks["QC differences explained"] = cl.explain_qc_differences(annual, log)
        if len(annual) != len(R["annual"]) or len(coverage) != len(R["annual"]) or len(stations) != len(R["city_info"]):
            raise ValueError(f"{origin}: a table has the wrong number of rows")
        cl.assert_no_post_cutoff(annual, "year", f"annual_clean {origin}")
        cl.assert_no_post_cutoff(coverage, "year", f"coverage_clean {origin}")
        daily = cl.assemble_daily(R, B)
        extremes = cl.precip_extremes(B["Q"]["W"], S["nbrs"])
    steps = B["steps"]
    return {"origin": origin, "annual": annual, "coverage": coverage, "stations": stations, "daily": daily, "log": log,
            "candidates": B["Q"]["candidates"], "overrides": overrides,
            "qc": {"daily_qc_log.csv": log, "stuck_runs.csv": B["Q"]["stuck"],
                   "repeated_precip_runs.csv": B["Q"]["prcp_runs"], "fragments.csv": B["Q"]["fragments"],
                   "record_breaks.csv": B["Q"]["breaks"], "precip_extremes.csv": extremes, "homog_steps.csv": steps,
                   "homog_placebo_calibration.csv": B["calib"], "fill_bakeoff.csv": bakeoff},
            "summary": {"fill_methods": methods, "checks": checks,
                        "qc_rules": {k: int(v) for k, v in log.groupby("rule").size().items()},
                        "outlier_candidates": {"total": len(B["Q"]["candidates"]),
                                               "isolated": int((B["Q"]["candidates"]["class"] == "isolated").sum())},
                        "homogenisation": {"clusters": len(B["clusters"]),
                                           "applied": {k: int(v) for k, v in steps.groupby("var")["applied"].sum().items()}}}}


def latest_year(table: pd.DataFrame) -> float:
    """The latest calendar year named by any time column; NaN if the table has no time column or no rows."""
    years = [pd.to_datetime(table[c]).dt.year for c in DATE_COLS if c in table.columns]
    years += [pd.to_numeric(table[c]) for c in YEAR_COLS if c in table.columns]
    return max((float(y.max()) for y in years if len(y)), default=np.nan)


def write_origin(built: dict, out_dir: Path, hashes: dict) -> Path:
    """origin-YYYY/ with the same tables as the full build, each checked to end by the origin."""
    origin = built["origin"]
    folder = Path(out_dir) / f"origin-{origin}"
    (folder / "qc").mkdir(parents=True, exist_ok=True)
    tables = {"annual_clean": built["annual"], "coverage_clean": built["coverage"], "stations_clean": built["stations"],
              "daily_clean": built["daily"]}
    for name, table in {**tables, **built["qc"]}.items():
        last = latest_year(table)
        if last > origin or (not len(table) == 0 and np.isnan(last) and name not in UNDATED_TABLES):
            raise ValueError(f"origin {origin}: {name} reaches {last} or has no time column")
    for name, table in tables.items():
        table.to_parquet(folder / f"{name}.parquet", index=False)
    for name, table in built["qc"].items():
        table.to_csv(folder / "qc" / name, index=False)
    cl.write_decisions(built["candidates"], built["overrides"], folder / "qc" / "isolated_outliers.csv")
    manifest = {"origin": origin, "data_through": f"{origin}-12-31", "cutoff_of_dataset": f"{CUTOFF_YEAR}-12-31",
                "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION}, "hashes": hashes,
                "constants": cl.constants(), "rows": {k: len(v) for k, v in tables.items()},
                "percentile_indices_blanked": origin < PERCENTILE_THRESHOLDS_END, **built["summary"]}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str, sort_keys=True), encoding="utf-8")
    return folder


def run(origins: tuple[int, ...] = ORIGINS, out_dir: Path = OUT_DIR) -> dict:
    started = time.time()
    cl.say("Loading the four subsets ...")
    full = cl.load_raw()
    overrides = cl.read_overrides()
    hashes = cl.reuse_hashes()
    hashes["src/protostar_challenge/cleaning_per_origin.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    folders = {}
    for origin in origins:
        begun = time.time()
        built = build_origin(full, origin, overrides)
        folders[origin] = write_origin(built, out_dir, hashes)
        cl.say(f"[{origin}] done in {time.time() - begun:.0f} s: {built['summary']['fill_methods']}, "
               f"{built['summary']['homogenisation']['applied']} station steps applied")
        del built
    cl.say(f"All {len(folders)} origins done in {time.time() - started:.0f} s -> {out_dir}")
    return folders
