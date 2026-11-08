"""Placeholder_v1.1: the 30-year average plus a centred graph-local term (plan A r6.2, section 1.2).

    forecast(c, m, O + h) = clip_m( ref30 + G ),   G = beta_m * centre(L_raw)

The cities are nodes of a fixed Gaussian graph (600 km). A is a city's anomaly from its 30-year mean, R = A minus
the kernel mean of the neighbours' A, and L_raw = EWMA(R) - mean(R) over the last 30 years (half-life 3, or 6 for
precipitation). centre() removes the median over defined cities and sets undefined cities to 0, so G has zero
cross-city median adjustment. beta_m is the first minimum of the mean absolute error over the grid 0-1 (step 0.05)
on pseudo-origins O-45..O-4 and leads 4-7 with t + h <= O. G is the same for all seven leads. clip_m = max(0, .)
for precipitation and CDD. ref30 and the rows without a 30-year mean come from reference(..., "average").

Reference implementation: docs/plans/creative/analysis/planA_walkforward.py; the model functions here are copies
of it, and docs/plans/creative/analysis/planA_parity.py checks that the forecasts match. Inputs: the harness
train table and data/city_info.csv (lat/lon) under PROTOSTAR_ASSET_ROOT, declared in
notebooks/Meet/placeholder-v0.1-assets.json. No feature or enhancement release and no daily data are read.

Single file on purpose: the runtime audit allows the entry point by exact hash, and helpers outside code/ and src/
would need a declared source asset. v1.1-v1.4 share this body; only the version block differs.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common import METRICS, assert_no_post_cutoff

# ---------------------------------------------------------------------------------------------
# Version block (the only lines that differ between v1.1, v1.2, v1.3 and v1.4)
# ---------------------------------------------------------------------------------------------
MODEL_NAME = "Placeholder_v1.1"
USE_LONG = False   # v1.2, v1.4: G = centre(beta L_raw + gamma L_long_raw), joint fit
NATIONAL = False   # v1.3, v1.4: + national level increment I, coefficients refitted with I in the base

# ---------------------------------------------------------------------------------------------
# Frozen by the plan (sections 2, 3.1 and 4.1). Nothing here is chosen from a development score.
# ---------------------------------------------------------------------------------------------
METRIC_ORDER = list(METRICS)
NON_NEGATIVE = {"total_precip_in", "cdd"}
HALF_LIFE = {"annual_avg_temp": 3, "hottest_day_temp": 3, "cdd": 3, "total_precip_in": 6}
KERNEL_KM = 600.0
EARTH_RADIUS_KM = 6371.0
MIN_KERNEL_MASS = 1.0
MIN_VALID_WINDOW = 10            # ref30 needs >= 10 valid years in t-29..t (the official baseline rule)
MIN_REL_YEARS = 10               # L_raw needs >= 10 valid R years in t-29..t
MIN_LONG_YEARS = 15              # L_long_raw needs >= 15 valid R years in t-59..t-30
GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)
FIT_LEADS = (4, 5, 6, 7)
TRAIN_YEARS = 45                 # pseudo-origins t in [O-45, O-4]
MIN_COEF_ROWS = 2000
KAPPA = 0.5                      # national increment: shrink of the floored, capped national 10-year gap
CAP_F = 1.0                      # amplitude bound on N (F); the temperature increment is at most 0.5 F
GAP_MIN_VALID = 5                # valid years needed in t-9..t for a city's 10-year gap
RHO_NEIGHBOURS = 15              # rho: CDD-on-temperature slope averaged over the city and its 15 nearest
MIN_RESPONSE_YEARS = 15          # rho: joint valid years needed in t-29..t
LOOKBACK = 110                   # NaN columns before the first year: the oldest window starts at (O - 45) - 59
KEY = ["city_id", "year", "metric"]
PLAN = "docs/plans/creative/plan-A-graph.md (draft r6.2)"


# ---------------------------------------------------------------------------------------------
# Geography (coordinates only from the declared static asset)
# ---------------------------------------------------------------------------------------------
def load_coords(cities) -> pd.DataFrame:
    """Latitude and longitude from the declared snapshot; the raw data/ path is deliberately not used."""
    root = os.environ.get("PROTOSTAR_ASSET_ROOT")
    if not root:
        raise RuntimeError("PROTOSTAR_ASSET_ROOT is not set: run through the harness with the asset manifest")
    table = pd.read_csv(Path(root) / "data" / "city_info.csv", dtype={"city_id": str}, usecols=["city_id", "lat", "lon"])
    return _check_coords(table.set_index("city_id"), cities)


def _check_coords(table: pd.DataFrame, cities) -> pd.DataFrame:
    coords = table.reindex(list(cities))[["lat", "lon"]]
    if coords.isna().any(axis=None):
        raise ValueError("city coordinates are missing for some forecast cities")
    return coords


def distances(coords: pd.DataFrame) -> np.ndarray:
    """Haversine distances in km."""
    lat, lon = np.radians(coords.lat.to_numpy(float)), np.radians(coords.lon.to_numpy(float))
    a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
         + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def graph(coords: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian kernel K_ij = exp(-(d_ij / 600 km)^2) with K_ii = 0, and each city's other cities by distance."""
    if list(coords.index) != sorted(coords.index):
        raise ValueError("coordinates must be sorted by city_id so distance ties break by city_id")
    d = distances(coords)
    kernel = np.exp(-(d / KERNEL_KM) ** 2)
    np.fill_diagonal(kernel, 0.0)
    dn = d.copy()
    np.fill_diagonal(dn, np.inf)
    order = np.argsort(dn, axis=1, kind="stable")
    return kernel, order


# ---------------------------------------------------------------------------------------------
# City x year panel. Everything below receives only columns <= the (pseudo-)origin it works at.
# ---------------------------------------------------------------------------------------------
class Panel:
    """One city x year array per metric, columns y0..last. Years after `last` are not stored."""

    def __init__(self, arrays: dict[str, np.ndarray], y0: int, last: int):
        self.arrays, self.y0, self.last = arrays, y0, last

    def col(self, year: int) -> int:
        if year < self.y0 or year > self.last:
            raise IndexError(f"year {year} is outside the panel {self.y0}..{self.last}")
        return year - self.y0


def build_panel(history: pd.DataFrame, cities, last_year: int) -> Panel:
    """Cut at last_year before anything else reads the table; years before the data are NaN padding."""
    frame = history[history.year <= last_year]
    assert_no_post_cutoff(frame, "year", f"{MODEL_NAME} history")
    if frame.empty:
        raise ValueError(f"no training history up to {last_year}")
    y0 = int(frame.year.min()) - LOOKBACK
    years = range(y0, last_year + 1)
    arrays = {m: frame.pivot(index="city_id", columns="year", values=m).reindex(index=list(cities), columns=years)
              .to_numpy(float) for m in METRIC_ORDER}
    return Panel(arrays, y0, last_year)


def nmean(x, axis=1):
    ok = np.isfinite(x)
    c = ok.sum(axis)
    return np.where(c > 0, np.where(ok, x, 0.0).sum(axis) / np.maximum(c, 1), np.nan)


def ref30(arr, c):
    w = arr[:, c - 29: c + 1]
    return np.where(np.isfinite(w).sum(1) >= MIN_VALID_WINDOW, nmean(w), np.nan)


def neighbour_mean(k, a):
    ok = np.isfinite(a)
    num = (k[:, :, None] * np.where(ok, a, 0.0)[None]).sum(1)      # broadcast sums: thread-independent order
    den = (k[:, :, None] * ok[None]).sum(1)
    return np.where(den >= MIN_KERNEL_MASS, num / np.where(den > 0, den, 1.0), np.nan)


def terms_at(panel: Panel, metric: str, t: int, kernel: np.ndarray):
    """ref30, L_raw and L_long_raw at pseudo-origin t, from columns <= t only."""
    c = panel.col(t)
    if c - 59 < 0:
        raise ValueError("panel padding does not cover the 60-year window")
    hist = panel.arrays[metric][:, : c + 1]                            # nothing after t is visible
    ref = ref30(hist, c)
    x = hist[:, c - 59: c + 1]
    a = x - ref[:, None]
    r = np.where(np.isfinite(x), a - neighbour_mean(kernel, a), np.nan)
    r30 = r[:, 30:]
    ok = np.isfinite(r30)
    nv = ok.sum(1)
    w = 0.5 ** (np.arange(29, -1, -1) / HALF_LIFE[metric])
    ewma = (np.where(ok, r30, 0.0) * w).sum(1) / np.maximum((ok * w).sum(1), 1e-12)
    m30 = np.where(ok, r30, 0.0).sum(1) / np.maximum(nv, 1)
    l_raw = np.where(nv >= MIN_REL_YEARS, ewma - m30, np.nan)
    ok_all = np.isfinite(r)
    m60 = np.where(ok_all, r, 0.0).sum(1) / np.maximum(ok_all.sum(1), 1)
    l_long = np.where((ok_all[:, :30].sum(1) >= MIN_LONG_YEARS) & (nv >= MIN_REL_YEARS), m60 - m30, np.nan)
    return ref, l_raw, l_long


def cdd_response(panel: Panel, t: int, order: np.ndarray) -> np.ndarray:
    """rho: OLS slope of annual CDD on annual mean temperature over t-29..t (>= 15 joint valid years), averaged
    over the city and its 15 nearest, NaN to 0, floored at 0. Columns <= t only."""
    c = panel.col(t)
    tw = panel.arrays["annual_avg_temp"][:, c - 29: c + 1]
    cw = panel.arrays["cdd"][:, c - 29: c + 1]
    ok = np.isfinite(tw) & np.isfinite(cw)
    nj = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = np.where(ok, tw, 0).sum(1) / nj
        ym = np.where(ok, cw, 0).sum(1) / nj
        dx = np.where(ok, tw - xm[:, None], 0)
        dy = np.where(ok, cw - ym[:, None], 0)
        resp = (dx * dy).sum(1) / (dx ** 2).sum(1)
    resp = np.where((nj >= MIN_RESPONSE_YEARS) & np.isfinite(resp), resp, np.nan)
    return np.maximum(np.nan_to_num(nmean(np.column_stack([resp, resp[order[:, :RHO_NEIGHBOURS]]]))), 0.0)


def national_gap(arr, c, ref):
    """N: cross-city median of (mean of valid years in t-9..t) - ref30(t); cities need >= 5 valid years and a ref30."""
    w = arr[:, c - 9: c + 1]
    gap = np.where((np.isfinite(w).sum(1) >= GAP_MIN_VALID) & np.isfinite(ref), nmean(w) - ref, np.nan)
    return float(np.nanmedian(gap)) if np.isfinite(gap).any() else 0.0


def national_shift(n_gap, rho, kappa, cap, floor):
    """Uniform temperature increment and its CDD conversion through rho. Returns per-metric arrays."""
    g = n_gap
    if floor:
        g = max(g, 0.0)
    if cap is not None:
        g = min(g, cap)
    inc = kappa * g
    zero = np.zeros_like(rho)
    return {"annual_avg_temp": np.full_like(rho, inc), "cdd": inc * rho, "total_precip_in": zero,
            "hottest_day_temp": zero}, inc


def precompute(panel: Panel, kernel, order, ts, kappa: float) -> dict:
    """Graph terms (and v1.3/v1.4's national increment) at every pseudo-origin t in ts, each from columns <= t."""
    pre = {}
    for t in ts:
        terms = {m: terms_at(panel, m, t, kernel) for m in METRIC_ORDER}
        entry = {}
        if NATIONAL:
            rho = cdd_response(panel, t, order)
            n_gap = national_gap(panel.arrays["annual_avg_temp"], panel.col(t), terms["annual_avg_temp"][0])
            nat, inc = national_shift(n_gap, rho, kappa, CAP_F, floor=True)
            entry.update(n_gap=n_gap, inc=inc, nat=nat)
        for m in METRIC_ORDER:
            ref, l_raw, l_long = terms[m]
            entry[m] = (ref, l_raw, l_long, ref + entry["nat"][m] if NATIONAL else ref)
        pre[t] = entry
    return pre


# ---------------------------------------------------------------------------------------------
# G, deploy and the coefficient fit (copies of planA_walkforward)
# ---------------------------------------------------------------------------------------------
def graph_shift(beta, gamma, l_raw, l_long, use_long):
    """G = centre(beta L_raw [+ gamma L_long_raw]): median over defined cities removed, undefined -> 0. Row-wise.

    use_long False: defined where L_raw is defined.  use_long True: defined where either part is.
    """
    defined = np.isfinite(l_raw) | (np.isfinite(l_long) if use_long else False)
    v = beta * np.nan_to_num(l_raw) + (gamma * np.nan_to_num(l_long) if use_long else 0.0)
    v = np.where(defined, v, np.nan)
    med = np.nanmedian(v, axis=-1, keepdims=True)
    return np.where(defined, v - med, 0.0)


def deploy(metric, base, g):
    """The only forecast formula, used in the fit and the forecast: base + G, clipped at 0 for precipitation and CDD."""
    f = base + g
    return np.maximum(f, 0.0) if metric in NON_NEGATIVE else f


def coef_rows(panel: Panel, pre: dict, metric: str, fit_origin: int):
    arr = panel.arrays[metric]
    ys, bs, ls, lls, target_years = [], [], [], [], []
    for t in range(fit_origin - TRAIN_YEARS, fit_origin - FIT_LEADS[0] + 1):
        if t not in pre:
            continue
        _, l_raw, l_long, base = pre[t][metric]
        for h in FIT_LEADS:
            if t + h > fit_origin:                         # the cutoff: no training target after the fit origin
                continue
            ys.append(arr[:, panel.col(t + h)])
            bs.append(base)
            ls.append(l_raw)
            lls.append(l_long)
            target_years.append(t + h)
    return np.array(ys), np.array(bs), np.array(ls), np.array(lls), target_years


def fit_coefs(panel: Panel, pre: dict, metric: str, fit_origin: int):
    """(beta, gamma, rows): first minimum of mean |y - deploy(base, G)| over the ascending grid; 0 under 2,000 rows."""
    y, b, l_raw, l_long, target_years = coef_rows(panel, pre, metric, fit_origin)
    assert target_years and max(target_years) <= fit_origin, f"training target after fit origin {fit_origin}"
    y = np.where(np.isfinite(b), y, np.nan)
    ok = np.isfinite(y)
    rows = int(ok.sum())
    if rows < MIN_COEF_ROWS:
        return 0.0, 0.0, rows
    best = (np.inf, 0.0, 0.0)
    for gamma in (GRID if USE_LONG else [0.0]):
        for beta in GRID:
            e = np.abs(y - deploy(metric, b, graph_shift(beta, gamma, l_raw, l_long, USE_LONG)))
            loss = np.where(ok, e, 0.0).sum() / ok.sum()
            if loss < best[0] - 1e-15:                     # ascending scan: ties keep the smallest coefficients
                best = (loss, beta, gamma)
    return float(best[1]), float(best[2]), rows


def fit_at(panel: Panel, kernel, order, fit_origin: int, kappa: float = KAPPA, coefs: dict | None = None) -> dict:
    """Coefficients fitted with targets <= fit_origin, and G and I at fit_origin, per metric.

    coefs ({metric: (beta, gamma)}) replaces the fit; it exists for the equality checks in the tests only.
    """
    if fit_origin > panel.last:
        raise ValueError("fit origin is after the panel")
    state = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # nanmedian of an all-NaN row is NaN by design
        pre = precompute(panel, kernel, order, range(fit_origin - TRAIN_YEARS, fit_origin + 1), kappa)
        for m in METRIC_ORDER:
            if coefs is None:
                beta, gamma, rows = fit_coefs(panel, pre, m, fit_origin)
            else:
                (beta, gamma), rows = coefs[m], None
            ref, l_raw, l_long, _ = pre[fit_origin][m]
            has_ref = np.isfinite(ref)
            defined = np.isfinite(l_raw) | (np.isfinite(l_long) if USE_LONG else False)
            v = beta * np.nan_to_num(l_raw) + (gamma * np.nan_to_num(l_long) if USE_LONG else 0.0)
            increment = np.where(has_ref, pre[fit_origin]["nat"][m], 0.0) if NATIONAL else np.zeros(len(ref))
            state[m] = {"beta": float(beta), "gamma": float(gamma), "fit_rows": rows, "ref": ref, "has_ref": has_ref,
                        "defined": defined, "G": graph_shift(beta, gamma, l_raw, l_long, USE_LONG), "I": increment,
                        "centring_median": float(np.median(v[defined])) if defined.any() else float("nan")}
    if NATIONAL:
        state["national"] = {"N": float(pre[fit_origin]["n_gap"]), "I_temp": float(pre[fit_origin]["inc"])}
    return state


# ---------------------------------------------------------------------------------------------
# Assembly, record and the harness entry point
# ---------------------------------------------------------------------------------------------
def assemble(base: pd.DataFrame, state: dict, cities) -> tuple[pd.DataFrame, dict]:
    """ref30 rows: deploy(ref30 + I, G) on every lead. Rows without a 30-year mean: reference()'s fallback as is."""
    index = {city: i for i, city in enumerate(cities)}
    parts, median_shift = [], {}
    for m in METRIC_ORDER:
        rows = base[base.metric == m].copy()
        pos = rows.city_id.map(index).to_numpy(int)
        s = state[m]
        value = rows.predicted_value.to_numpy(float)
        has_ref = s["has_ref"][pos]
        out = np.where(has_ref, deploy(m, value + s["I"][pos], s["G"][pos]), value)
        rows["predicted_value"] = out
        parts.append(rows[KEY + ["predicted_value"]])
        first = (rows.year == rows.year.min()).to_numpy() & has_ref
        median_shift[m] = float(np.median(out[first] - value[first])) if first.any() else float("nan")
    return pd.concat(parts, ignore_index=True), median_shift


def _num(x):
    x = float(x)
    return x if np.isfinite(x) else None


def selection_record(state: dict, median_shift: dict, origin: int, seed: int, kappa: float, coefs) -> dict:
    metrics = {}
    for m in METRIC_ORDER:
        s = state[m]
        has = s["has_ref"]
        g = s["G"][has]
        entry = {"beta": s["beta"]}
        if USE_LONG:
            entry["gamma"] = s["gamma"]
        entry.update(fit_rows=s["fit_rows"], cities_with_ref30=int(has.sum()), fallback_cities=int((~has).sum()),
                     graph_defined_cities=int(s["defined"].sum()),
                     mean_abs_G=_num(np.mean(np.abs(g))) if g.size else None,
                     share_G_nonzero=_num(np.mean(g != 0.0)) if g.size else None,
                     centring_median=_num(s["centring_median"]),
                     median_forecast_shift=_num(median_shift[m]))
        if NATIONAL:
            entry["mean_increment"] = _num(np.mean(s["I"][has])) if has.any() else None
        metrics[m] = entry
    record = {"model": MODEL_NAME, "plan": PLAN, "outer_origin": int(origin), "seed": int(seed),
              "status": "fitted" if coefs is None else "coefficients_overridden",
              "graph_long_term": USE_LONG, "national_increment": NATIONAL, "metrics": metrics,
              "constants": {"kernel_km": KERNEL_KM, "min_kernel_mass": MIN_KERNEL_MASS, "half_life": HALF_LIFE,
                            "min_rel_years": MIN_REL_YEARS, "min_long_years": MIN_LONG_YEARS,
                            "grid": [float(v) for v in GRID], "fit_leads": list(FIT_LEADS),
                            "train_years": TRAIN_YEARS, "min_coef_rows": MIN_COEF_ROWS}}
    if NATIONAL:
        nat = state["national"]
        record["national"] = {"N": _num(nat["N"]), "I_temp": _num(nat["I_temp"]), "cap_bound": bool(nat["N"] > CAP_F),
                              "mean_cdd_increment": metrics["cdd"]["mean_increment"], "kappa": float(kappa),
                              "cap_F": CAP_F, "floor": 0.0, "gap_min_valid": GAP_MIN_VALID,
                              "rho_neighbours": RHO_NEIGHBOURS, "min_response_years": MIN_RESPONSE_YEARS}
    return record


def run_forecast(train, targets, origin, seed, coords, coefs=None, kappa=KAPPA):
    """Forecast given coordinates; `forecast` adds the asset load and writes the artifact."""
    from evaluation.adapters import reference

    started = time.perf_counter()
    assert_no_post_cutoff(train, "year", f"{MODEL_NAME} training input")
    if train.year.max() > origin:
        raise ValueError(f"{MODEL_NAME} received observations after its origin")
    if not NATIONAL and kappa != KAPPA:
        raise ValueError(f"{MODEL_NAME} has no national increment")
    cities = sorted(targets.city_id.unique())
    history = train.copy()
    history["year"] = pd.to_numeric(history.year).astype(int)
    kernel, order = graph(_check_coords(coords, cities))
    panel = build_panel(history, cities, origin)
    state = fit_at(panel, kernel, order, origin, kappa, coefs)
    base = reference(history, targets, origin, seed, method="average")
    output, median_shift = assemble(base, state, cities)
    record = selection_record(state, median_shift, origin, seed, kappa, coefs)
    record["runtime_s"] = round(time.perf_counter() - started, 3)
    return output, record, state


def forecast(train, targets, origin, seed=42, **params):
    if params:
        raise ValueError(f"{MODEL_NAME} takes no tunable parameters; unknown keys: {sorted(params)}")
    started = time.perf_counter()
    cities = sorted(targets.city_id.unique())
    output, record, _ = run_forecast(train, targets, origin, seed, load_coords(cities))
    record["runtime_s_total"] = round(time.perf_counter() - started, 3)
    directory = os.environ.get("PROTOSTAR_ARTIFACT_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "placeholder_selection.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n",
                                                         encoding="utf-8")
    return output
