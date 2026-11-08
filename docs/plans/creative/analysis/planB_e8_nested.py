"""Plan B r3, E8: retrospective, fully nested evaluation of the version tree on outer origins with targets <= 1983.

usage: uv run python docs/plans/creative/analysis/planB_e8_nested.py CONFIG [--origins 1948,1955,1962,1969,1976] [--out DIR]
Reads only data/annual_valid.csv (cut at 1983) and data/city_info.csv; imports placeholder_v0_5 and harness functions
unchanged. Writes row errors to DIR (default %TEMP%/planB_e8) and intermediate checks to DIR/CONFIG-checks.json.
Every outcome-selected quantity at a fit origin F (phi, lambda) uses targets <= F; asserted below.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE), str(ROOT / "code"), str(ROOT / "notebooks" / "Meet")]
os.environ.pop("PROTOSTAR_FOLD_CUTOFF", None)
import planB_core as C  # noqa: E402
import placeholder_v0_5 as P  # noqa: E402
from evaluation.adapters import reference  # noqa: E402
from evaluation.core import build_fold  # noqa: E402
from evaluation.official_score import score  # noqa: E402

CUTOFF = 1983
MET = list(P.METRICS)
KEY = ["city_id", "year", "metric"]
PHI_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PHI_TIE = 0.002
COMBO_GETS_W = ("total_precip_in", "hottest_day_temp")


def S(W, cw, learn):
    return {"W": W, "cw": cw, "learn": learn}


V20 = {"annual_avg_temp": S("v03", 0.5, True), "cdd": S("v03", 0.5, True),
       "hottest_day_temp": S("none", 0.5, True), "total_precip_in": S("none", 0.5, True)}


def derive(parent, **changes):
    out = {m: dict(s) for m, s in parent.items()}
    for m, ch in changes.items():
        out[m].update(ch)
    return out


# r3 version tree (plan section 5.3). Names are the plan's version numbers.
T21 = derive(V20, annual_avg_temp={"W": "core"}, cdd={"W": "core"})                       # v2.1 core
T22 = derive(V20, annual_avg_temp={"W": "core", "cw": 0.0, "learn": False},
             cdd={"W": "core", "cw": 0.0, "learn": False})                                 # v2.2 core-only (conditional)
T23 = derive(T21, annual_avg_temp={"dep": True})                                           # v2.3 departure target, temperature
T23G = derive(T21, annual_avg_temp={"dep": True, "gate_rel": True})                       # v2.3 as declared in r3
T21G = derive(T21, annual_avg_temp={"gate_rel": True})                                     # diagnostic: relative gate only
T24 = derive(T23, hottest_day_temp={"W": "hot"})                                           # v2.4 hottest-day level
T24G = derive(T23G, hottest_day_temp={"W": "hot"})
T25G = derive(T24G, total_precip_in={"W": "prcp"})
T25 = derive(T24, total_precip_in={"W": "prcp"})                                           # v2.5 precipitation base
# r2-order configurations, kept as diagnostics
V21 = derive(V20, total_precip_in={"W": "prcp"})
V22 = derive(V21, annual_avg_temp={"W": "core"}, cdd={"W": "core"})
V23 = derive(V21, annual_avg_temp={"W": "core", "cw": 0.0, "learn": False}, cdd={"W": "core", "cw": 0.0, "learn": False})
V24A = derive(V22, hottest_day_temp={"W": "hot"})
V24B = derive(V23, hottest_day_temp={"W": "hot"})
V25A = derive(V24A, **{m: {"cw": 0.0} for m in MET})
V25B = derive(V24B, **{m: {"cw": 0.0} for m in MET})
CONFIGS = {"v05": None, "v20": V20, "t21": T21, "t22": T22, "t23": T23, "t24": T24, "t25": T25,
           "t21g": T21G, "t23g": T23G, "t24g": T24G, "t25g": T25G,
           "v21": V21, "v22": V22, "v23": V23, "v24a": V24A, "v24b": V24B, "v25a": V25A, "v25b": V25B}
DEP_NAMES = [f"dep{k}" for k in range(15)] + ["dep5", "dep15", "slope_diff", "sd30"]

ap = argparse.ArgumentParser()
ap.add_argument("config", choices=sorted(CONFIGS))
ap.add_argument("--origins", default="1948,1955,1962,1969,1976")
ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "planB_e8"))
ap.add_argument("--anchor-step", type=int, default=C.ANCHOR_STEP, choices=(5, 10))
args = ap.parse_args()
SET = CONFIGS[args.config]
OUTER = [int(x) for x in args.origins.split(",")]
OUT = Path(args.out)

annual = pd.read_csv(ROOT / "data" / "annual_valid.csv", dtype={"city_id": str}).drop(columns="name")
annual = annual[annual.year <= CUTOFF].reset_index(drop=True)
assert annual.year.max() <= CUTOFF
coords = pd.read_csv(ROOT / "data" / "city_info.csv", dtype={"city_id": str}).set_index("city_id")[["lat", "lon"]]
CITIES = sorted(annual.city_id.unique())
NEIGH = P.neighbour_index(P._check_coords(coords, CITIES))
assert (NEIGH == C.neighbours(coords.loc[CITIES].lat, coords.loc[CITIES].lon)).all()
NBS = np.column_stack([np.arange(len(CITIES)), NEIGH])
RHO = C.RhoCache(step=args.anchor_step)
CHECKS = {"config": args.config, "settings": SET, "anchor_step": args.anchor_step, "outer": {}}


def panel(table):
    return table.to_numpy(float), int(table.columns.min())


V03_CACHE, W_CACHE = {}, {}


def v03_W(tables, t, m):
    if t not in V03_CACHE:
        V03_CACHE[t] = P.build_states(tables, NEIGH, [t])[t]
    st = V03_CACHE[t]
    W = np.column_stack([P.warming_term(st["warming"][m], st["metric"][m]["mean_valid_year"], t, h) for h in C.LEADS])
    return np.where(np.isfinite(st["metric"][m]["ref30"])[:, None], W, 0.0)


def W_at(tables, t, phi):
    """Corrections per metric at pseudo-origin t from columns <= t (cached: inputs are the history cut at t)."""
    key = (t, phi)
    if key not in W_CACHE:
        out = {}
        for m in MET:
            mode = SET[m]["W"]
            M, y0 = panel(tables[m].loc[:, :t])
            if mode == "none":
                out[m] = np.zeros((len(CITIES), 7))
            elif mode == "v03":
                out[m] = v03_W(tables, t, m)
            elif mode == "prcp":
                out[m] = C.precip_W(M, y0, t)
            else:
                rho = RHO.get(m, M, y0, t, NBS, y0)
                out[m] = C.core_W(M, y0, t, rho, phi, m, NBS, NEIGH)[0]
        W_CACHE[key] = out
    return W_CACHE[key]


def W_frame(W, F):
    return {m: pd.DataFrame(W[m], index=CITIES, columns=range(F + 1, F + 8)) for m in MET}


DEP_CACHE = {}


def dep_features(table, t, m):
    """Departure-history features at pseudo-origin t (columns <= t): 15 yearly departures of (value - ref30) from the
    15-NN mean of the same (>= 4 valid; NaN -> 0), their 5- and 15-year means, OLS-30 slope minus the neighbours'
    mean slope, 30-year sd; plus the count of missing departures (used only to drop rows)."""
    key = (m, t)
    if key not in DEP_CACHE:
        M, y0 = panel(table.loc[:, :t])
        r = C.ref30(M, y0, t)
        hist = C.window(M, y0, t - 14, t, t) - r[:, None]
        dep = hist - C.nanmean(hist[NEIGH], 1, 4)
        w30 = C.window(M, y0, t - 29, t, t)
        s, _ = C.ols_slope(w30, np.arange(t - 29, t + 1, dtype=float)[None, :])
        f = np.column_stack([np.nan_to_num(dep), np.nan_to_num(C.nanmean(dep[:, -5:], 1)), np.nan_to_num(C.nanmean(dep, 1)),
                             np.nan_to_num(s - C.nanmean(s[NEIGH], 1)), np.nan_to_num(np.nanstd(w30, 1))])
        DEP_CACHE[key] = (f, np.isnan(dep).sum(1))
    return DEP_CACHE[key]


def fit_at(history, F, phi):
    """Learners for metrics with learn=True, on pseudo-origins F-45..F-1 with targets <= F."""
    assert history.year.max() <= F
    tables = P.tables_at(history, CITIES, F)
    first = min(int(tb.columns.min()) for tb in tables.values())
    t_values = list(range(max(first, F - P.TRAIN_YEARS), F))
    states = P.build_states(tables, NEIGH, t_values + [F])
    shifts = {}
    for m in MET:
        if not SET[m]["learn"]:
            shifts[m] = pd.DataFrame(0.0, index=CITIES, columns=range(F + 1, F + 8))
            continue
        dep = SET[m].get("dep", False)
        names = P.feature_names("all") + (DEP_NAMES if dep else [])
        table = tables[m]

        def feats(t, h):
            st = states[t]["metric"][m]
            base = P.feature_matrix(st, states[t]["pooled"][m], states[t]["metric"]["annual_avg_temp"], h, "all")
            if not dep:
                return base, np.zeros(len(CITIES))
            f, nmiss = dep_features(table, t, m)
            return np.column_stack([base, f]), nmiss

        X, y = [], []
        for h in P.LEADS:
            for t in t_values:
                if t + h > F or t + h not in table.columns:
                    continue
                assert t + h <= F
                st = states[t]["metric"][m]
                tgt = table[t + h].to_numpy(float) - st["ref30"] - W_at(tables, t, phi)[m][:, h - 1]
                fx, nmiss = feats(t, h)
                if dep:   # departure from the region: remove the 15-NN mean of the same residual, no median removal
                    tgt = tgt - C.nanmean(tgt[NEIGH], 1, 4)
                keep = np.isfinite(tgt) & np.isfinite(st["ref30"]) & (nmiss <= 5)
                if keep.any():
                    if not dep:
                        tgt = tgt - np.median(tgt[keep])
                    X.append(fx[keep])
                    y.append(tgt[keep])
        model = P.Members(names, "ensemble", 42).fit(np.vstack(X), np.concatenate(y))
        st = states[F]["metric"][m]
        by = {F + h: model.predict(feats(F, h)[0]) for h in P.LEADS}
        shifts[m] = pd.DataFrame(by, index=CITIES).where(pd.Series(st["ref30"], index=CITIES).notna(), 0.0)
    return shifts, W_frame(W_at(tables, F, phi), F)


def assemble(base, combo, W, shifts, lam):
    """THE assembly (plan 3.2), per-metric settings: used for phi selection, the lambda gate and the outer forecast."""
    parts = []
    for m in MET:
        rows = base[base.metric == m][KEY + ["predicted_value"]].copy()
        w = W[m].rename_axis(index="city_id", columns="year").stack().rename("W").reset_index()
        rows = rows.merge(w, on=["city_id", "year"], how="left", validate="one_to_one")
        if shifts is not None:
            s = shifts[m].rename_axis(index="city_id", columns="year").stack().rename("shift").reset_index()
            rows = rows.merge(s, on=["city_id", "year"], how="left", validate="one_to_one")
        else:
            rows["shift"] = 0.0
        own = rows.predicted_value + rows.W.fillna(0.0) + lam[m] * rows["shift"].fillna(0.0)
        cw = SET[m]["cw"]
        if cw > 0:
            cm = combo[combo.metric == m][KEY + ["predicted_value"]].rename(columns={"predicted_value": "combo"})
            rows = rows.merge(cm, on=KEY, how="left", validate="one_to_one")
            pred = (1 - cw) * own + cw * (rows.combo + (rows.W.fillna(0.0) if m in COMBO_GETS_W else 0.0))
        else:
            pred = own
        if m in P.NON_NEGATIVE:
            pred = pred.clip(lower=0.0)
        rows["predicted_value"] = pred
        assert rows.predicted_value.notna().all()
        parts.append(rows[KEY + ["predicted_value"]])
    return pd.concat(parts, ignore_index=True)


FOLDS = {}


def fold(I, ceiling):
    """Inner or outer fold at origin I whose targets I+1..I+7 must not pass `ceiling`."""
    assert I + 7 <= ceiling <= CUTOFF, f"fold {I} targets pass {ceiling}"
    if I not in FOLDS:
        tr, tg, truth, ref = build_fold(annual[annual.year <= I + 7], CITIES, I)
        base = reference(tr, tg, I, 42, method="average")
        combo = reference(tr, tg, I, 42, method="combo") if SET and any(SET[m]["cw"] > 0 for m in MET) else None
        FOLDS[I] = (tr, tg, truth, ref, base, combo, P.tables_at(tr, CITIES, I))
    return FOLDS[I]


PHI = {}


def select_phi(F):
    core = [m for m in MET if SET[m]["W"] == "core"]
    if not core:
        return 0.0, None
    if F in PHI:
        return PHI[F]
    sk = {p: [] for p in PHI_GRID}
    used = []
    for I in range(F - 21, F - 6):
        try:
            tr, tg, truth, ref, base, combo, tables = fold(I, F)
        except ValueError:
            continue
        used.append(I)
        for p in PHI_GRID:
            s = score(assemble(base, combo, W_frame(W_at(tables, I, p), I), None, {m: 0.0 for m in MET}), truth, ref, "final")["skill"]
            sk[p].append(float(np.mean([s[m] for m in core])))
    if not used:
        PHI[F] = (0.0, {"inner": [], "means": {}})
        return PHI[F]
    means = {p: float(np.mean(v)) for p, v in sk.items()}
    best = max(means.values())
    phi = min(p for p in PHI_GRID if means[p] >= best - PHI_TIE)
    PHI[F] = (phi, {"inner": used, "means": {str(k): v for k, v in means.items()}})
    return PHI[F]


def forecast(O):
    tr, tg, truth, ref, base, combo, tables = fold(O, CUTOFF)
    phi_O, phi_tab = select_phi(O)
    rec = {"phi": phi_O, "phi_table": phi_tab, "gate": {}}
    learn = [m for m in MET if SET[m]["learn"]]
    lambdas = {m: 0.0 for m in MET}
    if learn:
        skills = {lam: {m: [] for m in learn} for lam in P.LAMBDA_GRID}
        for I in [O - k for k in P.INNER_OFFSETS]:
            trI, tgI, truthI, refI, baseI, comboI, _ = fold(I, O)
            phi_I, tab_I = select_phi(I)
            shifts, W = fit_at(trI, I, phi_I)
            rec["gate"][str(I)] = {"phi": phi_I, "phi_table": tab_I, "skill": {}}
            for lam in P.LAMBDA_GRID:
                s = score(assemble(baseI, comboI, W, shifts, {m: (lam if m in learn else 0.0) for m in MET}), truthI, refI, "final")["skill"]
                rec["gate"][str(I)]["skill"][str(lam)] = {m: s[m] for m in learn}
                for m in learn:
                    skills[lam][m].append(s[m])
        for m in learn:
            means = {lam: float(np.mean(skills[lam][m])) for lam in P.LAMBDA_GRID}
            if SET[m].get("gate_rel"):   # gain over lambda = 0, not absolute skill, must be positive
                means = {lam: v - means[0.0] for lam, v in means.items()}
            lambdas[m] = P.select_lambda(means)
        shifts, W = fit_at(tr, O, phi_O)
    else:
        shifts, W = None, W_frame(W_at(tables, O, phi_O), O)
    rec["lambdas"] = lambdas
    rec["mean_W"] = {m: float(W[m].to_numpy().mean()) for m in MET}
    return assemble(base, combo, W, shifts, lambdas), rec


OUT.mkdir(parents=True, exist_ok=True)
for O in OUTER:
    assert O + 7 <= CUTOFF
    t0 = time.time()
    tr, tg, truth, ref, base, combo, tables = fold(O, CUTOFF)
    if args.config == "v05":
        pred, rec = P.run_forecast(tr, tg, O, 42, coords.loc[CITIES])[0], {}
    else:
        pred, rec = forecast(O)
    rows = truth[truth.scored.astype(bool)].merge(pred, on=KEY, how="left", validate="one_to_one")
    assert rows.predicted_value.notna().all() and rows.year.max() <= CUTOFF
    rows["absolute_error"] = (rows.predicted_value - rows.actual_value).abs()
    rows["baseline_absolute_error"] = (rows.baseline_value - rows.actual_value).abs()
    rows["origin"] = O
    rows[["city_id", "year", "metric", "split", "origin", "absolute_error", "baseline_absolute_error"]].to_csv(
        OUT / f"{args.config}-{O}.csv", index=False)
    rec["final_skill"] = score(pred, truth, ref, "final")["skill"]
    rec["live_skill"] = score(pred, truth, ref, "live")["skill"]
    rec["seconds"] = round(time.time() - t0, 1)
    CHECKS["outer"][str(O)] = rec
    print(args.config, O, f"{rec['seconds']}s", "phi", rec.get("phi"), "lambdas", rec.get("lambdas"), flush=True)
CHECKS["rho"] = {f"{m}@{a}": info for (m, a), info in sorted(RHO.info.items())}
(OUT / f"{args.config}-checks.json").write_text(json.dumps(CHECKS, indent=1, default=float), encoding="utf-8")
