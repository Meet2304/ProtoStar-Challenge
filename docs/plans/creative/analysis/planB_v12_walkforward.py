"""Plan B r6: pre-1983 walk-forward of the v1.2-based versions against a v1.2 base built from plan A's functions.

Run from the repository root:  uv run python docs/plans/creative/analysis/planB_v12_walkforward.py   (about 3 minutes)

What it computes (data and cutoffs exactly as planA_walkforward.py, which is imported and not edited):
- planA_walkforward.load(): data/annual_valid.csv and data/city_info.csv, years <= 1982 (asserted there).
- v1.2 per fit origin F in 1940..1975: plan A's precompute / fit_coefs / forecast_at (gauss kernel, base avg30,
  use_long=True); fit_coefs asserts that every coefficient-fit target year is <= F.
- Placeholder_v2.7 (primary, kappa 0.5) = v1.2's deployed forecast + W_hot for hottest_day_temp, added AFTER v1.2's
  centre(); v1.2's coefficients are not refitted. W_hot = planB_core.core_W(..., "hottest_day_temp", kappa) at F,
  which reads columns <= F only (planB_core.window asserts it). Diagnostic: kappa 1.0.
- Placeholder_v2.8 = v2.7 + the precipitation history blend 0.5 * (all-history mean - ref30) added after v1.2's
  centre(), clipped at 0. Diagnostic: weight 0.25.
- Scoring: plan A's city_sums and compare() (reward.py-matched: seed 20261108 reset per triplet, 2,000 draws, cities
  with a scored row resampled, promote = P >= 0.95, delta > 0, >= 2 of 3 origin wins, live not clearly worse,
  metric floor -2). Triplets (F, F+7, F+14), F = 1940..1961, all targets <= 1982 (asserted below).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import planA_walkforward as A  # noqa: E402  (imported, not edited)
import planB_core as C  # noqa: E402

KAPPA_HOT, KAPPA_HOT_DIAG = 0.5, 1.0          # frozen before this script was first run (plan r6, section 5.3)
PRCP_W, PRCP_W_DIAG = 0.5, 0.25
FIT_ORIGINS = A.FIT_ORIGINS                    # 1940..1975
TRIPS = [(f, f + 7, f + 14) for f in range(1940, 1962)]


def main():
    t0 = time.perf_counter()
    cities, coords, tab = A.load()
    d = A.distances(coords)
    dn = d.copy()
    np.fill_diagonal(dn, np.inf)
    order = np.argsort(dn, axis=1, kind="stable")
    pre = A.precompute(tab, A.gauss_kernel(d), order, range(FIT_ORIGINS[0] - A.TRAIN_YEARS, FIT_ORIGINS[-1] + 1))
    nb = C.neighbours(coords.lat, coords.lon)
    assert (nb == order[:, :15]).all(), "plan A and plan B neighbour orders differ"
    nbs = np.column_stack([np.arange(len(cities)), nb])
    rho_cache = C.RhoCache()
    print(f"setup {time.perf_counter() - t0:.1f} s; {len(cities)} cities; data years <= {A.LAST}")

    # retrospective-only check (not a development diagnostic): refit v1.2's hottest-day coefficients on base ref30 + W_hot,
    # so the coefficient fit sees the deployed formula. W_hot at every pseudo-origin t reads columns <= t only.
    for t in sorted(pre):
        w = C.core_W(tab["hottest_day_temp"], A.Y0, t, rho_cache.get("hottest_day_temp", tab["hottest_day_temp"], A.Y0, t, nbs, A.Y0),
                     0.0, "hottest_day_temp", nbs, nb, kappa=KAPPA_HOT)[0]
        ref_t, _, _, base_t = pre[t]["hottest_day_temp"]
        for h in range(1, 8):
            base_t[h]["avg30_hot"] = ref_t + w[:, h - 1]

    sums, n_f2, rhos = {}, [], {}
    for fo in FIT_ORIGINS:
        assert fo + 7 <= A.LAST, "scored target after the design cutoff"
        Mh, Mp = tab["hottest_day_temp"], tab["total_precip_in"]
        rho = rho_cache.get("hottest_day_temp", Mh, A.Y0, fo, nbs, A.Y0)
        rhos[fo] = rho
        w_hot = {k: C.core_W(Mh, A.Y0, fo, rho, 0.0, "hottest_day_temp", nbs, nb, kappa=k)
                 for k in (KAPPA_HOT, KAPPA_HOT_DIAG)}
        n_f2.append(w_hot[KAPPA_HOT][1])
        w_prcp = {w: C.precip_W(Mp, A.Y0, fo, weight=w) for w in (PRCP_W, PRCP_W_DIAG)}
        for m in A.METRICS:
            coefs = A.fit_coefs(tab, pre, m, fo, "avg30", True)
            fc, ref, _ = A.forecast_at(pre, m, fo, "v1.2", coefs)
            variants = {"v1.2": fc}
            for name, k, w in (("v2.7", KAPPA_HOT, None), ("v2.7 diag hot-k1", KAPPA_HOT_DIAG, None),
                               ("v2.8", KAPPA_HOT, PRCP_W), ("v2.8 diag prcp-w75", KAPPA_HOT, PRCP_W_DIAG)):
                out = {}
                for h in range(1, 8):
                    f = fc[h]
                    if m == "hottest_day_temp":
                        f = f + w_hot[k][0][:, h - 1]                         # added after centre(); no refit
                    if m == "total_precip_in" and w is not None:
                        f = np.maximum(f + w_prcp[w][:, h - 1], 0.0)          # added after centre(); clipped
                    out[h] = f
                variants[name] = out
            if m == "hottest_day_temp":
                b2, g2 = A.fit_coefs(tab, pre, m, fo, "avg30_hot", True)
                _, l_raw, l_long, base_f = pre[fo][m]
                g = A.graph_shift(b2, g2, l_raw, l_long, True)
                variants["check refit-hot"] = {h: A.deploy(m, base_f[h]["avg30_hot"], g) for h in range(1, 8)}
            else:
                variants["check refit-hot"] = variants["v2.7"]
            for name, f in variants.items():
                sums[(name, fo, m)] = A.city_sums(tab, m, fo, f, ref)
    print(f"walk-forward {time.perf_counter() - t0:.1f} s; hottest-day F2 cities per origin: "
          f"median {int(np.median(n_f2))}, max {max(n_f2)}; rho(hottest) range {min(rhos.values()):.4f}-{max(rhos.values()):.4f}")

    def skill(name, fo, m):
        e, r, _ = sums[(name, fo, m)]["final"]
        return 1 - e.sum() / r.sum()

    names = ["v1.2", "v2.7", "v2.7 diag hot-k1", "v2.8", "v2.8 diag prcp-w75", "check refit-hot"]
    print("\n== Held-out final points by metric, mean over the 36 fit origins 1940-1975 (vs average30) ==")
    for name in names:
        pm = [100 * np.mean([skill(name, fo, m) for fo in FIT_ORIGINS]) for m in A.METRICS]
        tot = 100 * np.mean([np.mean([skill(name, fo, m) for m in A.METRICS]) for fo in FIT_ORIGINS])
        print(f"   {name:20s} total {tot:+.4f} | temp {pm[0]:+.4f} precip {pm[1]:+.4f} hot {pm[2]:+.4f} cdd {pm[3]:+.4f}")
    for name in names[1:]:
        dm = [100 * np.mean([skill(name, fo, m) - skill("v1.2", fo, m) for fo in FIT_ORIGINS]) for m in A.METRICS]
        pos = np.mean([np.mean([skill(name, fo, m) - skill("v1.2", fo, m) for m in A.METRICS]) > 0 for fo in FIT_ORIGINS])
        print(f"   {name:20s} minus v1.2 by metric: temp {dm[0]:+.4f} precip {dm[1]:+.4f} hot {dm[2]:+.4f} cdd {dm[3]:+.4f}; "
              f"total better at {pos:.0%} of the 36 fit origins")

    print(f"\n== Promotion analogue: {len(TRIPS)} triplets (F, F+7, F+14), F = 1940..1961, reward.py-matched city bootstrap ==")
    pairs = [("v1.2", "average30"), ("v2.7", "average30"), ("v2.7", "v1.2"), ("v2.7 diag hot-k1", "v1.2"),
             ("v2.7 diag hot-k1", "v2.7"), ("v2.8", "v2.7"), ("v2.8 diag prcp-w75", "v2.7"), ("v2.8", "v1.2"),
             ("check refit-hot", "v1.2"), ("check refit-hot", "v2.7")]
    for cand, champ in pairs:
        res = pd.DataFrame([A.compare(sums, cand, champ, trip, A.SEED) for trip in TRIPS])
        if (cand, champ) in (("v2.7", "v1.2"), ("v2.7 diag hot-k1", "v1.2")):
            for i in np.flatnonzero(~res.promote.to_numpy()):
                r = res.iloc[i]
                print(f"      {cand} not promoting at {TRIPS[i]}: delta {r.pts:+.4f} P {r.p:.3f} LB90 {r.lb:+.4f} "
                      f"wins {r.wins}/3 floor {r.floor:+.3f} by origin {' / '.join(f'{v:+.2f}' for v in r.by_origin)}")
        weakest = TRIPS[int(res.p.idxmin())]
        print(f"   {cand:20s} vs {champ:9s} median delta {res.pts.median():+.4f}  mean {res.pts.mean():+.4f}  "
              f"median P {res.p.median():.3f}  min P {res.p.min():.4f} at {weakest}  median LB90 {res.lb.median():+.4f}  "
              f"min LB90 {res.lb.min():+.4f}  2+ wins {int((res.wins >= 2).sum())}/{len(res)}  "
              f"promote {int(res.promote.sum())}/{len(res)}  worst floor {res.floor.min():+.3f}")


if __name__ == "__main__":
    main()
