"""Plan A: pre-1983 walk-forward rows and checks for the conditional Placeholder_v1.4 (plan sections 4.3 step 4 and 5).

Run from the repository root:  uv run python docs/plans/creative/analysis/planA_v14_walkforward.py   (about 10 minutes)

v1.4 = v1.2's graph term (G = centre(beta L_raw + gamma L_long_raw), joint fit) on v1.3's base (ref30 + I, the national
increment with KAPPA 0.5, CAP_F 1.0 F, floor 0), with beta and gamma refitted with I in the base. It is built only from
planA_walkforward.py's functions: the candidate ("gauss", "avg30_nat", use_long=True, fitted) is added to the imported
module's CANDIDATES table in memory, exactly as planA_parity.py assembles it. planA_walkforward.py itself is not changed.

Data rule and cutoffs (audit points), all inherited from planA_walkforward.py:
- `load` keeps only years <= 1982 (asserted) before anything else is computed.
- `fit_coefs` asserts max(training target year) <= F before every fit; fit origins F = 1915..1975.
- Graph terms and the national increment at a pseudo-origin t read only columns <= t.
- The bootstrap is the script's `compare` (reward.py-matched: seed 20261108 reset per comparison, only cities with a
  scored row in the triplet resampled, 2,000 draws, the four promotion guards).
Section 5 checks for v1.4 (adapter notebooks/Meet/placeholder_v1_4.py, on data/annual_valid.csv kept to years <= 1982
and cut at F): KAPPA = 0 with refitting reproduces v1.2; fallback rows equal reference(..., "average"); no year after F
is read (cut-table check at F = 1940 and 1975, in the script and in the adapter); N <= 0 gives I = 0.
v1.1-v1.3 are recomputed only to confirm that this route reproduces the numbers already reported in the plan.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "notebooks" / "Meet"), str(ROOT / "code")]

spec = importlib.util.spec_from_file_location("planA_walkforward", Path(__file__).with_name("planA_walkforward.py"))
wf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wf)

wf.CANDIDATES["v1.4"] = ("gauss", "avg30_nat", True, True)        # v1.2's graph term on v1.3's base, refitted
NAMES = ("v1.1", "v1.2", "v1.3", "v1.4")
ALL_BLOCKS = {**wf.EARLY_BLOCKS, **wf.BLOCKS}                   # 1915-35, 1925-39, 1940-49, 1950-62, 1963-75
# Final points against average30 already reported in plan A section 3.4 (the reproduction check)
REPORTED = {"v1.1": (0.8983, 0.5109, 0.6582, 2.3694, 1.0970), "v1.2": (0.8731, 0.5611, 1.0576, 2.4880, 1.0703),
            "v1.3": (2.6324, 2.5033, 0.1861, 2.3823, 1.0989)}
MAIN_TRIPS = [(f, f + 7, f + 14) for f in range(1940, 1962)]
EARLY_TRIPS = [(f, f + 7, f + 14) for f in range(1915, 1926)]
CHECK_ORIGINS = (1940, 1960, 1975)
CUT_ORIGINS = (1940, 1975)


def walk_forward(tab, pre):
    coefs, sums, forecasts, shifts = {}, {}, {}, {}
    for fo in wf.ALL_ORIGINS:
        for m in wf.METRICS:
            for cand in NAMES:
                _, base_kind, use_long, _ = wf.CANDIDATES[cand]
                c = wf.fit_coefs(tab, pre, m, fo, base_kind, use_long)       # asserts max target year <= fo
                coefs[(cand, fo, m)] = c
                fc, ref, g = wf.forecast_at(pre, m, fo, cand, c)
                forecasts[(cand, fo, m)], shifts[(cand, fo, m)] = fc, g
                sums[(cand, fo, m)] = wf.city_sums(tab, m, fo, fc, ref)
    return coefs, sums, forecasts, shifts


def skill(sums, cand, fo, m, split):
    e, r, _ = sums[(cand, fo, m)][split]
    return 1 - e.sum() / r.sum()


def points(sums, cand, fo, split, minus=None):
    return 100 * np.mean([skill(sums, cand, fo, m, split) - (skill(sums, minus, fo, m, split) if minus else 0.0)
                          for m in wf.METRICS])


def bootstrap_rows(sums, trips, label):
    print(f"\n== Promotion analogue, {label}: {len(trips)} triplets, city resampling, {wf.DRAWS} draws, "
          f"seed {wf.SEED} reset per triplet ==")
    for cand, champ in (("v1.2", "average30"), ("v1.4", "average30"), ("v1.4", "v1.2")):
        res = pd.DataFrame([wf.compare(sums, cand, champ, trip, wf.SEED) for trip in trips])
        k = int(res.p.idxmin())
        weakest = trips[k]
        p0 = float(res.p.iloc[k])
        half = 1.6449 * np.sqrt(p0 * (1 - p0) / wf.DRAWS)
        ps = np.array([wf.compare(sums, cand, champ, weakest, s)["p"] for s in range(1, 31)])
        print(f"   {cand} vs {champ:9s}: median reward {res.pts.median():+.4f}  median P {res.p.median():.3f}  "
              f"median LB90 {res.lb.median():+.4f}  min LB90 {res.lb.min():+.4f}  worst floor {res.floor.min():+.4f}  "
              f"2+ wins {int((res.wins >= 2).sum())}/{len(res)}  promote {int(res.promote.sum())}/{len(res)}  "
              f"cities {res.n_cities.min()}-{res.n_cities.max()}")
        print(f"      weakest triplet {weakest}: P {p0:.4f} (binomial 90% [{p0 - half:.4f}, {p0 + half:.4f}]), "
              f"reward {res.pts.iloc[k]:+.4f}, wins {res.wins.iloc[k]}/3, promote {bool(res.promote.iloc[k])}; "
              f"seeds 1..30: min {ps.min():.4f} max {ps.max():.4f} mean {ps.mean():.4f}, P < 0.95 in {(ps < 0.95).sum()} of 30")


def adapter_checks(order, coefs, forecasts):
    """Plan section 5 checks for v1.4 through the adapter (and the script's cut-table check)."""
    import placeholder_v1_2 as v12
    import placeholder_v1_4 as v14
    from evaluation.adapters import reference

    annual = pd.read_csv(ROOT / "data" / "annual_valid.csv", dtype={"city_id": str})
    annual = annual[annual.year <= wf.LAST]
    assert annual.year.max() <= wf.LAST, "v1.4 analysis uses years <= 1982 only"
    info = pd.read_csv(ROOT / "data" / "city_info.csv", dtype={"city_id": str}).set_index("city_id")
    cities = sorted(annual.city_id.unique())
    failures = []

    def check(ok, text):
        print(f"   [{'ok' if ok else 'FAIL'}] {text}")
        if not ok:
            failures.append(text)

    print("\n== Section 5 checks for v1.4 (adapter notebooks/Meet/placeholder_v1_4.py, input cut at F) ==")
    for fo in CHECK_ORIGINS:
        train = annual[annual.year <= fo].copy()
        assert int(train.year.max()) <= fo, "adapter input must be cut at the fit origin"
        targets = pd.MultiIndex.from_product([cities, range(fo + 1, fo + 8), wf.METRICS],
                                             names=["city_id", "year", "metric"]).to_frame(index=False)
        coords = v14._check_coords(info, cities)
        base = reference(train, targets, fo, 42, method="average")
        out14, rec14, st14 = v14.run_forecast(train, targets, fo, 42, coords)
        out0, _, st0 = v14.run_forecast(train, targets, fo, 42, coords, kappa=0.0)       # KAPPA = 0, refitted
        out12, _, st12 = v12.run_forecast(train, targets, fo, 42, coords)
        print(f"-- F = {fo}: N {rec14['national']['N']:+.4f} F, I_temp {rec14['national']['I_temp']:.4f} F")
        same_coefs = all((st0[m]["beta"], st0[m]["gamma"]) == (st12[m]["beta"], st12[m]["gamma"]) for m in wf.METRICS)
        a = out0.merge(out12, on=["city_id", "year", "metric"], suffixes=("", "_v12"), validate="one_to_one")
        a = a.merge(base, on=["city_id", "year", "metric"], suffixes=("", "_ref"), validate="one_to_one")
        on_ref = np.zeros(len(a), dtype=bool)
        for m in wf.METRICS:
            on_ref |= ((a.metric == m) & a.city_id.isin(np.array(cities)[st12[m]["has_ref"]])).to_numpy()
        check(same_coefs and np.array_equal(a.predicted_value[on_ref], a.predicted_value_v12[on_ref]),
              f"F={fo}: KAPPA = 0 with refitting reproduces v1.2 exactly on rows with a 30-year mean "
              f"({int(on_ref.sum())} rows; coefficients equal: {same_coefs})")
        b = out14.merge(base, on=["city_id", "year", "metric"], suffixes=("", "_ref"), validate="one_to_one")
        counts, fallback_ok = [], True
        for m in wf.METRICS:
            miss = ~st14[m]["has_ref"]
            rows = b[(b.metric == m) & b.city_id.isin(np.array(cities)[miss])]
            fallback_ok &= bool(np.array_equal(rows.predicted_value, rows.predicted_value_ref)
                                and np.all(st14[m]["G"][miss] == 0.0) and np.all(st14[m]["I"][miss] == 0.0)
                                and len(rows) == 7 * int(miss.sum()))
            counts.append(f"{m} {int(miss.sum())}")
        check(fallback_ok, f"F={fo}: fallback rows equal reference(..., 'average') exactly, G = 0 and I = 0 there "
                           f"({', '.join(counts)})")
        if rec14["national"]["N"] <= 0:
            zero_i = rec14["national"]["I_temp"] == 0.0 and all(np.all(st14[m]["I"] == 0.0) for m in wf.METRICS)
            ref_plus_g = True
            for m in wf.METRICS:
                rows = b[(b.metric == m) & b.city_id.isin(np.array(cities)[st14[m]["has_ref"]])]
                pos = rows.city_id.map({c: i for i, c in enumerate(cities)}).to_numpy()
                ref_plus_g &= bool(np.array_equal(rows.predicted_value,
                                                  v14.deploy(m, rows.predicted_value_ref.to_numpy(), st14[m]["G"][pos])))
            equal12 = np.array_equal(out14.predicted_value, out12.predicted_value)
            check(zero_i and ref_plus_g, f"F={fo}: N <= 0 gives I = 0 for every city and metric, and the forecast is "
                                         f"ref30 + G with v1.4's own coefficients (equal to v1.2 here: {equal12})")

    print("-- No year after F is read: cut-table checks")
    for fo in CUT_ORIGINS:
        started = time.perf_counter()
        _, coords_f, tab_f = wf.load(cut=fo)                        # every year > F set to NaN
        pre_f = wf.precompute(tab_f, wf.gauss_kernel(wf.distances(coords_f)), order, range(fo - wf.TRAIN_YEARS, fo + 1))
        same = True
        for m in wf.METRICS:
            c = wf.fit_coefs(tab_f, pre_f, m, fo, "avg30_nat", True)
            fc, _, _ = wf.forecast_at(pre_f, m, fo, "v1.4", c)
            same &= c == coefs[("v1.4", fo, m)]
            same &= all(np.array_equal(fc[h], forecasts[("v1.4", fo, m)][h], equal_nan=True) for h in range(1, 8))
        elapsed = time.perf_counter() - started
        check(same, f"F={fo}: script, table with every year > F removed reproduces the walk-forward v1.4 coefficients "
                    f"and forecasts exactly ({elapsed:.1f} s for load + 46 pseudo-origins + v1.4 fit + forecast)")
        changed = annual.copy()
        late = changed.year > fo
        changed.loc[late, list(wf.METRICS)] += 1e4
        kernel, order_a = v14.graph(v14._check_coords(info, cities))
        fits = [v14.fit_at(v14.build_panel(frame, cities, last), kernel, order_a, fo)
                for frame, last in ((annual, wf.LAST), (changed, wf.LAST), (annual, fo))]
        same = all(fits[0][m][k] == f[m][k] for f in fits[1:] for m in wf.METRICS for k in ("beta", "gamma", "fit_rows"))
        same &= all(np.array_equal(fits[0][m][k], f[m][k], equal_nan=k != "has_ref") for f in fits[1:]
                    for m in wf.METRICS for k in ("ref", "has_ref", "G", "I"))        # ref is NaN at fallback cities
        same &= fits[0]["national"] == fits[1]["national"] == fits[2]["national"]
        check(same, f"F={fo}: adapter fit_at on a panel to 1982, the same with +1e4 after F, and a panel cut at F agree "
                    f"exactly (coefficients, G, I, N)")
    return failures


def main():
    started = time.perf_counter()
    cities, coords, tab = wf.load()
    d = wf.distances(coords)
    dn = d.copy()
    np.fill_diagonal(dn, np.inf)
    order = np.argsort(dn, axis=1, kind="stable")
    pre = wf.precompute(tab, wf.gauss_kernel(d), order, range(wf.ALL_ORIGINS[0] - wf.TRAIN_YEARS, wf.ALL_ORIGINS[-1] + 1))
    t0 = time.perf_counter()
    coefs, sums, forecasts, shifts = walk_forward(tab, pre)
    print(f"walk-forward ({len(wf.ALL_ORIGINS)} fit origins 1915-1975 x {len(NAMES)} candidates x 4 metrics): "
          f"{time.perf_counter() - t0:.1f} s")

    print("\n== Reproduction of the reported final points against average30 (plan section 3.4) ==")
    for cand, reported in REPORTED.items():
        got = [np.mean([points(sums, cand, fo, "final") for fo in fos]) for fos in ALL_BLOCKS.values()]
        assert np.allclose(got, reported, atol=5e-5), f"{cand}: {got} != reported {reported}"
        print(f"   {cand}: " + "  ".join(f"{b} {g:+.4f}" for b, g in zip(ALL_BLOCKS, got)) + "  (matches)")

    print("\n== v1.4 held-out points against average30 by block of fit origins (final = leads 4-7, live = 1-3) ==")
    for cand in ("v1.2", "v1.4"):
        print(cand)
        for bname, fos in ALL_BLOCKS.items():
            fin = np.array([points(sums, cand, fo, "final") for fo in fos])
            live = np.array([points(sums, cand, fo, "live") for fo in fos])
            per_m = [100 * np.mean([skill(sums, cand, fo, m, "final") for fo in fos]) for m in wf.METRICS]
            print(f"   {bname}: final {fin.mean():+.4f} (positive {np.mean(fin > 0):.0%}, min {fin.min():+.2f})  "
                  f"live {live.mean():+.4f} (positive {np.mean(live > 0):.0%})  | temp {per_m[0]:+.2f} "
                  f"precip {per_m[1]:+.2f} hot {per_m[2]:+.2f} cdd {per_m[3]:+.2f}")
    print("v1.4 minus v1.2 (the increment on v1.2's definition, including the refit it causes)")
    for bname, fos in ALL_BLOCKS.items():
        fin = np.array([points(sums, "v1.4", fo, "final", minus="v1.2") for fo in fos])
        live = np.array([points(sums, "v1.4", fo, "live", minus="v1.2") for fo in fos])
        ng = np.array([pre[fo]["n_gap"] for fo in fos])
        print(f"   {bname}: final {fin.mean():+.4f} (v1.4 > v1.2 at {np.mean(fin > 0):.0%}, equal at "
              f"{np.mean(fin == 0):.0%}, min {fin.min():+.2f}, max {fin.max():+.2f})  live {live.mean():+.4f}  "
              f"| N > 0 at {np.mean(ng > 0):.0%}, I mean {np.mean([pre[fo]['inc'] for fo in fos]):.3f} F")
    allfin = np.mean([points(sums, "v1.4", fo, "final", minus="v1.2") for fo in wf.ALL_ORIGINS])
    print(f"   all {len(wf.ALL_ORIGINS)} fit origins 1915-1975: final {allfin:+.4f}")
    print("   v1.4 minus v1.3 final by block (removing nothing but adding the long-window term): " + "  ".join(
        f"{b} {np.mean([points(sums, 'v1.4', fo, 'final', minus='v1.3') for fo in fos]):+.4f}"
        for b, fos in ALL_BLOCKS.items()))

    print("\n== v1.4 coefficients (beta, gamma) by fit origin, and v1.2 for comparison ==")
    show = (1915, 1925, 1935, 1940, 1950, 1960, 1970, 1975)
    for m in wf.METRICS:
        for cand in ("v1.2", "v1.4"):
            row = " ".join(f"{fo}:({coefs[(cand, fo, m)][0]:.2f},{coefs[(cand, fo, m)][1]:.2f})" for fo in show)
            print(f"   {cand} {m:17s} {row}")
    differ = sum(coefs[("v1.4", fo, m)] != coefs[("v1.2", fo, m)] for fo in wf.ALL_ORIGINS for m in wf.METRICS)
    print(f"   v1.4 coefficients differ from v1.2's in {differ} of {len(wf.ALL_ORIGINS) * 4} (origin, metric) fits")

    print("\n== N <= 0 gives I = 0 (walk-forward) ==")
    nonpos = [fo for fo in wf.ALL_ORIGINS if pre[fo]["n_gap"] <= 0]
    zero = all(pre[fo]["inc"] == 0.0 and all(np.array_equal(pre[fo][m][3][1]["avg30_nat"], pre[fo][m][3][1]["avg30"],
                                                            equal_nan=True) for m in wf.METRICS) for fo in nonpos)
    print(f"   [{'ok' if zero else 'FAIL'}] {len(nonpos)} of {len(wf.ALL_ORIGINS)} fit origins have N <= 0; at all of them "
          f"I = 0 and the v1.4 base equals ref30 exactly")
    print("\n== Cities without a 30-year mean in the walk-forward (G = 0 and I = 0 there for v1.4) ==")
    for fo in CHECK_ORIGINS:
        parts = []
        for m in wf.METRICS:
            ref = pre[fo][m][0]
            miss = ~np.isfinite(ref)
            g_ok = bool(np.all(shifts[("v1.4", fo, m)][miss] == 0.0))
            parts.append(f"{m} {int(miss.sum())} (G = 0 there: {g_ok})")
        print(f"   F={fo}: " + "; ".join(parts))

    bootstrap_rows(sums, MAIN_TRIPS, "main set F = 1940..1961")
    bootstrap_rows(sums, EARLY_TRIPS, "early set F = 1915..1925 (reused design-era years)")

    failures = adapter_checks(order, coefs, forecasts)
    if not zero:
        failures.append("N <= 0 without I = 0 in the walk-forward")
    print(f"\ntotal runtime {time.perf_counter() - started:.0f} s")
    if failures:
        print("CHECKS FAILED:\n   " + "\n   ".join(failures))
        sys.exit(1)
    print("ALL CHECKS OK")


if __name__ == "__main__":
    main()
