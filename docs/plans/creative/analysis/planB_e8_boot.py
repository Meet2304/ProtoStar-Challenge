"""Plan B r3, E8 summary: reward-v1 arithmetic on the retrospective folds, plus report-only sensitivity diagnostics.

usage: uv run python docs/plans/creative/analysis/planB_e8_boot.py CONFIG[,CONFIG...] [--against v05] [--out DIR]
Paired city bootstrap exactly as code/evaluation/reward.py (2,000 multinomial city draws, seed 20261108), and two
diagnostics that are never used to select variants: leave-one-origin-out points, and a spatial-block bootstrap
(5 x 5 degree lat/lon cells resampled with replacement, seed 20261108).
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CUTOFF = 1983
METRICS = ["annual_avg_temp", "total_precip_in", "hottest_day_temp", "cdd"]
SPLITS = ["live", "final"]
ap = argparse.ArgumentParser()
ap.add_argument("configs")
ap.add_argument("--against", default="v05")
ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "planB_e8"))
args = ap.parse_args()
OUT = Path(args.out)
names = args.configs.split(",")
if args.against and args.against not in names:
    names = [args.against] + names

runs = {c: pd.concat([pd.read_csv(p, dtype={"city_id": str}) for p in sorted(OUT.glob(f"{c}-[0-9][0-9][0-9][0-9].csv"))])
        for c in names}
for r in runs.values():
    assert r.year.max() <= CUTOFF
origins = sorted(runs[names[0]].origin.unique())
cities = sorted(set().union(*[set(r.city_id) for r in runs.values()]))
cells = pd.MultiIndex.from_tuples([(o, s, m) for o in origins for s in SPLITS for m in METRICS])


def mats(r):
    g = r.groupby(["city_id", "origin", "split", "metric"])[["absolute_error", "baseline_absolute_error"]].sum()
    e = g.absolute_error.unstack([1, 2, 3]).reindex(index=cities, columns=cells).fillna(0).to_numpy()
    b = g.baseline_absolute_error.unstack([1, 2, 3]).reindex(index=cities, columns=cells).fillna(0).to_numpy()
    return e, b


M = {c: mats(r) for c, r in runs.items()}
REFM = M[names[0]][1]
for c in names:
    assert np.allclose(M[c][1], REFM), "scored rows differ"
rng = np.random.default_rng(20261108)
WC = np.vstack([np.ones(len(cities)), rng.multinomial(len(cities), np.full(len(cities), 1 / len(cities)), size=2000).astype(float)])
coords = pd.read_csv(ROOT / "data" / "city_info.csv", dtype={"city_id": str}).set_index("city_id").loc[cities]
block = (np.floor(coords.lat / 5).astype(int).astype(str) + "_" + np.floor(coords.lon / 5).astype(int).astype(str)).to_numpy()
ublocks = np.unique(block)
rngb = np.random.default_rng(20261108)
WB = [np.ones(len(cities))]
for _ in range(2000):
    draw = rngb.integers(0, len(ublocks), len(ublocks))
    counts = pd.Series(ublocks[draw]).value_counts()
    WB.append(pd.Series(block).map(counts).fillna(0).to_numpy(float))
WB = np.vstack(WB)


def skills(c, W):
    return (1 - (W @ M[c][0]) / (W @ REFM)).reshape(len(W), len(origins), 2, 4)


def compare(c, against, W, keep_origins=None):
    s = skills(c, W)
    d = s - (skills(against, W) if against else 0.0)
    if keep_origins is not None:
        d = d[:, keep_origins]
        s = s[:, keep_origins]
    pts = 100 * d.mean(axis=3)
    tot = pts.mean(axis=1)
    fin, live = tot[:, 1], tot[:, 0]
    by_origin = pts[0, :, 1]
    metric_pts = 100 * s[0, :, 1, :].mean(axis=0)
    p = float((fin[1:] > 0).mean())
    wins = int((by_origin > 0).sum())
    need = max(2, int(np.ceil(2 / 3 * len(by_origin))))
    checks = dict(final=p >= 0.95 and fin[0] > 0, origins=wins >= need, live=(live[1:] < 0).mean() < 0.95,
                  floor=metric_pts.min() >= -2)
    verdict = "promote" if all(checks.values()) else ("worse" if (fin[1:] < 0).mean() >= 0.95 and fin[0] < 0 else "within_noise")
    return dict(delta=float(fin[0]), lo=float(np.quantile(fin[1:], 0.05)), hi=float(np.quantile(fin[1:], 0.95)), p=p,
                sd=float(fin[1:].std(ddof=1)), live=float(live[0]), wins=wins, n=len(by_origin), by_origin=by_origin,
                metric=metric_pts, verdict=verdict)


fmt = lambda v: " / ".join(f"{x:+.2f}" for x in v)  # noqa: E731
print(f"origins {list(map(int, origins))}; {len(cities)} cities; {len(ublocks)} spatial blocks; 2000 draws, seed 20261108")
print("\nA. vs average30 (champion analogue), city bootstrap")
for c in names:
    r = compare(c, None, WC)
    print(f"  {c:6s} final {r['delta']:+.4f} 90% [{r['lo']:+.2f}, {r['hi']:+.2f}] P {r['p']:.3f} sd {r['sd']:.2f} live {r['live']:+.2f} "
          f"wins {r['wins']}/{r['n']} origins [{fmt(r['by_origin'])}] metrics [{fmt(r['metric'])}] {r['verdict']}")
if args.against:
    print(f"\nB. paired vs {args.against}, city bootstrap")
    for c in names:
        if c != args.against:
            r = compare(c, args.against, WC)
            print(f"  {c:6s} delta {r['delta']:+.4f} 90% [{r['lo']:+.2f}, {r['hi']:+.2f}] P {r['p']:.3f} wins {r['wins']}/{r['n']} [{fmt(r['by_origin'])}]")
print("\nC. report-only: spatial-block bootstrap vs average30 (and vs %s)" % args.against)
for c in names:
    r = compare(c, None, WB)
    line = f"  {c:6s} vs avg30 {r['delta']:+.2f} 90% [{r['lo']:+.2f}, {r['hi']:+.2f}] P {r['p']:.3f}"
    if args.against and c != args.against:
        q = compare(c, args.against, WB)
        line += f" | vs {args.against} {q['delta']:+.2f} 90% [{q['lo']:+.2f}, {q['hi']:+.2f}] P {q['p']:.3f}"
    print(line)
print("\nD. report-only: leave-one-origin-out final points vs average30 (and delta vs %s)" % args.against)
for c in names:
    cells_ = []
    for i, o in enumerate(origins):
        keep = [j for j in range(len(origins)) if j != i]
        r = compare(c, None, WC, keep)
        txt = f"-{int(o)}:{r['delta']:+.2f}[lo {r['lo']:+.2f}]"
        if args.against and c != args.against:
            q = compare(c, args.against, WC, keep)
            txt += f"(vs {args.against} {q['delta']:+.2f}, lo {q['lo']:+.2f})"
        cells_.append(txt)
    print(f"  {c:6s} " + "  ".join(cells_))
