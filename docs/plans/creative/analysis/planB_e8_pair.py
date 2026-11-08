"""Plan B r6: one paired comparison of two E8 configurations at full precision (reward.py arithmetic, 2,000 city draws,
seed 20261108; promote with >= 4 of 5 origin wins on the five retrospective origins).

usage: uv run python docs/plans/creative/analysis/planB_e8_pair.py CANDIDATE AGAINST [--out DIR]
Example (r6 item 6): planB_e8_pair.py r21+prcpw25 r21  -> planB_e8_prcpw75_vs_v21.txt
"""
from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["annual_avg_temp", "total_precip_in", "hottest_day_temp", "cdd"]
SPLITS = ["live", "final"]
ap = argparse.ArgumentParser()
ap.add_argument("candidate")
ap.add_argument("against")
ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "planB_e8"))
a = ap.parse_args()
OUT = Path(a.out)


def load(c):
    r = pd.concat([pd.read_csv(p, dtype={"city_id": str}) for p in sorted(OUT.glob(f"{c}-[0-9][0-9][0-9][0-9].csv"))])
    assert r.year.max() <= 1983
    return r.groupby(["city_id", "origin", "split", "metric"])[["absolute_error", "baseline_absolute_error"]].sum()


gc, ga = load(a.candidate), load(a.against)
origins = sorted(gc.index.get_level_values("origin").unique())
cities = sorted(gc.index.get_level_values("city_id").unique())
cells = pd.MultiIndex.from_tuples([(o, s, m) for o in origins for s in SPLITS for m in METRICS])
mat = lambda g, col: g[col].unstack([1, 2, 3]).reindex(index=cities, columns=cells).fillna(0).to_numpy()  # noqa: E731
REF = mat(gc, "baseline_absolute_error")
assert np.allclose(REF, mat(ga, "baseline_absolute_error"))
rng = np.random.default_rng(20261108)
W = np.vstack([np.ones(len(cities)), rng.multinomial(len(cities), np.full(len(cities), 1 / len(cities)), size=2000).astype(float)])
sk = lambda g: (1 - (W @ mat(g, "absolute_error")) / (W @ REF)).reshape(len(W), len(origins), 2, 4)  # noqa: E731
s_c = sk(gc)
d = 100 * (s_c - sk(ga)).mean(axis=3)
fin, live = d.mean(axis=1)[:, 1], d.mean(axis=1)[:, 0]
by_origin = d[0, :, 1]
metric = 100 * s_c[0, :, 1, :].mean(axis=0)
need = max(2, math.ceil(2 / 3 * len(origins)))
checks = {"P>=0.95": bool((fin[1:] > 0).mean() >= 0.95 and fin[0] > 0), f"wins>={need}": int((by_origin > 0).sum()) >= need,
          "live not clearly worse": bool((live[1:] < 0).mean() < 0.95), "floor -2": bool(metric.min() >= -2)}
print(f"{a.candidate} vs {a.against}: delta {fin[0]:+.6f}, 90% [{np.quantile(fin[1:], 0.05):+.6f}, {np.quantile(fin[1:], 0.95):+.6f}], "
      f"P(better) {(fin[1:] > 0).mean():.4f}, origin wins {int((by_origin > 0).sum())}/{len(origins)} "
      f"({' / '.join(f'{v:+.4f}' for v in by_origin)}), live delta {live[0]:+.6f} (P worse {(live[1:] < 0).mean():.4f}), "
      f"candidate metric points {' / '.join(f'{v:+.4f}' for v in metric)}")
print(f"checks {checks} -> {'promote' if all(checks.values()) else 'not promote'}")
