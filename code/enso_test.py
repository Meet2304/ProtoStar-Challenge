"""One-year-ahead test of the El Nino adjustment in combo_model.py, over 38 forecast origins (1980-2017).

For every origin year X from 1980 to the last year with a year-end ONI (2017 with the archived
copies), forecast year X+1 with the combined model, with and without the El Nino shift (each city's
sensitivity fitted only on years before X), and compare the errors on the rows the challenge would
score. This complements the 4-window backtest in combo_model.py, which sees only 4 first years.

Usage (from the repository root):  python code/enso_test.py
Writes reports/enso_rolling_test.csv (one row per origin and metric).
"""

import pandas as pd

import combo_model as cm
from common import CUTOFF_YEAR, REPORTS_DIR, load_valid_annual, thirty_year_mean, wide

FIRST_TEST_ORIGIN = 1980


def main() -> None:
    annual = load_valid_annual()
    oni_end = cm.load_oni_year_end()
    last_origin = min(int(oni_end.index.max()), CUTOFF_YEAR - 1)   # X+1 must be a known year
    rows = []
    for metric in cm.ENSO_METRICS:
        table = wide(annual, metric)
        for x in range(FIRST_TEST_ORIGIN, last_origin + 1):
            if x not in oni_end.index:
                continue
            year = x + 1
            scored = thirty_year_mean(table.loc[:, :x], x).notna() & table[year].notna()
            base = cm.combo(table, metric, x, [year], oni_end, enso=False)[year]
            enso = cm.combo(table, metric, x, [year], oni_end, enso=True)[year]
            rows.append({"metric": metric, "origin": x, "oni_year_end": round(float(oni_end[x]), 3),
                         "mae_without": (base - table[year]).abs()[scored].mean(),
                         "mae_with": (enso - table[year]).abs()[scored].mean()})
    report = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    report.round(4).to_csv(REPORTS_DIR / "enso_rolling_test.csv", index=False)
    print(f"First forecast year, origins {FIRST_TEST_ORIGIN}-{last_origin}:")
    for metric, g in report.groupby("metric", sort=False):
        change = 1 - g["mae_with"].mean() / g["mae_without"].mean()
        better = int((g["mae_with"] < g["mae_without"]).sum())
        worst = g.assign(d=g["mae_with"] - g["mae_without"]).nlargest(1, "d").iloc[0]
        print(f"  {metric:17s} error change {-change:+.1%} with El Nino | better in {better}/{len(g)} years"
              f" | worst year: forecast for {int(worst['origin']) + 1} (ONI {worst['oni_year_end']:+.2f})")


if __name__ == "__main__":
    main()
