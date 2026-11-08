"""Plan A: parity of the Placeholder_v1.1-v1.4 adapters with the walk-forward reference (plan section 5).

Run from the repository root:  uv run python docs/plans/creative/analysis/planA_parity.py   (about 2 minutes)

For each fit origin F (1975, 1940, 1960):
- The walk-forward side uses planA_walkforward.load() (data/annual_valid.csv kept to years <= 1982, asserted there),
  its precompute and fit_coefs, and forecast_at for v1.1, v1.2 and v1.3. v1.4 is not a named candidate in the script;
  its reference forecast is assembled from the same script functions with v1.2's graph term (use_long=True) and
  v1.3's base (ref30 + national increment, "avg30_nat"), exactly as forecast_at would.
- The adapter side reads the same file, keeps years <= 1982, then cuts the table at F and asserts the cut before
  calling each adapter's run_forecast with coordinates from data/city_info.csv.
- Compared on rows with a 30-year mean: forecasts to 1e-9 (max abs difference reported per version and metric) and
  the fitted coefficients (exact). Rows without a 30-year mean must equal reference(..., "average") exactly.
At F = 1975 it also times run_forecast twice per adapter, compares SHA-256 digests of the two outputs, and runs the
harness entry point forecast() once with PROTOSTAR_ASSET_ROOT = the repository root and a temporary artifact directory.
Nothing here reads a year after 1982, and nothing is written inside the repository.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "notebooks" / "Meet"), str(ROOT / "code")]

from evaluation.adapters import reference  # noqa: E402

LAST = 1982
TOL = 1e-9
ORIGINS = (1975, 1940, 1960)
TIMED_ORIGIN = 1975
# version: (module, walk-forward base kind, use_long)
VERSIONS = {"v1.1": ("placeholder_v1_1", "avg30", False), "v1.2": ("placeholder_v1_2", "avg30", True),
            "v1.3": ("placeholder_v1_3", "avg30_nat", False), "v1.4": ("placeholder_v1_4", "avg30_nat", True)}


def load_walkforward():
    spec = importlib.util.spec_from_file_location("planA_walkforward", Path(__file__).with_name("planA_walkforward.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def walkforward_forecasts(wf, tab, pre, fit_origin):
    """{version: {metric: (coefs, {h: forecast}, ref)}} from the script's own functions."""
    out = {}
    for version, (_, base_kind, use_long) in VERSIONS.items():
        out[version] = {}
        for m in wf.METRICS:
            coefs = wf.fit_coefs(tab, pre, m, fit_origin, base_kind, use_long)
            if version in wf.CANDIDATES:
                assert wf.CANDIDATES[version][1:3] == (base_kind, use_long)
                fc, ref, _ = wf.forecast_at(pre, m, fit_origin, version, coefs)
            else:                                           # v1.4: forecast_at's body with v1.4's definition
                ref, l_raw, l_long, base = pre[fit_origin][m]
                g = wf.graph_shift(coefs[0], coefs[1], l_raw, l_long, use_long)
                fc = {h: wf.deploy(m, base[h][base_kind], g) for h in range(1, 8)}
            out[version][m] = (coefs, fc, ref)
    return out


def digest(frame: pd.DataFrame) -> str:
    frame = frame.sort_values(["metric", "city_id", "year"]).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()[:16]


def main():
    wf = load_walkforward()
    cities, coords_wf, tab = wf.load()
    d = wf.distances(coords_wf)
    dn = d.copy()
    np.fill_diagonal(dn, np.inf)
    order = np.argsort(dn, axis=1, kind="stable")
    kernel = wf.gauss_kernel(d)

    annual = pd.read_csv(ROOT / "data" / "annual_valid.csv", dtype={"city_id": str})
    annual = annual[annual.year <= LAST]
    assert annual.year.max() <= LAST, "parity uses years <= 1982 only"
    assert sorted(annual.city_id.unique()) == cities
    info = pd.read_csv(ROOT / "data" / "city_info.csv", dtype={"city_id": str}).set_index("city_id")
    adapters = {v: __import__(spec[0]) for v, spec in VERSIONS.items()}
    for version, module in adapters.items():
        assert module.MODEL_NAME == "Placeholder_" + version

    failures, table, runtimes = [], [], {}
    for fit_origin in ORIGINS:
        pre = wf.precompute(tab, kernel, order, range(fit_origin - wf.TRAIN_YEARS, fit_origin + 1))
        expected = walkforward_forecasts(wf, tab, pre, fit_origin)
        train = annual[annual.year <= fit_origin].copy()
        assert int(train.year.max()) <= fit_origin, "adapter input must be cut at the fit origin"
        targets = pd.MultiIndex.from_product([cities, range(fit_origin + 1, fit_origin + 8), wf.METRICS],
                                             names=["city_id", "year", "metric"]).to_frame(index=False)
        base = reference(train, targets, fit_origin, 42, method="average")
        print(f"\n== F = {fit_origin}: national gap N = {pre[fit_origin]['n_gap']:+.4f} F, "
              f"increment I = {pre[fit_origin]['inc']:.4f} F ==")
        for version, module in adapters.items():
            assert int(train.year.max()) <= fit_origin
            coords = module._check_coords(info, cities)
            output, record, state = module.run_forecast(train, targets, fit_origin, 42, coords)
            merged = output.merge(base, on=["city_id", "year", "metric"], suffixes=("", "_ref"), validate="one_to_one")
            row = {"F": fit_origin, "version": version}
            for m in wf.METRICS:
                coefs, fc, ref = expected[version][m]
                got = (state[m]["beta"], state[m]["gamma"])
                if got != (float(coefs[0]), float(coefs[1])):
                    failures.append(f"F={fit_origin} {version} {m}: coefficients {got} != script {coefs}")
                wide = output[output.metric == m].pivot(index="city_id", columns="year", values="predicted_value")
                wide = wide.reindex(index=cities)
                has_ref = np.isfinite(ref)
                if not np.array_equal(has_ref, state[m]["has_ref"]):
                    failures.append(f"F={fit_origin} {version} {m}: rows with a 30-year mean differ")
                diff = max(float(np.max(np.abs(wide[fit_origin + h].to_numpy(float)[has_ref] - fc[h][has_ref])))
                           for h in range(1, 8))
                row[m] = diff
                if not diff <= TOL:
                    failures.append(f"F={fit_origin} {version} {m}: max abs difference {diff:.3e} > {TOL}")
                fallback = merged[(merged.metric == m) & merged.city_id.isin(np.array(cities)[~has_ref])]
                if not np.array_equal(fallback.predicted_value.to_numpy(), fallback.predicted_value_ref.to_numpy()):
                    failures.append(f"F={fit_origin} {version} {m}: fallback rows differ from reference average")
                row[m + " coefs"] = got
                row[m + " fallback"] = int((~has_ref).sum())
            if module.NATIONAL:
                if record["national"]["N"] != pre[fit_origin]["n_gap"] or \
                        record["national"]["I_temp"] != pre[fit_origin]["inc"]:
                    failures.append(f"F={fit_origin} {version}: N or I differs from the script")
            table.append(row)

        if fit_origin == TIMED_ORIGIN:
            for version, module in adapters.items():
                coords = module._check_coords(info, cities)
                times, digests = [], []
                for _ in range(2):
                    started = time.perf_counter()
                    output, _, _ = module.run_forecast(train, targets, fit_origin, 42, coords)
                    times.append(time.perf_counter() - started)
                    digests.append(digest(output))
                with tempfile.TemporaryDirectory() as artifacts, \
                        mock.patch.dict(os.environ, {"PROTOSTAR_ASSET_ROOT": str(ROOT), "PROTOSTAR_ARTIFACT_DIR": artifacts}):
                    started = time.perf_counter()
                    entry = module.forecast(train, targets, fit_origin, 42)
                    entry_time = time.perf_counter() - started
                    record = json.loads((Path(artifacts) / "placeholder_selection.json").read_text(encoding="utf-8"))
                if digest(entry) != digests[0]:
                    failures.append(f"{version}: forecast() output differs from run_forecast()")
                if digests[0] != digests[1]:
                    failures.append(f"{version}: two runs differ")
                runtimes[version] = (times, digests, entry_time, record)

    print("\n== Max abs difference, adapter minus walk-forward, rows with a 30-year mean (all 7 leads) ==")
    print(f"{'F':>5} {'version':8}" + "".join(f"{m:>18}" for m in wf.METRICS))
    for row in table:
        print(f"{row['F']:>5} {row['version']:8}" + "".join(f"{row[m]:>18.3e}" for m in wf.METRICS))
    print("\n== Coefficients (beta, gamma), identical in adapter and script; rows without a 30-year mean ==")
    for row in table:
        print(f"{row['F']:>5} {row['version']:8}" + "".join(
            f"  {m}: ({row[m + ' coefs'][0]:.2f}, {row[m + ' coefs'][1]:.2f}) fallback {row[m + ' fallback']}"
            for m in wf.METRICS))
    print(f"\n== Runtime at F = {TIMED_ORIGIN} (210 cities; run_forecast twice, then forecast() with the asset load) ==")
    for version, (times, digests, entry_time, record) in runtimes.items():
        extra = ""
        if "national" in record:
            nat = record["national"]
            extra = (f"; N {nat['N']:+.4f}, I {nat['I_temp']:.4f}, cap bound {nat['cap_bound']}, "
                     f"mean CDD increment {nat['mean_cdd_increment']:.3f}")
        print(f"   {version}: {times[0]:.2f} s and {times[1]:.2f} s; forecast() {entry_time:.2f} s "
              f"(artifact runtime_s {record['runtime_s']}); sha256 {digests[0]} / {digests[1]} "
              f"identical={digests[0] == digests[1]}{extra}")
    if failures:
        print("\nPARITY FAILED:")
        for line in failures:
            print("   " + line)
        sys.exit(1)
    print(f"\nPARITY OK: every version, metric and origin within {TOL:g}; coefficients identical; fallback rows exact.")


if __name__ == "__main__":
    main()
