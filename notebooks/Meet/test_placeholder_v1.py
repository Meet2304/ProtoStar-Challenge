"""Checks for Placeholder_v1.1-v1.4 (plan A r6.2, section 5 'Checks before scoring') on synthetic data only.

No weather data is read. Run from the repository root:
    uv run python -m unittest notebooks/Meet/test_placeholder_v1.py -v

The parity of the adapters with the walk-forward reference on real pre-1983 data is a separate script:
docs/plans/creative/analysis/planA_parity.py.
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "code")]

import placeholder_v1_1  # noqa: E402
import placeholder_v1_2  # noqa: E402
import placeholder_v1_3  # noqa: E402
import placeholder_v1_4  # noqa: E402
from common import METRICS  # noqa: E402
from evaluation.adapters import reference  # noqa: E402
from evaluation.core import validate_predictions  # noqa: E402

N_ROWS, N_COLS = 14, 15
N_CITIES = N_ROWS * N_COLS                   # 210, like the challenge
FIRST, ORIGIN, INNER = 1880, 1990, 1970
CITIES = [f"C{i:03d}" for i in range(N_CITIES)]
LAT = np.repeat(np.linspace(30.0, 48.0, N_ROWS), N_COLS)
LON = np.tile(np.linspace(-120.0, -72.0, N_COLS), N_ROWS)
COORDS = pd.DataFrame({"lat": LAT, "lon": LON}, index=CITIES)
KEY = ["city_id", "year", "metric"]
WINDOW_FALLBACK, HISTORY_FALLBACK, MEDIAN_FALLBACK = "C000", "C001", "C002"


def synthetic(seed=0, recent=0.0, first=FIRST, last=ORIGIN) -> pd.DataFrame:
    """National, regional and persistent local temperature anomalies plus noise; CDD responds to temperature.

    recent: a shift (F) added to temperature in the last ten years, so the national gap N is about 2/3 of it.
    Three cities exercise reference()'s fallbacks: C000 (5 valid years in the 30-year window), C001 (none in the
    window: full-history mean) and C002 (no valid precipitation: metric-wide median).
    """
    rng = np.random.default_rng(seed)
    years = np.arange(first, last + 1)
    ny = len(years)

    def ar1(phi, sd, rows):
        e = rng.normal(0.0, sd, (rows, ny))
        x = np.zeros_like(e)
        for j in range(ny):
            x[:, j] = (phi * x[:, j - 1] if j else 0.0) + e[:, j]
        return x

    lat = LAT[:, None]
    d = placeholder_v1_1.distances(COORDS)
    smooth = np.exp(-(d / 800.0) ** 2)
    smooth /= smooth.sum(1, keepdims=True)
    national = ar1(0.8, 0.4, 1)
    regional = smooth @ ar1(0.5, 1.0, N_CITIES)
    local = ar1(0.9, 0.25, N_CITIES)
    climate = 70.0 - 0.8 * (lat - 30.0)
    temp = climate + national + regional + local + rng.normal(0.0, 0.6, (N_CITIES, ny))
    temp = temp + np.where(years > last - 10, recent, 0.0)[None, :]
    hottest = 95.0 - 0.3 * (lat - 30.0) + 0.8 * (temp - climate) + 1.5 * local + rng.normal(0.0, 1.5, (N_CITIES, ny))
    precip = np.maximum(35.0 + 4.0 * ar1(0.8, 0.5, N_CITIES) + rng.normal(0.0, 5.0, (N_CITIES, ny)), 0.0)
    cdd = np.maximum(1500.0 - 40.0 * (lat - 30.0) + 150.0 * (temp - climate) + rng.normal(0.0, 60.0, (N_CITIES, ny)), 0.0)
    values = {"annual_avg_temp": temp, "total_precip_in": precip, "hottest_day_temp": hottest, "cdd": cdd}
    missing = rng.random((4, N_CITIES, ny)) < 0.03
    for k, metric in enumerate(METRICS):
        v = np.where(missing[k], np.nan, values[metric])
        v[CITIES.index(WINDOW_FALLBACK), years > 1965] = np.nan
        v[CITIES.index(HISTORY_FALLBACK), years > 1950] = np.nan
        values[metric] = v
    values["total_precip_in"][CITIES.index(MEDIAN_FALLBACK)] = np.nan
    frame = pd.DataFrame({"city_id": np.repeat(CITIES, ny), "year": np.tile(years, N_CITIES)})
    frame["name"] = frame.city_id
    for metric in METRICS:
        frame[metric] = values[metric].reshape(-1)
    return frame


def grid(origin=ORIGIN) -> pd.DataFrame:
    return pd.MultiIndex.from_product([CITIES, range(origin + 1, origin + 8), list(METRICS)], names=KEY).to_frame(index=False)


def digest(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()


def with_reference(output: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    base = reference(train, grid(), ORIGIN, 42, method="average")
    return output.merge(base, on=KEY, suffixes=("", "_ref"), validate="one_to_one")


def city_rows(frame: pd.DataFrame, metric: str, mask: np.ndarray) -> pd.DataFrame:
    return frame[(frame.metric == metric) & frame.city_id.isin(np.array(CITIES)[mask])]


TRAIN = synthetic()


class VersionChecks:
    """Shared checks; each subclass sets `model`."""

    model = None

    @classmethod
    def setUpClass(cls):
        cls.train = TRAIN
        cls.output, cls.record, cls.state = cls.model.run_forecast(cls.train, grid(), ORIGIN, 42, COORDS)
        cls.merged = with_reference(cls.output, cls.train)

    def no_increment(self):
        return 0.0 if self.model.NATIONAL else self.model.KAPPA

    # ---------------------------------------------------------------- output grid and determinism
    def test_complete_finite_grid(self):
        self.assertEqual(len(self.output), N_CITIES * 7 * 4)
        self.assertEqual(list(self.output.columns), KEY + ["predicted_value"])
        self.assertFalse(self.output.duplicated(KEY).any())
        self.assertTrue(np.isfinite(self.output.predicted_value).all())
        validate_predictions(self.output, grid())

    def test_deterministic(self):
        again, record, _ = self.model.run_forecast(self.train, grid(), ORIGIN, 42, COORDS)
        pd.testing.assert_frame_equal(self.output, again)
        self.assertEqual(digest(self.output), digest(again))
        self.assertEqual({k: v for k, v in self.record.items() if k != "runtime_s"},
                         {k: v for k, v in record.items() if k != "runtime_s"})

    # ---------------------------------------------------------------- equality checks (plan section 5)
    def test_zero_coefficients_reproduce_reference_average_exactly(self):
        zero = {m: (0.0, 0.0) for m in METRICS}
        output, record, state = self.model.run_forecast(self.train, grid(), ORIGIN, 42, COORDS, coefs=zero,
                                                        kappa=self.no_increment())
        merged = with_reference(output, self.train)
        for metric in METRICS:
            rows = city_rows(merged, metric, state[metric]["has_ref"])
            self.assertGreater(len(rows), 0)
            np.testing.assert_array_equal(rows.predicted_value.to_numpy(), rows.predicted_value_ref.to_numpy())
        np.testing.assert_array_equal(merged.predicted_value.to_numpy(), merged.predicted_value_ref.to_numpy())
        self.assertEqual(record["status"], "coefficients_overridden")

    def test_fallback_rows_equal_reference_average(self):
        for metric in METRICS:
            s = self.state[metric]
            missing = ~s["has_ref"]
            expected = {WINDOW_FALLBACK, HISTORY_FALLBACK} | ({MEDIAN_FALLBACK} if metric == "total_precip_in" else set())
            self.assertTrue(expected <= set(np.array(CITIES)[missing]), metric)
            np.testing.assert_array_equal(s["G"][missing], 0.0)
            np.testing.assert_array_equal(s["I"][missing], 0.0)
            rows = city_rows(self.merged, metric, missing)
            self.assertEqual(len(rows), 7 * int(missing.sum()))
            np.testing.assert_array_equal(rows.predicted_value.to_numpy(), rows.predicted_value_ref.to_numpy())

    def test_rows_with_a_30_year_mean_are_deploy_of_ref30_plus_increment_plus_G(self):
        for metric in METRICS:
            s = self.state[metric]
            rows = city_rows(self.merged, metric, s["has_ref"])
            pos = rows.city_id.map({c: i for i, c in enumerate(CITIES)}).to_numpy()
            expected = self.model.deploy(metric, rows.predicted_value_ref.to_numpy() + s["I"][pos], s["G"][pos])
            np.testing.assert_array_equal(rows.predicted_value.to_numpy(), expected)

    def test_zero_cross_city_median_adjustment(self):
        for metric in METRICS:
            s = self.state[metric]
            g = s["G"][s["defined"]]
            self.assertGreater(len(g), N_CITIES // 2, metric)
            self.assertLess(abs(float(np.median(g))), 1e-12, metric)
            np.testing.assert_array_equal(s["G"][~s["defined"]], 0.0)

    def test_planted_persistent_relative_anomaly_is_recovered(self):
        self.assertGreater(self.state["annual_avg_temp"]["beta"], 0.0)
        self.assertGreater(self.record["metrics"]["annual_avg_temp"]["share_G_nonzero"], 0.5)

    def test_non_negative_metrics_are_clipped(self):
        for metric in self.model.NON_NEGATIVE:
            np.testing.assert_array_equal(self.model.deploy(metric, np.array([1.0, 5.0]), np.array([-3.0, -1.0])),
                                          [0.0, 4.0])
            self.assertTrue((self.output[self.output.metric == metric].predicted_value >= 0).all())

    # ---------------------------------------------------------------- information boundary
    def test_refuses_rows_after_origin(self):
        with self.assertRaises(ValueError):
            self.model.run_forecast(synthetic(last=ORIGIN + 1), grid(), ORIGIN, 42, COORDS)

    def test_graph_terms_ignore_years_after_t(self):
        t = 1960
        changed = self.train.copy()
        late = changed.year > t
        changed.loc[late, list(METRICS)] += 1e6
        a = self.model.build_panel(self.train, CITIES, ORIGIN)
        b = self.model.build_panel(changed, CITIES, ORIGIN)
        kernel, order = self.model.graph(COORDS)
        for metric in METRICS:
            for x, y in zip(self.model.terms_at(a, metric, t, kernel), self.model.terms_at(b, metric, t, kernel)):
                np.testing.assert_array_equal(x, y, err_msg=metric)
        ref_a = self.model.terms_at(a, "annual_avg_temp", t, kernel)[0]
        ref_b = self.model.terms_at(b, "annual_avg_temp", t, kernel)[0]
        self.assertEqual(self.model.national_gap(a.arrays["annual_avg_temp"], a.col(t), ref_a),
                         self.model.national_gap(b.arrays["annual_avg_temp"], b.col(t), ref_b))
        np.testing.assert_array_equal(self.model.cdd_response(a, t, order), self.model.cdd_response(b, t, order))

    def test_inner_fit_ignores_years_after_fit_origin(self):
        """Cutting the table at the fit origin, or adding huge values after it, changes nothing in the fit."""
        changed = self.train.copy()
        late = changed.year > INNER
        changed.loc[late, list(METRICS)] += 5000.0
        kernel, order = self.model.graph(COORDS)
        fits = [self.model.fit_at(self.model.build_panel(frame, CITIES, last), kernel, order, INNER)
                for frame, last in ((self.train, ORIGIN), (changed, ORIGIN), (self.train, INNER))]
        for metric in METRICS:
            first = fits[0][metric]
            self.assertIsNotNone(first["fit_rows"])
            for other in fits[1:]:
                for key in ("beta", "gamma", "fit_rows", "centring_median"):
                    self.assertEqual(first[key], other[metric][key], f"{metric} {key}")
                for key in ("ref", "has_ref", "defined", "G", "I"):
                    np.testing.assert_array_equal(first[key], other[metric][key], err_msg=f"{metric} {key}")
        if self.model.NATIONAL:
            self.assertEqual(fits[0]["national"], fits[1]["national"])
            self.assertEqual(fits[0]["national"], fits[2]["national"])

    def test_training_targets_never_pass_fit_origin(self):
        panel = self.model.build_panel(self.train, CITIES, ORIGIN)
        kernel, order = self.model.graph(COORDS)
        pre = self.model.precompute(panel, kernel, order, range(INNER - self.model.TRAIN_YEARS, INNER + 1),
                                    self.model.KAPPA)
        for metric in METRICS:
            y, _, _, _, years = self.model.coef_rows(panel, pre, metric, INNER)
            self.assertEqual(max(years), INNER)
            self.assertEqual(min(years), INNER - self.model.TRAIN_YEARS + min(self.model.FIT_LEADS))
            self.assertEqual(len(years), len(y))
        self.assertTrue(all(t <= INNER for t in pre))

    # ---------------------------------------------------------------- entry point and artifact
    def test_entry_point_reads_coordinates_from_the_asset_and_writes_the_record(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as artifacts:
            (Path(root) / "data").mkdir()
            COORDS.rename_axis("city_id").assign(name=CITIES).reset_index()[["city_id", "name", "lat", "lon"]].to_csv(
                Path(root) / "data" / "city_info.csv", index=False)
            with mock.patch.dict(os.environ, {"PROTOSTAR_ASSET_ROOT": root, "PROTOSTAR_ARTIFACT_DIR": artifacts}):
                output = self.model.forecast(self.train, grid(), ORIGIN, 42)
            record = json.loads((Path(artifacts) / "placeholder_selection.json").read_text(encoding="utf-8"))
        pd.testing.assert_frame_equal(output, self.output)
        self.assertEqual(record["model"], self.model.MODEL_NAME)
        self.assertGreater(record["runtime_s"], 0.0)
        for metric in METRICS:
            entry = record["metrics"][metric]
            for key in ("beta", "mean_abs_G", "share_G_nonzero", "centring_median", "median_forecast_shift"):
                self.assertIn(key, entry, f"{metric} {key}")
            self.assertEqual("gamma" in entry, self.model.USE_LONG)
        if self.model.NATIONAL:
            for key in ("N", "I_temp", "cap_bound", "mean_cdd_increment"):
                self.assertIn(key, record["national"])
        else:
            self.assertNotIn("national", record)

    def test_entry_point_takes_no_parameters(self):
        with self.assertRaises(ValueError):
            self.model.forecast(self.train, grid(), ORIGIN, 42, beta=0.5)

    def test_entry_point_needs_the_harness_snapshot(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("PROTOSTAR_ASSET_ROOT", None)
            with self.assertRaises(RuntimeError):
                self.model.forecast(self.train, grid(), ORIGIN, 42)


class LongTermChecks:
    """v1.2 and v1.4: gamma = 0 is needed as well as beta = 0 (Sol's r6 example)."""

    def test_gamma_alone_moves_the_forecast(self):
        l_raw, l_long = np.array([1.0, 1.0, 1.0]), np.array([-2.0, 0.0, 2.0])
        np.testing.assert_allclose(self.model.graph_shift(0.0, 0.4, l_raw, l_long, True), [-0.8, 0.0, 0.8])
        np.testing.assert_array_equal(self.model.graph_shift(0.0, 0.0, l_raw, l_long, True), [0.0, 0.0, 0.0])

    def test_defined_where_either_part_is(self):
        g = self.model.graph_shift(0.5, 0.5, np.array([np.nan, 1.0, np.nan]), np.array([2.0, np.nan, np.nan]), True)
        self.assertEqual(g[2], 0.0)
        self.assertNotEqual(g[0], g[1])


class NationalChecks:
    """v1.3 and v1.4: the national increment I."""

    def test_increment_is_zero_when_the_gap_is_not_positive(self):
        rho = np.array([0.0, 120.0, 300.0])
        for gap in (-0.4, 0.0):
            shift, inc = self.model.national_shift(gap, rho, self.model.KAPPA, self.model.CAP_F, floor=True)
            self.assertEqual(inc, 0.0)
            for metric in METRICS:
                np.testing.assert_array_equal(shift[metric], 0.0)

    def test_cap_bounds_the_increment(self):
        rho = np.array([0.0, 120.0, 300.0])
        shift, inc = self.model.national_shift(3.0, rho, self.model.KAPPA, self.model.CAP_F, floor=True)
        self.assertEqual(inc, 0.5)
        np.testing.assert_array_equal(shift["annual_avg_temp"], 0.5)
        np.testing.assert_array_equal(shift["cdd"], 0.5 * rho)
        np.testing.assert_array_equal(shift["total_precip_in"], 0.0)
        np.testing.assert_array_equal(shift["hottest_day_temp"], 0.0)

    def scenario(self, recent):
        train = synthetic(recent=recent)
        output, record, state = self.model.run_forecast(train, grid(), ORIGIN, 42, COORDS)
        return train, with_reference(output, train), record, state

    def assert_ref30_plus_increment_plus_g(self, merged, state):
        for metric in METRICS:
            s = state[metric]
            rows = city_rows(merged, metric, s["has_ref"])
            pos = rows.city_id.map({c: i for i, c in enumerate(CITIES)}).to_numpy()
            np.testing.assert_array_equal(
                rows.predicted_value.to_numpy(),
                self.model.deploy(metric, rows.predicted_value_ref.to_numpy() + s["I"][pos], s["G"][pos]))
            fallback = city_rows(merged, metric, ~s["has_ref"])
            np.testing.assert_array_equal(fallback.predicted_value.to_numpy(), fallback.predicted_value_ref.to_numpy())
            np.testing.assert_array_equal(s["I"][~s["has_ref"]], 0.0)

    def test_cool_recent_decade_gives_no_increment(self):
        _, merged, record, state = self.scenario(-1.5)
        self.assertLess(record["national"]["N"], 0.0)
        self.assertEqual(record["national"]["I_temp"], 0.0)
        self.assertFalse(record["national"]["cap_bound"])
        for metric in METRICS:
            np.testing.assert_array_equal(state[metric]["I"], 0.0)
        self.assert_ref30_plus_increment_plus_g(merged, state)        # ref30 + G with this version's own beta

    def test_warm_recent_decade_binds_the_cap(self):
        _, merged, record, state = self.scenario(3.0)
        self.assertGreater(record["national"]["N"], self.model.CAP_F)
        self.assertEqual(record["national"]["I_temp"], 0.5)
        self.assertTrue(record["national"]["cap_bound"])
        temp = state["annual_avg_temp"]
        np.testing.assert_array_equal(temp["I"][temp["has_ref"]], 0.5)
        self.assertTrue((state["cdd"]["I"] >= 0).all())
        self.assertGreater(record["national"]["mean_cdd_increment"], 0.0)
        np.testing.assert_array_equal(state["total_precip_in"]["I"], 0.0)
        np.testing.assert_array_equal(state["hottest_day_temp"]["I"], 0.0)
        self.assert_ref30_plus_increment_plus_g(merged, state)

    def test_kappa_zero_with_fitted_coefficients_has_no_increment(self):
        train = synthetic(recent=3.0)
        _, record, state = self.model.run_forecast(train, grid(), ORIGIN, 42, COORDS, kappa=0.0)
        self.assertEqual(record["national"]["I_temp"], 0.0)
        for metric in METRICS:
            np.testing.assert_array_equal(state[metric]["I"], 0.0)


class GraphOnlyChecks:
    """v1.1 and v1.2 have no national increment."""

    def test_has_no_increment(self):
        for metric in METRICS:
            np.testing.assert_array_equal(self.state[metric]["I"], 0.0)
        with self.assertRaises(ValueError):
            self.model.run_forecast(self.train, grid(), ORIGIN, 42, COORDS, kappa=0.0)


class PlaceholderV11(VersionChecks, GraphOnlyChecks, unittest.TestCase):
    model = placeholder_v1_1


class PlaceholderV12(VersionChecks, GraphOnlyChecks, LongTermChecks, unittest.TestCase):
    model = placeholder_v1_2


class PlaceholderV13(VersionChecks, NationalChecks, unittest.TestCase):
    model = placeholder_v1_3


class PlaceholderV14(VersionChecks, NationalChecks, LongTermChecks, unittest.TestCase):
    model = placeholder_v1_4


class FrozenConstants(unittest.TestCase):
    def test_versions_share_the_plan_constants(self):
        for model, name, use_long, national in ((placeholder_v1_1, "Placeholder_v1.1", False, False),
                                                (placeholder_v1_2, "Placeholder_v1.2", True, False),
                                                (placeholder_v1_3, "Placeholder_v1.3", False, True),
                                                (placeholder_v1_4, "Placeholder_v1.4", True, True)):
            self.assertEqual((model.MODEL_NAME, model.USE_LONG, model.NATIONAL), (name, use_long, national))
            self.assertEqual(model.KERNEL_KM, 600.0)
            self.assertEqual(model.MIN_KERNEL_MASS, 1.0)
            self.assertEqual(model.HALF_LIFE, {"annual_avg_temp": 3, "hottest_day_temp": 3, "cdd": 3,
                                               "total_precip_in": 6})
            self.assertEqual((model.MIN_REL_YEARS, model.MIN_LONG_YEARS, model.MIN_VALID_WINDOW), (10, 15, 10))
            self.assertEqual(model.GRID.tolist(), [round(0.05 * i, 2) for i in range(21)])
            self.assertEqual((model.FIT_LEADS, model.TRAIN_YEARS, model.MIN_COEF_ROWS), ((4, 5, 6, 7), 45, 2000))
            self.assertEqual((model.KAPPA, model.CAP_F, model.GAP_MIN_VALID), (0.5, 1.0, 5))
            self.assertEqual((model.RHO_NEIGHBOURS, model.MIN_RESPONSE_YEARS), (15, 15))

    def test_versions_share_one_body(self):
        def body(model):
            text = Path(model.__file__).read_text(encoding="utf-8")
            text = text[text.index("from __future__"):]
            return "\n".join(line for line in text.splitlines()
                             if not line.startswith(("MODEL_NAME =", "USE_LONG =", "NATIONAL =")))
        first = body(placeholder_v1_1)
        for model in (placeholder_v1_2, placeholder_v1_3, placeholder_v1_4):
            self.assertEqual(body(model), first, model.MODEL_NAME)


if __name__ == "__main__":
    unittest.main()
