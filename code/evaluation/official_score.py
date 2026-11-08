"""Official pure scorer, extracted unchanged from the pinned MIT-licensed Space.

Only network/config imports are replaced by local constants. See
../../evaluation/upstream/manifest.json and LICENSE for source and attribution.
"""
import math
import pandas as pd

METRICS = ("annual_avg_temp", "total_precip_in", "hottest_day_temp", "cdd")
KEY = ["city_id", "year", "metric"]
SPLITS = ("live", "final")

class ScoringError(ValueError):
    """The submission cannot be scored (should not happen after check_forecasts)."""


def score(submission: pd.DataFrame, answer_key: pd.DataFrame,
          reference_mae: pd.DataFrame, split: str) -> dict:
    """Score one submission on one split ("live" or "final").

    submission:    columns city_id, year, metric, predicted_value
    answer_key:    columns city_id, year, metric, actual_value, split, scored
    reference_mae: one row per split, one column per metric (30-year average MAE)

    Returns {"score", "mae": {metric: ...}, "skill": {metric: ...}, "n_rows"}.
    """
    if split not in SPLITS:
        raise ScoringError(f"unknown split {split!r}; expected one of {SPLITS}")

    truth = answer_key[(answer_key["split"] == split) & answer_key["scored"].astype(bool)]
    truth = truth[KEY + ["actual_value"]]
    pred = submission[KEY + ["predicted_value"]].astype({"city_id": str, "year": int, "metric": str})
    if pred.duplicated(KEY).any():
        raise ScoringError("duplicate (city_id, year, metric) rows in the submission")

    rows = truth.merge(pred, on=KEY, how="left")
    value = pd.to_numeric(rows["predicted_value"], errors="coerce")
    missing = value.isna() | ~value.map(math.isfinite)
    if missing.any():
        first = rows.loc[missing, KEY].iloc[0].tolist()
        raise ScoringError(f"{int(missing.sum())} scored rows have no finite prediction (first: {first})")

    error = (value - rows["actual_value"]).abs()
    mae = error.groupby(rows["metric"]).mean()

    ref = reference_mae.set_index("split").loc[split]
    mae_out, skill_out = {}, {}
    for m in METRICS:
        if m not in mae.index:
            raise ScoringError(f"no scored rows for metric {m!r} in split {split!r}")
        mae_out[m] = float(mae[m])
        skill_out[m] = 1.0 - mae_out[m] / float(ref[m])

    mean_skill = sum(skill_out.values()) / len(METRICS)
    return {"score": 100.0 * max(0.0, mean_skill),
            "mae": mae_out,
            "skill": skill_out,
            "n_rows": int(len(rows))}
