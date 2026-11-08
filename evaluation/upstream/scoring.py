"""Scoring: MAE per metric, skill against the 30-year average, Score 0–100.

`score()` is pure (no network, no files, no Gradio), so the tests can check it
against known answers. `load_answer_key()` is the only part that reads files.

    MAE_m   = mean |predicted − actual|            over scored rows of the split
    S_m     = 1 − MAE_m / MAE_ref,m                (ref = 30-year average baseline)
    Score   = 100 × max(0, mean of the four S_m)

Values are kept at full precision; round only for display.
"""

import math
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from config import (ANSWER_KEY_DIR, ANSWER_KEY_FILE, ANSWER_KEY_REPO,
                    BASELINE_SCORES_FILE, METRICS, REFERENCE_MAE_FILE, TOKEN)

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


def validate_truth(answer_key: pd.DataFrame, reference_mae: pd.DataFrame,
                   baseline_scores: pd.DataFrame) -> None:
    """Raise ScoringError if the answer key, reference MAEs or baselines are unusable.

    Checked once per cycle, before any team is scored: a broken answer key must fail
    the whole cycle (keeping the previous leaderboard), not show up as every team
    "could not be scored".
    """
    missing = set(KEY + ["actual_value", "split", "scored"]) - set(answer_key.columns)
    if missing:
        raise ScoringError(f"answer key is missing columns {sorted(missing)}")
    for name, df in (("reference MAE", reference_mae), ("baseline scores", baseline_scores)):
        cols = set(METRICS) | {"split"} | ({"baseline", "Score"} if name == "baseline scores" else set())
        if not cols <= set(df.columns):
            raise ScoringError(f"{name} is missing columns {sorted(cols - set(df.columns))}")
    for split in SPLITS:
        if not ((answer_key["split"] == split) & answer_key["scored"].astype(bool)).any():
            raise ScoringError(f"answer key has no scored rows for split {split!r}")
        if (reference_mae["split"] == split).sum() != 1:
            raise ScoringError(f"reference MAE needs exactly one row for split {split!r}")
        if not (reference_mae.loc[reference_mae["split"] == split, list(METRICS)] > 0).all(axis=None):
            raise ScoringError(f"reference MAE for split {split!r} must be positive")


def _path(filename: str) -> str:
    """Local file under ANSWER_KEY_DIR (tests) or a download from the private repo."""
    if ANSWER_KEY_DIR:
        return str(Path(ANSWER_KEY_DIR) / filename)
    return hf_hub_download(ANSWER_KEY_REPO, filename, repo_type="dataset", token=TOKEN)


def load_answer_key() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(answer_key, reference_mae, baseline_scores), read from ANSWER_KEY_DIR or the Hub."""
    answer_key = pd.read_csv(_path(ANSWER_KEY_FILE), dtype={"city_id": str, "metric": str, "split": str})
    reference_mae = pd.read_csv(_path(REFERENCE_MAE_FILE))
    baseline_scores = pd.read_csv(_path(BASELINE_SCORES_FILE))
    return answer_key, reference_mae, baseline_scores
