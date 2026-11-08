"""Mirror every complete, verified evaluation run into MLflow (idempotent).

    uv run python scripts/mlflow_sync.py
    uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db

Set MLFLOW_TRACKING_URI to use a shared tracking server instead of the local store.
Run it again after copying a teammate's run folders into outputs/evaluation/runs.
"""
import argparse

import _harness  # noqa: F401
from _harness import DEFAULT_RUNS

from evaluation.tracking import sync, tracking_uri


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    args = parser.parse_args()
    print(f"Tracking store: {tracking_uri()}")
    counts = sync(args.runs)
    print(", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
