"""Shared setup for the evaluation scripts: put code/ on the import path (as the worker does)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

DEFAULT_SUITE = ROOT / "outputs/evaluation/suite-v3"
DEFAULT_RUNS = ROOT / "outputs/evaluation/runs"
DEFAULT_REPORT = ROOT / "outputs/evaluation/report"
