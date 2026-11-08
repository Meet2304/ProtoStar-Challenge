"""Emit one deterministic synthetic suite identity for cross-OS CI comparison."""
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from evaluation.core import prepare, read_json, write_json


def main(output):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = []
        for city in ("A", "B"):
            for year in range(1950, 2019):
                value = 50 + (year - 1950) / 8 + 10 * (city == "B")
                rows.append({"ID": city, "Year": year, "Miss Tmax": 0, "Miss Tmin": 0,
                             "Miss Prcp": 0, "Avg Temp": value, "ToT Prcp": value / 2,
                             "Max Tmax": value + 40, "cdd": value * 10})
        pd.DataFrame(rows).to_csv(root / "annual.csv", index=False, lineterminator="\n")
        (root / "cities.csv").write_bytes(b"city_id\nA\nB\n")
        config = read_json(ROOT / "evaluation/config.json")
        config.update(origins=[2004, 2011], development_origins=[2004], assessment_origins=[2011])
        write_json(root / "config.json", config)
        result = prepare(root / "suite", root / "config.json", root / "annual.csv", root / "cities.csv")
        Path(output).write_bytes((result["suite_id"] + "\n").encode())
        print(result["suite_id"])


if __name__ == "__main__":
    main(sys.argv[1])
