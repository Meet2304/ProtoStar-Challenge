"""Download three ocean-climate indices as they were published before 2019, for forecasting tests.

    oni  Oceanic Nino Index: El Nino (+) / La Nina (-), affects the following US winter
    pdo  Pacific Decadal Oscillation: a slow North Pacific pattern that persists for years
    amo  Atlantic Multidecadal Oscillation (unsmoothed): a slow North Atlantic pattern

Source: NOAA Physical Sciences Laboratory (PSL), files under psl.noaa.gov/data/correlation/
(called www.esrl.noaa.gov/psd/data/correlation/ until 2019).

Data rule: the live NOAA files run to the present and cannot be cut at 2018, so this script never
calls them. It downloads the Internet Archive's copy of each file, taken on or before 2018-12-31
(the latest such copy), so no value from 2019 or later can be in the download. It then checks
every row with assert_no_post_cutoff(). Using the version that existed in 2018 also avoids later
revisions of the indices: the forecasts only see what was known at the time.

Usage (from the repository root):
    python code/prepare_ocean_indices.py

Writes:
    data/ocean_indices.csv          one row per index, year and month: index, year, month, value
    data/ocean_indices_sources.csv  which archived copy each index came from (URL, capture time)
"""

import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from common import CUTOFF_YEAR, DATA_DIR, assert_no_post_cutoff

ORIGINAL_BASE = "https://www.esrl.noaa.gov/psd/data/correlation/"
FILES = {"oni": "oni.data", "pdo": "pdo.data", "amo": "amon.us.data"}
CAPTURED_BY = f"{CUTOFF_YEAR}1231235959"   # newest archive capture allowed: 2018-12-31
USER_AGENT = "protostar-placeholders-data-prep/1.0"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def latest_capture_before_cutoff(original_url: str) -> tuple[str, str]:
    """(timestamp, original URL) of the newest successful archive capture on or before 2018-12-31."""
    query = urllib.parse.urlencode({
        "url": original_url, "to": CAPTURED_BY, "output": "json",
        "fl": "timestamp,original", "filter": "statuscode:200",
    })
    rows = json.loads(_get(f"https://web.archive.org/cdx/search/cdx?{query}") or b"[]")
    captures = [r for r in rows[1:] if r[0] <= CAPTURED_BY]   # rows[0] is the header
    if not captures:
        raise SystemExit(f"No archived copy of {original_url} captured on or before 2018-12-31.")
    timestamp, original = max(captures, key=lambda r: r[0])
    return timestamp, original


def parse_psl(text: str, name: str) -> pd.DataFrame:
    """Parse a PSL '.data' file: a 'first_year last_year' line, rows of 'year v1 ... v12',
    then a missing-value line and free-text notes."""
    lines = [line.split() for line in text.splitlines() if line.strip()]
    first, last = int(lines[0][0]), int(lines[0][1])
    rows, missing, i = [], None, 1
    while i < len(lines) and len(lines[i]) == 13 and lines[i][0].isdigit() and first <= int(lines[i][0]) <= last:
        rows.append([int(lines[i][0])] + [float(v) for v in lines[i][1:]])
        i += 1
    if i < len(lines) and len(lines[i]) == 1:
        try:
            missing = float(lines[i][0])
        except ValueError:
            pass
    wide = pd.DataFrame(rows, columns=["year"] + list(range(1, 13)))
    long = wide.melt(id_vars="year", var_name="month", value_name="value")
    if missing is not None:
        long.loc[np.isclose(long["value"], missing), "value"] = np.nan
    long.loc[long["value"] <= -99, "value"] = np.nan          # any other missing-value code
    long.insert(0, "index", name)
    return long.dropna(subset=["value"]).sort_values(["year", "month"]).reset_index(drop=True)


def main() -> None:
    tables, sources = [], []
    for name, filename in FILES.items():
        original = ORIGINAL_BASE + filename
        timestamp, captured_url = latest_capture_before_cutoff(original)
        archive_url = f"https://web.archive.org/web/{timestamp}id_/{captured_url}"
        table = parse_psl(_get(archive_url).decode("utf-8", errors="replace"), name)
        assert_no_post_cutoff(table, year_col="year", source=f"{name} ({archive_url})")
        last = table.iloc[-1]
        print(f"{name}: {len(table):,} monthly values, {int(table['year'].min())}-01 to "
              f"{int(last['year'])}-{int(last['month']):02d} (archive copy {timestamp[:8]})")
        tables.append(table)
        sources.append({"index": name, "original_url": original, "capture_timestamp": timestamp,
                        "archive_url": archive_url,
                        "last_value": f"{int(last['year'])}-{int(last['month']):02d}"})
        time.sleep(1)   # be polite to the archive

    out = pd.concat(tables, ignore_index=True)
    out["value"] = out["value"].round(3)
    out.to_csv(DATA_DIR / "ocean_indices.csv", index=False)
    pd.DataFrame(sources).to_csv(DATA_DIR / "ocean_indices_sources.csv", index=False)
    print(f"\nWrote {len(out):,} rows to {DATA_DIR / 'ocean_indices.csv'} "
          f"and sources to {DATA_DIR / 'ocean_indices_sources.csv'}")


if __name__ == "__main__":
    main()
