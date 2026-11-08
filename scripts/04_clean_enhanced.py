"""Check and clean the enhanced inputs (city facts and ocean indices) and assemble the next cleaned release.

Usage:   uv run python scripts/04_clean_enhanced.py [--out outputs/clean_v2] [--no-weather]

Needs the releases locally (`uv run python scripts/line.py pull enhance-v1 enhance-v2 clean-v1`). Writes:

    static/city_static_clean.csv            the 210 city facts, model-ready (constants dropped, units fixed, flags)
    static/qc/static_issues.csv             every change and flag, one row each
    static/manifest.json                    input and code hashes, constants, summary, model columns
    checks/enhance-ocean-verification.json  the recomputation of every enhance-v2 ocean snapshot (nothing changed)
    origin-YYYY/                            clean-v1's per-origin weather tables, byte for byte (unless --no-weather)

Hand it on with `line.py release clean <out> --upstream clean-v1 enhance-v1 --gate-eligible`. Page:
documentation/10-enhanced.md. Data rule: no weather value is read; the city facts are fixed structural facts.
"""

import argparse
from pathlib import Path

from protostar_challenge.cleaning_enhanced import run
from protostar_challenge.config import ROOT

RELEASES = ROOT / "data/processed/releases"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/clean_v2")
    parser.add_argument("--static", type=Path, default=RELEASES / "enhance-v1")
    parser.add_argument("--ocean", type=Path, default=RELEASES / "enhance-v2")
    parser.add_argument("--weather", type=Path, default=RELEASES / "clean-v1")
    parser.add_argument("--no-weather", action="store_true", help="do not copy the clean-v1 weather tables")
    args = parser.parse_args()
    manifest = run(args.static, args.ocean, ROOT / "data/city_info.csv", args.out,
                   None if args.no_weather else args.weather)
    summary = manifest["static"]
    print(f"city facts: {summary['rows']} cities, dropped {len(summary['constant_columns_dropped'])} constant columns, "
          f"{summary['resolution_converted']} resolutions converted, {summary['elevation_catalog_disagrees']} elevation "
          f"flags, location uncertain: {', '.join(summary['location_uncertain'])}")
    print(f"ocean indices: verification passed={manifest['ocean_passed']}")
    print(f"weather files copied from {manifest['inputs']['weather_release']}: {manifest['inputs']['weather_files_copied']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
