"""Create enhanced/<pinned-tag> using verified immutable assembly-line releases.

Run from any station after integrating this helper:
    uv run python scripts/materialize_enhanced.py

No weather download, reconstruction, model scoring, or release mutation occurs.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import stat

import line

ROOT = Path(__file__).resolve().parents[1]


def check_location(path: Path) -> None:
    if path.is_symlink() or path.is_junction():
        raise line.LineError(f"Expected a real local directory, not a link: {path}")
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise line.LineError("Enhanced dataset path must remain inside this worktree")


def main():
    pointer = json.loads((ROOT / "enhanced/latest.json").read_text(encoding="utf-8"))
    tag = pointer["release_id"]
    if not re.fullmatch(r"enhance-v[1-9]\d*", tag):
        raise line.LineError("Invalid enhancement release tag")
    if pointer["directory"] != f"enhanced/{tag}" or pointer["canonical_asset_directory"] != f"data/processed/releases/{tag}":
        raise line.LineError("Pointer paths disagree with the release tag")
    check_location(ROOT / "enhanced")
    check_location(ROOT / pointer["directory"])

    for destination in (ROOT / line.PULLED, ROOT / "enhanced"):
        manifest = line.pull_release(tag, destination=destination)
        folder = destination / tag
        if line.sha256_file(folder / "release.json") != pointer["release_manifest_sha256"]:
            raise line.LineError("Release manifest does not match the pinned latest dataset")
        expected = (tag, pointer["source_git_commit"], pointer["gate_eligible"], pointer["upstream"])
        actual = (manifest["id"], manifest["git_commit"], manifest["gate_eligible"], manifest["upstream"])
        if actual != expected:
            raise line.LineError("Release identity or eligibility disagrees with the pointer")
        print(f"Verified {tag}: {len(manifest['files'])} files in {folder}")

    tagged = ROOT / pointer["directory"]
    for path in tagged.rglob("*"):
        if path.is_file():
            path.chmod(stat.S_IREAD)
    print(f"Latest enhanced dataset: {tag}; gate_eligible={pointer['gate_eligible']}")
    print("Use canonical release asset paths for evaluation lineage; preserve the exploratory flag.")


if __name__ == "__main__":
    main()
