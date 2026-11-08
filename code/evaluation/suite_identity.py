"""Portable suite identity, separate from exact local artifact integrity.

Version 3 identifies the logical preparation inputs and LF-normalized source,
not platform-dependent derived floating-point bytes. Every physical artifact is
still hashed and bound by a separate manifest fingerprint. Versions 1 and 2 keep
their original raw-manifest identity so archived suites remain readable.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from common import ROOT


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def normalized_digest(path):
    """Hash text with Git LF normalization; retain exact binary input bytes."""
    path = Path(path)
    content = path.read_bytes()
    if path.suffix.lower() in {".py", ".csv", ".json", ".txt", ".toml", ".lock"}:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def preparation_source_pins(root=ROOT):
    """Pin preparation implementation and dependencies, normalized across Git checkouts.

    Whole modules deliberately conservatively include adjacent runtime helpers.
    These document the implementation that created a suite; verification does
    not require an archived suite to have been built by today's implementation.
    """
    root = Path(root)
    names = ("code/common.py", "code/combo_model.py", "code/evaluation/core.py",
             "code/evaluation/adapters.py", "code/evaluation/official_score.py",
             "code/evaluation/suite_identity.py", "code/requirements.txt", "uv.lock",
             "pyproject.toml", "requirements.txt")
    return {name: normalized_digest(root / name) for name in names}


def logical_identity(manifest):
    """Extract only stable preparation inputs, excluding numerical output artifacts."""
    source = manifest["source"]
    return {"identity_version": "logical-suite-v3", "schema_version": 3,
            "config": manifest["config"], "city_ids": manifest["city_ids"],
            "source": {name: source[name] for name in
                       ("repo", "revision", "annual_normalized_sha256", "cities_semantic_sha256")},
            "scorer_sha256": manifest["scorer_sha256"],
            "preparation_source_pins": manifest["preparation_source_pins"]}


def seal_suite_manifest(manifest):
    """Return a copy with schema-3 logical ID and exact physical-manifest seal."""
    sealed = copy.deepcopy(manifest)
    if sealed.get("schema_version") != 3:
        raise ValueError("only schema-3 suite manifests can be newly sealed")
    sealed.pop("suite_id", None)
    sealed.pop("manifest_fingerprint", None)
    sealed["suite_id"] = canonical_hash(logical_identity(sealed))
    sealed["manifest_fingerprint"] = canonical_hash(sealed)
    return sealed


def validate_suite_manifest(manifest):
    """Verify an in-memory manifest without mutating it or reading any artifact.

    Physical suite verification remains the caller's responsibility. This helper
    also validates copied suite manifests embedded in run records.
    """
    version = manifest.get("schema_version")
    if version not in (1, 2, 3):
        raise ValueError("unknown suite manifest version")
    if manifest.get("config", {}).get("schema_version") != version:
        raise ValueError("suite manifest and config schema versions disagree")
    claimed = manifest.get("suite_id")
    try:
        if version < 3:
            payload = {key: value for key, value in manifest.items() if key != "suite_id"}
            actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        else:
            cities = manifest["city_ids"]
            if (not isinstance(cities, list) or not cities
                    or any(not isinstance(city, str) or not city.strip() for city in cities)
                    or cities != sorted(set(cities))):
                raise ValueError("suite cities must be sorted unique nonempty strings")
            if manifest["source"]["cities_semantic_sha256"] != canonical_hash(cities):
                raise ValueError("suite semantic city fingerprint does not match")
            actual = canonical_hash(logical_identity(manifest))
            physical = {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
            if manifest.get("manifest_fingerprint") != canonical_hash(physical):
                raise ValueError("suite physical manifest fingerprint does not match")
    except (KeyError, TypeError) as exc:
        raise ValueError("incomplete suite identity metadata") from exc
    if claimed != actual:
        raise ValueError("suite manifest fingerprint does not match logical suite identity")
    return manifest
