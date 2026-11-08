"""End-to-end committed asset freeze workflow using synthetic observations."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from textwrap import dedent

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tests"))
from test_evaluation import annual_fixture


def exercise_asset_cli(fixture):
    fixture.mkdir(parents=True)
    for folder in ("code", "src"):
        for p in (PROJECT / folder).rglob("*.py"):
            out = fixture / p.relative_to(PROJECT)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(p.read_bytes())
    for name in ("uv.lock", "pyproject.toml", "requirements.txt", "code/requirements.txt"):
        p = fixture / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((PROJECT / name).read_bytes())
    (fixture / "evaluation").mkdir(exist_ok=True)
    (fixture / "data").mkdir()
    (fixture / "models").mkdir()
    config = json.loads((PROJECT / "evaluation/config.json").read_text())
    config.update(origins=[2004, 2011], development_origins=[2004], assessment_origins=[2011])
    (fixture / "evaluation/config.json").write_text(json.dumps(config), encoding="utf-8")
    annual_fixture().to_csv(fixture / "data/annual.csv", index=False)
    (fixture / "data/city_info.csv").write_text("city_id\nA\nB\n", encoding="utf-8")
    (fixture / "models/settings.json").write_text('{"level":55}', encoding="utf-8")
    (fixture / "models/learned.bin").write_bytes(b"synthetic weights, not a fitted model")
    manifest = {"schema_version": 1, "files": [
        {"path": "models/settings.json", "kind": "configuration"},
        {"path": "models/learned.bin", "kind": "model_weights", "information_end_year": 1980}
    ]}
    manifest_path = fixture / "evaluation/model-assets.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (fixture / "code/manifest_model.py").write_text(dedent("""
    import json
    import os
    from pathlib import Path

    def forecast(train, targets, origin, seed):
        asset_root = Path(os.environ["PROTOSTAR_ASSET_ROOT"])
        assert (asset_root / "models/learned.bin").read_bytes().startswith(b"synthetic")
        level = json.loads((asset_root / "models/settings.json").read_text())["level"]
        return targets.assign(predicted_value=float(level))
    """), encoding="utf-8")
    (fixture / ".gitignore").write_text("outputs/\n__pycache__/\n", encoding="utf-8")
    (fixture / "README.md").write_text("Synthetic asset integration fixture only.\n", encoding="utf-8")
    (fixture / "empty-hooks").mkdir()

    def shell(argv, expected=0):
        p = subprocess.run(argv, cwd=fixture, capture_output=True, text=True, timeout=120)
        assert p.returncode == expected, (argv, p.returncode, p.stdout, p.stderr)
        return p

    shell(["git", "init", "-q"])
    shell(["git", "add", "code", "src", "evaluation", "data", "models", "uv.lock",
           "pyproject.toml", "requirements.txt", ".gitignore", "README.md"])
    shell(["git", "-c", "user.name=Evaluation Test", "-c", "user.email=evaluation@example.invalid",
           "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={fixture / 'empty-hooks'}",
           "commit", "-qm", "Synthetic asset model"])
    cli = [sys.executable, str(fixture / "code/evaluate.py")]
    shell(cli + ["prepare", "--annual", str(fixture / "data/annual.csv")])
    entry = ["--adapter", "code/manifest_model.py:forecast", "--assets", str(manifest_path)]
    shell(cli + ["run", "--name", "asset-development", *entry])
    development = next((fixture / "outputs/evaluation/runs").iterdir())
    freeze = fixture / "outputs/asset-freeze.json"
    shell(cli + ["freeze", "--output", str(freeze), "--from-run", str(development)])
    # A documentation-only descendant commit must not invalidate the model freeze.
    (fixture / "README.md").write_text("Updated documentation only.\n", encoding="utf-8")
    shell(["git", "add", "README.md"])
    shell(["git", "-c", "user.name=Evaluation Test", "-c", "user.email=evaluation@example.invalid",
           "-c", "commit.gpgsign=false", "commit", "-qm", "Docs only"])
    shell(cli + ["run", "--name", "asset-assessment", "--phase", "assessment", "--freeze", str(freeze)])
    ledger = fixture / "outputs/evaluation/suite-v3-r5/assessment_access"
    accesses = len(list(ledger.glob("*.json")))
    assert accesses == 1
    baseline_freeze = fixture / "outputs/baseline-freeze.json"
    shell(cli + ["freeze", "--baseline", "average", "--output", str(baseline_freeze)])
    shell(cli + ["baselines", "--phase", "assessment", "--freeze", str(baseline_freeze)])
    accesses += 1
    assert len(list(ledger.glob("*.json"))) == accesses
    (fixture / "models/settings.json").write_text('{"level":900}', encoding="utf-8")
    shell(cli + ["run", "--name", "changed-asset", "--phase", "assessment", "--freeze", str(freeze), *entry], expected=1)
    assert len(list(ledger.glob("*.json"))) == accesses
    (fixture / "models/settings.json").write_text('{"level":55}', encoding="utf-8")
    manifest["files"][1]["information_end_year"] = 2018
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    shell(cli + ["run", "--name", "future-weights", *entry], expected=1)
    result = {"synthetic_only": True, "fixture": str(fixture), "development": "passed",
              "committed_asset_freeze_assessment": "passed", "changed_asset_rejected": True,
              "future_trained_weights_rejected": True}
    (fixture / "evidence.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


class AssetWorkflowTests(unittest.TestCase):
    def test_custom_adapter_assets_are_bound_through_committed_cli_assessment(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = exercise_asset_cli(Path(temporary) / "project")
            self.assertEqual(result["committed_asset_freeze_assessment"], "passed")
            self.assertTrue(result["changed_asset_rejected"])
            self.assertTrue(result["future_trained_weights_rejected"])


if __name__ == "__main__":
    unittest.main()
