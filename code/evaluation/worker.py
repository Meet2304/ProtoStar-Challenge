"""Run a trusted team adapter in its own process, with logs and timeout upstream.

This is reproducibility isolation, not an operating-system security sandbox.
"""
import argparse
import importlib
import importlib.util
import json
import inspect
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import assert_no_post_cutoff
from evaluation.assets import verify_asset_binding
from evaluation.runtime_audit import RuntimeAudit
from evaluation.audit_transport import child_sender


def main():
    parser = argparse.ArgumentParser()
    for name in ("adapter", "train", "targets", "output", "params"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--origin", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--asset-root")
    parser.add_argument("--asset-binding")
    parser.add_argument("--runtime-context", required=True)
    parser.add_argument("--audit-fd", type=int)
    parser.add_argument("--audit-handle", type=int)
    args = parser.parse_args()
    context = json.loads(Path(args.runtime_context).read_text(encoding="utf-8"))
    supplied = [args.train, args.targets, args.params, args.runtime_context]
    if args.asset_binding:
        supplied.append(args.asset_binding)
    sender = child_sender(args.audit_fd, args.audit_handle)
    audit = RuntimeAudit(context, input_files=supplied, sender=sender)
    try:
        audit.install()
        binding = json.loads(Path(args.asset_binding).read_text(encoding="utf-8")) if args.asset_binding else None
        if binding is not None and args.asset_root:
            audit.assets.update((Path(args.asset_root) / name).resolve()
                                for name in [binding["manifest"]["path"], *binding["files"]])
        _run(args, binding)
    finally:
        # Run before returning to the parent, including import/callback failures.
        # A swallowed PermissionError still fails this final audit before scoring.
        try:
            audit.finalize()
        finally:
            if sender is not None:
                sender.close()


def _run(args, binding):
    # Set context before importing any team code, including import-time loaders.
    existing = os.environ.get("PROTOSTAR_FOLD_CUTOFF")
    if existing is not None and int(existing) != args.origin:
        raise ValueError("worker origin disagrees with the parent cutoff")
    os.environ["PROTOSTAR_FOLD_CUTOFF"] = str(args.origin)
    for key in ("PROTOSTAR_ASSET_ROOT", "PROTOSTAR_ASSET_MANIFEST", "PROTOSTAR_ASSET_BINDING"):
        os.environ.pop(key, None)
    if bool(args.asset_root) != bool(args.asset_binding):
        raise ValueError("asset root and binding must be provided together")
    if args.asset_binding:
        if binding["origins"] != [args.origin]:
            raise ValueError("worker assets must be projected to its forecast origin only")
        verify_asset_binding(args.asset_root, binding)
        os.environ["PROTOSTAR_ASSET_ROOT"] = args.asset_root
        os.environ["PROTOSTAR_ASSET_MANIFEST"] = str(Path(args.asset_root) / binding["manifest"]["path"])
        os.environ["PROTOSTAR_ASSET_BINDING"] = args.asset_binding
    random.seed(args.seed)
    np.random.seed(args.seed)
    module_name, function = args.adapter.rsplit(":", 1)
    if Path(module_name).is_file():
        path = Path(module_name).resolve()
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("team_adapter", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    train = pd.read_csv(args.train, dtype={"city_id": str})
    targets = pd.read_csv(args.targets, dtype={"city_id": str, "metric": str})
    assert_no_post_cutoff(train, "year", "worker train")
    if train.year.max() > args.origin:
        raise ValueError("training rows exceed fold origin")
    params = json.loads(Path(args.params).read_text(encoding="utf-8"))
    callback = getattr(module, function)
    try:
        inspect.signature(callback).bind(train, targets, args.origin, args.seed, **params)
    except TypeError as exc:
        raise ValueError(f"adapter parameters do not match {args.adapter}: {exc}") from exc
    result = callback(train, targets, args.origin, args.seed, **params)
    if args.asset_binding:
        verify_asset_binding(args.asset_root, binding)
    if not isinstance(result, pd.DataFrame):
        raise TypeError("adapter must return a pandas DataFrame")
    for column in ("year", "predicted_value"):
        if column in result and result[column].map(lambda v: isinstance(v, (bool, np.bool_))).any():
            raise ValueError(f"booleans are not valid {column} values")
    result.to_csv(args.output, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
