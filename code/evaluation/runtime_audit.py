"""Practical Python IO/import tripwires for trusted model workers.

This is not an OS sandbox. Deliberate native-code, subprocess, module removal,
or interpreter manipulation can bypass it. Logs contain paths and hashes only,
never file contents, arguments, environment variables, or exception messages.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


class RuntimeAuditError(RuntimeError):
    """An adapter crossed the worker's declared runtime boundary."""


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _path(value):
    return Path(os.fsdecode(value)).resolve()


def _inside(path, root):
    return path == root or path.is_relative_to(root)


class RuntimeAudit:
    """Install before adapter import; finalize even when the callback raises.

    Context v1 requires project_root, suite_root, own_output and source_pins.
    adapter_file is null or {path, sha256}. declared_sources is a list of
    {original_path, snapshot_path, sha256}; all hashes normalize CRLF to LF.
    Paths except source_pins' project-relative POSIX keys are absolute.
    """

    def __init__(self, context, *, input_files, asset_files=()):
        if context.get("schema_version") != 1:
            raise ValueError("unsupported worker runtime context")
        self.root = _path(context["project_root"])
        self.suite = _path(context["suite_root"])
        self.output = _path(context["own_output"])
        self.protected = [self.root / "data", self.suite,
                          self.root / "outputs/evaluation", self.output.parent]
        self.protected = [_path(path) for path in self.protected]
        self.inputs = {_path(path) for path in input_files}
        self.assets = {_path(path) for path in asset_files}
        self.pins = {_path(self.root / name): value for name, value in context["source_pins"].items()}
        self.source_roles = {path: "pinned_source" for path in self.pins}
        adapter = context.get("adapter_file")
        if adapter:
            path = _path(adapter["path"])
            if path in self.pins and self.pins[path] != adapter["sha256"]:
                raise ValueError("conflicting adapter/runtime source hashes")
            self.pins[path] = adapter["sha256"]
            self.source_roles[path] = "adapter_entrypoint"
        for declaration in context.get("declared_sources", []):
            for key in ("original_path", "snapshot_path"):
                path = _path(declaration[key])
                if path in self.pins and self.pins[path] != declaration["sha256"]:
                    raise ValueError("conflicting runtime source hashes")
                self.pins[path] = declaration["sha256"]
                self.source_roles[path] = "declared_source"
        self.environment = {_path(sys.prefix), _path(sys.base_prefix)}
        self.preloaded = set(sys.modules)
        self.events = {}
        self.active = False
        self.inspecting = False
        self.import_report = None

    def _record(self, path, operation, decision, reason):
        key = (str(path), operation, decision, reason)
        self.events[key] = self.events.get(key, 0) + 1

    def _hook(self, event, args):
        if not self.active or self.inspecting or event != "open":
            return
        raw = args[0]
        if isinstance(raw, int) or not isinstance(raw, (str, bytes, os.PathLike)):
            return  # Existing file descriptors are outside this path tripwire.
        try:
            lexical = Path(os.path.abspath(os.fsdecode(raw)))
            path = _path(raw)
        except (OSError, ValueError):
            return  # The underlying open will report malformed paths.
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(char in mode for char in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        operation = "write" if writing else "read"
        # Exact supplied snapshots/inputs permit reads, never writes. This check
        # precedes own-output permission because asset snapshots live beneath it.
        if path in self.inputs or path in self.assets:
            decision, reason = ("deny", "immutable_supplied_input") if writing else ("allow", "supplied_input")
        elif _inside(path, self.output) and _inside(lexical, self.output):
            decision, reason = "allow", "own_origin_output"
        elif any(_inside(candidate, root) for candidate in (path, lexical) for root in self.protected):
            decision, reason = "deny", "undeclared_protected_path"
        else:
            return  # Package resources and ordinary imports remain usable.
        self._record(path, operation, decision, reason)
        if decision == "deny":
            raise RuntimeAuditError(f"worker file-open tripwire blocked {operation}: {path}")

    def install(self):
        self.active = True
        sys.addaudithook(self._hook)

    def _module_path(self, raw):
        path = _path(raw)
        if path.suffix.casefold() == ".pyc":
            try:
                path = _path(importlib.util.source_from_cache(str(path)))
            except ValueError:
                path = path.with_suffix(".py")
        return path

    def inspect_imports(self):
        """Observe all loaded source, including preloaded evaluator modules.

        Environment modules are summarized; every other concrete module must
        match a pin. Namespace packages contain no executed source; their loaded
        children receive the same checks. No discovered source is added to pins.
        """
        entries, violations, environment_count = [], [], 0
        self.inspecting = True  # Hashing source is evaluator observation, not model IO.
        try:
            for name, module in sorted(list(sys.modules.items())):
                raw = getattr(module, "__file__", None)
                if not raw or str(raw).startswith("<"):
                    continue
                try:
                    path = self._module_path(raw)
                    # A venv may live under the repository; only actual interpreter
                    # roots are exempt, never arbitrary preloaded project code.
                    if path not in self.pins and any(_inside(path, root) for root in self.environment):
                        environment_count += 1
                        continue
                    entry = {"module": name, "path": str(path), "preloaded": name in self.preloaded}
                    expected = self.pins.get(path)
                    observed = source_hash(path)
                    entry.update(sha256=observed, expected_sha256=expected,
                                 role=self.source_roles.get(path, "undeclared_source"))
                    entry["status"] = "verified" if expected == observed else ("undeclared" if expected is None else "changed")
                except (OSError, ValueError):
                    entry = {"module": name, "path": str(raw), "status": "unreadable"}
                entries.append(entry)
                if entry["status"] != "verified":
                    violations.append({"module": name, "path": entry["path"], "status": entry["status"]})
        finally:
            self.inspecting = False
        self.import_report = {"schema_version": 1, "status": "failed" if violations else "verified",
                              "environment_module_count": environment_count, "sources": entries,
                              "violations": violations,
                              "limits": "Loaded Python modules only; not a sandbox or proof against deliberate bypass."}
        return self.import_report

    def finalize(self):
        report = self.inspect_imports()
        self.output.mkdir(parents=True, exist_ok=True)
        # Stop recording the evaluator's own final artifact writes.
        self.active = False
        events = [{"path": key[0], "operation": key[1], "decision": key[2], "reason": key[3], "count": count}
                  for key, count in sorted(self.events.items())]
        io_report = {"schema_version": 1, "status": "failed" if any(e["decision"] == "deny" for e in events) else "verified",
                     "events": events, "denied_count": sum(e["count"] for e in events if e["decision"] == "deny"),
                     "limits": "Python file-open tripwire only; native IO, existing descriptors and subprocesses may bypass it."}
        for name, value in (("io_audit.json", io_report), ("imported_source.json", report)):
            (self.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if report["violations"]:
            paths = ", ".join(item["path"] for item in report["violations"])
            raise RuntimeAuditError(f"imported source is undeclared, changed or unreadable: {paths}")
        if io_report["denied_count"]:
            raise RuntimeAuditError("adapter attempted a denied file open; see io_audit.json")
