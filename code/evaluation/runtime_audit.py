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
from functools import wraps
from urllib.parse import urlsplit, unquote


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

    def __init__(self, context, *, input_files, asset_files=(), sender=None):
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
        self.sender = sender
        self.audit_paths = {self.output / name for name in ("io_audit.json", "imported_source.json", "runtime_receipt.json")}

    def check_network(self, operation):
        if not self.active or self.inspecting:
            return
        # Deliberately omit addresses, URLs and request arguments: they may
        # contain credentials. Workers have no authorized network inputs.
        self._record("<network>", operation, "deny", "network_disabled")
        raise RuntimeAuditError("worker network tripwire blocked network access")

    def check_path(self, raw, *, writing=False):
        """Shared decision used by Python opens and wrapped native readers."""
        if not self.active or self.inspecting:
            return
        if isinstance(raw, (list, tuple)):
            for item in raw:
                self.check_path(item, writing=writing)
            return
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return  # Existing native buffers/descriptors are outside this tripwire.
        raw = os.fsdecode(raw)
        if "://" in raw:
            uri = urlsplit(raw)
            if uri.scheme.casefold() != "file" or uri.netloc not in ("", "localhost"):
                self.check_network("native_uri")
            raw = unquote(uri.path)
            if os.name == "nt" and len(raw) > 2 and raw[0] == "/" and raw[2] == ":":
                raw = raw[1:]
        if raw.startswith(("\\\\", "//")):
            self.check_network("network_filesystem")
        try:
            lexical = Path(os.path.abspath(raw))
            path = _path(raw)
        except (OSError, ValueError):
            return
        operation = "write" if writing else "read"
        if writing and path in self.audit_paths:
            decision, reason = "deny", "parent_owned_audit_artifact"
        elif path in self.inputs or path in self.assets:
            decision, reason = ("deny", "immutable_supplied_input") if writing else ("allow", "supplied_input")
        elif path in self.pins or (path.suffix.casefold() == ".pyc" and self._module_path(path) in self.pins):
            decision, reason = ("deny", "immutable_pinned_source") if writing else ("allow", "pinned_source")
        elif _inside(path, self.output):
            decision, reason = "allow", "own_origin_output"
        elif any(_inside(path, root) for root in self.environment):
            # Package and stdlib resources are authorized by the installed
            # environment binding. Other files in e.g. .venv/cache are visible
            # as unreviewed inputs instead of silently inheriting that trust.
            lower_parts = {part.casefold() for part in path.parts}
            package = bool(lower_parts & {"site-packages", "dist-packages"})
            standard = path.suffix.casefold() in {".py", ".pyc", ".pyd", ".so", ".dll"}
            if package or standard:
                return
            decision, reason = "allow", "environment_nonpackage_path"
        elif any(_inside(candidate, root) for candidate in (path, lexical)
                 for root in [self.root, *self.protected]):
            decision, reason = "deny", "undeclared_protected_path"
        else:
            # OS resources and package temporary files outside the project are
            # observable but not claimed to be sandboxed or input-certified.
            decision, reason = "allow", "external_path_outside_project_boundary"
        self._record(path, operation, decision, reason)
        if decision == "deny":
            raise RuntimeAuditError(f"worker file-open tripwire blocked {operation}: {path}")

    def _record(self, path, operation, decision, reason):
        key = (str(path), operation, decision, reason)
        self.events[key] = self.events.get(key, 0) + 1
        if self.sender is not None:
            self.sender({"type": "event", "event": {"path": key[0], "operation": operation,
                         "decision": decision, "reason": reason}})

    def _mutation_path(self, value, dir_fd=None):
        # Relative dir_fd addressing could otherwise be checked against cwd
        # while mutating an entirely different directory. Refuse this less
        # common form rather than pretending a portable fd resolver exists.
        if dir_fd is not None and dir_fd != -1 and not Path(os.fsdecode(value)).is_absolute():
            self._record("<dir_fd>", "write", "deny", "relative_dir_fd_not_supported")
            raise RuntimeAuditError("worker mutation tripwire blocked relative dir_fd operation")
        self.check_path(value, writing=True)

    def _hook(self, event, args):
        if not self.active or self.inspecting:
            return
        if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
                     "socket.sendto", "urllib.Request"}:
            self.check_network(event)
        if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn"}:
            self._record("<subprocess>", event, "deny", "subprocess_disabled")
            raise RuntimeAuditError("worker subprocess tripwire blocked process launch")
        if event in {"shutil.copyfile", "shutil.copytree"}:
            self.check_path(args[0])
            self.check_path(args[1], writing=True)
            return
        if event == "shutil.rmtree":
            self._mutation_path(args[0], args[1] if len(args) > 1 else None)
            return
        if event == "sqlite3.connect":
            target = args[0]
            if isinstance(target, os.PathLike):
                target = os.fspath(target)
            if target != ":memory:":
                if isinstance(target, str) and target.startswith("file:"):
                    target = unquote(urlsplit(target).path)
                # Connections can create/modify databases; treating them as
                # writes preserves immutable supplied assets too.
                self.check_path(target)
                self.check_path(target, writing=True)
            return
        if event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.utime", "os.truncate"}:
            value = args[0]
            if isinstance(value, int):
                self._record("<descriptor>", "write", "deny", "descriptor_mutation_not_supported")
                raise RuntimeAuditError("worker mutation tripwire blocked descriptor mutation")
            fd_index = {"os.remove": 1, "os.rmdir": 1, "os.mkdir": 2, "os.chmod": 2, "os.utime": 3}.get(event)
            self._mutation_path(value, args[fd_index] if fd_index is not None and len(args) > fd_index else None)
            return
        if event in {"os.rename", "os.link", "os.symlink"}:
            if event == "os.symlink":
                # Resolving a relative symlink against cwd instead of its
                # destination would misstate the target. Models do not need
                # symlinks to produce forecast artifacts.
                self._record("<symlink>", "write", "deny", "symlink_creation_disabled")
                raise RuntimeAuditError("worker mutation tripwire blocked symlink creation")
            self._mutation_path(args[0], args[2] if len(args) > 2 else None)
            self._mutation_path(args[1], args[3] if len(args) > 3 else None)
            return
        if event != "open":
            return
        raw = args[0]
        if isinstance(raw, int) or not isinstance(raw, (str, bytes, os.PathLike)):
            return  # Existing file descriptors are outside this path tripwire.
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(char in mode for char in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
        self.check_path(raw, writing=writing)

    def install(self):
        # Import C-extension dependencies before replacing their standard API
        # entry points, but before any model code can capture those entry points.
        from evaluation.native_io import install_native_guards
        install_native_guards(self)
        # Windows copy2 can call CopyFile2 without CPython's copyfile/open
        # events. Guard these ordinary stdlib entry points as well as events.
        import shutil
        for name in ("copy", "copy2", "copyfile", "copytree", "copystat", "copymode", "move"):
            original = getattr(shutil, name)
            def guarded(source, destination, *args, _original=original, _name=name, **kwargs):
                self.check_path(source, writing=_name == "move")
                self.check_path(destination, writing=True)
                return _original(source, destination, *args, **kwargs)
            setattr(shutil, name, wraps(original)(guarded))
        sys.dont_write_bytecode = True
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
        # Stop recording the evaluator's own final artifact writes.
        self.active = False
        self.output.mkdir(parents=True, exist_ok=True)
        events = [{"path": key[0], "operation": key[1], "decision": key[2], "reason": key[3], "count": count}
                  for key, count in sorted(self.events.items())]
        io_report = {"schema_version": 1, "status": "failed" if any(e["decision"] == "deny" for e in events) else "verified",
                     "events": events, "denied_count": sum(e["count"] for e in events if e["decision"] == "deny"),
                     "limits": "Python opens, standard Arrow path APIs and Python socket/urllib events are checked. "
                               "External OS paths, existing descriptors, unwrapped/native internals, subprocesses and "
                               "interpreter manipulation remain outside this trusted-team tripwire; not an OS sandbox."}
        for name, value in (("io_audit.json", io_report), ("imported_source.json", report)):
            if self.sender is not None:
                self.sender({"type": "report", "name": name, "report": value})
            else:
                # Standalone synthetic worker tests retain file receipts. The
                # production parent always requires the inherited pipe.
                (self.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if report["violations"]:
            paths = ", ".join(item["path"] for item in report["violations"])
            raise RuntimeAuditError(f"imported source is undeclared, changed or unreadable: {paths}")
        if io_report["denied_count"]:
            raise RuntimeAuditError("adapter attempted a denied file open; see io_audit.json")
