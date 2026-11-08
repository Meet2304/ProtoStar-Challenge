"""Parent-owned worker audit receipts over an inherited local pipe.

Ordinary output-file forgery cannot erase events already delivered to the
parent. This is not cryptographic attestation of hostile code in the same
interpreter: deliberate inspection or mutation of interpreter state remains
outside the trusted-team boundary.
"""
from __future__ import annotations

from collections import Counter
import json
import os
import subprocess
import threading

REPORT_NAMES = frozenset({"io_audit.json", "imported_source.json"})
MAX_LINE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024


class AuditSender:
    def __init__(self, fd):
        self.fd = fd
        self.lock = threading.Lock()
        self.closed = False

    def __call__(self, message):
        raw = (json.dumps(message, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(raw) > MAX_LINE_BYTES:
            raise RuntimeError("worker audit message exceeds transport limit")
        with self.lock:
            if self.closed:
                raise RuntimeError("worker audit pipe is closed")
            pending = memoryview(raw)
            while pending:
                sent = os.write(self.fd, pending)
                if sent <= 0:
                    raise RuntimeError("worker audit pipe did not accept a message")
                pending = pending[sent:]

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                os.close(self.fd)


def child_sender(fd=None, handle=None):
    """Open the inherited writer before installing model audit hooks."""
    if fd is not None and handle is not None:
        raise ValueError("choose one worker audit pipe representation")
    if handle is not None:
        if os.name != "nt":
            raise ValueError("Windows audit handle on a non-Windows worker")
        import msvcrt
        fd = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
    return AuditSender(fd) if fd is not None else None


def summarize_events(events):
    def paths(reason, operation="read"):
        return [{"path": event["path"], "count": event["count"]}
                for event in events if event["reason"] == reason and event["operation"] == operation]
    external = paths("external_path_outside_project_boundary")
    environmental = paths("environment_nonpackage_path")
    subprocesses = [event for event in events if event["reason"] == "subprocess_disabled"]
    return {"external_reads": external, "external_read_count": sum(item["count"] for item in external),
            "external_writes": paths("external_path_outside_project_boundary", "write"),
            "environment_nonpackage_reads": environmental,
            "environment_nonpackage_read_count": sum(item["count"] for item in environmental),
            "subprocess_events": subprocesses,
            "subprocess_event_count": sum(item["count"] for item in subprocesses),
            "denied_count": sum(event["count"] for event in events if event["decision"] == "deny")}


class AuditPipe:
    """One process's receiver; start draining before launch to avoid backpressure.

    Use `child_args`, `popen_kwargs`, then `after_spawn()` immediately after
    Popen. Always wait/kill the process before `finish(timeout=5)`. `finish`
    returns a nonthrowing failure receipt for truncated/missing/invalid input.
    The context manager closes parent descriptors even after launch failures.
    """
    def __init__(self):
        self.read_fd, self.write_fd = os.pipe()
        self.events = Counter()
        self.reports = {}
        self.errors = []
        self.result = None
        self.total_bytes = 0
        self.reader_done = threading.Event()
        self.state_lock = threading.RLock()
        self.popen_kwargs = {"close_fds": True}
        if os.name == "nt":
            import msvcrt
            handle = msvcrt.get_osfhandle(self.write_fd)
            os.set_handle_inheritable(handle, True)
            startup = subprocess.STARTUPINFO()
            startup.lpAttributeList = {"handle_list": [handle]}
            self.child_args = ["--audit-handle", str(handle)]
            self.popen_kwargs["startupinfo"] = startup
        else:
            os.set_inheritable(self.write_fd, True)
            self.child_args = ["--audit-fd", str(self.write_fd)]
            self.popen_kwargs["pass_fds"] = (self.write_fd,)
        self.thread = threading.Thread(target=self._read, name="protostar-audit-receiver", daemon=True)
        self.thread.start()

    def __enter__(self):
        return self

    def _error(self, message):
        # Never include arbitrary child payloads or exception messages.
        with self.state_lock:
            if message not in self.errors:
                self.errors.append(message)

    def _accept(self, message):
        if not isinstance(message, dict):
            self._error("invalid audit message")
            return
        if message.get("type") == "event" and set(message) == {"type", "event"}:
            event = message["event"]
            fields = ("path", "operation", "decision", "reason")
            if (not isinstance(event, dict) or set(event) != set(fields)
                    or any(not isinstance(event.get(key), str) for key in fields)
                    or len(event["path"]) > 8192
                    or any(len(event[key]) > 128 for key in fields[1:])
                    or event["decision"] not in {"allow", "deny"}):
                self._error("invalid audit event")
                return
            self.events[tuple(event[key] for key in fields)] += 1
        elif message.get("type") == "report" and set(message) == {"type", "name", "report"}:
            name, report = message["name"], message["report"]
            if name not in REPORT_NAMES or not isinstance(report, dict):
                self._error("invalid final audit report")
            elif (report.get("schema_version") != 1 or report.get("status") not in {"verified", "failed"}
                    or (name == "io_audit.json" and (not isinstance(report.get("events"), list)
                        or type(report.get("denied_count")) is not int or report["denied_count"] < 0))
                    or (name == "imported_source.json" and (not isinstance(report.get("sources"), list)
                        or not isinstance(report.get("violations"), list)))):
                self._error("invalid final audit report schema")
            elif name in self.reports:
                self._error("duplicate final audit report")
            else:
                self.reports[name] = report
        else:
            self._error("unknown audit message")

    def _read(self):
        try:
            with os.fdopen(self.read_fd, "rb", closefd=True) as stream:
                self.read_fd = None
                while True:
                    line = stream.readline(MAX_LINE_BYTES + 1)
                    if not line:
                        break
                    self.total_bytes += len(line)
                    if len(line) > MAX_LINE_BYTES or self.total_bytes > MAX_TOTAL_BYTES:
                        self._error("audit stream exceeds transport limit")
                        break
                    if not line.endswith(b"\n"):
                        self._error("truncated audit message")
                        break
                    try:
                        with self.state_lock:
                            self._accept(json.loads(line))
                    except (ValueError, TypeError, KeyError, UnicodeError):
                        self._error("malformed audit message")
        except (OSError, ValueError):
            self._error("audit pipe read failed")
        finally:
            self.reader_done.set()

    def after_spawn(self):
        if self.write_fd is not None:
            os.close(self.write_fd)
            self.write_fd = None

    def finish(self, timeout=5):
        if self.result is not None:
            return self.result
        self.after_spawn()
        if not self.reader_done.wait(timeout):
            self._error("audit pipe did not close after worker termination")
        with self.state_lock:
            if set(self.reports) != REPORT_NAMES:
                self._error("missing final audit reports")
            for report in self.reports.values():
                if report.get("status") != "verified":
                    self._error("worker final audit failed")
            events = [{"path": key[0], "operation": key[1], "decision": key[2], "reason": key[3], "count": count}
                      for key, count in sorted(self.events.items())]
            summary = summarize_events(events)
            if summary["denied_count"]:
                self._error("parent received denied worker events")
            # The parent owns the canonical IO report and never accepts a worker
            # summary that could erase already received events.
            reports = dict(self.reports)
            if "io_audit.json" in reports:
                reports["io_audit.json"] = {**reports["io_audit.json"], "events": events,
                    "denied_count": summary["denied_count"], "status": "failed" if self.errors else "verified",
                    "transport": "parent_pipe_v1", **summary}
            self.result = {"transport": "parent_pipe_v1", "verified": not self.errors,
                           "reports": reports, "events": events, "summary": summary,
                           "errors": list(self.errors)}
        return self.result

    def __exit__(self, *unused):
        self.after_spawn()
        # Never block a caller's exception path. The normal finish() handles
        # EOF once the process has exited or been killed by its parent.
        if self.read_fd is not None:
            try:
                os.close(self.read_fd)
            except OSError:
                pass
            self.read_fd = None
