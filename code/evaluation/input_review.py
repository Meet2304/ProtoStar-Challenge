"""Visible, conservative external-input review for a trusted-team worker."""
from pathlib import Path
import re
CPU_PROBES = {
    "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
    "/sys/fs/cgroup/cpu/cpu.cfs_period_us", "/sys/fs/cgroup/cpu.cfs_quota_us",
    "/sys/fs/cgroup/cpu.cfs_period_us",
}


def review_input_summary(summary):
    result = dict(summary)
    result["subprocess_count"] = summary.get("subprocess_event_count", summary.get("subprocess_count", 0))
    external = summary.get("external_reads", [])
    def cpu_probe(item):
        path = item.get("path", "")
        normalized = path.replace("\\", "/")
        if normalized in CPU_PROBES:
            return True
        # joblib probes Linux cgroups on Windows too. Path.resolve records
        # those failed opens with the temporary work directory's drive. Never
        # approve an actual Windows data file at that spelling.
        if re.match(r"^[a-zA-Z]:/", normalized):
            return normalized[2:] in CPU_PROBES and not Path(path).exists()
        return False
    unexpected = [item for item in external if not cpu_probe(item)]
    environment = summary.get("environment_nonpackage_reads", [])
    result["non_allowlisted_external_reads"] = unexpected
    result["requires_review"] = bool(unexpected or environment or result["subprocess_count"])
    result["interpretation"] = ("CPU-limit probes are allowed. Other external reads and non-package environment files "
                                "require declared inputs before assessment; these counts are supported-API observations, not OS attestation.")
    return result
