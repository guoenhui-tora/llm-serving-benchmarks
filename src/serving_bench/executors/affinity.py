from __future__ import annotations

import inspect
import re
from pathlib import Path

from ..common import BenchError


def id_set(value: str) -> set[int]:
    """Parse the Linux CPU/node list syntax used by Docker and /proc."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", value):
        raise ValueError("Expected a quoted CPU/node list, e.g. '0-3,8'")
    result = set()
    for part in value.split(","):
        ends = part.split("-")
        start, end = int(ends[0]), int(ends[-1])
        if end < start or end > 1048575:
            raise ValueError("Invalid CPU/node range")
        items = set(range(start, end + 1))
        if result & items:
            raise ValueError("Duplicate CPU/node IDs")
        result.update(items)
    return result


def snapshot() -> dict:
    """Snapshot visible threads, including workers; vanished threads are normal."""
    result = {"threads": [], "unreadable": []}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            tasks = list((proc / "task").iterdir())
        except FileNotFoundError:
            continue
        except PermissionError:
            result["unreadable"].append(str(proc))
            continue
        for task in tasks:
            try:
                status = dict(line.split(":", 1) for line in (task / "status").read_text().splitlines() if ":" in line)
                result["threads"].append({"pid": int(proc.name), "tid": int(task.name),
                    "name": status["Name"].strip(), "cpus": status["Cpus_allowed_list"].strip(),
                    "mems": status["Mems_allowed_list"].strip()})
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                result["unreadable"].append(str(task))
    return result


def verify_snapshot(binding: dict, observed: dict) -> None:
    if observed["unreadable"] or not observed["threads"]:
        raise ValueError("Cannot verify container thread affinity")
    expected = {key: id_set(binding[key]) for key in ("cpus", "mems")}
    for thread in observed["threads"]:
        for key in expected:
            # Engines may narrow worker affinity within the container allocation.
            actual = id_set(thread[key])
            if not actual <= expected[key]:
                raise ValueError(f"Thread {thread['tid']} {key} escapes requested binding: {thread[key]}")


def probe_source() -> str:
    """Stdlib-only probe, sent to pinned containers without installing the runner."""
    return "import re, json\nfrom pathlib import Path\n" + "\n".join(
        inspect.getsource(function) for function in (id_set, snapshot, verify_snapshot))


def docker_args(target: dict, role: str) -> list[str]:
    binding = target.get("binding", {}).get(role)
    return ["--cpuset-cpus", binding["cpus"], "--cpuset-mems", binding["mems"]] if binding else []


def host_check(target: dict, sys_root: Path = Path("/sys/devices/system")) -> dict:
    bindings = target.get("binding", {})
    if not bindings:
        return {}
    try:
        online = id_set((sys_root / "cpu/online").read_text().strip())
        nodes = id_set((sys_root / "node/has_memory").read_text().strip())
        cores = {}
        for role, binding in bindings.items():
            cpus, mems = id_set(binding["cpus"]), id_set(binding["mems"])
            if not cpus <= online or not mems <= nodes:
                raise ValueError(f"{role}: requested offline/absent CPUs or memory nodes")
            cores[role] = set()
            for cpu in cpus:
                topology = sys_root / f"cpu/cpu{cpu}/topology"
                cores[role].add(((topology / "physical_package_id").read_text().strip(),
                                 (topology / "core_id").read_text().strip()))
        if cores.get("server", set()) & cores.get("client", set()):
            raise ValueError("Server/client bindings share physical cores (including SMT siblings)")
    except (OSError, ValueError) as exc:
        raise BenchError(f"Host affinity check failed: {exc}") from exc
    return {"requested": bindings, "online_cpus": sorted(online), "memory_nodes": sorted(nodes),
            "physical_cores": {role: sorted(values) for role, values in cores.items()}}


def verify_inspect(binding: dict, info: dict) -> None:
    try:
        for key, field in (("cpus", "CpusetCpus"), ("mems", "CpusetMems")):
            if id_set(info["HostConfig"][field]) != id_set(binding[key]):
                raise ValueError(f"Docker {field} does not match requested binding")
    except (KeyError, ValueError) as exc:
        raise BenchError(f"Container affinity check failed: {exc}") from exc


def client_prefix(prefix: list[str], binding: dict) -> list[str]:
    # The probe runs outside the client's internal benchmark timer. Evidence is
    # written before verification so even a mismatch remains reviewable.
    setup = probe_source() + f"\n_binding = {binding!r}\n"
    setup += ("def record_affinity(stage):\n"
              "    observed = snapshot()\n"
              "    Path('/results/client-affinity-' + stage + '.json').write_text(json.dumps(observed, indent=2))\n"
              "    verify_snapshot(_binding, observed)\n"
              "record_affinity('start')\n"
              "try:\n"
              f"    exec({prefix[1]!r})\n"
              "finally:\n"
              "    record_affinity('end')\n")
    return [prefix[0], setup, *prefix[2:]]
