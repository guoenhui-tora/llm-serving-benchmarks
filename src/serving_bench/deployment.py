"""Synchronized, equal-share local replicas using the existing client and gates."""
from __future__ import annotations

import copy
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import BenchError, read_json, write_json
from .executors import affinity, docker
from .clients.jsonl_dataset import validated_manifest


def preflight(case):
    if "replicas" not in case:
        return docker.preflight(case)
    affinity.host_check(case["target"])
    facts = [docker.preflight(member) for member in case["replicas"]]
    # Catch cross-replica SMT collisions, not just overlapping logical IDs.
    occupied = set()
    for item in facts:
        for cores in item.get("binding", {}).get("physical_cores", {}).values():
            cores = {tuple(core) for core in cores}
            if occupied & cores:
                raise BenchError("Replica bindings share physical cores through SMT")
            occupied.update(cores)
    for item in facts[1:]:
        for key in ("server_image", "client_image", "model"):
            identity = "identity" if key == "model" else "Id"
            if item[key][identity] != facts[0][key][identity]:
                raise BenchError("Replica image/model identities differ")
    combined = copy.deepcopy(facts[0])
    combined["replicas"] = facts
    combined["gpus"] = [gpu for item in facts for gpu in item["gpus"]]
    return combined


def percentile(values, p):
    values = sorted(values)
    position = (len(values)-1)*p/100
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper]-values[lower])*(position-lower)


def aggregate(directories: list[Path], expected: int, workload: dict) -> tuple[dict, dict]:
    windows = [read_json(p / "benchmark-window.json") for p in directories]
    normalized = [read_json(p / "metrics.json") for p in directories]
    parts = [read_json(p / "requests.json") for p in directories]
    rows = [row for part in parts for row in part]
    starts = [w["start_s"] for w in windows]
    ends = [w["end_s"] for w in windows]
    if not all(type(v) in (float, int) and math.isfinite(v) for v in starts + ends):
        raise BenchError("Invalid synchronized benchmark clocks")
    if max(starts)-min(starts) > 0.5:
        raise BenchError("Client benchmark start skew exceeds 0.5 seconds")
    for window, result, part in zip(windows, normalized, parts):
        if window["end_s"] <= window["start_s"] or abs(window["end_s"]-window["start_s"]-result["metrics"]["duration"]) > 0.01:
            raise BenchError("Client window does not match the official benchmark duration")
        if len(part) != expected//len(parts) or result["completed"] != len(part):
            raise BenchError("Per-replica detailed request count mismatch")
    if len(rows) != expected or any(not r["success"] for r in rows):
        raise BenchError("Synchronized request count/failure mismatch")
    d = workload["dataset"]
    is_jsonl = d.get("name") == "jsonl"
    if is_jsonl:
        manifests = []
        try:
            for i, (directory, part) in enumerate(zip(directories, parts)):
                evidence = validated_manifest(directory / "dataset-manifest.json", d, len(part), (i, len(parts), expected))
                manifests.extend(evidence["requests"])
                if [r["input_tokens"] for r in part] != [r["input_tokens"] for r in evidence["requests"]]:
                    raise ValueError("Detailed JSONL input token count mismatch")
            if len({r["id"] for r in manifests}) != expected:
                raise ValueError("Duplicate JSONL request IDs across replicas")
        except (OSError, ValueError) as exc:
            raise BenchError(f"Invalid JSONL replica evidence: {exc}") from exc
    for row in rows:
        if not is_jsonl and d["range_ratio"] == 0 and row["input_tokens"] != d["input_tokens"]:
            raise BenchError("Detailed input token count mismatch")
        if workload["sampling"]["ignore_eos"] and (is_jsonl or d["range_ratio"] == 0) and row["output_tokens"] != d["output_tokens"]:
            raise BenchError("Detailed output token count mismatch")
        times = [row["ttft_s"], row["latency_s"], *row["itl_s"]]
        if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in times) or row["latency_s"] < row["ttft_s"]:
            raise BenchError("Invalid per-request timing")
    duration = max(ends)-min(starts)
    output = sum(r["output_tokens"] for r in rows)
    inputs = sum(r["input_tokens"] for r in rows)
    if output != sum(r["metrics"]["total_output_tokens"] for r in normalized) or inputs != sum(r["metrics"]["total_input_tokens"] for r in normalized):
        raise BenchError("Detailed tokens differ from official client metrics")
    metrics = {"output_throughput": output/duration, "request_throughput": expected/duration,
               "duration": duration, "total_input_tokens": inputs, "total_output_tokens": output}
    values = {"ttft": [r["ttft_s"] for r in rows], "e2el": [r["latency_s"] for r in rows],
              "tpot": [(r["latency_s"]-r["ttft_s"])/(r["output_tokens"]-1) for r in rows if r["output_tokens"] > 1],
              "itl": [t for r in rows for t in r["itl_s"]]}
    for name, data in values.items():
        metrics[f"mean_{name}_ms"] = statistics.fmean(data)*1000 if data else None
        metrics[f"p95_{name}_ms"] = percentile(data, 95)*1000 if data else None
    metrics["median_ttft_ms"] = percentile(values["ttft"], 50)*1000
    metrics["p99_ttft_ms"] = percentile(values["ttft"], 99)*1000
    window = {"start_s": min(starts), "end_s": max(ends), "duration_s": duration,
              "start_skew_s": max(starts)-min(starts),
              "overlap_s": max(0, min(ends)-max(starts)),
              "replicas": [{"index": i, **w, "completed": normalized[i]["completed"],
                            "output_tokens": sum(r["output_tokens"] for r in parts[i]),
                            "output_throughput_own_window": normalized[i]["metrics"]["output_throughput"],
                            "output_throughput_common_window": sum(r["output_tokens"] for r in parts[i])/duration}
                           for i, w in enumerate(windows)],
              "interpretation": "Union of synchronized official client benchmark windows; latency percentiles pooled over requests/events."}
    return {"schema_version": 1, "completed": expected, "failed": 0, "failure_count_inferred": False,
            "metrics": metrics}, window


def phase(case, workload, concurrency, count, directory, sessions, owner, facts):
    from .runner import check_stop, phase as single_phase
    check_stop(directory)
    directory.mkdir(parents=True, exist_ok=False)
    coordination = directory / "coordination"
    coordination.mkdir()
    n = len(sessions)
    futures = []
    timeout = workload["measurement"]["timeout_s"]
    pool = ThreadPoolExecutor(max_workers=n)
    try:
        for index, (member, server, _) in enumerate(sessions):
            client_case = copy.deepcopy(member)
            client_case["_client_sync"] = {"directory": str(coordination), "index": index, "replicas": n,
                                           "global_requests": count, "timeout_s": min(timeout, 300)}
            futures.append(pool.submit(single_phase, client_case, workload, concurrency//n, count//n,
                                       directory/f"replica-{index}", server, f"{owner}-r{index}", facts["replicas"][index]))
        deadline = time.monotonic() + min(timeout, 300)
        while not all((coordination/f"ready-{i}.json").exists() for i in range(n)):
            check_stop(directory)
            for future in futures:
                if future.done():
                    future.result()
                    raise BenchError("Client exited before synchronized start")
            if time.monotonic() > deadline:
                raise BenchError("Clients did not reach benchmark barrier within timeout")
            time.sleep(0.05)
        # Atomic publication; both containers share the host monotonic clock.
        temporary = coordination / "release.tmp"
        write_json(temporary, {"start_ns": time.monotonic_ns()+100_000_000})
        temporary.replace(coordination / "release.json")
        while not all(f.done() for f in futures):
            check_stop(directory)
            for future in futures:
                if future.done():
                    future.result()
            time.sleep(0.05)
        results = [f.result() for f in futures]
        metrics, window = aggregate([directory/f"replica-{i}" for i in range(n)], count, workload)
        events = [{"replica": i, "event": event} for i, (_, entries) in enumerate(results) for event in entries]
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "benchmark-window.json", window)
        write_json(directory / "compilation.json", {"events": events, "interpretation": "All replica log gates must be quiet."})
        return metrics, events
    except BaseException:
        write_json(coordination / "abort.json", {"aborted": True})
        for i in range(n):
            try:
                docker.remove_owned(f"sb-{owner}-r{i}-client", f"{owner}-r{i}")
            except Exception:
                pass  # Lifecycle cleanup retries and reports errors.
        raise
    finally:
        pool.shutdown(wait=True)
