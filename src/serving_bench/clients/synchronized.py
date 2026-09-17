"""Stdlib-only instrumentation mounted read-only in the pinned benchmark client.

The official request generator, protocol implementation and metric calculation
are unchanged. Only the two benchmark clock assignments are instrumented, and
identical global input datasets are partitioned before entering the benchmark.
Unknown clock layouts fail closed instead of silently timing the wrong window.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import inspect
import json
import time
from pathlib import Path


def instrument(source: str) -> ast.Module:
    tree = ast.parse(source)
    found = {"benchmark_start_time": 0, "benchmark_duration": 0}
    expected = {"benchmark_start_time": "time.perf_counter()",
                "benchmark_duration": "time.perf_counter() - benchmark_start_time"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name not in found:
            continue
        if ast.dump(node.value) != ast.dump(ast.parse(expected[name], mode="eval").body):
            raise RuntimeError("Unsupported official benchmark clock expression: " + name)
        found[name] += 1
        expression = "_sb_clock_start()" if name == "benchmark_start_time" else "_sb_clock_end(benchmark_start_time)"
        node.value = ast.parse(expression, mode="eval").body
    if found != {"benchmark_start_time": 1, "benchmark_duration": 1}:
        raise RuntimeError("Unsupported official benchmark clock layout")
    return ast.fix_missing_locations(tree)


def install(serve, coordination: Path, output: Path, index: int, replicas: int, timeout: float):
    source = inspect.getsource(serve.benchmark)
    tree = instrument(source)
    deadline = time.monotonic() + timeout
    window = {}

    def check_abort():
        if (coordination / "abort.json").exists():
            raise RuntimeError("Synchronized client phase aborted")
        if time.monotonic() > deadline:
            raise TimeoutError("Synchronized client start barrier timed out")

    def start():
        (coordination / f"ready-{index}.json").write_text(json.dumps({"ready_ns": time.monotonic_ns()}))
        release = coordination / "release.json"
        while not release.exists():
            check_abort()
            time.sleep(0.01)
        go = json.loads(release.read_text())["start_ns"]
        while time.monotonic_ns() < go:
            check_abort()
            time.sleep(0.001)
        check_abort()
        begin = time.perf_counter()
        window.update(start_s=begin, clock="host CLOCK_MONOTONIC / perf_counter",
                      benchmark_source_sha256=hashlib.sha256(source.encode()).hexdigest())
        return begin

    def end(begin):
        finish = time.perf_counter()
        window.update(end_s=finish, duration_s=finish-begin)
        (output / "benchmark-window.json").write_text(json.dumps(window, indent=2))
        return finish-begin

    original_globals = serve.benchmark.__globals__
    namespace = dict(original_globals)
    namespace.update(_sb_clock_start=start, _sb_clock_end=end)
    exec(compile(tree, "<synchronized-official-benchmark>", "exec"), namespace)
    timed = namespace["benchmark"]
    signature = inspect.signature(timed)

    async def partitioned(*args, **kwargs):
        namespace.update(original_globals)
        namespace.update(_sb_clock_start=start, _sb_clock_end=end, calculate_metrics=recorded)
        bound = signature.bind(*args, **kwargs)
        requests = bound.arguments["input_requests"]
        if len(requests) % replicas:
            raise RuntimeError("Generated global dataset does not divide evenly")
        bound.arguments["input_requests"] = requests[index::replicas]
        (output / "request-partition.json").write_text(json.dumps({
            "global_requests": len(requests), "replica": index, "replicas": replicas,
            "indices": list(range(index, len(requests), replicas))}, indent=2))
        return await timed(*bound.args, **bound.kwargs)

    calculate = serve.calculate_metrics

    def recorded(*args, **kwargs):
        bound = inspect.signature(calculate).bind(*args, **kwargs)
        result = calculate(*args, **kwargs)
        rows = []
        for request, response, length in zip(bound.arguments["input_requests"], bound.arguments["outputs"], result[1]):
            rows.append({"success": response.success, "input_tokens": request.prompt_len,
                         "output_tokens": length, "ttft_s": response.ttft,
                         "latency_s": response.latency, "itl_s": response.itl,
                         "error": response.error})
        (output / "requests.json").write_text(json.dumps(rows))
        return result

    namespace["calculate_metrics"] = recorded
    serve.benchmark = partitioned
