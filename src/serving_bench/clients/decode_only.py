"""Local prefix-cached decode reference against an independently started vLLM server.

Derived from bench-runner's bounded decode pool and reset/prime protocol. This
tool does not start, stop, or modify the model server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler

from ..common import BenchError
from ..executors.docker import image_info, remove_owned
from . import vllm_bench
from .jsonl_dataset import read_records, validated_manifest


IMAGE_ID = "sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
PREFIX_SHA256 = "025246b18d8ba7e1bc4d98c1a20b32b570860bb06a0baf0cea4ae97403d982e7"
COUNTERS = (
    "vllm:prefix_cache_hits_total",
    "vllm:request_prefill_kv_computed_tokens_sum",
    "vllm:request_success_total",
    "vllm:num_preemptions_total",
)
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")
METRIC = re.compile(r'^([\w:]+)(?:\{([^}]*)\})?\s+([\d.eE+-]+)(?:\s|$)')
ENGINE = re.compile(r'(?:^|,)engine="([^"]+)"(?:,|$)')
HTTP = build_opener(ProxyHandler({}))


def parse_metrics(body: str, names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {name: {} for name in names}
    for line in body.splitlines():
        match = METRIC.match(line)
        if not match or match[1] not in values:
            continue
        engine = ENGINE.search(match[2] or "")
        key = engine[1] if engine else "all"
        number = float(match[3])
        if not math.isfinite(number) or number < 0:
            raise BenchError(f"Invalid metric {match[1]}: {number}")
        values[match[1]][key] = values[match[1]].get(key, 0) + number
    return values


def deltas(before: dict, after: dict) -> dict[str, float]:
    result = {}
    for name in COUNTERS:
        first, last = before.get(name), after.get(name)
        if not first or not last or first.keys() != last.keys():
            raise BenchError(f"Missing or changed metric series: {name}")
        changes = [last[k] - first[k] for k in first]
        if any(value < 0 for value in changes):
            raise BenchError(f"Counter reset during round: {name}")
        result[name] = sum(changes)
    return result


def check_round(counts: dict[str, float], n: int, input_tokens: int, block_size: int) -> None:
    if counts["vllm:request_success_total"] != n or counts["vllm:num_preemptions_total"] != 0:
        raise BenchError(f"Unexpected success/preemption count: {counts}")
    # A full shared prefix has at most one block of unavoidable local work.
    if counts["vllm:prefix_cache_hits_total"] < n * (input_tokens - block_size):
        raise BenchError(f"Shared-prefix cache hit deficit: {counts}")
    if counts["vllm:request_prefill_kv_computed_tokens_sum"] > n * block_size:
        raise BenchError(f"Excess local prefill computation: {counts}")


def check_primes(before: dict, after: dict, engines: int, input_tokens: int,
                 block_size: int) -> dict[str, float]:
    counts = deltas(before, after)
    hits_before = before["vllm:prefix_cache_hits_total"]
    hits_after = after["vllm:prefix_cache_hits_total"]
    if (hits_before.keys() != hits_after.keys() or len(hits_after) != engines or
            counts["vllm:request_success_total"] != engines * 2 or
            counts["vllm:num_preemptions_total"] != 0 or
            any(hits_after[k] - hits_before[k] < input_tokens - block_size
                for k in hits_before)):
        raise BenchError(f"Prime did not establish a reusable prefix on all {engines} engines: {counts}")
    return counts


def dataset_rows(source: Path, expected_sha256: str, n: int, output_tokens: int,
                 input_tokens: int) -> tuple[list[dict], str]:
    rows, digest = read_records(source, output_tokens, input_tokens, expected_sha256)
    first = rows[0]
    if first.get("input_tokens") != input_tokens:
        raise BenchError("The source's first prompt must declare the exact input length")
    return [dict(id=f"shared-{i:04d}", prompt=first["prompt"],
                 input_tokens=input_tokens, output_tokens=output_tokens) for i in range(n)], digest


def save_rows(path: Path, rows: list[dict]) -> str:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def get_json(port: int, path: str, *, post: bool = False) -> dict:
    request = Request(f"http://127.0.0.1:{port}{path}", method="POST" if post else "GET")
    with HTTP.open(request, timeout=30) as response:
        return json.load(response)


def profile_control(port: int, action: str, directory: Path) -> None:
    request = Request(f"http://127.0.0.1:{port}/{action}_profile", method="POST")
    with HTTP.open(request, timeout=120) as response:
        if response.status != 200:
            raise BenchError(f"{action}_profile returned HTTP {response.status}")
    (directory / f"profile-{action}").write_text(str(time.time()))


def snapshot(port: int) -> dict:
    with HTTP.open(f"http://127.0.0.1:{port}/metrics", timeout=30) as response:
        return parse_metrics(response.read().decode(), COUNTERS + GAUGES)


def assert_idle(metrics: dict) -> None:
    for name in GAUGES:
        if not metrics[name] or any(value != 0 for value in metrics[name].values()):
            raise BenchError(f"Cannot reset a busy or unobservable server: {name}={metrics[name]}")


def make_workload(path: Path, digest: str, output_tokens: int, n: int, args) -> dict:
    return {"id": f"shared-{n}-{output_tokens}",
            "dataset": {"name": "jsonl", "path": str(path), "sha256": digest,
                        "max_input_tokens": args.input_tokens, "output_tokens": output_tokens},
            "traffic": {"request_rate": "inf"},
            "sampling": {"seed": 0, "temperature": 0, "ignore_eos": True}}


def client_command(args, workload: dict, count: int, directory: Path, owner: str, name: str,
                   concurrency: int | None = None) -> list[str]:
    binding = ({"client": {"cpus": args.client_cpus, "mems": args.client_mems}}
               if args.client_cpus else {})
    case = {"target": {"port": args.port, "binding": binding},
            "model": {"served_name": args.served_name}, "model_path": str(args.model_dir),
            "dataset_paths": {workload["id"]: str(workload["dataset"]["path"])},
            "client": {"image": args.client_image, "trust_remote_code": True,
                       "environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}}}
    return vllm_bench.command(case, workload, concurrency or args.concurrency, count, directory,
                              name, owner, args.client_image_id)


def run_client(argv: list[str], directory: Path, name: str, owner: str, timeout: int) -> None:
    (directory / "command.json").write_text(json.dumps(argv, indent=2))
    (directory / "client-start-ns").write_text(str(time.time_ns()))
    try:
        with (directory / "client.log").open("w") as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    finally:
        try:
            (directory / "client-finish-ns").write_text(str(time.time_ns()))
        finally:
            remove_owned(name, owner)


def measure(args, root: Path, phase: str, n: int, rows: list[dict], owner: str,
            deadline: float) -> dict:
    directory = root / phase
    directory.mkdir()
    digest = save_rows(directory / "dataset.jsonl", rows[:n])
    workload = make_workload(directory / "dataset.jsonl", digest, args.output_tokens, n, args)
    assert_idle(snapshot(args.port))
    if get_json(args.port, "/reset_prefix_cache?reset_running_requests=true", post=True).get("success") is not True:
        raise BenchError("Prefix cache reset failed")
    before_prime = snapshot(args.port)
    # Each engine must compute the prefix once and demonstrate a cache hit once.
    prime_rows = [dict(rows[0], id="prime", output_tokens=1)]
    prime_digest = save_rows(directory / "prime.jsonl", prime_rows)
    prime_workload = make_workload(directory / "prime.jsonl", prime_digest, 1, 1, args)
    for i in range(2 * args.prime_engines):
        prime_dir = directory / f"prime-{i + 1}"
        prime_dir.mkdir()
        command = client_command(args, prime_workload, 1, prime_dir, owner,
                                 f"sb-decode-{owner}-{phase}-p{i}", concurrency=1)
        run_client(command, prime_dir, f"sb-decode-{owner}-{phase}-p{i}", owner, args.timeout_s)
        prime_result = json.loads((prime_dir / "raw.json").read_text())
        validated_manifest(prime_dir / "dataset-manifest.json", prime_workload["dataset"], 1)
        if (prime_result.get("completed") != 1 or prime_result.get("failed", 0) != 0 or
                prime_result.get("total_input_tokens") != args.input_tokens or
                prime_result.get("total_output_tokens") != 1):
            raise BenchError(f"Prime request did not do the expected work: {prime_result}")
    after_prime = snapshot(args.port)
    prime_counts = check_primes(before_prime, after_prime, args.prime_engines,
                                args.input_tokens, args.block_size)
    assert_idle(after_prime)
    command = client_command(args, workload, n, directory, owner, f"sb-decode-{owner}-{phase}")
    timeout = min(args.timeout_s, int(deadline - time.monotonic()))
    if timeout < 1:
        raise BenchError("Decode-only total time budget exhausted")
    if args.profile_window and phase == "round-1":
        profile_control(args.port, "start", directory)
        try:
            run_client(command, directory, f"sb-decode-{owner}-{phase}", owner, timeout)
        finally:
            profile_control(args.port, "stop", directory)
    else:
        run_client(command, directory, f"sb-decode-{owner}-{phase}", owner, timeout)
    result = vllm_bench.normalize(directory / "raw.json", n, workload)
    after = snapshot(args.port)
    counts = deltas(after_prime, after)
    (directory / "counters.json").write_text(json.dumps({"prime": prime_counts, "measured": counts}, indent=2))
    check_round(counts, n, args.input_tokens, args.block_size)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Original 16K GovReport JSONL")
    p.add_argument("--source-sha256", default=PREFIX_SHA256)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--served-name", default="deepseek-v4-flash")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--client-image", default="vllm/vllm-openai:v0.30.0")
    p.add_argument("--client-image-id", default=IMAGE_ID)
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--input-tokens", type=int, default=16384)
    p.add_argument("--output-tokens", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--prime-engines", type=int, default=1,
                   help="Number of DP engines to prime twice each; verify prefix hits per engine")
    p.add_argument("--warmup-requests", type=int, default=256)
    p.add_argument("--requests", type=int, default=512)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--budget-s", type=int, default=4200)
    p.add_argument("--timeout-s", type=int, default=900)
    p.add_argument("--client-cpus")
    p.add_argument("--client-mems")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--plan", action="store_true", help="Offline validation only; no server or Docker access")
    p.add_argument("--allow-cache-reset", action="store_true", help="Required for live operation")
    p.add_argument("--profile-window", action="store_true", help="Start/stop server profiler around the first formal round")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if (args.concurrency < 1 or args.requests < args.concurrency or
                args.warmup_requests < args.concurrency or args.repetitions < 1 or
                args.input_tokens < 1 or args.output_tokens < 2 or args.block_size < 1 or
                args.prime_engines < 1 or args.budget_s < 1 or args.timeout_s < 1 or
                not 1 <= args.port <= 65535 or
                bool(args.client_cpus) != bool(args.client_mems) or
                (args.profile_window and args.repetitions != 1)):
            raise BenchError("Invalid concurrency, request count, length, budget, port or binding")
        if not args.model_dir.is_dir() or args.run_root.exists():
            raise BenchError("Model directory must exist and run-root must not exist")
        source = args.source.resolve()
        rows, source_sha = dataset_rows(source, args.source_sha256,
                                        max(args.requests, args.warmup_requests),
                                        args.output_tokens, args.input_tokens)
        plan = {"source": str(source), "source_sha256": source_sha,
                "model_dir": str(args.model_dir.resolve()), "port": args.port,
                "client_image_id": args.client_image_id,
                "concurrency": args.concurrency, "warmup_requests": args.warmup_requests,
                "requests_per_round": args.requests, "repetitions": args.repetitions,
                "total_formal_requests": args.requests * args.repetitions,
                "input_tokens": args.input_tokens, "output_tokens": args.output_tokens,
                "prime_engines": args.prime_engines,
                "method": ("one shared prefix; reset/"
                           f"{args.prime_engines * 2} primes per round; continuous refill")}
        if args.plan:
            print(json.dumps(plan, indent=2))
            return 0
        if not args.allow_cache_reset:
            raise BenchError("Live operation requires --allow-cache-reset on a dedicated server")
        image_info({"image": args.client_image, "image_id": args.client_image_id})
        root = args.run_root.resolve()
        root.mkdir(parents=True)
        (root / "plan.json").write_text(json.dumps(plan, indent=2))
        owner = uuid.uuid4().hex[:12]
        deadline = time.monotonic() + args.budget_s
        results = []
        for phase, n in [("warmup", args.warmup_requests),
                         *((f"round-{i}", args.requests) for i in range(1, args.repetitions + 1))]:
            result = measure(args, root, phase, n, rows, owner, deadline)
            if phase != "warmup":
                results.append(result)
        (root / "summary.json").write_text(json.dumps({"plan": plan, "rounds": results}, indent=2))
        return 0
    except (BenchError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"decode-only: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
