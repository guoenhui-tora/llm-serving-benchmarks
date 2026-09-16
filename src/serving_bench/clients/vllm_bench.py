from __future__ import annotations

import math
from pathlib import Path

from ..common import BenchError, read_json
from ..executors.docker import OWNER_LABEL
from ..executors import affinity

ENTRYPOINT = "python3"
# vLLM 0.29's top-level CLI constructs server parsers even for `bench`, which
# requires a visible GPU. Invoke the same official benchmark functions directly.
PREFIX = ["-c", "from vllm.benchmarks.serve import add_cli_args, main; "
          "from vllm.utils.argparse_utils import FlexibleArgumentParser; "
          "p=FlexibleArgumentParser(); add_cli_args(p); main(p.parse_args())"]

METRICS = ("output_throughput", "request_throughput", "duration", "total_input_tokens", "total_output_tokens",
           "mean_ttft_ms", "median_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p95_tpot_ms",
           "mean_itl_ms", "p95_itl_ms", "mean_e2el_ms", "p95_e2el_ms")


def arguments(case: dict, workload: dict, concurrency: int, requests: int) -> list[str]:
    d, t, s = workload["dataset"], workload["traffic"], workload["sampling"]
    # "vllm" names the client's OpenAI completions transport, not the server engine.
    args = ["--backend", "vllm", "--base-url", f"http://127.0.0.1:{case['target']['port']}",
            "--endpoint", "/v1/completions", "--model", "/model", "--tokenizer", "/model",
            "--served-model-name", case["model"]["served_name"], "--dataset-name", "random",
            "--random-input-len", str(d["input_tokens"]), "--random-output-len", str(d["output_tokens"]),
            "--random-range-ratio", str(d["range_ratio"]), "--num-prompts", str(requests),
            "--max-concurrency", str(concurrency), "--request-rate", str(t["request_rate"]),
            "--num-warmups", "0", "--seed", str(s["seed"]), "--temperature", str(s["temperature"]),
            "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,95,99",
            "--save-result", "--result-dir", "/results", "--result-filename", "raw.json"]
    if s["ignore_eos"]:
        args.append("--ignore-eos")
    if case["client"]["trust_remote_code"]:
        args.append("--trust-remote-code")
    return args


def command(case: dict, workload: dict, concurrency: int, requests: int, directory: Path,
            name: str, owner: str, image_id: str | None = None) -> list[str]:
    argv = ["docker", "run", "--rm", "--pull", "never", "--name", name, "--label", f"{OWNER_LABEL}={owner}",
            "--network", "host", "-v", f"{case['model_path']}:/model:ro", "-v", f"{directory}:/results:rw"]
    binding = case["target"].get("binding", {}).get("client")
    prefix = PREFIX
    if binding:
        argv.remove("--rm")  # Retain inspect evidence until runner-owned cleanup.
        argv += affinity.docker_args(case["target"], "client")
        prefix = affinity.client_prefix(PREFIX, binding)
    for key, value in case["client"].get("environment", {}).items():
        argv += ["-e", f"{key}={value}"]
    return argv + ["--entrypoint", ENTRYPOINT, image_id or case["client"]["image"], *prefix,
                   *arguments(case, workload, concurrency, requests)]


def normalize(path: Path, expected: int, workload: dict | None = None) -> dict:
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise BenchError("Client result must be a JSON object")
    completed = raw.get("completed")
    failed = raw.get("failed", raw.get("num_failed_requests"))
    if type(completed) is not int or completed != expected:
        raise BenchError(f"Request count mismatch: completed={completed}, expected={expected}")
    # Some pinned vLLM clients omit failed; completed==requested proves no missing success.
    if failed is not None and (type(failed) is not int or failed != 0):
        raise BenchError(f"Client reported failed requests: {failed}")
    metrics = {}
    for key in METRICS:
        value = raw.get(key)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise BenchError(f"Invalid metric {key}: {value}")
        metrics[key] = value
    for key in ("output_throughput", "mean_ttft_ms", "mean_tpot_ms"):
        if metrics[key] is None:
            raise BenchError(f"Client result lacks required metric: {key}")
    if workload and workload["sampling"]["ignore_eos"] and workload["dataset"]["range_ratio"] == 0:
        requested_tokens = expected * workload["dataset"]["output_tokens"]
        if metrics["total_output_tokens"] != requested_tokens:
            raise BenchError(f"Fixed-length output mismatch: got {metrics['total_output_tokens']}, expected {requested_tokens}; verify ignore_eos compatibility")
    return {"schema_version": 1, "completed": completed, "failed": failed,
            "failure_count_inferred": failed is None, "metrics": metrics}
