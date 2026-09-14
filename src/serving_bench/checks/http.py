from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..common import BenchError, write_json
from ..executors import docker


def request(base: str, endpoint: str, payload: dict | None = None, timeout: int = 30, parse_json: bool = True):
    req = urllib.request.Request(base + endpoint, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
            if not parse_json:
                return data.decode(errors="replace")
            return json.loads(data) if data.strip() else None
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise BenchError(f"HTTP {exc.code} at {endpoint}: {body[:2000]}") from exc
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise BenchError(f"HTTP request failed at {endpoint}: {exc}") from exc


def base_url(case: dict) -> str:
    return f"http://127.0.0.1:{case['target']['port']}"


def wait_ready(case: dict, container: str, log) -> float:
    start = time.monotonic()
    deadline = start + case["runtime"]["ready_timeout_s"]
    last_progress = 0.0
    while time.monotonic() < deadline:
        info = docker.container_info(container)
        if not info or not info["State"]["Running"]:
            raise BenchError("Server exited before readiness; inspect server.log")
        try:
            request(base_url(case), "/health", timeout=3, parse_json=False)
            return time.monotonic() - start
        except BenchError:
            pass
        if time.monotonic() - last_progress >= 30:
            log(f"Waiting for {case['id']} readiness ({time.monotonic() - start:.0f}s)")
            last_progress = time.monotonic()
        time.sleep(2)
    raise BenchError("Server readiness timeout; inspect server.log")


def probes(case: dict, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    models = request(base_url(case), "/v1/models")
    write_json(directory / "models.json", models)
    if not isinstance(models, dict) or case["model"]["served_name"] not in {x.get("id") for x in models.get("data", []) if isinstance(x, dict)}:
        raise BenchError("/v1/models did not list the configured served model")
    probe = case["recipe"].get("probe")
    if not probe:
        return {"models": "PASS", "chat": "NOT_CONFIGURED"}
    payload = {"temperature": 0, **probe.get("request", {}), "model": case["model"]["served_name"],
               "messages": [{"role": "user", "content": probe["prompt"]}],
               "max_tokens": probe["max_tokens"], "stream": False}
    write_json(directory / "chat-request.json", payload)
    response = request(base_url(case), "/v1/chat/completions", payload, timeout=180)
    write_json(directory / "chat-response.json", response)
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise BenchError("Chat probe returned no message") from exc
    if not isinstance(content, str) or not content.strip():
        raise BenchError("Chat probe returned empty content (check thinking settings/token budget)")
    if "expect_regex" in probe and not re.search(probe["expect_regex"], content):
        raise BenchError(f"Chat probe failed expected content check: {content}")
    return {"models": "PASS", "chat": "PASS", "semantic_assertion": "expect_regex" in probe}
