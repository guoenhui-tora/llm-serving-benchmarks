from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BenchError(RuntimeError):
    """An actionable configuration, environment or experiment failure."""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise BenchError(f"Cannot read JSON {path}: {exc}") from exc


def save_command(directory: Path, argv: list[str]) -> None:
    write_json(directory / "argv.json", argv)
    (directory / "command.sh").write_text(shlex.join(argv) + "\n")


def capture(argv: list[str], timeout: float = 60, check: bool = True) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BenchError(f"Command failed: {shlex.join(argv)}: {exc}") from exc
    if check and result.returncode:
        raise BenchError(f"Command failed ({result.returncode}): {shlex.join(argv)}\n{result.stdout[-4000:]}{result.stderr[-4000:]}")
    return result


def option_names(help_text: str) -> set[str]:
    return set(re.findall(r"(?<![\w-])--([a-z][a-z0-9-]*)", help_text))


class Logger:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, message: str) -> None:
        line = f"[{utcnow()}] {message}"
        print(line, flush=True)
        with self.path.open("a") as stream:
            stream.write(line + "\n")
