from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path

from .common import BenchError


@contextmanager
def gpu_locks(gpus: list[int], root: Path = Path("/tmp/serving-bench-locks")):
    """Advisory locks coordinate this runner; they do not reserve GPUs in Docker."""
    root.mkdir(parents=True, exist_ok=True)
    streams = []
    try:
        for gpu in sorted(gpus):
            stream = (root / f"gpu-{gpu}.lock").open("a+")
            streams.append(stream)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BenchError(f"GPU {gpu} is locked by another benchmark") from exc
        yield
    finally:
        for stream in reversed(streams):
            stream.close()
