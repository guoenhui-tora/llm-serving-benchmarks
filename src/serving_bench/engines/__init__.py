from . import sglang, vllm
from ..common import BenchError


def get_engine(name: str):
    engines = {"vllm": vllm, "sglang": sglang}
    if not isinstance(name, str) or name not in engines:
        raise BenchError(f"Unsupported serving engine: {name}")
    return engines[name]
