"""vLLM native launch and resource semantics."""
from .base import native_args, positive_option
from ..common import BenchError

ENTRYPOINT = "vllm"
PREFIX = ["serve"]
HELP_FLAG = "--help=all"
COMPILATION_PATTERNS = [r"JIT compilation during inference", r"Triton autotun(?:ing|e)",
                        r"\bAutotuning kernel\b", r"\[AutoTuner\].*(?:Tuning|process starts)",
                        r"torch\.compile.*(?:compiling|compilation took)"]


def server_args(case: dict) -> list[str]:
    return PREFIX + ["/model", "--served-model-name", case["model"]["served_name"],
                     "--host", "127.0.0.1", "--port", str(case["target"]["port"])] + native_args(case["recipe"])


def validate(case: dict) -> int:
    recipe = case["recipe"]
    world = (positive_option(recipe, "tensor-parallel-size") *
             positive_option(recipe, "pipeline-parallel-size") *
             positive_option(recipe, "data-parallel-size"))
    flags = recipe["flags"]
    if "no-enable-prefix-caching" not in flags or "enable-prefix-caching" in flags:
        raise BenchError("vLLM cache=disabled requires no-enable-prefix-caching")
    if world != len(case["target"]["gpus"]):
        raise BenchError("vLLM TP×PP×DP must match selected GPU count")
    return positive_option(recipe, "max-model-len")
