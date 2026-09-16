"""SGLang native launch and resource semantics."""
from .base import native_args, positive_option
from ..common import BenchError

ENTRYPOINT = "sglang"
PREFIX = ["serve"]
HELP_FLAG = "--help"
COMPILATION_PATTERNS = [r"JIT compilation during inference", r"\bAutotuning kernel\b",
                        r"\[AutoTuner\]\s*:?\s*(?:Tuning\b|(?:Autotuning\s+)?process starts\b)",
                        r"Running FlashInfer autotune", r"(?:nvcc|ninja).*compil"]


def server_args(case: dict) -> list[str]:
    return PREFIX + ["--model-path", "/model", "--served-model-name", case["model"]["served_name"],
                     "--host", "127.0.0.1", "--port", str(case["target"]["port"])] + native_args(case["recipe"])


def validate(case: dict) -> int:
    recipe = case["recipe"]
    tp = positive_option(recipe, "tp-size")
    pp = positive_option(recipe, "pp-size")
    dp = positive_option(recipe, "dp-size")
    # SGLang DP attention partitions a TP group; it is not an extra replica factor.
    if dp > 1 and ("enable-dp-attention" not in recipe["flags"] or tp % dp):
        raise BenchError("SGLang dp-size>1 requires enable-dp-attention and TP divisible by DP")
    if tp * pp != len(case["target"]["gpus"]):
        raise BenchError("SGLang TP×PP must match selected GPU count")
    if "disable-radix-cache" not in recipe["flags"]:
        raise BenchError("SGLang cache=disabled requires disable-radix-cache")
    return positive_option(recipe, "context-length")
