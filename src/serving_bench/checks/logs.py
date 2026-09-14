import re

from ..engines import get_engine


def matching(text: str, patterns: list[str]) -> list[str]:
    compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
    return [line for line in text.splitlines() if any(p.search(line) for p in compiled)]


def compilation_events(case: dict, text: str) -> list[str]:
    patterns = get_engine(case["runtime"]["engine"]).COMPILATION_PATTERNS + case["recipe"]["checks"]["compilation"]
    return matching(text, patterns)


def kernel_checks(case: dict, text: str) -> dict:
    checks = case["recipe"]["checks"]
    missing = [p for p in checks["required"] if not re.search(p, text, re.IGNORECASE)]
    forbidden = matching(text, checks["forbidden"])
    return {"status": "FAIL" if missing or forbidden else ("PASS" if checks["required"] else "UNVERIFIED"),
            "missing": missing, "forbidden": forbidden,
            "evidence": matching(text, checks["required"]), "warnings": matching(text, checks["warnings"]),
            "scope": "Configured log patterns only; not proof of full optimization coverage or numerical accuracy."}
