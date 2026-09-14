from __future__ import annotations

import json
import re

from ..common import BenchError

MANAGED_OPTIONS = {"model", "model-path", "served-model-name", "host", "port"}


def native_args(recipe: dict) -> list[str]:
    result = [f"--{flag}" for flag in recipe["flags"]]
    for key, value in recipe["options"].items():
        result.append(f"--{key}")
        if isinstance(value, dict):
            result.append(json.dumps(value, separators=(",", ":")))
        elif isinstance(value, list):
            result.extend(str(item) for item in value)
        else:
            result.append(str(value))
    return result


def positive_option(recipe: dict, key: str, default: int = 1) -> int:
    value = recipe["options"].get(key, default)
    if type(value) is not int or value < 1:
        raise BenchError(f"recipe.options.{key} must be a positive integer")
    return value


def check_native(recipe: dict) -> None:
    names = recipe["flags"] + list(recipe["options"])
    if len(names) != len(set(names)):
        raise BenchError("Duplicate recipe flag/option")
    for name in names:
        if name.startswith("no-") and name[3:] in names:
            raise BenchError(f"Conflicting native flags: {name} and {name[3:]}")
    for name in names:
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
            raise BenchError(f"Invalid native CLI option: {name}")
        if name in MANAGED_OPTIONS:
            raise BenchError(f"{name} is managed by the runner, not recipe")
    for key, value in recipe["options"].items():
        if isinstance(value, bool) or value is None:
            raise BenchError(f"options.{key}: use explicit native switches in flags, not boolean/null options")
        if not isinstance(value, (str, int, float, list, dict)):
            raise BenchError(f"Invalid option value: {key}")
        if isinstance(value, list) and (not value or any(type(x) not in (str, int, float) for x in value)):
            raise BenchError(f"options.{key}: list must contain scalar arguments")
