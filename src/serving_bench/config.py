from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

import yaml

from .common import BenchError, fingerprint
from .engines import get_engine
from .engines.base import check_native
from .executors.affinity import id_set

FIELDS = {
    "target": ({"executor", "address", "gpus", "gpu", "model_root", "cache_root", "port"},
               {"model_paths", "environment", "ulimits", "cap_add", "binding"}),
    "model": ({"directory", "served_name", "architecture", "required_files"}, {"quantization"}),
    "runtime": ({"engine", "image", "version"}, {"image_id", "environment", "ready_timeout_s"}),
    "recipe": ({"compatible", "mode", "options", "flags"}, {"environment", "checks", "probe", "provenance"}),
    "client": ({"tool", "image", "version"}, {"image_id", "environment", "trust_remote_code"}),
    "workload": ({"dataset", "traffic", "sampling", "measurement", "cache", "purpose"}, set()),
    "campaign": ({"target", "client", "workloads", "cases"}, set()),
}


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise BenchError("YAML mapping keys must be strings")
        if key in result:
            raise BenchError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def mapping(value, where, required=(), optional=()):
    if not isinstance(value, dict):
        raise BenchError(f"{where} must be a mapping")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise BenchError(f"{where}: missing={sorted(missing)}, unknown={sorted(extra)}")
    return value


def string(value, where):
    if not isinstance(value, str) or not value.strip():
        raise BenchError(f"{where} must be a nonempty string")
    return value


def slug(value, where):
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,79}", string(value, where)) or value in {".", ".."}:
        raise BenchError(f"{where}: use a short lowercase ID containing letters, numbers, '.', '_' or '-'")


def integer(value, where, minimum=1):
    if type(value) is not int or value < minimum:
        raise BenchError(f"{where} must be an integer >= {minimum}")
    return value


def strings(value, where, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise BenchError(f"{where} must be a list")
    for item in value:
        string(item, where)
    return value


def enum(value, choices, where):
    if value not in choices:
        raise BenchError(f"{where} must be one of {choices}")


def number(value, where, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise BenchError(f"{where} must be a finite number >= {minimum}")


def environment(value, where):
    if not isinstance(value, dict):
        raise BenchError(f"{where} must be a mapping")
    for key, val in value.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or not isinstance(val, str):
            raise BenchError(f"{where}: environment names must be valid and values quoted strings")


def relative(value, where):
    path = Path(string(value, where))
    if path.is_absolute() or ".." in path.parts or str(path) == ".":
        raise BenchError(f"{where} must be a relative path without '..'")


def absolute(value, where):
    if not Path(string(value, where)).expanduser().is_absolute():
        raise BenchError(f"{where} must be absolute (or start with ~)")
    if ":" in value or "," in value or "\n" in value:
        raise BenchError(f"{where} contains unsupported mount-path characters")


def load_document(path: Path, kind: str) -> dict:
    try:
        value = yaml.load(path.read_text(), Loader=UniqueLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise BenchError(f"Cannot load {path}: {exc}") from exc
    required, optional = FIELDS[kind]
    mapping(value, str(path), required | {"schema_version", "kind", "id"}, optional | {"description"})
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["kind"] != kind:
        raise BenchError(f"{path}: expected schema_version: 1 and kind: {kind}")
    slug(value["id"], str(path))
    if "description" in value:
        string(value["description"], "description")
    if "environment" in value:
        environment(value["environment"], "environment")
    validate_document(value, kind)
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise BenchError(f"{path}: use finite, JSON-compatible YAML values without cyclic aliases") from exc
    return value


def validate_document(v: dict, kind: str) -> None:
    if kind == "target":
        enum(v["executor"], ["docker"], "target.executor")
        string(v["address"], "target.address")
        if not isinstance(v["gpus"], list) or not v["gpus"]:
            raise BenchError("target.gpus must be a nonempty list")
        for gpu in v["gpus"]:
            integer(gpu, "target.gpus", 0)
        if len(v["gpus"]) != len(set(v["gpus"])):
            raise BenchError("Duplicate GPU IDs")
        mapping(v["gpu"], "target.gpu", {"vendor", "name_regex", "compute_capability", "min_memory_mib"})
        enum(v["gpu"]["vendor"], ["nvidia"], "target.gpu.vendor")
        regex(v["gpu"]["name_regex"])
        string(v["gpu"]["compute_capability"], "compute_capability (quote it)")
        integer(v["gpu"]["min_memory_mib"], "min_memory_mib")
        for key in ("model_root", "cache_root"):
            absolute(v[key], key)
        integer(v["port"], "target.port", 1024)
        if v["port"] > 65535:
            raise BenchError("target.port exceeds 65535")
        if not isinstance(v.get("model_paths", {}), dict):
            raise BenchError("model_paths must be a mapping of model IDs to absolute paths")
        for key, val in v.get("model_paths", {}).items():
            slug(key, "model_paths key")
            absolute(val, "model_paths value")
        environment(v.get("ulimits", {}), "ulimits")
        strings(v.get("cap_add", []), "cap_add")
        if "binding" in v:
            bindings = mapping(v["binding"], "target.binding", (), {"server", "client"})
            if not bindings:
                raise BenchError("target.binding must specify server and/or client")
            for role, binding in bindings.items():
                mapping(binding, "target.binding." + role, {"cpus", "mems"})
                try:
                    for value in binding.values():
                        id_set(value)
                except ValueError as exc:
                    raise BenchError(f"target.binding.{role}: {exc}") from exc
            if "server" in bindings and "client" in bindings:
                if id_set(bindings["server"]["cpus"]) & id_set(bindings["client"]["cpus"]):
                    raise BenchError("Server/client CPU bindings overlap")
    elif kind == "model":
        relative(v["directory"], "model.directory")
        string(v["served_name"], "model.served_name")
        string(v["architecture"], "model.architecture")
        if "quantization" in v:
            string(v["quantization"], "model.quantization")
        for item in strings(v["required_files"], "required_files", True):
            relative(item, "required_files")
    elif kind in {"runtime", "client"}:
        string(v["image"], "image")
        string(v["version"], "version (quote it)")
        if "image_id" in v and not re.fullmatch(r"sha256:[a-f0-9]{64}", v["image_id"]):
            raise BenchError("image_id must be a complete sha256 Docker image ID")
        if kind == "runtime":
            get_engine(v["engine"])
            integer(v.setdefault("ready_timeout_s", 1800), "ready_timeout_s")
        else:
            enum(v["tool"], ["vllm-bench"], "client.tool")
            if type(v.setdefault("trust_remote_code", False)) is not bool:
                raise BenchError("trust_remote_code must be boolean")
    elif kind == "recipe":
        if "provenance" in v:
            string(v["provenance"], "recipe.provenance")
        c = mapping(v["compatible"], "recipe.compatible", {"engine", "models", "compute_capabilities"}, {"runtime_versions"})
        get_engine(c["engine"])
        strings(c["models"], "compatible.models", True)
        strings(c["compute_capabilities"], "compatible.compute_capabilities", True)
        strings(c.get("runtime_versions", []), "compatible.runtime_versions")
        enum(v["mode"], ["smoke", "performance"], "recipe.mode")
        if not isinstance(v["options"], dict):
            raise BenchError("recipe.options must be a mapping")
        strings(v["flags"], "recipe.flags")
        check_native(v)
        checks = mapping(v.setdefault("checks", {}), "recipe.checks", (), {"required", "forbidden", "warnings", "compilation"})
        for key in ("required", "forbidden", "warnings", "compilation"):
            for pattern in strings(checks.setdefault(key, []), "checks." + key):
                regex(pattern)
        if "probe" in v:
            p = mapping(v["probe"], "recipe.probe", {"prompt", "max_tokens"}, {"request", "expect_regex"})
            string(p["prompt"], "probe.prompt")
            integer(p["max_tokens"], "probe.max_tokens")
            if not isinstance(p.setdefault("request", {}), dict):
                raise BenchError("probe.request must be a mapping")
            reserved = {"model", "messages", "prompt", "max_tokens", "stream"} & p["request"].keys()
            if reserved:
                raise BenchError(f"probe.request overrides managed fields: {reserved}")
            if "expect_regex" in p:
                regex(p["expect_regex"])
    elif kind == "workload":
        d = mapping(v["dataset"], "dataset", {"name", "input_tokens", "output_tokens"}, {"range_ratio"})
        enum(d["name"], ["random"], "dataset.name")
        for key in ("input_tokens", "output_tokens"):
            integer(d[key], "dataset." + key)
        number(d.setdefault("range_ratio", 0), "range_ratio")
        if d["range_ratio"] > 1:
            raise BenchError("range_ratio must be <=1")
        t = mapping(v["traffic"], "traffic", {"concurrency", "requests", "request_rate"})
        if not isinstance(t["concurrency"], list) or not t["concurrency"]:
            raise BenchError("traffic.concurrency must be a nonempty list")
        for c in t["concurrency"]:
            integer(c, "traffic.concurrency")
        if len(t["concurrency"]) != len(set(t["concurrency"])):
            raise BenchError("Duplicate concurrency values")
        integer(t["requests"], "traffic.requests", max(t["concurrency"]))
        if t["request_rate"] != "inf":
            number(t["request_rate"], "request_rate", 0.000001)
        s = mapping(v["sampling"], "sampling", {"seed", "temperature", "ignore_eos"})
        integer(s["seed"], "sampling.seed", 0)
        number(s["temperature"], "temperature")
        if type(s["ignore_eos"]) is not bool:
            raise BenchError("ignore_eos must be boolean")
        if isinstance(v["measurement"], dict):
            old = {"warmup_requests", "min_warmup_rounds", "max_warmup_rounds", "max_attempts"} & v["measurement"].keys()
            if old or "protocol" not in v["measurement"]:
                raise BenchError("Replace the entire measurement block with protocol: quick|jit_clean|stable "
                                 "and budget_s; old warmup/retry fields are no longer supported. "
                                 "See docs/configuration.md#workload")
        m = mapping(v["measurement"], "measurement", {"protocol", "budget_s"},
                    {"repetitions", "timeout_s", "max_rounds", "warmup_rounds", "stability_threshold"})
        enum(m["protocol"], ["quick", "jit_clean", "stable"], "measurement.protocol")
        integer(m["budget_s"], "measurement.budget_s")
        minimum = 1 if v["purpose"] == "smoke" and m["protocol"] != "stable" else 3
        integer(m.setdefault("repetitions", minimum), "measurement.repetitions", minimum)
        integer(m.setdefault("timeout_s", 1800), "measurement.timeout_s")
        if m["protocol"] == "quick":
            integer(m.setdefault("warmup_rounds", 1), "measurement.warmup_rounds")
            if m["warmup_rounds"] not in (1, 2):
                raise BenchError("quick warmup_rounds must be 1 or 2")
            integer(m.setdefault("max_rounds", m["repetitions"]), "measurement.max_rounds")
            if m["max_rounds"] != m["repetitions"]:
                raise BenchError("quick max_rounds must equal repetitions (no automatic retries)")
        else:
            if "warmup_rounds" in m:
                raise BenchError("Only quick accepts warmup_rounds; other protocols use full rounds")
            integer(m.setdefault("max_rounds", 12), "measurement.max_rounds", m["repetitions"])
        if m["protocol"] == "stable":
            number(m.setdefault("stability_threshold", 0.02), "measurement.stability_threshold")
            if not 0 < m["stability_threshold"] <= 1:
                raise BenchError("stability_threshold is a fraction, e.g. 0.02 for 2%")
        elif "stability_threshold" in m:
            raise BenchError("Only stable accepts stability_threshold")
        enum(v["cache"], ["disabled"], "workload.cache")
        enum(v["purpose"], ["smoke", "calibration", "performance"], "workload.purpose")
    elif kind == "campaign":
        strings(v["workloads"], "campaign.workloads", True)
        if not isinstance(v["cases"], list) or not v["cases"]:
            raise BenchError("campaign.cases must be a nonempty list")
        seen = set()
        for case in v["cases"]:
            mapping(case, "campaign.case", {"id", "model", "runtime", "recipe"}, {"replica_targets"})
            if "replica_targets" in case:
                strings(case["replica_targets"], "case.replica_targets", True)
                if len(case["replica_targets"]) > 2:
                    raise BenchError("Synchronized deployments currently support one or two replicas")
            slug(case["id"], "case.id")
            if case["id"] in seen:
                raise BenchError("Duplicate case ID")
            seen.add(case["id"])


def regex(pattern):
    try:
        re.compile(string(pattern, "regex"))
    except re.error as exc:
        raise BenchError(f"Invalid regex {pattern!r}: {exc}") from exc


def resolve(campaign_path: str | Path, config_root: str | Path | None = None, case_ids: list[str] | None = None) -> dict:
    path = Path(campaign_path).expanduser().resolve()
    root = Path(config_root).expanduser().resolve() if config_root else next((p for p in path.parents if p.name == "configs"), None)
    if root is None or not path.is_relative_to(root):
        raise BenchError("Campaign must be inside configs/ or an explicit --config-root")
    documents = {}

    def load(ref, kind):
        relative(ref, f"{kind} reference")
        resolved = (root / ref).resolve()
        if not resolved.is_relative_to(root):
            raise BenchError(f"Config reference escapes root: {ref}")
        doc = load_document(resolved, kind)
        documents[str(resolved.relative_to(root))] = doc
        return copy.deepcopy(doc)

    campaign = load(str(path.relative_to(root)), "campaign")
    target = load(campaign["target"], "target")
    client = load(campaign["client"], "client")
    workloads = [load(ref, "workload") for ref in campaign["workloads"]]
    if len({w["id"] for w in workloads}) != len(workloads):
        raise BenchError("Duplicate workload IDs")
    selected = set(case_ids or [c["id"] for c in campaign["cases"]])
    if selected - {c["id"] for c in campaign["cases"]}:
        raise BenchError(f"Unknown case IDs: {selected}")
    cases = []
    for spec in campaign["cases"]:
        if spec["id"] not in selected:
            continue
        case = {"id": spec["id"], "target": copy.deepcopy(target), "client": copy.deepcopy(client), "workloads": copy.deepcopy(workloads)}
        for kind in ("model", "runtime", "recipe"):
            case[kind] = load(spec[kind], kind)
        compatible = case["recipe"]["compatible"]
        if compatible["engine"] != case["runtime"]["engine"]:
            raise BenchError(f"{case['id']}: recipe/runtime engine mismatch")
        if case["model"]["id"] not in compatible["models"]:
            raise BenchError(f"{case['id']}: recipe/model mismatch")
        if target["gpu"]["compute_capability"] not in compatible["compute_capabilities"]:
            raise BenchError(f"{case['id']}: recipe/hardware mismatch")
        if compatible.get("runtime_versions") and case["runtime"]["version"] not in compatible["runtime_versions"]:
            raise BenchError(f"{case['id']}: runtime version outside recipe compatibility list")
        members = [case]
        if "replica_targets" in spec:
            if client["version"] != "0.29.0":
                raise BenchError("Synchronized client requires the verified vLLM benchmark version 0.29.0")
            members = []
            for index, ref in enumerate(spec["replica_targets"]):
                member = copy.deepcopy(case)
                member["id"] = f"replica-{index}"
                member["target"] = load(ref, "target")
                mt = member["target"]
                for field in ("address", "gpu", "model_root", "model_paths", "cache_root"):
                    if mt.get(field) != target.get(field):
                        raise BenchError(f"Replica target {field} differs from campaign target")
                members.append(member)
            validate_replica_targets(target, [m["target"] for m in members])
            n = len(members)
            for w in workloads:
                if w["traffic"]["request_rate"] != "inf":
                    raise BenchError("Synchronized deployments require request_rate=inf")
                for value in [*w["traffic"]["concurrency"], w["traffic"]["requests"]]:
                    if value % n:
                        raise BenchError("Global concurrency/request counts must divide evenly across replicas")
            case["replicas"] = members
        max_len = min(get_engine(m["runtime"]["engine"]).validate(m) for m in members)
        for w in workloads:
            if w["dataset"]["input_tokens"] + w["dataset"]["output_tokens"] > max_len:
                raise BenchError(f"{case['id']}/{w['id']}: input+output exceeds context length")
            if case["recipe"]["mode"] == "smoke" and w["purpose"] != "smoke":
                raise BenchError("A smoke recipe cannot run calibration/performance workloads")
        model = case["model"]
        case["model_path"] = str(Path(target.get("model_paths", {}).get(model["id"], str(Path(target["model_root"]) / model["directory"]))).expanduser())
        if "replicas" in case:
            for member in case["replicas"]:
                member["model_path"] = case["model_path"]
        cases.append(case)
    plan = {"schema_version": 1, "campaign": campaign["id"], "cases": cases, "documents": documents}
    plan["fingerprint"] = fingerprint(plan)
    return plan


def validate_replica_targets(target: dict, members: list[dict]) -> None:
    """Require an exact partition of the campaign GPU and CPU allocations."""
    ports, gpus = set(), set()
    used_cpus = set()
    role_cpus = {"server": set(), "client": set()}
    role_mems = {"server": set(), "client": set()}
    for member in members:
        if member["port"] in ports or gpus & set(member["gpus"]):
            raise BenchError("Replica ports and GPU allocations must be disjoint")
        ports.add(member["port"])
        gpus.update(member["gpus"])
        if set(member.get("binding", {})) != {"server", "client"}:
            raise BenchError("Each replica requires explicit server and client CPU/memory binding")
        for role, binding in member["binding"].items():
            cpus = id_set(binding["cpus"])
            if used_cpus & cpus:
                raise BenchError("Replica CPU allocations overlap")
            used_cpus.update(cpus)
            role_cpus[role].update(cpus)
            role_mems[role].update(id_set(binding["mems"]))
    if gpus != set(target["gpus"]):
        raise BenchError("Replica GPUs must exactly partition campaign target GPUs")
    if set(target.get("binding", {})) != {"server", "client"}:
        raise BenchError("Campaign target requires the complete CPU/memory budget")
    for role in role_cpus:
        if role_cpus[role] != id_set(target["binding"][role]["cpus"]) or role_mems[role] != id_set(target["binding"][role]["mems"]):
            raise BenchError("Replica CPU/memory unions must match campaign budget")
