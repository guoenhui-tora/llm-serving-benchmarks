from __future__ import annotations

import csv
import hashlib
import json
import re
import socket
import subprocess
import uuid
from pathlib import Path

from ..common import BenchError, capture, fingerprint, option_names, read_json
from ..engines import get_engine
from . import affinity

OWNER_LABEL = "io.serving-bench.run"


def image_info(config: dict) -> dict:
    result = json.loads(capture(["docker", "image", "inspect", config["image"]]).stdout)[0]
    if config.get("image_id") and config["image_id"] != result["Id"]:
        raise BenchError(f"Image ID mismatch for {config['image']}: {result['Id']}")
    return result


def local_addresses() -> set[str]:
    values = json.loads(capture(["ip", "-j", "address", "show"]).stdout)
    return {item["local"] for interface in values for item in interface.get("addr_info", [])}


def gpu_inventory() -> list[dict]:
    text = capture(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"]).stdout
    result = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            raise BenchError(f"Unexpected nvidia-smi GPU row: {row}")
        result.append(dict(zip(("index", "uuid", "name", "memory_mib", "driver", "compute_capability"), [x.strip() for x in row])))
    return result


def validate_gpus(target: dict, inventory: list[dict]) -> list[dict]:
    by_index = {int(g["index"]): g for g in inventory}
    selected = []
    for index in target["gpus"]:
        if index not in by_index:
            raise BenchError(f"GPU {index} is not present")
        gpu = by_index[index]
        expected = target["gpu"]
        if not re.search(expected["name_regex"], gpu["name"]):
            raise BenchError(f"GPU {index} name mismatch: {gpu['name']}")
        if gpu["compute_capability"] != expected["compute_capability"]:
            raise BenchError(f"GPU {index} compute capability mismatch")
        if float(gpu["memory_mib"]) < expected["min_memory_mib"]:
            raise BenchError(f"GPU {index} has insufficient physical memory")
        selected.append(gpu)
    return selected


def check_idle(selected: list[dict]) -> None:
    result = capture(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"])
    uuids = {gpu["uuid"] for gpu in selected}
    active = [row for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True) if row and row[0].strip() in uuids]
    if active:
        raise BenchError(f"Selected GPUs already have compute processes: {active}")


def model_info(case: dict) -> dict:
    root = Path(case["model_path"]).resolve()
    if not root.is_dir():
        raise BenchError(f"Model directory not found: {root}")

    def within(relative: str) -> Path:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise BenchError(f"Missing model file or path outside model directory: {relative}")
        return path

    for relative in case["model"]["required_files"]:
        within(relative)
    config = read_json(within("config.json"))
    if case["model"]["architecture"] not in config.get("architectures", []):
        raise BenchError(f"Model architecture mismatch: {config.get('architectures')}")
    index_path = root / "model.safetensors.index.json"
    shards = {}
    if index_path.is_file():
        for relative in sorted(set(read_json(index_path).get("weight_map", {}).values())):
            shards[relative] = within(relative).stat().st_size
        if not shards:
            raise BenchError("Checkpoint index has no weight shards")
    hashes = {}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"):
        path = root / name
        if path.is_file():
            hashes[name] = hashlib.sha256(within(name).read_bytes()).hexdigest()
    return {"config": config, "file_hashes": hashes, "shard_sizes": shards,
            "identity": fingerprint({"hashes": hashes, "shards": shards}),
            "weight_content_hashed": False}


def inspect_cli(image: str, entrypoint: str, arguments: list[str], environment: dict, help_flag: str = "--help", gpus: list[int] | None = None) -> str:
    owner = uuid.uuid4().hex[:16]
    name = f"sb-help-{owner}"
    argv = ["docker", "run", "--rm", "--pull", "never", "--network", "none", "--name", name, "--label", f"{OWNER_LABEL}={owner}"]
    if gpus:
        argv += ["--gpus", '"device=' + ",".join(map(str, gpus)) + '"']
    for key, value in environment.items():
        argv += ["-e", f"{key}={value}"]
    argv += ["--entrypoint", entrypoint, image, *arguments, help_flag]
    try:
        result = capture(argv, timeout=180)
    finally:
        remove_owned(name, owner)
    return result.stdout + result.stderr


def preflight(case: dict) -> dict:
    target = case["target"]
    capture(["docker", "info"])
    if target["address"] not in local_addresses():
        raise BenchError(f"Run this target locally on {target['address']}; this runner does not SSH automatically")
    binding_facts = affinity.host_check(target)
    selected = validate_gpus(target, gpu_inventory())
    check_idle(selected)
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", target["port"]))
        except OSError as exc:
            raise BenchError(f"Port {target['port']} is busy") from exc
    server_image = image_info(case["runtime"])
    client_image = image_info(case["client"])
    engine = get_engine(case["runtime"]["engine"])
    server_help = inspect_cli(server_image["Id"], engine.ENTRYPOINT, engine.PREFIX,
                              {**target.get("environment", {}), **case["runtime"].get("environment", {}), **case["recipe"].get("environment", {})},
                              help_flag=engine.HELP_FLAG, gpus=[target["gpus"][0]])
    required = set(case["recipe"]["flags"]) | case["recipe"]["options"].keys() | {"host", "port", "served-model-name"}
    missing = required - option_names(server_help)
    if missing:
        raise BenchError(f"Server CLI lacks configured options: {sorted(missing)}")
    from ..clients.vllm_bench import ENTRYPOINT, PREFIX, arguments
    client_help = inspect_cli(client_image["Id"], ENTRYPOINT, PREFIX, case["client"].get("environment", {}))
    args = arguments(case, case["workloads"][0], 1, 1)
    missing = {x[2:] for x in args if x.startswith("--")} - option_names(client_help)
    if missing:
        raise BenchError(f"Client CLI lacks required options: {sorted(missing)}")
    return {"hostname": socket.gethostname(), "gpus": selected, "server_image": server_image,
            "client_image": client_image, "model": model_info(case), "binding": binding_facts,
            "server_help": server_help, "client_help": client_help}


def cache_directory(case: dict, image_id: str) -> Path:
    key = fingerprint({"image": image_id, "cc": case["target"]["gpu"]["compute_capability"],
                       "model": case["model"]["id"], "recipe": case["recipe"]})[:20]
    return Path(case["target"]["cache_root"]).expanduser() / case["runtime"]["engine"] / key


def server_command(case: dict, name: str, owner: str, image_id: str | None = None) -> list[str]:
    target = case["target"]
    image = image_id or case["runtime"]["image"]
    engine = get_engine(case["runtime"]["engine"])
    argv = ["docker", "run", "-d", "--pull", "never", "--name", name, "--label", f"{OWNER_LABEL}={owner}",
            "--network", "host", "--ipc", "host", "--gpus", '"device=' + ",".join(map(str, target["gpus"])) + '"',
            "-v", f"{case['model_path']}:/model:ro", "-v", f"{cache_directory(case, image)}:/root/.cache:rw"]
    argv += affinity.docker_args(target, "server")
    for name_, value in target.get("ulimits", {}).items():
        argv += ["--ulimit", f"{name_}={value}"]
    for capability in target.get("cap_add", []):
        argv += ["--cap-add", capability]
    env = {**target.get("environment", {}), **case["runtime"].get("environment", {}), **case["recipe"].get("environment", {})}
    for key, value in env.items():
        argv += ["-e", f"{key}={value}"]
    return argv + ["--entrypoint", engine.ENTRYPOINT, image, *engine.server_args(case)]


def container_info(name: str) -> dict | None:
    result = capture(["docker", "inspect", name], check=False)
    if result.returncode:
        diagnostic = (result.stderr + result.stdout).lower()
        if "no such object" in diagnostic or "no such container" in diagnostic:
            return None
        raise BenchError(f"Cannot inspect {name}: {result.stderr}")
    return json.loads(result.stdout)[0]


def binding_evidence(target: dict, role: str, name: str, owner: str, directory: Path,
                     *, live: bool = False) -> None:
    binding = target.get("binding", {}).get(role)
    if not binding:
        return
    from ..common import write_json
    info = container_info(name)
    if info is None or (info.get("Config", {}).get("Labels") or {}).get(OWNER_LABEL) != owner:
        raise BenchError(f"Cannot inspect owned {role} container binding")
    write_json(directory / f"{role}-binding-inspect.json", info)
    affinity.verify_inspect(binding, info)
    if live:
        result = capture(["docker", "exec", name, "python3", "-c",
                          affinity.probe_source() + "\nprint(json.dumps(snapshot()))"], timeout=30)
        observed = json.loads(result.stdout)
        write_json(directory / f"{role}-affinity.json", observed)
        try:
            affinity.verify_snapshot(binding, observed)
        except ValueError as exc:
            raise BenchError(str(exc)) from exc


def remove_owned(name: str, owner: str) -> None:
    info = container_info(name)
    if info is None:
        return
    if (info.get("Config", {}).get("Labels") or {}).get(OWNER_LABEL) != owner:
        raise BenchError(f"Refusing to remove container not owned by this run: {name}")
    capture(["docker", "rm", "-f", name], timeout=120)


def server_logs(name: str, since: str | None = None) -> str:
    argv = ["docker", "logs", "--timestamps"]
    if since:
        argv += ["--since", since]
    result = capture(argv + [name], timeout=120)
    return result.stdout + result.stderr


def run_client(argv: list[str], log_path: Path, timeout: int) -> None:
    try:
        with log_path.open("w") as stream:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BenchError(f"Client failed or timed out; see {log_path}: {exc}") from exc
    if result.returncode:
        raise BenchError(f"Client exited {result.returncode}; see {log_path}")


def environment_snapshot() -> dict:
    result = {}
    for name, argv in {"gpu": ["nvidia-smi", "-q"], "topology": ["nvidia-smi", "topo", "-m"],
                       "cpu": ["lscpu"], "memory": ["free", "-h"], "docker": ["docker", "info"]}.items():
        completed = capture(argv, timeout=120, check=False)
        result[name] = {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
    return result
