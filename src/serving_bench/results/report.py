from __future__ import annotations

import statistics
from pathlib import Path

from ..common import BenchError, fingerprint, read_json


def artifact(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise BenchError(f"Result reference must stay inside the run: {relative}")
    return path


def report(run_roots: list[Path]) -> dict:
    groups = {}
    runs = []
    seen = set()
    seen_owners = set()
    for root in run_roots:
        root = root.expanduser().resolve()
        if root in seen:
            continue
        seen.add(root)
        run = read_json(root / "run.json")
        if run.get("schema_version") != 1 or not all(k in run for k in ("owner", "runner_source_fingerprint", "campaign", "cases", "status")):
            raise BenchError(f"Unsupported or incomplete run manifest: {root}")
        if run["owner"] in seen_owners:
            continue
        seen_owners.add(run["owner"])
        overview = {"path": str(root), "campaign": run["campaign"], "status": run["status"], "cases": []}
        runs.append(overview)
        for ref in run["cases"]:
            directory = artifact(root, ref["path"])
            if not (directory / "case.json").is_file():
                overview["cases"].append({"id": ref["id"], "status": "PENDING"})
                continue
            state = read_json(directory / "case.json")
            overview["cases"].append({k: state[k] for k in ("id", "status", "error", "kernel_checks", "warning_count", "cleanup_errors") if k in state})
            # Partial/failed campaigns may contain successful cases. Only terminal successful cases enter statistics.
            if state["status"] != "PASS":
                continue
            case = read_json(directory / "resolved.json")
            facts = read_json(directory / "environment.json")
            for trial in state["trials"]:
                key = {"target": case["target"]["id"], "hostname": facts["hostname"],
                       "hardware": fingerprint(facts["gpus"]), "model": case["model"]["id"],
                       "model_identity": facts["model"]["identity"], "engine": case["runtime"]["engine"],
                       "server_image": facts["server_image"]["Id"], "runtime": case["runtime"]["id"],
                       "client_image": facts["client_image"]["Id"], "client_config": fingerprint(case["client"]),
                       "recipe": case["recipe"]["id"], "recipe_fingerprint": fingerprint(case["recipe"]),
                       "workload": trial["workload"], "workload_fingerprint": trial["workload_fingerprint"],
                       "concurrency": trial["concurrency"], "purpose": trial["purpose"],
                       "execution_target": fingerprint(case["target"]),
                       "runtime_config": fingerprint(case["runtime"])}
                if "replicas" in case:
                    key["replica_targets"] = fingerprint([m["target"] for m in case["replicas"]])
                    key["replica_count"] = len(case["replicas"])
                    key["measurement_protocol"] = "synchronized-equal-share-v1"
                key["runner_source"] = run["runner_source_fingerprint"]
                token = fingerprint(key)
                group = groups.setdefault(token, {**key, "samples": [], "sources": []})
                group["samples"].append(trial["metrics"]["metrics"])
                group["sources"].append({"run": str(root), "case": case["id"], "trial": trial["path"]})
    rows = []
    for group in groups.values():
        samples = group.pop("samples")
        group["n"] = len(samples)
        group["statistics"] = {}
        for field in sorted({key for sample in samples for key in sample}):
            values = [sample[field] for sample in samples if sample.get(field) is not None]
            group["statistics"][field] = {"mean": statistics.fmean(values) if values else None,
                                         "sd": statistics.stdev(values) if len(values) > 1 else None,
                                         "n": len(values)}
        rows.append(group)
    return {"schema_version": 1, "runs": runs, "groups": rows,
            "notes": ["Latency percentile means are means of per-trial percentiles, not pooled percentiles.",
                      "Smoke/calibration/performance are separate groups. Missing metrics remain null.",
                      "Model identity uses configuration/tokenizer hashes and shard sizes, not full weight hashes."]}


def markdown(value: dict) -> str:
    def safe(text):
        return str(text).replace("|", "\\|").replace("\n", " ")

    lines = []
    for run in value["runs"]:
        lines.append(f"{safe(run['campaign'])}: **{safe(run['status'])}**")
        for case in run["cases"]:
            line = f"- {safe(case['id'])}: {safe(case['status'])}"
            if case.get("error"):
                line += f" — {safe(case['error'])}"
            if case.get("warning_count"):
                line += f"; {case['warning_count']} warning log lines (see kernel-checks.json)"
            lines.append(line)
        lines.append("")
    lines += ["| Target | Model | Runtime | Recipe | Workload | Purpose | C | N | Output tok/s mean ± SD | Mean TTFT ms | Mean TPOT ms |",
              "|---|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for group in value["groups"]:
        stats = group["statistics"]

        def metric(name, field="mean"):
            value_ = stats.get(name, {}).get(field)
            return "N/A" if value_ is None else f"{value_:.2f}"

        row = [group[k] for k in ("target", "model", "runtime", "recipe", "workload", "purpose", "concurrency", "n")]
        row += [f"{metric('output_throughput')} ± {metric('output_throughput', 'sd')}", metric("mean_ttft_ms"), metric("mean_tpot_ms")]
        lines.append("| " + " | ".join(map(safe, row)) + " |")
    return "\n".join(lines) + "\n"
