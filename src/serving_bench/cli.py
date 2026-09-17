from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .common import BenchError
from . import deployment
from .clients import vllm_bench
from .config import resolve
from .executors import docker
from .results.report import markdown, report
from .runner import run, stop


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="bench", description="Configuration-driven LLM serving experiments")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("validate", "plan", "preflight", "run"):
        p = commands.add_parser(name)
        p.add_argument("campaign", type=Path)
        p.add_argument("--config-root", type=Path)
        p.add_argument("--case", action="append", dest="case_ids", help="Select case ID (repeatable)")
        if name == "run":
            p.add_argument("--run-root", type=Path, help="New directory; refuses to overwrite any existing directory")
    p = commands.add_parser("report", help="Report or compare one or more portable run directories")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--json", action="store_true")
    p = commands.add_parser("stop", help="Stop only containers owned by this run, on the original host")
    p.add_argument("run_root", type=Path)
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command in {"validate", "plan", "preflight", "run"}:
            plan = resolve(args.campaign, args.config_root, args.case_ids)
            if args.command == "validate":
                print(f"Valid: {plan['campaign']}, {len(plan['cases'])} case(s), fingerprint={plan['fingerprint'][:16]}")
            elif args.command == "plan":
                plan["commands"] = {}
                for case in plan["cases"]:
                    workload = case["workloads"][0]
                    if "replicas" in case:
                        previews = []
                        n = len(case["replicas"])
                        for i, member in enumerate(case["replicas"]):
                            from copy import deepcopy
                            preview = deepcopy(member)
                            preview["_client_sync"] = {"directory": "/COORDINATION_DIRECTORY", "index": i,
                                "replicas": n, "global_requests": workload["traffic"]["requests"], "timeout_s": 300}
                            previews.append({"server": docker.server_command(member, f"sb-preview-r{i}-server", f"preview-r{i}"),
                                "client_first_workload": vllm_bench.command(preview, workload, workload["traffic"]["concurrency"][0]//n,
                                    workload["traffic"]["requests"]//n, Path(f"/RESULT_DIRECTORY/replica-{i}"), f"sb-preview-r{i}-client", f"preview-r{i}")})
                        plan["commands"][case["id"]] = {"replicas": previews, "dispatch": "equal-share, interleaved global dataset"}
                    else:
                        plan["commands"][case["id"]] = {
                            "server": docker.server_command(case, "sb-preview-server", "preview"),
                            "client_first_workload": vllm_bench.command(case, workload, workload["traffic"]["concurrency"][0],
                                workload["traffic"]["requests"], Path("/RESULT_DIRECTORY"), "sb-preview-client", "preview")}
                print(json.dumps(plan, indent=2, ensure_ascii=False))
            elif args.command == "preflight":
                for case in plan["cases"]:
                    facts = deployment.preflight(case)
                    print(f"Preflight passed: {case['id']}; server={facts['server_image']['Id']}; client={facts['client_image']['Id']}")
            else:
                root = args.run_root or Path("results") / plan["campaign"] / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                state = run(plan, root)
                return 0 if state["status"] == "PASS" else (130 if state["status"] == "INTERRUPTED" else 1)
        elif args.command == "report":
            value = report(args.runs)
            print(json.dumps(value, indent=2, ensure_ascii=False) if args.json else markdown(value), end="\n")
        elif args.command == "stop":
            stop(args.run_root.expanduser().resolve())
            print("Stop requested; the runner will preserve logs and close the run.")
        return 0
    except (BenchError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
