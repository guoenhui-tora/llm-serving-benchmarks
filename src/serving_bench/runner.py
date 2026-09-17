from __future__ import annotations

import signal
import socket
import time
import uuid
from pathlib import Path

from . import __version__
from .checks import http, logs
from .clients import vllm_bench
from .common import BenchError, Logger, fingerprint, read_json, save_command, utcnow, write_json
from .executors import docker
from .locks import gpu_locks


def check_stop(directory: Path) -> None:
    for parent in (directory, *directory.parents):
        if (parent / "stop-requested.json").exists():
            raise KeyboardInterrupt
        if (parent / "run.json").exists():
            return


def phase(case, workload, concurrency, count, directory, server, owner, facts):
    if "replicas" in case:
        from .deployment import phase as replica_phase
        return replica_phase(case, workload, concurrency, count, directory, server, owner, facts)
    check_stop(directory)
    directory.mkdir(parents=True, exist_ok=False)
    name = f"sb-{owner}-client"
    argv = vllm_bench.command(case, workload, concurrency, count, directory, name, owner, facts["client_image"]["Id"])
    save_command(directory, argv)
    started = utcnow()
    phase_error = None
    try:
        docker.run_client(argv, directory / "client.log", workload["measurement"]["timeout_s"])
    except BaseException as exc:
        phase_error = exc
        raise
    finally:
        try:
            docker.binding_evidence(case["target"], "client", name, owner, directory)
        except Exception as exc:
            # Stop may already have removed the client. Preserve interruption or
            # the original load-generator error instead of replacing it.
            write_json(directory / "client-binding-error.json", {"error": str(exc)})
            if phase_error is None:
                raise
        finally:
            docker.remove_owned(name, owner)
    # These observations are outside the load generator's benchmark timer.
    docker.binding_evidence(case["target"], "server", server, owner, directory, live=True)
    normalized = vllm_bench.normalize(directory / "raw.json", count, workload)
    time.sleep(1)  # Allow asynchronous server logs to flush before reading this phase's window.
    window = docker.server_logs(server, since=started)
    (directory / "server-window.log").write_text(window)
    events = logs.compilation_events(case, window)
    write_json(directory / "compilation.json", {"started_at": started, "events": events,
                                               "interpretation": "No matches means no known compilation log events detected."})
    write_json(directory / "metrics.json", normalized)
    return normalized, events


def trial(case, workload, concurrency, repetition, directory, server, owner, facts, log):
    m = workload["measurement"]
    attempts = []
    warmups = []

    def warmup(batch):
        quiet = 0
        for index in range(1, m["max_warmup_rounds"] + 1):
            where = directory / f"warmup-{batch:02d}-{index:02d}"
            _, events = phase(case, workload, concurrency, max(m["warmup_requests"], concurrency), where, server, owner, facts)
            warmups.append({"path": where.name, "events": len(events)})
            quiet = 0 if events else quiet + 1
            log(f"{case['id']}/{workload['id']} C{concurrency}: warmup {index}, known compile events={len(events)}")
            if quiet >= m["min_warmup_rounds"]:
                return
        raise BenchError("Warmup did not reach the required consecutive quiet rounds")

    write_json(directory / "workload.json", workload)
    for attempt in range(1, m["max_attempts"] + 1):
        warmup(attempt)
        where = directory / f"measurement-{attempt:02d}"
        metrics, events = phase(case, workload, concurrency, workload["traffic"]["requests"], where, server, owner, facts)
        attempts.append({"path": where.name, "events": len(events), "accepted": not events})
        write_json(directory / "measurement-checks.json", {"warmups": warmups, "attempts": attempts})
        if not events:
            return {"workload": workload["id"], "workload_fingerprint": fingerprint(workload),
                    "concurrency": concurrency, "repetition": repetition,
                    "metrics": metrics, "measurement": str(where.relative_to(directory)),
                    "compilation_check": "NO_KNOWN_EVENTS", "purpose": workload["purpose"]}
        log(f"Discarded attempt {attempt}: known compilation events detected")
    raise BenchError("All measurement attempts contained known compilation events")


def run_case(case: dict, directory: Path, owner: str, log) -> dict:
    from . import deployment
    directory.mkdir(parents=True, exist_ok=False)
    state = {"id": case["id"], "status": "STARTING", "started_at": utcnow(), "trials": []}
    write_json(directory / "case.json", state)
    write_json(directory / "resolved.json", case)
    grouped = "replicas" in case
    members = case.get("replicas", [case])
    sessions = [(member, f"sb-{owner}-r{i}-server" if grouped else f"sb-{owner}-server",
                 directory / "replicas" / f"replica-{i}" if grouped else directory)
                for i, member in enumerate(members)]
    cleanup_errors = []
    started = set()
    recorder = None
    if grouped:
        from .telemetry import Recorder
        recorder = Recorder(directory, log)
        recorder.start()
    try:
        check_stop(directory)
        log(f"Preflight {case['id']}")
        facts = deployment.preflight(case)
        write_json(directory / "environment.json", facts)
        write_json(directory / "host.json", docker.environment_snapshot())
        from .common import capture
        state["replicas"] = []
        for i, (member, server, where) in enumerate(sessions):
            member_owner = f"{owner}-r{i}" if grouped else owner
            member_facts = facts["replicas"][i] if grouped else facts
            where.mkdir(parents=True, exist_ok=True)
            if grouped:
                write_json(where / "resolved.json", member)
                write_json(where / "environment.json", member_facts)
            docker.cache_directory(member, member_facts["server_image"]["Id"]).mkdir(parents=True, exist_ok=True)
            argv = docker.server_command(member, server, member_owner, member_facts["server_image"]["Id"])
            save_command(where, argv)
            check_stop(directory)
            log(f"Starting {case['id']}/{member['id']}: {member['runtime']['image']}")
            capture(argv, timeout=120)
            started.add(server)
            startup = http.wait_ready(member, server, log)
            docker.binding_evidence(member["target"], "server", server, member_owner, where, live=True)
            probes = http.probes(member, where / "probes")
            state["replicas"].append({"id": member["id"], "startup_seconds": startup, "probes": probes})
            if not grouped:
                state.update(startup_seconds=startup, probes=probes)
        state["status"] = "MEASURING"
        write_json(directory / "case.json", state)
        for workload in case["workloads"]:
            for concurrency in workload["traffic"]["concurrency"]:
                for repetition in range(1, workload["measurement"]["repetitions"] + 1):
                    trial_dir = directory / "trials" / workload["id"] / f"c{concurrency:04d}" / f"r{repetition:02d}"
                    result = trial(case, workload, concurrency, repetition, trial_dir,
                                   sessions if grouped else sessions[0][1], owner, facts, log)
                    result["path"] = str(trial_dir.relative_to(directory))
                    state["trials"].append(result)
                    write_json(directory / "case.json", state)
                    log(f"Accepted {case['id']}/{workload['id']} C{concurrency} repeat={repetition}")
        state["status"] = "PASS"
    except KeyboardInterrupt:
        state.update(status="INTERRUPTED", error="Interrupted by user or signal")
        raise
    except Exception as exc:
        state.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        log(f"Failed {case['id']}: {exc}")
    finally:
        if recorder:
            try:
                recorder.close()
            except Exception as exc:
                cleanup_errors.append(str(exc))
        checks_status, warnings = [], 0
        for i, (member, server, where) in enumerate(sessions):
            member_owner = f"{owner}-r{i}" if grouped else owner
            where.mkdir(parents=True, exist_ok=True)
            try:
                info = docker.container_info(server)
                if info is not None:
                    if (info.get("Config", {}).get("Labels") or {}).get(docker.OWNER_LABEL) != member_owner:
                        raise BenchError("Server ownership mismatch during evidence capture")
                    write_json(where / "server-inspect.json", info)
                    text = docker.server_logs(server)
                    (where / "server.log").write_text(text)
                    checks = logs.kernel_checks(member, text)
                    write_json(where / "kernel-checks.json", checks)
                    checks_status.append(checks["status"])
                    warnings += len(checks["warnings"])
                elif server in started:
                    raise BenchError("Server disappeared before evidence collection")
            except Exception as exc:
                cleanup_errors.append(f"{member['id']} evidence capture: {exc}")
            for container in (f"sb-{member_owner}-client", server):
                try:
                    docker.remove_owned(container, member_owner)
                except Exception as exc:
                    cleanup_errors.append(str(exc))
        if checks_status:
            state["kernel_checks"] = "FAIL" if "FAIL" in checks_status else ("UNVERIFIED" if "UNVERIFIED" in checks_status else "PASS")
            state["warning_count"] = warnings
            if state["kernel_checks"] == "FAIL" and state["status"] == "PASS":
                state.update(status="FAIL", error="Kernel log checks failed")
        if cleanup_errors:
            state["cleanup_errors"] = cleanup_errors
            if state["status"] != "INTERRUPTED":
                state["status"] = "FAIL"
        state["finished_at"] = utcnow()
        write_json(directory / "case.json", state)
    return state


def run(plan: dict, run_root: Path) -> dict:
    run_root = run_root.expanduser().resolve()
    if run_root.exists():
        raise BenchError(f"Run directory already exists: {run_root}")
    run_root.mkdir(parents=True)
    log = Logger(run_root / "run.log")
    owner = uuid.uuid4().hex[:16]
    package = Path(__file__).parent
    source_id = fingerprint({str(p.relative_to(package)): p.read_text() for p in sorted(package.rglob("*.py"))})
    state = {"schema_version": 1, "runner_version": __version__, "runner_source_fingerprint": source_id, "campaign": plan["campaign"],
             "owner": owner, "hostname": socket.gethostname(), "status": "RUNNING", "cases": [], "started_at": utcnow()}
    write_json(run_root / "plan.json", plan)
    write_json(run_root / "run.json", state)
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        with gpu_locks(sorted({gpu for case in plan["cases"] for gpu in case["target"]["gpus"]})):
            for case in plan["cases"]:
                check_stop(run_root)
                path = run_root / "cases" / case["id"]
                # Save the case path before starting so stop/report work during startup.
                item = {"id": case["id"], "path": str(path.relative_to(run_root)), "status": "RUNNING"}
                state["cases"].append(item)
                write_json(run_root / "run.json", state)
                result = run_case(case, path, owner, log)
                item["status"] = result["status"]
                write_json(run_root / "run.json", state)
                if result.get("cleanup_errors"):
                    raise BenchError("Cleanup/evidence failure; stopped campaign before starting another case")
        state["status"] = "PASS" if all(x["status"] == "PASS" for x in state["cases"]) else "FAIL"
    except KeyboardInterrupt:
        state["status"] = "INTERRUPTED"
    except Exception as exc:
        state.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        log(str(exc))
    finally:
        for item in state["cases"]:
            case_path = run_root / item["path"] / "case.json"
            if case_path.is_file():
                item["status"] = read_json(case_path)["status"]
        if (run_root / "stop-requested.json").exists():
            state["status"] = "INTERRUPTED"
        state["finished_at"] = utcnow()
        write_json(run_root / "run.json", state)
        signal.signal(signal.SIGTERM, previous)
    log(f"Campaign {state['status']}; results: {run_root}")
    return state


def stop(run_root: Path) -> None:
    state = read_json(run_root / "run.json")
    if state["hostname"] != socket.gethostname():
        raise BenchError("Stop must run on the original host")
    if state["status"] not in {"RUNNING"}:
        raise BenchError(f"Run is already terminal: {state['status']}")
    write_json(run_root / "stop-requested.json", {"at": utcnow()})
    plan = read_json(run_root / "plan.json")
    owners = {state["owner"]}
    for case in plan["cases"]:
        for i in range(len(case.get("replicas", []))):
            owners.add(f"{state['owner']}-r{i}")
    for owner in owners:
        for role in ("client", "server"):
            docker.remove_owned(f"sb-{owner}-{role}", owner)
