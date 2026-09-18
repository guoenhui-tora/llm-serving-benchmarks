"""Bounded measurement protocols; server lifecycle remains owned by runner."""
from __future__ import annotations

import math
import statistics
import subprocess
import time
from pathlib import Path

from .common import BenchError, fingerprint, utcnow, write_json

VERSION = 1
STABILITY_METRICS = ("output_throughput", "mean_ttft_ms", "mean_tpot_ms")


def summary(rows):
    result = {}
    rows = [r for r in rows if "metrics" in r]
    fields = {k for row in rows for k in row["metrics"]["metrics"]}
    for key in sorted(fields):
        values = [r["metrics"]["metrics"][key] for r in rows
                  if r["metrics"]["metrics"].get(key) is not None]
        result[key] = {"n": len(values), "mean": statistics.fmean(values) if values else None,
                       "sd": statistics.stdev(values) if len(values) > 1 else None}
    return result


def relative_ranges(rows):
    result = {}
    for key in STABILITY_METRICS:
        values = [r["metrics"]["metrics"].get(key) for r in rows]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise BenchError(f"Stability requires finite positive {key} in every round")
        result[key] = (max(values) - min(values)) / statistics.fmean(values)
    return result


def collect(workload, concurrency, directory: Path, measure, log, check_stop):
    """measure(count, directory, timeout_s) runs one full client phase.

    All decisions and original round identities survive faults/interruption.
    The budget covers this workload/concurrency, not model startup or cleanup.
    """
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "workload.json", workload)
    m = workload["measurement"]
    mode, needed = m["protocol"], m["repetitions"]
    started = time.monotonic()
    deadline = started + m["budget_s"]
    state = {"version": VERSION, "protocol": mode, "workload": workload["id"],
             "concurrency": concurrency, "status": "RUNNING", "started_at": utcnow(),
             "budget_s": m["budget_s"], "max_rounds": m["max_rounds"],
             "warmups": [], "rounds": [], "selected_rounds": [], "windows": []}
    clean, selected, consecutive = [], [], []

    def save():
        state["elapsed_s"] = time.monotonic() - started
        state["event_rounds"] = sum(bool(r.get("events", 0)) for r in state["rounds"])
        state["clean_rounds"] = len(clean)
        state["all_clean_statistics"] = summary(clean)
        state["all_round_statistics"] = summary(state["rounds"])
        state["selected_statistics"] = summary(selected)
        state["selected_rounds"] = [r["round"] for r in selected]
        write_json(directory / "protocol.json", state)

    def execute(count, name, records):
        check_stop(directory)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            state.update(status="PARTIAL", stop_reason="time_budget")
            return None
        record = {"path": name, "started_at": utcnow(), "status": "RUNNING"}
        records.append(record)
        # Save before launching so a failed or interrupted round is identifiable.
        write_json(directory / "protocol.json", state)
        try:
            metrics, events = measure(count, directory / name, min(m["timeout_s"], remaining))
        except KeyboardInterrupt:
            record.update(status="INTERRUPTED", finished_at=utcnow())
            raise
        except Exception as exc:
            record.update(status="FAILED", error=str(exc), finished_at=utcnow())
            # Only an actual client timeout may become budget exhaustion.
            # An OOM/token/binding failure must never be hidden by the clock.
            if isinstance(exc.__cause__, subprocess.TimeoutExpired) and time.monotonic() >= deadline:
                state.update(status="PARTIAL", stop_reason="time_budget", aborted_phase=True)
                return None
            raise
        record.update(status="COMPLETE", finished_at=utcnow(), metrics=metrics, events=len(events))
        if time.monotonic() > deadline:
            record["over_budget"] = True
            state.update(status="PARTIAL", stop_reason="time_budget")
        return record

    save()
    try:
        if mode == "quick":
            for i in range(m["warmup_rounds"]):
                record = execute(2 * concurrency, f"warmup-{i+1:02d}", state["warmups"])
                log(f"{workload['id']} C{concurrency}: quick warmup={i+1}, "
                    f"events={record.get('events', 'unknown') if record else 'unknown'}")
                save()
                if state["status"] == "PARTIAL":
                    break
        if state["status"] != "PARTIAL":
            for i in range(1, m["max_rounds"] + 1):
                record = execute(workload["traffic"]["requests"], f"measurement-{i:02d}", state["rounds"])
                if record is None:
                    break
                record.update(round=i, eligible=not record["events"] and not record.get("over_budget", False))
                if record["eligible"]:
                    clean.append(record)
                    consecutive.append(record)
                else:
                    consecutive.clear()
                if not record.get("over_budget"):
                    if mode == "quick":
                        selected.append(record)
                    elif mode == "jit_clean":
                        selected = list(clean)
                    elif len(consecutive) >= needed:
                        window = consecutive[-needed:]
                        ranges = relative_ranges(window)
                        passed = all(v <= m["stability_threshold"] + 1e-12 for v in ranges.values())
                        state["windows"].append({"rounds": [r["round"] for r in window],
                                                 "relative_ranges": ranges, "passed": passed})
                        if passed:
                            selected = list(window)
                log(f"{workload['id']} C{concurrency}: {mode} round={i}, events={record['events']}, "
                    f"clean={len(clean)}, selected={len(selected)}/{needed}")
                if len(selected) == needed:
                    state.update(status="PASS", stop_reason="target_reached")
                save()
                if state["status"] != "RUNNING":
                    break
        if state["status"] == "RUNNING":
            state.update(status="PARTIAL", stop_reason="round_budget")
    except KeyboardInterrupt:
        state.update(status="INTERRUPTED", stop_reason="interrupted")
        raise
    except Exception as exc:
        state.update(status="FAIL", stop_reason="runtime_error", error=str(exc))
        raise
    finally:
        # Failed rounds may have no metrics; retain them without aggregating.
        state["finished_at"] = utcnow()
        save()

    log(f"{workload['id']} C{concurrency}: {mode} {state['status']}, "
        f"reason={state['stop_reason']}, elapsed={state['elapsed_s']:.1f}s")
    trials = []
    for n, row in enumerate(selected, 1):
        trials.append({"workload": workload["id"], "workload_fingerprint": fingerprint(workload),
                       "concurrency": concurrency, "repetition": n, "round": row["round"],
                       "metrics": row["metrics"], "measurement": row["path"],
                       "compilation_check": "KNOWN_EVENTS" if row["events"] else "NO_KNOWN_EVENTS",
                       "event_count": row["events"], "protocol": mode, "protocol_version": VERSION,
                       "protocol_status": state["status"], "purpose": workload["purpose"]})
    return trials, state
