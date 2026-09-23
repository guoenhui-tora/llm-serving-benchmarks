#!/usr/bin/env python3
"""Postprocess a FINISHED study once; never read logs or alter run artifacts.

Usage: python3 experiments/dsv4-prefill-ep-20260923/reports/analysis.py
"""
import argparse
import csv
import json
import math
import re
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
CONCURRENCIES = (8, 16, 24, 32, 40, 48, 56, 64)
ENGINE_FIELDS = {
    'vllm:num_requests_running': 'running',
    'vllm:num_requests_waiting': 'waiting',
    'vllm:kv_cache_usage_perc': 'kv_usage_fraction',
    'vllm:gpu_cache_usage_perc': 'legacy_gpu_cache_usage_fraction',
}


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def read_jsonl(path):
    if not path.exists():
        return [], ['missing: ' + str(path)]
    rows, errors = [], []
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row.get('unix_s'), (float, int)):
                    raise ValueError('missing unix_s')
                rows.append(row)
            except (ValueError, TypeError) as exc:
                errors.append(f'{path.name}:{line_no}: {exc}')
    return sorted(rows, key=lambda r: r['unix_s']), errors


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def label(key, name):
    match = re.search(r'(?:\{|,)' + re.escape(name) + r'="([^"]*)"', key)
    return match.group(1) if match else None


def weighted(rows, start, end, values, max_gap_s=15):
    """Left-hold sampled gauges, clipped to window and 15 s max sample age.

    Error/missing samples terminate a prior value. Means divide by covered
    seconds, not full window; coverage is reported explicitly per field.
    """
    totals, durations, maxima, samples = {}, {}, {}, {}
    for i, row in enumerate(rows):
        t = row['unix_s']
        following = rows[i + 1]['unix_s'] if i + 1 < len(rows) else end
        left, right = max(start, t), min(end, following, t + max_gap_s)
        if t >= end:
            break
        if right <= left:
            continue
        for key, value in values(row).items():
            if not finite(value):
                continue
            duration = right - left
            totals[key] = totals.get(key, 0) + value * duration
            durations[key] = durations.get(key, 0) + duration
            maxima[key] = max(maxima.get(key, value), value)
            samples[key] = samples.get(key, 0) + 1
    return {key: {'time_weighted_mean': totals[key] / durations[key],
                  'sampled_max': maxima[key], 'covered_seconds': durations[key],
                  'window_coverage_fraction': durations[key] / (end - start),
                  'contributing_samples': samples[key]}
            for key in sorted(totals)}


def engine_values(row):
    result = {}
    for key, value in row.get('values', {}).items():
        name = key.split('{', 1)[0]
        field = ENGINE_FIELDS.get(name)
        if name == 'vllm:num_requests_waiting_by_reason':
            field = 'waiting_by_reason.' + (label(key, 'reason') or 'UNLABELED')
        if field:
            engine = label(key, 'engine')
            identity = f'engine_{engine if engine is not None else "UNLABELED"}.{field}'
            if identity in result:
                raise ValueError('Duplicate gauge identity; inspect metric labels: ' + identity)
            result[identity] = value
    return result


def gpu_values(row, wanted):
    result = {}
    gpu = row.get('gpu', {})
    if gpu.get('returncode') != 0:
        return result
    for columns in csv.reader(gpu.get('stdout', '').splitlines(), skipinitialspace=True):
        if len(columns) != 8:
            continue
        try:
            index = int(columns[0])
        except ValueError:
            continue
        if index not in wanted:
            continue
        for pos, name in ((2, 'utilization_percent'), (3, 'memory_used_mib'),
                          (4, 'power_w'), (5, 'temperature_c'),
                          (6, 'sm_clock_mhz'), (7, 'memory_clock_mhz')):
            try:
                result[f'gpu_{index}.{name}'] = float(columns[pos])
            except ValueError:
                pass
    return result


def cpu_windows(rows, start, end, wanted):
    """CPU tick deltas represent preceding intervals, unlike sampled GPU gauges."""
    busy_s = duration_s = 0.0
    peak = None
    for previous, current in zip(rows, rows[1:]):
        left, right = max(start, previous['unix_s']), min(end, current['unix_s'])
        if right <= left or current['unix_s'] - previous['unix_s'] > 15:
            continue
        deltas = []
        for cpu in wanted:
            before = previous.get('cpu_ticks', {}).get(cpu)
            after = current.get('cpu_ticks', {}).get(cpu)
            if before is None or after is None:
                break
            deltas.append([b - a for a, b in zip(before, after)])
        else:
            total = sum(sum(d) for d in deltas)
            idle = sum(d[3] + d[4] for d in deltas)
            if total <= 0 or any(v < 0 for d in deltas for v in d):
                continue
            busy = 1 - idle / total
            busy_s += busy * (right - left)
            duration_s += right - left
            peak = busy if peak is None else max(peak, busy)
    return {'time_weighted_mean_busy_fraction': busy_s / duration_s if duration_s else None,
            'sampled_max_busy_fraction': peak, 'covered_seconds': duration_s,
            'window_coverage_fraction': duration_s / (end - start)}


def mode_summary(root, mode, resources):
    here = root / mode
    state = read_json(here / 'status.json', {'status': 'MISSING'})
    engines, errors = read_jsonl(here / 'engine-telemetry.jsonl')
    rows = []
    for concurrency in CONCURRENCIES:
        block = here / 'trials' / f'c{concurrency:02d}'
        protocol = read_json(block / 'protocol.json', {})
        rounds = protocol.get('rounds', [])
        selected = protocol.get('selected_rounds', [])
        record = rounds[0] if len(rounds) == 1 else {}
        measurement = block / record.get('path', 'measurement-01')
        metric_file = read_json(measurement / 'metrics.json', {})
        metric = metric_file.get('metrics', {})
        audit = read_json(measurement / 'workload-audit.json', {})
        window = read_json(measurement / 'benchmark-window.json', {})
        compilation = read_json(measurement / 'compilation.json', {})
        checks = read_json(measurement / 'window-checks.json', {})
        issues = []
        if protocol.get('status') != 'PASS': issues.append('protocol_not_PASS')
        if len(rounds) != 1 or selected != [1] or record.get('status') != 'COMPLETE':
            issues.append('not_exactly_one_selected_complete_round')
        if audit.get('status') != 'PASS': issues.append('audit_not_PASS')
        if metric_file.get('completed') != 256: issues.append('client_completed_not_256')
        if checks.get('forbidden') or not checks: issues.append('missing_or_forbidden_window_checks')
        if compilation.get('events') is None: issues.append('missing_compilation_check')
        if metric.get('total_input_tokens') != 256 * 16384 or metric.get('total_output_tokens') != 256:
            issues.append('unexpected_workload')
        start, end = window.get('start_unix'), window.get('end_unix')
        window_valid = finite(start) and finite(end) and end > start
        if not window_valid: issues.append('missing_or_invalid_benchmark_window')
        duration = metric.get('duration')
        if not finite(duration) or duration <= 0: issues.append('missing_or_invalid_duration')
        row = {'mode': mode, 'concurrency': concurrency,
               'group_status': state.get('status'),
               'protocol_status': protocol.get('status', 'NOT_RUN'),
               'measurement_status': record.get('status', 'NOT_RUN'),
               'formal_result': not issues, 'acceptance_issues': issues,
               'measurement_path': str(measurement.relative_to(root)),
               'duration_s': duration, 'request_per_s': metric.get('request_throughput'),
               'actual_input_tokens_per_s': audit.get('input_tokens', 0) / duration
                   if audit.get('input_tokens') is not None and finite(duration) and duration > 0 else None,
               'mean_ttft_s': metric.get('mean_ttft_ms') / 1000 if finite(metric.get('mean_ttft_ms')) else None,
               'p50_ttft_s': metric.get('median_ttft_ms') / 1000 if finite(metric.get('median_ttft_ms')) else None,
               'p95_ttft_s': metric.get('p95_ttft_ms') / 1000 if finite(metric.get('p95_ttft_ms')) else None,
               'mean_queue_s': audit.get('mean_queue_s'), 'mean_prefill_s': audit.get('mean_prefill_s'),
               'tpot_s': None, 'completed': audit.get('completed'),
               'input_tokens': audit.get('input_tokens'), 'output_tokens': audit.get('output_tokens'),
               'computed_prefill_tokens': audit.get('computed_prefill_tokens'),
               'prefix_hits': audit.get('prefix_hits'), 'preemptions': audit.get('preemptions'),
               'jit_event_count': len(compilation['events']) if 'events' in compilation else None,
               'jit_events': compilation.get('events'), 'warning_matches': checks.get('warnings'),
               'completed_by_engine': audit.get('completed_by_engine', {}),
               'benchmark_window': window,
               'engine_scrape_errors_in_window': None,
               'engine_telemetry': {}, 'gpu_telemetry': {}, 'cpu_telemetry': {}}
        if window_valid:
            row['engine_telemetry'] = weighted(engines, start, end, engine_values)
            row['engine_scrape_errors_in_window'] = sum('error' in r for r in engines if start <= r['unix_s'] <= end)
            gpu_ids = set(range(4, 8) if mode == 'on' else range(4))
            row['gpu_telemetry'] = weighted(resources, start, end, lambda r: gpu_values(r, gpu_ids))
            server_cpus = range(32, 48) if mode == 'on' else range(0, 16)
            client_cpus = range(48, 52) if mode == 'on' else range(16, 20)
            row['cpu_telemetry'] = {
                'host': cpu_windows(resources, start, end, ['cpu']),
                'server_cpuset': cpu_windows(resources, start, end, [f'cpu{x}' for x in server_cpus]),
                'client_cpuset': cpu_windows(resources, start, end, [f'cpu{x}' for x in client_cpus]),
                'server_smt_siblings': cpu_windows(resources, start, end, [f'cpu{x + 64}' for x in server_cpus]),
                'client_smt_siblings': cpu_windows(resources, start, end, [f'cpu{x + 64}' for x in client_cpus]),
            }
        rows.append(row)
    return {'original_state': state, 'telemetry_parse_errors': errors, 'cells': rows}


def write_csv(path, rows):
    keys = list(rows[0]) if rows else ['mode', 'concurrency', 'formal_result']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=WORKSPACE / 'results/scan-01')
    parser.add_argument('--output-dir', type=Path, default=WORKSPACE / 'reports/scan-01-analysis')
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not (root / 'FINISHED').is_file():
        raise SystemExit('Refusing analysis: controller FINISHED marker is absent.')
    controller = read_json(root / 'summary.json', {})
    if any(controller.get('groups', {}).get(mode, {}).get('status') not in ('COMPLETE', 'PARTIAL', 'FAIL')
           for mode in ('on', 'off')):
        raise SystemExit('Refusing analysis: both groups must have terminal states.')
    resources, resource_errors = read_jsonl(root / 'resource-telemetry.jsonl')
    groups = {mode: mode_summary(root, mode, resources) for mode in ('on', 'off')}
    rows = [row for mode in ('on', 'off') for row in groups[mode]['cells']]
    output = args.output_dir.resolve()
    if not output.is_relative_to(WORKSPACE / 'reports'):
        raise SystemExit('Output must remain inside this experiment reports directory.')
    output.mkdir(parents=True, exist_ok=False)
    summary = {'run_root': str(root), 'controller': controller, 'groups': groups,
               'formal_result_count': sum(row['formal_result'] for row in rows),
               'resource_parse_errors': resource_errors,
               'limitations': [
                   'Only formal_result=true rows are accepted measurements; original group/protocol states are retained.',
                   'One official full-window measurement per point; no repetition standard deviation or stability claim.',
                   'TTFT includes HTTP, queueing, prefill and first sampling/return; server prefill histogram is not pure GPU time.',
                   'TPOT is null for single-token responses; output tokens/s equals requests/s and is not decode capability.',
                   'Engine/GPU gauges use left-held samples clipped to the benchmark window, at most 15 s sample age; means divide by observed seconds.',
                   'Sampled maxima are not continuous peaks. Edge values may carry from the immediately preceding sample; per-field coverage is reported.',
                   'Resource recorder samples at approximately 5 s. CPU means use overlapping preceding tick-delta intervals; values include other tasks on those CPUs.',
                   'CPU cpusets and SMT mapping are the verified study bindings, not process-level utilization or proof of exclusive ownership.',
                   'JIT event windows include phase setup outside benchmark timing; no log rereading is done by this analyzer.',
                   'EP mode is confounded with GPU/CPU position; both services share host resources and run independent scans.',
               ]}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    write_csv(output / 'all-cells.csv', rows)
    write_csv(output / 'accepted-cells.csv', [row for row in rows if row['formal_result']])
    print(json.dumps({'output_dir': str(output), 'formal_results': summary['formal_result_count'],
                      'group_status': {mode: groups[mode]['original_state'].get('status') for mode in groups}}))


if __name__ == '__main__':
    main()
