from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serving_bench.common import BenchError, read_json
from serving_bench.config import load_document
from serving_bench.protocols import collect, relative_ranges

ROOT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'block'
        self.workload = {'id': 'test', 'purpose': 'performance', 'traffic': {'requests': 128},
                         'measurement': {'protocol': 'jit_clean', 'repetitions': 3, 'max_rounds': 12,
                                         'timeout_s': 100, 'budget_s': 1000}}
        self.calls = []
        self.events = iter([])
        self.values = iter([])

    def measure(self, count, path, timeout):
        self.calls.append((count, path.name, timeout))
        metrics = {'output_throughput': 100, 'mean_ttft_ms': 20, 'mean_tpot_ms': 5}
        metrics.update(next(self.values, {}))
        return {'completed': count, 'failed': 0, 'metrics': metrics}, ['JIT'] * next(self.events, 0)

    def run_protocol(self):
        return collect(self.workload, 32, self.path, self.measure, lambda _: None, lambda _: None)

    def test_cumulative_reuses_service_and_keeps_earlier_samples(self):
        self.events = iter([1, 0, 1, 0, 0])
        trials, state = self.run_protocol()
        self.assertEqual(state['selected_rounds'], [2, 4, 5])
        self.assertEqual(state['status'], 'PASS')
        self.assertEqual(state['event_rounds'], 2)
        self.assertEqual(len(trials), 3)
        self.assertEqual([c[0] for c in self.calls], [128]*5)
        self.assertEqual(len(state['warmups']), 0)

    def test_quick_keeps_jit_samples_and_only_warms_once(self):
        self.workload['measurement'].update(protocol='quick', warmup_rounds=1, max_rounds=3)
        self.events = iter([1, 1, 0, 1])
        trials, state = self.run_protocol()
        self.assertEqual([c[0] for c in self.calls], [64, 128, 128, 128])
        self.assertEqual([t['compilation_check'] for t in trials], ['KNOWN_EVENTS', 'NO_KNOWN_EVENTS', 'KNOWN_EVENTS'])
        self.assertEqual(state['status'], 'PASS')

    def test_stable_slides_window_and_checks_latency_too(self):
        self.workload['measurement'].update(protocol='stable', stability_threshold=.02)
        self.values = iter([{'mean_ttft_ms': 30}, {}, {}, {}, {}])
        trials, state = self.run_protocol()
        self.assertEqual(state['selected_rounds'], [2, 3, 4])
        self.assertEqual([w['passed'] for w in state['windows']], [False, True])
        self.assertEqual(state['all_clean_statistics']['mean_ttft_ms']['n'], 4)

    def test_stable_jit_breaks_consecutive_window(self):
        self.workload['measurement'].update(protocol='stable', stability_threshold=.02)
        self.events = iter([0, 0, 1, 0, 0, 0])
        _, state = self.run_protocol()
        self.assertEqual(state['selected_rounds'], [4, 5, 6])
        self.assertEqual(state['clean_rounds'], 5)

    def test_budget_returns_partial_samples_without_retries(self):
        self.workload['measurement']['max_rounds'] = 3
        self.events = iter([0, 1, 0])
        trials, state = self.run_protocol()
        self.assertEqual(state['status'], 'PARTIAL')
        self.assertEqual(state['selected_rounds'], [1, 3])
        self.assertEqual([t['protocol_status'] for t in trials], ['PARTIAL']*2)

    def test_unstable_finishes_at_budget_and_retains_all_clean(self):
        self.workload['measurement'].update(protocol='stable', stability_threshold=.02, max_rounds=4)
        self.values = iter([{'output_throughput': x} for x in (100, 110, 100, 110)])
        trials, state = self.run_protocol()
        self.assertEqual(trials, [])
        self.assertEqual(state['stop_reason'], 'round_budget')
        self.assertEqual(state['all_clean_statistics']['output_throughput']['n'], 4)

    def test_runtime_failure_retains_prior_rounds(self):
        actual = self.measure
        def fail(*args):
            if len(self.calls) == 1:
                raise BenchError('wrong token count')
            return actual(*args)
        self.measure = fail
        with self.assertRaisesRegex(BenchError, 'wrong token'):
            self.run_protocol()
        state = read_json(self.path/'protocol.json')
        self.assertEqual(state['status'], 'FAIL')
        self.assertEqual(state['selected_rounds'], [1])
        self.assertEqual(state['rounds'][1]['status'], 'FAILED')

    def test_interrupt_is_saved(self):
        self.measure = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.run_protocol()
        self.assertEqual(read_json(self.path/'protocol.json')['status'], 'INTERRUPTED')

    def test_time_budget_caps_phase_timeout_and_preserves_completed_round(self):
        clock = [0.0]
        actual = self.measure
        self.workload['measurement']['budget_s'] = 5
        def advance(*args):
            value = actual(*args)
            clock[0] += 3
            return value
        self.measure = advance
        with patch('serving_bench.protocols.time.monotonic', side_effect=lambda: clock[0]):
            trials, state = self.run_protocol()
        self.assertEqual([c[2] for c in self.calls], [5, 2])
        self.assertEqual(state['stop_reason'], 'time_budget')
        self.assertEqual(state['selected_rounds'], [1])
        self.assertTrue(state['rounds'][1]['over_budget'])

    def test_budget_timeout_stops_case_but_runtime_fault_is_not_hidden(self):
        import subprocess
        for timeout in (True, False):
            with self.subTest(timeout=timeout):
                self.path = Path(self.temp.name) / str(timeout)
                clock = [0.0]
                self.workload['measurement']['budget_s'] = 5
                def fail(*args):
                    clock[0] = 6
                    if timeout:
                        raise BenchError('client timed out') from subprocess.TimeoutExpired('client', 5)
                    raise BenchError('OOM')
                self.measure = fail
                with patch('serving_bench.protocols.time.monotonic', side_effect=lambda: clock[0]):
                    if timeout:
                        _, state = self.run_protocol()
                        self.assertTrue(state['aborted_phase'])
                        self.assertEqual(state['status'], 'PARTIAL')
                    else:
                        with self.assertRaisesRegex(BenchError, 'OOM'):
                            self.run_protocol()
                        self.assertEqual(read_json(self.path/'protocol.json')['status'], 'FAIL')

    def test_stability_formula_and_missing_metric(self):
        rows = [{'metrics': {'metrics': dict(output_throughput=x, mean_ttft_ms=20, mean_tpot_ms=5)}} for x in (99,100,101)]
        self.assertAlmostEqual(relative_ranges(rows)['output_throughput'], .02)
        rows[0]['metrics']['metrics']['mean_ttft_ms'] = None
        with self.assertRaisesRegex(BenchError, 'mean_ttft_ms'):
            relative_ranges(rows)

    def test_config_validation_and_defaults(self):
        import yaml
        source = yaml.safe_load((ROOT/'tests/fixtures/configs/workloads/smoke-128-32.yaml').read_text())
        dest = Path(self.temp.name)/'workload.yaml'
        for mode in ('quick','jit_clean','stable'):
            source['measurement'] = {'protocol': mode, 'budget_s': 3600}
            dest.write_text(yaml.safe_dump(source))
            m = load_document(dest, 'workload')['measurement']
            self.assertEqual(m['repetitions'], 3)
            self.assertEqual(m['max_rounds'], 3 if mode=='quick' else 12)
        for m in ({'protocol':'jit_clean'}, {'protocol':'bad','budget_s':10},
                  {'protocol':'jit_clean','budget_s':10,'max_attempts':2},
                  {'protocol':'stable','budget_s':10,'stability_threshold':float('nan')},
                  {'protocol':'stable','budget_s':10,'stability_threshold':2},
                  {'protocol':'stable','budget_s':10,'stability_threshold':0},
                  {'protocol':'jit_clean','budget_s':10,'max_rounds':2},
                  {'protocol':'quick','budget_s':10,'max_rounds':12},
                  {'protocol':'quick','budget_s':10,'warmup_rounds':3}):
            with self.subTest(measurement=m):
                source['measurement']=m;dest.write_text(yaml.safe_dump(source))
                with self.assertRaises(BenchError): load_document(dest,'workload')


if __name__ == '__main__': unittest.main()
