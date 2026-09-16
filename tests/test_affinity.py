from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serving_bench.common import BenchError
from serving_bench.config import resolve, validate_document
from serving_bench.executors import affinity, docker
from serving_bench.clients import vllm_bench

ROOT = Path(__file__).resolve().parents[1]


class AffinityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = resolve(ROOT / 'tests/fixtures/configs/campaigns/46-glm52-vllm-smoke.yaml')['cases'][0]
        self.binding = {'server': {'cpus': '0-1', 'mems': '0'},
                        'client': {'cpus': '4-5', 'mems': '1'}}
        self.case['target']['binding'] = copy.deepcopy(self.binding)

    def test_schema_rejects_malformed_or_overlapping_bindings(self):
        validate_document(self.case['target'], 'target')
        invalid = [{}, {'worker': {}}, {'server': {'cpus': '0-1'}},
                   {'server': {'cpus': 1, 'mems': '0'}},
                   {'server': {'cpus': '2-1', 'mems': '0'}},
                   {'server': {'cpus': '0-2,2', 'mems': '0'}},
                   {'server': {'cpus': '0;echo x', 'mems': '0'}},
                   {'server': {'cpus': '0', 'mems': ''}},
                   {'server': {'cpus': '0-1', 'mems': '0'}, 'client': {'cpus': '1-2', 'mems': '1'}}]
        for binding in invalid:
            with self.subTest(binding=binding), self.assertRaises(BenchError):
                target = copy.deepcopy(self.case['target'])
                target['binding'] = binding
                validate_document(target, 'target')

    def test_role_specific_docker_flags_and_old_command_compatibility(self):
        case = self.case
        server = docker.server_command(case, 'server', 'owner')
        client = vllm_bench.command(case, case['workloads'][0], 1, 2, self.root, 'client', 'owner')
        self.assertEqual(server[server.index('--cpuset-cpus') + 1], '0-1')
        self.assertEqual(client[client.index('--cpuset-cpus') + 1], '4-5')
        self.assertEqual(client[client.index('--cpuset-mems') + 1], '1')
        self.assertNotIn('--rm', client)
        self.assertIn('record_affinity', client[client.index('-c') + 1])
        del case['target']['binding']
        self.assertNotIn('--cpuset-cpus', docker.server_command(case, 'server', 'owner'))
        legacy = vllm_bench.command(case, case['workloads'][0], 1, 2, self.root, 'client', 'owner')
        self.assertIn('--rm', legacy)
        self.assertEqual(legacy[legacy.index('-c') + 1], vllm_bench.PREFIX[1])

    def host_tree(self):
        cpu = self.root / 'cpu'
        cpu.mkdir()
        (cpu / 'online').write_text('0-5')
        node = self.root / 'node'
        node.mkdir()
        (node / 'has_memory').write_text('0-1')
        for index in range(6):
            topology = cpu / f'cpu{index}/topology'
            topology.mkdir(parents=True)
            (topology / 'physical_package_id').write_text('0')
            (topology / 'core_id').write_text(str(index % 4))

    def test_host_check_rejects_smt_overlap_even_with_different_cpu_ids(self):
        self.host_tree()
        with self.assertRaisesRegex(BenchError, 'physical cores'):
            affinity.host_check(self.case['target'], self.root)
        self.case['target']['binding']['client']['cpus'] = '2-3'
        facts = affinity.host_check(self.case['target'], self.root)
        self.assertEqual(len(facts['physical_cores']['server']), 2)
        for key, val in [('cpus', '6'), ('mems', '2')]:
            target = copy.deepcopy(self.case['target'])
            target['binding']['client'][key] = val
            with self.subTest(key=key), self.assertRaisesRegex(BenchError, 'offline/absent'):
                affinity.host_check(target, self.root)
        self.assertEqual(affinity.host_check({}, self.root / 'absent'), {})

    def test_snapshot_verification_allows_narrowing_but_rejects_escape(self):
        observed = {'threads': [{'tid': 2, 'cpus': '1', 'mems': '0'}], 'unreadable': []}
        affinity.verify_snapshot(self.binding['server'], observed)
        for key, value in [('cpus', '0-2'), ('mems', '1')]:
            bad = copy.deepcopy(observed)
            bad['threads'][0][key] = value
            with self.assertRaisesRegex(ValueError, 'escapes'):
                affinity.verify_snapshot(self.binding['server'], bad)
        with self.assertRaisesRegex(ValueError, 'Cannot verify'):
            affinity.verify_snapshot(self.binding['server'], {'threads': [], 'unreadable': []})

    def test_docker_and_live_evidence_are_saved_before_rejecting_mismatch(self):
        info = {'Config': {'Labels': {docker.OWNER_LABEL: 'owner'}},
                'HostConfig': {'CpusetCpus': '0,1', 'CpusetMems': '0'}}
        observed = {'threads': [{'tid': 2, 'cpus': '2', 'mems': '0'}], 'unreadable': []}
        result = subprocess.CompletedProcess([], 0, json.dumps(observed), '')
        with patch.object(docker, 'container_info', return_value=info), patch.object(docker, 'capture', return_value=result):
            with self.assertRaisesRegex(BenchError, 'escapes'):
                docker.binding_evidence(self.case['target'], 'server', 'server', 'owner', self.root, live=True)
        self.assertEqual(json.loads((self.root / 'server-affinity.json').read_text()), observed)
        self.assertTrue((self.root / 'server-binding-inspect.json').is_file())
        info['HostConfig']['CpusetMems'] = '0-1'
        with self.assertRaisesRegex(BenchError, 'CpusetMems'):
            affinity.verify_inspect(self.binding['server'], info)
        with patch.object(docker, 'container_info', return_value=info), patch.object(docker, 'capture') as execute:
            with self.assertRaisesRegex(BenchError, 'owned'):
                docker.binding_evidence(self.case['target'], 'server', 'server', 'other-owner', self.root, live=True)
            execute.assert_not_called()

    def fake_proc(self, cpus='4-5'):
        proc = self.root / 'proc'
        task = proc / '1/task/1'
        task.mkdir(parents=True)
        (task / 'status').write_text(f'Name:\tclient\nCpus_allowed_list:\t{cpus}\nMems_allowed_list:\t1\n')
        return proc

    def run_wrapper(self, cpus, body):
        proc = self.fake_proc(cpus)
        # Run the real generated Python, with a controlled /proc and output root.
        code = affinity.client_prefix(['-c', body], self.binding['client'])[1]
        code = code.replace('Path("/proc")', f'Path({str(proc)!r})')
        code = code.replace("'/results/client-affinity-'", repr(str(self.root / 'client-affinity-')))
        return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)

    def test_client_wrapper_preserves_execution_and_records_both_snapshots(self):
        result = self.run_wrapper('4-5', "print('benchmark invoked')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('benchmark invoked', result.stdout)
        for stage in ['start', 'end']:
            self.assertTrue((self.root / f'client-affinity-{stage}.json').is_file())

    def test_client_mismatch_stops_before_benchmark_and_keeps_evidence(self):
        result = self.run_wrapper('4-6', "print('benchmark invoked')")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('benchmark invoked', result.stdout)
        self.assertIn('escapes', result.stderr)
        self.assertTrue((self.root / 'client-affinity-start.json').is_file())

    def test_client_failure_still_records_final_affinity(self):
        result = self.run_wrapper('4-5', "raise RuntimeError('benchmark failed')")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('benchmark failed', result.stderr)
        self.assertTrue((self.root / 'client-affinity-end.json').is_file())
