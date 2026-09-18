from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from serving_bench.clients import jsonl_dataset, vllm_bench
from serving_bench.common import BenchError, fingerprint, write_json
from serving_bench.config import resolve
from serving_bench import deployment

ROOT = Path(__file__).resolve().parents[1]


class JsonlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / 'tests/fixtures/configs', self.root / 'configs')
        (self.root / 'datasets').mkdir()
        self.path = self.root / 'datasets/prompts.jsonl'
        self.rows = [{'id': str(i), 'prompt': 'word ' * (i + 1), 'input_tokens': i + 1,
                      'output_tokens': 4} for i in range(4)]
        self.write_rows()
        self.workload_path = self.root / 'configs/workloads/smoke-128-32.yaml'
        self.workload = yaml.safe_load(self.workload_path.read_text())
        self.workload['dataset'] = dict(name='jsonl', path='datasets/prompts.jsonl',
                                        max_input_tokens=128, output_tokens=4)
        self.workload_path.write_text(yaml.safe_dump(self.workload))
        self.campaign = self.root / 'configs/campaigns/46-glm52-vllm-smoke.yaml'

    def write_rows(self):
        self.path.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))

    def case(self):
        return resolve(self.campaign)['cases'][0]

    def test_dataset_identity_changes_with_content_but_not_project_location(self):
        first = self.case()
        relocated = self.root / 'relocated'
        shutil.copytree(self.root / 'configs', relocated / 'configs')
        shutil.copytree(self.root / 'datasets', relocated / 'datasets')
        second = resolve(relocated / 'configs/campaigns' / self.campaign.name)['cases'][0]
        self.assertEqual(first['workloads'], second['workloads'])
        self.assertNotEqual(first['dataset_paths'], second['dataset_paths'])
        self.rows[0]['prompt'] = 'changed'
        self.write_rows()
        self.assertNotEqual(fingerprint(first['workloads']), fingerprint(self.case()['workloads']))

    def test_pin_count_schema_and_escape_fail_before_launch(self):
        for change, pattern in [
            ({'sha256': '0' * 64}, 'SHA256'),
            ({'range_ratio': 0}, 'unknown'),
            ({'path': '../outside.jsonl'}, 'relative'),
            ({'max_input_tokens': 4096}, 'context length'),
        ]:
            w = copy.deepcopy(self.workload)
            w['dataset'].update(change)
            self.workload_path.write_text(yaml.safe_dump(w))
            with self.subTest(change=change), self.assertRaisesRegex(BenchError, pattern):
                self.case()
        w = copy.deepcopy(self.workload)
        w['traffic']['requests'] = 5
        self.workload_path.write_text(yaml.safe_dump(w))
        with self.assertRaisesRegex(BenchError, 'needs 5 rows'):
            self.case()
        w['traffic'].update(requests=4, concurrency=[4])
        w['measurement'] = dict(protocol='quick', budget_s=100)
        self.workload_path.write_text(yaml.safe_dump(w))
        with self.assertRaisesRegex(BenchError, 'needs 8 rows'):
            self.case()

    def test_symlink_escape_rejected(self):
        outside = self.root.parent / (self.root.name + '-outside.jsonl')
        outside.write_text(self.path.read_text())
        self.addCleanup(outside.unlink)
        self.path.unlink()
        self.path.symlink_to(outside)
        with self.assertRaisesRegex(BenchError, 'escapes project'):
            self.case()

    def test_bad_records_rejected(self):
        for mutate in [lambda r: r[1].update(id='0'), lambda r: r[0].update(input_tokens=True),
                       lambda r: r[0].update(output_tokens=5), lambda r: r[0].update(prompt=''),
                       lambda r: r[0].update(extra='unknown')]:
            rows = copy.deepcopy(self.rows)
            mutate(rows)
            self.path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaises(BenchError):
                self.case()

    def sample(self, case, count=4, index=0, replicas=1, modify_sample=None):
        # Stand in for the pinned official sampler; the real image is checked separately.
        class CustomDataset:
            pass

        def original(args, tokenizer):
            ds = CustomDataset()
            ds.load_data()
            requests = [SimpleNamespace(prompt=r['prompt'], prompt_len=len(tokenizer(r['prompt']).input_ids),
                                        expected_output_len=args.custom_output_len) for r in ds.data[:args.num_prompts]]
            if modify_sample:
                modify_sample(requests)
            return requests

        tokenizer = lambda prompt: SimpleNamespace(input_ids=prompt.split())
        serve = SimpleNamespace(get_samples=original)
        output = self.root / f'out-{index}'
        output.mkdir(exist_ok=True)
        actual_read = jsonl_dataset.read_records
        with patch.dict(sys.modules, {'vllm.benchmarks.datasets': SimpleNamespace(CustomDataset=CustomDataset)}), \
                patch.object(jsonl_dataset, 'read_records', side_effect=lambda _, *a: actual_read(self.path, *a)):
            jsonl_dataset.install(serve, case['workloads'][0]['dataset'], output, index, replicas)
        args = SimpleNamespace(dataset_name='custom', dataset_path='/dataset.jsonl', skip_chat_template=True,
                               disable_shuffle=True, no_oversample=True, custom_output_len=4, num_prompts=count)
        requests = serve.get_samples(args, tokenizer)
        return requests, json.loads((output / 'dataset-manifest.json').read_text()), output

    def test_raw_prompts_real_lengths_and_interleaved_manifests(self):
        case = self.case()
        selected = []
        for i in range(2):
            requests, manifest, output = self.sample(case, index=i, replicas=2)
            self.assertEqual([r.prompt for r in requests], [r['prompt'] for r in self.rows])
            self.assertEqual([r['index'] for r in manifest['requests']], list(range(i, 4, 2)))
            jsonl_dataset.validated_manifest(output / 'dataset-manifest.json', case['workloads'][0]['dataset'], 2, (i, 2, 4))
            selected.extend(manifest['requests'])
        self.assertEqual(len({r['id'] for r in selected}), 4)
        self.assertEqual(sum(r['input_tokens'] for r in selected), 10)

    def test_token_metadata_lies_and_actual_oversize_rejected(self):
        self.rows[0]['input_tokens'] = 2
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'metadata differs'):
            self.sample(self.case())
        del self.rows[0]['input_tokens']
        self.rows[0]['prompt'] = 'word ' * 129
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'actual input exceeds'):
            self.sample(self.case())

    def test_mutation_after_resolve_and_sampler_changes_rejected(self):
        case = self.case()
        self.rows[0]['prompt'] = 'changed'
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.sample(case)
        with self.assertRaisesRegex(ValueError, 'changed prompt'):
            self.sample(self.case(), modify_sample=lambda reqs: reqs.reverse())
        with self.assertRaisesRegex(ValueError, 'insufficient rows'):
            self.sample(self.case(), count=5)

    def test_normalize_variable_input_and_fixed_or_natural_output(self):
        case = self.case()
        _, _, output = self.sample(case, count=2)
        raw = dict(completed=2, failed=0, output_throughput=10, mean_ttft_ms=1,
                   mean_tpot_ms=1, total_input_tokens=3, total_output_tokens=8)
        path = output / 'raw.json'
        w = case['workloads'][0]
        write_json(path, raw)
        vllm_bench.normalize(path, 2, w, (0, 1, 2))
        for key, value, pattern in [('total_input_tokens', 4, 'JSONL input mismatch'),
                                    ('total_output_tokens', 7, 'output mismatch')]:
            write_json(path, {**raw, key: value})
            with self.assertRaisesRegex(BenchError, pattern):
                vllm_bench.normalize(path, 2, w)
        w['sampling']['ignore_eos'] = False
        vllm_bench.normalize(path, 2, w)
        with self.assertRaisesRegex(BenchError, 'scheduled partition'):
            vllm_bench.normalize(path, 2, w, (1, 2, 4))
        (output / 'dataset-manifest.json').unlink()
        with self.assertRaisesRegex(BenchError, 'manifest'):
            vllm_bench.normalize(path, 2, w)

    def test_replica_aggregate_uses_variable_lengths_and_rejects_duplicate_partition(self):
        case = self.case()
        directories = []
        for i in range(2):
            _, manifest, output = self.sample(case, index=i, replicas=2)
            directories.append(output)
            write_json(output / 'benchmark-window.json', dict(start_s=100, end_s=102))
            write_json(output / 'metrics.json', dict(completed=2, metrics=dict(duration=2,
                total_input_tokens=manifest['total_input_tokens'], total_output_tokens=8, output_throughput=4)))
            rows = [dict(success=True, input_tokens=r['input_tokens'], output_tokens=4,
                         ttft_s=.1, latency_s=.4, itl_s=[.1]*3) for r in manifest['requests']]
            write_json(output / 'requests.json', rows)
        result, _ = deployment.aggregate(directories, 4, case['workloads'][0])
        self.assertEqual(result['metrics']['total_input_tokens'], 10)
        self.assertEqual(result['metrics']['total_output_tokens'], 16)
        shutil.copy(directories[0] / 'dataset-manifest.json', directories[1] / 'dataset-manifest.json')
        with self.assertRaisesRegex(BenchError, 'scheduled partition'):
            deployment.aggregate(directories, 4, case['workloads'][0])

    def test_command_keeps_official_client_and_combines_hooks(self):
        case = self.case()
        case['_client_sync'] = dict(directory='/coord', index=1, replicas=2, global_requests=4, timeout_s=30)
        args = vllm_bench.command(case, case['workloads'][0], 1, 2, self.root, 'test', 'owner')
        code = args[args.index('-c') + 1]
        self.assertIn('/jsonl_dataset.py', code)
        self.assertIn('/synchronized.py', code)
        self.assertIn('main(p.parse_args())', code)
        self.assertNotIn('--random-input-len', args)
        self.assertIn('--skip-chat-template', args)
        self.assertIn(str(self.path) + ':/dataset.jsonl:ro', args)
