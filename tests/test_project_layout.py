from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serving_bench.common import BenchError
from serving_bench.config import resolve
from serving_bench.clients.vllm_bench import command as client_command
from serving_bench.executors.docker import cache_directory, server_command
from scripts.prepare_logging import prepare
from scripts.package import source_files

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / 'projects/dsv4-rtx6000d/configs'
TEMPLATE = ROOT / 'projects/_template/configs'


class ProjectLayoutTests(unittest.TestCase):
    def test_all_node_budgets_and_capacity(self):
        for name,expected in [('dual-tp4',1),('dual-candidates',2)]:
            plan=resolve(DS/f'campaigns/{name}.yaml')
            self.assertEqual(len(plan['cases']),expected)
            for case in plan['cases']:
                members=case['replicas'];o=case['recipe']['options']
                self.assertEqual(len(members)*o['data-parallel-size']*o['max-num-seqs'],64)
                self.assertEqual(sum(len(m['target']['gpus']) for m in members),8)
                self.assertEqual(o['kernel-config']['enable_flashinfer_autotune'],False)

    def test_all_project_campaigns_resolve_offline(self):
        with patch('subprocess.run', side_effect=AssertionError('Offline resolution ran a command')):
            paths = list((ROOT / 'projects').glob('*/configs/campaigns/*.yaml'))
            self.assertGreater(len(paths), 0)
            for path in paths:
                with self.subTest(campaign=str(path)):
                    plan = resolve(path)
                    for case in plan['cases']:
                        self.assertIn('--pull', server_command(case, 'preview', 'preview'))

    def test_workspace_configs_can_be_published_without_changing_execution(self):
        for project in sorted((ROOT / 'projects').iterdir()):
            if not (project / 'configs').is_dir():
                continue
            with self.subTest(project=project.name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                working = root / 'experiments' / project.name / 'configs'
                published = root / 'projects' / project.name / 'configs'
                shutil.copytree(project / 'configs', working)
                plans = {p.name: resolve(p) for p in (working / 'campaigns').glob('*.yaml')}
                published.parent.mkdir(parents=True)
                shutil.move(str(working), published)
                # The original workspace is absent: references must be self-contained.
                for name, before in plans.items():
                    after = resolve(published / 'campaigns' / name)
                    self.assertEqual(before, after)
                    for old, new in zip(before['cases'], after['cases']):
                        image_id = new['runtime']['image_id']
                        self.assertEqual(server_command(old, 'preview', 'preview', image_id),
                                         server_command(new, 'preview', 'preview', image_id))
                        self.assertEqual(cache_directory(old, image_id), cache_directory(new, image_id))
                        for workload in new['workloads']:
                            for concurrency in workload['traffic']['concurrency']:
                                args = (workload, concurrency, workload['traffic']['requests'],
                                        root / 'output', 'preview', 'preview')
                                self.assertEqual(client_command(old, *args), client_command(new, *args))
                        # Logging files must also work from the published copy.
                        isolated = copy.deepcopy(new)
                        isolated['target']['cache_root'] = str(root / 'cache')
                        destination = prepare(isolated, published)
                        if destination is not None:
                            self.assertTrue(destination.is_file())
                            self.assertEqual(prepare(isolated, published, check=True), destination)

    def test_logging_preparation_uses_actual_cache_and_preserves_existing_files(self):
        case = copy.deepcopy(resolve(TEMPLATE / 'campaigns/c32-comparison.yaml')['cases'][1])
        with tempfile.TemporaryDirectory() as directory:
            case['target']['cache_root'] = directory
            with self.assertRaisesRegex(BenchError, 'not prepared'):
                prepare(case, TEMPLATE, check=True)
            self.assertEqual(list(Path(directory).iterdir()), [])
            destination = prepare(case, TEMPLATE)
            argv = server_command(case, 'preview', 'preview', case['runtime']['image_id'])
            cache = cache_directory(case, case['runtime']['image_id'])
            self.assertIn(f'{cache}:/root/.cache:rw', argv)
            self.assertEqual(json.loads(destination.read_text()), json.loads((TEMPLATE / 'logging/sglang-jit-info.json').read_text()))
            before = destination.stat().st_mtime_ns
            self.assertEqual(prepare(case, TEMPLATE, check=True), destination)
            self.assertEqual(prepare(case, TEMPLATE), destination)
            self.assertEqual(destination.stat().st_mtime_ns, before)
            destination.write_text('{}')
            with self.assertRaisesRegex(BenchError, 'differs'):
                prepare(case, TEMPLATE)
            self.assertEqual(destination.read_text(), '{}')

    def test_logging_path_escape_rejected(self):
        case = copy.deepcopy(resolve(TEMPLATE / 'campaigns/c32-comparison.yaml')['cases'][1])
        case['recipe']['environment']['SGLANG_LOGGING_CONFIG_PATH'] = '/root/.cache/logging/../outside.json'
        with self.assertRaisesRegex(BenchError, 'Logging config must'):
            prepare(case, TEMPLATE)

    def test_package_keeps_curated_reports_but_excludes_local_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keep = ['README.md', 'AGENTS.md', '.github/workflows/tests.yml',
                    'src/serving_bench/results/report.py',
                    'projects/demo/reports/final.md', 'projects/demo/data/samples.json',
                    'projects/demo/configs/campaigns/run.yaml']
            drop = ['results/.gitkeep', 'reports/private.md', 'artifacts/backup.tar.gz', 'results/raw.json',
                    'experiments/demo/configs/campaigns/draft.yaml',
                    'experiments/demo/results/raw.json', 'experiments/demo/reports/draft.md',
                    'projects/demo/results/raw.json', 'projects/demo/.env',
                    'projects/demo/.env.private', 'src/serving_bench/__pycache__/file.pyc',
                    'projects/demo/cache/kernel.bin', 'unknown/secret.txt']
            for name in keep + drop:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('test')
            actual = {str(p.relative_to(root)) for p in source_files(root)}
            self.assertEqual(actual, set(keep))
