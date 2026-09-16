from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serving_bench.common import BenchError
from serving_bench.config import resolve
from serving_bench.executors.docker import cache_directory, server_command
from scripts.prepare_logging import prepare
from scripts.package import source_files

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / 'projects/dsv4-rtx6000d/configs'


class ProjectLayoutTests(unittest.TestCase):
    def test_all_project_campaigns_resolve_offline(self):
        with patch('subprocess.run', side_effect=AssertionError('Offline resolution ran a command')):
            paths = list((ROOT / 'projects').glob('*/configs/campaigns/*.yaml'))
            self.assertGreater(len(paths), 0)
            for path in paths:
                with self.subTest(campaign=str(path)):
                    plan = resolve(path)
                    for case in plan['cases']:
                        self.assertIn('--pull', server_command(case, 'preview', 'preview'))

    def test_logging_preparation_uses_actual_cache_and_preserves_existing_files(self):
        case = copy.deepcopy(resolve(DS / 'campaigns/48-dsv4-aligned-final-c16-c32.yaml')['cases'][1])
        with tempfile.TemporaryDirectory() as directory:
            case['target']['cache_root'] = directory
            with self.assertRaisesRegex(BenchError, 'not prepared'):
                prepare(case, DS, check=True)
            self.assertEqual(list(Path(directory).iterdir()), [])
            destination = prepare(case, DS)
            argv = server_command(case, 'preview', 'preview', case['runtime']['image_id'])
            cache = cache_directory(case, case['runtime']['image_id'])
            self.assertIn(f'{cache}:/root/.cache:rw', argv)
            self.assertEqual(json.loads(destination.read_text()), json.loads((DS / 'logging/sglang-jit-info.json').read_text()))
            before = destination.stat().st_mtime_ns
            self.assertEqual(prepare(case, DS, check=True), destination)
            self.assertEqual(prepare(case, DS), destination)
            self.assertEqual(destination.stat().st_mtime_ns, before)
            destination.write_text('{}')
            with self.assertRaisesRegex(BenchError, 'differs'):
                prepare(case, DS)
            self.assertEqual(destination.read_text(), '{}')

    def test_logging_path_escape_rejected(self):
        case = copy.deepcopy(resolve(DS / 'campaigns/48-dsv4-aligned-final-c16-c32.yaml')['cases'][1])
        case['recipe']['environment']['SGLANG_LOGGING_CONFIG_PATH'] = '/root/.cache/logging/../outside.json'
        with self.assertRaisesRegex(BenchError, 'Logging config must'):
            prepare(case, DS)

    def test_package_keeps_curated_reports_but_excludes_local_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keep = ['README.md', 'AGENTS.md', 'results/.gitkeep', 'src/serving_bench/results/report.py',
                    'projects/demo/reports/final.md', 'projects/demo/data/samples.json',
                    'projects/demo/configs/campaigns/run.yaml']
            drop = ['reports/private.md', 'artifacts/backup.tar.gz', 'results/raw.json',
                    'projects/demo/results/raw.json', 'projects/demo/.env',
                    'projects/demo/.env.private', 'src/serving_bench/__pycache__/file.pyc',
                    'projects/demo/cache/kernel.bin', 'unknown/secret.txt']
            for name in keep + drop:
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text('test')
            actual = {str(p.relative_to(root)) for p in source_files(root)}
            self.assertEqual(actual, set(keep))
