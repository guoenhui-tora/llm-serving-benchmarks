#!/usr/bin/env python3
"""Stage the reviewed local DSpark overlay for a resolved campaign; no service start."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / 'bench').is_file())
sys.path.insert(0, str(REPO / 'src'))
from serving_bench.config import resolve
from serving_bench.executors.docker import cache_directory

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('campaign', type=Path)
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
manifest = json.loads((HERE / 'manifest.json').read_text())
for name, digest in manifest['files'].items():
    if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f'Patch hash mismatch: {name}')
for case in resolve(args.campaign)['cases']:
    assert case['runtime']['image_id'] == 'sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1'
    assert case['recipe']['environment']['PYTHONPATH'] == '/root/.cache/dspark-native-mxfp4'
    destination = cache_directory(case, case['runtime']['image_id']) / 'dspark-native-mxfp4'
    for name in (*manifest['files'], 'manifest.json'):
        source = HERE / name
        target = destination / name
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise SystemExit(f'Existing overlay differs; preserve and review: {target}')
        elif args.check:
            raise SystemExit(f'Missing overlay: {target}')
        else:
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    print(f'{case["id"]}: {destination} verified')
