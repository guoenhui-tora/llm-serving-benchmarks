#!/usr/bin/env python3
"""Stage project-owned SGLang logging JSON without starting a service."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from serving_bench.common import BenchError
from serving_bench.config import resolve
from serving_bench.executors.docker import cache_directory


def prepare(case: dict, config_root: Path, *, check: bool = False) -> Path | None:
    env = {**case['target'].get('environment', {}),
           **case['runtime'].get('environment', {}),
           **case['recipe'].get('environment', {})}
    value = env.get('SGLANG_LOGGING_CONFIG_PATH')
    if not value:
        return None
    container_path = PurePosixPath(value)
    if container_path.parent != PurePosixPath('/root/.cache/logging') or container_path.suffix != '.json':
        raise BenchError('Logging config must be /root/.cache/logging/<name>.json')
    source_root = (config_root / 'logging').resolve()
    source = (source_root / container_path.name).resolve()
    if not source.is_relative_to(source_root) or not source.is_file():
        raise BenchError(f'Missing project logging file or path outside logging/: {source}')
    content = source.read_bytes()
    json.loads(content)
    image_id = case['runtime'].get('image_id')
    if not image_id:
        raise BenchError('Pin runtime.image_id before preparing recipe-specific logging')
    cache = cache_directory(case, image_id).resolve()
    destination = (cache / 'logging' / container_path.name).resolve()
    if not destination.is_relative_to(cache):
        raise BenchError('Logging destination escapes the recipe cache directory')
    if destination.exists():
        if destination.read_bytes() != content:
            raise BenchError(f'Existing logging file differs; preserve and review it: {destination}')
    elif check:
        raise BenchError(f'Logging file is not prepared: {destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(content)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--config-root', type=Path)
    parser.add_argument('--case', action='append', dest='case_ids')
    parser.add_argument('--check', action='store_true', help='Read-only check; do not create files')
    args = parser.parse_args(argv)
    try:
        path = args.campaign.expanduser().resolve()
        plan = resolve(path, args.config_root, args.case_ids)
        config_root = args.config_root.expanduser().resolve() if args.config_root else next(p for p in path.parents if p.name == 'configs')
        for case in plan['cases']:
            destination = prepare(case, config_root, check=args.check)
            print(f"{case['id']}: {destination or 'no custom logging file required'}")
        return 0
    except (BenchError, OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
