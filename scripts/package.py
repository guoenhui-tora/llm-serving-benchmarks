#!/usr/bin/env python3
"""Package framework and curated projects, excluding local experiment artifacts."""
from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path

SOURCE_DIRS = ('src', 'tests', 'scripts', 'docs', 'projects')
SOURCE_FILES = ('README.md', 'AGENTS.md', '.gitignore', 'bench', 'pyproject.toml', 'requirements.txt', 'results/.gitkeep')
EXCLUDED_DIRS = {'.git', '.venv', 'venv', '__pycache__', 'dist', 'build', '.pytest_cache',
                 '.mypy_cache', '.ruff_cache', 'cache', '.cache', 'results', 'artifacts', 'tmp', 'logs'}


def source_files(root: Path):
    for name in SOURCE_FILES:
        path = root / name
        if path.is_file() and not path.is_symlink():
            yield path
    for name in SOURCE_DIRS:
        for path in sorted((root / name).rglob('*')):
            relative = path.relative_to(root)
            if not path.is_file() or path.is_symlink():
                continue
            if any(p in EXCLUDED_DIRS or p.endswith('.egg-info') for p in relative.parts[:-1]):
                # src/serving_bench/results is code, unlike a project's raw results.
                if not (relative.parts[:3] == ('src', 'serving_bench', 'results')
                        and len(relative.parts) == 4):
                    continue
            if path.name == '.env' or path.name.startswith('.env.') or path.name == '.DS_Store' or path.suffix in {'.pyc', '.pyo'}:
                continue
            yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (args.output or root / 'dist/llm-serving-benchmarks-source.tar.gz').expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz') as archive:
        for path in source_files(root):
            if path.resolve() != output:
                archive.add(path, arcname=str(Path('llm-serving-benchmarks') / path.relative_to(root)), recursive=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + '.sha256').write_text(f'{digest}  {output.name}\n')
    print(output)
    print(digest)


if __name__ == '__main__':
    main()
