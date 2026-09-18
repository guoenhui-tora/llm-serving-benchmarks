"""Overlay one audited module, failing closed if the image or patch differs."""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
try:
    MANIFEST = json.loads((ROOT / 'manifest.json').read_text())
    for name, digest in MANIFEST['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
except BaseException as error:
    print(f'DSpark overlay validation failed: {error}', file=sys.stderr, flush=True)
    os._exit(78)

class DraftOverlay(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != MANIFEST['module']:
            return None
        original = importlib.machinery.PathFinder.find_spec(fullname, path)
        if not original or hashlib.sha256(Path(original.origin).read_bytes()).hexdigest() != MANIFEST['original_sha256']:
            raise RuntimeError('DSpark source differs from pinned vLLM image; refusing overlay')
        return importlib.util.spec_from_file_location(fullname, ROOT / 'utils.py')

sys.meta_path.insert(0, DraftOverlay())
