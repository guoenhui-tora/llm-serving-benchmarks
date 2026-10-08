"""Backport only the profiling hunk of vLLM PR #54856 to the pinned image."""
import hashlib
import importlib.machinery
import linecache
from pathlib import Path
import sys

OLD = """                num_tokens_across_dp=(
                    dp_sync.num_tokens_across_dp if dp_sync is not None else None
                ),"""
NEW = "                num_tokens_across_dp=None,"


def patch_source(source: bytes, expected_sha256: str) -> bytes:
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise RuntimeError('DSpark DP source differs from pinned vLLM image; refusing overlay')
    text = source.decode('utf-8')
    if text.count(OLD) != 1:
        raise RuntimeError('Expected exactly one DSpark profiling hunk')
    return text.replace(OLD, NEW).encode('utf-8')


class ProfileLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, path, expected_sha256):
        super().__init__(fullname, path)
        self.expected_sha256 = expected_sha256
        self.source_filename = path + ".dspark-dp-profile.py"

    def get_source(self, fullname):
        return patch_source(Path(self.path).read_bytes(), self.expected_sha256).decode('utf-8')

    def get_code(self, fullname):
        # Triton uses inspect.getsource: its text/line numbers must match the
        # executed code. A virtual filename prevents a later linecache clear
        # from falling back to the original file; get_source restores this view.
        source = self.get_source(fullname)
        linecache.cache[self.source_filename] = (len(source), None, source.splitlines(True), self.source_filename)
        # Compile directly: never read/write a .pyc shared with the unpatched image.
        code = compile(source, self.source_filename, 'exec', dont_inherit=True)
        print('UPSTREAM_DSPARK_DP_PROFILE_FIX: PR #54856 facd9a74a1; '
              'profiling uses draft-local DP token synchronization',
              file=sys.stderr, flush=True)
        return code
