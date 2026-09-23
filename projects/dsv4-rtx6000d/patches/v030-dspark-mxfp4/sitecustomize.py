"""Pinned v0.30 loader hook: correct only the audited DSpark expert format."""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import linecache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / 'manifest.json').read_text())
OLD = '    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)'
NEW = '''    from dspark_native_mxfp4 import native_draft_config
    draft_vllm_config.quant_config = native_draft_config(
        vllm_config, draft_model_config, get_draft_quant_config(vllm_config)
    )'''


def patched_source(source):
    if hashlib.sha256(source).hexdigest() != MANIFEST['original_sha256']:
        raise RuntimeError('DSpark loader differs from audited v0.30 source')
    text = source.decode()
    if text.count(OLD) != 1:
        raise RuntimeError('Expected exactly one draft quantization assignment')
    text = text.replace(OLD, NEW)
    if text.count('    return draft_model') != 1:
        raise RuntimeError('Expected exactly one draft return')
    return text.replace('    return draft_model',
        '    from dspark_native_mxfp4 import verify_loaded_draft\n'
        '    verify_loaded_draft(draft_model)\n'
        '    return draft_model')


class Loader(importlib.machinery.SourceFileLoader):
    def get_source(self, fullname):
        return patched_source(Path(self.path).read_bytes())

    def get_code(self, fullname):
        source = self.get_source(fullname)
        filename = self.path + '.local-mxfp4.py'
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        return compile(source, filename, 'exec', dont_inherit=True)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != MANIFEST['module']:
            return None
        original = importlib.machinery.PathFinder.find_spec(fullname, path)
        if original is None or original.origin is None:
            raise RuntimeError('Cannot locate original DSpark loader')
        vllm_root = Path(original.origin).parents[5]
        quant = vllm_root / 'models/deepseek_v4/quant_config.py'
        if hashlib.sha256(quant.read_bytes()).hexdigest() != MANIFEST['quant_source_sha256']:
            raise RuntimeError('Quant dispatcher differs from audited v0.30 source')
        for relative, digest in MANIFEST['extra_source_sha256'].items():
            if hashlib.sha256((vllm_root / relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Registry/CUDA quant dispatcher differs: ' + relative)
        return importlib.util.spec_from_file_location(
            fullname, original.origin, loader=Loader(fullname, original.origin))


sys.meta_path.insert(0, Finder())
