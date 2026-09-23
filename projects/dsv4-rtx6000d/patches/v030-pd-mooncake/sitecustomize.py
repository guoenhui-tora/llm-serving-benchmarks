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

# Observe original worker counters; the image has no Mooncake Prom exporter.
class MoonStatsLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        data=Path(self.path).read_bytes()
        if hashlib.sha256(data).hexdigest() != MOON_STATS_SHA:
            raise RuntimeError('Mooncake stats source differs from audited image')
        old='        return MooncakeKVConnectorStats(data=snapshot_data)'
        new="        import os\n        import time\n        global _audit_last\n        now = time.monotonic()\n        if any(snapshot_data.values()) or now - globals().get('_audit_last', 0) >= 5:\n            _audit_last = now\n            print('LOCAL_MOONCAKE_STATS ' + json.dumps({\n                'pid': os.getpid(), 'transfers': len(snapshot_data['transfer_duration']),\n                'bytes': sum(snapshot_data['bytes_transferred']),\n                'failed_transfers': sum(snapshot_data['num_failed_transfers']),\n                'failed_recvs': sum(snapshot_data['num_failed_recvs']),\n                'expired': sum(snapshot_data['num_kv_expired_reqs'])}), flush=True)\n        return MooncakeKVConnectorStats(data=snapshot_data)"
        source=data.decode()
        if source.count(old)!=1:raise RuntimeError('Mooncake observer anchor mismatch')
        return compile('import json\n'+source.replace(old,new),self.path,'exec')
class MoonStatsFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path,target=None):
        if fullname!='vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats':return None
        original=importlib.machinery.PathFinder.find_spec(fullname,path)
        return importlib.util.spec_from_file_location(fullname,original.origin,loader=MoonStatsLoader(fullname,original.origin))
MOON_STATS_SHA='f39b2f0b1db2e4f922e27e684a3f7bbf5bb279b82a0a8c3bb0c7ae69b6d14d24'
sys.meta_path.insert(0,MoonStatsFinder())

# Observe empty worker intervals too, keeping the original None return intact.
class MoonWorkerLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        data=Path(self.path).read_bytes()
        if hashlib.sha256(data).hexdigest() != MOON_WORKER_SHA:
            raise RuntimeError('Mooncake worker source differs from audited image')
        source=data.decode()
        if source.count(MOON_WORKER_OLD)!=1:
            raise RuntimeError('Mooncake worker observer anchor mismatch')
        if source.count(MOON_REQUEST_OLD)!=1:raise RuntimeError('Request audit anchor mismatch')
        return compile(source.replace(MOON_WORKER_OLD,MOON_WORKER_NEW).replace(MOON_REQUEST_OLD,MOON_REQUEST_NEW),self.path,'exec')
class MoonWorkerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path,target=None):
        if fullname!='vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector':return None
        original=importlib.machinery.PathFinder.find_spec(fullname,path)
        return importlib.util.spec_from_file_location(fullname,original.origin,loader=MoonWorkerLoader(fullname,original.origin))
MOON_WORKER_SHA='516be01320e64b0110e102eeb69a6ba766dd5ac8cb3036db6b97b7d54413323d'
MOON_WORKER_OLD='        if self.xfer_stats.is_empty():\n            return None\n        return self.xfer_stats.clone_and_reset()'
MOON_WORKER_NEW="        if self.xfer_stats.is_empty():\n            import time, os, json\n            now = time.monotonic()\n            if now - getattr(self, '_local_audit_last', 0) >= 5:\n                self._local_audit_last = now\n                print('LOCAL_MOONCAKE_STATS ' + json.dumps({\n                    'pid': os.getpid(), 'transfers': 0, 'bytes': 0,\n                    'failed_transfers': 0, 'failed_recvs': 0, 'expired': 0}), flush=True)\n            return None\n        return self.xfer_stats.clone_and_reset()"
sys.meta_path.insert(0,MoonWorkerFinder())

MOON_REQUEST_OLD='            response = MooncakeXferResponse(\n                status=response_status,\n                ok_reqs=[d_req_id for d_req_id, _ in ok_ready_reqs] or None,\n                err_reqs=err_reqs or None,\n                err_msg=err_msg,\n            )\n            await sock.send_multipart((identity, self._encoder.encode(response)))'
MOON_REQUEST_NEW="            response = MooncakeXferResponse(\n                status=response_status,\n                ok_reqs=[d_req_id for d_req_id, _ in ok_ready_reqs] or None,\n                err_reqs=err_reqs or None,\n                err_msg=err_msg,\n            )\n            import os, json\n            print('LOCAL_MOONCAKE_REQUESTS ' + json.dumps({\n                'pid': os.getpid(), 'tp_rank': self.tp_rank, 'pp_rank': self.pp_rank,\n                'ok_reqs': [d_req_id for d_req_id, _ in ok_ready_reqs],\n                'err_reqs': err_reqs or []}), flush=True)\n            await sock.send_multipart((identity, self._encoder.encode(response)))"
