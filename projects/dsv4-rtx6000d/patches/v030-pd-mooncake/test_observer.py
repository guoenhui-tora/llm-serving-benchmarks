import ast,contextlib,io,json,tempfile,types
from pathlib import Path
hook=Path(__file__).resolve().with_name('sitecustomize.py')
# Execute only the observer definitions, avoiding the unrelated quantization importer.
s=hook.read_text();ns={};exec('import importlib.abc, importlib.machinery, importlib.util, hashlib, sys\nfrom pathlib import Path\n'+s[s.index('# Observe original worker'):],ns)
import vllm
base=Path(vllm.__file__).parent/'distributed/kv_transfer/kv_connector/v1/mooncake'
name='vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats'
loader=ns['MoonStatsLoader'](name,str(base/'stats.py'));module=types.ModuleType(name);exec(loader.get_code(name),module.__dict__)
Stats=module.MooncakeKVConnectorStats
connector=(base/'mooncake_connector.py').read_text().replace(ns['MOON_WORKER_OLD'],ns['MOON_WORKER_NEW'])
tree=ast.parse(connector)
method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='get_kv_connector_stats' and 'self.xfer_stats' in ast.unparse(n))
method.returns=None;gns={};exec(compile(ast.Module(body=[method],type_ignores=[]),'audited-worker','exec'),gns)
getstats=gns['get_kv_connector_stats'];worker=types.SimpleNamespace(xfer_stats=Stats())
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
 assert getstats(worker) is None
 worker.xfer_stats.record_transfer(.01,4096,2)
 snapshot=getstats(worker);assert snapshot.data['bytes_transferred']==[4096]
 worker.xfer_stats.record_failed_transfer();worker.xfer_stats.record_failed_recv();worker.xfer_stats.record_kv_expired_req()
 snapshot=getstats(worker);assert snapshot.data['num_failed_recvs']==[1]
 assert worker.xfer_stats.is_empty()
records=[json.loads(x.split('LOCAL_MOONCAKE_STATS ',1)[1]) for x in buf.getvalue().splitlines()]
assert len(records)==3 and records[0]['transfers']==0 and records[1]['transfers']==1 and records[1]['bytes']==4096
assert records[2]['failed_transfers']==records[2]['failed_recvs']==records[2]['expired']==1
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'bad.py';p.write_text('wrong')
 for cls in ['MoonStatsLoader','MoonWorkerLoader']:
  try:ns[cls](name,str(p)).get_code(name)
  except RuntimeError:pass
  else:raise AssertionError('source mismatch accepted')
print('PASS: positive bytes, all failure counters, empty worker observation, reset semantics, source mismatch rejected')
