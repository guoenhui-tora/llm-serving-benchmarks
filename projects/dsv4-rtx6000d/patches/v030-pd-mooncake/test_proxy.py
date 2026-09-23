import asyncio, importlib.util, json, os
from pathlib import Path
from aiohttp import web,ClientSession
p=Path(__file__).resolve().with_name('pd_proxy.py');s=importlib.util.spec_from_file_location('candidate',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
async def main():
 pseen=[];dseen=[];runners=[]
 async def start(app):
  runner=web.AppRunner(app);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start();runners.append(runner);return 'http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1])
 async def query(req):return web.json_response({'0':{'engine_id':'fixed-engine','worker_addr':{'0':{'0':'a','1':'b'},'1':{'0':'c','1':'d'}}}})
 async def prefill(req):
  b=await req.json();assert b['max_tokens']==1 and b['kv_transfer_params']['do_remote_decode'];pseen.append(b['kv_transfer_params']['transfer_id']);return web.json_response({'choices':[{'text':'x'}],'usage':{'prompt_tokens':16384}})
 async def decode(req):
  b=await req.json();v=b['kv_transfer_params'];assert v['transfer_id'] in pseen and v['remote_engine_id']=='fixed-engine' and v['do_remote_prefill'];dseen.append(v['transfer_id']);return web.json_response({'choices':[{'text':'done'}]})
 a=web.Application();a.router.add_get('/query',query);a.router.add_post('/v1/completions',prefill);purl=await start(a)
 a=web.Application();a.router.add_post('/v1/completions',decode);durl=await start(a)
 os.environ['MOONCAKE_BOOTSTRAP_MAP']=json.dumps({purl:purl});url=await start(m.make_app([purl],[durl]))
 try:
  async with ClientSession() as client:
   for _ in range(2):
    async with client.post(url+'/v1/completions',json={'prompt':'test','max_tokens':16,'stream':False}) as r:assert r.status==200;await r.read()
   async with client.post(url+'/v1/completions',json={'prompt':'test','kv_transfer_params':{'transfer_id':'injected'}}) as r:assert r.status==502;await r.read()
  assert len(set(pseen))==2 and dseen==pseen
  for wrong in [{},{'0':{'engine_id':'x','worker_addr':{'0':{'0':'a'}}}}, {'0':{},'1':{}}]:
   try:m.mooncake_registration(wrong)
   except ValueError:pass
   else:raise AssertionError('Malformed registration accepted')
 finally:
  for r in reversed(runners):await r.cleanup()
 print('PASS: unique transfer IDs, P-to-D metadata, missing ranks and client metadata rejected')
asyncio.run(main())
