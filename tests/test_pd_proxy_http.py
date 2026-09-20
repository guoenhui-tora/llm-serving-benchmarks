"""Run with aiohttp installed (the pinned client image provides it)."""
import unittest
try:
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
except ImportError:
    web = None
from serving_bench.pd_proxy import make_app

@unittest.skipUnless(web, "aiohttp is available in the pinned client image")
class ProxyHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_metadata_never_reaches_decoder(self):
        calls=[]
        async def p(r):return web.json_response({'choices':[{'text':'x'}]})
        async def d(r):calls.append(await r.json());return web.json_response({})
        pa=web.Application();pa.router.add_post('/v1/completions',p)
        da=web.Application();da.router.add_post('/v1/completions',d)
        async with TestServer(pa) as ps, TestServer(da) as ds:
            async with TestClient(TestServer(make_app(str(ps.make_url('')).rstrip('/'),str(ds.make_url('')).rstrip('/')))) as c:
                r=await c.post('/v1/completions',json={'prompt':'x','max_tokens':1024})
                self.assertEqual(r.status,502);self.assertEqual(calls,[])

    async def test_stream_forwards_metadata_and_full_output_budget(self):
        calls=[]
        meta=dict(do_remote_prefill=True,remote_engine_id='e',remote_request_id='r',remote_host='host',remote_port=5600,remote_block_ids=[[1]],transfer_mode='pull')
        async def p(r):
            b=await r.json();self.assertEqual(b['max_tokens'],1);self.assertFalse(b['stream'])
            return web.json_response({'kv_transfer_params':meta})
        async def d(r):
            calls.append(await r.json());return web.Response(text='data: {"choices":[{"text":"ok"}]}\n\ndata: [DONE]\n\n',content_type='text/event-stream')
        pa=web.Application();pa.router.add_post('/v1/completions',p)
        da=web.Application();da.router.add_post('/v1/completions',d)
        async with TestServer(pa) as ps, TestServer(da) as ds:
            async with TestClient(TestServer(make_app(str(ps.make_url('')).rstrip('/'),str(ds.make_url('')).rstrip('/')))) as c:
                r=await c.post('/v1/completions',json={'prompt':'x','max_tokens':1024,'stream':True})
                self.assertEqual(r.status,200);self.assertIn('[DONE]',await r.text())
                self.assertEqual(calls[0]['max_tokens'],1024);self.assertEqual(calls[0]['kv_transfer_params'],meta)

if __name__=='__main__':unittest.main()
