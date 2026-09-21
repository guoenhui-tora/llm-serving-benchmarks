"""Run with aiohttp installed (the pinned client image provides it)."""
import asyncio
from contextlib import AsyncExitStack
import unittest
try:
    from aiohttp import web, TCPConnector
    from aiohttp.test_utils import TestClient, TestServer
except ImportError:
    web = None
from serving_bench.pd_proxy import make_app

@unittest.skipUnless(web, "aiohttp is available in the pinned client image")
class ProxyHTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_least_inflight_tracks_stream_and_releases_errors(self):
        opened = asyncio.Event()
        release = asyncio.Event()
        calls = []
        async def a(r):
            body = await r.json()
            calls.append(('a', body['prompt']))
            if body['prompt'] == 'hold':
                response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
                await response.prepare(r)
                await response.write(b'data: first\n\n')
                opened.set()
                await release.wait()
                await response.write_eof()
                return response
            return web.Response(status=500)
        async def b(r):
            calls.append(('b', (await r.json())['prompt']))
            return web.json_response({})
        aa = web.Application(); aa.router.add_post('/v1/completions', a)
        ba = web.Application(); ba.router.add_post('/v1/completions', b)
        async with TestServer(aa) as sa, TestServer(ba) as sb:
            urls = [str(s.make_url('')).rstrip('/') for s in (sa, sb)]
            async with TestClient(TestServer(make_app(None, None, urls, ordinary_policy='least-inflight'))) as c:
                try:
                    held = await c.post('/v1/completions', json={'prompt': 'hold', 'stream': True})
                    await asyncio.wait_for(opened.wait(), 2)
                    for prompt in ('fast1', 'fast2'):
                        r = await c.post('/v1/completions', json={'prompt': prompt})
                        self.assertEqual(r.status, 200)
                        await r.read()
                    self.assertEqual(calls, [('a', 'hold'), ('b', 'fast1'), ('b', 'fast2')])
                finally:
                    release.set()
                await held.read()
                # A failed upstream must also release its in-flight reservation.
                for prompt in ('error1', 'error2'):
                    r = await c.post('/v1/completions', json={'prompt': prompt})
                    self.assertEqual(r.status, 502)
                    await r.read()
                self.assertEqual(calls[-2:], [('a', 'error1'), ('a', 'error2')])

    async def test_upstream_connections_are_not_reused_or_retried(self):
        transports = []
        async def upstream(request):
            transports.append(request.transport)
            body = await request.json()
            if body['prompt'] == 'disconnect':
                request.transport.close()
                return web.Response()
            return web.json_response({'ok': True})
        app = web.Application(); app.router.add_post('/v1/completions', upstream)
        async with TestServer(app) as server:
            url = str(server.make_url('')).rstrip('/')
            async with TestClient(TestServer(make_app(None, None, [url]))) as client:
                for prompt, status in [('first', 200), ('second', 200), ('disconnect', 502), ('after', 200)]:
                    response = await client.post('/v1/completions', json={'prompt': prompt})
                    self.assertEqual(response.status, status)
                    await response.read()
                self.assertEqual(len(transports), 4)
                self.assertEqual(len(set(transports)), 4)

    async def test_more_than_100_requests_reach_upstreams(self):
        # Hold all streams open: a hidden pool cap must not serialize C128.
        count = 128
        for mode in ('ordinary', 'pd'):
            with self.subTest(mode=mode):
                p_calls = []; d_calls = []
                all_p = asyncio.Event(); all_d = asyncio.Event()
                release_p = asyncio.Event(); release_d = asyncio.Event()
                async def prefill(request):
                    body = await request.json(); p_calls.append(body['prompt'])
                    if len(p_calls) == count: all_p.set()
                    await release_p.wait()
                    return web.json_response({'kv_transfer_params': dict(
                        do_remote_prefill=True, remote_engine_id='p',
                        remote_request_id=body['prompt'], remote_host='host',
                        remote_port=5600, remote_block_ids=[[1]])})
                async def stream(request):
                    body = await request.json(); d_calls.append(body['prompt'])
                    if len(d_calls) == count: all_d.set()
                    response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
                    await response.prepare(request); await response.write(b'data: first\n\n')
                    await release_d.wait(); await response.write(b'data: [DONE]\n\n')
                    await response.write_eof(); return response
                pa = web.Application(); pa.router.add_post('/v1/completions', prefill)
                da = web.Application(); da.router.add_post('/v1/completions', stream)
                async with TestServer(pa) as ps, TestServer(da) as ds:
                    pu = str(ps.make_url('')).rstrip('/')
                    du = str(ds.make_url('')).rstrip('/')
                    app = make_app(pu, du) if mode == 'pd' else make_app(None, None, [du])
                    async with TestClient(TestServer(app), connector=TCPConnector(limit=0)) as client:
                        async def request(index):
                            response = await client.post('/v1/completions', json={
                                'prompt': str(index), 'max_tokens': 1024, 'stream': True})
                            self.assertEqual(response.status, 200)
                            return await response.text()
                        tasks = [asyncio.create_task(request(i)) for i in range(count)]
                        try:
                            if mode == 'pd':
                                await asyncio.wait_for(all_p.wait(), 5)
                                release_p.set()
                            await asyncio.wait_for(all_d.wait(), 5)
                            self.assertEqual(len(set(d_calls)), count)
                        finally:
                            release_p.set(); release_d.set()
                            bodies = await asyncio.wait_for(asyncio.gather(*tasks), 10)
                        self.assertTrue(all('[DONE]' in body for body in bodies))
                        if mode == 'pd':
                            self.assertEqual(len(set(p_calls)), count)
                            self.assertFalse(any(app['pd_pool'].prefill.values()))
                            self.assertFalse(any(app['pd_pool'].decode.values()))

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

    async def test_multiple_p_d_routes_and_reservation_lifetimes(self):
        all_p = asyncio.Event(); release_p = asyncio.Event(); release_d = asyncio.Event()
        p_calls = []; pairs = []
        def prefill(index):
            async def handler(request):
                body = await request.json(); p_calls.append((index, body['prompt']))
                self.assertEqual(body['max_tokens'], 1)
                if len(p_calls) == 8: all_p.set()
                await release_p.wait()
                meta = dict(do_remote_prefill=True, remote_engine_id=f'p{index}',
                            remote_request_id=body['prompt'], remote_host='host',
                            remote_port=5600+index, remote_block_ids=[[index+1]])
                return web.json_response({'kv_transfer_params': meta})
            return handler
        def decode(index):
            async def handler(request):
                body = await request.json()
                self.assertEqual(body['max_tokens'], 1024)
                meta = body['kv_transfer_params']
                self.assertEqual(meta['remote_request_id'], body['prompt'])
                pairs.append((meta['remote_engine_id'], index))
                response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
                await response.prepare(request); await response.write(b'data: first\n\n')
                await release_d.wait(); await response.write(b'data: [DONE]\n\n')
                await response.write_eof(); return response
            return handler
        async with AsyncExitStack() as stack:
            urls = []
            for handler in [prefill(i) for i in range(4)] + [decode(i) for i in range(2)]:
                app = web.Application(); app.router.add_post('/v1/completions', handler)
                server = await stack.enter_async_context(TestServer(app, handler_cancellation=True))
                urls.append(str(server.make_url('')).rstrip('/'))
            app = make_app(urls[:4], urls[4:]); pool = app['pd_pool']
            client = await stack.enter_async_context(TestClient(TestServer(app, handler_cancellation=True)))
            tasks = [asyncio.create_task(client.post('/v1/completions', json=dict(prompt=str(i), max_tokens=1024, stream=True))) for i in range(8)]
            try:
                await asyncio.wait_for(all_p.wait(), 3)
                self.assertEqual(list(pool.prefill.values()), [2]*4)
                self.assertEqual(list(pool.decode.values()), [4]*2)
                release_p.set()
                responses = await asyncio.wait_for(asyncio.gather(*tasks), 3)
                self.assertEqual(set(pairs), {(f'p{p}', d) for p in range(4) for d in range(2)})
                self.assertEqual(list(pool.prefill.values()), [0]*4)
                self.assertEqual(list(pool.decode.values()), [4]*2)
                release_d.set()
                bodies = await asyncio.gather(*(r.text() for r in responses))
                self.assertTrue(all('[DONE]' in b for b in bodies))
                self.assertEqual(list(pool.decode.values()), [0]*2)
            finally:
                release_p.set(); release_d.set()
                for task in tasks:
                    if not task.done(): task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def test_pd_http_errors_release_both_pools(self):
        stage = 'p'; decoder_calls = []
        async def prefill(request):
            if stage == 'p': return web.Response(status=500)
            return web.json_response({'kv_transfer_params': dict(do_remote_prefill=True,
                remote_engine_id='p', remote_request_id='r', remote_host='host',
                remote_port=5600, remote_block_ids=[[1]])})
        async def decode(request):
            decoder_calls.append(await request.json()); return web.Response(status=500)
        pa=web.Application(); pa.router.add_post('/v1/completions',prefill)
        da=web.Application(); da.router.add_post('/v1/completions',decode)
        async with TestServer(pa) as ps, TestServer(da) as ds:
            app=make_app(str(ps.make_url('')).rstrip('/'),str(ds.make_url('')).rstrip('/'))
            async with TestClient(TestServer(app)) as client:
                for stage in ['p','d','d']:
                    response=await client.post('/v1/completions',json={'prompt':'x','max_tokens':1024})
                    self.assertEqual(response.status,502); await response.read()
                    self.assertFalse(any(app['pd_pool'].prefill.values()))
                    self.assertFalse(any(app['pd_pool'].decode.values()))
                self.assertEqual(len(decoder_calls),2)

    async def test_disconnect_during_prefill_and_decode_releases_pools(self):
        for stage in ['p','d']:
            entered=asyncio.Event(); release=asyncio.Event()
            async def prefill(request):
                if stage == 'p': entered.set(); await release.wait()
                return web.json_response({'kv_transfer_params':dict(do_remote_prefill=True,
                    remote_engine_id='p',remote_request_id='r',remote_host='host',
                    remote_port=5600,remote_block_ids=[[1]])})
            async def decode(request):
                response=web.StreamResponse(headers={'Content-Type':'text/event-stream'})
                await response.prepare(request); await response.write(b'data: first\n\n')
                entered.set(); await release.wait(); await response.write_eof(); return response
            pa=web.Application(); pa.router.add_post('/v1/completions',prefill)
            da=web.Application(); da.router.add_post('/v1/completions',decode)
            async with TestServer(pa,handler_cancellation=True) as ps, TestServer(da,handler_cancellation=True) as ds:
                app=make_app(str(ps.make_url('')).rstrip('/'),str(ds.make_url('')).rstrip('/'))
                async with TestClient(TestServer(app,handler_cancellation=True)) as client:
                    task=asyncio.create_task(client.post('/v1/completions',json={'prompt':'x','max_tokens':1024,'stream':True}))
                    try:
                        await asyncio.wait_for(entered.wait(),3)
                        if stage == 'p':
                            task.cancel()
                            with self.assertRaises(asyncio.CancelledError): await task
                        else:
                            response=await asyncio.wait_for(task,3); response.close()
                        for _ in range(100):
                            if not any(app['pd_pool'].prefill.values()) and not any(app['pd_pool'].decode.values()): break
                            await asyncio.sleep(.01)
                        self.assertFalse(any(app['pd_pool'].prefill.values()))
                        self.assertFalse(any(app['pd_pool'].decode.values()))
                    finally:
                        release.set()
                        if not task.done(): task.cancel()
                        await asyncio.gather(task,return_exceptions=True)

if __name__=='__main__':unittest.main()
