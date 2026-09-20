"""Small experimental NIXL pull proxy; no retries or local-prefill fallback."""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time
import uuid


def transfer_params(value):
    if not isinstance(value, dict):
        raise ValueError('Missing KV transfer metadata')
    required = ('remote_engine_id', 'remote_request_id', 'remote_host', 'remote_port', 'remote_block_ids')
    if not value.get('do_remote_prefill') or any(not value.get(k) for k in required):
        raise ValueError('Incomplete remote-prefill metadata')
    if not isinstance(value['remote_block_ids'], list) or not any(value['remote_block_ids']):
        raise ValueError('Empty remote KV blocks')
    if value.get('transfer_mode', 'pull') not in ('pull', 'READ'):
        raise ValueError('Expected pull KV transfer')
    return value


def prefill_request(body):
    body = dict(body)
    body.update(stream=False, max_tokens=1, kv_transfer_params={'do_remote_decode': True, 'do_remote_prefill': False})
    if 'max_completion_tokens' in body:
        body['max_completion_tokens'] = 1
    for key in ('min_tokens', 'min_completion_tokens', 'stream_options'):
        body.pop(key, None)
    return body


def emit(event, **kw):
    print(json.dumps(dict(event=event, time=time.time(), monotonic=time.monotonic(), **kw)), flush=True)


class PDPool:
    """Reserve D for the whole request and P only through metadata retrieval."""
    def __init__(self, prefill, decode):
        def urls(value):
            items = [value] if isinstance(value, str) else list(value or ())
            items = [item.rstrip('/') for item in items]
            if not items or len(set(items)) != len(items):
                raise ValueError('PD pools require nonempty, unique service URLs')
            return items
        self.prefill = dict.fromkeys(urls(prefill), 0)
        self.decode = dict.fromkeys(urls(decode), 0)
        self.routes = {(p, d): 0 for p in self.prefill for d in self.decode}

    def reserve(self):
        ps = [p for p, n in self.prefill.items() if n == min(self.prefill.values())]
        ds = [d for d, n in self.decode.items() if n == min(self.decode.values())]
        # Balance equal-load pairs too; no await may split this reservation.
        p, d = min(((p, d) for p in ps for d in ds), key=self.routes.get)
        self.prefill[p] += 1
        self.decode[d] += 1
        self.routes[p, d] += 1
        return p, d

    def release_prefill(self, url):
        assert self.prefill[url] > 0
        self.prefill[url] -= 1

    def release_decode(self, url):
        assert self.decode[url] > 0
        self.decode[url] -= 1


def make_app(prefill, decode, ordinary=(), timeout=180, ordinary_policy='round-robin'):
    from aiohttp import web, ClientSession, ClientTimeout, TCPConnector
    app = web.Application()
    cycle = itertools.cycle(ordinary)
    if ordinary_policy not in ('round-robin', 'least-inflight'):
        raise ValueError('Unknown ordinary routing policy')
    inflight = dict.fromkeys(ordinary, 0)
    pool = None if ordinary else PDPool(prefill, decode)
    app['pd_pool'] = pool

    async def lifecycle(app):
        # Each upstream POST gets a fresh connection: an idle peer may close
        # a pooled socket while P and D have very different service times.
        # Never retry a non-idempotent model request after a disconnect.
        app['client'] = ClientSession(timeout=ClientTimeout(total=timeout),
                                      connector=TCPConnector(force_close=True), trust_env=False)
        yield
        await app['client'].close()
    app.cleanup_ctx.append(lifecycle)

    async def health(request):
        return web.json_response({'mode': 'ordinary' if ordinary else 'pd'})

    async def completion(request):
        started = time.monotonic()
        rid = request.headers.get('X-Request-Id') or str(uuid.uuid4())
        headers = {'X-Request-Id': rid}
        response = None
        ordinary_destination = None
        prefill_destination = None
        decode_destination = None
        prefill_reserved = False
        try:
            body = await request.json()
            if 'kv_transfer_params' in body:
                raise ValueError('Client-supplied transfer metadata is not accepted')
            emit('start', request_id=rid)
            if ordinary:
                destination = (min(inflight, key=inflight.get)
                               if ordinary_policy == 'least-inflight' else next(cycle))
                # Reserve without an await; count the full upstream stream.
                ordinary_destination = destination
                inflight[destination] += 1
                emit('route', request_id=rid, destination=destination,
                     policy=ordinary_policy, inflight=dict(inflight))
            else:
                prefill_destination, decode_destination = pool.reserve()
                prefill_reserved = True
                emit('pd_route', request_id=rid, prefill=prefill_destination,
                     decode=decode_destination, prefill_inflight=dict(pool.prefill),
                     decode_inflight=dict(pool.decode))
                try:
                    async with app['client'].post(prefill_destination + '/v1/completions', json=prefill_request(body), headers=headers) as p:
                        p.raise_for_status()
                        payload = await p.json()
                    body['kv_transfer_params'] = transfer_params(payload.get('kv_transfer_params'))
                finally:
                    pool.release_prefill(prefill_destination)
                    prefill_reserved = False
                emit('prefill_done', request_id=rid, elapsed=time.monotonic()-started,
                     kv_transfer_params=body['kv_transfer_params'], usage=payload.get('usage'),
                     prefill=prefill_destination, decode=decode_destination)
                destination = decode_destination
            async with app['client'].post(destination + '/v1/completions', json=body, headers=headers) as d:
                d.raise_for_status()
                if body.get('stream'):
                    response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
                    await response.prepare(request)
                    async for chunk in d.content.iter_any():
                        await response.write(chunk)
                    await response.write_eof()
                else:
                    response = web.Response(body=await d.read(), content_type='application/json')
            emit('done', request_id=rid, elapsed=time.monotonic()-started, destination=destination)
            return response
        except asyncio.CancelledError:
            emit('cancelled', request_id=rid, elapsed=time.monotonic()-started)
            raise
        except Exception as exc:
            emit('error', request_id=rid, elapsed=time.monotonic()-started, error=repr(exc))
            if response is not None and response.prepared:
                if request.transport is not None:
                    request.transport.close()
                return response
            return web.json_response({'error': str(exc), 'request_id': rid}, status=502)
        finally:
            if ordinary_destination is not None:
                inflight[ordinary_destination] -= 1
            if prefill_reserved:
                pool.release_prefill(prefill_destination)
            if decode_destination is not None:
                pool.release_decode(decode_destination)
                emit('pd_release', request_id=rid,
                     prefill_inflight=dict(pool.prefill), decode_inflight=dict(pool.decode))

    app.router.add_get('/health', health)
    app.router.add_post('/v1/completions', completion)
    return app


def main():
    from aiohttp import web
    p = argparse.ArgumentParser()
    p.add_argument('--prefill', nargs='+'); p.add_argument('--decode', nargs='+')
    p.add_argument('--ordinary', nargs='*', default=[])
    p.add_argument('--ordinary-policy', choices=('round-robin', 'least-inflight'), default='round-robin')
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=31250)
    p.add_argument('--timeout', type=int, default=180)
    a = p.parse_args()
    if not a.ordinary and not (a.prefill and a.decode):
        p.error('prefill and decode are required for PD')
    web.run_app(make_app(a.prefill, a.decode, a.ordinary, a.timeout, a.ordinary_policy), host=a.host, port=a.port, access_log=None, handler_cancellation=True)


if __name__ == '__main__':
    main()
