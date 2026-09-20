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


def make_app(prefill, decode, ordinary=(), timeout=180, ordinary_policy='round-robin'):
    from aiohttp import web, ClientSession, ClientTimeout
    app = web.Application()
    cycle = itertools.cycle(ordinary)
    if ordinary_policy not in ('round-robin', 'least-inflight'):
        raise ValueError('Unknown ordinary routing policy')
    inflight = dict.fromkeys(ordinary, 0)

    async def lifecycle(app):
        app['client'] = ClientSession(timeout=ClientTimeout(total=timeout), trust_env=False)
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
                async with app['client'].post(prefill + '/v1/completions', json=prefill_request(body), headers=headers) as p:
                    p.raise_for_status()
                    payload = await p.json()
                body['kv_transfer_params'] = transfer_params(payload.get('kv_transfer_params'))
                emit('prefill_done', request_id=rid, elapsed=time.monotonic()-started,
                     kv_transfer_params=body['kv_transfer_params'], usage=payload.get('usage'))
                destination = decode
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
        except Exception as exc:
            emit('error', request_id=rid, elapsed=time.monotonic()-started, error=repr(exc))
            if response is not None and response.prepared:
                request.transport.close()
                return response
            return web.json_response({'error': str(exc), 'request_id': rid}, status=502)
        finally:
            if ordinary_destination is not None:
                inflight[ordinary_destination] -= 1

    app.router.add_get('/health', health)
    app.router.add_post('/v1/completions', completion)
    return app


def main():
    from aiohttp import web
    p = argparse.ArgumentParser()
    p.add_argument('--prefill'); p.add_argument('--decode')
    p.add_argument('--ordinary', nargs='*', default=[])
    p.add_argument('--ordinary-policy', choices=('round-robin', 'least-inflight'), default='round-robin')
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=31250)
    p.add_argument('--timeout', type=int, default=180)
    a = p.parse_args()
    if not a.ordinary and not (a.prefill and a.decode):
        p.error('prefill and decode are required for PD')
    web.run_app(make_app(a.prefill, a.decode, a.ordinary, a.timeout, a.ordinary_policy), host=a.host, port=a.port, access_log=None)


if __name__ == '__main__':
    main()
