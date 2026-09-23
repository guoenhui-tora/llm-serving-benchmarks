import asyncio,json,time
from aiohttp import web,ClientSession
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput,async_request_openai_completions
async def main():
    async def respond(req):
        payload=await req.json();assert payload['max_tokens']==1
        r=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await r.prepare(req)
        await asyncio.sleep(.06)
        await r.write(('data: '+json.dumps({'choices':[{'text':payload['prompt'],'finish_reason':'length'}]})+'\n\n').encode())
        await asyncio.sleep(.18)
        await r.write(b'data: {"choices":[],"usage":{"completion_tokens":1,"prompt_tokens":16384}}\n\ndata: [DONE]\n\n')
        await r.write_eof();return r
    app=web.Application();app.router.add_post('/v1/completions',respond);runner=web.AppRunner(app);await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',18989);await site.start()
    results=[]
    try:
        async with ClientSession() as session:
            for text in ('x',''):
                begin=time.perf_counter()
                row=await async_request_openai_completions(RequestFuncInput(prompt=text,api_url='http://127.0.0.1:18989/v1/completions',prompt_len=16384,output_len=1,model='test',ignore_eos=True),session)
                assert row.success and row.output_tokens==1 and row.ttft>=.05 and time.perf_counter()-begin-row.ttft>=.15 and abs(row.latency-row.ttft)<.01 and row.itl==[],repr(row)
                results.append({'empty_special_token':text=='','ttft_s':row.ttft,'latency_s':row.latency,'output_tokens':row.output_tokens})
    finally:await runner.cleanup()
    print(json.dumps({'status':'PASS','cases':results}))
asyncio.run(main())
