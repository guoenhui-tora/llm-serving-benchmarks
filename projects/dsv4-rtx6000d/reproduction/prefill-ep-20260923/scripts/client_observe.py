"""Observe official benchmark boundaries without changing request generation or clocks."""
import inspect,json,time,urllib.request
from pathlib import Path
from serving_bench.clients.synchronized import instrument

def install(serve,port):
    out=Path('/results');source=inspect.getsource(serve.benchmark);tree=instrument(source);window={}
    def scrape(name):
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics',timeout=15) as r:
            (out/name).write_bytes(r.read())
    def start():
        time.sleep(2) # Flush the official client's preliminary test request, outside timing.
        scrape('metrics-before.txt')
        begin=time.perf_counter();window.update(start_s=begin,start_unix=time.time());return begin
    def end(begin):
        finish=time.perf_counter();window.update(end_s=finish,end_unix=time.time(),duration_s=finish-begin)
        (out/'benchmark-window.json').write_text(json.dumps(window,indent=2))
        time.sleep(2)
        scrape('metrics-after.txt')
        return finish-begin
    calculate=serve.calculate_metrics
    def recorded(*args,**kwargs):
        bound=inspect.signature(calculate).bind(*args,**kwargs);result=calculate(*args,**kwargs)
        rows=[{'success':r.success,'input_tokens':q.prompt_len,'output_tokens':n,'ttft_s':r.ttft,'latency_s':r.latency,'itl_s':r.itl,'error':r.error} for q,r,n in zip(bound.arguments['input_requests'],bound.arguments['outputs'],result[1])]
        (out/'requests.json').write_text(json.dumps(rows))
        return result
    namespace=dict(serve.benchmark.__globals__);namespace.update(_sb_clock_start=start,_sb_clock_end=end,calculate_metrics=recorded)
    exec(compile(tree,'<observed-official-benchmark>','exec'),namespace);serve.benchmark=namespace['benchmark']
