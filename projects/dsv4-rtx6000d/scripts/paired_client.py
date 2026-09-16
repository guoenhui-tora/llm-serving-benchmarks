"""Pinned official client, with a barrier immediately before its measured timer.
Only timing/barrier instrumentation is inserted; generation and metrics are unchanged.
"""
import inspect
import os
import time
from pathlib import Path
import vllm.benchmarks.serve as serve
from vllm.utils.argparse_utils import FlexibleArgumentParser

def mark(name):
    Path('/results/'+name+'.time').write_text(str(time.time()))

async def ready():
    mark('ready')
    if os.environ.get('PAIRED_BARRIER') == '1':
        deadline=time.monotonic()+7200
        while not Path('/results/release').exists():
            if time.monotonic()>deadline: raise TimeoutError('Paired benchmark barrier timed out')
            await serve.asyncio.sleep(0.01)
    mark('benchmark-start')

source=inspect.getsource(serve.benchmark)
start='    benchmark_start_time = time.perf_counter()'
end='    benchmark_duration = time.perf_counter() - benchmark_start_time'
assert source.count(start)==source.count(end)==1
source=source.replace(start,'    await _paired_ready()\n'+start).replace(end,end+'\n    _paired_mark("benchmark-end")')
Path('/results/instrumented-benchmark.py').write_text(source)
serve.__dict__.update(_paired_ready=ready,_paired_mark=mark)
exec(compile(source,'<paired official benchmark>','exec'),serve.__dict__)
p=FlexibleArgumentParser()
serve.add_cli_args(p)
serve.main(p.parse_args())
