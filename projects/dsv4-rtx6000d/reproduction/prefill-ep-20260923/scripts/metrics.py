import json,re,statistics,threading,time,urllib.request
from pathlib import Path

def parse(text):
    result={}
    for line in text.splitlines():
        if not line or line.startswith('#'):continue
        m=re.match(r'([^\s{]+)(\{.*\})?\s+([^ ]+)',line)
        if not m:continue
        name,labels,value=m.groups()
        if name.endswith(('_bucket','_created')):continue
        if not name.startswith('vllm:'):continue
        result[name+(labels or '')]=float(value)
    return result

def label(key,field):
    m=re.search(r'(?:\{|,)'+field+r'="([^"]*)"',key);return m.group(1) if m else None

def group(values,name):
    return {k:v for k,v in values.items() if k.split('{')[0]==name}

def audit(directory,count):
    before=parse((directory/'metrics-before.txt').read_text());after=parse((directory/'metrics-after.txt').read_text())
    delta={k:v-before.get(k,0) for k,v in after.items()}
    def total(name,required=True):
        rows=group(delta,name)
        if required and not rows:raise RuntimeError('Required metric absent: '+name)
        return sum(rows.values())
    n=total('vllm:request_success_total');assert n==count,('server completed',n,count)
    prompt=total('vllm:request_prompt_tokens_sum');assert prompt==count*16384,('server input',prompt)
    outputs=total('vllm:request_generation_tokens_sum');assert outputs==count,('server output',outputs)
    computed=total('vllm:request_prefill_kv_computed_tokens_sum');assert computed==count*16384,('actual prefill',computed)
    hits=total('vllm:prefix_cache_hits_total');assert hits==0,('prefix hits',hits)
    preemptions=total('vllm:num_preemptions_total');assert preemptions==0,('preemptions',preemptions)
    completed=group(delta,'vllm:request_success_total');engines={}
    for k,v in completed.items():
        engine=label(k,'engine');engines[engine]=engines.get(engine,0)+v
    assert len(engines)==2 and all(v>0 for v in engines.values()),('DP distribution',engines)
    rows=json.loads((directory/'requests.json').read_text())
    assert len(rows)==count and all(r['success'] and r['input_tokens']==16384 and r['output_tokens']==1 and r['ttft_s']>0 for r in rows)
    result={'status':'PASS','completed':n,'input_tokens':prompt,'output_tokens':outputs,'computed_prefill_tokens':computed,'prefix_hits':hits,'preemptions':preemptions,'completed_by_engine':engines,'mean_queue_s':total('vllm:request_queue_time_seconds_sum')/count,'mean_prefill_s':total('vllm:request_prefill_time_seconds_sum')/count,'counter_deltas':{k:v for k,v in delta.items() if any(t in k for t in ('_sum','_count','_total'))}}
    (directory/'workload-audit.json').write_text(json.dumps(result,indent=2));return result

class MetricsRecorder:
    def __init__(self,port,path):self.port=port;self.path=path;self.stop=threading.Event();self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        with self.path.open('w',buffering=1) as f:
            while not self.stop.is_set():
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/metrics',timeout=5) as r:values=parse(r.read().decode())
                    row={'unix_s':time.time(),'values':values}
                except Exception as e:row={'unix_s':time.time(),'error':str(e)}
                f.write(json.dumps(row)+'\n');self.stop.wait(5)
    def start(self):self.thread.start()
    def close(self):self.stop.set();self.thread.join(10)
