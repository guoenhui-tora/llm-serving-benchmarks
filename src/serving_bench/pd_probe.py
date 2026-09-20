"""Bounded functional checks for an external completions PD endpoint."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request


def post(url, endpoint, body, rid, timeout):
    req=urllib.request.Request(url+endpoint, data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json','X-Request-Id':rid})
    start=time.monotonic()
    with urllib.request.urlopen(req,timeout=timeout) as r: value=json.load(r)
    return {'request_id':rid,'duration':time.monotonic()-start,'response':value}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--proxy',required=True);p.add_argument('--direct',required=True)
    p.add_argument('--model',required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['correctness','capacity'],default='correctness')
    p.add_argument('--budget',type=int,default=900)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    records=[json.loads(l) for l in a.dataset.read_text().splitlines()]
    deadline=time.monotonic()+a.budget
    def request(url, body, rid):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Probe budget exhausted')
        result=post(url,'/v1/completions',body,rid,min(180,remaining))
        (a.output/(rid+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
        usage=result['response']['usage']
        if usage['completion_tokens']!=body['max_tokens']:raise ValueError(f'Output workload mismatch {rid}: {usage}')
        print(json.dumps({'request_id':rid,'duration':result['duration'],'usage':usage}),flush=True)
        return result['response']
    status={'phase':a.phase,'started':time.time(),'status':'RUNNING'}
    try:
        base={'model':a.model,'stream':False,'temperature':0,'seed':0,'ignore_eos':True,'max_tokens':256}
        if a.phase=='correctness':
            tokens=post(a.direct,'/tokenize',{'model':a.model,'prompt':records[0]['prompt']},'tokenize',60)['response']['tokens']
            lengths=[len(tokens),len(tokens)-1,127,128,129,255,256,257]
            (a.output/'input-lengths.json').write_text(json.dumps(lengths))
            for i,n in enumerate(lengths):
                body=dict(base,prompt=tokens[:n],add_special_tokens=False)
                ordinary=request(a.direct,body,f'direct-{i:02}')
                pd=request(a.proxy,body,f'pd-{i:02}')
                if ordinary['usage']['prompt_tokens']!=n or pd['usage']['prompt_tokens']!=n:
                    raise ValueError('Input token count differs')
                if ordinary['choices'][0]['text']!=pd['choices'][0]['text']:
                    raise ValueError(f'Greedy output mismatch at input length {n}; stop for diagnosis')
        else:
            for concurrency,count,length in [(8,16,256),(32,32,1024)]:
                def one(i):
                    return request(a.proxy,dict(base,prompt=records[i]['prompt'],max_tokens=length),f'c{concurrency}-{i:03}')
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
                    list(ex.map(one,range(count)))
        status['status']='PASS'
    except Exception as e:
        status['status']='FAIL';status['error']=repr(e)
    finally:
        status['ended']=time.time();(a.output/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    if status['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
