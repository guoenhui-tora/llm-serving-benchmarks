#!/usr/bin/env python3
"""Local paired orchestration; reuse repository validation, Docker and log checks.
Capacity mode never starts a benchmark. Both modes retain all artifacts and JIT caches.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from serving_bench import __version__
from serving_bench.common import BenchError, Logger, capture, fingerprint, write_json, save_command, utcnow
from serving_bench.config import resolve
from serving_bench.executors import docker
from serving_bench.checks import http, logs
from serving_bench.clients import vllm_bench
from serving_bench.locks import gpu_locks
from prepare_logging import prepare

BINDINGS={'vllm':('0-13,64-77','14-15,78-79','0'), 'sglang':('32-45,96-109','46-47,110-111','2')}

def bound(argv,engine,client=False):
    cpus=BINDINGS[engine][int(client)]
    return argv[:2]+['--cpuset-cpus',cpus,'--cpuset-mems',BINDINGS[engine][2]]+argv[2:]

def phase(item,where,count,barrier=False):
    where.mkdir(parents=True,exist_ok=False)
    c=item['case']; w=c['workloads'][0]; owner=item['owner']; engine=c['runtime']['engine']
    argv=vllm_bench.command(c,w,32,count,where,f'sb-{owner}-client',owner,item['facts']['client_image']['Id'])
    at=argv.index('--entrypoint')
    prefix=['-v',str(Path(__file__).with_name('paired_client.py'))+':/paired_client.py:ro','-e',f'PAIRED_BARRIER={int(barrier)}']
    argv=argv[:at]+prefix+argv[at:]
    k=argv.index('-c',argv.index('--entrypoint'))
    argv=argv[:k]+['/paired_client.py']+argv[k+2:]
    argv=bound(argv,engine,True)
    save_command(where,argv)
    start=utcnow()
    write_json(where/'phase.json',{'started_at':start,'requests':count,'paired':barrier})
    try:
        docker.run_client(argv,where/'client.log',w['measurement']['timeout_s'])
    finally:
        docker.remove_owned(f'sb-{owner}-client',owner)
        window=docker.server_logs(item['server'],since=start)
        (where/'server-window.log').write_text(window)
    metrics=vllm_bench.normalize(where/'raw.json',count,w)
    if metrics['metrics']['total_input_tokens'] != count*w['dataset']['input_tokens']:
        raise BenchError('Input workload mismatch')
    time.sleep(1)
    window=docker.server_logs(item['server'],since=start)
    (where/'server-window.log').write_text(window)
    events=logs.compilation_events(c,window)
    bad=logs.matching(window,c['recipe']['checks']['forbidden'])
    write_json(where/'metrics.json',metrics)
    write_json(where/'compilation.json',{'started_at':start,'events':events})
    write_json(where/'phase.json',{'started_at':start,'finished_at':utcnow(),'requests':count,'paired':barrier,'forbidden':bad})
    if bad: raise BenchError('Forbidden runtime event: '+bad[0])
    return metrics,events

def warmup(item,trial,attempt,log):
    m=item['case']['workloads'][0]['measurement']; quiet=0; previous=None; records=[]
    for index in range(1,m['max_warmup_rounds']+1):
        where=trial/f'warmup-{attempt:02d}-{index:02d}'
        metrics,events=phase(item,where,m['warmup_requests'])
        rate=metrics['metrics']['output_throughput']
        change=abs(rate/previous-1) if previous else None
        quiet=0 if events else quiet+1
        records.append({'path':where.name,'events':len(events),'output_throughput':rate,'relative_change':change})
        write_json(trial/f'warmup-checks-{attempt:02d}.json',records)
        log(f"{item['case']['id']}: warmup {index}, events={len(events)}, throughput={rate:.2f}, relative change={change}")
        if quiet>=m['min_warmup_rounds'] and change is not None and change<=0.05:
            return records
        previous=rate
    raise BenchError('Warmup failed quiet/convergence gate within five rounds')

def monitor(root,stop):
    with (root/'telemetry.jsonl').open('w') as f:
        while not stop.is_set():
            row={'at':utcnow(),'loadavg':os.getloadavg()}
            for key,cmd in {'gpu':['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem','--format=csv,noheader,nounits'], 'processes':['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']}.items():
                r=capture(cmd,check=False); row[key]=r.stdout; row[key+'_error']=r.stderr
            row['cpu_stat']=Path('/proc/stat').read_text()
            f.write(json.dumps(row)+'\n'); f.flush(); stop.wait(5)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('campaigns',nargs=2,type=Path)
    p.add_argument('--run-root',required=True,type=Path)
    p.add_argument('--capacity-only',action='store_true')
    args=p.parse_args(); root=args.run_root.resolve(); root.mkdir(parents=True,exist_ok=False)
    log=Logger(root/'run.log'); owner=uuid.uuid4().hex[:12]; items=[]
    plans=[resolve(c) for c in args.campaigns]
    assert [p['cases'][0]['runtime']['engine'] for p in plans]==['vllm','sglang']
    assert all(len(p['cases'])==1 for p in plans)
    assert plans[0]['cases'][0]['workloads']==plans[1]['cases'][0]['workloads']
    state={'status':'RUNNING','started_at':utcnow(),'hostname':socket.gethostname(),'owner':owner,'runner_version':__version__,'mode':'capacity' if args.capacity_only else 'paired','cases':[],'pairs':[]}
    write_json(root/'run.json',state); write_json(root/'plans.json',plans)
    write_json(root/'host.json',docker.environment_snapshot())
    write_json(root/'protocol.json',{'bindings':BINDINGS,'warmup_relative_change_limit':0.05,'throughput_repeat_cv_limit':0.05,'throughput_repeat_max_min_limit':0.10,'repetitions':3,'max_pair_attempts':2,'warmup_max_rounds':5,'warmup_requests':64,'measurement_requests':128,'concurrency_per_instance':32,'barrier':'immediately before official benchmark timer, after initial test prompt','tail_policy':'both services stay loaded; earlier finisher remains idle; actual overlap reported','startup_max_per_configuration':1})
    (root/'git-head.txt').write_text(capture(['git','rev-parse','HEAD']).stdout)
    (root/'git-diff.patch').write_text(capture(['git','diff','HEAD']).stdout)
    shutil.copytree(ROOT/'src',root/'source')
    shutil.copytree(Path(__file__).parent,root/'scripts')
    source_id=fingerprint({str(x.relative_to(ROOT/'src')):x.read_text() for x in (ROOT/'src').rglob('*.py')})
    state['runner_source_fingerprint']=source_id
    stop=threading.Event(); thread=threading.Thread(target=monitor,args=(root,stop),daemon=True); thread.start()
    previous=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
      with gpu_locks([0,1,2,3,4,5,6,7]):
        # Both CLI/model/resource checks finish before either model starts.
        for index,plan in enumerate(plans):
            c=plan['cases'][0]; engine=c['runtime']['engine']; where=root/'cases'/c['id']; where.mkdir(parents=True)
            write_json(where/'resolved.json',c)
            prepare(c,next(x for x in args.campaigns[index].resolve().parents if x.name=='configs'))
            log('Preflight '+c['id']); facts=docker.preflight(c); write_json(where/'environment.json',facts)
            item={'case':c,'dir':where,'facts':facts,'owner':owner+str(index),'server':f'sb-{owner}{index}-server','state':{'id':c['id'],'status':'STARTING','trials':[]}}
            items.append(item); state['cases'].append(item['state'])
        if items[0]['facts']['model']['identity']!=items[1]['facts']['model']['identity']: raise BenchError('Model identity mismatch')
        for item in items:
            c=item['case']; where=item['dir']; docker.cache_directory(c,item['facts']['server_image']['Id']).mkdir(parents=True,exist_ok=True)
            argv=bound(docker.server_command(c,item['server'],item['owner'],item['facts']['server_image']['Id']),c['runtime']['engine'])
            save_command(where,argv); log('Starting '+c['id']); capture(argv,timeout=120)
            item['state']['startup_seconds']=http.wait_ready(c,item['server'],log)
            item['state']['probes']=http.probes(c,where/'probes')
            text=docker.server_logs(item['server']); (where/'startup.log').write_text(text)
            checks=logs.kernel_checks(c,text); write_json(where/'startup-checks.json',checks)
            if checks['status']!='PASS': raise BenchError('Startup kernel check failed: '+str(checks['missing'])+str(checks['forbidden']))
            item['state']['status']='READY'; write_json(root/'run.json',state); log(c['id']+' ready; startup checks PASS')
            if not args.capacity_only:
                trial=where/'trials'/'r01'; trial.mkdir(parents=True)
                warmup(item,trial,1,log)
        if args.capacity_only:
            state['status']='CAPACITY_CHECKED'
        else:
            # Capacity must have been reviewed before these frozen plans are launched.
            for repetition in range(1,4):
                for attempt in range(1,3):
                    dirs=[]
                    for item in items:
                        trial=item['dir']/'trials'/f'r{repetition:02d}'; trial.mkdir(parents=True,exist_ok=True)
                        if repetition!=1 or attempt!=1: warmup(item,trial,attempt,log)
                        dirs.append(trial/f'measurement-{attempt:02d}')
                    log(f'Pair R{repetition} attempt {attempt}: launching clients to timer barrier')
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futures=[pool.submit(phase,item,d,128,True) for item,d in zip(items,dirs)]
                        try:
                            deadline=time.monotonic()+7200
                            while not all((d/'ready.time').exists() for d in dirs):
                                for future in futures:
                                    if future.done(): future.result(); raise BenchError('Client ended before barrier')
                                if time.monotonic()>deadline: raise BenchError('Barrier deadline')
                                time.sleep(0.2)
                            # Catch compilation in client initial probes before releasing either measurement.
                            probe_events=[]
                            for item,d in zip(items,dirs):
                                start=json.loads((d/'phase.json').read_text())['started_at']
                                probe_events += logs.compilation_events(item['case'],docker.server_logs(item['server'],since=start))
                            for d in dirs: (d/'release').write_text(utcnow())
                            results=[f.result() for f in futures]
                        except BaseException:
                            # Unblock a surviving client before the thread pool joins.
                            # remove_owned checks labels and cannot target another run.
                            for item in items:
                                docker.remove_owned(f"sb-{item['owner']}-client",item['owner'])
                            raise
                    windows=[(float((d/'benchmark-start.time').read_text()),float((d/'benchmark-end.time').read_text())) for d in dirs]
                    accepted=not probe_events and all(not events for _,events in results)
                    pair={'repetition':repetition,'attempt':attempt,'accepted':accepted,'probe_compile_events':probe_events,'windows_epoch_s':windows,'start_skew_s':abs(windows[0][0]-windows[1][0]),'overlap_s':max(0,min(w[1] for w in windows)-max(w[0] for w in windows)),'tail_s':abs(windows[0][1]-windows[1][1])}
                    if pair['start_skew_s']>1: raise BenchError('Paired timer start skew exceeds one second')
                    state['pairs'].append(pair); write_json(root/'run.json',state)
                    if accepted:
                        for item,d,(metrics,_) in zip(items,dirs,results):
                            item['state']['trials'].append({'repetition':repetition,'path':str(d.relative_to(item['dir'])),'metrics':metrics})
                        write_json(root/'run.json',state); log(f'Pair R{repetition} accepted: '+', '.join(f"{m[0]['metrics']['output_throughput']:.2f}" for m in results)); break
                    log(f'Pair R{repetition} attempt {attempt} rejected due to compilation; keeping both samples')
                else: raise BenchError('Pair exhausted two measurement attempts')
            stability={}
            for item in items:
                rates=[t['metrics']['metrics']['output_throughput'] for t in item['state']['trials']]
                mean=statistics.mean(rates); cv=statistics.stdev(rates)/mean; span=max(rates)/min(rates)-1
                stability[item['case']['id']]={'rates':rates,'mean':mean,'sample_sd':statistics.stdev(rates),'cv':cv,'max_min_relative_range':span,'pass':cv<=.05 and span<=.10}
            write_json(root/'stability.json',stability)
            state['status']='PASS' if all(x['pass'] for x in stability.values()) else 'UNSTABLE'
    except BaseException as exc:
        state.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL',error=f'{type(exc).__name__}: {exc}')
        log(state['error'])
    finally:
        stop.set(); thread.join(timeout=15)
        for item in items:
            try:
                info=docker.container_info(item['server'])
                if info:
                    if (info['Config'].get('Labels') or {}).get(docker.OWNER_LABEL)!=item['owner']: raise BenchError('Ownership mismatch')
                    write_json(item['dir']/'server-inspect.json',info)
                    text=docker.server_logs(item['server']); (item['dir']/'server.log').write_text(text)
                    checks=logs.kernel_checks(item['case'],text); write_json(item['dir']/'kernel-checks.json',checks)
                    if checks['status']!='PASS': state.update(status='FAIL',error='Final kernel gate failed')
                for name in (f"sb-{item['owner']}-client",item['server']): docker.remove_owned(name,item['owner'])
                item['state']['status']=state['status']; write_json(item['dir']/'case.json',item['state'])
            except Exception as exc:
                state.setdefault('cleanup_errors',[]).append(str(exc)); state['status']='FAIL'
        state['finished_at']=utcnow(); write_json(root/'run.json',state)
        signal.signal(signal.SIGTERM,previous); log('Finished '+state['status'])
    return 0 if state['status'] in ('PASS','CAPACITY_CHECKED') else 1
if __name__=='__main__': raise SystemExit(main())
