"""Bounded experimental SSH service lifecycle, driven by resolved campaigns.

No requests are generated here. READY gates the separate functional client.
STOP or deadline triggers ownership-checked cleanup on all participating hosts.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import urllib.request
from .config import resolve
from .executors.docker import server_command, OWNER_LABEL


def remote(host, argv, timeout=60):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, shlex.join(argv)],
                          text=True, capture_output=True, timeout=timeout, check=True).stdout


def owned(info, owner):
    return bool(info and info[0].get('Config', {}).get('Labels', {}).get(OWNER_LABEL) == owner)


def cleanup(member, owner):
    try:
        info = json.loads(remote(member['host'], ['docker', 'inspect', member['name']]))
    except subprocess.CalledProcessError as e:
        if 'No such' in e.stderr:
            return
        raise
    if not owned(info, owner):
        raise RuntimeError('Refusing to remove an unowned container')
    remote(member['host'], ['docker', 'rm', '-f', member['name']], timeout=90)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('deployment', type=Path); p.add_argument('--run-root', required=True, type=Path)
    a = p.parse_args(); root = a.run_root.resolve(); root.mkdir(parents=True, exist_ok=False)
    spec = json.loads(a.deployment.read_text()); owner = spec['owner']
    (root/'deployment.json').write_text(json.dumps(spec, indent=2))
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    members = []; log_procs = []; started = time.monotonic()
    state = {'status': 'PREPARING', 'owner': owner, 'started': time.time()}
    def save():
        (root/'state.json').write_text(json.dumps(state, indent=2))
        print(json.dumps(state), flush=True)
    save()
    watched = list(Path('src').rglob('*.py')) + list(a.deployment.parent.joinpath('configs').rglob('*.yaml'))
    frozen = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    (root/'frozen-files.json').write_text(json.dumps(frozen,indent=2))
    try:
        for entry in spec['members']:
            case = resolve(Path(entry['campaign']))['cases'][0]
            m = dict(entry, case=case, name=f"sb-{owner}-{entry['role']}")
            members.append(m)
            md = root/entry['role']; md.mkdir(); m['directory'] = str(md)
            (md/'resolved.json').write_text(json.dumps(case, indent=2))
            argv = server_command(case, m['name'], owner, case['runtime']['image_id'])
            (md/'command.json').write_text(json.dumps(argv, indent=2))
            m['argv'] = argv
            facts = remote(m['host'], ['python3', '-c', '''import json,subprocess,pathlib,hashlib,socket
out={}
for key,args in [('image',['docker','image','inspect','vllm/vllm-openai:v0.29.0']),('gpus',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']),('topology',['nvidia-smi','topo','-m']),('rdma',['rdma','link','show']),('cpu',['lscpu','-e=CPU,NODE,SOCKET,CORE'])]:
 p=subprocess.run(args,capture_output=True,text=True,check=True);out[key]=p.stdout
model=pathlib.Path('/data/models/DeepSeek-V4-Flash-0731-NVFP4')
out['model_hashes']={n:hashlib.sha256((model/n).read_bytes()).hexdigest() for n in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']}
print(json.dumps(out))'''])
            (md/'preflight.json').write_text(facts)
            f = json.loads(facts)
            if json.loads(f['image'])[0]['Id'] != case['runtime']['image_id'] or f['gpus'].strip():
                raise RuntimeError('Image mismatch or GPU compute processes present')
        hashes = [json.loads((Path(m['directory'])/'preflight.json').read_text())['model_hashes'] for m in members]
        if any(h != hashes[0] for h in hashes): raise RuntimeError('Model metadata differs')
        state['status'] = 'STARTING'; save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(members)) as ex:
            jobs = [(m, ex.submit(remote, m['host'], m['argv'], 120)) for m in members]
            for m, job in jobs:
                (Path(m['directory'])/'container-id.txt').write_text(job.result())
                f = (Path(m['directory'])/'server.log').open('w')
                proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', m['host'], shlex.join(['docker','logs','--timestamps','-f',m['name']])], stdout=f, stderr=subprocess.STDOUT)
                log_procs.append((proc,f))
        deadline = started + spec.get('budget_s',7200)
        ready_deadline = time.monotonic() + spec.get('ready_timeout_s',1800)
        ready = set()
        while not stopping and not (root/'STOP').exists() and time.monotonic() < deadline:
            for m in members:
                md=Path(m['directory'])
                info=json.loads(remote(m['host'],['docker','inspect',m['name']]))
                (md/'inspect.json').write_text(json.dumps(info,indent=2))
                if not owned(info,owner) or not info[0]['State']['Running']:
                    raise RuntimeError(f"{m['role']} service exited: {info[0]['State']}")
                if m['role'] not in ready:
                    try:
                        with urllib.request.urlopen(m['url']+'/health',timeout=2) as r:
                            if r.status==200: ready.add(m['role'])
                    except OSError: pass
                metrics=''
                if m['role'] in ready:
                    try:
                        with urllib.request.urlopen(m['url']+'/metrics',timeout=3) as r: metrics=r.read().decode()
                    except OSError: pass
                with (md/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps({'time':time.time(),'metrics':metrics})+'\n')
                telemetry=remote(m['host'], ['python3','-c', '''import pathlib,json,subprocess,time
p=pathlib.Path('/sys/class/infiniband'); d={'time':time.time(),'proc_stat':pathlib.Path('/proc/stat').read_text(),'net':{}}
for n in ['mlx5_4','mlx5_5','mlx5_6','mlx5_7']:
 d['net'][n]={f.name:f.read_text().strip() for f in (p/n/'ports/1/counters').glob('*') if f.name in ['port_xmit_data','port_rcv_data','port_xmit_packets','port_rcv_packets']}
d['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout
print(json.dumps(d))'''],timeout=20)
                with (md/'telemetry.jsonl').open('a') as f: f.write(telemetry)
            if len(ready)==len(members) and state['status']!='READY':
                state['status']='READY';state['ready_at']=time.time();(root/'READY').touch();save()
            if len(ready)<len(members) and time.monotonic()>ready_deadline:
                raise TimeoutError('Service readiness budget exhausted')
            time.sleep(5)
        state['status']='STOPPED' if stopping or (root/'STOP').exists() else 'BUDGET_EXHAUSTED'
    except Exception as e:
        state['status']='FAIL';state['error']=repr(e)
    finally:
        errors=[]
        for m in members:
            try:
                logs=remote(m['host'],['docker','logs','--timestamps',m['name']],timeout=60)
                (Path(m['directory'])/'server-final.stdout.log').write_text(logs)
            except Exception: pass
            try: cleanup(m,owner)
            except Exception as e: errors.append(repr(e))
        for proc,f in log_procs:
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.terminate()
            f.close()
        state['changed_files']=[p for p,h in frozen.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
        state['cleanup_errors']=errors;state['ended']=time.time();save()

if __name__=='__main__': main()
