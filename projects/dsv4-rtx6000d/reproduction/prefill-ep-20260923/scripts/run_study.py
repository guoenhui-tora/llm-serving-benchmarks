"""Bounded local composition of existing preflight, phase, protocol and telemetry tools."""
import argparse,copy,hashlib,json,os,re,signal,subprocess,sys,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[3];W=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from serving_bench.config import resolve
from serving_bench.common import BenchError,capture,write_json,read_json,save_command,Logger,utcnow
from serving_bench.executors import docker
from serving_bench.checks import http,logs
from serving_bench.clients import vllm_bench
from serving_bench.runner import phase
from serving_bench.protocols import collect
from serving_bench.telemetry import Recorder
from serving_bench.locks import gpu_locks
from metrics import MetricsRecorder,audit

def deadline_check():
    if time.time()>read_json(W/'evidence/authorized-budget.json')['deadline_unix']:raise TimeoutError('Four-hour total budget exhausted')

def verify_frozen():
    for name,digest in read_json(W/'evidence/frozen.json').items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen file changed: '+name)

def startup_coverage(text):
    text=re.sub(r'\x1b\[[0-9;]*m','',text);result={}
    for marker in ('DSV4_INPUT_KERNEL_WARMUP','DSV4_TARGETED_WARMUP','DSV4_DSPARK_WARMUP'):
        records=[]
        for line in text.splitlines():
            if marker+' ' in line:
                raw=line.split(marker+' ',1)[1]
                try:r=json.JSONDecoder().raw_decode(raw)[0]
                except ValueError:continue
                if r.get('phase')=='COMPLETE':records.append(r)
        workers={(r['dp_rank'],r['rank']) for r in records}
        assert len(workers)==4,(marker,'missing workers',workers)
        assert all(r.get('memory_allocated_delta',0)==0 for r in records),(marker,'scratch memory')
        result[marker]=records
    graph=[s for s in text.splitlines() if 'Graph capturing finished' in s]
    assert len(graph)>=2,('Missing final graph capture records',graph)
    result['graph_completed']=graph
    result['graph_progress']=[s for s in text.splitlines() if 'Capturing CUDA graphs' in s or 'cudagraph_capture_sizes' in s]
    return result

def install_client_observer():
    original=vllm_bench.command
    def command(case,*args,**kwargs):
        argv=original(case,*args,**kwargs);i=argv.index('-c');port=case['target']['port']
        prefix="import sys; sys.path.insert(0,'/benchsrc'); sys.path.insert(0,'/hooks'); import vllm.benchmarks.serve as serve; from client_observe import install; install(serve,"+str(port)+")\n"
        argv[i+1]=prefix+argv[i+1]
        argv[2:2]=['-v',str(R/'src')+':/benchsrc:ro','-v',str(W/'scripts')+':/hooks:ro']
        return argv+['--save-detailed']
    vllm_bench.command=command

def worker(mode,root):
    here=root/mode;here.mkdir();state={'mode':mode,'status':'STARTING','started_at':utcnow(),'cells':[]};owner='prefill-ep-'+mode+'-'+root.name;server='sb-'+owner+'-server';recorder=None
    log=Logger(here/'run.log');case=resolve(W/f'configs/campaigns/prefill-{mode}.yaml')['cases'][0];facts=read_json(W/f'evidence/preflight-{mode}.json');workload=case['workloads'][0]
    def save():write_json(here/'status.json',state)
    def stop(*_):raise TimeoutError('Study deadline or controller stop')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop)
    remaining=read_json(W/'evidence/authorized-budget.json')['deadline_unix']-time.time()
    signal.alarm(max(1,int(min(2700,remaining))));save();install_client_observer()
    try:
        with gpu_locks(case['target']['gpus']):
            deadline_check();verify_frozen();docker.check_idle(docker.validate_gpus(case['target'],docker.gpu_inventory()))
            write_json(here/'resolved.json',case);write_json(here/'environment.json',facts)
            argv=docker.server_command(case,server,owner,facts['server_image']['Id']);save_command(here,argv);capture(argv,timeout=120)
            state['startup_seconds']=http.wait_ready(case,server,log)
            docker.binding_evidence(case['target'],'server',server,owner,here,live=True)
            write_json(here/'models.json',http.request(http.base_url(case),'/v1/models'))
            text=docker.server_logs(server);(here/'startup.log').write_text(text)
            write_json(here/'startup-coverage.json',startup_coverage(text))
            check=logs.kernel_checks(case,text);write_json(here/'startup-checks.json',check);assert check['status']=='PASS',check
            recorder=MetricsRecorder(case['target']['port'],here/'engine-telemetry.jsonl');recorder.start()
            # Eight complete real 16K/1 requests, separate from all scan warmups.
            functional=copy.deepcopy(workload);functional['measurement']['timeout_s']=min(600,max(1,int(2700-state['startup_seconds'])))
            phase(case,functional,8,8,here/'functional',server,owner,facts)
            audit(here/'functional',8)
            state.update(status='READY',ready_at=utcnow());save();(here/'READY').touch()
            signal.alarm(max(1,int(read_json(W/'evidence/authorized-budget.json')['deadline_unix']-time.time())))
            while not (root/'RELEASE').exists():
                if (root/'ABORT').exists():raise RuntimeError('Peer failed startup/functional gate')
                deadline_check();time.sleep(5)
            verify_frozen();state.update(status='SCANNING',scan_started_at=utcnow());save()
            for concurrency in workload['traffic']['concurrency']:
                deadline_check();verify_frozen();block=here/'trials'/f'c{concurrency:02d}'
                def measure(count,directory,timeout):
                    bounded=copy.deepcopy(workload);bounded['measurement']['timeout_s']=min(timeout,read_json(W/'evidence/authorized-budget.json')['deadline_unix']-time.time())
                    metrics,events=phase(case,bounded,concurrency,count,directory,server,owner,facts)
                    audit(directory,count)
                    check=logs.kernel_checks(case,(directory/'server-window.log').read_text())
                    write_json(directory/'window-checks.json',check)
                    if check['forbidden']:raise RuntimeError('Forbidden server event: '+str(check['forbidden']))
                    return metrics,events
                trials,protocol=collect(workload,concurrency,block,measure,log,lambda _:deadline_check())
                item={'concurrency':concurrency,'status':protocol['status'],'path':str(block),'trials':trials}
                state['cells'].append(item);save()
                if protocol['status']!='PASS':raise TimeoutError('Cell budget exhausted; no retry')
            state['status']='COMPLETE'
    except BaseException as e:
        state.update(status='PARTIAL' if isinstance(e,TimeoutError) else 'FAIL',error=f'{type(e).__name__}: {e}')
        (here/'failure.txt').write_text(traceback.format_exc());log(state['error'])
    finally:
        signal.alarm(0)
        if recorder:recorder.close()
        try:
            info=docker.container_info(server)
            if info:
                assert (info['Config'].get('Labels') or {}).get(docker.OWNER_LABEL)==owner
                write_json(here/'server-inspect.json',info);text=docker.server_logs(server);(here/'server.log').write_text(text);write_json(here/'kernel-checks.json',logs.kernel_checks(case,text))
        except Exception as e:state['evidence_error']=str(e)
        for name in ('sb-'+owner+'-client',server):
            try:docker.remove_owned(name,owner)
            except Exception as e:state.setdefault('cleanup_errors',[]).append(str(e))
        if state.get('cleanup_errors'):state['status']='FAIL'
        state.update(finished_at=utcnow(),exit_code=0 if state['status']=='COMPLETE' else 2);save();(here/('DONE' if state['exit_code']==0 else 'FAILED')).touch()
    return state['exit_code']

def control(root):
    root.mkdir(parents=True,exist_ok=False);verify_frozen();deadline_check();write_json(root/'host-start.json',docker.environment_snapshot());write_json(root/'budget.json',read_json(W/'evidence/authorized-budget.json'))
    recorder=Recorder(root,lambda s:None,interval=5);recorder.start();processes={};streams=[];released=False
    try:
        for mode in ('on','off'):
            f=(root/f'{mode}-controller.log').open('w');streams.append(f)
            processes[mode]=subprocess.Popen([sys.executable,__file__,'--worker',mode,'--run-root',str(root)],stdout=f,stderr=subprocess.STDOUT)
        while any(p.poll() is None for p in processes.values()):
            if not released:
                if all((root/m/'READY').exists() for m in processes):
                    (root/'RELEASE').write_text(utcnow());released=True;print('BOTH_READY_SCAN_RELEASED',flush=True)
                elif any(p.poll() is not None for p in processes.values()):
                    (root/'ABORT').touch()
            if time.time()>read_json(W/'evidence/authorized-budget.json')['deadline_unix']:
                (root/'ABORT').touch()
                for p in processes.values():
                    if p.poll() is None:p.terminate()
            time.sleep(10)
        codes={m:p.wait() for m,p in processes.items()};summaries={m:read_json(root/m/'status.json') if (root/m/'status.json').exists() else {'status':'FAIL','error':'worker exited without status'} for m in processes}
        write_json(root/'summary.json',{'exit_codes':codes,'groups':summaries,'scan_released':released,'finished_at':utcnow()});(root/'FINISHED').touch();print(json.dumps({'finished':True,'exit_codes':codes,'statuses':{m:v['status'] for m,v in summaries.items()}}),flush=True)
    finally:
        for p in processes.values():
            if p.poll() is None:p.terminate()
        for p in processes.values():p.wait(timeout=180)
        recorder.close()
        for f in streams:f.close()
    return 0 if all(p.returncode==0 for p in processes.values()) else 2
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=['on','off']);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();root=args.run_root.resolve();assert root.is_relative_to(W/'results')
    sys.exit(worker(args.worker,root) if args.worker else control(root))
